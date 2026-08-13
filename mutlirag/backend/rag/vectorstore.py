"""Vector store: hybrid retrieval over chunks, with a pluggable DB backend.

Search combines two signals:
  1. Semantic — cosine similarity of sentence-transformer embeddings.
     Great at meaning ("who earns the most"), weak at exact tokens.
  2. Keyword — BM25 over tokenized chunk text.
     Great at exact tokens ("ZEBRA-42", invoice numbers), weak at paraphrase.
The final score is a weighted blend (config.HYBRID_ALPHA).

Two persistence backends are available (config.VECTOR_BACKEND):
  * "weaviate" — self-hosted Weaviate (see docker-compose.yml). Vectors + text
                 live in a Docker volume, so the index survives restarts.
  * "chroma"   — the original embedded ChromaDB under index_store/chroma.
If Weaviate is selected but unreachable, we fall back to Chroma automatically.

Both backends embed locally and keep an in-memory copy of the vectors + a BM25
index, so the *search* (the blended scoring above) is identical regardless of
which DB stores the data. The DB is the source of record; NumPy does the maths.
"""

from __future__ import annotations

import os
import json
import uuid
import re

import numpy as np
from rank_bm25 import BM25Okapi

import config
from rag.embeddings import embed, embedding_dim
from rag.ingestion import Document

_token_re = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


def _is_toc(meta: dict) -> bool:
    """True when a chunk's page was flagged TOC/index at ingestion.

    Checks both the current `is_index` key (see ingestion._looks_like_toc)
    and a possible future `is_toc` key, so a metadata rename never has to
    touch this logic again. Missing on both (pre-existing indexed chunks) ->
    False, i.e. treated as ordinary content — never crashes, just requires
    re-ingestion to get accurate TOC tagging.
    """
    return bool(meta.get("is_index") or meta.get("is_toc"))


# --------------------------------------------------------------------------- #
# Shared search / scoring logic
# --------------------------------------------------------------------------- #
class _BaseStore:
    """Holds the in-memory documents + vectors + BM25 index and implements the
    hybrid search. Subclasses add persistence (add / remove / load / hydrate)."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._dim = embedding_dim()
        self._matrix = np.empty((0, self._dim), dtype="float32")
        self._bm25: BM25Okapi | None = None

    # -- in-memory index maintenance -- #
    def _reset_memory(self) -> None:
        self.documents = []
        self._matrix = np.empty((0, self._dim), dtype="float32")
        self._bm25 = None

    def _rebuild_bm25(self) -> None:
        corpus = [_tokenize(d.text) for d in self.documents]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    # -- search (identical for every backend) -- #
    def search(
        self,
        query: str,
        top_k: int = 5,
        chat_id: str | None = None,
        explain: bool = False,
    ) -> list[tuple[Document, float]] | list[dict]:
        """Return the top_k (Document, blended score) pairs for a query.

        When `chat_id` is given, only chunks belonging to that chat are
        considered — so a chat can never retrieve another chat's document.

        `explain=True` returns the same top_k selection, but as
        `[{"doc":, "score":, "dense":, "bm25":}, ...]` dicts exposing the two
        raw signals `score` (the fused hybrid score) is blended from —
        for retrieval diagnostics (see RagService._log_retrieval_diagnostics)
        without a second, separately-computed call: dense/bm25 are already
        sitting right here either way, this just exposes them instead of
        discarding them.
        """
        if not self.documents:
            return []

        # Semantic: vectors are L2-normalized, so dot product == cosine.
        sims = self._matrix @ embed([query])[0]

        # Keyword: BM25 scores are unbounded, so normalize by the max to get
        # a 0..1 signal comparable with cosine before blending.
        bm = np.asarray(self._bm25.get_scores(_tokenize(query)), dtype="float32")
        if bm.max() > 0:
            bm = bm / bm.max()

        alpha = config.HYBRID_ALPHA
        fused = alpha * sims + (1.0 - alpha) * bm

        # Per-chat isolation: mask out every chunk not owned by this chat.
        # TOC/index-page exclusion: same mechanism, ANDed in — those pages are
        # pure noise (see ingestion._looks_like_toc), never worth retrieving
        # regardless of chat.
        mask = None
        if chat_id is not None:
            mask = np.array(
                [d.meta.get("chat_id") == chat_id for d in self.documents],
                dtype=bool,
            )
        if config.EXCLUDE_INDEX_PAGES:
            not_index = np.array(
                [not _is_toc(d.meta) for d in self.documents], dtype=bool
            )
            mask = not_index if mask is None else (mask & not_index)

        # Dedup: exact-duplicate chunk text (same page/content indexed more
        # than once — e.g. a file re-uploaded to a chat before
        # RagService.ingest_file started replacing the previous one, or a
        # retried upload request) must not occupy more than one of the final
        # top_k slots. Scoped to this chat only — two DIFFERENT chats sharing
        # identical boilerplate text (e.g. the same template document) is not
        # a bug, so it's left alone. Only the first occurrence (by
        # self.documents' order — oldest-indexed first) survives; the rest
        # get masked out here, same as any other excluded chunk.
        seen_texts: set[str] = set()
        dedupe_mask = np.ones(len(self.documents), dtype=bool)
        for i, d in enumerate(self.documents):
            if chat_id is not None and d.meta.get("chat_id") != chat_id:
                continue
            if d.text in seen_texts:
                dedupe_mask[i] = False
            else:
                seen_texts.add(d.text)
        mask = dedupe_mask if mask is None else (mask & dedupe_mask)

        if mask is not None:
            if not mask.any():
                return []
            fused = np.where(mask, fused, -np.inf)
            limit = int(mask.sum())
        else:
            limit = len(self.documents)

        k = min(top_k, limit)
        idxs = np.argsort(-fused)[:k]
        idxs = [i for i in idxs if np.isfinite(fused[i])]
        if explain:
            return [
                {
                    "doc": self.documents[i],
                    "score": float(fused[i]),
                    "dense": float(sims[i]),
                    "bm25": float(bm[i]),
                }
                for i in idxs
            ]
        return [(self.documents[i], float(fused[i])) for i in idxs]

    # -- deterministic, metadata-only retrieval (TOC / page requests) -- #
    # A structural request ("give me the TOC", "what's on page 247") is not a
    # semantic-similarity question — answering it via embedding/BM25 search
    # risks missing pages whose wording doesn't score highly, or returning
    # only one of several TOC pages. These filter self.documents directly
    # (same in-memory list search() uses) and always bypass the TOC exclusion
    # mask above: an explicit request for the TOC or a specific page wins
    # over the generic "TOC is noise" rule that mask encodes.
    def _sort_key(self, d: Document):
        m = d.meta
        return (m.get("page") or 0, m.get("chunk") or 0, m.get("child") or 0)

    @staticmethod
    def _dedupe_by_text(docs: list[Document]) -> list[Document]:
        """Drop exact-duplicate chunk text, keeping the first occurrence.

        Same reasoning as search()'s dedupe_mask: a page/chunk indexed more
        than once (duplicate upload, retried request) must not show up twice
        in a TOC or page-request answer either.
        """
        seen: set[str] = set()
        out: list[Document] = []
        for d in docs:
            if d.text in seen:
                continue
            seen.add(d.text)
            out.append(d)
        return out

    def get_toc_chunks(self, chat_id: str | None = None) -> list[Document]:
        """Every chunk from a page flagged TOC/index, in page order,
        deduplicated.

        Chat-isolated exactly like search(): when `chat_id` is given, only
        that chat's chunks are considered, so TOC retrieval can never leak
        another chat's document.
        """
        matched = [
            d for d in self.documents
            if _is_toc(d.meta) and (chat_id is None or d.meta.get("chat_id") == chat_id)
        ]
        return self._dedupe_by_text(sorted(matched, key=self._sort_key))

    def get_by_pages(self, pages, chat_id: str | None = None) -> list[Document]:
        """Every chunk belonging to one of `pages`, in page order,
        deduplicated — regardless of is_index, since an explicit page
        request always wins over the generic TOC-exclusion rule.
        Chat-isolated like search()/get_toc_chunks.
        """
        wanted = set(pages)
        matched = [
            d for d in self.documents
            if d.meta.get("page") in wanted and (chat_id is None or d.meta.get("chat_id") == chat_id)
        ]
        return self._dedupe_by_text(sorted(matched, key=self._sort_key))

    def get_neighbors(self, doc: Document, window: int = 1) -> list[Document]:
        """Parent chunks immediately before/after `doc`'s own parent chunk,
        in this document's natural (page, chunk) reading order — same
        source + chat_id only, and never a TOC-flagged page (expansion is
        for structural content completeness, not for pulling in unrelated
        structural noise; TOC pages are transparently skipped over, so a
        TOC page sitting between two content sections doesn't break their
        adjacency).

        This is how a chunk that scores far too low to enter any reasonable
        candidate pool on its own — confirmed live: a real "Guideline #3"
        chunk ranked ~140th for its own query, entirely on its own wording —
        still reaches the LLM when the chunk immediately before/after it
        (same page, same section) DID score well. See
        RagService.retrieve's neighbor-expansion step and
        config.NEIGHBOR_EXPANSION_WINDOW.

        Returns one representative Document per neighboring parent chunk
        (its first child, whose meta["parent_text"] already carries the
        full parent text — the same text generation resolves to anyway).
        """
        page, chunk = doc.meta.get("page"), doc.meta.get("chunk")
        if page is None or chunk is None or window <= 0:
            return []

        chat_id = doc.meta.get("chat_id")
        scoped = [
            d for d in self.documents
            if d.source == doc.source
            and d.meta.get("chat_id") == chat_id
            and not _is_toc(d.meta)
        ]
        if not scoped:
            return []

        order: list[tuple] = []
        representative: dict[tuple, Document] = {}
        for d in sorted(scoped, key=self._sort_key):
            pid = (d.meta.get("page"), d.meta.get("chunk"))
            if pid not in representative:
                order.append(pid)
                representative[pid] = d

        this_id = (page, chunk)
        if this_id not in representative:
            return []
        pos = order.index(this_id)
        neighbor_ids = order[max(0, pos - window):pos] + order[pos + 1:pos + 1 + window]
        return [representative[pid] for pid in neighbor_ids]

    # -- read-only views used by the UI -- #
    @property
    def size(self) -> int:
        return len(self.documents)

    @property
    def sources(self) -> list[str]:
        """Unique source filenames currently in the index, in first-seen order."""
        seen: dict[str, None] = {}
        for d in self.documents:
            seen.setdefault(d.source, None)
        return list(seen)

    def size_for_chat(self, chat_id: str) -> int:
        """Number of chunks belonging to a single chat."""
        return sum(1 for d in self.documents if d.meta.get("chat_id") == chat_id)

    def sources_for_chat(self, chat_id: str) -> list[str]:
        """Unique source filenames for a single chat, in first-seen order."""
        seen: dict[str, None] = {}
        for d in self.documents:
            if d.meta.get("chat_id") == chat_id:
                seen.setdefault(d.source, None)
        return list(seen)

    def remove_chat(self, chat_id: str) -> int:
        """Delete every chunk owned by a chat (called when a chat is deleted)."""
        for src in self.sources_for_chat(chat_id):
            self.remove_source(src, chat_id=chat_id)
        return 0

    # -- backend hooks (overridden) -- #
    def add(self, docs: list[Document], progress_callback=None) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def remove_source(
        self, source: str, chat_id: str | None = None
    ) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def clear(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def save(self, dirpath: str = config.INDEX_DIR) -> None:
        # Both backends persist on write, so save() is a no-op. Kept for the
        # call sites in app.py that expect it.
        pass


# --------------------------------------------------------------------------- #
# Weaviate backend (default)
# --------------------------------------------------------------------------- #
class _WeaviateStore(_BaseStore):
    """Persists chunks + our locally-computed vectors in a Weaviate collection.

    We disable Weaviate's own vectorizer (Configure.Vectorizer.none) and hand it
    the sentence-transformer vectors directly, keeping embeddings free/offline.
    On startup every object is pulled back into memory to rebuild the search
    index — same as the Chroma backend does.
    """

    def __init__(self, dirpath: str = config.INDEX_DIR) -> None:
        super().__init__()
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType

        self.backend = "hybrid (Weaviate + BM25)"
        self._name = config.WEAVIATE_COLLECTION

        # Raises if the container isn't up -> the factory falls back to Chroma.
        self.client = weaviate.connect_to_local(
            host=config.WEAVIATE_HOST,
            port=config.WEAVIATE_HTTP_PORT,
            grpc_port=config.WEAVIATE_GRPC_PORT,
        )

        if not self.client.collections.exists(self._name):
            self.client.collections.create(
                name=self._name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="kind", data_type=DataType.TEXT),
                    # Which chat owns this chunk — enables per-chat isolation.
                    Property(name="chat_id", data_type=DataType.TEXT),
                    # Full metadata dict, JSON-encoded (Weaviate props are flat).
                    Property(name="meta_json", data_type=DataType.TEXT),
                ],
            )
        self.collection = self.client.collections.get(self._name)

        self._hydrate()

    def _hydrate(self) -> None:
        """Load every stored object (text + vector + meta) into memory."""
        self._reset_memory()
        docs: list[Document] = []
        vectors: list[list[float]] = []

        for obj in self.collection.iterator(include_vector=True):
            props = obj.properties or {}
            # v4 returns vectors as {"default": [...]}
            vec = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
            if not vec:
                continue
            try:
                meta = json.loads(props.get("meta_json") or "{}")
            except Exception:
                meta = {}
            docs.append(
                Document(
                    text=props.get("text", ""),
                    source=props.get("source", ""),
                    kind=props.get("kind", ""),
                    meta=meta,
                )
            )
            vectors.append(vec)

        self.documents = docs
        self._matrix = (
            np.array(vectors, dtype="float32")
            if vectors
            else np.empty((0, self._dim), dtype="float32")
        )
        self._rebuild_bm25()

    def add(self, docs: list[Document], progress_callback=None) -> None:
        if not docs:
            return
        batch_size = 32
        vectors_list = []
        total_docs = len(docs)
        for i in range(0, total_docs, batch_size):
            batch = docs[i:i + batch_size]
            batch_vecs = embed([d.text for d in batch])
            vectors_list.append(batch_vecs)
            if progress_callback:
                processed = min(i + batch_size, total_docs)
                progress_callback(processed, total_docs, f"Embedding chunk {processed} of {total_docs}")
        vectors = np.vstack(vectors_list) if vectors_list else np.empty((0, self._dim), dtype="float32")

        with self.collection.batch.dynamic() as batch:
            for doc, vector in zip(docs, vectors):
                meta = dict(doc.meta)
                batch.add_object(
                    uuid=str(uuid.uuid4()),
                    properties={
                        "text": doc.text,
                        "source": doc.source,
                        "kind": doc.kind,
                        "chat_id": meta.get("chat_id", ""),
                        "meta_json": json.dumps(meta),
                    },
                    vector=vector.tolist(),
                )

        # Update the in-memory index to match, without a full reload.
        self.documents.extend(docs)
        self._matrix = np.vstack([self._matrix, vectors])
        self._rebuild_bm25()

    def remove_source(self, source: str, chat_id: str | None = None) -> int:
        """Drop chunks from `source`; if `chat_id` is given, only that chat's."""
        from weaviate.classes.query import Filter

        def match(d: Document) -> bool:
            return d.source == source and (
                chat_id is None or d.meta.get("chat_id") == chat_id
            )

        removed = sum(1 for d in self.documents if match(d))
        if removed:
            where = Filter.by_property("source").equal(source)
            if chat_id is not None:
                where = where & Filter.by_property("chat_id").equal(chat_id)
            self.collection.data.delete_many(where=where)

            keep = [i for i, d in enumerate(self.documents) if not match(d)]
            self.documents = [self.documents[i] for i in keep]
            self._matrix = (
                self._matrix[keep] if keep else np.empty((0, self._dim), dtype="float32")
            )
            self._rebuild_bm25()
        return removed

    def clear(self) -> None:
        """Delete the whole collection and recreate it empty."""
        from weaviate.classes.config import Configure, Property, DataType

        try:
            self.client.collections.delete(self._name)
        except Exception:
            pass
        self.client.collections.create(
            name=self._name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="kind", data_type=DataType.TEXT),
                Property(name="chat_id", data_type=DataType.TEXT),
                Property(name="meta_json", data_type=DataType.TEXT),
            ],
        )
        self.collection = self.client.collections.get(self._name)
        self._reset_memory()


# --------------------------------------------------------------------------- #
# Chroma backend (fallback / opt-in)
# --------------------------------------------------------------------------- #
class _ChromaStore(_BaseStore):
    """The original embedded-ChromaDB backend, kept as an automatic fallback."""

    def __init__(self, dirpath: str = config.INDEX_DIR) -> None:
        super().__init__()
        import chromadb

        self.backend = "hybrid (Chroma + BM25)"
        chroma_path = os.path.join(dirpath, "chroma")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="multi_rag_documents",
            metadata={"hnsw:space": "cosine"},
        )

        try:
            self._load_from_chroma()
        except Exception as e:
            print(f"WARNING: Chroma DB failed to load: {e}. Re-creating collection...")
            try:
                self.client.delete_collection("multi_rag_documents")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name="multi_rag_documents",
                metadata={"hnsw:space": "cosine"},
            )
            self._reset_memory()

    def _load_from_chroma(self) -> None:
        data = self.collection.get(include=["embeddings", "documents", "metadatas"])
        ids = data.get("ids", [])
        embeddings = data.get("embeddings", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        if not ids:
            self._reset_memory()
            return

        self.documents = []
        for doc_text, meta in zip(documents, metadatas):
            orig_meta = dict(meta) if meta else {}
            if "images" in orig_meta and isinstance(orig_meta["images"], str):
                try:
                    orig_meta["images"] = json.loads(orig_meta["images"])
                except Exception:
                    orig_meta["images"] = []

            self.documents.append(
                Document(
                    text=doc_text,
                    source=orig_meta.get("source", ""),
                    kind=orig_meta.get("kind", ""),
                    meta=orig_meta,
                )
            )

        if embeddings is not None and len(embeddings) > 0:
            self._matrix = np.array(embeddings, dtype="float32")
        else:
            self._matrix = np.empty((0, self._dim), dtype="float32")

        self._rebuild_bm25()

    def add(self, docs: list[Document], progress_callback=None) -> None:
        if not docs:
            return
        batch_size = 32
        vectors_list = []
        total_docs = len(docs)
        for i in range(0, total_docs, batch_size):
            batch = docs[i:i + batch_size]
            batch_vecs = embed([d.text for d in batch])
            vectors_list.append(batch_vecs)
            if progress_callback:
                processed = min(i + batch_size, total_docs)
                progress_callback(processed, total_docs, f"Embedding chunk {processed} of {total_docs}")
        vectors = np.vstack(vectors_list) if vectors_list else np.empty((0, self._dim), dtype="float32")

        ids, embeddings, documents, metadatas = [], [], [], []
        for doc, vector in zip(docs, vectors):
            ids.append(str(uuid.uuid4()))
            embeddings.append(vector.tolist())
            documents.append(doc.text)

            meta = dict(doc.meta)
            meta["source"] = doc.source
            meta["kind"] = doc.kind
            for k, v in list(meta.items()):
                if isinstance(v, (list, dict, set)):
                    meta[k] = json.dumps(v)
            metadatas.append(meta)

        self.collection.add(
            embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
        )

        self.documents.extend(docs)
        self._matrix = np.vstack([self._matrix, vectors])
        self._rebuild_bm25()

    def remove_source(self, source: str, chat_id: str | None = None) -> int:
        def match(d: Document) -> bool:
            return d.source == source and (
                chat_id is None or d.meta.get("chat_id") == chat_id
            )

        removed = sum(1 for d in self.documents if match(d))
        if removed:
            if chat_id is None:
                where = {"source": source}
            else:
                where = {"$and": [{"source": source}, {"chat_id": chat_id}]}
            self.collection.delete(where=where)
            keep = [i for i, d in enumerate(self.documents) if not match(d)]
            self.documents = [self.documents[i] for i in keep]
            self._matrix = (
                self._matrix[keep] if keep else np.empty((0, self._dim), dtype="float32")
            )
            self._rebuild_bm25()
        return removed

    def clear(self) -> None:
        """Delete the whole collection and recreate it empty."""
        try:
            self.client.delete_collection("multi_rag_documents")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="multi_rag_documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._reset_memory()


# --------------------------------------------------------------------------- #
# Public factory — picks the backend, with Weaviate->Chroma fallback
# --------------------------------------------------------------------------- #
def _make_store(dirpath: str) -> _BaseStore:
    backend = getattr(config, "VECTOR_BACKEND", "weaviate").lower()
    if backend == "weaviate":
        try:
            return _WeaviateStore(dirpath)
        except Exception as e:
            print(
                f"WARNING: Weaviate backend unavailable ({e!r}). "
                f"Falling back to embedded Chroma. "
                f"Start Weaviate with `docker compose up -d` to use it."
            )
    return _ChromaStore(dirpath)


class VectorStore:
    """Backend-agnostic entry point used by the app.

    `VectorStore(...)` returns a ready store instance for the configured backend
    (Weaviate by default, Chroma as fallback). It's a factory: the object you
    get back is a `_WeaviateStore`/`_ChromaStore`, both of which expose the same
    API (add / search / remove_source / save / size / sources / backend).
    """

    def __new__(cls, dirpath: str = config.INDEX_DIR) -> "_BaseStore":
        return _make_store(dirpath)

    @classmethod
    def load(cls, dirpath: str = config.INDEX_DIR) -> "_BaseStore | None":
        """Load a previously saved index, or None if there isn't one yet."""
        try:
            store = _make_store(dirpath)
        except Exception:
            return None
        return store if store.size > 0 else None
