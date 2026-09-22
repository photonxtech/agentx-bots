"""Local, file-persisted vector store using ChromaDB + sentence-transformers.

Every chunk is tagged with who shared it AND where. Retrieval is scoped
by two rules: you can always retrieve anything you personally shared
(DM or channel, from anywhere you ask), and anyone asking inside a
specific channel can also retrieve anything shared IN that channel by
anyone — so a file someone drops in #team is answerable by their
teammates asking in that same channel, while a DM stays private to the
person who shared it.
"""

import logging
import uuid

import chromadb
from chromadb.utils import embedding_functions

log = logging.getLogger("knowledgebot")


class MemoryStore:
    def __init__(
        self,
        persist_path: str,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        relevance_threshold: float = 0.45,
    ):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        # Cosine distance (0 = identical, 2 = opposite) makes the relevance
        # threshold below meaningful and consistent regardless of embedding
        # model. Only takes effect on a freshly created collection — if
        # ./chroma_data already exists from before this was added, delete
        # it once to pick this up (you'll just re-share your links/docs).
        self.collection = self.client.get_or_create_collection(
            name="knowledge",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.relevance_threshold = relevance_threshold

    def add_document(
        self,
        user_id: str,
        chunks: list[str],
        source: str,
        source_type: str,
        title: str = "",
        channel_id: str = "",
        is_shared: bool = False,
        content_hash: str = "",
    ) -> int:
        """Embeds and stores chunks for one ingested document/link/image.

        channel_id: where it was shared (a Slack channel or DM id).
        is_shared: True if shared in a channel (visible to that channel),
        False if shared in a DM (private to user_id only).
        content_hash: identifies the actual content (not just the filename
        or URL string) — used by has_content() to detect true duplicates.
        """
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "user_id": user_id,
                "source": source,
                "source_type": source_type,  # "link" | "pdf" | "docx" | "txt" | "image" | ...
                "title": title or source,
                "chunk_index": i,
                "channel_id": channel_id,
                "is_shared": is_shared,
                "content_hash": content_hash,
                "is_summary": i == 0,  # chunk 0 is always the summary, by convention
            }
            for i in range(len(chunks))
        ]

        self.collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def has_content(self, user_id: str, channel_id: str, content_hash: str) -> bool:
        """Checks whether this exact content is already stored for this
        user in this scope, so re-sharing the same link/file doesn't pile
        up duplicate chunks (which would waste storage and skew retrieval
        toward whatever's been re-shared most, rather than what's most
        relevant).
        """
        if not content_hash:
            return False
        results = self.collection.get(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"channel_id": channel_id},
                    {"content_hash": content_hash},
                ]
            },
            limit=1,
        )
        return len(results["ids"]) > 0

    def query(self, user_id: str, question: str, channel_id: str, top_k: int = 5) -> list[dict]:
        """Retrieves chunks the asker is allowed to see: anything they
        personally shared, plus anything shared in this specific channel
        by anyone.
        """
        scope_filter = {
            "$or": [
                {"user_id": user_id},
                {"$and": [{"channel_id": channel_id}, {"is_shared": True}]},
            ]
        }

        results = self.collection.query(
            query_texts=[question], n_results=top_k, where=scope_filter
        )

        if not results["documents"] or not results["documents"][0]:
            log.info("Vector search returned no candidates at all (nothing stored in scope)")
            return []

        # Logged unconditionally (not just on failure) so the actual
        # distances are visible in the terminal for calibrating
        # RELEVANCE_THRESHOLD against real data, not guesswork.
        log.info(f"Raw candidate distances: {[round(d, 3) for d in results['distances'][0]]}")

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            if dist > self.relevance_threshold:
                continue
            hits.append({"text": doc, "metadata": meta, "distance": dist})

        if hits:
            return hits

        # Nothing cleared the bar via pure similarity — this is exactly the
        # failure mode for broad/meta questions ("what does this contain")
        # asked right after sharing something: the phrasing doesn't embed
        # close to topical content even when a summary chunk exists that
        # would answer it well. Fall back to summary chunks specifically,
        # with a looser bar, rather than flatly saying "nothing relevant"
        # when a summary is sitting right there in scope.
        summary_filter = {"$and": [scope_filter, {"is_summary": True}]}
        fallback = self.collection.query(
            query_texts=[question], n_results=top_k, where=summary_filter
        )

        if not fallback["documents"] or not fallback["documents"][0]:
            return []

        log.info(
            f"No hits above threshold; falling back to summary chunks, "
            f"distances: {[round(d, 3) for d in fallback['distances'][0]]}"
        )

        fallback_threshold = self.relevance_threshold * 1.6  # deliberately looser
        for doc, meta, dist in zip(
            fallback["documents"][0], fallback["metadatas"][0], fallback["distances"][0]
        ):
            if dist > fallback_threshold:
                continue
            hits.append({"text": doc, "metadata": meta, "distance": dist})

        return hits

    def has_any_knowledge(self, user_id: str) -> bool:
        results = self.collection.get(where={"user_id": user_id}, limit=1)
        return len(results["ids"]) > 0