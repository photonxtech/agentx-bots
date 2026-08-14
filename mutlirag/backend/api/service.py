"""RagService — the stateful core shared by every API request.

Streamlit kept the vector index in `st.session_state`, reloaded per browser
session, and used `current_chat_id` as implicit state. An API has none of that,
so this service does the FastAPI-appropriate equivalent:

  * The ChromaDB-backed VectorStore is loaded ONCE (at app startup, see
    `main.lifespan`) and shared across all requests — not per user.
  * All INDEX WRITES (add / remove / clear) and the in-memory SEARCH are guarded
    by a single re-entrant lock, because that shared store is mutated in place
    and concurrent requests would otherwise corrupt the NumPy matrix / BM25 index.
  * `chat_id` is passed explicitly on every call (no "current chat").
  * Extracted images are namespaced per chat so two chats can't overwrite each
    other's files.

The retrieval logic (rewrite -> hybrid search -> type/source filtering ->
confidence) is ported faithfully from the Streamlit `app.py` so answers are
identical to the UI.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid

import config
import chat_store
import db
from rag import chunking, evaluation, generator, golden_set, ingestion, langsmith_logging, reranker
from rag.ingestion import Document
from rag.query_intent import CLARIFY_QUERY, NORMAL_QUERY, PAGE_QUERY, TOC_QUERY, detect_intent
from rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Question keywords that hint at a desired source *type*, used to bias retrieval
# toward the right kind of chunk when a chat somehow holds more than one source.
_TYPE_KEYWORDS = {
    "pptx":  ["ppt", "powerpoint", "presentation", "slide", "slides"],
    "pdf":   ["pdf"],
    "docx":  ["word", "docx", "document"],
    "image": ["image", "photo", "picture", "png", "jpg", "jpeg"],
    "txt":   ["text file", "txt", "markdown", "md"],
}

# --------------------------------------------------------------------------- #
# Small talk — greetings and pleasantries are answered directly, WITHOUT going
# through retrieval/generation, so a friendly "hi" never becomes "I don't know".
# We only treat a message as small talk when the WHOLE message is a greeting
# (so "hello, what does the report say?" still runs the normal RAG flow).
# --------------------------------------------------------------------------- #
_GREETINGS = {
    "hi", "hii", "hiii", "hiya", "hello", "helloo", "hellooo", "hey", "heyy",
    "heya", "hey there", "hello there", "hi there", "yo", "sup", "wassup",
    "whats up", "whatsup", "good morning", "good afternoon", "good evening",
    "good night", "gm", "gn", "greetings", "howdy", "hola", "namaste",
}
_HOWAREYOU = {
    "how are you", "how are you doing", "how r u", "how are u", "hows it going",
    "hows it goin", "how is it going", "whats good",
}
_THANKS = {
    "thanks", "thank you", "thankyou", "thank u", "thx", "ty", "thanks a lot",
    "thank you so much", "much appreciated", "appreciate it",
}
_BYES = {
    "bye", "goodbye", "good bye", "see you", "see ya", "cya", "take care", "later",
}
_ABOUT = {
    "who are you", "what are you", "what can you do", "what do you do",
    "what is this", "how do you work", "help", "what can i ask", "what can i ask you",
}


def smalltalk_reply(question: str, has_file: bool) -> str | None:
    """Return a friendly canned reply if `question` is pure small talk, else None."""
    q = re.sub(r"[^\w\s]", "", (question or "").strip().lower()).strip()
    q = re.sub(r"\s+", " ", q)
    if not q:
        return None
    if q in _GREETINGS:
        if has_file:
            return ("Hello! 👋 I'm your document assistant. Ask me anything about "
                    "the file in this chat and I'll answer from it.")
        return ("Hello! 👋 I'm your document assistant. Upload a file to this chat "
                "and I'll answer your questions grounded in it.")
    if q in _HOWAREYOU:
        return ("I'm doing great, thanks for asking! 😊 Ask me anything about your "
                "uploaded document.")
    if q in _THANKS:
        return "You're welcome! 😊 Happy to help — ask me anything else about your document."
    if q in _BYES:
        return "Goodbye! 👋 Come back anytime to ask about your documents."
    if q in _ABOUT:
        return ("I'm a document Q&A assistant. Upload a file (PDF, Word, PowerPoint, "
                "text, or image) to a chat, and I'll answer your questions grounded "
                "strictly in that document — with sources.")
    return None


class ChatNotFoundError(Exception):
    """Raised when an operation targets a chat id that does not exist."""


class NoDocumentError(Exception):
    """Raised when a chat is asked a question but has no indexed file."""


class MessageNotFoundError(Exception):
    """Raised when an operation targets a message id that does not exist in the chat."""


class RagService:
    """Owns the shared vector store + chat registry for the whole process."""

    def __init__(self) -> None:
        # One store for the whole process, hydrated from disk (Chroma). Using
        # VectorStore() (not .load) guarantees a non-None store even when empty.
        self.store = VectorStore()
        # Chat metadata, kept in memory and persisted to disk on every mutation
        # via chat_store (which writes the whole dict). Mirrors app.py's state.
        self.chats: dict = chat_store.load_chats()
        # Guards index writes AND the in-memory hybrid search (both touch the
        # shared NumPy matrix / BM25 index). Re-entrant so nested calls are safe.
        self._lock = threading.RLock()

    # --------------------------------------------------------------------- #
    # Chats
    # --------------------------------------------------------------------- #
    def create_chat(self, title: str = "New Chat") -> str:
        with self._lock:
            return chat_store.create_chat(self.chats, title=title)

    def get_chat(self, chat_id: str) -> dict:
        chat = self.chats.get(chat_id)
        if not chat:
            raise ChatNotFoundError(chat_id)
        return chat

    def list_chats(self) -> list[dict]:
        """All chats, newest-updated first, each annotated with file info."""
        ordered = sorted(
            self.chats.values(), key=lambda c: c.get("updated_at", ""), reverse=True
        )
        return [self._annotate(c) for c in ordered]

    def delete_chat(self, chat_id: str) -> None:
        """Delete a chat and drop its indexed chunks from the store."""
        if chat_id not in self.chats:
            raise ChatNotFoundError(chat_id)
        with self._lock:
            self.store.remove_chat(chat_id)
            self.store.save(config.INDEX_DIR)
            chat_store.delete_chat(self.chats, chat_id)

    def _annotate(self, chat: dict) -> dict:
        """Attach file / chunk_count / message_count to a raw chat dict."""
        cid = chat["id"]
        srcs = self.store.sources_for_chat(cid)
        return {
            "id": cid,
            "title": chat.get("title", "New Chat"),
            "created_at": chat.get("created_at", ""),
            "updated_at": chat.get("updated_at", ""),
            "message_count": len(chat.get("messages", [])),
            "file": srcs[0] if srcs else None,
            "chunk_count": self.store.size_for_chat(cid),
        }

    def chat_detail(self, chat_id: str) -> dict:
        chat = self.get_chat(chat_id)
        detail = self._annotate(chat)
        detail["messages"] = chat.get("messages", [])
        return detail

    def chat_file(self, chat_id: str) -> str | None:
        srcs = self.store.sources_for_chat(chat_id)
        return srcs[0] if srcs else None

    # --------------------------------------------------------------------- #
    # Ingestion (upload)
    # --------------------------------------------------------------------- #
    def ingest_file(self, chat_id: str, file_bytes: bytes, filename: str) -> dict:
        """Ingest + chunk + tag + index one file into a chat. Returns a summary.

        Mirrors app.index_file_for_chat, minus the Streamlit status widget.
        """
        self.get_chat(chat_id)  # validate chat exists (raises ChatNotFoundError)

        docs = ingestion.ingest_cached(file_bytes, filename)
        chunks = chunking.chunk_documents(docs)
        for c in chunks:
            c.meta["chat_id"] = chat_id  # tag for per-chat isolation

        if not chunks:
            return {
                "chat_id": chat_id,
                "file": filename,
                "chunks_indexed": 0,
                "backend": self.store.backend,
                "message": "No extractable content found in the file.",
            }

        # Index write — serialize against other writers/searchers.
        with self._lock:
            replaced = self._replace_existing_file(chat_id)
            self.store.add(chunks)
            self.store.save(config.INDEX_DIR)
            indexed = self.store.size_for_chat(chat_id)

        message = f"Indexed {filename} — {indexed} chunk(s) for this chat."
        if replaced:
            message = f"Replaced {replaced!r} with {filename} — {indexed} chunk(s) indexed."

        return {
            "chat_id": chat_id,
            "file": filename,
            "chunks_indexed": len(chunks),
            "backend": self.store.backend,
            "message": message,
        }

    def _replace_existing_file(self, chat_id: str) -> str | None:
        """A chat holds exactly one file at a time (see chat_file/_annotate) —
        uploading a new one REPLACES whatever was there, instead of
        accumulating alongside it. Without this, re-uploading the same file
        (or a revised version of it) leaves the old chunks in the index
        forever, so the same content ends up indexed multiple times and
        duplicate chunks compete for retrieval's top_k slots.

        Must be called under self._lock, after confirming the new upload
        actually produced chunks (never delete the old file over a failed
        upload) and before the new ones are added. Returns the removed
        filename, or None if the chat had no file yet.
        """
        existing = self.store.sources_for_chat(chat_id)
        for src in existing:
            self.store.remove_source(src, chat_id=chat_id)
        return existing[0] if existing else None

    def ingest_file_stream(self, chat_id: str, file_bytes: bytes, filename: str):
        """Generator yielding real-time indexing progress events (0-100%, stage, ETA)."""
        self.get_chat(chat_id)

        start_time = time.time()
        parse_times = []
        last_time = [time.time()]

        yield {
            "status": "processing",
            "stage": "parsing",
            "current": 0,
            "total": 1,
            "percent": 0,
            "eta_seconds": 0,
            "message": f"Starting ingestion for {filename}..."
        }

        # Step 1: Parsing & OCR
        events_queue = []

        def on_parse_progress(current, total, msg):
            now = time.time()
            dt = now - last_time[0]
            last_time[0] = now
            if dt > 0.005 and current > 1:
                parse_times.append(dt)
            avg_page_time = (sum(parse_times) / len(parse_times)) if parse_times else 0.4
            remaining_pages = max(0, total - current)
            est_parse_rem = remaining_pages * avg_page_time

            pct = int(round((current / max(1, total)) * 50))
            est_embed_time = max(0.5, total * 0.04)
            total_eta = round(est_parse_rem + est_embed_time, 1)

            events_queue.append({
                "status": "processing",
                "stage": "parsing",
                "current": current,
                "total": total,
                "percent": min(50, pct),
                "eta_seconds": total_eta,
                "message": f"{msg} (~{int(total_eta)}s left)"
            })

        docs = ingestion.ingest_cached(file_bytes, filename, progress_callback=on_parse_progress)

        for ev in events_queue:
            yield ev
        events_queue.clear()

        # Step 2: Chunking (50% mark)
        yield {
            "status": "processing",
            "stage": "chunking",
            "current": len(docs),
            "total": len(docs),
            "percent": 50,
            "eta_seconds": round(max(0.5, len(docs) * 0.03), 1),
            "message": f"Splitting document into search chunks..."
        }

        chunks = chunking.chunk_documents(docs)
        for c in chunks:
            c.meta["chat_id"] = chat_id

        if not chunks:
            yield {
                "status": "error",
                "percent": 100,
                "message": "No extractable content found in the file."
            }
            return

        # Step 3: Embedding generation (50% to 90%)
        embed_start = time.time()

        def on_embed_progress(current, total, msg):
            pct = 50 + int(round((current / max(1, total)) * 40))
            elapsed = time.time() - embed_start
            rate = current / max(0.001, elapsed)
            rem_chunks = max(0, total - current)
            rem_eta = round(rem_chunks / max(0.1, rate), 1)
            events_queue.append({
                "status": "processing",
                "stage": "embedding",
                "current": current,
                "total": total,
                "percent": min(90, pct),
                "eta_seconds": rem_eta,
                "message": f"{msg} (~{int(rem_eta)}s left)"
            })

        with self._lock:
            replaced = self._replace_existing_file(chat_id)
            self.store.add(chunks, progress_callback=on_embed_progress)
            self.store.save(config.INDEX_DIR)
            indexed = self.store.size_for_chat(chat_id)

        for ev in events_queue:
            yield ev
        events_queue.clear()

        # Step 4: Completion (100%)
        total_elapsed = round(time.time() - start_time, 1)
        completion_message = f"Indexed {filename} — {indexed} chunk(s) ready in {total_elapsed}s!"
        if replaced:
            completion_message = (
                f"Replaced {replaced!r} with {filename} — {indexed} chunk(s) ready in {total_elapsed}s!"
            )
        yield {
            "status": "completed",
            "stage": "indexing",
            "current": len(chunks),
            "total": len(chunks),
            "percent": 100,
            "eta_seconds": 0,
            "chunks_indexed": len(chunks),
            "message": completion_message,
        }

    def remove_file(self, chat_id: str, filename: str | None = None) -> str:
        """Remove this chat's file (defaults to whatever file it holds)."""
        self.get_chat(chat_id)
        target = filename or self.chat_file(chat_id)
        if not target:
            raise NoDocumentError(chat_id)
        with self._lock:
            self.store.remove_source(target, chat_id=chat_id)
            self.store.save(config.INDEX_DIR)
        return target

    # --------------------------------------------------------------------- #
    # Retrieval — ported from app.py's `if question:` block
    # --------------------------------------------------------------------- #
    def retrieve(self, chat_id: str, question: str, history: list[dict]) -> dict:
        """Route by query intent, then rewrite/search/rerank (NORMAL_QUERY) or
        deterministically pull TOC/page chunks straight from metadata
        (TOC_QUERY/PAGE_QUERY — see rag.query_intent), and format.

        Returns a dict with: search_query, hits (list[(Document, score)]),
        raw_hits (the pre-rerank candidate pool, for LangSmith's before/after
        view — see rag.langsmith_logging.log_retrieval), sources (list[dict]),
        confidence_pct, rewrite_ms, search_ms, rerank_ms, query_type, and
        direct_answer (str | None — set for TOC_QUERY/PAGE_QUERY when nothing
        matched, so the caller can skip the LLM and answer with a clear
        "not found" message instead of risking a hallucinated one).
        """
        if self.store.size_for_chat(chat_id) == 0:
            raise NoDocumentError(chat_id)

        intent = detect_intent(question)

        if intent.kind == TOC_QUERY:
            return self._retrieve_toc(chat_id)
        if intent.kind == PAGE_QUERY:
            return self._retrieve_pages(chat_id, intent.pages)

        logger.info("query_type=%s exclude_toc=%s", NORMAL_QUERY, config.EXCLUDE_INDEX_PAGES)

        # 1. Understand the follow-up: standalone / contextual (both need a
        #    new search) / clarify (asking to rephrase/simplify/elaborate on
        #    the PREVIOUS answer — no new search at all, see
        #    _retrieve_clarification).
        t0 = time.perf_counter()
        understanding = generator.classify_followup(question, history)
        rewrite_ms = int((time.perf_counter() - t0) * 1000)

        if understanding["mode"] == "clarify":
            clarification = self._retrieve_clarification(history, question, rewrite_ms)
            if clarification is not None:
                return clarification
            # No usable previous answer to clarify (e.g. this is actually the
            # first turn) -> fall through and treat it as a normal question.

        search_q = understanding["query"]

        # 2. Hybrid retrieve a WIDE candidate pool, scoped to THIS chat.
        #    candidate_k (fed to the reranker) is deliberately kept distinct
        #    from final_k (what actually reaches the LLM) — see
        #    config.RETRIEVAL_CANDIDATE_K/FINAL_CONTEXT_K. TOC/index pages are
        #    excluded here (config.EXCLUDE_INDEX_PAGES), inside
        #    store.search() — untouched by the TOC/PAGE routing above.
        num_sources = len(self.store.sources_for_chat(chat_id)) or 1
        final_k = max(config.FINAL_CONTEXT_K, min(num_sources * 2, 30))
        candidate_k = max(config.RETRIEVAL_CANDIDATE_K, final_k)

        t1 = time.perf_counter()
        with self._lock:  # search touches the shared matrix/BM25 index
            explained = self.store.search(
                search_q, top_k=candidate_k, chat_id=chat_id, explain=True
            )
            raw_hits = [(r["doc"], r["score"]) for r in explained]
            search_ms = int((time.perf_counter() - t1) * 1000)

            # 2b. Neighbor/parent expansion: a chunk can be essential
            #     evidence yet score far outside ANY reasonable candidate
            #     pool on its own wording (confirmed live: a real
            #     "Guideline #3" chunk ranked ~140th for its own query,
            #     entirely below candidate_k). For each of the top-scoring
            #     candidates, also pull its immediately adjacent chunk(s)
            #     from the same page/section — see vectorstore.get_neighbors.
            #     Timed separately from the hybrid search above so a slow
            #     expansion pass is distinguishable from a slow vector DB.
            t_expand = time.perf_counter()
            expanded_keys: set[int] = set()
            if config.NEIGHBOR_EXPANSION_WINDOW > 0 and raw_hits:
                seen_texts = {d.text for d, _ in raw_hits}
                added: list[tuple[Document, float]] = []
                for d, _ in raw_hits[: config.NEIGHBOR_EXPANSION_MAX_ANCHORS]:
                    if len(added) >= config.NEIGHBOR_EXPANSION_MAX_ADDED:
                        break
                    for nb in self.store.get_neighbors(
                        d, window=config.NEIGHBOR_EXPANSION_WINDOW
                    ):
                        if nb.text in seen_texts:
                            continue
                        if len(added) >= config.NEIGHBOR_EXPANSION_MAX_ADDED:
                            break
                        seen_texts.add(nb.text)
                        added.append((nb, 0.0))
                        expanded_keys.add(id(nb))
                raw_hits = raw_hits + added
        neighbor_expansion_ms = int((time.perf_counter() - t_expand) * 1000)

        # 3. Rerank the (candidate + expanded) pool with a cross-encoder
        #    (query+chunk scored jointly, far better than hybrid search's
        #    fixed blend). Neighbor-expanded chunks are exempt from the
        #    RERANK_MIN_SCORE floor — they were added for structural
        #    completeness, not because they're expected to score well
        #    standing alone.
        t2 = time.perf_counter()
        hits = (
            reranker.rerank(search_q, raw_hits, protect=expanded_keys)
            if config.RERANK_ENABLED
            else raw_hits
        )
        rerank_ms = int((time.perf_counter() - t2) * 1000)

        # 4. Diversity pass (MMR) so several near-duplicate restatements
        #    don't crowd out genuinely different evidence — same protect
        #    exemption as above — then the existing type/source-keyword
        #    bias. Timed separately from the cross-encoder scoring above so
        #    a slow reranker model is distinguishable from a slow selection
        #    pass.
        t3 = time.perf_counter()
        hits = reranker.diversity_select(hits, final_k, protect=expanded_keys)
        hits = self._filter_hits(question, search_q, hits, chat_id, final_k)
        diversity_ms = int((time.perf_counter() - t3) * 1000)

        diagnostics = self._build_retrieval_diagnostics(explained, expanded_keys, hits)
        self._log_retrieval_diagnostics(search_q, diagnostics)

        # 5. Confidence = top score (0..1) -> percentage.
        top_score = hits[0][1] if hits else 0.0
        confidence_pct = int(round(min(top_score, 1.0) * 100))

        return {
            "search_query": search_q,
            "hits": hits,
            "raw_hits": raw_hits,
            "sources": self._build_sources(hits),
            "confidence_pct": confidence_pct,
            "rewrite_ms": rewrite_ms,
            "search_ms": search_ms,
            "neighbor_expansion_ms": neighbor_expansion_ms,
            "rerank_ms": rerank_ms,
            "diversity_ms": diversity_ms,
            "query_type": NORMAL_QUERY,
            "direct_answer": None,
            "retrieval_diagnostics": diagnostics,
            "retrieval_meta": {
                "embedding_model": config.EMBEDDING_MODEL,
                "rerank_model": config.RERANK_MODEL,
                "candidate_k": candidate_k,
                "final_k": final_k,
                "expanded_count": len(expanded_keys),
            },
        }

    def _retrieve_clarification(
        self, history: list[dict], question: str, rewrite_ms: int
    ) -> dict | None:
        """Answer a "make it simpler / explain more / give an example"
        follow-up by reusing the immediately preceding answer's OWN retrieved
        context, instead of running a new search.

        Confirmed live: a follow-up like "more clearly" has no topical
        content of its own, so a fresh hybrid search on it retrieves
        whatever text in the document happens to also literally discuss
        clarity/writing — nothing to do with what was actually being
        discussed — and produces a wrong, unrelated answer. The previous
        turn's context was already the right grounding; reusing it verbatim
        can't drift onto a different topic the way a brand-new search can.

        Returns None if there's no usable previous assistant answer to
        clarify (e.g. this is actually the first turn), so the caller can
        fall back to treating this as an ordinary new question.
        """
        prev_assistant = next(
            (m for m in reversed(history) if m.get("role") == "assistant" and m.get("contexts")),
            None,
        )
        if prev_assistant is None:
            return None
        prev_user = next((m for m in reversed(history) if m.get("role") == "user"), None)

        hits = [
            (Document(text=ctx, source="(previous answer)", kind="text", meta={}), 1.0)
            for ctx in prev_assistant["contexts"]
        ]
        if not hits:
            return None

        search_query = (
            f'{question} (follow-up asking to clarify/rephrase/expand your previous '
            f'answer to: "{prev_user["content"]}")'
            if prev_user else question
        )

        return {
            "search_query": search_query,
            "hits": hits,
            "raw_hits": hits,
            "sources": prev_assistant.get("sources") or [],
            "confidence_pct": 100,
            "rewrite_ms": rewrite_ms,
            "search_ms": 0,
            "rerank_ms": 0,
            "query_type": CLARIFY_QUERY,
            "direct_answer": None,
            "retrieval_diagnostics": None,
        }

    def _build_retrieval_diagnostics(
        self,
        explained: list[dict],
        expanded_keys: set[int],
        final_hits: list[tuple[Document, float]],
    ) -> list[dict]:
        """One row per FINAL chunk — rank/chunk_id/page/section/chunk_type/
        dense/bm25/hybrid/rerank score/whether it was neighbor-expanded — so
        "did the required evidence enter the candidate pool, and if so did
        reranking/filtering drop it" is answerable by reading one table
        instead of guessing. Cheap (dict lookups + float rounding, no extra
        model calls), so always built — this is what gets attached to the
        rerank_documents span in LangSmith (see
        rag.langsmith_logging.log_retrieval) as well as the DEBUG console log
        below. Only a short text preview is kept, never the full chunk text.
        """
        components = {id(r["doc"]): (r["dense"], r["bm25"], r["score"]) for r in explained}
        rows = []
        for rank, (d, score) in enumerate(final_hits, 1):
            dense, bm25, hybrid = components.get(id(d), (None, None, None))
            parent_text = d.meta.get("parent_text") or d.text
            rows.append({
                "rank": rank,
                "chunk_id": f"{d.meta.get('page')}:{d.meta.get('chunk')}:{d.meta.get('child')}",
                "page": d.meta.get("page"),
                "section": d.meta.get("section") or "",
                "chunk_type": d.meta.get("chunk_type", "?"),
                "dense_score": round(dense, 4) if dense is not None else None,
                "bm25_score": round(bm25, 4) if bm25 is not None else None,
                "hybrid_score": round(hybrid, 4) if hybrid is not None else None,
                "reranker_score": round(score, 4),
                "neighbor_expanded": id(d) in expanded_keys,
                "text_preview": d.text[:120].replace("\n", " "),
                # Parent/child visibility: `chunk_id`'s "page:chunk" prefix is
                # the PARENT identity (one PARENT_CHUNK_SIZE-character section
                # of the document); "child" is this hit's own small
                # CHILD_CHUNK_SIZE-character match within it. `used_parent_context`
                # confirms whether the LLM actually got the wider parent_text
                # instead of just the narrow child match that scored well —
                # false for chunks indexed before parent-child chunking existed.
                "parent_id": f"{d.meta.get('page')}:{d.meta.get('chunk')}",
                "child_index": d.meta.get("child"),
                "child_chars": len(d.text),
                "parent_chars": len(parent_text),
                "used_parent_context": parent_text != d.text,
            })
        return rows

    def _log_retrieval_diagnostics(self, query: str, rows: list[dict]) -> None:
        """Debug-only console rendering of `rows` as a fixed-width table.
        No-op unless DEBUG logging is enabled — never adds noise to
        production logs by default. The LangSmith copy of the same rows
        (see `retrieval_diagnostics` in retrieve()'s return dict) is sent
        regardless of log level, since that's a UI the user opts into per
        question rather than a firehose console stream.
        """
        if not logger.isEnabledFor(logging.DEBUG) or not rows:
            return
        lines = [f"retrieval diagnostics query={query!r}"]
        lines.append(
            f"{'rank':<5}{'chunk_id':<14}{'page':<6}{'type':<8}"
            f"{'dense':<8}{'bm25':<8}{'hybrid':<8}{'rerank':<8}{'expanded':<9}section / text"
        )
        for r in rows:
            dense_s = "-" if r["dense_score"] is None else f"{r['dense_score']:.3f}"
            bm25_s = "-" if r["bm25_score"] is None else f"{r['bm25_score']:.3f}"
            hybrid_s = "-" if r["hybrid_score"] is None else f"{r['hybrid_score']:.3f}"
            lines.append(
                f"{r['rank']:<5}{r['chunk_id']:<14}{str(r['page']):<6}"
                f"{r['chunk_type']:<8}{dense_s:<8}{bm25_s:<8}{hybrid_s:<8}"
                f"{r['reranker_score']:<8.3f}{str(r['neighbor_expanded']):<9}"
                f"{r['section']} | {r['text_preview'][:50]}"
            )
        logger.debug("\n".join(lines))

    def _build_sources(self, hits: list[tuple[Document, float]]) -> list[dict]:
        return [
            {
                "index": i,
                "source": d.source,
                "kind": d.kind,
                "score": round(float(s), 4),
                "text": d.text[:400],
                "images": [p for p in (d.meta.get("images") or []) if p],
            }
            for i, (d, s) in enumerate(hits, 1)
        ]

    def _retrieve_toc(self, chat_id: str) -> dict:
        """Deterministic TOC retrieval (TOC_QUERY): pull every chunk from
        pages flagged TOC/index for this chat, in page order, bypassing the
        exclusion mask entirely — "give me the TOC" is a structural request,
        not a semantic-similarity one, so it shouldn't depend on whether
        embedding/BM25 search happens to rank the TOC page highly.
        """
        t0 = time.perf_counter()
        with self._lock:
            toc_docs = self.store.get_toc_chunks(chat_id=chat_id)
        search_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("query_type=%s toc_pages=%s", TOC_QUERY,
                    sorted({d.meta.get("page") for d in toc_docs if d.meta.get("page") is not None}))

        hits = [(d, 1.0) for d in toc_docs]
        direct_answer = None
        if not hits:
            direct_answer = (
                "I couldn't find a table of contents in this document. "
                "It may not have one, or it wasn't detected during ingestion "
                "(older documents may need re-ingestion for TOC detection)."
            )

        return {
            "search_query": "table of contents",
            "hits": hits,
            "raw_hits": hits,
            "sources": self._build_sources(hits),
            "confidence_pct": 100 if hits else 0,
            "rewrite_ms": 0,
            "search_ms": search_ms,
            "rerank_ms": 0,
            "query_type": TOC_QUERY,
            "direct_answer": direct_answer,
        }

    def _retrieve_pages(self, chat_id: str, pages: list[int]) -> dict:
        """Deterministic page retrieval (PAGE_QUERY): pull exactly the
        requested page(s) for this chat, regardless of is_index — an
        explicit page request always wins over the generic TOC-exclusion
        rule for NORMAL_QUERY.
        """
        t0 = time.perf_counter()
        with self._lock:
            page_docs = self.store.get_by_pages(pages, chat_id=chat_id)
        search_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("query_type=%s requested_pages=%s found_chunks=%d",
                    PAGE_QUERY, pages, len(page_docs))

        hits = [(d, 1.0) for d in page_docs]
        direct_answer = None
        if not hits:
            label = "Page" if len(pages) == 1 else "Pages"
            pages_str = ", ".join(str(p) for p in pages)
            direct_answer = f"{label} {pages_str} could not be found in this document."

        return {
            "search_query": f"page {', '.join(str(p) for p in pages)}",
            "hits": hits,
            "raw_hits": hits,
            "sources": self._build_sources(hits),
            "confidence_pct": 100 if hits else 0,
            "rewrite_ms": 0,
            "search_ms": search_ms,
            "rerank_ms": 0,
            "query_type": PAGE_QUERY,
            "direct_answer": direct_answer,
        }

    def _filter_hits(self, question, search_q, hits, chat_id, final_k):
        """Type/source-aware narrowing of the retrieved pool (from app.py)."""
        q_lower = (question + " " + search_q).lower()

        wanted_kind = None
        for kind, kws in _TYPE_KEYWORDS.items():
            if any(kw in q_lower for kw in kws):
                wanted_kind = kind
                break

        wanted_source = None
        if not wanted_kind:
            q_words = set(re.split(r"\W+", q_lower))
            for src in self.store.sources_for_chat(chat_id):
                src_words = set(re.split(r"[-_.\s]+", src.lower()))
                meaningful = {w for w in (src_words & q_words) if len(w) > 3}
                if len(meaningful) >= 2:
                    wanted_source = src
                    break

        if wanted_kind:
            filtered = [(d, s) for d, s in hits if d.kind == wanted_kind]
            return filtered[:final_k] if filtered else hits[:final_k]
        if wanted_source:
            filtered = [(d, s) for d, s in hits if d.source == wanted_source]
            return filtered[:final_k] if filtered else hits[:final_k]
        return hits[:final_k]

    # --------------------------------------------------------------------- #
    # Message persistence
    # --------------------------------------------------------------------- #
    def append_user_message(self, chat_id: str, question: str) -> list[dict]:
        """Append the user's question and (if first) auto-title the chat."""
        chat = self.get_chat(chat_id)
        messages = chat.get("messages", [])
        if not messages:
            chat["title"] = question[:40] + "..." if len(question) > 40 else question
        messages.append({"id": str(uuid.uuid4()), "role": "user", "content": question})
        with self._lock:
            chat_store.update_chat_messages(self.chats, chat_id, messages)
        return messages

    def append_assistant_message(
        self,
        chat_id: str,
        answer: str,
        sources: list[dict],
        metrics: dict | None,
        question: str = "",
        contexts: list[str] | None = None,
        langsmith_run_id: str | None = None,
    ) -> str:
        """Append the assistant's answer. Returns the new message's id.

        `question`/`contexts`/`langsmith_run_id` are persisted (not shown in
        the UI) purely so evaluate_message() can score this exact turn later,
        on demand — DeepEval no longer runs automatically at answer time, and
        the original in-memory hits/RunTree are long gone by the time a user
        clicks "Calculate Metrics" in a later, separate request.
        """
        chat = self.get_chat(chat_id)
        messages = chat.get("messages", [])
        # "I don't know" answers are stored without sources, matching the UI.
        is_dont_know = "don't know" in answer.lower()
        message_id = str(uuid.uuid4())
        messages.append({
            "id": message_id,
            "role": "assistant",
            "content": answer,
            "sources": [] if is_dont_know else sources,
            "metrics": metrics,
            "question": question,
            "contexts": [] if is_dont_know else (contexts or []),
            "langsmith_run_id": langsmith_run_id,
        })
        with self._lock:
            chat_store.update_chat_messages(self.chats, chat_id, messages)
        return message_id

    # --------------------------------------------------------------------- #
    # Generation
    # --------------------------------------------------------------------- #
    def answer_stream(
        self,
        search_query: str,
        hits,
        original_question: str | None = None,
        verified_context: bool = False,
        usage: dict | None = None,
    ):
        """Yield answer text deltas from Groq (grounded in `hits`).

        `verified_context` — pass True for PAGE_QUERY/TOC_QUERY results (see
        rag.query_intent), whose hits were pulled deterministically by page
        metadata rather than semantic search, so the model should describe
        them directly instead of treating them as a search result to verify.

        `usage` — an optional dict populated in place with token counts once
        the stream finishes (see generator.answer), for cost/latency tracking.
        """
        return generator.answer(
            search_query,
            hits,
            model=config.DEFAULT_MODEL,
            temperature=config.DEFAULT_TEMPERATURE,
            original_question=original_question,
            verified_context=verified_context,
            usage=usage,
        )

    # --------------------------------------------------------------------- #
    # Evaluation (DeepEval metrics) — computed ONLY on demand, when the user
    # clicks "Calculate Metrics" in the UI, via evaluate_message() below. Not
    # run automatically at answer time (human-in-the-loop, since a DeepEval
    # pass costs one judge call per metric and can take 10s of seconds).
    # --------------------------------------------------------------------- #
    def evaluate_answer(self, question: str, answer: str, contexts: list[str]) -> dict | None:
        """Score a produced answer on the three reference-free DeepEval metrics,
        plus context_precision/context_recall/answer_correctness when `question`
        closely matches one of the curated golden_set questions (see rag.golden_set).

        `contexts` should be the same parent-chunk text the answer was actually
        generated from (see generator.context_texts) — persisted on the message
        at generation time so it's still available here, computed on demand,
        long after the original retrieval hits are gone. Returns None when
        evaluation is disabled, there is nothing to score, or the answer was a
        refusal — never raises.
        """
        if not config.DEEPEVAL_ENABLED or not contexts or not answer:
            return None
        if "don't know" in answer.lower():
            return None
        try:
            ground_truth = golden_set.lookup(question)
            if ground_truth:
                return evaluation.evaluate_with_ground_truth(question, answer, ground_truth, contexts)
            return evaluation.evaluate(question, answer, contexts)
        except Exception:
            return None

    def evaluate_message(self, chat_id: str, message_id: str) -> dict | None:
        """Compute DeepEval metrics for one already-answered message, on demand.

        Looks up the persisted question/answer/contexts from chat_store (saved
        by append_assistant_message at generation time), scores them, merges
        the result into that message's existing (timing-only) metrics dict,
        and propagates it to Postgres + LangSmith feedback — same destinations
        the old automatic scoring used to reach, just deferred until now.
        """
        chat = self.get_chat(chat_id)
        messages = chat.get("messages", [])
        message = next((m for m in messages if m.get("id") == message_id), None)
        if message is None or message.get("role") != "assistant":
            raise MessageNotFoundError(message_id)

        ragas = self.evaluate_answer(
            message.get("question", ""), message.get("content", ""), message.get("contexts") or []
        )

        metrics = dict(message.get("metrics") or {})
        if ragas:
            metrics.update(ragas)
        message["metrics"] = metrics
        with self._lock:
            chat_store.update_chat_messages(self.chats, chat_id, messages)

        db.update_qa_metrics(message_id, metrics)
        langsmith_logging.attach_feedback(message.get("langsmith_run_id"), metrics)
        return metrics

    # --------------------------------------------------------------------- #
    # Admin
    # --------------------------------------------------------------------- #
    def reset(self) -> None:
        """Wipe every chat and every indexed chunk (like the UI reset button)."""
        with self._lock:
            try:
                self.store.clear()
            except Exception:
                pass
            self.chats = {}
            chat_store.save_chats({})

    def health(self) -> dict:
        return {
            "status": "ok",
            "backend": self.store.backend,
            "total_chunks": self.store.size,
            "chats": len(self.chats),
        }
