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

import os
import re
import threading
import time

import config
import chat_store
from rag import chunking, generator, ingestion
from rag.ingestion import Document
from rag.vectorstore import VectorStore

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
            self.store.add(chunks)
            self.store.save(config.INDEX_DIR)
            indexed = self.store.size_for_chat(chat_id)

        return {
            "chat_id": chat_id,
            "file": filename,
            "chunks_indexed": len(chunks),
            "backend": self.store.backend,
            "message": f"Indexed {filename} — {indexed} chunk(s) for this chat.",
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
        """Rewrite the query, hybrid-search within the chat, filter, and format.

        Returns a dict with: search_query, hits (list[(Document, score)]),
        sources (list[dict]), confidence_pct, rewrite_ms, search_ms.
        """
        if self.store.size_for_chat(chat_id) == 0:
            raise NoDocumentError(chat_id)

        top_k = config.TOP_K

        # 1. Rewrite follow-ups into a standalone query.
        t0 = time.perf_counter()
        search_q = generator.rewrite_query(question, history)
        rewrite_ms = int((time.perf_counter() - t0) * 1000)

        # 2. Hybrid retrieve, scoped to THIS chat. Scale k with #sources (~1).
        num_sources = len(self.store.sources_for_chat(chat_id)) or 1
        effective_top_k = max(top_k, min(num_sources * 2, 30))

        t1 = time.perf_counter()
        with self._lock:  # search touches the shared matrix/BM25 index
            hits = self.store.search(
                search_q, top_k=effective_top_k * 3, chat_id=chat_id
            )
        search_ms = int((time.perf_counter() - t1) * 1000)

        # 3. Bias by source type or filename keywords (same heuristics as the UI).
        hits = self._filter_hits(question, search_q, hits, chat_id, effective_top_k)

        # 4. Confidence = top blended score (0..1) -> percentage.
        top_score = hits[0][1] if hits else 0.0
        confidence_pct = int(round(min(top_score, 1.0) * 100))

        sources = [
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

        return {
            "search_query": search_q,
            "hits": hits,
            "sources": sources,
            "confidence_pct": confidence_pct,
            "rewrite_ms": rewrite_ms,
            "search_ms": search_ms,
        }

    def _filter_hits(self, question, search_q, hits, chat_id, effective_top_k):
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
            return filtered[:effective_top_k] if filtered else hits[:effective_top_k]
        if wanted_source:
            filtered = [(d, s) for d, s in hits if d.source == wanted_source]
            return filtered[:effective_top_k] if filtered else hits[:effective_top_k]
        return hits[:effective_top_k]

    # --------------------------------------------------------------------- #
    # Message persistence
    # --------------------------------------------------------------------- #
    def append_user_message(self, chat_id: str, question: str) -> list[dict]:
        """Append the user's question and (if first) auto-title the chat."""
        chat = self.get_chat(chat_id)
        messages = chat.get("messages", [])
        if not messages:
            chat["title"] = question[:40] + "..." if len(question) > 40 else question
        messages.append({"role": "user", "content": question})
        with self._lock:
            chat_store.update_chat_messages(self.chats, chat_id, messages)
        return messages

    def append_assistant_message(
        self, chat_id: str, answer: str, sources: list[dict], metrics: dict | None
    ) -> None:
        chat = self.get_chat(chat_id)
        messages = chat.get("messages", [])
        # "I don't know" answers are stored without sources, matching the UI.
        is_dont_know = "don't know" in answer.lower()
        messages.append({
            "role": "assistant",
            "content": answer,
            "sources": [] if is_dont_know else sources,
            "metrics": metrics,
        })
        with self._lock:
            chat_store.update_chat_messages(self.chats, chat_id, messages)

    # --------------------------------------------------------------------- #
    # Generation
    # --------------------------------------------------------------------- #
    def answer_stream(self, search_query: str, hits):
        """Yield answer text deltas from Groq (grounded in `hits`)."""
        return generator.answer(
            search_query,
            hits,
            model=config.DEFAULT_MODEL,
            temperature=config.DEFAULT_TEMPERATURE,
        )

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
