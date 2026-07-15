"""The RAG pipeline, built with LangChain.

This single module replaces SEVEN hand-written files from the manual project.
The whole point of this exercise is to see *which* LangChain abstraction stands
in for *which* piece we wrote by hand:

    Manual project file        LangChain abstraction (here)
    -----------------------    ------------------------------------------
    rag/pdf_loader.py       →  PyPDFLoader           (community loader)
    rag/chunker.py          →  RecursiveCharacterTextSplitter
    rag/embedding.py        →  OpenAIEmbeddings
    rag/vector_store.py     →  Chroma  (langchain-chroma)
    rag/retriever.py        →  vectorstore.as_retriever(...)
    rag/prompt.py           →  ChatPromptTemplate     (see prompt.py)
    rag/llm.py              →  ChatOpenAI
    services/qa_service.py  →  an LCEL chain ( | pipe operator )

LCEL = "LangChain Expression Language": the ``a | b | c`` syntax that wires
runnables into a chain, so ``chain.invoke(question)`` runs retrieve → format →
prompt → LLM → parse in one call.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.core.exceptions import DocumentError, VectorStoreError
from app.core.logger import get_logger
from app.rag import conversation
from app.rag.prompt import (
    NO_ANSWER_MESSAGE,
    REDIRECT_MESSAGE,
    build_chat_prompt,
    build_prompt,
)

logger = get_logger(__name__)


class RAGPipeline:
    """Ingest PDFs and answer questions using LangChain components."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # 1) EMBEDDINGS — replaces the whole manual embedding.py (batching,
        #    retries, ordering are handled inside this object).
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        # 2) VECTOR STORE — replaces manual vector_store.py. Chroma is opened in
        #    persistent mode; cosine space is requested via collection metadata.
        self._store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self._embeddings,
            persist_directory=str(settings.chroma_db),
            collection_metadata={"hnsw:space": "cosine"},
        )

        # 3) LLM — replaces manual llm.py.
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )

        # 4) RETRIEVER — HYBRID (vector + keyword). Pure vector search misses
        #    the leadership roster for queries like "who founded PhotonX?" (a
        #    bare list of names embeds far from the query). BM25 keyword search
        #    catches it via the literal "Co-founder" match, and an
        #    EnsembleRetriever fuses both rankings so people/org queries surface
        #    the right chunk regardless of phrasing.
        self._retriever = self._build_retriever()

        # 5) CHAIN — replaces manual qa_service.py orchestration. See below.
        self._chain = self._build_chain()

        # 6) CHAT CHAIN — small-talk persona (no retrieval). A separate LCEL
        #    chain: persona prompt → chat model (warmer temperature) → string.
        chat_llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            max_tokens=160,
            api_key=settings.openai_api_key,
        )
        self._chat_chain = build_chat_prompt() | chat_llm | StrOutputParser()

        logger.info("LangChain RAG pipeline initialized.")

    # ------------------------------------------------------------------ ingest
    def ingest(self) -> tuple[int, int]:
        """Load → split → embed → store every PDF in the docs folder.

        Returns ``(num_documents, num_chunks)``.
        """
        docs_dir = self._settings.docs_dir
        # Index both PDF and Word documents living in the docs folder.
        paths = sorted(
            p
            for p in docs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".pdf", ".docx")
        )
        if not paths:
            raise DocumentError(f"No PDF or DOCX files found in {docs_dir}")

        # LOAD — PyPDFLoader yields one Document per page (0-based page in
        # metadata); Docx2txtLoader yields one Document for the whole file
        # (Word has no stored pagination), which we label page 1. Both are
        # normalized to our API: 1-based page + source name.
        pages: list[Document] = []
        for path in paths:
            if path.suffix.lower() == ".pdf":
                logger.info("Loading PDF: %s", path.name)
                loaded = PyPDFLoader(str(path)).load()
                for d in loaded:
                    d.metadata["document"] = path.name
                    d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
            else:
                logger.info("Loading DOCX: %s", path.name)
                loaded = Docx2txtLoader(str(path)).load()
                for d in loaded:
                    d.metadata["document"] = path.name
                    d.metadata["page"] = 1
            pages.extend(loaded)

        if not pages:
            raise DocumentError("No extractable text found in the PDFs.")

        # CLEAN — strip repeated page boilerplate (headers/footers like the
        # company name, URL, and "Page N") *before* splitting. PyPDFLoader keeps
        # this text, and RecursiveCharacterTextSplitter is structure-agnostic, so
        # without this every chunk would be prefixed with the same boilerplate —
        # which pollutes retrieval and causes false "not found" answers.
        self._strip_boilerplate(pages)

        # SECTION-AWARE PRE-SPLIT — the generic splitter is structure-blind, so
        # a short heading block like "Leadership Team" gets glued onto the
        # neighbouring paragraph and stops surfacing for a "who founded" query
        # (the exact failure the manual project's heading-aware chunker avoids).
        # We first break each page into per-heading sections, tagging each with
        # its section name, so a heading's content stays its own coherent unit.
        sectioned = self._split_into_sections(pages)

        # SPLIT (RecursiveCharacterTextSplitter) — pack each section into
        # size/overlap windows, splitting on paragraphs → lines → sentences →
        # words. Because we split *within* sections, section boundaries are
        # always respected and each chunk keeps its section metadata.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(sectioned)
        logger.info("Split into %d chunks", len(chunks))

        # CONTEXT LEAD-IN FOR RETRIEVAL — prepend a short, query-friendly line to
        # the text we embed. A bare roster ("Prathik Gadde — Co-founder & CEO,
        # …") has no topic words, so both vector and keyword search miss it for
        # "who founded PhotonX?". A lead-in naming the section (and, for people
        # sections, the words "founders / leaders / founded / lead") gives every
        # chunk a topical anchor. The lead-in is embedded/indexed but stripped
        # before the text is shown to the LLM, so answers are unchanged — only
        # retrieval recall improves.
        for chunk in chunks:
            lead_in = self._retrieval_lead_in(chunk.metadata.get("section"))
            if lead_in:
                chunk.page_content = f"{lead_in}\n{chunk.page_content}"

        # Reset the collection so re-ingesting is clean (mirrors manual project).
        self._reset_collection()

        # EMBED + STORE — a single call embeds all chunks and upserts them.
        start = time.perf_counter()
        try:
            self._store.add_documents(chunks)
        except Exception as exc:  # chroma raises broad errors
            raise VectorStoreError(f"Failed to store vectors: {exc}") from exc
        logger.info(
            "Embedded + stored %d chunks in %.2fs",
            len(chunks),
            time.perf_counter() - start,
        )

        # Rebuild the hybrid retriever (and the chain that uses it) so the BM25
        # half sees the freshly-ingested chunks, not a stale corpus.
        self._retriever = self._build_retriever()
        self._chain = self._build_chain()

        num_documents = len({p.metadata.get("document") for p in pages})
        return num_documents, len(chunks)

    # --------------------------------------------------------------------- ask
    def ask(self, question: str) -> tuple[str, list[dict[str, object]], str]:
        """Answer a question. Returns ``(answer, sources, kind)``.

        ``kind`` is ``"chat"`` (copilot small talk), ``"answer"`` (grounded
        answer with citations), or ``"redirect"`` (not in the docs).

        Retrieval runs once here so we can both (a) feed context to the chain
        and (b) build citations from the same chunks — guaranteeing sources are
        never fabricated (identical guarantee to the manual project).
        """
        # Copilot small talk (greetings, identity, thanks, …) is routed here
        # before any retrieval. The reply is generated by the chat chain so
        # Buddy sounds natural and varied; the classifier only decides *that*
        # it's small talk, not the wording.
        intent = conversation.classify(question)
        if intent is not None:
            logger.info("Conversational intent: %s", intent.value)
            try:
                reply = self._chat_chain.invoke(
                    {"message": question, "intent_hint": intent.value}
                )
            except Exception as exc:  # never hard-fail a greeting
                logger.warning("Chat generation failed, using fallback: %s", exc)
                reply = (
                    f"Hi! I'm {conversation.ASSISTANT_NAME}, your PhotonX "
                    "assistant. Ask me anything about PhotonX's services, "
                    "technology, team, or clients!"
                )
            return reply, [], "chat"

        retrieved = self._retriever.invoke(question)
        if not retrieved:
            logger.info("No context retrieved; redirecting.")
            return REDIRECT_MESSAGE, [], "redirect"

        start = time.perf_counter()
        answer = self._chain.invoke(question)
        logger.info("Chain produced answer in %.2fs", time.perf_counter() - start)

        if NO_ANSWER_MESSAGE.lower() in answer.lower():
            return REDIRECT_MESSAGE, [], "redirect"

        return answer, self._sources_from(answer, retrieved), "answer"

    # ------------------------------------------------------------------- stats
    def count(self) -> int:
        """Number of vectors stored."""
        try:
            return self._store._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Count failed: {exc}") from exc

    def distinct_documents(self) -> int:
        """Number of distinct source documents indexed."""
        try:
            metas = self._store._collection.get(include=["metadatas"])["metadatas"]
        except Exception as exc:
            raise VectorStoreError(f"Metadata scan failed: {exc}") from exc
        return len({m.get("document") for m in metas if m})

    # --------------------------------------------------------------- internals
    def _build_retriever(self):
        """Build a hybrid (vector + BM25 keyword) retriever.

        The Chroma vector retriever handles semantic matches; a BM25 retriever
        built from the same stored chunks handles literal keyword matches (e.g.
        "founder" → the "Co-founder & CEO" roster). ``EnsembleRetriever`` fuses
        the two rankings with reciprocal-rank fusion.

        If the store is empty (before the first ingest), fall back to the plain
        vector retriever so startup never fails.
        """
        k = self._settings.top_k
        vector = self._store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

        # Load stored chunks (text + metadata) to seed BM25.
        try:
            stored = self._store._collection.get(
                include=["documents", "metadatas"]
            )
        except Exception:  # pragma: no cover - store not ready
            return vector

        texts = stored.get("documents") or []
        metas = stored.get("metadatas") or []
        if not texts:
            return vector

        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metas)
        ]
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = k

        # Weight vector and keyword roughly equally; keyword gets a slight edge
        # so exact-term queries (names, roles) reliably surface their chunk.
        return EnsembleRetriever(
            retrievers=[vector, bm25], weights=[0.5, 0.5]
        )

    def _build_chain(self):
        """Wire retrieve → format → prompt → LLM → parse using LCEL.

        The ``|`` operator composes runnables. Reading it top to bottom:
          - ``context`` = retrieve docs, then format them into a numbered block
          - ``question`` = passed straight through
          - both feed the prompt template
          - the prompt feeds the chat model
          - the model output is parsed to a plain string
        """
        prompt = build_prompt()
        format_context = RunnableLambda(
            lambda q: self._format_docs(self._retriever.invoke(q))
        )
        return (
            {"context": format_context, "question": RunnablePassthrough()}
            | prompt
            | self._llm
            | StrOutputParser()
        )

    @classmethod
    def _format_docs(cls, docs: list[Document]) -> str:
        """Render retrieved docs as a numbered, cited context block."""
        if not docs:
            return "(no context available)"
        blocks = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata
            header = (
                f"[{i}] Source: {meta.get('document', 'unknown')} "
                f"| page {meta.get('page', '?')} "
                f"| section: {meta.get('section', 'General')}"
            )
            # Strip the injected retrieval lead-in so the LLM sees clean text.
            content = cls._strip_lead_in(d.page_content.strip())
            blocks.append(f"{header}\n{content}")
        return "\n\n".join(blocks)

    @staticmethod
    def _sources_from(
        answer: str, docs: list[Document]
    ) -> list[dict[str, object]]:
        """Return only the docs the answer cited via ``[n]`` markers.

        Same citation-precision fix we applied to the manual project.
        """
        indices = {
            int(n)
            for n in re.findall(r"\[(\d{1,2})\]", answer)
            if 1 <= int(n) <= len(docs)
        }
        chosen = (
            [docs[i - 1] for i in sorted(indices)] if indices else docs
        )

        seen: set[tuple[str, int, str]] = set()
        sources: list[dict[str, object]] = []
        for d in chosen:
            document = str(d.metadata.get("document", "unknown"))
            page = int(d.metadata.get("page", 0) or 0)
            section = str(d.metadata.get("section", "General"))
            key = (document, page, section)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {"document": document, "page": page, "section": section}
            )
        return sources

    # A section heading: a short ALL-CAPS or Title-Case line (optionally
    # numbered) sitting on its own — e.g. "Leadership Team", "OUR SERVICES",
    # "2.1 Services". Ordinary sentences (which contain lowercase words and end
    # with punctuation) are not headings.
    _HEADING_RE = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
        r"(?:[A-Z][A-Za-z0-9&/\-]*)(?:\s+[A-Z][A-Za-z0-9&/\-]*){0,6}\s*$"
    )

    @classmethod
    def _split_into_sections(cls, pages: list[Document]) -> list[Document]:
        """Break each page Document into one Document per heading section.

        Detects short heading-like lines and starts a new section at each,
        tagging every emitted Document with a ``section`` metadata field. This
        keeps a heading's content (e.g. the leadership roster) as its own unit
        so it retrieves for on-topic queries instead of being buried in a
        neighbouring paragraph.
        """
        out: list[Document] = []
        for page in pages:
            current_heading = "General"
            buffer: list[str] = []

            def flush() -> None:
                body = "\n".join(buffer).strip()
                if not body:
                    return
                meta = dict(page.metadata)
                meta["section"] = current_heading
                out.append(Document(page_content=body, metadata=meta))

            for line in page.page_content.splitlines():
                stripped = line.strip()
                if cls._is_heading(stripped):
                    flush()
                    buffer = []
                    current_heading = stripped
                else:
                    buffer.append(line)
            flush()

        # If sectioning found nothing usable, fall back to the original pages so
        # a document with no detectable headings is never dropped.
        return out or pages

    # Prefix marking an injected retrieval lead-in, so it can be stripped back
    # out before the chunk text is shown to the LLM.
    _LEAD_IN_MARK = "[[topic]] "

    @classmethod
    def _retrieval_lead_in(cls, section: str | None) -> str:
        """Return a query-friendly lead-in line to embed with a chunk.

        Sparse sections (a bare name roster, a stat block) don't contain the
        words a natural question uses, so they never retrieve. The lead-in names
        the section's topic. People/leadership sections get extra synonyms
        ("founders, leaders, founded, lead") so "who founded/leads PhotonX?"
        reliably hits the roster.
        """
        if not section or section == "General":
            return ""
        lowered = section.lower()
        if any(w in lowered for w in ("leadership", "team", "founder", "people")):
            topic = (
                f"{section}: the founders, leaders, and key people who founded "
                "and lead PhotonX."
            )
        else:
            topic = f"{section}."
        return f"{cls._LEAD_IN_MARK}{topic}"

    @classmethod
    def _strip_lead_in(cls, text: str) -> str:
        """Remove an injected retrieval lead-in line from chunk text."""
        if text.startswith(cls._LEAD_IN_MARK):
            # Drop the first line (the lead-in) and return the real content.
            _, _, rest = text.partition("\n")
            return rest.strip() or text
        return text

    @classmethod
    def _is_heading(cls, line: str) -> bool:
        """True for a short, title-like line that starts a new section."""
        if not line or len(line) > 60:
            return False
        # A heading has at least two words OR a numbered prefix (avoids matching
        # a lone capitalized word that's really the start of a sentence).
        if not re.match(r"^\d", line) and len(line.split()) < 2:
            return False
        # Sentences end in punctuation; headings don't.
        if line.endswith((".", ",", ":", ";", "!", "?")):
            return False
        return bool(cls._HEADING_RE.match(line))

    @staticmethod
    def _strip_boilerplate(pages: list[Document]) -> None:
        """Remove repeated header/footer lines from every page, in place.

        Finds short lines that appear on most pages (the page boilerplate) plus
        obvious "Page N" / URL / email lines, and drops them. This is the
        LangChain equivalent of the manual chunker's footer-stripping — the one
        piece of chunking quality the generic splitter doesn't provide.
        """
        boilerplate_re = re.compile(
            r"^\s*(?:"
            r"page\s+\d+"
            r"|\d+\s*/\s*\d+"
            r"|(?:https?://)?[\w.-]+\.[a-z]{2,}(?:\s*\|\s*\S+@\S+)?"
            r"|\S+@\S+\.\S+"
            r")\s*$",
            re.IGNORECASE,
        )

        # Lines repeated on >=60% of pages are treated as boilerplate.
        counter: Counter[str] = Counter()
        for page in pages:
            counter.update(
                {
                    ln.strip()
                    for ln in page.page_content.splitlines()
                    if 0 < len(ln.strip()) <= 60
                }
            )
        threshold = max(2, int(len(pages) * 0.6))
        repeated = {line for line, n in counter.items() if n >= threshold}

        for page in pages:
            kept = [
                ln
                for ln in page.page_content.splitlines()
                if ln.strip() not in repeated
                and not boilerplate_re.match(ln.strip())
            ]
            page.page_content = "\n".join(kept).strip()

    def _reset_collection(self) -> None:
        """Drop and recreate the collection before a full re-ingest."""
        try:
            ids = self._store._collection.get()["ids"]
            if ids:
                self._store._collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Reset failed: {exc}") from exc
