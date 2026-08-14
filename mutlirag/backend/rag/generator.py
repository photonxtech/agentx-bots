"""Generation: build a grounded prompt from retrieved context and call Groq.

This is the "G" in RAG. The key idea: we do NOT ask the model from memory. We
hand it the retrieved chunks and instruct it to answer *only* from them,
which is what keeps answers grounded and citable.
"""

from __future__ import annotations

import json
import logging
import os
import re

import numpy as np
from groq import Groq

import config
from rag.embeddings import embed
from rag.ingestion import Document

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant for question-answering over the user's own "
    "documents (spreadsheets, PDFs, presentations, and scanned/photographed "
    "images). Answer using the provided context. "
    "For a normal, specific question, answer only what was asked — do not pad "
    "the response with tangential material from other retrieved chunks just "
    "because it was in the context. "
    "For explicitly broad questions like 'what does this document contain', "
    "'give an overview', or 'summarize', synthesize across ALL the retrieved "
    "context chunks even if no single chunk perfectly answers the question. "
    "Questions like 'what is on page X', 'what does page X say', or "
    "'summarize page X' have ALREADY been resolved to the exact page by the "
    "retrieval system before you see the context — treat them the same as a "
    "broad/overview question and describe everything relevant in the "
    "provided context. Do not refuse just because the context text itself "
    "doesn't literally repeat the page number back to you — the retrieval "
    "system already guarantees the context IS that page's content. "
    "Only respond with: \"I don't know. Please ask a question related to the uploaded documents.\" "
    "when the context is completely empty or entirely unrelated to the question — "
    "never refuse when relevant context chunks are present. "
    "Do not invent facts beyond what is in the context. "
    "Be tolerant of slight misspellings or name variations in the question (e.g., 'doremon' "
    "referring to 'Doraemon', 'ppt' referring to a .pptx file) by mapping them to the "
    "closest match in the context. "
    "Do not include source filenames, page numbers, or scores in your generated response. Provide a direct, clean answer."
)


def _client() -> Groq:
    # The key lives in .env (loaded via python-dotenv) — never hardcode it.
    key = os.getenv("GROQ_API_KEY")
    if not key or key.startswith("gsk_your"):
        raise RuntimeError("No Groq API key set. Add GROQ_API_KEY to your .env file.")
    return Groq(api_key=key)


def list_models() -> list[str]:
    """Fetch the current chat models from Groq, newest list at runtime.

    Falls back to the curated list in config if the call fails (offline, bad
    key, etc.). We filter out whisper/tts/guard models that aren't for chat.
    """
    try:
        models = _client().models.list().data
        ids = [
            m.id
            for m in models
            if not any(x in m.id.lower() for x in ("whisper", "tts", "guard", "embed"))
        ]
        return sorted(ids) or config.FALLBACK_GROQ_MODELS
    except Exception:
        return config.FALLBACK_GROQ_MODELS


_VALID_MODES = ("standalone", "contextual", "clarify")


def _query_understanding(question: str, history: list[dict]) -> dict:
    """One LLM call that reads the latest question against chat history and
    decides which of three modes it's in — a single structured-JSON call
    instead of separate classify/rewrite calls, since the model already has
    to read the history to answer either question:

      standalone  — fully understandable on its own, no reference to earlier
                     conversation. A NEW document search, using the question
                     as-is.
      contextual  — relies on chat history to resolve a reference (pronoun,
                     "earlier", "the same") before it's searchable. Still a
                     NEW document search, once resolved.
      clarify     — not a new question about the document at all: the user
                     is asking to rephrase/simplify/expand/elaborate on the
                     answer just given ("more clearly", "simplify that",
                     "can you elaborate", "explain like I'm five", "give an
                     example"). No new search — see
                     RagService._retrieve_clarification, which reuses the
                     previous turn's own retrieved context instead. Confirmed
                     live: routing "more clearly" through a fresh document
                     search retrieved unrelated text that merely happened to
                     also discuss "clarity", producing a wrong answer.

    Returns {"mode": ..., "query": ...}. Raises on any call/parse failure so
    the caller can fall back to standalone.
    """
    turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]

    # Include full list of user questions so the model can resolve references to earlier topics
    user_questions = [f"- Turn {i+1}: {m['content'][:150]}" for i, m in enumerate(turns) if m.get("role") == "user"]
    user_q_summary = "\n".join(user_questions[-10:])

    # Recent turns context
    recent_convo = "\n".join(
        f"{m['role']}: {m['content'][:300]}"
        for m in turns[-config.REWRITE_HISTORY_TURNS:]
    )

    prompt_content = (
        f"Past user questions in this chat:\n{user_q_summary}\n\n"
        f"Recent conversation turns:\n{recent_convo}\n\n"
        f"Latest user question: {question}"
    )

    resp = _client().chat.completions.create(
        model=config.REWRITE_MODEL,
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query-understanding stage for a RAG document search engine.\n"
                    "Classify the user's LATEST message into exactly one mode:\n\n"
                    "STANDALONE — fully understandable on its own, no reference to anything "
                    "said earlier in this chat.\n"
                    "CONTEXTUAL — relies on chat history to make sense (a pronoun, 'earlier', "
                    "'the same', an unresolved reference) and needs to be rewritten into a "
                    "standalone search query before it can be searched.\n"
                    "CLARIFY — the user is NOT asking a new question about the document. They "
                    "are asking you to rephrase, simplify, expand, elaborate on, or otherwise "
                    "redo the ANSWER YOU JUST GAVE in the previous turn (e.g. 'more clearly', "
                    "'simplify that', 'can you elaborate', 'explain like I'm 5', 'give an "
                    "example', 'too complicated', 'in more detail', 'shorter please'). This "
                    "applies whenever the message reads as feedback/instruction about HOW the "
                    "previous answer was delivered, not a new fact being asked about the "
                    "document.\n\n"
                    "CRITICAL RULES:\n"
                    "1. NO TOPIC POISONING: Do NOT force a previous topic (like 'LEAP framework') into a question about a different topic (like '3 zones', 'smart zone', 'color zones'). If the latest question introduces its own clear subject, it is STANDALONE even if earlier turns discussed something else entirely.\n"
                    "2. EARLIER REFERENCES: If the user refers to something discussed earlier (e.g. 'earlier there 3 zones right'), check the past user questions and recent conversation to identify what '3 zones' refers to, and produce a query specific to that (mode=CONTEXTUAL).\n"
                    "3. RESOLVE PRONOUNS: Replace ambiguous pronouns (it, that, they, these) with the exact subject being discussed, only when the pronoun's subject genuinely isn't in the latest question itself.\n"
                    "4. WHEN IN DOUBT BETWEEN STANDALONE AND CONTEXTUAL, PICK STANDALONE. When in doubt about CLARIFY, ask: does this message contain its own new subject/topic? If yes, it's STANDALONE or CONTEXTUAL, not CLARIFY.\n\n"
                    "Respond with ONLY a JSON object: "
                    '{"mode": "standalone"|"contextual"|"clarify", "query": "..."}. '
                    "For standalone or clarify, \"query\" must be the latest question, verbatim, unchanged. "
                    "For contextual, \"query\" is the resolved standalone search query — concise, no quotes/labels."
                ),
            },
            {"role": "user", "content": prompt_content},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    parsed = json.loads(content)
    mode = str(parsed.get("mode") or "").strip().lower()
    if mode not in _VALID_MODES:
        mode = "standalone"
    query = str(parsed.get("query") or "").strip().strip('"\'')
    return {"mode": mode, "query": query or question}


def _rewrite_is_valid(question: str, rewritten: str) -> bool:
    """Independent, non-LLM safety net on a "contextual" rewrite — the
    classifier can still be wrong (or a small model can ignore an
    instruction), so its output isn't trusted blindly.

    A genuine follow-up rewrite stays close in meaning to the original
    question ("and when is it due?" -> "when is the electricity bill due" —
    same subject, just resolved). A topic-poisoned rewrite (confirmed live:
    a standalone question about "analytical vs critical thinking" rewritten
    into an unrelated earlier topic, "Cool Red vs Warm Red mindsets") does
    not. Cosine similarity between the two, via the same embedding model
    already used for retrieval, catches that drift regardless of wording —
    no need to enumerate every way a rewrite can go wrong.
    """
    if not rewritten or rewritten.strip() == question.strip():
        return True
    if len(rewritten) > max(200, 3 * len(question)):
        return False
    vecs = embed([question, rewritten])
    similarity = float(np.dot(vecs[0], vecs[1]))
    return similarity >= config.REWRITE_MIN_SIMILARITY


def classify_followup(question: str, history: list[dict]) -> dict:
    """Understand how `question` relates to the conversation so far.

    Returns {"mode": "standalone"|"contextual"|"clarify", "query": str}.
    `query` is the validated, resolved standalone search query for
    "contextual" (falls back to "standalone" with the original question if
    the rewrite looks topic-poisoned — see _rewrite_is_valid), and the
    original question, unchanged, for "standalone"/"clarify" (for "clarify"
    it's not used as a search query at all — see
    RagService._retrieve_clarification).
    """
    turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
    if not turns:
        return {"mode": "standalone", "query": question}

    try:
        result = _query_understanding(question, history)
    except Exception:
        logger.warning("Query-understanding call failed; treating as standalone", exc_info=True)
        return {"mode": "standalone", "query": question}

    if result["mode"] == "clarify":
        return {"mode": "clarify", "query": question}

    if result["mode"] != "contextual" or not result["query"]:
        return {"mode": "standalone", "query": question}

    if not _rewrite_is_valid(question, result["query"]):
        logger.info(
            "Rewrite failed validation (likely topic drift); falling back to the original question. "
            "original=%r rewritten=%r",
            question, result["query"],
        )
        return {"mode": "standalone", "query": question}

    return {"mode": "contextual", "query": result["query"]}


def rewrite_query(question: str, history: list[dict]) -> str:
    """Backward-compatible convenience wrapper around classify_followup() for
    callers that only want the resolved search string, not the mode
    (RagService.retrieve() calls classify_followup() directly so it can also
    branch on "clarify")."""
    result = classify_followup(question, history)
    return question if result["mode"] == "clarify" else result["query"]


_HEADER_RE = re.compile(r"^\[(PDF|File|Image|DOCX|PPTX):[^\]]+\]\n?", re.MULTILINE)


def context_texts(hits: list[tuple[Document, float]]) -> list[str]:
    """Resolve each hit to its PARENT chunk text, deduplicated.

    rag.chunking indexes small CHILD chunks for precise retrieval, but
    generation and DeepEval evaluation should see the larger PARENT chunk each
    child belongs to (meta["parent_text"]) — not the narrow snippet that
    happened to match. Several retrieved children can share the same parent
    (adjacent chunks of one section), so parents are deduplicated here, first
    (highest-ranked) occurrence wins, or every hit would otherwise repeat the
    same text and waste context budget. Chunks indexed before parent-child
    chunking existed have no parent_text and fall back to their own text.
    """
    seen: set[str] = set()
    texts: list[str] = []
    for doc, _ in hits:
        text = doc.meta.get("parent_text") or doc.text
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return texts


def build_context(hits: list[tuple[Document, float]]) -> str:
    """Format retrieved chunks' parent text into a numbered context block for
    the prompt, omitting filenames, page numbers, and scores."""
    blocks = []
    for i, text in enumerate(context_texts(hits), 1):
        clean_text = _HEADER_RE.sub("", text).strip()
        blocks.append(f"[{i}]\n{clean_text}")
    return "\n\n".join(blocks)


def answer(
    question: str,
    hits: list[tuple[Document, float]],
    model: str = config.DEFAULT_MODEL,
    temperature: float = config.DEFAULT_TEMPERATURE,
    original_question: str | None = None,
    verified_context: bool = False,
    usage: dict | None = None,
):
    """Stream an answer from Groq, grounded in the retrieved context.

    `verified_context` — set for PAGE_QUERY/TOC_QUERY (see
    rag.query_intent), where `hits` were pulled deterministically by page
    metadata rather than by semantic search. Without this, a model asked
    "what's on page 247" but shown context with no literal "page 247" text
    in it tends to hedge ("there is no content from page 247 in the
    context") even though the retrieval system already guarantees the
    context IS that page — confirmed live: the SYSTEM_PROMPT rule alone
    wasn't enough to stop this, since nothing next to the context itself
    asserted it. Asserting it directly above the context (not just in the
    system prompt) is what actually changes the model's behavior.

    `usage` — an optional dict this function populates in place with
    prompt_tokens/completion_tokens/total_tokens once the stream's final
    chunk arrives, so a caller consuming this generator token-by-token
    (for the live UI) can still recover real token counts afterward for
    cost/latency tracking, without giving up streaming to get them.

    Yields text deltas so the UI can render the answer live.
    """
    context = build_context(hits) or "(no context retrieved)"
    if verified_context:
        context = (
            "(The following is the verified, exact content of the requested "
            "page(s)/section — already resolved by the retrieval system, "
            "not a search result to be second-guessed.)\n\n" + context
        )

    q_text = f"Question: {original_question}\n(Search topic: {question})" if original_question and original_question != question else f"Question: {question}"

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"{q_text}\n\n"
        "Answer using only the context above."
    )

    stream = _client().chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=config.MAX_TOKENS,
        stream=True,
        # Not a typed kwarg on this SDK version (groq==1.6.0) — confirmed
        # live it TypeErrors as an unexpected keyword if passed directly,
        # breaking every answer. The underlying Groq API does honor it
        # (confirmed live too, real usage numbers came back), so it's
        # forwarded via extra_body instead, same as any other
        # OpenAI-wire-compatible param the SDK hasn't added a typed
        # parameter for yet.
        extra_body={"stream_options": {"include_usage": True}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        # The final chunk (stream_options.include_usage) has empty choices
        # and carries token counts instead of a delta.
        if usage is not None and getattr(chunk, "usage", None):
            usage["prompt_tokens"] = chunk.usage.prompt_tokens
            usage["completion_tokens"] = chunk.usage.completion_tokens
            usage["total_tokens"] = chunk.usage.total_tokens
