"""Generation: build a grounded prompt from retrieved context and call Groq.

This is the "G" in RAG. The key idea: we do NOT ask the model from memory. We
hand it the retrieved chunks and instruct it to answer *only* from them,
which is what keeps answers grounded and citable.
"""

from __future__ import annotations

import os

from groq import Groq

import config
from rag.ingestion import Document

SYSTEM_PROMPT = (
    "You are a helpful assistant for question-answering over the user's own "
    "documents (spreadsheets, PDFs, presentations, and scanned/photographed "
    "images). Answer using the provided context. "
    "For broad questions like 'what does this document contain', 'give an overview', "
    "or 'summarize', synthesize a helpful answer from ALL the retrieved context chunks "
    "even if no single chunk perfectly answers the question. "
    "Only respond with: \"I don't know. Please ask a question related to the uploaded documents.\" "
    "when the context is completely empty or entirely unrelated to the question — "
    "never refuse when relevant context chunks are present. "
    "Do not invent facts beyond what is in the context. "
    "Be tolerant of slight misspellings or name variations in the question (e.g., 'doremon' "
    "referring to 'Doraemon', 'ppt' referring to a .pptx file) by mapping them to the "
    "closest match in the context. "
    "When the answer is found in the context, cite the source filename in parentheses."
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


def rewrite_query(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone search query.

    "and when is it due?" is unsearchable on its own — the embedding and BM25
    stages need the missing subject ("the electricity bill") resolved from the
    chat history. First questions and failures fall back to the original text.
    """
    turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
    if not turns:
        return question

    convo = "\n".join(
        f"{m['role']}: {m['content'][:300]}"
        for m in turns[-config.REWRITE_HISTORY_TURNS:]
    )
    try:
        resp = _client().chat.completions.create(
            model=config.REWRITE_MODEL,
            temperature=0.0,
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite follow-up questions into standalone search "
                        "queries. Replace every pronoun or vague reference "
                        "(it, that, the document, and...) with the concrete "
                        "subject from the conversation. Keep it short.\n"
                        "Example: after a discussion of an electricity bill, "
                        "'and when is it due?' becomes "
                        "'When is the electricity bill due?'\n"
                        "Return ONLY the rewritten question — no explanation, "
                        "no quotes."
                    ),
                },
                {"role": "user", "content": f"Conversation:\n{convo}\n\nLatest question: {question}"},
            ],
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        # Sanity guard: a rambling or empty rewrite is worse than the original.
        if rewritten and len(rewritten) <= max(200, 3 * len(question)):
            return rewritten
    except Exception:
        pass
    return question


def build_context(hits: list[tuple[Document, float]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, (doc, score) in enumerate(hits, 1):
        blocks.append(
            f"[{i}] (source: {doc.source}, type: {doc.kind}, score: {score:.2f})\n{doc.text}"
        )
    return "\n\n".join(blocks)


def answer(
    question: str,
    hits: list[tuple[Document, float]],
    model: str = config.DEFAULT_MODEL,
    temperature: float = config.DEFAULT_TEMPERATURE,
):
    """Stream an answer from Groq, grounded in the retrieved context.

    Yields text deltas so the Streamlit UI can render the answer live.
    """
    context = build_context(hits) or "(no context retrieved)"
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    stream = _client().chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=config.MAX_TOKENS,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
