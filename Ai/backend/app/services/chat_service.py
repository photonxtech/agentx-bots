import json
import time

from sqlalchemy.orm import Session

from app.database.models import Conversation, Message, MessageRole, Website
from app.llm.llm_client import generate_structured
from app.llm.prompts import CHAT_RESPONSE_SCHEMA, build_messages, build_no_context_messages
from app.llm.query_normalizer import correct_query_spelling
from app.llm.retrieval import compute_confidence, retrieve_relevant_chunks
from app.services import cache_service
from app.utils.logging import get_logger
from app.utils.metrics import chat_metrics

logger = get_logger(__name__)

_GENERATION_ERROR_TEXT = "Something went wrong generating a response. Please try again."


def get_or_create_conversation(
    db: Session, website_id: int, session_id: str, conversation_id: int | None
) -> Conversation:
    if conversation_id:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            return conversation

    conversation = Conversation(website_id=website_id, session_id=session_id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _dedupe_sources(chunks: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for chunk in chunks:
        meta = chunk["metadata"]
        if meta["url"] in seen:
            continue
        seen.add(meta["url"])
        deduped.append({"page_id": meta["page_id"], "url": meta["url"], "title": meta.get("title", "")})
    return deduped


def _chunk_for_replay(text: str, group_size: int = 4) -> list[str]:
    # Structured Outputs returns the complete answer in one call rather than a token
    # stream (see llm_client.generate_structured) — this splits it back into several SSE
    # "delta" events so the frontend still sees the familiar typing effect either way.
    words = text.split(" ")
    if not words:
        return []
    pieces = []
    for i in range(0, len(words), group_size):
        piece = " ".join(words[i : i + group_size])
        if i + group_size < len(words):
            piece += " "
        pieces.append(piece)
    return pieces


async def _generate_and_persist(
    db: Session,
    conversation: Conversation,
    messages: list[dict],
    sources: list[dict],
    confidence: float | None,
    website_id: int,
    question: str,
    cache_response: bool,
    request_started: float,
    retrieval_ms: float,
):
    usage: dict = {}
    generation_started = time.perf_counter()
    grounded = False
    try:
        result = await generate_structured(messages, CHAT_RESPONSE_SCHEMA, usage_holder=usage)
        full_text = (result.get("answer") or "").strip()
        grounded = bool(result.get("grounded"))
        if not full_text:
            full_text = _GENERATION_ERROR_TEXT
    except Exception:
        logger.exception("LLM generation failed for conversation %s", conversation.id)
        full_text = _GENERATION_ERROR_TEXT
    generation_ms = round((time.perf_counter() - generation_started) * 1000, 1)

    for piece in _chunk_for_replay(full_text):
        yield _sse({"type": "delta", "content": piece})

    total_ms = round((time.perf_counter() - request_started) * 1000, 1)

    # Retrieval may have surfaced context that merely looked closest, without the model
    # actually using it (small talk, or a genuine question the context didn't answer) —
    # showing sources/confidence in that case would misleadingly imply the reply is
    # grounded in site content when the model itself said it isn't.
    if not grounded:
        sources = []
        confidence = None

    assistant_message = Message(
        conversation_id=conversation.id, role=MessageRole.assistant,
        content=full_text, sources=sources, confidence=confidence,
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    is_error_response = full_text == _GENERATION_ERROR_TEXT

    logger.info(
        "chat_response conversation_id=%s total_ms=%s retrieval_ms=%s generation_ms=%s "
        "prompt_tokens=%s completion_tokens=%s confidence=%s sources=%s error=%s",
        conversation.id, total_ms, retrieval_ms, generation_ms,
        usage.get("prompt_tokens"), usage.get("completion_tokens"), confidence, len(sources), is_error_response,
    )
    chat_metrics.record_request(
        total_ms=total_ms, retrieval_ms=retrieval_ms, generation_ms=generation_ms,
        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
        error=is_error_response,
    )

    if cache_response and not is_error_response:
        cache_service.set_cached_response(
            website_id, question, {"answer": full_text, "sources": sources, "confidence": confidence},
        )

    yield _sse({
        "type": "done", "sources": sources, "confidence": confidence,
        "message_id": assistant_message.id, "conversation_id": conversation.id,
    })


async def stream_chat_response(
    db: Session, website_id: int, conversation: Conversation, question: str, regenerate: bool = False,
):
    request_started = time.perf_counter()
    db.add(Message(conversation_id=conversation.id, role=MessageRole.user, content=question))
    db.commit()

    # "Regenerate" means the visitor explicitly wants a new attempt, so it must bypass the
    # response cache (a cache hit would just replay the exact same answer) — but retrieval
    # itself is deterministic, so re-using its cache is still fine there.
    if not regenerate:
        cached = cache_service.get_cached_response(website_id, question)
        if cached is not None:
            assistant_message = Message(
                conversation_id=conversation.id, role=MessageRole.assistant,
                content=cached["answer"], sources=cached["sources"], confidence=cached["confidence"],
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)
            total_ms = round((time.perf_counter() - request_started) * 1000, 1)
            logger.info("chat_cache_hit conversation_id=%s total_ms=%s", conversation.id, total_ms)
            chat_metrics.record_request(total_ms=total_ms, cache_hit=True)
            for piece in _chunk_for_replay(cached["answer"]):
                yield _sse({"type": "delta", "content": piece})
            yield _sse({
                "type": "done", "sources": cached["sources"], "confidence": cached["confidence"],
                "message_id": assistant_message.id, "conversation_id": conversation.id,
            })
            return

    website = db.query(Website).filter(Website.id == website_id).first()

    # Retrieval already spell-corrects internally to find the right chunks, but a heavily
    # garbled question (multiple typos, a word accidentally split in two) is also just
    # genuinely harder for the LLM itself to parse — even with perfectly relevant context
    # handed to it. Use the same corrected text for generation, not just retrieval; the
    # visitor's original wording is still what's stored in conversation history.
    corrected_question = correct_query_spelling(website_id, question)

    retrieval_started = time.perf_counter()
    chunks = retrieve_relevant_chunks(
        website_id, question, use_cache=not regenerate,
        website_name=website.name if website else None,
    )
    retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 1)

    if not chunks:
        # No relevant content found. Rather than returning a fixed canned string ourselves,
        # ask the model to decide how to respond — naturally if this is small talk, or with
        # the required fallback line if it's a real question it has no grounds to answer.
        messages = build_no_context_messages(corrected_question, website.name if website else "this website")
        async for event in _generate_and_persist(
            db, conversation, messages, [], None, website_id, question,
            cache_response=not regenerate, request_started=request_started, retrieval_ms=retrieval_ms,
        ):
            yield event
        return

    sources = _dedupe_sources(chunks)
    confidence = compute_confidence(chunks)
    messages = build_messages(corrected_question, chunks)

    async for event in _generate_and_persist(
        db, conversation, messages, sources, confidence, website_id, question,
        cache_response=not regenerate, request_started=request_started, retrieval_ms=retrieval_ms,
    ):
        yield event
