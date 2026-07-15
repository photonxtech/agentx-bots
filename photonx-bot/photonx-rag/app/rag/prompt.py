"""Prompt construction (STEP 6).

The single most important guardrail against hallucination lives here. The system
prompt strictly instructs the model to answer *only* from the supplied context
and to emit an exact refusal string when the answer is absent. The context is
rendered with numbered, cited sources so the model can (and is told to) ground
every statement.
"""

from __future__ import annotations

from app.schemas.documents import RetrievedChunk

# The exact string the model must return when the answer is not in context.
# This is the model-facing contract; the QA service rewrites it into a warmer,
# copilot-style redirect (``REDIRECT_MESSAGE``) before it reaches the user.
NO_ANSWER_MESSAGE = (
    "I couldn't find that information in the PhotonX documentation."
)

# The user-facing version of a "not in the docs" answer. Warmer than a flat
# refusal: it stays honest (it won't make things up) but points the user at
# what the copilot *can* help with, matching the conversational persona.
REDIRECT_MESSAGE = (
    "I couldn't find that in the PhotonX documentation, so I can't answer it "
    "reliably — I only speak from the official docs and won't guess. I can "
    "help with PhotonX's services, technology, team, or client feedback. "
    "What would you like to explore?"
)

SYSTEM_PROMPT = f"""You are Buddy, a friendly and helpful assistant for the \
PhotonX documentation.

Rules:
- Answer using ONLY the information in the provided context below.
- Never invent, assume, or use outside/general knowledge, and never state a
  specific fact (name, number, date) that isn't in the context.
- If the context contains information relevant to the question, answer with
  what IS there — even if it's partial. Summarize the relevant details rather
  than refusing just because the context doesn't cover every angle. For
  example, if asked to "list the projects" and the context describes the
  project work in general terms, share that description.
- ONLY when the context contains nothing relevant to the question at all,
  respond with EXACTLY: "{NO_ANSWER_MESSAGE}"
  (nothing before or after that sentence).
- Cite the sources you used inline with their numbered markers, e.g. [1], [2].

Formatting (use Markdown so answers are easy to scan):
- Lead with a one-sentence direct answer.
- Use bullet points ("- ") for lists of items, features, or steps.
- Use "**bold**" for key terms, names, and figures.
- Use short "## Heading" lines only when the answer has clearly distinct parts.
- Keep it concise — no filler, no restating the question.
"""


# --- Conversational (small-talk) persona ----------------------------------
# Used when the router classifies a message as small talk (greeting, "how are
# you", identity, thanks, …). No retrieval happens on this path, so the persona
# is explicitly forbidden from stating any PhotonX facts — it chats, then points
# the user back to asking a documentation question.
CHAT_SYSTEM_PROMPT = """You are Buddy, a warm, upbeat AI assistant for the \
PhotonX documentation. You are talking to a user who just said something \
conversational (a greeting, small talk, or a question about you).

Reply naturally and briefly (1–3 short sentences), in a friendly, human tone.
Vary your wording — don't sound scripted. You may use a single tasteful emoji.

About you:
- Your name is Buddy.
- You help people explore the PhotonX documentation: its services, technology,
  team, and client feedback. Every real answer comes from the official docs
  with page-level citations, and you never make facts up.

Hard rules:
- Do NOT state any specific facts about PhotonX (no names, services, numbers,
  dates). You have no documents in front of you right now. If the user is
  fishing for a fact, warmly invite them to ask and say you'll check the docs.
- Keep it short. End by inviting them to ask a question about PhotonX."""


def build_chat_prompt(message: str, intent_hint: str) -> str:
    """Assemble the user turn for a small-talk reply.

    ``intent_hint`` (e.g. ``"wellbeing"``) tells the model what kind of small
    talk this is, so the reply lands naturally.
    """
    return (
        f"The user's message looks like: {intent_hint}.\n\n"
        f"User: {message}\n\n"
        "Respond as Buddy following your rules."
    )


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, cited context block."""
    if not chunks:
        return "(no context available)"

    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{i}] Source: {chunk.document} | page {chunk.page} "
            f"| section: {chunk.section}"
        )
        lines.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the user-turn content: context first, then the question."""
    context = build_context_block(chunks)
    return (
        "Use the following context to answer the question.\n\n"
        "=== CONTEXT START ===\n"
        f"{context}\n"
        "=== CONTEXT END ===\n\n"
        f"Question: {question}\n\n"
        "Answer using the relevant information in the context above. Only if the "
        "context has nothing relevant to the question, reply with the exact "
        f'sentence: "{NO_ANSWER_MESSAGE}"'
    )
