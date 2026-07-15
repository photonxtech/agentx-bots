"""Prompt construction (same guardrail as the manual project).

The anti-hallucination contract is identical: answer only from context, emit
the exact refusal string otherwise, cite sources. What changes is *how* the
prompt is expressed — here as a LangChain ``ChatPromptTemplate`` rather than
hand-built strings.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

NO_ANSWER_MESSAGE = (
    "I couldn't find that information in the PhotonX documentation."
)

# User-facing "not in the docs" reply — warmer than a flat refusal. Stays honest
# (won't make things up) but points the user at what the copilot can help with.
REDIRECT_MESSAGE = (
    "I couldn't find that in the PhotonX documentation, so I can't answer it "
    "reliably — I only speak from the official docs and won't guess. I can "
    "help with PhotonX's services, technology, team, or client feedback. "
    "What would you like to explore?"
)

_SYSTEM = f"""You are Buddy, a friendly and helpful assistant for the PhotonX \
documentation.

Rules:
- Answer using ONLY the information in the provided context below.
- Never invent, assume, or use outside/general knowledge, and never state a
  specific fact (name, number, date) that isn't in the context.
- If the context contains information relevant to the question, answer with
  what IS there — even if it's partial. Summarize the relevant details rather
  than refusing just because the context doesn't cover every angle.
- ONLY when the context contains nothing relevant to the question at all,
  respond with EXACTLY: "{NO_ANSWER_MESSAGE}" (nothing before or after it).
- Cite the sources you used inline with their numbered markers, e.g. [1], [2].

Formatting (use Markdown so answers are easy to scan):
- Lead with a one-sentence direct answer.
- Use bullet points ("- ") for lists of items, features, or steps.
- Use "**bold**" for key terms, names, and figures.
- Use short "## Heading" lines only when the answer has clearly distinct parts.
- Keep it concise — no filler, no restating the question."""

_HUMAN = """Use the following context to answer the question.

=== CONTEXT START ===
{context}
=== CONTEXT END ===

Question: {question}

Answer using the relevant information in the context above. Only if the context \
has nothing relevant to the question, reply with the exact sentence: \
"{no_answer}\""""


def build_prompt() -> ChatPromptTemplate:
    """Return the chat prompt template used by the RAG chain."""
    return ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _HUMAN)]
    ).partial(no_answer=NO_ANSWER_MESSAGE)


# --- Conversational (small-talk) persona ----------------------------------
# Used when the router classifies a message as small talk. No retrieval happens
# on this path, so the persona is forbidden from stating any PhotonX facts.
_CHAT_SYSTEM = """You are Buddy, a warm, upbeat AI assistant for the PhotonX \
documentation. You are talking to a user who just said something conversational \
(a greeting, small talk, or a question about you).

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

_CHAT_HUMAN = """The user's message looks like: {intent_hint}.

User: {message}

Respond as Buddy following your rules."""


def build_chat_prompt() -> ChatPromptTemplate:
    """Return the prompt template for generating small-talk replies."""
    return ChatPromptTemplate.from_messages(
        [("system", _CHAT_SYSTEM), ("human", _CHAT_HUMAN)]
    )
