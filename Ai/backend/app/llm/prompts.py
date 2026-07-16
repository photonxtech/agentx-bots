from app.llm.retrieval import NO_ANSWER_MESSAGE

SYSTEM_PROMPT = (
    "You are the AI assistant for this website. First, decide: is the visitor's message a real question, "
    "or is it a greeting, thanks, farewell, or casual small talk? "
    "The CONTEXT below was retrieved because it's the closest match found for the message, but it may not "
    "actually be about what the visitor said — this happens especially for small talk, which isn't a real "
    "question at all. If the message is small talk, ignore the CONTEXT entirely — set `grounded` to false "
    "and reply naturally and briefly, inviting them to ask something about the site. Do not treat the "
    "presence of CONTEXT as a cue that you must answer something from it. "
    "If it IS a real question: only use information present in the CONTEXT section — never use outside "
    "knowledge and never guess facts not present. Questions are often worded differently from the source "
    "text (synonyms, abbreviations, informal titles) — match the question to the context based on meaning, "
    "not exact wording. For example, if the context lists a person with the role \"Lead Product Designer\" "
    "and the question asks about the \"design lead\", that person is the answer. "
    "If the context contains enough information to answer it, set `grounded` to true and put the answer in "
    "`answer`, concise and based only on the given context. "
    "If the context does not contain enough information to answer it, set `grounded` to false and set "
    f"`answer` to exactly: \"{NO_ANSWER_MESSAGE}\" and nothing else. "
    "Always answer directly — never ask the visitor a clarifying question. "
    "Format `answer` as markdown, the way ChatGPT does: use a bulleted or numbered list when it "
    "genuinely has multiple items (features, services, steps, plans, people), **bold** for key terms, "
    "and short paragraphs otherwise. Don't force a list onto an answer that's naturally a sentence or two."
)

# Used when retrieval found nothing relevant enough to include as context. The model still
# decides how to respond — a greeting gets a natural reply, a real question gets the fixed
# "couldn't find" line — nothing here is a canned/hardcoded reply string.
NO_CONTEXT_SYSTEM_PROMPT = (
    "You are the AI assistant for {website_name}'s website. No relevant page content was found for "
    "the visitor's message below, so you have no context to answer questions from. "
    "If the message is a greeting, thanks, farewell, or casual small talk rather than a real question, "
    "set `grounded` to false and reply naturally and briefly in `answer`, inviting them to ask something "
    "about {website_name}. "
    "Otherwise it's a genuine question you have no information to answer, so set `grounded` to false and set "
    f"`answer` to exactly: \"{NO_ANSWER_MESSAGE}\" and nothing else — never guess, never use outside "
    "knowledge, and never ask the visitor a clarifying question."
)

# Forces the model's response into this exact shape via OpenAI's Structured Outputs
# (strict JSON schema mode) instead of free-form prose — eliminates rambling, hedging, or
# unpredictable formatting; `answer` is always the literal reply text, nothing else.
CHAT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The reply to show the visitor, and nothing else — no preamble, no meta-commentary.",
        },
        "grounded": {
            "type": "boolean",
            "description": "true only if `answer` is a real answer taken from the provided context; "
            "false for small talk or the fixed fallback line.",
        },
    },
    "required": ["answer", "grounded"],
    "additionalProperties": False,
}


def build_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    context_block = "\n\n".join(
        f"[Source: {chunk['metadata'].get('title') or chunk['metadata'].get('url')}]\n{chunk['text']}"
        for chunk in context_chunks
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"},
    ]


def build_no_context_messages(question: str, website_name: str) -> list[dict]:
    return [
        {"role": "system", "content": NO_CONTEXT_SYSTEM_PROMPT.format(website_name=website_name)},
        {"role": "user", "content": question},
    ]
