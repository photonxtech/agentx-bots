"""
QA test case generation using Groq, Gemini, or OpenAI.
Reads the uploaded document and generates Q&A pairs via LLM.
"""
import os
import json
import re
import math
import logging
import httpx
from dotenv import load_dotenv

from backend.services.document_reader import read_documents
from backend.services.groq_keys import make_async_client, key_pool_size
from backend.services.llm_endpoint import resolve_endpoint, auth_headers

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _chunk_text(text: str, max_chars: int = 8000) -> list[str]:
    """Split text into chunks that fit within LLM context window."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind("\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        chunks.append(text[start:boundary].strip())
        start = boundary
    return [c for c in chunks if c]


def _parse_json_from_response(text: str) -> list[dict]:
    """Extract a JSON array from LLM response text."""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning("Could not parse JSON from response: %s", text[:200])
        return []


QA_GENERATION_PROMPT = """You are a QA dataset generator for evaluating RAG (Retrieval-Augmented Generation) systems.

Given the document text below, generate exactly {n} diverse question-answer pairs.

Rules:
- Questions must be specific and answerable ONLY from the given text
- Answers must be concise, factual, and directly supported by the text
- Cover different sections and topics in the document
- Avoid yes/no questions — prefer questions that need a substantive answer
- Do NOT add any explanation outside the JSON

Document:
\"\"\"
{text}
\"\"\"

Return ONLY a valid JSON array with this exact structure (no markdown, no backticks):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


async def _generate_with_groq(prompt: str) -> str:
    ep = await resolve_endpoint()
    async with make_async_client(timeout=60.0) as client:
        res = await client.post(
            ep.chat_completions,
            headers=auth_headers(ep),
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                # Explicit: an absent `stream` makes some gateway providers
                # reply with SSE, which .json() below cannot parse.
                "stream": False,
            },
        )
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]


async def _generate_with_gemini(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


async def _generate_with_openai(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
        )
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]


async def generate_qa_from_document(
    document_paths: list[str] | str,
    target_count: int = 15,
    pairs_per_chunk: int | None = None,
) -> list[dict]:
    """
    Generate question-answer pairs from a document using whichever provider has
    a key configured (Groq, Gemini, or OpenAI).

    `target_count` is the total number of pairs wanted across the whole
    document; the per-chunk quota is derived from it. `pairs_per_chunk` is kept
    as an explicit override for callers that want to drive that directly.
    """
    # Check provider availability
    provider = None
    if key_pool_size() > 0:
        provider = "groq"
    elif GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"):
        provider = "gemini"
    elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"):
        provider = "openai"

    if not provider:
        raise ValueError("No valid API Key found in .env! Please set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.")

    try:
        paths = document_paths if isinstance(document_paths, list) else [document_paths]
        docs = read_documents(paths)
    except Exception as e:
        raise RuntimeError(f"Could not read document(s): {e}")

    if not docs:
        raise ValueError("Document(s) are empty")
    # Concatenated here: this fallback has no source tracking to preserve.
    content = "\n\n".join(text for _filename, text in docs)

    chunks = _chunk_text(content, max_chars=8000)[:4]
    all_pairs: list[dict] = []
    seen_questions: set[str] = set()

    # Spread the requested total across the chunks we actually have. Ask for a
    # little extra per chunk, since dedup and parse failures thin the results.
    if pairs_per_chunk is None:
        per_chunk = math.ceil(max(target_count, 1) / max(len(chunks), 1))
        pairs_per_chunk = max(1, min(per_chunk + 1, 15))

    for i, chunk in enumerate(chunks):
        if len(all_pairs) >= target_count:
            break
        prompt = QA_GENERATION_PROMPT.format(n=pairs_per_chunk, text=chunk)
        try:
            if provider == "groq":
                raw_response = await _generate_with_groq(prompt)
            elif provider == "gemini":
                raw_response = await _generate_with_gemini(prompt)
            else:
                raw_response = await _generate_with_openai(prompt)

            pairs = _parse_json_from_response(raw_response)
            for pair in pairs:
                if (
                    isinstance(pair, dict)
                    and pair.get("question")
                    and pair.get("answer")
                    and pair["question"] not in seen_questions
                ):
                    seen_questions.add(pair["question"])
                    all_pairs.append({
                        "question": pair["question"].strip(),
                        "answer": pair["answer"].strip(),
                    })
        except Exception as e:
            logger.error(f"{provider.capitalize()} QA generation failed for chunk {i}: {e}")
            raise RuntimeError(f"{provider.capitalize()} API call failed: {e}")

    if not all_pairs:
        raise RuntimeError(f"{provider.capitalize()} returned no valid QA pairs — check document content and API key.")

    return all_pairs[:target_count]
