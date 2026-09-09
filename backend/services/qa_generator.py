"""
QA test case generation using Groq, Gemini, or OpenAI.
Reads the uploaded document and generates Q&A pairs via LLM.
"""
import os
import json
import re
import math
import random
import asyncio
import logging
import httpx
from dotenv import load_dotenv

from backend.services.document_reader import read_documents
from backend.services.groq_keys import make_async_client, key_pool_size
from backend.services.llm_endpoint import resolve_endpoint, auth_headers

load_dotenv()

logger = logging.getLogger(__name__)

# Retry configuration for transient network errors
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 10.0

async def _retry_with_backoff(coro_factory, operation_name: str):
    """Execute async operation with exponential backoff retry for transient errors."""
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_factory()
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, 
                httpx.PoolTimeout, OSError) as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = min(BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.5), MAX_RETRY_DELAY)
                logger.warning(f"{operation_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"{operation_name} failed after {MAX_RETRIES} attempts: {e}")
        except httpx.HTTPStatusError as e:
            # Don't retry 4xx errors (client errors), only 5xx and 429
            if e.response.status_code >= 500 or e.response.status_code == 429:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.5), MAX_RETRY_DELAY)
                    logger.warning(f"{operation_name} attempt {attempt + 1} failed with {e.response.status_code}: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{operation_name} failed after {MAX_RETRIES} attempts: {e}")
            else:
                raise
    raise last_exception

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL   = os.getenv("NVIDIA_MODEL", "moonshotai/kimi-k3")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEY_2 = os.getenv("OPENROUTER_API_KEY_2", "")
OPENROUTER_API_KEY_3 = os.getenv("OPENROUTER_API_KEY_3", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
OPENROUTER_OX_ALPHA_MODEL = os.getenv("OPENROUTER_OX_ALPHA_MODEL", "stealth/ox-alpha")

QA_TEMPERATURE = 0.7


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


def _build_genesis_prompt(text: str, n: int, fields: str, variation: int) -> str:
    """Build a varied prompt for Genesis Q&A generation to ensure diversity across calls."""
    focus_variations = [
        "Focus on extracting specific field values and their relationships.",
        "Prioritize questions about numeric fields (amounts, percentages, counts, dates, square footage).",
        "Emphasize dropdown/categorical fields (Project Type, Property Type, Plan Status, Permit Status, etc.).",
        "Target narrative fields (Additional Comments, Budget Comments) and calculated fields.",
        "Cover header fields (Sponsor, Borrower Entity, Project Address) and timeline fields.",
        "Mix field types: ask about field definitions, field dependencies, and field cross-references.",
        "Focus on Loan Summary fields (Rehab Amount, Holdback, Contingency, Cost metrics, Budget Review).",
        "Target Executive Summary fields (Third-Party Review, Draw Hold, Timeline, Special Conditions).",
        "Emphasize Finished Product fields (Property Type, Units, Stories, Structures, GFA, Region).",
    ]
    
    # Cycle through variations and add a random element
    focus = focus_variations[variation % len(focus_variations)]
    random.seed(variation)  # Deterministic variation per call index
    
    return f"""You are a QA dataset generator for evaluating RAG systems on Genesis Capital Feasibility Review documents.

Given the document text below, generate exactly {n} DIVERSE question-answer pairs that focus EXCLUSIVELY on the 39 standard feasibility fields used in Genesis Capital reviews.

The 39 Genesis Feasibility Fields:
{fields}

STRICT RULES:
- Questions MUST ask about one or more of the 39 fields listed above
- Questions must be specific and answerable ONLY from the given document text
- Answers must be concise, factual, and directly supported by the text
- Focus on extracting field values, field relationships, and field-based analysis
- Examples: "What is the Rehab Amount?", "What is the Project Address?", "How many Units are in the project?", "What is the Third-Party Review status?", "What is the Contingency percentage?"
- Avoid questions about general construction topics not related to the 39 fields
- Avoid yes/no questions — prefer questions that extract specific field values
- Do NOT add any explanation outside the JSON

{focus} Vary your question styles: some should ask "What is...", others "How many...", "What is the status of...", "Which option...", "What does the document state about...".

CRITICAL - NUMERIC DATA RULES:
- For numbers (costs, amounts, percentages, counts, square footage, dates): COPY EXACTLY from the document text
- DO NOT calculate, estimate, round, or modify any numbers UNLESS the field is defined as a calculated field
- Calculated fields (ONLY these 5): Project Cost per Square Foot, Cost per Structure, Cost per Unit, Contingency %, Project Complete Percentage
- For calculated fields: Use the value stated in the document if available; otherwise acknowledge it needs calculation
- For all other numeric fields: COPY the exact value from the source text without modification
- DO NOT generate plausible-sounding numbers - use ONLY numbers that appear in the source text
- If a number includes units ($, %, SF, etc.), include those units in the answer
- Examples of CORRECT answers: "$374.74 per square foot" (from document), "51 units" (from document), "5-story" (from document)
- Examples of INCORRECT answers: Making up numbers that don't appear in the document

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
                "temperature": QA_TEMPERATURE,
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
                "temperature": QA_TEMPERATURE,
            },
        )
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]


async def _generate_with_nvidia(prompt: str) -> str:
    """Call Nvidia NIM API (Priority 1) with failover across configured keys/models (e.g. moonshotai/kimi-k3, deepseek/deepseek-v4-pro-0813)."""
    slots = []
    k1 = os.getenv("NVIDIA_API_KEY", "").strip()
    if k1 and not k1.startswith("your_"):
        slots.append({
            "key": k1,
            "base": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip().rstrip("/"),
            "model": os.getenv("NVIDIA_MODEL", "moonshotai/kimi-k3").strip(),
        })
    for idx in range(2, 10):
        k = os.getenv(f"NVIDIA_API_KEY_{idx}", "").strip()
        if k and not k.startswith("your_"):
            default_m = "deepseek/deepseek-v4-pro-0813" if idx == 2 else "moonshotai/kimi-k3"
            slots.append({
                "key": k,
                "base": os.getenv(f"NVIDIA_BASE_URL_{idx}", os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")).strip().rstrip("/"),
                "model": os.getenv(f"NVIDIA_MODEL_{idx}", default_m).strip(),
            })

    if not slots:
        raise ValueError("NVIDIA_API_KEY is not set or is a placeholder")

    for attempt, slot in enumerate(slots):
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                res = await client.post(
                    f"{slot['base']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {slot['key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": slot["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": QA_TEMPERATURE,
                    },
                )
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(
                "Nvidia NIM slot #%d (%s) failed (%s), %s",
                attempt + 1, slot["model"], str(e)[:100],
                f"trying slot #{attempt + 2}..." if attempt + 1 < len(slots) else "falling back to OpenRouter/Groq...",
            )

    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return await _generate_with_OPENROUTER(prompt)
    return await _generate_with_groq(prompt)


async def _generate_with_OPENROUTER(prompt: str) -> str:
    """Call OpenRouter with 3 keys: key1, key2 (nemotron), key3 (ox-alpha), then Groq."""
    OPENROUTER_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    OPENROUTER_key_2 = os.getenv("OPENROUTER_API_KEY_2", "").strip()
    OPENROUTER_key_3 = os.getenv("OPENROUTER_API_KEY_3", "").strip()
    OPENROUTER_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free").strip()
    OPENROUTER_ox_alpha_model = os.getenv("OPENROUTER_OX_ALPHA_MODEL", "stealth/ox-alpha").strip()
    
    if not OPENROUTER_model:
        OPENROUTER_model = "nvidia/nemotron-3.5-lightning:free"
    if not OPENROUTER_ox_alpha_model:
        OPENROUTER_ox_alpha_model = "stealth/ox-alpha"
    
    # Try primary OpenRouter key (nemotron)
    if OPENROUTER_key:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENROUTER_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": QA_TEMPERATURE,
                    },
                )
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Primary OpenRouter Nvidia failed ({e}), trying secondary key...")
    
    # Try secondary OpenRouter Nvidia key
    if OPENROUTER_key_2:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_key_2}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENROUTER_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": QA_TEMPERATURE,
                    },
                )
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Secondary OpenRouter Nvidia failed ({e}), trying ox-alpha key...")
    
    # Try third OpenRouter key (ox-alpha model)
    if OPENROUTER_key_3:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_key_3}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENROUTER_ox_alpha_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": QA_TEMPERATURE,
                    },
                )
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OpenRouter ox-alpha failed ({e}), falling back to Groq...")
    
    # Final fallback to Groq
    return await _generate_with_groq(prompt)


async def generate_qa_from_document(
    document_paths: list[str] | str,
    target_count: int = 15,
    pairs_per_chunk: int | None = None,
) -> list[dict]:
    """
    Generate question-answer pairs from a document using whichever provider has
    a key configured (Groq, Gemini, or OpenAI).
    
    For Genesis Capital feasibility documents, generates Q&A focused on the 39
    standard feasibility fields. For other documents, generates general Q&A.

    `target_count` is the total number of pairs wanted across the whole
    document; the per-chunk quota is derived from it. `pairs_per_chunk` is kept
    as an explicit override for callers that want to drive that directly.
    """
    from backend.services.genesis_field_detector import (
        is_genesis_feasibility_package, GENESIS_39_FIELDS
    )
    
    # Check provider availability - Priority 1: Nvidia NIM (moonshotai/kimi-k3 / deepseek)
    provider = None
    nvidia_has_key = any(
        os.getenv(f"NVIDIA_API_KEY{'' if i == 1 else f'_{i}'}", "").strip()
        and not os.getenv(f"NVIDIA_API_KEY{'' if i == 1 else f'_{i}'}", "").strip().startswith("your_")
        for i in range(1, 10)
    )
    if nvidia_has_key:
        provider = "NVIDIA"
    elif OPENROUTER_API_KEY and not OPENROUTER_API_KEY.startswith("your_"):
        provider = "OPENROUTER"
    elif key_pool_size() > 0:
        provider = "groq"
    elif GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"):
        provider = "gemini"
    elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"):
        provider = "openai"

    if not provider:
        raise ValueError("No valid API Key found in .env! Please set NVIDIA_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.")

    try:
        paths = document_paths if isinstance(document_paths, list) else [document_paths]
        docs = read_documents(paths)
    except Exception as e:
        raise RuntimeError(f"Could not read document(s): {e}")

    if not docs:
        raise ValueError("Document(s) are empty")
    
    # Detect if this is a Genesis feasibility package
    is_genesis, genesis_details = is_genesis_feasibility_package(paths)
    
    if is_genesis:
        logger.info(f"Genesis feasibility package detected with {genesis_details['confidence']} confidence. "
                   f"Generating field-focused Q&A for the 39 standard fields.")
    else:
        logger.info("General document detected. Generating standard Q&A.")
    
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

    # Use a random variation seed to ensure different questions each generation
    variation_seed = random.randint(0, 10000)

    for i, chunk in enumerate(chunks):
        if len(all_pairs) >= target_count:
            break
        
        # Use appropriate prompt based on document type
        if is_genesis:
            prompt = _build_genesis_prompt(chunk, pairs_per_chunk, GENESIS_39_FIELDS, variation_seed + i)
        else:
            prompt = QA_GENERATION_PROMPT.format(n=pairs_per_chunk, text=chunk)
        
        try:
            if provider == "NVIDIA":
                raw_response = await _retry_with_backoff(
                    lambda: _generate_with_nvidia(prompt), "Nvidia NIM QA generation"
                )
            elif provider == "OPENROUTER":
                raw_response = await _retry_with_backoff(
                    lambda: _generate_with_OPENROUTER(prompt), "OpenRouter QA generation"
                )
            elif provider == "groq":
                raw_response = await _retry_with_backoff(
                    lambda: _generate_with_groq(prompt), "Groq QA generation"
                )
            elif provider == "gemini":
                raw_response = await _retry_with_backoff(
                    lambda: _generate_with_gemini(prompt), "Gemini QA generation"
                )
            else:
                raw_response = await _retry_with_backoff(
                    lambda: _generate_with_openai(prompt), "OpenAI QA generation"
                )

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


