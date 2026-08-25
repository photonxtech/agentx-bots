"""
Custom RAGAS-style eval — ported from a coworker's hand-written approach
(no `ragas` package, no LangChain). Runs the SAME 4 reference-free metrics,
with the SAME logic/prompts, against OUR real 109 production Q&A pairs from
sessions.db — so the resulting numbers are genuinely comparable to hers,
not just similarly named.

REQUIRES a Groq API key with quota still available today. If you reuse the
same key from eval_ragas.py, this will likely hit the same daily-limit wall
immediately — use a FRESH key from a different account if today's quota is
already spent (check console.groq.com on that account to confirm).

Metrics (all reference-free, no ground truth needed):
  - faithfulness       : fraction of the answer's claims supported by context
  - answer_relevancy    : does the answer address the question (via generated-
                          question embedding similarity)
  - context_precision   : are relevant chunks ranked near the top?
  - context_relevancy    : what fraction of retrieved text is actually relevant
                          (density/signal-to-noise, ignores ranking)

Run this in the SAME isolated venv as eval_ragas.py (it already has the groq
SDK and sentence-transformers installed).
"""

import json
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
DB_PATH = BASE.parent / "sessions.db"

print("Using DB:", DB_PATH.resolve())
# --------------------------------------------------------------------------- #
# Config — tune here if needed
# --------------------------------------------------------------------------- #
# eval_custom_ragas.py, eval_custom_ragas_ground_truth.py, eval_custom_ragas_ground_truth_json.py
RAGAS_EVAL_MODEL = "openai/gpt-oss-20b"   # was: "llama-3.1-8b-instant"
RAGAS_MAX_RETRIES = 3
RAGAS_TIMEOUT_S = 30
RAGAS_MAX_RETRY_WAIT = 60  # give up on a rate-limit wait longer than this
RAGAS_MAX_TOKENS_CAP = 2000
RAGAS_RELEVANCY_N = 3
JUDGE_CONTEXT_CHAR_CAP = 3000  # cap per-context text sent to the judge model —
# broad-summary contexts can be ~10,000 chars, and llama-3.1-8b-instant
# frequently fails to return valid JSON when given that much text at once
# (see json_validate_failed warnings). This caps what the JUDGE sees only —
# sessions.db and the real app answer are unaffected.

# Use GROQ_API_KEY_RAGAS if set (a separate/fresh key), else fall back
_API_KEY = os.getenv("GROQ_API_KEY_RAGAS") or os.getenv("GROQ_API_KEY")
if not _API_KEY:
    raise SystemExit(
        "No Groq API key found. Set GROQ_API_KEY_RAGAS (recommended: a FRESH "
        "key from a different account, since today's quota on the main key "
        "is likely already spent) or GROQ_API_KEY in your .env."
    )

_groq_client = Groq(api_key=_API_KEY)

# Local embedding model — same family (MiniLM) as the app's retriever, no
# LangChain wrapper needed for this script.
from sentence_transformers import SentenceTransformer

_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(texts: list[str]) -> np.ndarray:
    return _embed_model.encode(texts, normalize_embeddings=True)


# --------------------------------------------------------------------------- #
# Judge call helpers (ported as-is)
# --------------------------------------------------------------------------- #
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)\b", re.IGNORECASE)


def _retry_after_seconds(message: str):
    m = _RETRY_AFTER_RE.search(message or "")
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    value = float(m.group(2))
    seconds = value / 1000.0 if m.group(3).lower() == "ms" else value
    return minutes * 60 + seconds


def _judge_json(system: str, user: str, max_tokens: int = 500):
    text = None
    current_max_tokens = max_tokens
    for attempt in range(RAGAS_MAX_RETRIES + 1):
        try:
            resp = _groq_client.chat.completions.create(
                model=RAGAS_EVAL_MODEL,
                temperature=0.0,
                max_tokens=current_max_tokens,
                timeout=RAGAS_TIMEOUT_S,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
            is_token_limit = "json_validate_failed" in msg or "max completion tokens" in msg.lower()
            wait = _retry_after_seconds(msg)
            if is_rate_limit and attempt < RAGAS_MAX_RETRIES and wait is not None and wait <= RAGAS_MAX_RETRY_WAIT:
                time.sleep(wait + 0.5)
                continue
            if is_token_limit and attempt < RAGAS_MAX_RETRIES and current_max_tokens < RAGAS_MAX_TOKENS_CAP:
                current_max_tokens = min(current_max_tokens * 2, RAGAS_MAX_TOKENS_CAP)
                continue
            logger.warning("Judge call failed: %s", msg[:200])
            return None

    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        logger.warning("Judge returned non-JSON-object reply: %.200r", text)
        return None
    return parsed


def _parse_json(text):
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _round(x):
    return None if x is None else round(float(x), 3)


def _int_list(values):
    out = []
    for v in values:
        try:
            out.append(1 if int(v) == 1 else 0)
        except (TypeError, ValueError):
            out.append(0)
    return out


# --------------------------------------------------------------------------- #
# 1. Faithfulness
# --------------------------------------------------------------------------- #
_CLAIMS_SYS = (
    "You break an answer into a list of standalone, atomic factual claims. "
    "Each claim must be a single, self-contained statement that can be checked "
    "as true or false. Ignore opinions, questions, filler, and citations. "
    'Respond with a JSON object of the form {"claims": ["Claim one.", "Claim two."]}. '
    'If the answer contains no factual claims, respond with {"claims": []}.'
)
_VERIFY_SYS = (
    "You are given a CONTEXT and a numbered list of CLAIMS. For each claim, decide "
    "whether it can be directly inferred from the context. Respond with a JSON "
    'object of the form {"verdicts": [1, 0, ...]}: exactly one integer per claim, '
    "in the same order — 1 if the context supports the claim, 0 if it does not "
    "or is unclear."
)


def _claim_coverage(text, context):
    try:
        claims_resp = _judge_json(_CLAIMS_SYS, f"Answer:\n{text}", max_tokens=1000)
        if claims_resp is None:
            return None
        claims = claims_resp.get("claims")
        if not isinstance(claims, list):
            return None
        claims = [str(c) for c in claims if str(c).strip()]
        if not claims:
            return 1.0
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
        verify_resp = _judge_json(_VERIFY_SYS, f"CONTEXT:\n{context}\n\nCLAIMS:\n{numbered}")
        if verify_resp is None:
            return None
        verdicts = verify_resp.get("verdicts")
        if not isinstance(verdicts, list) or len(verdicts) != len(claims):
            return None
        supported = sum(_int_list(verdicts))
        return supported / len(claims)
    except Exception:
        logger.exception("claim coverage error")
        return None


def faithfulness(answer, context):
    return _claim_coverage(answer, context)


# --------------------------------------------------------------------------- #
# 2. Answer relevancy
# --------------------------------------------------------------------------- #
_GENQ_SYS = (
    "Given an ANSWER, generate {n} diverse questions that the answer would be a "
    "direct and complete response to. Do not use outside knowledge. "
    'Respond with a JSON object of the form {{"questions": [... {n} strings ...]}}.'
)


def answer_relevancy(question, answer, n=None):
    n = n or RAGAS_RELEVANCY_N
    try:
        resp = _judge_json(_GENQ_SYS.format(n=n), f"ANSWER:\n{answer}")
        if resp is None:
            return None
        gen = resp.get("questions")
        if not isinstance(gen, list):
            return None
        gen_qs = [str(q) for q in gen if str(q).strip()]
        if not gen_qs:
            return None
        vecs = embed([question] + gen_qs)
        q_vec, gen_vecs = vecs[0], vecs[1:]
        sims = gen_vecs @ q_vec
        return float(np.clip(sims, 0.0, 1.0).mean())
    except Exception:
        logger.exception("answer_relevancy error")
        return None


# --------------------------------------------------------------------------- #
# 3. Context precision
# --------------------------------------------------------------------------- #
_CTXREL_SYS = (
    "You are given a QUESTION and a numbered list of retrieved CONTEXT chunks. "
    "For each chunk, decide whether it is useful for answering the question. "
    'Respond with a JSON object of the form {"verdicts": [1, 0, ...]}: exactly '
    "one integer per chunk, in the same order — 1 if the chunk is relevant/useful, "
    "0 if not."
)


def context_precision(question, contexts):
    if not contexts:
        return None
    try:
        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        resp = _judge_json(_CTXREL_SYS, f"QUESTION: {question}\n\nCONTEXT CHUNKS:\n{numbered}")
        if resp is None:
            return None
        verdicts = resp.get("verdicts")
        if not isinstance(verdicts, list) or len(verdicts) != len(contexts):
            return None
        rel = _int_list(verdicts)
        total_relevant = sum(rel)
        if total_relevant == 0:
            return 0.0
        hits, precision_sum = 0, 0.0
        for k, r in enumerate(rel, 1):
            if r:
                hits += 1
                precision_sum += hits / k
        return precision_sum / total_relevant
    except Exception:
        logger.exception("context_precision error")
        return None


# --------------------------------------------------------------------------- #
# 4. Context relevancy
# --------------------------------------------------------------------------- #
_CTXREV_SYS = (
    "You are given a QUESTION and a CONTEXT passage made up of several chunks. "
    "Extract ONLY the sentences from the context that are directly relevant to "
    "answering the question, copied verbatim (do not paraphrase). "
    'Respond with a JSON object of the form {"relevant_sentences": ["sentence one.", ...]}. '
    'If no sentences are relevant, respond with {"relevant_sentences": []}.'
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentence_count(text):
    return len([s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()])


def context_relevancy(question, contexts):
    if not contexts:
        return None
    context_blob = "\n\n".join(contexts)
    total = _sentence_count(context_blob)
    if total == 0:
        return None
    try:
        resp = _judge_json(_CTXREV_SYS, f"QUESTION: {question}\n\nCONTEXT:\n{context_blob}", max_tokens=800)
        if resp is None:
            return None
        sentences = resp.get("relevant_sentences")
        if not isinstance(sentences, list):
            return None
        relevant = len([s for s in sentences if str(s).strip()])
        return min(1.0, relevant / total)
    except Exception:
        logger.exception("context_relevancy error")
        return None


# --------------------------------------------------------------------------- #
# Per-question evaluate (4 metrics concurrently, like the original)
# --------------------------------------------------------------------------- #
def evaluate_one(question, answer, contexts):
    capped_contexts = [c[:JUDGE_CONTEXT_CHAR_CAP] for c in contexts]
    context_blob = "\n\n".join(capped_contexts)
    with ThreadPoolExecutor(max_workers=4) as pool:
        faith_f = pool.submit(faithfulness, answer, context_blob)
        rel_f = pool.submit(answer_relevancy, question, answer)
        prec_f = pool.submit(context_precision, question, capped_contexts)
        ctxrel_f = pool.submit(context_relevancy, question, capped_contexts)
        faith, rel, prec, ctxrel = faith_f.result(), rel_f.result(), prec_f.result(), ctxrel_f.result()
    return {
        "faithfulness": _round(faith),
        "answer_relevancy": _round(rel),
        "context_precision": _round(prec),
        "context_relevancy": _round(ctxrel),
    }


# --------------------------------------------------------------------------- #
# Data loading — same filtering as eval_ragas.py
# --------------------------------------------------------------------------- #
def is_scoreable(sources):
    if not sources:
        return False
    return any(s.get("text", "").strip() for s in sources)

# Standalone copy of rag_utils.is_broad_summary_question's detection logic —
# duplicated here (not imported) because rag_utils.py pulls in the full RAG
# stack (flashrank, langchain, etc.) which isn't installed in this eval venv.
# KEEP IN SYNC WITH rag_utils.py if BROAD_SUMMARY_PATTERNS ever changes there.
_FILE_NOUN = r"(pdf|document|doc|file|photo|image|picture|pic)s?"
_BROAD_SUMMARY_PATTERNS = [
    r"\bsummar(y|ize|ise|isation|ization)\b",
    rf"\bexplain\b.{{0,20}}\b{_FILE_NOUN}\b",
    r"\bexplain (this|these|them|it|that)\b",
    r"\b(give|provide) (me )?(an )?overview\b",
    r"\ball (of )?the details\b",
    rf"\bwhat('?s| is| does) (this|the) {_FILE_NOUN} (about|consist of|have|contain|show)\b",
    r"\btl;?dr\b",
    rf"\bwhat('?s| is|s)( there)? in (this|the) {_FILE_NOUN}\b",
    rf"\bwalk me through (this|the) {_FILE_NOUN}\b",
]


def is_broad_summary_question(question):
    q = question.lower()
    return any(re.search(pattern, q) for pattern in _BROAD_SUMMARY_PATTERNS)

# Standalone Python port of the frontend's isFallbackAnswer() (frontend/static/js/...).
# Duplicated here — different language, can't import — so answers like "I don't
# know" get EXCLUDED from scoring instead of being counted as faithfulness=1.0.
# KEEP IN SYNC WITH the JS version if those patterns ever change.
_FALLBACK_PATTERNS = [
    r"^i don'?t know\.?$",
    r"^i do not know\.?$",
    r"^i'?m not sure\.?$",
    r"^oops i don'?t know\.?$",
    r"i don'?t have (enough|sufficient) information",
    r"i (could not|couldn'?t) find",
    r"no relevant information",
    r"not (mentioned|found|available) in the (provided|given|uploaded)? ?(context|document|pdf)",
    r"the (document|context|pdf) does(n'?t| not) (contain|mention|include)",
]


def is_fallback_answer(text):
    if not text:
        return False
    t = text.strip()
    return any(re.search(pattern, t, re.IGNORECASE) for pattern in _FALLBACK_PATTERNS)


def load_real_qa_pairs():
    if not DB_PATH.exists():
        raise SystemExit(f"Couldn't find sessions.db at {DB_PATH}.")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT question, answer, sources FROM chat_turns ORDER BY id"
    ).fetchall()
    conn.close()

    total = 0
    json_errors = 0
    unscoreable = 0
    unscoreable_broad = 0
    duplicates = 0
    fallback_answers = 0

    records = []
    seen = set()

    for question, answer, sources_json in rows:
        total += 1

        try:
            sources = json.loads(sources_json)
        except Exception:
            json_errors += 1
            continue

        if not is_scoreable(sources):
            unscoreable += 1
            if is_broad_summary_question(question):
                unscoreable_broad += 1
            continue

        if is_fallback_answer(answer):
            fallback_answers += 1
            continue

        if question in seen:
            duplicates += 1
            continue

        seen.add(question)

        contexts = [
            s["text"]
            for s in sources
            if s.get("text", "").strip()
        ]

        records.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
        )

    print(f"Total rows      : {total}")
    print(f"JSON errors     : {json_errors}")
    print(f"Unscoreable     : {unscoreable}")
    print(f"Unscoreable (broad-summary): {unscoreable_broad}")
    print(f"Fallback answers (excluded): {fallback_answers}")
    print(f"Duplicates      : {duplicates}")
    print(f"Final records   : {len(records)}")

    return records


def main():
    records = load_real_qa_pairs()
    print(f"Loaded {len(records)} scoreable Q&A pairs from sessions.db.")
    print(f"Judge model: {RAGAS_EVAL_MODEL}\n")

    results = []
    for i, r in enumerate(records, 1):
        scores = evaluate_one(r["question"], r["answer"], r["contexts"])
        scores["question"] = r["question"]
        results.append(scores)
        print(
            f"[{i}/{len(records)}] "
            f"faith={scores['faithfulness']} rel={scores['answer_relevancy']} "
            f"prec={scores['context_precision']} ctxrel={scores['context_relevancy']}"
        )

    df = pd.DataFrame(results)
    df.to_csv("custom_ragas_results.csv", index=False)

    print("\n=== Aggregate scores (custom method, matches coworker's metrics) ===")
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_relevancy"]:
        print(f"{col}: {df[col].mean():.3f}  (n={df[col].notna().sum()})")

    print(f"\nPer-question breakdown saved to custom_ragas_results.csv ({len(df)} rows).")


if __name__ == "__main__":
    main()
