"""
Live RAGAS-style evaluation — 4 reference-free metrics, computed right after
every real chat answer, using only the question/answer/retrieved context
(no ground truth needed, so this is safe to run on real user questions).

context_recall and answer_correctness are intentionally NOT here — both need
a known-correct reference answer, which a real user's live question doesn't
have. Those stay offline-only (see eval_custom_ragas_ground_truth.py).

Uses GROQ_API_KEY_RAGAS if set (recommended: keep this separate from your
app's main GROQ_API_KEY so live eval calls don't compete with answer
generation for the same rate limit), falling back to GROQ_API_KEY.
"""

import json
import logging
from multiprocessing import pool
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

print("LOADED METRICS FILE")
print(__file__)

key = os.getenv("GROQ_API_KEY_RAGAS")
print("RAGAS key:", key[:10] + "..." if key else None)

logger = logging.getLogger(__name__)

RAGAS_EVAL_MODEL = "llama-3.1-8b-instant"
RAGAS_MAX_RETRIES = 2          # kept low — this runs inline in a live request
RAGAS_TIMEOUT_S = 15
RAGAS_MAX_RETRY_WAIT = 10       # don't stall a live chat response for long
RAGAS_MAX_TOKENS_CAP = 1500
RAGAS_RELEVANCY_N = 3

_API_KEY = os.getenv("GROQ_API_KEY_RAGAS") or os.getenv("GROQ_API_KEY")
_groq_client = Groq(api_key=_API_KEY) if _API_KEY else None

_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(texts):
    return _embed_model.encode(texts, normalize_embeddings=True)


_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)\b", re.IGNORECASE)


def _retry_after_seconds(message):
    m = _RETRY_AFTER_RE.search(message or "")
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    value = float(m.group(2))
    seconds = value / 1000.0 if m.group(3).lower() == "ms" else value
    return minutes * 60 + seconds


def _judge_json(system, user, max_tokens=500):
    if _groq_client is None:
        return None
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
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
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
            logger.warning("Live eval judge call failed: %s", msg[:200])
            return None
    parsed = _parse_json(text)
    return parsed if isinstance(parsed, dict) else None


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
    "in the same order — 1 if the context supports the claim, 0 if it does not or is unclear."
)


def _claim_coverage(text, context):
    try:
        claims_resp = _judge_json(_CLAIMS_SYS, f"Answer:\n{text}", max_tokens=800)
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


_CTXPREC_SYS = (
    "You are given a QUESTION and a numbered list of retrieved CONTEXT chunks. "
    "For each chunk, decide whether it is useful for answering the question. "
    'Respond with a JSON object of the form {"verdicts": [1, 0, ...]}: exactly '
    "one integer per chunk, in the same order — 1 if the chunk is relevant/useful, "
    "0 if not."
)


def context_precision(question, contexts):
    print("Number of contexts:", len(contexts), flush=True)
    if not contexts:
        return None
    try:
        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        resp = _judge_json(_CTXPREC_SYS, f"QUESTION: {question}\n\nCONTEXT CHUNKS:\n{numbered}")
        print("\n===== CONTEXT PRECISION RAW RESPONSE =====")
        print(resp)
        print("=========================================\n")
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


_CTXREL_SYS = (
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
        resp = _judge_json(_CTXREL_SYS, f"QUESTION: {question}\n\nCONTEXT:\n{context_blob}", max_tokens=700)
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


def evaluate(question, answer, contexts):
    """Score one real answer on all 4 reference-free metrics, concurrently.
    Never raises — any judge failure yields None for that metric so a live
    chat answer can never break because of this."""
    context_blob = "\n\n".join(contexts)
    with ThreadPoolExecutor(max_workers=4) as pool:
        faith_f = pool.submit(faithfulness, answer, context_blob)
        rel_f = pool.submit(answer_relevancy, question, answer)
        print("Submitting context_precision...")
        prec_f = pool.submit(context_precision, question, contexts)
        print("Submitted context_precision")
        ctxrel_f = pool.submit(context_relevancy, question, contexts)
        faith = faith_f.result()
        print("Faithfulness:", faith)

        rel = rel_f.result()
        print("Answer relevancy:", rel)

        prec = prec_f.result()
        print("Context precision:", prec)

        ctxrel = ctxrel_f.result()
        print("Context relevancy:", ctxrel)
    return {
        "faithfulness": _round(faith),
        "answer_relevancy": _round(rel),
        "context_precision": _round(prec),
        "context_relevancy": _round(ctxrel),
    }
