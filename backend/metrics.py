"""
Live RAGAS-style evaluation for frontend chat metrics.

Scores each live chat turn with 4 reference-free metrics using the existing
judge-based live evaluator. This file does not compute ground-truth-only
metrics such as context_recall or answer_correctness, which remain
offline-only in the evaluation scripts.
"""

import importlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from sentence_transformers import SentenceTransformer
import contextvars

load_dotenv()

print("LOADED METRICS FILE")
print(__file__)

logger = logging.getLogger(__name__)

# Upgraded from llama-3.1-8b-instant — the small model was noisier on
# structured JSON tasks (it's what caused the verdict-count mismatches you
# saw).
# eval_custom_ragas.py, eval_custom_ragas_ground_truth.py, eval_custom_ragas_ground_truth_json.py
#
# UPDATE (Aug 2026): llama-3.3-70b-versatile — what this eval pipeline was
# originally built and validated against — was deprecated by Groq (announced
# June 17, 2026) and now 404s as model_not_found. Groq's own migration
# guidance for it is openai/gpt-oss-120b or qwen/qwen3.6-27b.
#
# openai/gpt-oss-120b was already ruled out for THIS use case: it has an
# 8,000 TPM quota on this account tier, and evaluate() fires ~4 concurrent
# judge calls (each carrying ~2-4K tokens of retrieved context), which
# triggers a 429 storm almost immediately — that's what caused
# faithfulness/context_relevancy to come back None before. (It's still fine
# as the main answer/condense LLM in rag_utils.py, where calls are
# sequential, not concurrent — this constant is judge-only and independent
# of that.)
#
# Using qwen/qwen3.6-27b instead. Same deprecation-migration recommendation
# from Groq, without the 120b's TPM ceiling. Re-verify actual TPM limits on
# the Groq console for this account tier, and watch for 429s under the same
# 4-concurrent-call load before treating this as settled — if it turns out
# quota-constrained too, the next fallback is serializing/staggering the
# judge calls in evaluate() rather than switching models again.
RAGAS_EVAL_MODEL = "qwen/qwen3.6-27b"
RAGAS_MAX_RETRIES = 1          # kept low — this runs inline in a live request
RAGAS_TIMEOUT_S = 15
RAGAS_MAX_RETRY_WAIT = 6       # don't stall a live chat response for long
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


@traceable(name="ragas_judge_call", run_type="llm")
def _judge_json(system, user, max_tokens=500):
    if _groq_client is None:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.metadata["judge_call_failed"] = True
            run_tree.metadata["judge_call_error"] = "GROQ_API_KEY_RAGAS/GROQ_API_KEY not set"
            run_tree.tags = (run_tree.tags or []) + ["judge_error"]
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
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.metadata["judge_call_failed"] = True
                run_tree.metadata["judge_call_error"] = msg[:200]
                run_tree.tags = (run_tree.tags or []) + ["judge_error"]
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


def _salvage_squashed_verdicts(verdicts, expected_len):
    """openai/gpt-oss-20b has a reliable quirk: instead of N separate 0/1
    integers, it sometimes squashes them into ONE integer whose digits ARE
    the verdicts (e.g. 5 contexts -> [11101] instead of [1,1,1,0,1]). This
    is systematic, not random — decode it directly instead of burning a
    full retry (extra quota + extra latency) every single time it happens.
    Returns a clean list of exactly expected_len 0/1 ints, or None if the
    shape doesn't match this specific pattern."""
    if not isinstance(verdicts, list) or len(verdicts) != 1:
        return None
    digits = str(verdicts[0])
    if len(digits) != expected_len or not all(c in "01" for c in digits):
        return None
    return [int(c) for c in digits]


def _valid_binary_verdicts(verdicts):
    """True only if every entry cleanly parses to exactly 0 or 1.

    Guards against openai/gpt-oss-20b occasionally squashing multiple
    digits into one value (e.g. a verdicts list of [11000] instead of
    [1, 1, 0, 0, 0]). Coercing garbage like that through _int_list
    silently turns it into all-zeros, which then reads as "the judge
    found nothing relevant" instead of what actually happened: the
    judge's output was malformed and untrustworthy. Confirmed this was
    happening live — DeepEval scored context_relevancy 0.517 on the same
    context where this bug produced a fabricated 0.0.
    """
    for v in verdicts:
        try:
            if int(v) not in (0, 1):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _cosine_sim(a, b):
    if a is None or b is None:
        return None
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return None
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))

try:
    xai_chat = importlib.import_module("xai_sdk.chat")
    _orig_xai_user = xai_chat.user

    def _patched_xai_user(*args):
        fixed_args = []
        for arg in args:
            if isinstance(arg, list):
                if len(arg) == 1 and isinstance(arg[0], dict) and arg[0].get("type") == "text":
                    fixed_args.append(arg[0].get("text", ""))
                else:
                    fixed_args.append(str(arg))
            else:
                fixed_args.append(arg)
        return _orig_xai_user(*fixed_args)

    xai_chat.user = _patched_xai_user
except ImportError:
    xai_chat = None


def faithfulness_local(answer, context):
    if not answer or not context:
        return None
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    if not sentences:
        return None
    ctx_vec = embed([context])[0]
    sentence_vecs = embed(sentences)
    sims = [_cosine_sim(vec, ctx_vec) for vec in sentence_vecs]
    sims = [s for s in sims if s is not None]
    return float(np.mean(sims)) if sims else None


def answer_relevancy_local(question, answer):
    if not question or not answer:
        return None
    q_vec, a_vec = embed([question, answer])
    return _cosine_sim(q_vec, a_vec)


def context_precision_local(question, contexts):
    if not question or not contexts:
        return None
    q_vec = embed([question])[0]
    ctx_vecs = embed(contexts)
    sims = [float(np.dot(vec, q_vec)) for vec in ctx_vecs]
    rel = [1 if s >= 0.6 else 0 for s in sims]
    total_relevant = sum(rel)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, r in enumerate(rel, 1):
        if r:
            hits += 1
            precision_sum += hits / i
    return precision_sum / total_relevant


def context_relevancy_local(question, contexts):
    if not question or not contexts:
        return None
    q_vec = embed([question])[0]
    ctx_vecs = embed(contexts)
    sims = [float(np.dot(vec, q_vec)) for vec in ctx_vecs]
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def evaluate_local(question, answer, contexts):
    return {
        "faithfulness": _round(faithfulness_local(answer, "\n\n".join(contexts))),
        "answer_relevancy": _round(answer_relevancy_local(question, answer)),
        "context_precision": _round(context_precision_local(question, contexts)),
        "context_relevancy": _round(context_relevancy_local(question, contexts)),
    }


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

        def _get_verdicts():
            resp = _judge_json(_VERIFY_SYS, f"CONTEXT:\n{context}\n\nCLAIMS:\n{numbered}")
            if resp is None:
                return None
            v = resp.get("verdicts")
            if isinstance(v, list) and len(v) == len(claims) and _valid_binary_verdicts(v):
                return v
            salvaged = _salvage_squashed_verdicts(v, len(claims))
            if salvaged is not None:
                print(f"[faithfulness] salvaged squashed-digit verdicts: {salvaged}", flush=True)
                return salvaged
            return None

        verdicts = _get_verdicts()
        if verdicts is None:
            print("[faithfulness] missing/mismatched/malformed verdicts — retrying once", flush=True)
            verdicts = _get_verdicts()

        if verdicts is None:
            return None

        supported = sum(_int_list(verdicts))
        return supported / len(claims)
    except Exception:
        logger.exception("claim coverage error")
        return None


@traceable(name="metric_faithfulness")
def faithfulness(answer, context):
    return _claim_coverage(answer, context)


_GENQ_SYS = (
    "Given an ANSWER, generate {n} diverse questions that the answer would be a "
    "direct and complete response to. Do not use outside knowledge. "
    'Respond with a JSON object of the form {{"questions": [... {n} strings ...]}}.'
)


@traceable(name="metric_answer_relevancy")
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


@traceable(name="metric_context_precision")
def context_precision(question, contexts):
    print("Number of contexts:", len(contexts), flush=True)
    if not contexts:
        return None
    try:
        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))

        def _get_verdicts():
            resp = _judge_json(_CTXPREC_SYS, f"QUESTION: {question}\n\nCONTEXT CHUNKS:\n{numbered}")
            print("\n===== CONTEXT PRECISION RAW RESPONSE =====")
            print(resp)
            print("=========================================\n")
            if resp is None:
                return None
            v = resp.get("verdicts")
            if isinstance(v, list) and len(v) == len(contexts) and _valid_binary_verdicts(v):
                return v
            salvaged = _salvage_squashed_verdicts(v, len(contexts))
            if salvaged is not None:
                print(f"[context_precision] salvaged squashed-digit verdicts: {salvaged}", flush=True)
                return salvaged
            return None

        verdicts = _get_verdicts()
        if verdicts is None:
            print("[context_precision] missing/mismatched/malformed verdicts — retrying once", flush=True)
            verdicts = _get_verdicts()

        if verdicts is None:
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

# Rewritten from verbatim-extraction to numbered-sentence classification —
# the old approach asked the judge to copy relevant sentences EXACTLY, and
# any tiny wording drift (whitespace, a paraphrase, a typo fix) meant a truly
# relevant sentence silently failed to match and got undercounted. WE split
# the sentences ourselves (deterministic, no LLM involved), number them, and
# ask the judge for a straight relevant/not-relevant verdict per number —
# same reliable pattern as metric_context_precision, no string-matching involved.
_CTXREL_SYS = (
    "You are given a QUESTION and a numbered list of SENTENCES extracted from "
    "retrieved context. For each sentence, decide whether it is directly relevant "
    "to answering the question. Respond with a JSON object of the form "
    '{"verdicts": [1, 0, ...]}: exactly one integer per sentence, in the same '
    "order — 1 if relevant, 0 if not."
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@traceable(name="metric_context_relevancy")
def context_relevancy(question, contexts):
    if not contexts:
        return None
    context_blob = "\n\n".join(contexts)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(context_blob.strip()) if s.strip()]
    total = len(sentences)
    print(f"[context_relevancy] total sentences counted: {total}", flush=True)
    if total == 0:
        return None
    try:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))

        def _get_verdicts():
            resp = _judge_json(_CTXREL_SYS, f"QUESTION: {question}\n\nSENTENCES:\n{numbered}", max_tokens=700)
            if resp is None:
                return None
            v = resp.get("verdicts")
            if isinstance(v, list) and len(v) == total and _valid_binary_verdicts(v):
                return v
            salvaged = _salvage_squashed_verdicts(v, total)
            if salvaged is not None:
                print(f"[context_relevancy] salvaged squashed-digit verdicts: {salvaged}", flush=True)
                return salvaged
            return None

        verdicts = _get_verdicts()
        if verdicts is None:
            print("[context_relevancy] missing/mismatched/malformed verdicts — retrying once", flush=True)
            verdicts = _get_verdicts()

        if verdicts is None:
            return None

        relevant = sum(_int_list(verdicts))
        return min(1.0, relevant / total)
    except Exception:
        logger.exception("context_relevancy error")
        return None

@traceable(name="live_ragas_evaluation")
def evaluate(question, answer, contexts):
    """Score one real answer on all 4 reference-free metrics, concurrently.
    Never raises — any judge failure yields None for that metric so a live
    chat answer can never break because of this."""
    context_blob = "\n\n".join(contexts)

    # Each thread needs its OWN copy of the context — a single Context object
    # can't be entered by two threads at once (that's what caused the
    # "cannot enter context: already entered" RuntimeError). Copying once per
    # submission gives each thread an independent snapshot, so they can all
    # run concurrently while still correctly nesting under this run in LangSmith.

    # Small stagger between submissions — Groq's rate limit is per-minute
    # request count, not concurrency, so firing all 4 (really ~5-6 judge
    # calls once faithfulness's internal 2-call sequence and any mismatch
    # retries are counted) in the same instant is what triggers 429s. A
    # 300ms stagger spreads the burst without materially slowing down the
    # overall evaluate() wall-clock time, since the calls still run concurrently.
    STAGGER_S = 0.3

    with ThreadPoolExecutor(max_workers=4) as pool:
        faith_f = pool.submit(contextvars.copy_context().run, faithfulness, answer, context_blob)
        time.sleep(STAGGER_S)

        rel_f = pool.submit(contextvars.copy_context().run, answer_relevancy, question, answer)
        time.sleep(STAGGER_S)

        print("Submitting context_precision...")
        prec_f = pool.submit(contextvars.copy_context().run, context_precision, question, contexts)
        print("Submitted context_precision")
        time.sleep(STAGGER_S)

        ctxrel_f = pool.submit(contextvars.copy_context().run, context_relevancy, question, contexts)
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