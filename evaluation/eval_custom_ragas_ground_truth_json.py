"""
Ground-truth-based custom eval — adds the 2 metrics that need a correct
reference answer (ported from the same coworker's approach as
eval_custom_ragas.py): context_recall, answer_correctness.

Reads ground_truth_template.csv (export your filled-in xlsx/Sheets to CSV
first, into this same folder) and scores only the rows with a non-empty
ground_truth cell.

Uses the SAME fresh GROQ_API_KEY_RAGAS as eval_custom_ragas.py — run this
in a new terminal tab if that script is still finishing, or wait for it to
free up the key's rate limit.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

JSON_PATH = Path(__file__).parent / "benchmark_120.json"

RAGAS_EVAL_MODEL = "llama-3.1-8b-instant"
RAGAS_MAX_RETRIES = 3
RAGAS_TIMEOUT_S = 30
RAGAS_MAX_RETRY_WAIT = 60
RAGAS_MAX_TOKENS_CAP = 2000

_API_KEY = os.getenv("GROQ_API_KEY_RAGAS") or os.getenv("GROQ_API_KEY")
if not _API_KEY:
    raise SystemExit("Set GROQ_API_KEY_RAGAS (or GROQ_API_KEY) in your .env / shell.")

_groq_client = Groq(api_key=_API_KEY)

from sentence_transformers import SentenceTransformer

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
            logger.warning("Judge call failed: %s", msg[:200])
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


def context_recall(ground_truth, contexts):
    if not ground_truth or not ground_truth.strip() or not contexts:
        return None
    context_blob = "\n\n".join(contexts)
    return _claim_coverage(ground_truth, context_blob)


_CORRECTNESS_SYS = (
    "You compare a candidate ANSWER to a GROUND TRUTH reference answer for the "
    "same question. Judge them at the level of individual factual statements. "
    'Respond with a JSON object of the form {"tp": <int>, "fp": <int>, "fn": <int>}:\n'
    "tp = number of statements in the ANSWER that are also supported by the GROUND TRUTH.\n"
    "fp = number of statements in the ANSWER that are NOT supported by the GROUND TRUTH "
    "(wrong, invented, or contradicting it).\n"
    "fn = number of statements in the GROUND TRUTH that are missing from the ANSWER.\n"
    "If both texts are empty or contain no checkable statements, respond with "
    '{"tp": 0, "fp": 0, "fn": 0}.'
)


def answer_correctness(answer, ground_truth):
    if not answer or not answer.strip() or not ground_truth or not ground_truth.strip():
        return None
    try:
        resp = _judge_json(_CORRECTNESS_SYS, f"GROUND TRUTH:\n{ground_truth}\n\nANSWER:\n{answer}")
        if resp is None:
            return None
        try:
            tp, fp, fn = int(resp["tp"]), int(resp["fp"]), int(resp["fn"])
        except (KeyError, TypeError, ValueError):
            return None
        f1 = 1.0 if tp + fp + fn == 0 else tp / (tp + 0.5 * (fp + fn))
        vecs = embed([answer, ground_truth])
        semantic_sim = float(np.clip(vecs[0] @ vecs[1], 0.0, 1.0))
        return 0.75 * f1 + 0.25 * semantic_sim
    except Exception:
        logger.exception("answer_correctness error")
        return None


def main():
    if not JSON_PATH.exists():
        raise SystemExit(
            f"Couldn't find {JSON_PATH}"
        )

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    print(f"Found {len(data)} labeled rows.\n")

    results = []
    for i, row in enumerate(data):
        contexts = row.get("contexts", [])
        gt = row["ground_truth"]
        answer = row.get("answer", "")
        gt = row.get("ground_truth", "")
        correctness = answer_correctness(answer, gt)
        results.append({
            "question": row["query"],
            "context_recall": _round(recall),
            "answer_correctness": _round(correctness),
        })
        print(f"[{i+1}/{len(data)}] recall={_round(recall)} correctness={_round(correctness)}")

    out_df = pd.DataFrame(results)
    out_df.to_csv("custom_ragas_ground_truth_results.csv", index=False)

    print("\n=== Aggregate scores (ground-truth-based) ===")
    for col in ["context_recall", "answer_correctness"]:
        print(f"{col}: {out_df[col].mean():.3f}  (n={out_df[col].notna().sum()})")
    print(f"\nSaved to custom_ragas_ground_truth_results.csv ({len(out_df)} rows).")


if __name__ == "__main__":
    main()
