"""
Ground-truth-based metrics for live chat turns — same claim-decomposition
judge logic as evaluation/eval_custom_ragas_ground_truth.py, just callable
per-turn instead of as a batch script. Only runs when ground_truth_lookup
finds a matching dataset row for the live question.
"""

import json
import logging
import os
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# eval_custom_ragas.py, eval_custom_ragas_ground_truth.py, eval_custom_ragas_ground_truth_json.py
RAGAS_EVAL_MODEL = "openai/gpt-oss-20b"   # was: "llama-3.1-8b-instant"
_API_KEY = os.getenv("GROQ_API_KEY_RAGAS") or os.getenv("GROQ_API_KEY")
_groq_client = Groq(api_key=_API_KEY)
_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _judge_json(system, user, max_tokens=1000):
    try:
        resp = _groq_client.chat.completions.create(
            model=RAGAS_EVAL_MODEL,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=30,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception as e:
        logger.warning("GT judge call failed: %s", str(e)[:200])
        return None


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


def context_recall(ground_truth, contexts):
    if not ground_truth.strip() or not contexts:
        return None
    claims_resp = _judge_json(_CLAIMS_SYS, f"Answer:\n{ground_truth}")
    if not claims_resp:
        return None
    claims = [str(c) for c in claims_resp.get("claims", []) if str(c).strip()]
    if not claims:
        return 1.0
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    context_blob = "\n\n".join(contexts)
    verify_resp = _judge_json(_VERIFY_SYS, f"CONTEXT:\n{context_blob}\n\nCLAIMS:\n{numbered}")
    if not verify_resp:
        return None
    verdicts = verify_resp.get("verdicts", [])
    if len(verdicts) != len(claims):
        return None
    supported = sum(1 for v in verdicts if int(v) == 1)
    return round(supported / len(claims), 3)


_CORRECTNESS_SYS = (
    "You compare a candidate ANSWER to a GROUND TRUTH reference answer for the "
    "same question. Judge them at the level of individual factual statements. "
    'Respond with a JSON object of the form {"tp": <int>, "fp": <int>, "fn": <int>}:\n'
    "tp = statements in ANSWER also supported by GROUND TRUTH.\n"
    "fp = statements in ANSWER NOT supported by GROUND TRUTH.\n"
    "fn = statements in GROUND TRUTH missing from ANSWER."
)


def answer_correctness(answer, ground_truth):
    if not answer.strip() or not ground_truth.strip():
        return None
    resp = _judge_json(_CORRECTNESS_SYS, f"GROUND TRUTH:\n{ground_truth}\n\nANSWER:\n{answer}")
    if not resp:
        return None
    try:
        tp, fp, fn = int(resp["tp"]), int(resp["fp"]), int(resp["fn"])
    except (KeyError, TypeError, ValueError):
        return None
    f1 = 1.0 if tp + fp + fn == 0 else tp / (tp + 0.5 * (fp + fn))
    vecs = _embed_model.encode([answer, ground_truth], normalize_embeddings=True)
    semantic_sim = float(np.clip(vecs[0] @ vecs[1], 0.0, 1.0))
    return round(0.75 * f1 + 0.25 * semantic_sim, 3)
