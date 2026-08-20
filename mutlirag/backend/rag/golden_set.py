"""Golden-set lookup — matches a live question against curated Q&A pairs
from LangSmith dataset(s) the USER explicitly selects, per question.

context_recall and answer_correctness (see rag.evaluation) need a ground-truth
reference answer, which a real user's question never has. Earlier design used
one fixed global dataset for the whole app — wrong once a chat can hold
multiple documents, since the system has no reliable way to know which
document's golden set applies to a given question (a compound question can
span two documents; the retrieved evidence doesn't map cleanly to just one).

Instead, the "Calculate Metrics" UI shows a dropdown of every dataset
available in LangSmith (list_available_datasets(), multi-select) and the user
picks whichever one(s) are relevant to THAT question before scoring — sidesteps
the attribution problem entirely, and naturally supports checking a compound
answer against more than one dataset at once. No selection, or no close-enough
match within the selected set(s), and the answer still gets scored on the
three reference-free metrics only (see RagService.evaluate_answer).

Each named dataset is fetched from LangSmith once and cached in memory for the
process lifetime (avoids hammering the API on every question). There's no
local-file fallback — a dataset has to actually exist in LangSmith to be
selectable at all, since the dropdown is populated straight from
client.list_datasets().
"""

from __future__ import annotations

import logging
import os

import numpy as np

import config
from rag import embeddings

logger = logging.getLogger(__name__)

# Cache keyed by dataset name -> (questions, ground_truths, embedding matrix).
_dataset_cache: dict[str, tuple[list[str], list[str], np.ndarray]] = {}


def _langsmith_configured() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def list_available_datasets() -> list[str]:
    """Names of every LangSmith dataset available to pick from, or [] if
    LangSmith isn't configured or unreachable. Powers the "Calculate Metrics"
    dataset picker — never raises, a LangSmith outage just means an empty
    dropdown, not a broken UI.
    """
    if not _langsmith_configured():
        return []
    try:
        from langsmith import Client

        client = Client()
        return sorted(ds.name for ds in client.list_datasets())
    except Exception:
        logger.warning("Failed to list LangSmith datasets", exc_info=True)
        return []


def _load_dataset(name: str) -> tuple[list[str], list[str], np.ndarray]:
    """Fetch one dataset's question/ground_truth pairs from LangSmith,
    embed the questions once, and cache. Raises on failure — the caller
    (lookup) treats a single bad dataset name as a skip, not a hard failure
    for the whole call.
    """
    if name in _dataset_cache:
        return _dataset_cache[name]

    from langsmith import Client

    client = Client()
    dataset = client.read_dataset(dataset_name=name)
    examples = [ex for ex in client.list_examples(dataset_id=dataset.id) if ex.inputs.get("question")]
    questions = [ex.inputs["question"] for ex in examples]
    ground_truths = [ex.outputs.get("ground_truth", "") for ex in examples]
    vecs = (
        embeddings.embed(questions)
        if questions
        else np.zeros((0, embeddings.embedding_dim()), dtype="float32")
    )
    _dataset_cache[name] = (questions, ground_truths, vecs)
    return _dataset_cache[name]


def lookup(question: str, dataset_names: list[str] | None = None) -> str | None:
    """Return the ground_truth of the closest curated question across all
    `dataset_names`, or None if none were selected, none could be loaded, or
    the best match anywhere in them is below config.GOLDEN_SET_MATCH_THRESHOLD.

    Checking multiple datasets at once takes the single best match across all
    of them — not one match per dataset — so a compound question is scored
    against whichever curated question (from either dataset) it actually
    resembles most.
    """
    if not dataset_names or not question or not question.strip():
        return None

    q_vec = embeddings.embed([question])[0]
    best_score = -1.0
    best_ground_truth = None
    for name in dataset_names:
        try:
            questions, ground_truths, vecs = _load_dataset(name)
        except Exception:
            logger.warning("Failed to load golden dataset %r", name, exc_info=True)
            continue
        if not questions:
            continue
        sims = vecs @ q_vec
        idx = int(np.argmax(sims))
        if sims[idx] > best_score:
            best_score = float(sims[idx])
            best_ground_truth = ground_truths[idx]

    if best_ground_truth is not None and best_score >= config.GOLDEN_SET_MATCH_THRESHOLD:
        return best_ground_truth
    return None
