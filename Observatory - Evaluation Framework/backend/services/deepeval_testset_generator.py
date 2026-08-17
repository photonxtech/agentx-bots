"""
Synthetic test set generation using DeepEval's Synthesizer.

Chosen deliberately over `generate_goldens_from_docs`: that path builds its own
contexts via `ContextConstructionConfig`, which needs an embedder and defaults
to OpenAI. Feeding our own chunks through `generate_goldens_from_contexts`
instead means:

  * no OpenAI key and no second embedding stack,
  * the same `document_reader` the retriever uses — including the docx table
    extraction fix, so the test set can never reference text the retriever
    was never given,
  * the same chunking knobs the rest of the app exposes.

DeepEval's *evolutions* harden generated questions: a base
question is rewritten to be harder along one of seven axes (reasoning,
multi-context, concretizing, constrained, comparative, hypothetical,
in-breadth).
"""
import os
import asyncio
import logging

from dotenv import load_dotenv

from backend.services.document_reader import read_documents
from backend.services.deepeval_llm import (
    build_groq_judge, groq_ready, DEEPEVAL_MODEL,
    model_fallback_stats, reset_model_fallback_stats,
)
from backend.services.groq_keys import rotation_stats, reset_rotation_stats

load_dotenv()

logger = logging.getLogger(__name__)

# Chunking for context construction. Larger than the retrieval default because
# a golden needs enough surrounding material to support a non-trivial question.
TESTSET_CHUNK_SIZE = int(os.getenv("DEEPEVAL_TESTSET_CHUNK_SIZE", "1024"))
TESTSET_CHUNK_OVERLAP = int(os.getenv("DEEPEVAL_TESTSET_CHUNK_OVERLAP", "100"))

# How many chunks are grouped into one "context" handed to the synthesizer.
# >1 is what makes MULTICONTEXT evolutions possible at all.
CHUNKS_PER_CONTEXT = int(os.getenv("DEEPEVAL_CHUNKS_PER_CONTEXT", "2"))


def _build_contexts(document_paths: list[str], num_contexts: int) -> list[list[str]]:
    """Chunk every document and group chunks into contexts for the synthesizer.

    Grouping stays within a single file: a context mixing two unrelated documents
    would produce a question no retrieved chunk can answer coherently.
    """
    from backend.services.rag_evaluator import _chunk_document

    docs = read_documents(document_paths)
    if not docs:
        raise ValueError("Document(s) are empty or produced no extractable text.")

    contexts: list[list[str]] = []
    for _filename, text in docs:
        chunks = _chunk_document(text, TESTSET_CHUNK_SIZE, TESTSET_CHUNK_OVERLAP)
        for i in range(0, len(chunks), CHUNKS_PER_CONTEXT):
            group = chunks[i:i + CHUNKS_PER_CONTEXT]
            if group:
                contexts.append(group)

    if not contexts:
        raise ValueError("Document(s) produced no chunks.")

    # Spread the requested count across the whole corpus rather than taking the
    # first N, which with several files would only ever question the first one.
    if num_contexts and len(contexts) > num_contexts:
        step = len(contexts) / num_contexts
        contexts = [contexts[int(i * step)] for i in range(num_contexts)]

    return contexts


def _generate_sync(document_paths: list[str], testset_size: int, notes: list[str]) -> list[dict]:
    """Blocking — call via asyncio.to_thread."""
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import (
        EvolutionConfig, FiltrationConfig, StylingConfig,
    )
    from deepeval.synthesizer.types import Evolution

    judge = build_groq_judge()

    # Two goldens per context keeps the number of contexts (and therefore the
    # context-construction cost) roughly half the requested size.
    max_per_context = 2
    num_contexts = max(1, -(-testset_size // max_per_context))  # ceil
    contexts = _build_contexts(document_paths, num_contexts)

    evolution_config = EvolutionConfig(
        num_evolutions=1,
        evolutions={
            Evolution.REASONING: 0.25,
            Evolution.MULTICONTEXT: 0.25,
            Evolution.CONCRETIZING: 0.2,
            Evolution.CONSTRAINED: 0.15,
            Evolution.COMPARATIVE: 0.15,
        },
    )

    # The critic re-rolls a question that scores below the threshold, and each
    # retry re-sends the full context — the main driver of tokens-per-day burn.
    # One retry rather than two: measured quality scores sat at 0.3-0.5 either
    # way, so the second retry was paying tokens for no observed gain.
    filtration_config = FiltrationConfig(
        synthetic_input_quality_threshold=0.5,
        max_quality_retries=int(os.getenv("DEEPEVAL_MAX_QUALITY_RETRIES", "1")),
        critic_model=judge,
    )

    styling_config = StylingConfig(
        scenario="Someone evaluating a RAG system asks factual questions answerable only from the supplied document.",
        task="Answer strictly from the given context, without adding outside knowledge.",
        input_format="A natural question a real user would type, phrased as they would actually phrase it.",
        expected_output_format="A concise, factual answer drawn only from the context.",
    )

    synthesizer = Synthesizer(
        model=judge,
        # Serial: Groq's free tier is ~30 req/min and DeepEval's default of 100
        # concurrent requests would be throttled into failure immediately.
        async_mode=False,
        max_concurrent=1,
        evolution_config=evolution_config,
        filtration_config=filtration_config,
        styling_config=styling_config,
        cost_tracking=False,
    )

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=max_per_context,
        _send_data=False,          # never publish a run to Confident AI
    )

    pairs: list[dict] = []
    for g in goldens:
        question = (g.input or "").strip()
        answer = (g.expected_output or "").strip()
        if not question or not answer:
            continue
        meta = g.additional_metadata or {}
        pairs.append({
            # The contract the evaluator consumes.
            "question": question,
            "answer": answer,
            # Provenance, so a generated case can be traced back to how it was
            # produced and which chunks it came from.
            "reference_contexts": list(g.context or []),
            "synthesizer": "deepeval",
            # Which of the seven evolution axes were applied to harden this
            # question, and the critic's score for it. `context_quality` is
            # NOT captured: DeepEval only populates it on the from-docs path,
            # and it is commented out in generate_goldens_from_contexts.
            "evolutions": meta.get("evolutions") or [],
            "synthetic_input_quality": meta.get("synthetic_input_quality"),
        })

    if not pairs:
        raise RuntimeError("DeepEval returned no usable goldens.")

    if len(pairs) > testset_size:
        pairs = pairs[:testset_size]
    return pairs


async def generate_testset(
    document_paths: list[str],
    testset_size: int = 6,
) -> tuple[list[dict], dict]:
    """Generate a test set with DeepEval. Returns (pairs, meta)."""
    notes: list[str] = []

    if not document_paths:
        raise ValueError("No documents available to generate a test set from.")
    if not groq_ready():
        raise ValueError("No GROQ_API_KEY configured — cannot generate a test set.")

    # Scope both stats to this run.
    reset_rotation_stats()
    reset_model_fallback_stats()

    pairs = await asyncio.to_thread(_generate_sync, document_paths, testset_size, notes)

    if len(pairs) < testset_size:
        notes.append(
            f"DeepEval produced {len(pairs)} of the {testset_size} requested questions — "
            f"the document supported no more contexts of usable quality."
        )

    # Report quota pressure. A run that only completed because it fell back is
    # not the same run as one served entirely by the requested model.
    keys = rotation_stats()
    if keys["rotations"]:
        notes.append(
            f"Groq key failover occurred {keys['rotations']} time(s) — an account hit "
            f"its rate limit and requests moved to another (pool of {keys['keys']} keys)."
        )
    fallbacks = model_fallback_stats()
    if fallbacks:
        served = ", ".join(f"{m} ({n} call(s))" for m, n in fallbacks.items())
        notes.append(
            f"'{DEEPEVAL_MODEL}' was rate-limited on every key, so part of this test set "
            f"was produced by: {served}. Tokens-per-day is per model per organisation, "
            f"so a substitute model is a separate daily budget."
        )

    return pairs, {
        "generator": "deepeval",
        "framework": "deepeval.synthesizer.Synthesizer",
        "model": DEEPEVAL_MODEL,
        "embedding_model": None,   # contexts are supplied, so no embedder is used
        "num_documents": len(document_paths),
        "requested": testset_size,
        "produced": len(pairs),
        "key_pool": keys,
        "model_fallbacks": fallbacks,
        "notes": notes,
    }
