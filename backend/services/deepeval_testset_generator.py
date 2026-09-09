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
    build_groq_judge, build_generator_llm, groq_ready, DEEPEVAL_MODEL,
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


def _build_contexts(document_paths: list[str], num_contexts: int,
                    randomize: bool = False) -> list[list[str]]:
    """Chunk every document and group chunks into contexts for the synthesizer.

    Grouping stays within a single file: a context mixing two unrelated documents
    would produce a question no retrieved chunk can answer coherently.

    When *randomize* is True the pool is shuffled before selection so that
    successive runs sample different parts of the corpus — essential for Genesis
    feasibility packages where 6-10 Q&A must cover as many of the 39 fields as
    possible rather than hitting the same few chunks every time.
    """
    import random
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

    # Down-select when we have more contexts than requested.
    if num_contexts and len(contexts) > num_contexts:
        if randomize:
            # Shuffle then take the first N — every run sees a different slice
            # of the corpus, so each generation covers different fields.
            random.shuffle(contexts)
            contexts = contexts[:num_contexts]
        else:
            # Deterministic: evenly-spaced selection across the whole corpus so
            # that multi-file packages don't only question the first file.
            step = len(contexts) / num_contexts
            contexts = [contexts[int(i * step)] for i in range(num_contexts)]

    return contexts


# Mapping from each of the 39 field names to keyword sets that reliably
# detect whether a generated question/answer is *about* that field.  Keyword
# matching is case-insensitive; a question matches a field when ANY keyword
# for that field appears in the question text or the answer text.
_GENESIS_FIELD_KEYWORDS: dict[str, list[str]] = {
    "date of report approved":       ["date of report", "report approved", "report date"],
    "project address / title":       ["project address", "project title", "property address", "street address"],
    "sponsor":                       ["sponsor", "development company", "sponsoring"],
    "borrower entity":               ["borrower entity", "borrower", "borrowing entity", "llc"],
    "project status":                ["project status", "approval status", "approved with conditions"],
    "report created by":             ["report created by", "analyst", "prepared by"],
    "additional comments":           ["additional comments", "executive narrative", "sponsor track record"],
    "third-party review":            ["third-party review", "third party review", "third-party report"],
    "third-party reviewer":          ["third-party reviewer", "third party reviewer", "inspection company", "trinity", "granite", "dcmi"],
    "third-party review (good/bad)": ["appropriate", "not appropriate", "good/bad", "good or bad"],
    "third-party review (meet or fail)": ["meet or fail", "meets best practice", "best practice", "recommended for approval"],
    "genesis agree":                 ["genesis agree", "agrees", "disagree"],
    "project timeline to date":      ["timeline to date", "elapsed time", "project commencement"],
    "remaining timeline":            ["remaining timeline", "remaining time", "certificate of occupancy"],
    "appropriate / not appropriate (timeline)": ["timeline appropriate", "timeline.*appropriate"],
    "draw hold":                     ["draw hold"],
    "specify draw hold items":       ["draw hold items", "hold items"],
    "other special conditions":      ["special conditions"],
    "project type":                  ["project type", "rehab", "new construction", "gut renovation"],
    "rehab amount":                  ["rehab amount", "rehabilitation amount", "rehab cost"],
    "construction holdback amount":  ["construction holdback", "holdback amount"],
    "project cost per square foot":  ["cost per square foot", "cost per sq", "price per square"],
    "cost per structure":            ["cost per structure"],
    "cost per unit":                 ["cost per unit"],
    "contingency amount":            ["contingency amount", "contingency fund"],
    "contingency (%)":               ["contingency %", "contingency percent"],
    "project complete percentage":   ["project complete", "complete percentage", "completion percentage", "percent complete"],
    "budget review":                 ["budget review"],
    "additional budget comments":    ["budget comments", "additional budget"],
    "plan status":                   ["plan status", "plans status"],
    "plan review status":            ["plan review"],
    "permit status":                 ["permit status", "permits"],
    "permits (post funding)":        ["post funding", "permits.*post"],
    "property type":                 ["property type", "multi-family", "mixed use", "single family"],
    "region":                        ["region"],
    "no. of units":                  ["no. of units", "number of units", "units planned", "how many units", "total units"],
    "no. of stories":                ["no. of stories", "number of stories", "how many stories", "stories"],
    "no. of structures":             ["no. of structures", "number of structures", "how many structures"],
    "gross buildable square footage": ["gross buildable", "square footage", "gfa", "gross floor"],
}


def _match_field(question: str, answer: str) -> str | None:
    """Return the Genesis field name a Q&A pair is about, or None."""
    combined = (question + " " + answer).lower()
    for field, keywords in _GENESIS_FIELD_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return field
    return None


def _enforce_genesis_field_diversity(pairs: list[dict], target_count: int) -> list[dict]:
    """Select *target_count* Q&A pairs that cover the widest range of the 39
    Genesis feasibility fields.

    Strategy (two-pass):
      1. Walk the pool in shuffled order.  For each pair, detect which field it
         targets.  If that field has not been covered yet, accept the pair.
      2. Fill remaining slots from pairs whose fields have NOT yet been covered
         (even if we couldn't detect the field name, they may still add
         breadth), then from any remaining pairs.

    The shuffle in pass 1 ensures that successive runs don't always pick the
    same pair for a given field.
    """
    import random

    # Pre-classify every pair
    tagged: list[tuple[dict, str | None]] = []
    for pair in pairs:
        q = pair.get("question", "")
        a = pair.get("answer", "")
        # Skip useless answers
        a_lower = a.lower()
        if any(ph in a_lower for ph in (
            "not provided", "not specified", "does not contain",
            "does not provide", "no information",
        )):
            continue
        tagged.append((pair, _match_field(q, a)))

    # Shuffle so each run picks a different representative per field
    random.shuffle(tagged)

    covered_fields: set[str] = set()
    selected: list[dict] = []
    remaining: list[tuple[dict, str | None]] = []

    # Pass 1 — one question per unique field
    for pair, field in tagged:
        if len(selected) >= target_count:
            break
        if field and field not in covered_fields:
            selected.append(pair)
            covered_fields.add(field)
        else:
            remaining.append((pair, field))

    # Pass 2 — fill with uncovered-field or untagged pairs, then leftovers
    if len(selected) < target_count:
        # Prioritise pairs about fields we haven't covered yet
        uncovered = [(p, f) for p, f in remaining if f and f not in covered_fields]
        generic   = [(p, f) for p, f in remaining if f is None]
        covered_dupes = [(p, f) for p, f in remaining if f and f in covered_fields]
        for pool in (uncovered, generic, covered_dupes):
            for pair, field in pool:
                if len(selected) >= target_count:
                    break
                selected.append(pair)
                if field:
                    covered_fields.add(field)

    logger.info(
        f"Genesis field diversity: {len(selected)} questions covering "
        f"{len(covered_fields)} unique fields out of {len(_GENESIS_FIELD_KEYWORDS)}"
    )
    return selected


def _generate_sync(document_paths: list[str], testset_size: int, notes: list[str]) -> list[dict]:
    """Blocking — call via asyncio.to_thread."""
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import (
        EvolutionConfig, FiltrationConfig, StylingConfig,
    )
    from deepeval.synthesizer.types import Evolution
    from backend.services.genesis_field_detector import (
        is_genesis_feasibility_package, GENESIS_39_FIELDS
    )

    # Detect if this is a Genesis feasibility package
    is_genesis, genesis_details = is_genesis_feasibility_package(document_paths)

    # generator_llm drives Q&A synthesis — prefers OmniRoute when available.
    # critic_judge scores/filters the generated questions — always Groq so that
    # quality thresholds are consistent regardless of the generator backend.
    generator_llm = build_generator_llm()
    critic_judge  = build_groq_judge()

    # Two goldens per context keeps the number of contexts (and therefore the
    # context-construction cost) roughly half the requested size.
    max_per_context = 2
    num_contexts = max(1, -(-testset_size // max_per_context))  # ceil

    # For Genesis: over-generate with more contexts so the diversity filter has
    # a large pool covering many fields.  3× gives enough headroom for 6-10 Q&A
    # to span most of the 39 fields without blowing through rate limits.
    if is_genesis:
        num_contexts = min(num_contexts * 3, max(num_contexts, 15))

    contexts = _build_contexts(document_paths, num_contexts, randomize=is_genesis)

    evolution_config = EvolutionConfig(
        num_evolutions=1,  # Enable evolution (Highest Quality)
        evolutions={
            Evolution.REASONING: 0.25,
            Evolution.MULTICONTEXT: 0.25,
            Evolution.CONCRETIZING: 0.2,
            Evolution.CONSTRAINED: 0.15,
            Evolution.COMPARATIVE: 0.15,
        },
    )

    # HIGH QUALITY CONFIGURATION: Strict quality filtering and evolution
    filtration_config = FiltrationConfig(
        synthetic_input_quality_threshold=0.5,  # Strict quality threshold
        max_quality_retries=1,  # Allow 1 retry if question is below threshold
        critic_model=critic_judge,   # keeps quality threshold consistent
    )

    # Configure styling based on document type
    if is_genesis:
        logger.info(f"Genesis feasibility package detected with {genesis_details['confidence']} confidence. "
                   f"Generating field-focused Q&A for the 39 standard fields.")
        
        styling_config = StylingConfig(
            scenario=f"""You are evaluating a RAG system for Genesis Capital Feasibility Review documents. 
The system must accurately extract and answer questions about the 39 standard feasibility fields:
{GENESIS_39_FIELDS}

Generate questions that test the system's ability to extract specific field values and understand field relationships.

DIVERSITY REQUIREMENT: Ask about DIFFERENT fields in each question. Do NOT repeat questions about the same field (e.g., avoid asking "What is the Rehab Amount?" multiple times). Spread questions across the 39 fields.

CRITICAL: For numeric data (costs, amounts, percentages, counts, square footage, dates) - answers MUST contain EXACT numbers from the source document. DO NOT calculate, estimate, round, or generate plausible numbers EXCEPT for the 5 calculated fields (Project Cost per Square Foot, Cost per Structure, Cost per Unit, Contingency %, Project Complete Percentage). For all other numeric fields: COPY the exact numeric values that appear in the context.""",
            task="Extract and answer questions ONLY about the 39 Genesis feasibility fields from the document context. Focus on field values, calculations, and field-based analysis. For all numeric data: use EXACT values from the context without modification. PRIORITIZE VARIETY - ask about different fields, not the same field repeatedly.",
            input_format="A natural question asking about one or more of the 39 feasibility fields (e.g., 'What is the Rehab Amount?', 'What is the Project Address?', 'How many Units are planned?'). Each question should target a DIFFERENT field.",
            expected_output_format="A concise, factual answer containing the specific field value(s) drawn only from the context. Numeric values must be EXACT copies from the source text, not calculated or estimated. If the field is not in the context, state that clearly.",
        )
        
        notes.append(
            f"Genesis feasibility package detected ({genesis_details['genesis_files']}/{genesis_details['total_files']} files) — "
            f"Q&A generation focused on the 39 standard feasibility fields."
        )
    else:
        logger.info("General document detected. Generating standard Q&A.")
        
        styling_config = StylingConfig(
            scenario="Someone evaluating a RAG system asks factual questions answerable only from the supplied document.",
            task="Answer strictly from the given context, without adding outside knowledge.",
            input_format="A natural question a real user would type, phrased as they would actually phrase it.",
            expected_output_format="A concise, factual answer drawn only from the context.",
        )

    synthesizer = Synthesizer(
        model=generator_llm,
        async_mode=False,            # SERIAL: prevents rate-limit storms on Groq free tier
        max_concurrent=1,
        evolution_config=evolution_config,
        filtration_config=filtration_config,
        styling_config=styling_config,
        cost_tracking=False,
    )

    # Note which backend is serving this generation run so it appears in the
    # session's qa_meta and the UI's provenance banner.
    notes.append(
        f"Q&A generated via {generator_llm.get_model_name()} · "
        f"quality-filtered by Groq/{critic_judge.model_name}"
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
        cleaned_contexts = [
            c.replace("\x00", "").replace("\u0000", "") if isinstance(c, str) else str(c)
            for c in (g.context or [])
        ]
        pairs.append({
            # The contract the evaluator consumes.
            "question": question,
            "answer": answer,
            # Provenance, so a generated case can be traced back to how it was
            # produced and which chunks it came from.
            "reference_contexts": cleaned_contexts,
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

    # For Genesis documents: enforce field diversity to avoid repeating same field questions
    if is_genesis:
        pairs = _enforce_genesis_field_diversity(pairs, testset_size)

    if len(pairs) > testset_size:
        pairs = pairs[:testset_size]
    return pairs


async def generate_testset(
    document_paths: list[str],
    testset_size: int = 6,
    seed_questions: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """Generate a test set with DeepEval. Returns (pairs, meta).

    seed_questions — optional list of questions already present in the session's
    seed (user-uploaded) test set.  Any generated question that matches a seed
    question (case-insensitive, normalised) is filtered out so the final
    generated set contains only NEW questions.
    """
    notes: list[str] = []

    if not document_paths:
        raise ValueError("No documents available to generate a test set from.")
    if not groq_ready():
        raise ValueError("No GROQ_API_KEY configured — cannot generate a test set.")

    # Scope both stats to this run.
    reset_rotation_stats()
    reset_model_fallback_stats()

    pairs = await asyncio.to_thread(_generate_sync, document_paths, testset_size, notes)

    # ── Deduplicate against seed questions ────────────────────────────────────
    import re

    def _norm(q: str) -> str:
        return re.sub(r"\s+", " ", (q or "").strip().lower())

    filtered_count = 0
    if seed_questions:
        seed_norms = {_norm(q) for q in seed_questions}
        before = len(pairs)
        pairs = [p for p in pairs if _norm(p["question"]) not in seed_norms]
        filtered_count = before - len(pairs)
        if filtered_count:
            notes.append(
                f"{filtered_count} generated question(s) matched your uploaded seed Q&A and "
                f"were removed to avoid duplicates."
            )

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
        "duplicates_filtered": filtered_count,
        "key_pool": keys,
        "model_fallbacks": fallbacks,
        "notes": notes,
    }

