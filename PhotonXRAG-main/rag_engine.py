"""
PhotonX RAG - Retrieval + Generation Engine

Hybrid retrieval (BM25 + dense) -> Reciprocal Rank Fusion -> Cross-encoder rerank -> Groq Llama generation

Import `ask()` from a Streamlit (or FastAPI, later) app to power the copilot.
Assumes ingest.py has already been run against the .docx files in
source_docs/ and ./chroma_db exists.
"""

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Config - keep in sync with ingest.py
# ---------------------------------------------------------------------------
DB_DIR = "./chroma_db"
COLLECTION_NAME = "photonxtech"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Set your own key: export GROQ_API_KEY=... (or put it in .streamlit/secrets.toml).
# ONE key covers every Groq model - keys are per account, not per model - so the
# answer model here and the judge model in llm_metrics.py both authenticate with
# GROQ_API_KEY. A second key is only needed to move scoring to another provider.
#
# THIS MODEL AND THE JUDGE MODEL MUST DIFFER - see llm_metrics.JUDGE_MODEL for
# the full reasoning. Groq's token-per-day limit is per model, so keeping them
# apart means evaluation cannot spend the allowance the chat needs to answer.
#
# openai/gpt-oss-120b replaces llama-3.3-70b-versatile, which Groq deprecated on
# 2026-06-17. It is Groq's recommended successor - same quality tier, faster
# inference - and carries a 200k/day free-tier token allowance against
# llama-3.3's 100k, which roughly doubles how many questions a free key answers.
LLM_MODEL_NAME = os.environ.get("ANSWER_MODEL", "openai/gpt-oss-120b")

# gpt-oss models think before answering, and those reasoning tokens are billed
# like any other output token. "low" is a deliberate choice rather than a
# default: this task is grounded extraction from text already retrieved for the
# model, which does not reward deliberation, and on a free tier the tokens saved
# are questions answered. Raise it if answers start feeling shallow.
ANSWER_REASONING_EFFORT = os.environ.get("ANSWER_REASONING_EFFORT", "low")


def _reasoning_kwargs(model: str) -> dict:
    """reasoning_effort is a gpt-oss feature; other models reject it as an
    unknown field, so it is sent only where it is understood. Keeps
    ANSWER_MODEL free to point back at llama-3.3 or anything else."""
    return {"reasoning_effort": ANSWER_REASONING_EFFORT} if "gpt-oss" in model else {}

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60  # reciprocal rank fusion constant
FINAL_TOP_N = 6  # hard ceiling -- upper bound on chunks sent to the LLM

# We don't have a reliable way to hand-pick an absolute reranker-score
# threshold offline (its raw logit scale isn't something we can calibrate
# without seeing real score distributions from this corpus), and two
# attempts at fixed numbers proved it either way: too loose let unrelated
# sections through, too strict collapsed every query down to one source.
# Instead, _select_relevant() below looks at each query's own ranked score
# list and cuts at the first real "cliff" -- a drop-off that's large
# relative to that query's own spread of scores -- rather than a fixed
# number tuned to a scale we can't see. See that function for how.

# How many fused candidates get handed to the (more expensive) cross-encoder
# reranker. This used to be max(DENSE_TOP_K, BM25_TOP_K) = 15, which silently
# dropped anything ranked below 15th in the fused list before the reranker
# ever saw it. On a small site like this (tens of chunks total), one longer
# page can produce several near-duplicate chunks that all score decently on
# both dense and BM25 search -- flooding the top of the fused ranking and
# pushing a single, uniquely-relevant chunk from another page (e.g. a
# specific project write-up) below the cutoff entirely. Since reranking a
# few dozen pairs is cheap, we simply rerank everything retrieved instead of
# pre-truncating -- let the reranker (which is actually good at judging
# relevance) make the call instead of an earlier, cruder ranking stage.
RERANK_CANDIDATE_CAP = 60

# Cap on how many of the FINAL_TOP_N slots a single URL can occupy. Without
# this, a verbose, generically-worded page (e.g. a long blog post chunked
# into 7 pieces) can dominate the context sent to the LLM purely by having
# more chunks in the index, crowding out other relevant sources.
MAX_CHUNKS_PER_SOURCE = 2

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = """You are a helpful and knowledgeable assistant answering questions based on the provided context from our documents.

Rules:
- Synthesize the retrieved context to answer the question as best as you can.
- If the exact answer isn't explicitly stated, try to infer it or provide the most relevant related information from the context.
- If the context is completely unrelated to the question, you can state that the documents don't provide a clear answer, but still offer a helpful summary of the related concepts that were retrieved.
- Keep answers conversational, direct, and helpful. Do not mention the internal mechanics of the retrieval or the file names.
"""


# ---------------------------------------------------------------------------
# Resource loading (cache these in the calling app - see app.py)
# ---------------------------------------------------------------------------
@dataclass
class RagResources:
    collection: any
    embedder: SentenceTransformer
    reranker: CrossEncoder
    bm25: BM25Okapi
    bm25_tokens: list
    all_ids: list
    all_docs: list
    all_metadatas: list


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_resources() -> RagResources:
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    raw = collection.get(include=["documents", "metadatas"])
    all_ids = raw["ids"]
    all_docs = raw["documents"]
    all_metadatas = raw["metadatas"]

    if not all_ids:
        raise RuntimeError(
            "Chroma collection is empty. Run ingest.py first to index the .docx "
            "files in source_docs/ (e.g. the PhotonX Company Profile)."
        )

    bm25_tokens = [_tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(bm25_tokens)

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    return RagResources(
        collection=collection,
        embedder=embedder,
        reranker=reranker,
        bm25=bm25,
        bm25_tokens=bm25_tokens,
        all_ids=all_ids,
        all_docs=all_docs,
        all_metadatas=all_metadatas,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _dense_search(res: RagResources, query: str, k: int) -> list[str]:
    """Returns a ranked list of chunk ids."""
    q_emb = res.embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    result = res.collection.query(query_embeddings=q_emb, n_results=min(k, len(res.all_ids)))
    return result["ids"][0]


def _bm25_search(res: RagResources, query: str, k: int) -> list[str]:
    """Returns a ranked list of chunk ids."""
    scores = res.bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [res.all_ids[i] for i in ranked_indices]


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)


def _id_to_doc(res: RagResources, doc_id: str) -> tuple[str, dict]:
    idx = res.all_ids.index(doc_id)
    return res.all_docs[idx], res.all_metadatas[idx]


MIN_RELEVANT = 3  # never return fewer than this many, if the pool has them --
                  # see _select_relevant for why


def _select_relevant(candidates: list[dict], max_n: int, min_n: int = MIN_RELEVANT) -> list[dict]:
    """
    Given candidates already sorted best-first by rerank_score, decides how
    many are actually relevant to this particular query -- 1, 2, or several,
    never a fixed count.

    Rather than a hand-picked absolute score cutoff (whose right value
    depends on this reranker's raw logit scale, which we can't calibrate
    offline), this looks at the shape of THIS query's own score list: it
    walks down the sorted scores and stops at the first "cliff" -- a
    drop-off between consecutive chunks that's large compared to the rest of
    the gaps in this list. Everything above the cliff is kept; everything
    below it is cut. A single standout match returns alone. Several
    similarly-strong matches (no real cliff between them) all survive
    together, up to max_n.
    """
    pool = candidates[: max(max_n, 10)]
    if len(pool) <= 1:
        return pool

    scores = [c["rerank_score"] for c in pool]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    avg_gap = sum(gaps) / len(gaps)
    score_range = scores[0] - scores[-1]

    cutoff = len(pool)  # no real cliff found in the pool -- keep it all
    for i, gap in enumerate(gaps):
        # A gap only counts as a genuine cliff if it's both clearly bigger
        # than the "background noise" between otherwise-similar chunks in
        # THIS list, and a meaningful chunk of this list's total spread --
        # so a list where every score is nearly identical (all similarly
        # on- or off-topic) doesn't get chopped on tiny noise.
        is_cliff = gap > avg_gap * 1.6 and (score_range == 0 or gap > 0.12 * score_range)
        if is_cliff:
            cutoff = i + 1
            break

    # On a small corpus (tens of chunks, like this one), a single misranked
    # top chunk -- e.g. a client testimonial that happens to share surface
    # wording with the query -- can create an artificial cliff right after
    # rank 1, discarding every other candidate including the one that's
    # actually correct. min_n is a safety floor: even when a cliff is
    # detected very early, at least min_n candidates (if the pool has that
    # many) still reach the LLM, which can read full context and discount an
    # irrelevant one itself -- cheap insurance on a small corpus where extra
    # context costs little, versus the alternative of retrieval silently
    # deciding the correct chunk doesn't exist.
    cutoff = max(cutoff, min(min_n, len(pool)))
    return pool[:cutoff]


def retrieve(res: RagResources, query: str) -> list[dict]:
    dense_ids = _dense_search(res, query, DENSE_TOP_K)
    bm25_ids = _bm25_search(res, query, BM25_TOP_K)
    fused_ids = _reciprocal_rank_fusion([dense_ids, bm25_ids])

    candidates = []
    for doc_id in fused_ids[: max(DENSE_TOP_K, BM25_TOP_K)]:
        text, meta = _id_to_doc(res, doc_id)
        candidates.append({"id": doc_id, "text": text, "metadata": meta})

    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = res.reranker.predict(pairs)
    for c, score in zip(candidates, rerank_scores):
        c["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    relevant = _select_relevant(candidates, FINAL_TOP_N)
    if not relevant:
        relevant = candidates[:1]

    return relevant[:FINAL_TOP_N]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _build_context_block(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        heading = " / ".join(h for h in [meta.get("h1"), meta.get("h2"), meta.get("h3")] if h)
        blocks.append(
            f"[Source {i} - {meta.get('title', 'Untitled document')}"
            f"{' - ' + heading if heading else ''}]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer_stream(query: str, chunks: list[dict], chat_history: list[dict]):
    """Yields text tokens as they stream in from Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        yield (
            "GROQ_API_KEY isn't set. Add it to your environment or "
            ".streamlit/secrets.toml as GROQ_API_KEY, then restart the app."
        )
        return

    client = Groq(api_key=api_key)

    context_block = _build_context_block(chunks)

    history_text = ""
    for turn in chat_history[-6:]:  # keep last few turns for follow-up context
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Conversation so far:
{history_text}

Context from the indexed company documents:
{context_block}

Current question: {query}

Answer the current question using the context above."""}
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=1024,
        **_reasoning_kwargs(LLM_MODEL_NAME),
    )

    # gpt-oss streams its reasoning in a separate `delta.reasoning` field rather
    # than mixing it into `delta.content`, so reading content alone yields the
    # answer only - no <think> block ever reaches the chat bubble. The falsy
    # check also covers the content-less deltas sent while the model reasons.
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def ask(res: RagResources, query: str, chat_history: list[dict]):
    """
    Full pipeline for one turn. Returns (chunks_used, generator_of_text).
    Call chunks_used to render source chips; iterate the generator to stream the answer.
    """
    chunks = retrieve(res, query)
    if not chunks:
        def empty_gen():
            yield (
                "I couldn't find anything relevant to that in the indexed documents. "
                "Try rephrasing, or ask about services, projects, or platform features."
            )
        return [], empty_gen()

    return chunks, generate_answer_stream(query, chunks, chat_history)
