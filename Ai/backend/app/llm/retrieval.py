import re
from datetime import datetime, timezone

from app.config import settings
from app.embeddings.embedder import embed_query
from app.llm.bm25_index import bm25_search
from app.llm.query_normalizer import correct_query_spelling
from app.llm.reranker import rerank_best_of
from app.services import cache_service
from app.utils.text_dedup import dedupe_key
from app.vectordb.collection import find_chunks_containing, query_collection

NO_ANSWER_MESSAGE = "I couldn't find that information on this website."

# Generic words too common to usefully narrow a keyword search — skipping them keeps
# _keyword_terms() focused on the distinctive nouns (brand names, product names) that
# semantic search tends to underrepresent.
_STOPWORDS = {
    "this", "that", "with", "from", "have", "does", "what", "who", "how", "why", "when",
    "where", "which", "about", "your", "their", "they", "them", "website", "project",
    "projects", "company", "please", "could", "would", "should", "some", "any", "there",
}

# A genuinely distinctive term (a brand/product name) should only appear in a handful of
# chunks. A term matching more than this is common boilerplate (e.g. the site's own name,
# which appears on nearly every page) — keyword-boosting it would just add noise that
# crowds out the real match, so those hits are discarded rather than included.
_MAX_USEFUL_KEYWORD_HITS = 5

_RRF_K = 60


def _keyword_terms(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z]{4,}", question)
    return [w for w in words if w.lower() not in _STOPWORDS]


def _keyword_candidates(website_id: int, question: str) -> list[dict]:
    # Small general-purpose embedding models represent rare proper nouns/brand names
    # poorly, so a chunk that literally names what was asked about can still rank low on
    # pure cosine similarity. Look the term up verbatim as a second, independent path —
    # this is exact substring matching (classic keyword search), not a canned answer.
    matches = []
    for term in _keyword_terms(question):
        term_hits = []
        seen_for_term: set[str] = set()
        for variant in {term, term.lower(), term.capitalize(), term.upper()}:
            for hit in find_chunks_containing(website_id, variant):
                key = dedupe_key(hit["text"])
                if key in seen_for_term:
                    continue
                seen_for_term.add(key)
                term_hits.append(hit)
        if 0 < len(term_hits) <= _MAX_USEFUL_KEYWORD_HITS:
            matches.extend(term_hits)
    return matches


def _rrf_merge(ranked_lists: list[list[dict]], k: int = _RRF_K) -> list[dict]:
    # Reciprocal Rank Fusion: combines independently-ranked lists (semantic, BM25, exact
    # keyword) by rank position rather than raw score, so a chunk ranking well on ANY
    # signal surfaces — the lists use incomparable scales (cosine similarity vs. BM25
    # score vs. "found at all"), so fusing on raw scores directly wouldn't be meaningful.
    pool: dict[str, dict] = {}
    rrf_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_position, item in enumerate(ranked_list):
            key = dedupe_key(item["text"])
            if key not in pool:
                pool[key] = item
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank_position + 1)

    merged = list(pool.values())
    merged.sort(key=lambda item: rrf_scores[dedupe_key(item["text"])], reverse=True)
    return merged


_MAX_CHUNKS_PER_SOURCE = 2


def _cap_chunks_per_source(chunks: list[dict], max_per_source: int = _MAX_CHUNKS_PER_SOURCE) -> list[dict]:
    # A page that happens to repeat a theme (e.g. several paragraphs all mentioning
    # "digital") can otherwise place 3-4 of its own chunks near the top of the ranking,
    # crowding out a different page that holds the actual answer but only scored one
    # relevant chunk. Capping how many chunks any single source contributes keeps the
    # final context diverse instead of dominated by one repetitive page.
    counts: dict[str, int] = {}
    capped = []
    for chunk in chunks:
        url = chunk["metadata"].get("url", "")
        if counts.get(url, 0) >= max_per_source:
            continue
        counts[url] = counts.get(url, 0) + 1
        capped.append(chunk)
    return capped


def _freshness_factor(indexed_at_iso: str | None) -> float:
    # Content doesn't really "expire" on a marketing/informational site the way a news
    # article does, so this is a gentle factor (0.85-1.0), not a hard penalty — recent
    # content scores a bit higher, but old-but-correct content is barely discounted.
    if not indexed_at_iso:
        return 0.9
    try:
        indexed_at = datetime.fromisoformat(indexed_at_iso)
    except ValueError:
        return 0.9
    if indexed_at.tzinfo is None:
        indexed_at = indexed_at.replace(tzinfo=timezone.utc)

    days_old = (datetime.now(timezone.utc) - indexed_at).days
    if days_old <= 30:
        return 1.0
    if days_old >= 365:
        return 0.85
    return 1.0 - 0.15 * ((days_old - 30) / (365 - 30))


def retrieve_relevant_chunks(
    website_id: int,
    question: str,
    top_k: int | None = None,
    threshold: float | None = None,
    use_cache: bool = True,
    website_name: str | None = None,
) -> list[dict]:
    top_k = top_k if top_k is not None else settings.top_k_chunks
    threshold = threshold if threshold is not None else settings.relevance_threshold

    if use_cache:
        cached = cache_service.get_cached_retrieval(website_id, question)
        if cached is not None:
            return cached

    # Small embedding models and cross-encoders are surprisingly fragile against typos —
    # a heavily misspelled question can embed nowhere near its intended meaning. Correct
    # obvious spelling mistakes before every retrieval signal; the original, uncorrected
    # question is still what's shown to the user and sent to the LLM for the final answer.
    corrected_question = correct_query_spelling(website_id, question)

    # A short/generic question ("how to contact", "team", "services") is implicitly about
    # THIS website — a real visitor never needs to say the site's name because they're
    # already looking at its page. But out of context, a few-word fragment embeds and
    # reranks weakly (the same fragility short queries show elsewhere). Naming the site
    # explicitly in the retrieval query restores that missing context — it doesn't change
    # what's being asked, just makes the "about this site" framing explicit for the models.
    search_query = f"{website_name}: {corrected_question}" if website_name else corrected_question

    query_embedding = embed_query(search_query)
    fetch_n = top_k * 8

    # Hybrid retrieval: semantic (embeddings/ANN) + BM25 (keyword statistics) + exact
    # substring matches, each an independent signal with different strengths/blind spots.
    semantic_candidates = query_collection(website_id, query_embedding, fetch_n)
    bm25_candidates = bm25_search(website_id, search_query, fetch_n)
    keyword_candidates = _keyword_candidates(website_id, search_query)

    fused = _rrf_merge([keyword_candidates, semantic_candidates, bm25_candidates])
    rerank_pool = fused[: max(top_k * 3, 20)]

    # Rerank the fused candidates with a cross-encoder for the final, most accurate
    # ordering — ANN/BM25 are fast-but-approximate; this trades a bit of latency for a
    # real (question, chunk) relevance judgment instead of separately-embedded similarity.
    # Score against both the bare and site-enriched query and keep each candidate's best
    # (see rerank_best_of) — enrichment helps some questions and actively hurts others.
    rerank_queries = [corrected_question] if search_query == corrected_question else [corrected_question, search_query]
    reranked = rerank_best_of(rerank_queries, rerank_pool, top_k=top_k * 2)

    above_threshold = [c for c in reranked if c["similarity"] >= threshold]
    filtered = _cap_chunks_per_source(above_threshold)[:top_k]

    if use_cache:
        cache_service.set_cached_retrieval(website_id, question, filtered)
    return filtered


def compute_confidence(chunks: list[dict]) -> float:
    # Composite confidence: relevance (post-rerank score) + freshness (how recently the
    # best source was indexed) + consistency (how many of the selected chunks corroborate
    # the top result, vs. resting on a single match) — not just raw similarity.
    if not chunks:
        return 0.0

    best = chunks[0]
    relevance = max(0.0, min(1.0, best["similarity"]))
    freshness = _freshness_factor(best["metadata"].get("indexed_at"))

    agreeing = sum(1 for c in chunks if c["similarity"] >= relevance - 0.15)
    consistency = min(1.0, agreeing / 3)

    composite = 0.7 * relevance + 0.15 * freshness + 0.15 * consistency
    return round(max(0.0, min(1.0, composite)) * 100, 1)
