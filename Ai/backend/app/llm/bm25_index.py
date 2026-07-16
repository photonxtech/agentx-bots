import re

from rank_bm25 import BM25Okapi

from app.vectordb.collection import get_all_chunks

# In-process cache: rebuilding BM25 requires re-fetching and re-tokenizing every chunk in
# the collection, which is wasteful to redo on every single chat request. Invalidated
# explicitly by the crawl/sync/reindex pipeline whenever a website's content changes,
# rather than on a timer — so it never serves a stale index after new content is indexed.
_INDEX_CACHE: dict[int, tuple[BM25Okapi, list[dict]]] = {}

# Site vocabulary (unique tokens seen in the site's own content) is derived from the same
# corpus and shares its invalidation — used by the query spelling normalizer so brand
# names/abbreviations that aren't in a general English dictionary (e.g. "photonx", "ceo")
# never get "corrected" into a wrong real word.
_VOCAB_CACHE: dict[int, set[str]] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def invalidate_bm25_index(website_id: int) -> None:
    _INDEX_CACHE.pop(website_id, None)
    _VOCAB_CACHE.pop(website_id, None)


def site_vocabulary(website_id: int) -> set[str]:
    cached = _VOCAB_CACHE.get(website_id)
    if cached is not None:
        return cached

    corpus = get_all_chunks(website_id)
    vocab = {token for item in corpus for token in _tokenize(item["text"])}
    _VOCAB_CACHE[website_id] = vocab
    return vocab


def _get_or_build_index(website_id: int) -> tuple[BM25Okapi, list[dict]] | tuple[None, list]:
    cached = _INDEX_CACHE.get(website_id)
    if cached is not None:
        return cached

    corpus = get_all_chunks(website_id)
    if not corpus:
        return None, []

    tokenized = [_tokenize(item["text"]) for item in corpus]
    bm25 = BM25Okapi(tokenized)
    _INDEX_CACHE[website_id] = (bm25, corpus)
    return bm25, corpus


def bm25_search(website_id: int, question: str, top_k: int) -> list[dict]:
    bm25, corpus = _get_or_build_index(website_id)
    if bm25 is None:
        return []

    tokens = _tokenize(question)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [
        {"text": corpus[i]["text"], "metadata": corpus[i]["metadata"], "bm25_score": float(scores[i])}
        for i in ranked[:top_k]
        if scores[i] > 0
    ]
