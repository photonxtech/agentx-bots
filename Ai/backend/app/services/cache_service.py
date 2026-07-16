from app.config import settings
from app.utils.cache import TTLCache

# Generation counters version-scope both caches below: every crawl/sync/reindex/delete
# bumps a website's generation, which changes the cache key, so stale entries are simply
# never looked up again rather than needing an explicit invalidation sweep.
_generations: dict[int, int] = {}

# Retrieval results (post-rerank chunk lists) are deterministic and moderately expensive
# (embedding + BM25 + cross-encoder rerank) — safe to reuse across identical questions,
# including "Regenerate", since only the LLM call itself needs to vary for that.
_retrieval_cache = TTLCache(ttl_seconds=settings.response_cache_ttl_seconds, max_size=512)

# Final generated answers ("common responses"). Deliberately NOT consulted for
# "Regenerate" requests — a cached answer would defeat the point of asking for another
# attempt — but a fresh regenerate result still overwrites the cache for next time.
_response_cache = TTLCache(ttl_seconds=settings.response_cache_ttl_seconds, max_size=512)


def bump_generation(website_id: int) -> None:
    _generations[website_id] = _generations.get(website_id, 0) + 1


def _key(website_id: int, question: str) -> str:
    generation = _generations.get(website_id, 0)
    normalized = " ".join(question.strip().lower().split())
    return f"{website_id}:{generation}:{normalized}"


def get_cached_retrieval(website_id: int, question: str) -> list[dict] | None:
    return _retrieval_cache.get(_key(website_id, question))


def set_cached_retrieval(website_id: int, question: str, chunks: list[dict]) -> None:
    _retrieval_cache.set(_key(website_id, question), chunks)


def get_cached_response(website_id: int, question: str) -> dict | None:
    return _response_cache.get(_key(website_id, question))


def set_cached_response(website_id: int, question: str, payload: dict) -> None:
    _response_cache.set(_key(website_id, question), payload)
