import time
from threading import Lock


class _ChatMetrics:
    """Process-local counters for basic observability. Not distributed — a
    multi-instance deployment would need a shared metrics backend (e.g. Prometheus)."""

    def __init__(self):
        self._lock = Lock()
        self.request_count = 0
        self.error_count = 0
        self.cache_hit_count = 0
        self.total_latency_ms = 0.0
        self.total_retrieval_ms = 0.0
        self.total_generation_ms = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.started_at = time.time()

    def record_request(
        self,
        total_ms: float,
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        error: bool = False,
        cache_hit: bool = False,
    ) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += total_ms
            self.total_retrieval_ms += retrieval_ms
            self.total_generation_ms += generation_ms
            if prompt_tokens:
                self.total_prompt_tokens += prompt_tokens
            if completion_tokens:
                self.total_completion_tokens += completion_tokens
            if error:
                self.error_count += 1
            if cache_hit:
                self.cache_hit_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            n = self.request_count or 1
            return {
                "uptime_seconds": round(time.time() - self.started_at),
                "chat_requests": self.request_count,
                "chat_errors": self.error_count,
                "cache_hits": self.cache_hit_count,
                "cache_hit_rate_pct": round(100 * self.cache_hit_count / n, 1),
                "avg_total_latency_ms": round(self.total_latency_ms / n, 1),
                "avg_retrieval_ms": round(self.total_retrieval_ms / n, 1),
                "avg_generation_ms": round(self.total_generation_ms / n, 1),
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
            }


chat_metrics = _ChatMetrics()
