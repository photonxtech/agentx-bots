import time
from threading import Lock
from typing import Any


class TTLCache:
    """A small in-process TTL+LRU cache. Not distributed — fine for a single backend
    process; a multi-instance deployment would need a shared cache (e.g. Redis) instead."""

    def __init__(self, ttl_seconds: int, max_size: int = 256):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self._max_size and key not in self._store:
                oldest_key = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest_key]
            self._store[key] = (time.time() + self._ttl, value)

    def clear_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in [k for k in self._store if k.startswith(prefix)]:
                del self._store[key]
