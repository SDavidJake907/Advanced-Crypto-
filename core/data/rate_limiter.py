from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate          # tokens refilled per second
        self._capacity = capacity  # max burst
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until the requested tokens are available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self._rate
            time.sleep(min(wait, 0.1))


# Kraken REST: safe at ~1 req/sec with burst of 3
kraken_rest_limiter = TokenBucketRateLimiter(rate=1.0, capacity=3.0)
