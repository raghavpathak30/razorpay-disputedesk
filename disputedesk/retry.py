"""Generic exponential-backoff retry helper, shared by every network-calling
client (`disputedesk/client/razorpay.py`, `disputedesk/evidence/llm.py`) so
retry/backoff behaviour is written and tested exactly once (SPEC.md §7
failure path 1; PHASES.md Phase 4 - "retry with exponential backoff on
timeout" and "backoff lives in the shared client path, not only in the eval
script").

Retries only on two transient, infrastructure-level conditions:
- `httpx.TimeoutException` (the network call itself timed out)
- `httpx.HTTPStatusError` with status 429 (rate limited)

Any other exception - including a non-429 `httpx.HTTPStatusError` - is not
retried and propagates on the first attempt, since it is not the kind of
condition a retry can fix.
"""

import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")

# (attempt_number, outcome, error) - outcome is one of "success", "timeout_retry",
# "rate_limited_retry", "failed". Called after every attempt, success or not, if given.
AttemptCallback = Callable[[int, str, Exception | None], None]


def call_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_attempt: AttemptCallback | None = None,
) -> T:
    """Call `fn()`, retrying with exponential backoff (`base_delay_seconds *
    2**(attempt-1)`, capped at `max_delay_seconds`) up to `max_retries` times
    past the first attempt. A 429's `Retry-After` header, if present,
    overrides the computed delay for that wait. Raises the triggering error
    once retries are exhausted.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            result = fn()
        except httpx.TimeoutException as error:
            if attempt > max_retries:
                if on_attempt:
                    on_attempt(attempt, "failed", error)
                raise
            if on_attempt:
                on_attempt(attempt, "timeout_retry", error)
            sleep_fn(min(base_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds))
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 429:
                if on_attempt:
                    on_attempt(attempt, "failed", error)
                raise
            if attempt > max_retries:
                if on_attempt:
                    on_attempt(attempt, "failed", error)
                raise
            computed_delay = base_delay_seconds * (2 ** (attempt - 1))
            wait_seconds = float(error.response.headers.get("retry-after", computed_delay))
            if on_attempt:
                on_attempt(attempt, "rate_limited_retry", error)
            sleep_fn(min(wait_seconds, max_delay_seconds))
        else:
            if on_attempt:
                on_attempt(attempt, "success", None)
            return result
