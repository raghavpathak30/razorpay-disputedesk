"""Unit tests for the shared backoff helper (PHASES.md Phase 4: "retry with
exponential backoff on timeout" / "backoff lives in the shared client path,
not only in the eval script"). No network, no real sleeping - `sleep_fn` is
a no-op that only records what it was asked to wait.
"""

import httpx
import pytest

from disputedesk.retry import call_with_backoff


def _make_429(retry_after: str | None = None) -> httpx.HTTPStatusError:
    headers = {"retry-after": retry_after} if retry_after else {}
    request = httpx.Request("POST", "https://example.test/x")
    response = httpx.Response(429, headers=headers, request=request)
    return httpx.HTTPStatusError("rate limited", request=request, response=response)


def _make_500() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/x")
    response = httpx.Response(500, request=request)
    return httpx.HTTPStatusError("server error", request=request, response=response)


def test_succeeds_immediately_when_the_first_attempt_works():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    result = call_with_backoff(fn, sleep_fn=lambda _s: None)

    assert result == "ok"
    assert calls["n"] == 1


def test_retries_on_timeout_then_succeeds():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("timed out")
        return "recovered"

    sleeps: list[float] = []
    result = call_with_backoff(fn, sleep_fn=sleeps.append)

    assert result == "recovered"
    assert calls["n"] == 2
    assert sleeps == [0.5]  # base_delay_seconds * 2**0


def test_raises_after_exhausting_retries_on_persistent_timeout():
    def fn():
        raise httpx.ReadTimeout("always times out")

    with pytest.raises(httpx.ReadTimeout):
        call_with_backoff(fn, max_retries=2, sleep_fn=lambda _s: None)


def test_backoff_delay_grows_exponentially_and_is_capped():
    def fn():
        raise httpx.ConnectTimeout("always times out")

    sleeps: list[float] = []
    with pytest.raises(httpx.ConnectTimeout):
        call_with_backoff(
            fn,
            max_retries=4,
            base_delay_seconds=1.0,
            max_delay_seconds=3.0,
            sleep_fn=sleeps.append,
        )

    assert sleeps == [1.0, 2.0, 3.0, 3.0]  # 1, 2, 4->capped 3, 8->capped 3


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_429()
        return "ok"

    result = call_with_backoff(fn, sleep_fn=lambda _s: None)

    assert result == "ok"
    assert calls["n"] == 2


def test_429_honors_retry_after_header_over_computed_delay():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_429(retry_after="7")
        return "ok"

    sleeps: list[float] = []
    call_with_backoff(fn, base_delay_seconds=0.5, sleep_fn=sleeps.append)

    assert sleeps == [7.0]


def test_non_429_http_error_is_not_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _make_500()

    with pytest.raises(httpx.HTTPStatusError):
        call_with_backoff(fn, sleep_fn=lambda _s: None)

    assert calls["n"] == 1


def test_other_exceptions_are_not_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("not a network problem")

    with pytest.raises(ValueError):
        call_with_backoff(fn, sleep_fn=lambda _s: None)

    assert calls["n"] == 1


def test_on_attempt_callback_reports_every_attempt_and_outcome():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("timed out")
        return "ok"

    log: list[tuple[int, str]] = []
    call_with_backoff(
        fn,
        sleep_fn=lambda _s: None,
        on_attempt=lambda attempt, outcome, _error: log.append((attempt, outcome)),
    )

    assert log == [(1, "timeout_retry"), (2, "success")]
