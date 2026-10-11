from __future__ import annotations

import pytest

from callinsights.ratelimit import SlidingWindowLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(max_attempts: int = 3, window: float = 60.0) -> tuple[SlidingWindowLimiter, Clock]:
    clock = Clock()
    return SlidingWindowLimiter(max_attempts, window, clock=clock), clock


def test_blocks_after_max_attempts() -> None:
    lim, _ = make(3)
    for _ in range(2):
        lim.hit("k")
        assert lim.is_blocked("k") is False
    lim.hit("k")
    assert lim.is_blocked("k") is True
    assert lim.is_blocked("other") is False


def test_window_expiry_unblocks() -> None:
    lim, clock = make(2, window=10)
    lim.hit("k")
    clock.now += 5
    lim.hit("k")
    assert lim.is_blocked("k") is True
    clock.now += 4.5
    assert lim.is_blocked("k") is True
    clock.now += 0.5  # first hit is now exactly one window old -> pruned
    assert lim.is_blocked("k") is False


def test_hits_accumulate_across_prunes() -> None:
    lim, clock = make(2, window=10)
    lim.hit("k")
    clock.now += 11
    lim.hit("k")  # old hit pruned, new one recorded
    assert list(lim._hits["k"]) == [1011.0]
    assert lim.is_blocked("k") is False
    lim.hit("k")
    assert lim.is_blocked("k") is True


def test_retry_after() -> None:
    lim, clock = make(2, window=10)
    assert lim.retry_after("k") == 0
    lim.hit("k")
    assert lim.retry_after("k") == 0
    lim.hit("k")
    assert lim.retry_after("k") == 10


def test_retry_after_uses_oldest_hit() -> None:
    lim, clock = make(2, window=10)
    lim.hit("k")
    clock.now += 4
    lim.hit("k")
    assert lim.retry_after("k") == 6  # the oldest hit leaves the window first
    clock.now += 3.5
    assert lim.retry_after("k") == 7  # ceil(6.5)
    clock.now += 6.25
    assert lim.retry_after("k") == 1  # ceil(0.25)


def test_reset_clears_key() -> None:
    lim, _ = make(1)
    lim.hit("k")
    assert lim.is_blocked("k") is True
    lim.reset("k")
    assert lim.is_blocked("k") is False
    lim.reset("missing")  # no error


def test_memory_is_bounded() -> None:
    lim, clock = make(5, window=1)
    lim.hit("a")
    assert "a" in lim._hits
    clock.now += 2
    assert lim.is_blocked("a") is False
    assert "a" not in lim._hits


@pytest.mark.parametrize(("attempts", "window"), [(0, 10), (1, 0), (1, -1)])
def test_invalid_config(attempts: int, window: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        SlidingWindowLimiter(attempts, window)


def test_minimum_valid_config() -> None:
    lim = SlidingWindowLimiter(1, 0.001)
    assert lim.max_attempts == 1
    assert lim.window == 0.001
