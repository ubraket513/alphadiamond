"""A per-game wall-clock budget, measured monotonically and injectable in tests."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from diamond.alphazero.deadline import MAX_GAME_TIME_EXCEEDED, Deadline


class FakeClock:
    """A monotonic clock the test drives explicitly; no sleeping, ever."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_no_budget_means_no_deadline() -> None:
    assert Deadline.start(None) is None


@pytest.mark.parametrize("budget", (0, -1.0))
def test_an_unusable_budget_is_rejected(budget: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        Deadline.start(budget)


def test_expiry_is_inclusive_at_the_budget_boundary() -> None:
    clock = FakeClock()
    deadline = Deadline.start(900.0, clock=clock)
    assert deadline is not None

    assert not deadline.expired
    clock.advance(899.9)
    assert not deadline.expired
    clock.advance(0.1)  # exactly 900.0 elapsed
    assert deadline.expired
    clock.advance(1.0)
    assert deadline.expired


def test_remaining_seconds_count_down_and_clamp_at_zero() -> None:
    clock = FakeClock()
    deadline = Deadline.start(900.0, clock=clock)
    assert deadline is not None

    assert deadline.remaining_s == pytest.approx(900.0)
    clock.advance(300.0)
    assert deadline.remaining_s == pytest.approx(600.0)
    clock.advance(1000.0)
    # Never negative: callers use this to size waits.
    assert deadline.remaining_s == 0.0


def test_elapsed_seconds_track_the_injected_clock() -> None:
    clock = FakeClock()
    deadline = Deadline.start(900.0, clock=clock)
    assert deadline is not None

    clock.advance(12.5)
    assert deadline.elapsed_s == pytest.approx(12.5)


def test_deadline_is_immutable() -> None:
    deadline = Deadline.start(900.0, clock=FakeClock())
    assert deadline is not None
    with pytest.raises(FrozenInstanceError):
        deadline.budget_s = 1.0  # type: ignore[misc]


def test_the_abort_reason_is_distinct_from_the_move_cap_reason() -> None:
    assert MAX_GAME_TIME_EXCEEDED == "max_game_time_exceeded"
    assert MAX_GAME_TIME_EXCEEDED != "max_game_moves_exceeded"


def test_the_default_clock_is_monotonic() -> None:
    from time import monotonic

    deadline = Deadline.start(900.0)
    assert deadline is not None
    assert deadline.clock is monotonic


def test_deadline_module_does_not_import_torch() -> None:
    """Spawn workers import this module; torch must stay out of that path."""
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment = os.environ | {"PYTHONPATH": source_root}
    code = """
import sys
import diamond.alphazero.deadline
assert 'torch' not in sys.modules, tuple(sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
