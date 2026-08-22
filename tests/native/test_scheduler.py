"""Gate C scheduler correctness: the properties the throughput numbers rest on.

None of this measures speed.  These are the invariants that have to hold before
any number the sweep produces means anything, plus the three tests section 5 of
``docs/native_selfplay_phase0.md`` requires of the batching design.
"""

from __future__ import annotations

import statistics
import threading

import pytest

from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.native import native_game, require_native
from diamond.game.state import build_players

WATCHDOG_SECONDS = 60.0
"""A hang must fail the test rather than wedge CI."""


class _Harness:
    def __init__(self) -> None:
        self.players = build_players(2)
        self.module = require_native()
        self.native = native_game(self.players)
        state = AlphaZeroGameAdapter(self.players).initial_state()
        self.opening = self.module.State(
            occupancy=list(state.occupancy),
            current_player=state.current_player_id,
            turn_number=state.turn_number,
            status=0,
            finish_order=[],
        )

    def run(self, **kwargs):
        """Run the scheduler under a watchdog, never on the calling thread."""
        config = self.module.SchedulerConfig(**kwargs)
        box: dict = {}

        def target() -> None:
            try:
                box["result"] = self.native.schedule(self.opening, config)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
                box["error"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(WATCHDOG_SECONDS)
        assert not thread.is_alive(), f"scheduler did not finish in {WATCHDOG_SECONDS}s: {kwargs}"
        if "error" in box:
            raise box["error"]
        return box["result"]


_HARNESS: _Harness | None = None


def _harness() -> _Harness:
    global _HARNESS
    if _HARNESS is None:
        _HARNESS = _Harness()
    return _HARNESS


def test_a_lane_trajectory_does_not_depend_on_the_thread_count() -> None:
    """The strongest correctness statement available about the scheduler.

    A lane's evaluations depend only on its own request and its own salt, so its
    game must be identical whether one worker ran it or eight -- regardless of
    how batches happened to form or in what order lanes were scheduled.  Any
    cross-lane contamination, torn state or misrouted result breaks this.
    """
    baseline = None
    for threads in (1, 2, 4, 8):
        result = _harness().run(
            games=32,
            threads=threads,
            max_batch=8,
            max_wait_us=500,
            simulations=16,
            trace_moves=True,
            stop_after_moves=8,
        )
        moves = [list(lane) for lane in result["lane_moves"]]
        assert all(len(lane) == 8 for lane in moves), threads
        if baseline is None:
            baseline = moves
        else:
            assert moves == baseline, f"trajectories diverged at threads={threads}"


def test_lanes_play_different_games() -> None:
    """Guards a bug that would flatter every number in this gate.

    The dummy evaluator is request-deterministic and temperature is 0, so lanes
    starting from the same opening only diverge if the per-lane salt actually
    reaches the value and priors.  A first implementation mixed the salt into
    bits that ``hash >> 11`` discards: lane values differed at the 11th decimal
    place, all lanes played the same moves, and N "independent" games were one
    game replayed N times -- which would have made batch formation look far
    better than it is.
    """
    result = _harness().run(
        games=32,
        threads=4,
        max_batch=8,
        max_wait_us=500,
        simulations=16,
        trace_moves=True,
        stop_after_moves=6,
    )
    trajectories = {tuple(lane) for lane in result["lane_moves"]}
    assert len(trajectories) == 32, f"only {len(trajectories)} distinct games from 32 lanes"


def test_a_single_lane_still_makes_progress() -> None:
    """Section 5's single-lane tail.

    One lane can never be joined by a second request, so it must ride the full
    ``max_wait_us`` and dispatch alone.  A batcher with any minimum batch size
    would hang here instead.
    """
    result = _harness().run(
        games=1, threads=1, max_batch=32, max_wait_us=2000, simulations=8, seconds=1.0
    )
    assert result["moves"] > 0
    assert result["batches"] > 0
    assert set(result["batch_sizes"]) == {1}, "a solitary lane must dispatch alone"


def test_batches_are_batch_scale_not_per_evaluation() -> None:
    """Section 5's callback-frequency test.

    With many lanes in flight the dispatch count must track
    ``evaluations / mean_batch``, not ``evaluations``.  If a worker blocked on
    its own evaluation instead of handing the lane over, batches would collapse
    towards the thread count and this would fail.
    """
    result = _harness().run(
        games=256, threads=2, max_batch=32, max_wait_us=2000, simulations=16, seconds=2.0
    )
    mean_batch = statistics.mean(result["batch_sizes"])
    assert mean_batch > 8, f"mean batch {mean_batch:.1f} with 256 lanes over 2 threads"
    assert result["batches"] == pytest.approx(
        result["evaluations"] / mean_batch, rel=0.01
    )
    # The decisive one: batch size must exceed the thread count, which is
    # impossible if a lane is pinned to the worker that started it.
    assert mean_batch > 2


def test_more_lanes_than_threads_keeps_workers_fed() -> None:
    """The architectural criterion, asserted rather than eyeballed.

    A worker must not sleep alongside a game that is waiting on the evaluator.
    With a slow evaluator and many more lanes than threads, there must still be
    runnable work queued at dispatch time.
    """
    result = _harness().run(
        games=128,
        threads=2,
        max_batch=32,
        max_wait_us=2000,
        simulations=16,
        eval_latency_ms=1.0,
        seconds=2.0,
    )
    assert result["batches"] > 0
    waiting = statistics.mean(result["waiting"])
    assert waiting > 2, f"only {waiting:.1f} lanes in flight; workers are blocking"


def test_scheduler_terminates_under_every_shape() -> None:
    """Watchdog coverage across the corners, including the degenerate ones."""
    for games, threads, max_batch in ((1, 1, 1), (1, 4, 32), (8, 1, 32), (64, 8, 1)):
        result = _harness().run(
            games=games,
            threads=threads,
            max_batch=max_batch,
            max_wait_us=200,
            simulations=8,
            seconds=0.5,
        )
        assert result["evaluations"] > 0, (games, threads, max_batch)
