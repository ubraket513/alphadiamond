"""Gate D: the Python batch callback, its GIL discipline, and end-to-end parity.

Section 4 (the ABI) and section 5 (the threading design) of
``docs/native_selfplay_phase0.md``, including the three tests section 5
requires.

No throughput is asserted here.  Gate D's A/B throughput table needs the GPU
host; what these establish is that the boundary is correct, does not deadlock,
does not lose exceptions, and produces the *same game* as the Python backend.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path

import pytest

from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.native import native_game, require_native
from diamond.game.state import build_players

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "runtime" / "runs" / "soo" / "cpu8h-soo-20260819" / "latest.pt"
CHECKPOINT_SHA = "1634b901e213b065c107eea734b8c172c14babb1c2565352203961e86ea165af"
"""The immutable step-80 checkpoint every measurement in the design rests on."""

WATCHDOG_SECONDS = 120.0


class _Harness:
    def __init__(self) -> None:
        self.players = build_players(2)
        self.module = require_native()
        self.native = native_game(self.players)
        self.game = AlphaZeroGameAdapter(self.players)
        self.search = DiamondSearchAdapter(self.game)
        state = self.game.initial_state()
        self.opening = self.module.State(
            occupancy=list(state.occupancy),
            current_player=state.current_player_id,
            turn_number=state.turn_number,
            status=0,
            finish_order=[],
        )

    def run(self, callback, mode: str = "value_only", **kwargs):
        """Always on a worker thread under a watchdog: a GIL deadlock must fail."""
        config = self.module.SchedulerConfig(**kwargs)
        box: dict = {}

        def target() -> None:
            try:
                box["result"] = self.native.schedule_with_callback(
                    self.opening, config, callback, mode
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
                box["error"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(WATCHDOG_SECONDS)
        assert not thread.is_alive(), (
            f"callback run did not finish in {WATCHDOG_SECONDS}s -- likely a GIL deadlock"
        )
        if "error" in box:
            raise box["error"]
        return box["result"]


_HARNESS: _Harness | None = None


def _harness() -> _Harness:
    global _HARNESS
    if _HARNESS is None:
        _HARNESS = _Harness()
    return _HARNESS


def _constant(features):
    return np.zeros(features.shape[0], dtype=np.float32)


# --------------------------------------------------------------------------
# Section 5's required tests
# --------------------------------------------------------------------------


def test_a_callback_that_touches_python_does_not_deadlock() -> None:
    """Section 5's deadlock guard.

    The entry point releases the GIL for the whole run and the evaluator thread
    takes it back only around the callback.  If the entry point kept the GIL,
    the evaluator thread would block on it forever and this would hang -- so
    the callback allocates, imports and collects to be sure it is genuinely
    exercising the interpreter, and a watchdog fails rather than wedging CI.
    """
    seen: list[dict] = []

    def busy_callback(features):
        import json  # a real import inside the callback

        payload = {"rows": int(features.shape[0]), "sum": float(features.sum())}
        seen.append(json.loads(json.dumps(payload)))
        if len(seen) % 32 == 0:
            gc.collect()
        return np.full(features.shape[0], 0.5, dtype=np.float32)

    result = _harness().run(
        busy_callback, games=32, threads=4, max_batch=8, max_wait_us=500,
        simulations=8, seconds=1.5,
    )
    assert result["evaluations"] > 0
    assert len(seen) == result["batches"]


def test_a_single_lane_still_makes_progress_through_the_callback() -> None:
    """Section 5's single-lane tail, across the Python boundary."""
    result = _harness().run(
        _constant, games=1, threads=1, max_batch=32, max_wait_us=2000,
        simulations=8, seconds=1.0,
    )
    assert result["moves"] > 0
    assert set(result["batch_sizes"]) == {1}


def test_python_crossings_are_batch_scale() -> None:
    """Section 5's callback-frequency test.

    With N lanes the crossing count must be close to
    ``evaluations / mean_batch``, not to ``evaluations``.  This is the whole
    economic case for the batcher: the GIL is taken once per batch.
    """
    calls = {"n": 0, "rows": 0}

    def counting(features):
        calls["n"] += 1
        calls["rows"] += features.shape[0]
        return _constant(features)

    result = _harness().run(
        counting, games=128, threads=2, max_batch=32, max_wait_us=2000,
        simulations=16, seconds=2.0,
    )
    assert calls["rows"] == result["evaluations"]
    assert calls["n"] == result["batches"]
    evals_per_crossing = calls["rows"] / calls["n"]
    assert evals_per_crossing > 8, f"only {evals_per_crossing:.1f} evaluations per GIL acquisition"


# --------------------------------------------------------------------------
# ABI contract
# --------------------------------------------------------------------------


def test_features_arrive_as_float32_batch_by_board_by_channels() -> None:
    shapes: set = set()
    dtypes: set = set()

    def inspect(features):
        shapes.add(features.shape[1:])
        dtypes.add(features.dtype)
        # Features are occupancy channels plus finished flags: 0.0 or 1.0 only.
        assert set(np.unique(features)).issubset({0.0, 1.0})
        return _constant(features)

    _harness().run(inspect, games=8, threads=2, max_batch=4, max_wait_us=500,
                   simulations=8, seconds=1.0)
    assert shapes == {(73, 4)}, shapes
    assert dtypes == {np.dtype(np.float32)}, dtypes


def test_a_raising_callback_propagates_instead_of_terminating() -> None:
    """An exception escaping a std::thread would abort the process.

    It must instead tear the run down and surface on the calling thread, with
    the original Python exception type intact.
    """

    class Boom(Exception):
        pass

    def raising(features):
        raise Boom("callback failed on purpose")

    with pytest.raises(Boom, match="on purpose"):
        _harness().run(raising, games=16, threads=2, max_batch=8, max_wait_us=500,
                       simulations=8, seconds=1.0)

    # And the scheduler is still usable afterwards.
    result = _harness().run(_constant, games=8, threads=2, max_batch=4,
                            max_wait_us=500, simulations=8, seconds=0.5)
    assert result["evaluations"] > 0


def test_a_callback_returning_the_wrong_shape_is_rejected() -> None:
    def too_few(features):
        return np.zeros(max(1, features.shape[0] - 1), dtype=np.float32)

    with pytest.raises(Exception, match="wrong number of values"):
        _harness().run(too_few, games=8, threads=1, max_batch=4, max_wait_us=500,
                       simulations=8, seconds=1.0)


# --------------------------------------------------------------------------
# End to end, against the immutable checkpoint
# --------------------------------------------------------------------------


def _model():
    torch = pytest.importorskip("torch")
    if not CHECKPOINT.is_file():  # pragma: no cover - checkpoint ships with the repo
        pytest.skip(f"missing checkpoint: {CHECKPOINT}")
    from diamond.alphazero.config import NetworkConfig
    from diamond.alphazero.network.soo import SooModel

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = SooModel(NetworkConfig())
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def test_the_checkpoint_is_the_one_every_measurement_rests_on() -> None:
    import hashlib

    if not CHECKPOINT.is_file():  # pragma: no cover
        pytest.skip(f"missing checkpoint: {CHECKPOINT}")
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert digest == CHECKPOINT_SHA, "checkpoint changed; parity below is no longer comparable"


def test_value_only_is_bit_identical_to_the_full_forward() -> None:
    """Justifies skipping the policy head rather than computing and discarding it.

    If these differed at all, ValueOnly would be a different evaluator and not
    merely a cheaper one, and no parity claim below would hold.
    """
    torch = pytest.importorskip("torch")
    model = _model()
    features = torch.randn(7, 73, 4)
    with torch.inference_mode():
        _, full = model(features)
        nodes = model.trunk(features)
        trunk_only = model.value_head(nodes.mean(dim=1))
    assert bool((full == trunk_only).all()), "trunk-only value differs from the full forward"


@pytest.mark.parametrize(("simulations", "moves"), [(32, 20), (64, 12)])
def test_native_plays_the_same_game_as_the_python_backend(simulations: int, moves: int) -> None:
    """Gate D's correctness claim, on the immutable checkpoint.

    The native backend computes the vacancy prior itself and takes only values
    across the boundary; the Python backend runs ``VacancyPriorEvaluator`` over
    ``TorchEvaluator``.  Same weights, same simulations, epsilon and temperature
    at 0 -- so the two must select the same move every time.

    A single lane is used deliberately: both sides then evaluate at batch 1.
    Batch size perturbs the model's own output by ~3e-8 (measured), which is far
    larger than the <=1e-12 the native and Python priors differ by, so comparing
    across different batch shapes would be testing the model's numerics rather
    than the backend.
    """
    pytest.importorskip("torch")
    from diamond.alphazero.bootstrap.evaluator import VacancyPriorEvaluator
    from diamond.alphazero.config import MCTSConfig
    from diamond.alphazero.evaluator.torch import TorchEvaluator
    from diamond.alphazero.mcts.search_2p import MCTS2P
    from diamond.alphazero.native.backend import value_only_callback

    harness = _harness()
    model = _model()

    evaluator = VacancyPriorEvaluator(TorchEvaluator(model, value_size=1, device="cpu"))
    state = harness.game.initial_state()
    expected = []
    for _ in range(moves):
        result = MCTS2P(
            harness.search,
            evaluator,
            MCTSConfig(simulations=simulations, dirichlet_epsilon=0.0),
        ).run(state, temperature=0.0)
        expected.append(result.selected_action)
        state = harness.search.apply_action(state, result.selected_action)

    actual = harness.run(
        value_only_callback(model, device="cpu"),
        games=1,
        threads=1,
        max_batch=1,
        max_wait_us=200,
        simulations=simulations,
        trace_moves=True,
        stop_after_moves=moves,
    )
    assert list(actual["lane_moves"][0]) == expected


def test_policy_value_mode_returns_usable_priors() -> None:
    """The reference path stays compiled and exercised, per section 4."""
    pytest.importorskip("torch")
    from diamond.alphazero.native.backend import policy_value_callback

    model = _model()
    seen = {"rows": 0}
    inner = policy_value_callback(model, device="cpu")

    def checked(features, actions, offsets):
        assert offsets[0] == 0
        assert offsets[-1] == actions.shape[0]
        assert offsets.shape[0] == features.shape[0] + 1
        seen["rows"] += features.shape[0]
        priors, values = inner(features, actions, offsets)
        assert priors.shape[0] == actions.shape[0]
        # Each ragged row must be a distribution.
        for index in range(features.shape[0]):
            row = priors[offsets[index] : offsets[index + 1]]
            assert abs(float(row.sum()) - 1.0) < 1e-4
        return priors, values

    result = _harness().run(
        checked, mode="policy_value", games=8, threads=2, max_batch=4,
        max_wait_us=500, simulations=8, seconds=1.5,
    )
    assert seen["rows"] == result["evaluations"]
