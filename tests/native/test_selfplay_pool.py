"""Gate F: the native pool produces the episodes the Python backend would.

Gate D proved the native *search* plays the same game. This is the layer above:
that a whole episode -- samples, policy targets, value targets, provenance --
comes out identical to what ``SooSelfPlayRunner`` produces for the same job.

Determinism is the only setting where the two can be compared move for move, so
the parity tests run at ``epsilon = 0, temperature = 0``. What that cannot cover
is asserted separately: the stochastic path has its own gate (Gate E), and the
things that are *structural* rather than stochastic -- shapes, sorting, value
signs, abort semantics -- are checked in the stochastic configuration too.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from diamond.alphazero.config import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
    MCTSConfig,
    NetworkConfig,
    SelfPlayConfig,
)
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.protocol import ModelKey
from diamond.alphazero.native import require_native
from diamond.alphazero.native.selfplay_pool import NativeSelfPlayPool
from diamond.alphazero.orchestration.selfplay_workers import SelfPlayJob
from diamond.contract.state import GameState, GameStatus, build_players

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "runtime" / "runs" / "soo" / "cpu8h-soo-20260819" / "latest.pt"

SIMULATIONS = 16
"""Small on purpose: these tests compare whole games move by move against the
Python runner, which is ~400x slower per evaluation."""

FIXTURE = Path(__file__).parent / "fixtures" / "positions.jsonl"


ENDGAME_TAG = "packing-s1-m338"
"""The corpus position full-episode parity starts from.

Pinned by tag rather than chosen as "the deepest", because depth does not
predict what is wanted here.  Measured across the corpus with the step-80
checkpoint at 16 simulations, greedy: turn 383 ends in 1 move, turn 360 in 4,
turn 340 in 6 -- and turns 300 and 375 do not terminate at all, burning the full
2000-move cap in ~137 s each.  This one gives six moves in 0.35 s: enough
samples for the comparison to mean something, cheap enough for CI.
"""


def _endgame_state() -> GameState:
    """A near-terminal 2P position, for a parity comparison that terminates.

    Full-episode parity has to be compared in the deterministic regime -- it is
    the only one where the two backends are defined to agree move for move --
    and **a deterministic game from the opening does not terminate**.  Measured
    with this checkpoint at 4, 8 and 16 simulations: all three burn the full
    2000-move cap and produce zero samples.  A parity test starting from the
    opening therefore compares two empty episodes and asserts nothing, which is
    exactly what the first version of this test did.  See
    ``test_the_parity_comparison_is_not_vacuous``.
    """
    with FIXTURE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            # The corpus holds both 2P and 3P positions; Soo is the 2P game.
            if record["tag"] == ENDGAME_TAG and record["player_count"] == 2:
                return GameState(
                    occupancy=tuple(record["occupancy"]),
                    current_player_id=record["current_player_id"],
                    turn_number=record["turn_number"],
                    status=GameStatus(record["status"]),
                    finish_order=tuple(record["finish_order"]),
                )
    raise AssertionError(f"corpus position {ENDGAME_TAG!r} is missing")


def _model():
    from diamond.alphazero.network.soo import SooModel

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = SooModel(NetworkConfig())
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


class _Harness:
    def __init__(self) -> None:
        torch.set_num_threads(1)
        self.players = build_players(2)
        self.game = AlphaZeroGameAdapter(self.players)
        self.module = require_native()
        self.model = _model()
        self.compatibility = CheckpointCompatibilitySpec.soo(
            model_version="2.0.0", network_config=NetworkConfig()
        )
        self.model_key = ModelKey(
            model_name="Soo",
            model_version="2.0.0",
            checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        )

    def job(
        self,
        index: int,
        *,
        prior: str = CANONICAL_TARGET_VACANCY_DISTANCE_V2,
        epsilon: float = 0.0,
        temperature: float = 0.0,
        max_moves: int = 2000,
        endgame: bool = False,
    ) -> SelfPlayJob:
        return SelfPlayJob(
            run_seed=7,
            iteration=0,
            game_index=index,
            retry_id="attempt-0",
            model_key=self.model_key,
            compatibility=self.compatibility,
            players=self.players,
            initial_state=_endgame_state() if endgame else self.game.initial_state(),
            mcts_config=MCTSConfig(
                simulations=SIMULATIONS,
                c_puct=1.5,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=epsilon,
                seed=7,
            ),
            selfplay_config=SelfPlayConfig(
                max_moves=max_moves,
                temperature_moves=20 if temperature > 0 else 0,
                temperature=temperature,
                seed=7,
                bootstrap_prior=prior,
            ),
        )

    def pool(self, **kwargs) -> NativeSelfPlayPool:
        return NativeSelfPlayPool(
            self.model, device="cpu", threads=2, max_batch=8, **kwargs
        )


_HARNESS: _Harness | None = None


def _harness() -> _Harness:
    global _HARNESS
    if _HARNESS is None:
        _HARNESS = _Harness()
    return _HARNESS


def _python_episode(harness: _Harness, job: SelfPlayJob):
    """The oracle: the same job, wired exactly as ``run_selfplay_job`` wires it.

    Including the two ``replace`` calls that reseed both configs from the job --
    miss those and the two backends are seeded differently, which looks like a
    parity failure and is not one.
    """
    from dataclasses import replace

    from diamond.alphazero.bootstrap.evaluator import bootstrap_evaluator
    from diamond.alphazero.evaluator.torch import TorchEvaluator
    from diamond.alphazero.game_adapter import DiamondSearchAdapter
    from diamond.alphazero.selfplay.runner_2p import SooSelfPlayRunner

    game = DiamondSearchAdapter(
        AlphaZeroGameAdapter(job.players, initial=job.initial_state)
    )
    evaluator = bootstrap_evaluator(
        TorchEvaluator(harness.model, value_size=1, device="cpu"),
        job.selfplay_config.bootstrap_prior,
    )
    return SooSelfPlayRunner(
        game,
        evaluator,
        replace(job.mcts_config, seed=job.seed),
        replace(job.selfplay_config, seed=job.seed),
        job.compatibility,
    ).run()


# --------------------------------------------------------------------------
# Parity against the oracle
# --------------------------------------------------------------------------


def test_a_deterministic_episode_matches_the_python_runner() -> None:
    """The whole point: identical games, identical training data.

    Compared field by field rather than by digest, so a failure says *which*
    part diverged instead of only that something did.
    """
    harness = _harness()
    job = harness.job(0, endgame=True)
    expected = _python_episode(harness, job)
    (actual,) = harness.pool().run((job,))

    assert actual.completed == expected.completed
    assert actual.move_count == expected.move_count
    assert actual.final_order == expected.final_order
    assert len(actual.samples) == len(expected.samples)

    for index, (got, want) in enumerate(zip(actual.samples, expected.samples)):
        where = f"sample {index} of {len(expected.samples)}"
        assert got.canonical_player_ids == want.canonical_player_ids, where
        assert got.value_target == want.value_target, where
        assert [a for a, _ in got.sparse_policy] == [a for a, _ in want.sparse_policy], where
        for (_, p), (_, q) in zip(got.sparse_policy, want.sparse_policy):
            assert p == pytest.approx(q, rel=1e-12), where
        # approx does not descend into nested tuples; flatten so a mismatch in
        # any one of the 73 x 4 cells is actually compared.
        assert [x for row in got.node_features for x in row] == pytest.approx(
            [x for row in want.node_features for x in row], rel=1e-6
        ), where


def test_the_parity_comparison_is_not_vacuous() -> None:
    """Guard the guard.

    A deterministic game that never terminates yields zero samples on *both*
    sides, so the parity test above would pass while comparing nothing.  That is
    not hypothetical -- it is what the first version of this file did.  If the
    endgame position ever stops completing, this fails loudly instead.
    """
    harness = _harness()
    (episode,) = harness.pool().run((harness.job(0, endgame=True),))
    assert episode.completed, "the endgame parity position no longer terminates"
    assert len(episode.samples) >= 4, (
        f"only {len(episode.samples)} samples from {ENDGAME_TAG}: too few for the "
        "comparison to mean anything"
    )


def test_a_deterministic_game_from_the_opening_does_not_terminate() -> None:
    """Records why the parity test starts from an endgame, and why exploration
    is not optional for data generation.

    With the step-80 checkpoint and the heuristic prior, greedy self-play from
    the opening burns the full move cap: both sides shuffle rather than commit.
    Temperature and Dirichlet noise are what make games finish, so a native
    self-play pool run deterministically would produce no training data at all.
    """
    harness = _harness()
    job = harness.job(0, max_moves=60)
    (episode,) = harness.pool().run((job,))
    assert not episode.completed
    assert episode.move_count == 60


def test_identity_and_provenance_come_from_the_job() -> None:
    harness = _harness()
    job = harness.job(3, endgame=True)
    (episode,) = harness.pool().run((job,))
    assert episode.game_id == job.game_id
    assert episode.seed == job.seed
    assert episode.retry_id == job.retry_id
    assert episode.model_key == job.model_key
    assert episode.compatibility == job.compatibility
    assert episode.bootstrap_prior == CANONICAL_TARGET_VACANCY_DISTANCE_V2


def test_many_jobs_run_in_one_call_and_stay_addressed_to_their_own_job() -> None:
    """A batched scheduler that mixed up which lane wrote which episode would
    still produce plausible games; only the identity mapping catches it."""
    harness = _harness()
    jobs = tuple(harness.job(index, endgame=True) for index in range(6))
    episodes = harness.pool(lanes=3).run(jobs)
    assert len(episodes) == len(jobs)
    for job, episode in zip(jobs, episodes):
        assert episode.game_id == job.game_id
        assert episode.seed == job.seed

    # Same seed for every job here, so the games must be identical too -- which
    # doubles as a check that lane reuse does not leak state between games.
    move_counts = {episode.move_count for episode in episodes}
    assert len(move_counts) == 1, f"identically seeded jobs diverged: {move_counts}"


def test_lane_reuse_does_not_leak_between_games() -> None:
    """More jobs than lanes forces reuse; one lane plays several games."""
    harness = _harness()
    jobs = tuple(harness.job(index, endgame=True) for index in range(4))
    one_lane = harness.pool(lanes=1).run(jobs)
    four_lanes = harness.pool(lanes=4).run(jobs)
    for a, b in zip(one_lane, four_lanes):
        assert a.move_count == b.move_count
        assert a.final_order == b.final_order
        assert len(a.samples) == len(b.samples)


# --------------------------------------------------------------------------
# Structure, in the configuration production actually uses
# --------------------------------------------------------------------------


def test_samples_are_well_formed_under_exploration() -> None:
    """``TrainingSample.__post_init__`` enforces most of this; running it at all
    is the assertion.  What is checked here is what it does not: that the policy
    is the visit distribution and the value target is the game's own result."""
    harness = _harness()
    jobs = tuple(
        # Capped: Dirichlet noise from a near-terminal position can wander for
        # hundreds of moves, and this test is about sample shape, not length.
        harness.job(index, epsilon=0.25, temperature=1.0, endgame=True, max_moves=120)
        for index in range(4)
    )
    episodes = harness.pool(lanes=4).run(jobs)
    assert any(episode.completed for episode in episodes)

    for episode in episodes:
        if not episode.completed:
            continue
        winner = episode.final_order[0]
        for sample in episode.samples:
            assert len(sample.node_features) == 73
            assert sum(p for _, p in sample.sparse_policy) == pytest.approx(1.0, abs=1e-9)
            assert list(sample.sparse_policy) == sorted(sample.sparse_policy)
            expected = 1.0 if sample.canonical_player_ids[0] == winner else -1.0
            assert sample.value_target == (expected,)


def test_an_aborted_game_contributes_nothing() -> None:
    """Python returns zero samples past the move cap; inventing value targets
    for a game with no terminal outcome is the failure mode being excluded."""
    harness = _harness()
    job = harness.job(0, max_moves=4)
    (episode,) = harness.pool().run((job,))
    assert not episode.completed
    assert episode.samples == ()
    assert episode.final_order is None
    assert episode.aborted_reason == "max_game_moves_exceeded"
    assert episode.move_count == 4


# --------------------------------------------------------------------------
# The mapping that must not be allowed to drift
# --------------------------------------------------------------------------


def test_the_bootstrap_prior_selects_the_evaluator_mode() -> None:
    """``value_only`` computes the vacancy prior natively, so it *is*
    ``canonical-target-vacancy-distance-v2``.  A mislabelled run would train on
    data whose provenance is wrong and nothing downstream could tell."""
    harness = _harness()
    heuristic = harness.pool().run((harness.job(0, endgame=True),))[0]
    assert heuristic.bootstrap_prior == CANONICAL_TARGET_VACANCY_DISTANCE_V2

    neural_job = harness.job(1, prior=BOOTSTRAP_PRIOR_NONE, endgame=True)
    (neural,) = harness.pool().run((neural_job,))
    assert neural.bootstrap_prior == BOOTSTRAP_PRIOR_NONE
    # Different priors must actually produce different searches, or the
    # mapping above would be untested: two modes that agree prove nothing.
    assert (neural.move_count, [s.sparse_policy for s in neural.samples]) != (
        heuristic.move_count,
        [s.sparse_policy for s in heuristic.samples],
    ), "the two evaluator modes produced identical play"


def test_an_unknown_prior_is_refused_rather_than_defaulted() -> None:
    harness = _harness()
    job = harness.job(0)
    broken = dataclasses.replace(
        job,
        selfplay_config=dataclasses.replace(
            job.selfplay_config, bootstrap_prior="canonical-target-distance-v1"
        ),
    )
    with pytest.raises(ValueError, match="no evaluator mode"):
        harness.pool().run((broken,))


def test_jobs_that_disagree_on_configuration_are_refused() -> None:
    """One call, one configuration.  Silently using the first job's config for
    all of them would produce data that does not match its own provenance."""
    harness = _harness()
    jobs = (harness.job(0), harness.job(1, temperature=1.0))
    with pytest.raises(ValueError, match="one configuration"):
        harness.pool().run(jobs)


def test_duplicate_job_identities_are_refused() -> None:
    harness = _harness()
    job = harness.job(0)
    with pytest.raises(ValueError, match="distinct"):
        harness.pool().run((job, job))
