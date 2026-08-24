"""Min self-play runs on the native pool, and its value targets are per seat.

The pool was two-player end to end: a `SearchSession` that refuses a three-seat
match, and a sample builder that wrote Soo's scalar outcome. Both are the kind
of limit that produces no error for Min -- just no native path at all, so Min
trained on the Python engine while Soo did not.

The value target is the part worth pinning. Min's is a placement utility
ordered by `canonical_player_ids`, not by seat id, because that is the order the
value head is trained in. Ordering it by seat trains two of the three heads on
somebody else's result and nothing says so.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from diamond.alphazero.config import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
    MCTSConfig,
    NetworkConfig,
    SelfPlayConfig,
)
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.native.selfplay_pool import NativeSelfPlayPool
from diamond.alphazero.network.min import MinModel
from diamond.alphazero.orchestration.selfplay_workers import ModelKey, SelfPlayJob
from diamond.game.state import build_players

SIMULATIONS = 4
NETWORK = NetworkConfig(width=16, residual_blocks=1)


@pytest.fixture(scope="module")
def harness():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    players = build_players(3)
    model = MinModel(NETWORK, model_version="0.1.0")
    model.eval()
    return {
        "players": players,
        "game": AlphaZeroGameAdapter(players),
        "model": model,
        "compatibility": CheckpointCompatibilitySpec.min(
            model_version="0.1.0", network_config=NETWORK
        ),
        "model_key": ModelKey(
            model_name="Min",
            model_version="0.1.0",
            checkpoint_sha256=hashlib.sha256(b"min-test").hexdigest(),
        ),
    }


def _job(
    harness,
    index: int,
    *,
    max_moves: int = 24,
    prior: str = BOOTSTRAP_PRIOR_NONE,
) -> SelfPlayJob:
    return SelfPlayJob(
        run_seed=7,
        iteration=0,
        game_index=index,
        retry_id="attempt-0",
        model_key=harness["model_key"],
        compatibility=harness["compatibility"],
        players=harness["players"],
        initial_state=harness["game"].initial_state(),
        mcts_config=MCTSConfig(
            simulations=SIMULATIONS,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.0,
            seed=7,
        ),
        selfplay_config=SelfPlayConfig(
            max_moves=max_moves,
            temperature_moves=0,
            temperature=0.0,
            seed=7,
            bootstrap_prior=prior,
        ),
    )


def test_min_plays_through_the_native_pool(harness) -> None:
    pool = NativeSelfPlayPool(harness["model"], device="cpu", threads=2, max_batch=4)
    results = pool.run(tuple(_job(harness, index) for index in range(2)))

    assert len(results) == 2
    for result in results:
        assert result.move_count > 0
        # 24 moves is nowhere near a finished three-player game, so these abort
        # by design; what is under test is that the search ran at all.
        assert not result.completed
        assert result.samples == ()
    assert pool.metrics["moves"] > 0


def test_min_runs_on_the_vacancy_prior_too(harness) -> None:
    """value_only mode, which is what Min's shipped config asks for.

    The vacancy prior is computed from the acting seat's own pieces in
    canonical space, so it is defined for either seat count -- only the value
    width differs, and that is the whole of what had to change.
    """
    pool = NativeSelfPlayPool(harness["model"], device="cpu", threads=1, max_batch=4)
    results = pool.run(
        (_job(harness, 0, prior=CANONICAL_TARGET_VACANCY_DISTANCE_V2),)
    )
    assert results[0].move_count > 0
    assert pool.metrics["evaluations"] > 0


def test_a_min_value_target_is_the_placement_in_canonical_order() -> None:
    value_target = NativeSelfPlayPool._value_target

    # finish_order is by seat: 2 placed first, then 3, then 1.
    final_order = (2, 3, 1)
    assert value_target((2, 3, 1), final_order) == (1.0, 0.0, -1.0)
    # The same game seen from another seat's canonical rotation.
    assert value_target((3, 1, 2), final_order) == (0.0, -1.0, 1.0)
    assert value_target((1, 2, 3), final_order) == (-1.0, 1.0, 0.0)


def test_a_soo_value_target_is_still_the_scalar_outcome() -> None:
    value_target = NativeSelfPlayPool._value_target
    assert value_target((1, 2), (1, 2)) == (1.0,)
    assert value_target((2, 1), (1, 2)) == (-1.0,)


def test_min_samples_carry_a_three_component_target(harness) -> None:
    """The sample builder, on a finished game.

    Driven from a recorded episode rather than by playing until a random
    network happens to finish a three-player game: that takes thousands of
    moves and would make this test a coin flip on the cap.
    """
    pool = NativeSelfPlayPool(harness["model"], device="cpu")
    job = _job(harness, 0)
    encoded = harness["game"].encoder.encode(
        harness["game"].initial_state(), harness["players"]
    )
    record = {
        "move_count": 3,
        "completed": True,
        "finish_order": [2, 3, 1],
        "moves": [
            {
                "node_features": [list(row) for row in encoded.node_features],
                "canonical_player_ids": list(encoded.canonical_player_ids),
                "root_actions": [10, 20, 30],
                "visit_counts": [3, 1, 0],
            }
        ],
    }

    result = pool._episode_result(job, record)
    assert result.completed
    assert result.final_order == (2, 3, 1)
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert len(sample.canonical_player_ids) == 3
    # Min's replay semantics: a permutation of (+1, 0, -1), and TrainingSample
    # validates that on construction -- so reaching this line is half the test.
    assert sorted(sample.value_target) == [-1.0, 0.0, 1.0]
    placement = dict(zip((2, 3, 1), (1.0, 0.0, -1.0)))
    assert sample.value_target == tuple(
        placement[seat] for seat in sample.canonical_player_ids
    )
    # Zero-visit actions are dropped and the rest are sorted, as the Python
    # runner does, so replay digests stay comparable across backends.
    assert sample.sparse_policy == ((10, 0.75), (20, 0.25))
