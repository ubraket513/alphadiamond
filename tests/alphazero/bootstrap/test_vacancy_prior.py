"""canonical-target-vacancy-distance-v2 and the v1 stall it exists to fix."""

from __future__ import annotations

import math

import pytest

from diamond.alphazero.action_codec import ActionCodec, ActionSpaceSpec
from diamond.alphazero.bootstrap.evaluator import (
    BootstrapPriorEvaluator,
    VacancyPriorEvaluator,
    bootstrap_evaluator,
)
from diamond.alphazero.bootstrap.heuristic import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_DISTANCE_V1,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
    CanonicalTargetVacancyDistancePrior,
    pairwise_distance_table,
)
from diamond.alphazero.evaluator.dummy import DummyEvaluator
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.contract.board import Board, Camp
from diamond.contract.state import build_players


def best_of(priors):
    return max(priors.items(), key=lambda item: item[1])[0]


@pytest.fixture(scope="module")
def board() -> Board:
    return Board()


@pytest.fixture(scope="module")
def codec() -> ActionCodec:
    return ActionCodec(ActionSpaceSpec.diamond73())


@pytest.fixture(scope="module")
def pairwise(board: Board):
    return pairwise_distance_table(board)


@pytest.fixture(scope="module")
def target(board: Board) -> frozenset:
    return frozenset(board.camp_positions(Camp.Z_NEG))


@pytest.fixture(scope="module")
def stall_state():
    """The reproducible position where v1 greedy play freezes.

    Reached by following the v1 argmax from the opening: eight pieces are home,
    the last two sit one step out, and every v1 progress score is zero.
    """
    game = DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2)))
    wrapped = BootstrapPriorEvaluator(DummyEvaluator(0.0))
    state = game.initial_state()
    for _ in range(160):
        request = game.evaluation_request(state)
        state = game.apply_action(state, best_of(wrapped.evaluate((request,))[0].priors))
    return game, state


def features_with_self_at(positions, size: int = 73, channels: int = 2):
    """Node features whose channel 0 marks the acting player holes."""
    return tuple(
        (1.0 if index in positions else 0.0,) + (0.0,) * (channels - 1)
        for index in range(size)
    )


# -- the regression the whole of v2 exists for -------------------------------


def test_v1_cannot_distinguish_moves_in_the_stall_fixture(stall_state) -> None:
    """Recorded limitation: the best v1 moves here are a flat tie."""
    game, state = stall_state
    request = game.evaluation_request(state)
    priors = BootstrapPriorEvaluator(DummyEvaluator(0.0)).evaluate((request,))[0].priors
    best = max(priors.values())
    tied = [a for a, v in priors.items() if math.isclose(v, best, rel_tol=1e-12)]
    assert len(tied) > 1, "v1 was expected to tie; the fixture no longer reproduces"


def test_v1_stall_fixture_has_unfilled_target_holes(stall_state, target) -> None:
    """The tie is not a finished game: target holes remain to be filled."""
    game, state = stall_state
    request = game.evaluation_request(state)
    mine = {i for i, row in enumerate(request.node_features) if row[0] == 1.0}
    assert target - mine, "fixture should still have unfilled target holes"
    assert not game.is_terminal(state)


def test_v2_prefers_filling_a_target_hole_in_the_stall_fixture(
    stall_state, codec, target, pairwise
) -> None:
    game, state = stall_state
    request = game.evaluation_request(state)
    priors = CanonicalTargetVacancyDistancePrior().priors(
        request.legal_action_ids, codec, target, pairwise, request.node_features
    )
    top = best_of(priors)
    _, destination = codec.decode(top)
    assert destination in target
    ties = [v for v in priors.values() if math.isclose(v, priors[top], rel_tol=1e-12)]
    assert len(ties) == 1, "v2 should express a strict preference, not a tie"


def test_v2_breaks_the_v1_tie_into_distinct_scores(
    stall_state, codec, target, pairwise
) -> None:
    game, state = stall_state
    request = game.evaluation_request(state)
    priors = CanonicalTargetVacancyDistancePrior().priors(
        request.legal_action_ids, codec, target, pairwise, request.node_features
    )
    assert len({round(v, 9) for v in priors.values()}) > 1


# -- U is "not yet mine", never "physically empty" ---------------------------


def test_vacancy_includes_target_holes_held_by_an_opponent(target, pairwise) -> None:
    """An opponent-held target hole is still a slot this player must fill.

    Channel 0 is self-occupancy only, so a hole occupied by an opponent is not in
    ``occupied`` and therefore remains in ``U``.
    """
    prior = CanonicalTargetVacancyDistancePrior()
    ordered = sorted(target)
    mine = frozenset(ordered[:-1])
    contested = ordered[-1]
    outside = max(range(73), key=lambda p: pairwise[contested][p])
    potential = prior.potential(mine | {outside}, target, pairwise)
    assert potential == pairwise[outside][contested]
    assert potential > 0.0


def test_potential_is_zero_once_every_target_hole_is_self_occupied(
    target, pairwise
) -> None:
    prior = CanonicalTargetVacancyDistancePrior()
    assert prior.potential(frozenset(target), target, pairwise) == 0.0


def test_potential_ignores_pieces_already_inside_the_target_camp(
    target, pairwise
) -> None:
    prior = CanonicalTargetVacancyDistancePrior()
    partial = frozenset(sorted(target)[:5])
    assert prior.potential(partial, target, pairwise) == 0.0


# -- invariants the design requires to hold for every prior ------------------


def test_all_legal_actions_keep_strictly_positive_priors(codec, target, pairwise) -> None:
    features = features_with_self_at({40, 41, 42})
    actions = (codec.encode(40, 33), codec.encode(41, 48), codec.encode(42, 43))
    priors = CanonicalTargetVacancyDistancePrior().priors(
        actions, codec, target, pairwise, features
    )
    assert set(priors) == set(actions)
    assert all(v > 0.0 for v in priors.values())
    assert math.isclose(sum(priors.values()), 1.0, rel_tol=1e-12)


def test_action_ordering_does_not_change_probabilities(codec, target, pairwise) -> None:
    features = features_with_self_at({40, 41, 42})
    actions = (codec.encode(40, 33), codec.encode(41, 48), codec.encode(42, 43))
    prior = CanonicalTargetVacancyDistancePrior()
    forward = prior.priors(actions, codec, target, pairwise, features)
    backward = prior.priors(tuple(reversed(actions)), codec, target, pairwise, features)
    for action in actions:
        assert math.isclose(forward[action], backward[action], rel_tol=1e-12)


def test_empty_action_set_is_rejected(codec, target, pairwise) -> None:
    with pytest.raises(ValueError):
        CanonicalTargetVacancyDistancePrior().priors(
            (), codec, target, pairwise, features_with_self_at({0})
        )


def test_wrapper_preserves_values_and_replaces_priors_only() -> None:
    game = DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2)))
    request = game.evaluation_request(game.initial_state())
    result = VacancyPriorEvaluator(DummyEvaluator(-0.5)).evaluate((request,))
    assert result[0].value == -0.5
    assert tuple(sorted(result[0].priors)) == tuple(sorted(request.legal_action_ids))


def test_pairwise_table_is_symmetric_and_zero_on_the_diagonal(pairwise) -> None:
    assert all(pairwise[i][i] == 0 for i in range(len(pairwise)))
    assert all(
        pairwise[i][j] == pairwise[j][i]
        for i in range(0, len(pairwise), 7)
        for j in range(0, len(pairwise), 5)
    )


# -- factory dispatch --------------------------------------------------------


def test_factory_selects_the_configured_prior() -> None:
    base = DummyEvaluator(0.0)
    assert bootstrap_evaluator(base, BOOTSTRAP_PRIOR_NONE) is base
    assert isinstance(
        bootstrap_evaluator(base, CANONICAL_TARGET_DISTANCE_V1), BootstrapPriorEvaluator
    )
    assert isinstance(
        bootstrap_evaluator(base, CANONICAL_TARGET_VACANCY_DISTANCE_V2),
        VacancyPriorEvaluator,
    )


def test_identities_are_reported_for_provenance() -> None:
    base = DummyEvaluator(0.0)
    assert BootstrapPriorEvaluator(base).identity == CANONICAL_TARGET_DISTANCE_V1
    assert VacancyPriorEvaluator(base).identity == CANONICAL_TARGET_VACANCY_DISTANCE_V2
