"""Geometry and prior-shape guarantees for canonical-target-distance-v1."""

from __future__ import annotations

import math

import pytest

from diamond.alphazero.action_codec import ActionCodec, ActionSpaceSpec
from diamond.alphazero.bootstrap.heuristic import (
    CanonicalTargetDistancePrior,
    target_distance_table,
)
from diamond.game.board import Board, Camp


@pytest.fixture(scope="module")
def board() -> Board:
    return Board()


@pytest.fixture(scope="module")
def codec() -> ActionCodec:
    return ActionCodec(ActionSpaceSpec.diamond73())


@pytest.fixture(scope="module")
def distance(board: Board) -> tuple[int, ...]:
    return target_distance_table(board)


def test_target_camp_positions_have_distance_zero(board: Board, distance) -> None:
    for position in board.camp_positions(Camp.Z_NEG):
        assert distance[position] == 0


def test_every_position_is_finite_and_non_negative(board: Board, distance) -> None:
    assert len(distance) == len(board)
    assert all(isinstance(value, int) and value >= 0 for value in distance)


def test_table_is_deterministic(board: Board) -> None:
    assert target_distance_table(board) == target_distance_table(Board())


def test_neighbour_distances_differ_by_at_most_one(board: Board, distance) -> None:
    for source, destination in board.edges:
        assert abs(distance[source] - distance[destination]) <= 1


def test_progress_ranks_closer_destinations_higher(board: Board, codec, distance) -> None:
    """A source with a strictly closer and a strictly farther neighbour."""
    for source in range(len(board)):
        neighbours = [n for n in board.neighbours(source) if n is not None]
        closer = [n for n in neighbours if distance[n] < distance[source]]
        farther = [n for n in neighbours if distance[n] > distance[source]]
        if not (closer and farther):
            continue
        forward = codec.encode(source, closer[0])
        backward = codec.encode(source, farther[0])
        priors = CanonicalTargetDistancePrior().priors(
            (forward, backward), codec, distance
        )
        assert priors[forward] > priors[backward]
        return
    pytest.fail("no fixture position offered both a closer and a farther neighbour")


def test_backward_actions_keep_non_zero_prior(board: Board, codec, distance) -> None:
    for source in range(len(board)):
        farther = [
            n
            for n in board.neighbours(source)
            if n is not None and distance[n] > distance[source]
        ]
        if not farther:
            continue
        backward = codec.encode(source, farther[0])
        closer = [
            n
            for n in board.neighbours(source)
            if n is not None and distance[n] < distance[source]
        ]
        actions = (backward,) + (
            (codec.encode(source, closer[0]),) if closer else ()
        )
        priors = CanonicalTargetDistancePrior().priors(actions, codec, distance)
        assert priors[backward] > 0.0
        return
    pytest.fail("no fixture position offered a backward move")


def test_priors_cover_exactly_the_supplied_actions(codec, distance) -> None:
    actions = (codec.encode(0, 1), codec.encode(5, 9), codec.encode(20, 30))
    priors = CanonicalTargetDistancePrior().priors(actions, codec, distance)
    assert tuple(sorted(priors)) == tuple(sorted(actions))


def test_priors_are_finite_positive_and_normalised(codec, distance) -> None:
    actions = tuple(codec.encode(source, source + 1) for source in range(0, 40, 3))
    priors = CanonicalTargetDistancePrior().priors(actions, codec, distance)
    assert all(math.isfinite(p) and p > 0.0 for p in priors.values())
    assert math.isclose(sum(priors.values()), 1.0, rel_tol=1e-12)


def test_action_ordering_does_not_change_probabilities(codec, distance) -> None:
    actions = (codec.encode(0, 1), codec.encode(30, 40), codec.encode(60, 55))
    prior = CanonicalTargetDistancePrior()
    forward = prior.priors(actions, codec, distance)
    reversed_ = prior.priors(tuple(reversed(actions)), codec, distance)
    for action in actions:
        assert math.isclose(forward[action], reversed_[action], rel_tol=1e-12)


def test_empty_action_set_is_rejected(codec, distance) -> None:
    with pytest.raises(ValueError):
        CanonicalTargetDistancePrior().priors((), codec, distance)
