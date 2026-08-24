"""Geometry and prior-shape guarantees for canonical-target-distance-v1."""

from __future__ import annotations

import math

import pytest

from diamond.alphazero.action_codec import ActionCodec, ActionSpaceSpec
from diamond.alphazero.bootstrap.heuristic import (
    CanonicalTargetDistancePrior,
    target_distance_table,
)
from diamond.alphazero.native.topology import camp_positions, neighbour_table
from diamond.contract.camps import PLAYABLE_HOLES, Camp


@pytest.fixture(scope="module")
def codec() -> ActionCodec:
    return ActionCodec(ActionSpaceSpec.diamond73())


@pytest.fixture(scope="module")
def distance() -> tuple[int, ...]:
    return target_distance_table()


def test_target_camp_positions_have_distance_zero(distance) -> None:
    for position in camp_positions(Camp.Z_NEG):
        assert distance[position] == 0


def test_every_position_is_finite_and_non_negative(distance) -> None:
    assert len(distance) == PLAYABLE_HOLES
    assert all(isinstance(value, int) and value >= 0 for value in distance)


def test_table_is_deterministic() -> None:
    assert target_distance_table() == target_distance_table()


def test_neighbour_distances_differ_by_at_most_one(distance) -> None:
    for source, row in enumerate(neighbour_table()):
        for destination in row:
            if destination >= 0:
                assert abs(distance[source] - distance[destination]) <= 1


def test_progress_ranks_closer_destinations_higher(codec, distance) -> None:
    """A source with a strictly closer and a strictly farther neighbour."""
    for source in range(PLAYABLE_HOLES):
        neighbours = [n for n in neighbour_table()[source] if n >= 0]
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


def test_backward_actions_keep_non_zero_prior(codec, distance) -> None:
    for source in range(PLAYABLE_HOLES):
        neighbours = [n for n in neighbour_table()[source] if n >= 0]
        farther = [n for n in neighbours if distance[n] > distance[source]]
        if not farther:
            continue
        backward = codec.encode(source, farther[0])
        closer = [n for n in neighbours if distance[n] < distance[source]]
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
