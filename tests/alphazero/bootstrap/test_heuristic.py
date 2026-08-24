"""Geometry and prior-shape guarantees for canonical-target-distance-v1."""

from __future__ import annotations

import math

import pytest

from diamond.alphazero.bootstrap.heuristic import (
    CanonicalTargetDistancePrior,
    target_distance_table,
)
from diamond.alphazero.native import require_native
from diamond.alphazero.native.topology import camp_positions, neighbour_table
from diamond.contract.camps import PLAYABLE_HOLES, Camp


@pytest.fixture(scope="module")
def native():
    return require_native()


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


def test_progress_ranks_closer_destinations_higher(native, distance) -> None:
    """A source with a strictly closer and a strictly farther neighbour."""
    for source in range(PLAYABLE_HOLES):
        neighbours = [n for n in neighbour_table()[source] if n >= 0]
        closer = [n for n in neighbours if distance[n] < distance[source]]
        farther = [n for n in neighbours if distance[n] > distance[source]]
        if not (closer and farther):
            continue
        forward = native.encode_action(source, closer[0])
        backward = native.encode_action(source, farther[0])
        priors = CanonicalTargetDistancePrior().priors((forward, backward), distance)
        assert priors[forward] > priors[backward]
        return
    pytest.fail("no fixture position offered both a closer and a farther neighbour")


def test_backward_actions_keep_non_zero_prior(native, distance) -> None:
    for source in range(PLAYABLE_HOLES):
        neighbours = [n for n in neighbour_table()[source] if n >= 0]
        farther = [n for n in neighbours if distance[n] > distance[source]]
        if not farther:
            continue
        backward = native.encode_action(source, farther[0])
        closer = [n for n in neighbours if distance[n] < distance[source]]
        actions = (backward,) + (
            (native.encode_action(source, closer[0]),) if closer else ()
        )
        priors = CanonicalTargetDistancePrior().priors(actions, distance)
        assert priors[backward] > 0.0
        return
    pytest.fail("no fixture position offered a backward move")


def test_priors_cover_exactly_the_supplied_actions(native, distance) -> None:
    actions = (native.encode_action(0, 1), native.encode_action(5, 9), native.encode_action(20, 30))
    priors = CanonicalTargetDistancePrior().priors(actions, distance)
    assert tuple(sorted(priors)) == tuple(sorted(actions))


def test_priors_are_finite_positive_and_normalised(native, distance) -> None:
    actions = tuple(native.encode_action(source, source + 1) for source in range(0, 40, 3))
    priors = CanonicalTargetDistancePrior().priors(actions, distance)
    assert all(math.isfinite(p) and p > 0.0 for p in priors.values())
    assert math.isclose(sum(priors.values()), 1.0, rel_tol=1e-12)


def test_action_ordering_does_not_change_probabilities(native, distance) -> None:
    actions = (native.encode_action(0, 1), native.encode_action(30, 40), native.encode_action(60, 55))
    prior = CanonicalTargetDistancePrior()
    forward = prior.priors(actions, distance)
    reversed_ = prior.priors(tuple(reversed(actions)), distance)
    for action in actions:
        assert math.isclose(forward[action], reversed_[action], rel_tol=1e-12)


def test_empty_action_set_is_rejected(distance) -> None:
    with pytest.raises(ValueError):
        CanonicalTargetDistancePrior().priors((), distance)
