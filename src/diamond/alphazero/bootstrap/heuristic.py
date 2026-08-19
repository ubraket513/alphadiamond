"""Board-topology prior that points cold-start self-play at the target camp.

Temporary bootstrap scaffolding.  It exists only so an untrained network can
reach real terminal Diamond states often enough to fill replay; it is switched
off once a learned policy can finish games on its own.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp

from ..action_codec import ActionCodec
from ...game.board import Board, Camp

BOOTSTRAP_PRIOR_NONE = "none"
CANONICAL_TARGET_DISTANCE_V1 = "canonical-target-distance-v1"

TEMPERATURE_V1 = 1.0
"""Part of the ``canonical-target-distance-v1`` identity, not a tuning knob."""


def target_distance_table(board: Board, target: Camp = Camp.Z_NEG) -> tuple[int, ...]:
    """Shortest neighbour-graph distance from every hole to ``target``.

    Canonicalisation always rotates the acting player's home camp to ``z+``, so
    that player's target camp is canonical ``z-``.  The table therefore depends
    only on board topology and is identical for every player and every state.
    """
    sources = board.camp_positions(target)
    if not sources:
        raise ValueError(f"camp {target} has no positions")

    distance: list[int | None] = [None] * len(board)
    queue: deque[int] = deque()
    for position in sources:
        distance[position] = 0
        queue.append(position)

    while queue:
        current = queue.popleft()
        for neighbour in board.neighbours(current):
            if neighbour is not None and distance[neighbour] is None:
                distance[neighbour] = distance[current] + 1  # type: ignore[operator]
                queue.append(neighbour)

    if any(value is None for value in distance):
        raise ValueError("board graph is disconnected; distance table is undefined")
    return tuple(value for value in distance)  # type: ignore[misc]


@dataclass(frozen=True, slots=True)
class CanonicalTargetDistancePrior:
    """Softmax over per-action progress toward the canonical target camp.

    Every legal action keeps a strictly positive probability: backward, lateral
    and tactically unusual moves stay reachable, they are merely less likely.
    """

    identity: str = CANONICAL_TARGET_DISTANCE_V1
    temperature: float = TEMPERATURE_V1

    @classmethod
    def for_diamond73(cls) -> "CanonicalTargetDistancePrior":
        return cls()

    def priors(
        self,
        legal_action_ids: tuple[int, ...],
        codec: ActionCodec,
        distance: tuple[int, ...],
    ) -> dict[int, float]:
        if not legal_action_ids:
            raise ValueError("evaluation requires at least one legal action")

        progress: list[float] = []
        for action_id in legal_action_ids:
            source, destination = codec.decode(action_id)
            progress.append(float(distance[source] - distance[destination]))

        # Shift by the max before exponentiating: identical probabilities, no
        # overflow, and the result is independent of action ordering.
        highest = max(progress)
        weights = [exp((value - highest) / self.temperature) for value in progress]
        total = sum(weights)
        return {
            action_id: weight / total
            for action_id, weight in zip(legal_action_ids, weights)
        }





CANONICAL_TARGET_VACANCY_DISTANCE_V2 = "canonical-target-vacancy-distance-v2"


def _self_occupancy(node_features: tuple[tuple[float, ...], ...]) -> frozenset[int]:
    """Canonical holes held by the acting player.

    Channel 0 of every row is the acting player's occupancy: the encoder rotates
    occupancy channels to ``self, next[, previous]`` before features are built.
    """
    return frozenset(
        position
        for position, row in enumerate(node_features)
        if row and row[0] == 1.0
    )


@dataclass(frozen=True, slots=True)
class CanonicalTargetVacancyDistancePrior:
    """Potential-based prior that can still see the target-camp endgame.

    ``canonical-target-distance-v1`` measures each move against a fixed table and
    is therefore blind to which target holes are already filled.  Once the last
    stragglers sit one step out, every legal move scores zero progress and the
    player shuffles forever.

    v2 scores the whole position instead of the single moved piece::

        U   = target holes the acting player does NOT yet occupy
        O   = acting player's pieces outside the target camp
        Phi = sum over O of the distance to the nearest hole in U

    An action is scored by ``Phi(before) - Phi(after)``, updating only the acting
    player's own occupancy.  A camp-internal move that opens a vacancy the
    stragglers can actually reach now scores above an idle shuffle.

    ``U`` deliberately means "not yet mine", not "physically empty": a target hole
    held by an opponent is still a slot this player must eventually fill.  Whether
    a move is playable right now stays entirely the authoritative rules' business.
    """

    identity: str = CANONICAL_TARGET_VACANCY_DISTANCE_V2
    temperature: float = TEMPERATURE_V1

    def potential(
        self,
        occupied: frozenset[int],
        target: frozenset[int],
        pairwise: tuple[tuple[int, ...], ...],
    ) -> float:
        vacancies = target - occupied
        if not vacancies:
            return 0.0
        return float(
            sum(
                min(pairwise[piece][vacancy] for vacancy in vacancies)
                for piece in occupied - target
            )
        )

    def priors(
        self,
        legal_action_ids: tuple[int, ...],
        codec: ActionCodec,
        target: frozenset[int],
        pairwise: tuple[tuple[int, ...], ...],
        node_features: tuple[tuple[float, ...], ...],
    ) -> dict[int, float]:
        if not legal_action_ids:
            raise ValueError("evaluation requires at least one legal action")

        occupied = _self_occupancy(node_features)
        before = self.potential(occupied, target, pairwise)

        scores: list[float] = []
        for action_id in legal_action_ids:
            source, destination = codec.decode(action_id)
            moved = (occupied - {source}) | {destination}
            scores.append(before - self.potential(moved, target, pairwise))

        highest = max(scores)
        weights = [exp((score - highest) / self.temperature) for score in scores]
        total = sum(weights)
        return {
            action_id: weight / total
            for action_id, weight in zip(legal_action_ids, weights)
        }


def pairwise_distance_table(board: Board) -> tuple[tuple[int, ...], ...]:
    """All-pairs neighbour-graph distances; 73x73 BFS, computed once."""
    size = len(board)
    rows: list[tuple[int, ...]] = []
    for origin in range(size):
        distance: list[int | None] = [None] * size
        distance[origin] = 0
        queue: deque[int] = deque([origin])
        while queue:
            current = queue.popleft()
            for neighbour in board.neighbours(current):
                if neighbour is not None and distance[neighbour] is None:
                    distance[neighbour] = distance[current] + 1  # type: ignore[operator]
                    queue.append(neighbour)
        if any(value is None for value in distance):
            raise ValueError("board graph is disconnected; distances are undefined")
        rows.append(tuple(value for value in distance))  # type: ignore[misc]
    return tuple(rows)


__all__ = [
    "BOOTSTRAP_PRIOR_NONE",
    "CANONICAL_TARGET_DISTANCE_V1",
    "CANONICAL_TARGET_VACANCY_DISTANCE_V2",
    "TEMPERATURE_V1",
    "CanonicalTargetDistancePrior",
    "CanonicalTargetVacancyDistancePrior",
    "pairwise_distance_table",
    "target_distance_table",
]
