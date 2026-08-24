"""A uniformly random legal-move agent — the placeholder for AlphaZero."""

from __future__ import annotations

import random

from ..contract.move import Move
from .base import Agent, MoveProposal, MoveRequest


def _sort_key(move: Move) -> tuple[int, int, tuple[int, ...]]:
    return (move.source, move.destination, move.path)


class RandomAgent(Agent):
    """Picks uniformly at random among legal moves.

    Uses a private :class:`random.Random`, never the global RNG, so tests and
    replays are reproducible: the same seed with the same sequence of states
    yields the same sequence of proposals.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "RandomAgent"

    @property
    def seed(self) -> int | None:
        return self._seed

    def reset(self, seed: int | None = None) -> None:
        self._seed = seed if seed is not None else self._seed
        self._rng = random.Random(self._seed)

    def choose_move(self, request: MoveRequest) -> MoveProposal | None:
        candidates = sorted(request.legal_moves, key=_sort_key)
        if not candidates:
            return None

        rng = random.Random(request.seed) if request.seed is not None else self._rng

        # "Think Again" asks for something different; fall back to the full set
        # when the avoided moves are all we have.
        avoided = {(m.source, m.destination) for m in request.avoid}
        pool = [m for m in candidates if (m.source, m.destination) not in avoided] or candidates

        chosen = rng.choice(pool)
        return MoveProposal.from_move(
            chosen,
            metadata={
                "agent": self.name,
                "seed": self._seed if request.seed is None else request.seed,
                "legal_move_count": len(candidates),
            },
        )
