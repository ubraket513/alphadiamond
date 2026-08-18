"""Framework-independent PUCT helpers."""

from __future__ import annotations

import math
import random


def exploration_bonus(prior: float, parent_visits: int, edge_visits: int, c_puct: float) -> float:
    return c_puct * prior * math.sqrt(max(1, parent_visits)) / (1 + edge_visits)


def add_dirichlet_noise(
    priors: dict[int, float], *, alpha: float, epsilon: float, rng: random.Random
) -> dict[int, float]:
    if epsilon <= 0:
        return dict(priors)
    if alpha <= 0 or not 0 <= epsilon <= 1:
        raise ValueError("Dirichlet alpha must be positive and epsilon in [0, 1]")
    samples = [rng.gammavariate(alpha, 1.0) for _ in priors]
    total = sum(samples)
    if total <= 0:
        raise ValueError("failed to sample root Dirichlet noise")
    noise = [sample / total for sample in samples]
    return {
        action: (1.0 - epsilon) * prior + epsilon * component
        for (action, prior), component in zip(priors.items(), noise)
    }


def select_from_visits(
    visits: dict[int, int], *, temperature: float, rng: random.Random
) -> int:
    if not visits:
        raise ValueError("cannot select from an empty root")
    if temperature <= 0:
        return min(visits, key=lambda action: (-visits[action], action))
    weights = [visits[action] ** (1.0 / temperature) for action in visits]
    if sum(weights) <= 0:
        weights = [1.0] * len(visits)
    return rng.choices(tuple(visits), weights=weights, k=1)[0]


__all__ = ["add_dirichlet_noise", "exploration_bonus", "select_from_visits"]
