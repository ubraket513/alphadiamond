"""What a search returns, independent of which engine ran it.

These types live outside ``diamond.alphazero.mcts`` on purpose. The Python
search is being retired (see
``docs/architecture/retiring_the_python_engine.md``); the shape of a search
result is not, and the native search returns the same thing. Keeping the type
in the package being deleted would make every consumer a dependent of it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchResult2P:
    """Soo: one scalar q per root action, from the acting player's view."""

    selected_action: int
    visit_counts: dict[int, int]
    policy: dict[int, float]
    q_values: dict[int, float]


@dataclass(frozen=True, slots=True)
class SearchResult3P:
    """Min: one q *vector* per root action -- a value for every seat."""

    selected_action: int
    visit_counts: dict[int, int]
    policy: dict[int, float]
    q_values: dict[int, dict[int, float]]


__all__ = ["SearchResult2P", "SearchResult3P"]
