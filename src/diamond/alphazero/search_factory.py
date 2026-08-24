"""Which search engine to use, decided in one place.

The C++ core is the authority, so callers should get it without asking. They
should also not have to know when they cannot have it: the native search is
built around the 73-hole two-seat game, so a three-player match, a reduced
board or a test double falls back to the Python search rather than failing.

Resolved once per caller rather than per move -- an arena that changed engines
halfway through would be reporting a win rate over two different searches.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import MCTSConfig
from .evaluator.base import Evaluator
from .mcts.search_2p import MCTS2P

SearchFactory = Callable[[Any, Evaluator, MCTSConfig], Any]


def two_player_search() -> SearchFactory:
    """A factory that prefers the native search and falls back per game."""
    from .native import is_available

    if not is_available():
        return MCTS2P

    from .native.search import NativeSearch2P

    def factory(game: Any, evaluator: Evaluator, config: MCTSConfig, **kwargs: Any) -> Any:
        # `deadline` and other Python-search extras are not implemented natively;
        # a caller that needs one keeps the Python search rather than silently
        # losing its wall-clock bound.
        if not kwargs and NativeSearch2P.can_drive(game):
            return NativeSearch2P(game, evaluator, config)
        return MCTS2P(game, evaluator, config, **kwargs)

    return factory


__all__ = ["SearchFactory", "two_player_search"]
