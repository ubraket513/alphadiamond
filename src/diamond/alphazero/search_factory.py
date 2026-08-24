"""Which search engine to use.

There is one: the C++ core. Decision 1 in docs/architecture/decisions.md
retired the pure-Python search as a fallback, so this no longer chooses between
two engines -- it checks that the one engine is available and says plainly what
is wrong when it is not.

What it still does is decline games the native core cannot play. Those are not
fallbacks, they are unsupported inputs, and a caller handed one gets an error
naming the reason rather than a different engine's answer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import MCTSConfig
from .evaluator.base import Evaluator

SearchFactory = Callable[[Any, Evaluator, MCTSConfig], Any]


class NativeSearchUnavailable(RuntimeError):
    """The native extension is required to search and is not importable."""


def _require_native() -> None:
    from .native import is_available, native_error

    if not is_available():
        raise NativeSearchUnavailable(
            "searching requires the native extension, which is not importable: "
            f"{native_error()}. Build it with `python tools/build_native.py`."
        )


def two_player_search() -> SearchFactory:
    """The native two-seat search."""
    _require_native()
    from .native.search import NativeSearch2P

    def factory(game: Any, evaluator: Evaluator, config: MCTSConfig, **kwargs: Any) -> Any:
        if not NativeSearch2P.can_drive(game):
            raise NativeSearchUnavailable(
                "the native search plays the 73-hole two-seat game; this game is "
                "neither, and there is no Python search to fall back to"
            )
        return NativeSearch2P(game, evaluator, config, **kwargs)

    return factory


def three_player_search() -> SearchFactory:
    """The native three-seat search."""
    _require_native()
    from .native.search import NativeSearch3P

    def factory(game: Any, evaluator: Evaluator, config: MCTSConfig, **kwargs: Any) -> Any:
        if not NativeSearch3P.can_drive(game):
            raise NativeSearchUnavailable(
                "the native search plays the 73-hole three-seat game; this game is "
                "neither, and there is no Python search to fall back to"
            )
        return NativeSearch3P(game, evaluator, config, **kwargs)

    return factory


__all__ = [
    "NativeSearchUnavailable",
    "SearchFactory",
    "three_player_search",
    "two_player_search",
]
