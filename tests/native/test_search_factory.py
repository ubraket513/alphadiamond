"""Which search a caller gets, and what it is refused.

There is one engine. What the selector still decides is whether a game is one
the core can play at all -- a seat count it was not compiled for, or a caller's
stand-in that is not the 73-hole board. Those are unsupported inputs, not slow
paths, and the caller is told so rather than handed another engine's answer
(decision 1 in docs/architecture/decisions.md).

The search's own behaviour is not tested here: it is C++, and it is tested by
CTest without an interpreter. This file is the Python half of the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from diamond.alphazero.deadline import Deadline
from diamond.alphazero.evaluator.base import EvalResult
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.native import native_module
from diamond.alphazero.native.search import NativeSearch2P, NativeSearch3P
from diamond.alphazero.search_factory import (
    NativeSearchUnavailable,
    three_player_search,
    two_player_search,
)
from diamond.contract.state import build_players

pytestmark = pytest.mark.skipif(
    native_module() is None, reason="the native extension is not built"
)


class _Evaluator:
    def evaluate(self, requests):
        return tuple(
            EvalResult({action: 1.0 / len(r.legal_action_ids) for action in r.legal_action_ids}, 0.0)
            for r in requests
        )


@dataclass(frozen=True)
class _Config:
    simulations = 4
    c_puct = 1.5
    dirichlet_alpha = 0.3
    dirichlet_epsilon = 0.0
    seed = 0


def _adapter(player_count: int = 2) -> DiamondSearchAdapter:
    return DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(player_count)))


def test_the_real_game_goes_to_the_native_core() -> None:
    assert isinstance(two_player_search()(_adapter(), _Evaluator(), _Config()), NativeSearch2P)
    assert isinstance(
        three_player_search()(_adapter(3), _Evaluator(), _Config()), NativeSearch3P
    )


def test_a_three_seat_game_is_refused_by_the_two_seat_search() -> None:
    assert not NativeSearch2P.can_drive(_adapter(3))
    with pytest.raises(NativeSearchUnavailable, match="two-seat"):
        two_player_search()(_adapter(3), _Evaluator(), _Config())


def test_a_deadline_reaches_the_native_search() -> None:
    """The bound survives the crossing; what it then does is `budget_test`."""
    search = two_player_search()(_adapter(), _Evaluator(), _Config(), deadline=Deadline.start(5.0))
    assert isinstance(search, NativeSearch2P)


def test_an_option_the_native_search_does_not_understand_is_not_swallowed() -> None:
    with pytest.raises(TypeError, match="some_future_option"):
        two_player_search()(_adapter(), _Evaluator(), _Config(), some_future_option=1)


def test_a_caller_that_is_not_the_board_is_refused() -> None:
    """A stand-in has no board size, which is the question `can_drive` asks."""

    class NotAGame:
        players = build_players(2)

    assert not NativeSearch2P.can_drive(NotAGame())
    with pytest.raises(NativeSearchUnavailable):
        two_player_search()(NotAGame(), _Evaluator(), _Config())
