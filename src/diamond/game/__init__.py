"""The Python reading of Diamond's rules: legality, turn order, the podium.

**Not the source of truth.** The C++ core under ``native/`` is the authority for
rules, encoding, search and self-play. What is left here is the oracle that
``tests/golden/`` was generated from and the other half of the bridge parity
gates, and it is scheduled for deletion once neither job needs it --
``tests/test_engine_retirement.py`` fails if shipped code reaches for it.

What a position *is* -- the board, the seats, ``GameState``, ``Move`` -- moved to
:mod:`diamond.contract`, which this package imports. Nothing is re-exported from
there: a caller that wants the definitions should say so, and only the rules
below are this package's own.

See ``docs/architecture/retiring_the_python_engine.md``.
"""

from .history import MoveRecord
from .rules import (
    find_legal_move,
    find_winner,
    legal_moves,
    moves_from,
    next_player_id,
    update_ranking,
    validate_move,
)
from .session import SCHEMA_VERSION, GameSession

__all__ = [
    "SCHEMA_VERSION",
    "GameSession",
    "MoveRecord",
    "find_legal_move",
    "find_winner",
    "legal_moves",
    "moves_from",
    "next_player_id",
    "update_ranking",
    "validate_move",
]
