"""Agent abstraction and its current implementations.

::

    Agent
    ├── RandomAgent      # implemented today
    ├── AlphaZeroAgent   # future: MCTS + OpenVINO
    ├── OpenVINOAgent    # future
    └── RemoteAgent      # future, if needed
"""

from .base import Agent, MoveProposal, MoveRequest, NoLegalMoveError
from .random_agent import RandomAgent

__all__ = ["Agent", "MoveProposal", "MoveRequest", "NoLegalMoveError", "RandomAgent"]
