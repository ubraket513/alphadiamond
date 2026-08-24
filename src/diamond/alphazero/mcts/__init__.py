"""PUCT search with distinct 2P scalar and 3P vector semantics.

The C++ search under ``native/`` is the authority. This implementation is the
oracle the Gate B golden files were generated from, and the other half of the
bridge parity gates; see ``docs/architecture/retiring_the_python_engine.md``.
"""

from .search_2p import MCTS2P, SearchResult2P
from .search_3p import MCTS3P, SearchResult3P

__all__ = ["MCTS2P", "MCTS3P", "SearchResult2P", "SearchResult3P"]
