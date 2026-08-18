"""Single-process Soo and Min self-play runners."""

from .runner_2p import SooSelfPlayRunner
from .runner_3p import MinSelfPlayRunner

__all__ = ["MinSelfPlayRunner", "SooSelfPlayRunner"]
