"""Asynchronous boundary between the GUI thread and an :class:`Agent`.

Nothing here knows what an agent actually does.  Today it wraps ``RandomAgent``
and finishes in microseconds; the same boundary will carry an AlphaZero MCTS
search that takes seconds without the GUI ever blocking.

Every request carries a *generation* token.  The controller bumps the
generation whenever the authoritative state changes (commit, undo, new game,
load), and drops any result whose token no longer matches — that is what stops
a proposal computed against an old board from being applied to the current one.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ..agents.base import Agent, MoveProposal, MoveRequest

DEFAULT_THINKING_DELAY_MS = 400
"""Artificial pause so the 'thinking' UX exists before a real search does.

It runs on the worker thread; the GUI thread never sleeps.
"""


class _TaskSignals(QObject):
    finished = Signal(int, object)  # generation, MoveProposal | None
    failed = Signal(int, str)  # generation, error message


class _AgentTask(QRunnable):
    def __init__(
        self,
        agent: Agent,
        request: MoveRequest,
        generation: int,
        signals: _TaskSignals,
        delay_ms: int,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._request = request
        self._generation = generation
        self._signals = signals
        self._delay_ms = delay_ms

    def run(self) -> None:  # executed on a pool thread
        try:
            if self._delay_ms > 0:
                time.sleep(self._delay_ms / 1000.0)
            proposal = self._agent.choose_move(self._request)
        except Exception as exc:  # never let a worker exception kill the pool
            self._signals.failed.emit(self._generation, f"{type(exc).__name__}: {exc}")
            return
        self._signals.finished.emit(self._generation, proposal)


class AiWorker(QObject):
    """Runs agent queries on a background thread pool."""

    proposalReady = Signal(int, object)  # generation, MoveProposal | None
    proposalFailed = Signal(int, str)

    def __init__(self, parent: QObject | None = None, delay_ms: int = DEFAULT_THINKING_DELAY_MS):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)  # one search at a time
        self._delay_ms = delay_ms
        self._signals = _TaskSignals(self)
        self._signals.finished.connect(self.proposalReady)
        self._signals.failed.connect(self.proposalFailed)

    @property
    def thinking_delay_ms(self) -> int:
        return self._delay_ms

    def set_thinking_delay_ms(self, value: int) -> None:
        self._delay_ms = max(0, int(value))

    def submit(self, agent: Agent, request: MoveRequest, generation: int) -> None:
        self._pool.start(_AgentTask(agent, request, generation, self._signals, self._delay_ms))

    @Slot()
    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Wait for in-flight work so the app can close without a crash."""
        self._pool.waitForDone(timeout_ms)
