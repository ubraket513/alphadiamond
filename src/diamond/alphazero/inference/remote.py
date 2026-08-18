"""Worker-side evaluator that submits correlated requests to central inference."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Lock
from time import monotonic
from typing import Protocol

from ..evaluator.base import EvalRequest, EvalResult
from .coordinator import InferenceConfig
from .protocol import InferenceFailure, InferenceRequest, InferenceResponse, ModelKey


class RequestCoordinator(Protocol):
    config: InferenceConfig

    def submit(self, request: InferenceRequest, reply_queue: Queue[object]) -> None: ...


class RemoteEvaluator:
    """Adapt the local evaluator protocol to a central coordinator request queue."""

    def __init__(
        self,
        coordinator: RequestCoordinator,
        *,
        model_key: ModelKey,
        client_id: str,
        response_timeout_s: float | None = None,
    ) -> None:
        if not client_id.strip():
            raise ValueError("client_id must be a non-empty string")
        self.coordinator = coordinator
        self.model_key = model_key
        self.client_id = client_id
        self.response_timeout_s = (
            coordinator.config.response_timeout_s
            if response_timeout_s is None
            else response_timeout_s
        )
        if self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")
        self._replies: Queue[object] = Queue()
        self._lock = Lock()
        self._next_request_id = 0

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        if not isinstance(requests, tuple):
            raise ValueError("evaluation requests must be a tuple")
        if not requests:
            return ()
        with self._lock:
            envelopes = tuple(
                InferenceRequest.from_eval_request(
                    client_id=self.client_id,
                    request_id=self._new_request_id(),
                    model_key=self.model_key,
                    request=request,
                )
                for request in requests
            )
            waiting = {envelope.correlation_id: envelope for envelope in envelopes}
            received: dict[tuple[str, str], EvalResult] = {}
            for envelope in envelopes:
                self.coordinator.submit(envelope, self._replies)

            deadline = monotonic() + self.response_timeout_s
            while waiting:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for central inference response")
                try:
                    response = self._replies.get(timeout=remaining)
                except Empty as error:
                    raise TimeoutError("timed out waiting for central inference response") from error
                if isinstance(response, InferenceFailure):
                    if response.correlation_id in waiting:
                        raise RuntimeError(f"{response.error_type}: {response.message}")
                    continue
                if not isinstance(response, InferenceResponse):
                    raise RuntimeError("coordinator returned a malformed inference response")
                if response.correlation_id not in waiting:
                    continue
                received[response.correlation_id] = response.to_eval_result()
                del waiting[response.correlation_id]

            return tuple(received[envelope.correlation_id] for envelope in envelopes)

    def _new_request_id(self) -> str:
        request_id = f"{self.client_id}:{self._next_request_id}"
        self._next_request_id += 1
        return request_id


__all__ = ["RemoteEvaluator"]
