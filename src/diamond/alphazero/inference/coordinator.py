"""Bounded single-worker batching for centralized AlphaZero inference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Protocol

from .protocol import InferenceFailure, InferenceRequest, InferenceResponse, ModelKey


class BatchEvaluator(Protocol):
    def evaluate(self, requests: tuple[InferenceRequest, ...]) -> tuple[InferenceResponse, ...]: ...


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    max_batch_size: int
    max_wait_ms: int
    request_queue_capacity: int
    response_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.max_wait_ms < 1:
            raise ValueError("max_wait_ms must be positive")
        if self.request_queue_capacity < 1:
            raise ValueError("request_queue_capacity must be positive")
        if self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    batches_completed: int = 0
    requests_completed: int = 0
    max_batch_size: int = 0
    total_queue_latency_s: float = 0.0


@dataclass(frozen=True, slots=True)
class _QueuedRequest:
    request: object
    reply_queue: Queue[object]
    submitted_at: float


_STOP = object()


class InferenceCoordinator:
    """Batch requests for one model key without allowing unbounded backlog."""

    def __init__(self, evaluator: BatchEvaluator, config: InferenceConfig) -> None:
        self.evaluator = evaluator
        self.config = config
        self._requests: Queue[object] = Queue(maxsize=config.request_queue_capacity)
        self._metrics = InferenceMetrics()
        self._metrics_lock = Lock()
        self._worker: Thread | None = None

    @property
    def metrics(self) -> InferenceMetrics:
        with self._metrics_lock:
            return replace(self._metrics)

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._worker = Thread(target=self._run, name="alphazero-inference", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if worker.is_alive():
            self._requests.put(_STOP)
            worker.join()
        self._worker = None

    def submit(self, request: object, reply_queue: Queue[object]) -> None:
        item = _QueuedRequest(request=request, reply_queue=reply_queue, submitted_at=monotonic())
        try:
            self._requests.put_nowait(item)
        except Full:
            self._reply_failure(item, Full("inference request queue is full"))

    def _run(self) -> None:
        pending: dict[ModelKey, list[_QueuedRequest]] = {}
        stopping = False
        while not stopping:
            now = monotonic()
            due_keys = [
                key
                for key, batch in pending.items()
                if batch[0].submitted_at + self.config.max_wait_ms / 1000 <= now
            ]
            if due_keys:
                for key in due_keys:
                    self._flush(pending.pop(key))
                continue

            timeout: float | None = None
            if pending:
                deadline = min(
                    batch[0].submitted_at + self.config.max_wait_ms / 1000
                    for batch in pending.values()
                )
                timeout = max(0.0, deadline - now)
            try:
                item = self._requests.get(timeout=timeout)
            except Empty:
                continue
            if item is _STOP:
                stopping = True
                continue
            assert isinstance(item, _QueuedRequest)
            request = self._validated_request(item)
            if request is None:
                continue
            batch = pending.setdefault(request.model_key, [])
            batch.append(item)
            if len(batch) == self.config.max_batch_size:
                self._flush(pending.pop(request.model_key))

        for batch in pending.values():
            self._flush(batch)

    def _validated_request(self, item: _QueuedRequest) -> InferenceRequest | None:
        request = item.request
        try:
            if not isinstance(request, InferenceRequest):
                raise ValueError("inference request must be an InferenceRequest")
            return InferenceRequest(
                client_id=request.client_id,
                request_id=request.request_id,
                model_key=request.model_key,
                node_features=request.node_features,
                legal_action_ids=request.legal_action_ids,
                canonical_player_ids=request.canonical_player_ids,
            )
        except Exception as error:
            self._reply_failure(item, error)
            return None

    def _flush(self, batch: list[_QueuedRequest]) -> None:
        requests = tuple(item.request for item in batch)
        try:
            assert all(isinstance(request, InferenceRequest) for request in requests)
            responses = self.evaluator.evaluate(requests)  # type: ignore[arg-type]
            if len(responses) != len(batch):
                raise ValueError("inference worker returned an unexpected response count")
            validated: list[InferenceResponse] = []
            for request, response in zip(requests, responses, strict=True):
                if not isinstance(response, InferenceResponse):
                    raise ValueError("inference worker returned a malformed response")
                if response.correlation_id != request.correlation_id or response.model_key != request.model_key:
                    raise ValueError("inference worker returned a mismatched response")
                validated.append(response)
            for item, response in zip(batch, validated, strict=True):
                item.reply_queue.put(response)
            self._record_batch(batch)
        except Exception as error:
            for item in batch:
                self._reply_failure(item, error)
            self._record_batch(batch)

    def _reply_failure(self, item: _QueuedRequest, error: Exception) -> None:
        request = item.request
        if not isinstance(request, InferenceRequest) or not isinstance(request.model_key, ModelKey):
            return
        item.reply_queue.put(
            InferenceFailure(
                client_id=request.client_id,
                request_id=request.request_id,
                model_key=request.model_key,
                error_type=type(error).__name__,
                message=str(error) or type(error).__name__,
            )
        )

    def _record_batch(self, batch: list[_QueuedRequest]) -> None:
        latency = sum(monotonic() - item.submitted_at for item in batch)
        with self._metrics_lock:
            self._metrics = InferenceMetrics(
                batches_completed=self._metrics.batches_completed + 1,
                requests_completed=self._metrics.requests_completed + len(batch),
                max_batch_size=max(self._metrics.max_batch_size, len(batch)),
                total_queue_latency_s=self._metrics.total_queue_latency_s + latency,
            )


__all__ = ["InferenceConfig", "InferenceCoordinator", "InferenceMetrics"]
