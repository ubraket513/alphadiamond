"""Bounded single-worker batching for centralized AlphaZero inference."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Lock, Thread
from time import monotonic
from typing import Protocol

from .protocol import InferenceFailure, InferenceRequest, InferenceResponse, ModelKey
from .summary import StreamingSeries


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


@dataclass(slots=True)
class InferenceMetrics:
    """Batching and latency statistics accumulated in constant memory.

    Series are reservoir-summarized rather than retained per request: a GPU
    iteration issues tens of thousands of requests, and keeping every sample
    made recording quadratic in the inference thread's own hot path.
    """

    batches_completed: int = 0
    requests_completed: int = 0
    total_queue_latency_s: float = 0.0
    batch_size_series: StreamingSeries = field(default_factory=StreamingSeries)
    admission_latency_series: StreamingSeries = field(default_factory=StreamingSeries)
    queue_to_dispatch_series: StreamingSeries = field(default_factory=StreamingSeries)
    inference_latency_series: StreamingSeries = field(default_factory=StreamingSeries)
    response_latency_series: StreamingSeries = field(default_factory=StreamingSeries)

    @property
    def max_batch_size(self) -> int:
        largest = self.batch_size_series.maximum
        return 0 if largest is None else int(largest)

    # Retained sample views, kept under their original names for the profiler.
    # Complete for a profiling run and bounded past RESERVOIR_CAPACITY, so a
    # long GPU session cannot grow them without limit.
    @property
    def batch_sizes(self) -> tuple[float, ...]:
        return self.batch_size_series.reservoir

    @property
    def admission_latency_samples(self) -> tuple[float, ...]:
        return self.admission_latency_series.reservoir

    @property
    def queue_to_dispatch_latency_samples(self) -> tuple[float, ...]:
        return self.queue_to_dispatch_series.reservoir

    @property
    def inference_latency_samples(self) -> tuple[float, ...]:
        return self.inference_latency_series.reservoir

    @property
    def response_latency_samples(self) -> tuple[float, ...]:
        return self.response_latency_series.reservoir

    def record_batch(
        self,
        *,
        batch_size: int,
        queue_to_dispatch_s: Sequence[float],
        inference_duration_s: float,
        response_latencies_s: Sequence[float],
        admission_latencies_s: Sequence[float],
        rng: random.Random,
    ) -> "InferenceMetrics":
        """Record one completed batch in place; returns self so calls can chain."""
        self.batches_completed += 1
        self.requests_completed += batch_size
        self.total_queue_latency_s += sum(queue_to_dispatch_s)
        self.batch_size_series.observe(batch_size, rng=rng)
        self.inference_latency_series.observe(max(0.0, inference_duration_s), rng=rng)
        for value in admission_latencies_s:
            self.admission_latency_series.observe(value, rng=rng)
        for value in queue_to_dispatch_s:
            self.queue_to_dispatch_series.observe(value, rng=rng)
        for value in response_latencies_s:
            self.response_latency_series.observe(value, rng=rng)
        return self

    def snapshot(self) -> "InferenceMetrics":
        """An independent copy, safe to read while the worker keeps recording."""
        return InferenceMetrics(
            batches_completed=self.batches_completed,
            requests_completed=self.requests_completed,
            total_queue_latency_s=self.total_queue_latency_s,
            batch_size_series=self.batch_size_series.snapshot(),
            admission_latency_series=self.admission_latency_series.snapshot(),
            queue_to_dispatch_series=self.queue_to_dispatch_series.snapshot(),
            inference_latency_series=self.inference_latency_series.snapshot(),
            response_latency_series=self.response_latency_series.snapshot(),
        )


@dataclass(slots=True)
class _QueuedRequest:
    request: object
    reply_queue: Queue[object]
    submitted_at: float
    admitted_at: float | None
    admitted: bool
    client_id: str | None
    request_id: str | None
    model_key: ModelKey | None
    completed: bool = False


_STOP = object()


class InferenceCoordinator:
    """Batch ``InferenceRequest`` objects or mapping payloads with bounded admission.

    Malformed transports with recoverable client, request, and model identities
    receive ``InferenceFailure``. If any failure-envelope identity is unavailable,
    the reply queue receives a coordinator-level ``ValueError`` instead.
    """

    def __init__(self, evaluator: BatchEvaluator, config: InferenceConfig) -> None:
        self.evaluator = evaluator
        self.config = config
        self._requests: Queue[object] = Queue(maxsize=config.request_queue_capacity)
        self._admission = BoundedSemaphore(config.request_queue_capacity)
        self._metrics = InferenceMetrics()
        # Fixed seed: reservoir sampling only shapes which latency samples are
        # retained, and a reproducible choice keeps ledgers comparable.
        self._sample_rng = random.Random(0)
        self._metrics_lock = Lock()
        self._state_lock = Lock()
        self._closed = False
        self._stop_sent = False
        self._outstanding: dict[int, _QueuedRequest] = {}
        self._worker: Thread | None = None

    @property
    def metrics(self) -> InferenceMetrics:
        with self._metrics_lock:
            return self._metrics.snapshot()

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("inference coordinator is closed")
            if self.is_running:
                return
            self._worker = Thread(target=self._run, name="alphazero-inference", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        with self._state_lock:
            self._closed = True
            worker = self._worker
            should_signal = worker is not None and worker.is_alive() and not self._stop_sent
        if should_signal:
            try:
                self._requests.put(_STOP, timeout=self.config.response_timeout_s)
                with self._state_lock:
                    self._stop_sent = True
            except Full:
                pass

        self._fail_outstanding(RuntimeError("inference coordinator is closed"))
        if worker is None:
            self._drain_closed_queue()
            return
        if worker.is_alive():
            worker.join(timeout=self.config.response_timeout_s)
        if not worker.is_alive():
            with self._state_lock:
                if self._worker is worker:
                    self._worker = None

    def submit(self, request: object, reply_queue: Queue[object]) -> None:
        submitted_at = monotonic()
        client_id, request_id, model_key = self._extract_identity(request)
        item = _QueuedRequest(
            request=request,
            reply_queue=reply_queue,
            submitted_at=submitted_at,
            admitted_at=None,
            admitted=False,
            client_id=client_id,
            request_id=request_id,
            model_key=model_key,
        )
        rejection: Exception | None = None
        with self._state_lock:
            if self._closed:
                rejection = RuntimeError("inference coordinator is closed")
            elif not self._admission.acquire(blocking=False):
                rejection = Full("inference request capacity is full")
            else:
                item.admitted = True
                item.admitted_at = monotonic()
                self._outstanding[id(item)] = item
                try:
                    self._requests.put_nowait(item)
                except Full as error:
                    rejection = error
        if rejection is not None:
            self._reply_failure(item, rejection)

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
            if self._is_closed():
                self._reply_failure(item, RuntimeError("inference coordinator is closed"))
                self._drain_closed_queue()
                break
            request = self._validated_request(item)
            if request is None:
                continue
            item.request = request
            batch = pending.setdefault(request.model_key, [])
            batch.append(item)
            if len(batch) == self.config.max_batch_size:
                self._flush(pending.pop(request.model_key))

        for batch in pending.values():
            for item in batch:
                self._reply_failure(item, RuntimeError("inference coordinator is closed"))

    def _validated_request(self, item: _QueuedRequest) -> InferenceRequest | None:
        request = item.request
        try:
            if isinstance(request, InferenceRequest):
                return InferenceRequest(
                    client_id=request.client_id,
                    request_id=request.request_id,
                    model_key=request.model_key,
                    node_features=request.node_features,
                    legal_action_ids=request.legal_action_ids,
                    canonical_player_ids=request.canonical_player_ids,
                )
            if isinstance(request, Mapping):
                return InferenceRequest.from_payload(request)
            raise ValueError("inference transport must be an InferenceRequest or mapping")
        except Exception as error:
            self._reply_failure(item, error)
            return None

    def _flush(self, batch: list[_QueuedRequest]) -> None:
        requests = tuple(item.request for item in batch)
        dispatch_started = monotonic()
        inference_started = dispatch_started
        try:
            assert all(isinstance(request, InferenceRequest) for request in requests)
            responses = self.evaluator.evaluate(requests)  # type: ignore[arg-type]
            inference_completed = monotonic()
            if len(responses) != len(batch):
                raise ValueError("inference worker returned an unexpected response count")
            validated: list[InferenceResponse] = []
            for request, response in zip(requests, responses, strict=True):
                if not isinstance(response, InferenceResponse):
                    raise ValueError("inference worker returned a malformed response")
                if response.correlation_id != request.correlation_id or response.model_key != request.model_key:
                    raise ValueError("inference worker returned a mismatched response")
                validated.append(response)
            if self._is_closed():
                raise RuntimeError("inference coordinator is closed")
            for item, response in zip(batch, validated, strict=True):
                self._complete(item, response)
            self._record_batch(
                batch,
                dispatch_started=dispatch_started,
                inference_duration_s=inference_completed - inference_started,
                response_completed=monotonic(),
            )
        except Exception as error:
            inference_completed = monotonic()
            for item in batch:
                self._reply_failure(item, error)
            self._record_batch(
                batch,
                dispatch_started=dispatch_started,
                inference_duration_s=inference_completed - inference_started,
                response_completed=monotonic(),
            )

    def _reply_failure(self, item: _QueuedRequest, error: Exception) -> None:
        if item.client_id is None or item.request_id is None or item.model_key is None:
            self._complete(
                item,
                ValueError(
                    "uncorrelated inference request could not produce an InferenceFailure: "
                    f"{str(error) or type(error).__name__}"
                ),
            )
            return
        self._complete(
            item,
            InferenceFailure(
                client_id=item.client_id,
                request_id=item.request_id,
                model_key=item.model_key,
                error_type=type(error).__name__,
                message=str(error) or type(error).__name__,
            ),
        )

    @staticmethod
    def _extract_identity(request: object) -> tuple[str | None, str | None, ModelKey | None]:
        if isinstance(request, InferenceRequest):
            raw_client_id = request.client_id
            raw_request_id = request.request_id
            raw_model_key: object = request.model_key
        elif isinstance(request, Mapping):
            raw_client_id = request.get("client_id")
            raw_request_id = request.get("request_id")
            raw_model_key = request.get("model_key")
        else:
            return None, None, None

        client_id = (
            raw_client_id
            if isinstance(raw_client_id, str) and raw_client_id.strip()
            else None
        )
        request_id = (
            raw_request_id if isinstance(raw_request_id, str) and raw_request_id.strip() else None
        )
        if isinstance(raw_model_key, ModelKey):
            model_key = raw_model_key
        else:
            try:
                model_key = ModelKey.from_payload(raw_model_key)
            except ValueError:
                model_key = None
        return client_id, request_id, model_key

    def _complete(self, item: _QueuedRequest, response: object) -> None:
        with self._state_lock:
            if item.completed:
                return
            item.completed = True
            item.reply_queue.put(response)
            if item.admitted:
                self._outstanding.pop(id(item), None)
                self._admission.release()

    def _is_closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def _fail_outstanding(self, error: Exception) -> None:
        with self._state_lock:
            outstanding = tuple(self._outstanding.values())
        for item in outstanding:
            self._reply_failure(item, error)

    def _drain_closed_queue(self) -> None:
        while True:
            try:
                item = self._requests.get_nowait()
            except Empty:
                return
            if isinstance(item, _QueuedRequest):
                self._reply_failure(item, RuntimeError("inference coordinator is closed"))

    def _record_batch(
        self,
        batch: list[_QueuedRequest],
        *,
        dispatch_started: float,
        inference_duration_s: float,
        response_completed: float,
    ) -> None:
        admission = tuple(
            max(0.0, (item.admitted_at or item.submitted_at) - item.submitted_at)
            for item in batch
        )
        queue = tuple(
            max(0.0, dispatch_started - (item.admitted_at or item.submitted_at))
            for item in batch
        )
        response = tuple(
            max(0.0, response_completed - item.submitted_at) for item in batch
        )
        with self._metrics_lock:
            self._metrics = self._metrics.record_batch(
                batch_size=len(batch),
                queue_to_dispatch_s=queue,
                inference_duration_s=inference_duration_s,
                response_latencies_s=response,
                admission_latencies_s=admission,
                rng=self._sample_rng,
            )


__all__ = ["InferenceConfig", "InferenceCoordinator", "InferenceMetrics"]
