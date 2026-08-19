from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event
from time import monotonic

from diamond.alphazero.evaluator.base import EvalResult
from diamond.alphazero.inference.coordinator import (
    InferenceConfig,
    InferenceCoordinator,
)
from diamond.alphazero.inference.protocol import (
    InferenceFailure,
    InferenceRequest,
    InferenceResponse,
    ModelKey,
)


def _key(digest: str = "a" * 64) -> ModelKey:
    return ModelKey("Soo", "1.2.3", digest)


def _request(
    number: int,
    *,
    key: ModelKey | None = None,
    client_id: str = "client",
) -> InferenceRequest:
    return InferenceRequest(
        client_id=client_id,
        request_id=f"request-{number}",
        model_key=key or _key(),
        node_features=((0.0, 1.0, 2.0, 3.0),),
        legal_action_ids=(7, 19),
        canonical_player_ids=(1, 2),
    )


class RecordingEvaluator:
    def __init__(self, error: Exception | None = None) -> None:
        self.batches: list[tuple[InferenceRequest, ...]] = []
        self.error = error

    def evaluate(self, requests: tuple[InferenceRequest, ...]) -> tuple[InferenceResponse, ...]:
        self.batches.append(requests)
        if self.error is not None:
            raise self.error
        return tuple(
            InferenceResponse.from_eval_result(
                request, EvalResult(priors={7: 0.75, 19: 0.25}, value=0.5)
            )
            for request in requests
        )


class BlockingEvaluator(RecordingEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def evaluate(self, requests: tuple[InferenceRequest, ...]) -> tuple[InferenceResponse, ...]:
        self.started.set()
        self.release.wait(timeout=1.0)
        return super().evaluate(requests)


def _responses(reply_queue: Queue[object], count: int) -> list[object]:
    return [reply_queue.get(timeout=1.0) for _ in range(count)]


def _response_or_none(reply_queue: Queue[object]) -> object | None:
    try:
        return reply_queue.get(timeout=0.2)
    except Empty:
        return None


def test_coordinator_flushes_a_full_single_key_batch() -> None:
    evaluator = RecordingEvaluator()
    coordinator = InferenceCoordinator(
        evaluator, InferenceConfig(max_batch_size=3, max_wait_ms=500, request_queue_capacity=3)
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    try:
        for number in range(3):
            coordinator.submit(_request(number), replies)

        responses = _responses(replies, 3)

        assert len(evaluator.batches) == 1
        assert [request.request_id for request in evaluator.batches[0]] == [
            "request-0",
            "request-1",
            "request-2",
        ]
        assert all(isinstance(response, InferenceResponse) for response in responses)
        assert coordinator.metrics.batches_completed == 1
        assert coordinator.metrics.requests_completed == 3
    finally:
        coordinator.stop()


def test_coordinator_flushes_an_underfull_batch_at_its_monotonic_deadline() -> None:
    evaluator = RecordingEvaluator()
    coordinator = InferenceCoordinator(
        evaluator, InferenceConfig(max_batch_size=2, max_wait_ms=30, request_queue_capacity=2)
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    try:
        started = monotonic()
        coordinator.submit(_request(1), replies)

        response = replies.get(timeout=1.0)

        assert isinstance(response, InferenceResponse)
        assert len(evaluator.batches) == 1
        assert len(evaluator.batches[0]) == 1
        assert monotonic() - started >= 0.02
        assert coordinator.metrics.max_batch_size == 1
        assert coordinator.metrics.total_queue_latency_s > 0
    finally:
        coordinator.stop()


def test_coordinator_never_batches_distinct_model_keys_together() -> None:
    evaluator = RecordingEvaluator()
    coordinator = InferenceCoordinator(
        evaluator, InferenceConfig(max_batch_size=2, max_wait_ms=20, request_queue_capacity=2)
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    try:
        coordinator.submit(_request(1, key=_key("a" * 64)), replies)
        coordinator.submit(_request(2, key=_key("b" * 64)), replies)

        _responses(replies, 2)

        assert {batch[0].model_key for batch in evaluator.batches} == {_key("a" * 64), _key("b" * 64)}
        assert all(len({request.model_key for request in batch}) == 1 for batch in evaluator.batches)
    finally:
        coordinator.stop()


def test_coordinator_returns_a_correlated_failure_when_its_request_queue_is_full() -> None:
    coordinator = InferenceCoordinator(
        RecordingEvaluator(),
        InferenceConfig(max_batch_size=2, max_wait_ms=500, request_queue_capacity=1),
    )
    replies: Queue[object] = Queue()

    coordinator.submit(_request(1), replies)
    coordinator.submit(_request(2), replies)

    failure = replies.get(timeout=1.0)

    assert isinstance(failure, InferenceFailure)
    assert failure.correlation_id == ("client", "request-2")
    assert failure.error_type == "Full"


def test_capacity_includes_requests_already_dequeued_for_evaluation() -> None:
    evaluator = BlockingEvaluator()
    coordinator = InferenceCoordinator(
        evaluator,
        InferenceConfig(max_batch_size=1, max_wait_ms=500, request_queue_capacity=1),
    )
    first_replies: Queue[object] = Queue()
    second_replies: Queue[object] = Queue()
    coordinator.start()
    try:
        coordinator.submit(_request(1), first_replies)
        assert evaluator.started.wait(timeout=1.0)

        coordinator.submit(_request(2), second_replies)
        evaluator.release.set()

        assert isinstance(first_replies.get(timeout=1.0), InferenceResponse)
        failure = second_replies.get(timeout=1.0)
        assert isinstance(failure, InferenceFailure)
        assert failure.correlation_id == ("client", "request-2")
        assert failure.error_type == "Full"
    finally:
        evaluator.release.set()
        coordinator.stop()


def test_coordinator_returns_a_correlated_failure_for_a_malformed_request() -> None:
    evaluator = RecordingEvaluator()
    coordinator = InferenceCoordinator(
        evaluator, InferenceConfig(max_batch_size=1, max_wait_ms=500, request_queue_capacity=1)
    )
    replies: Queue[object] = Queue()
    malformed = _request(1)
    object.__setattr__(malformed, "node_features", ())
    coordinator.start()
    try:
        coordinator.submit(malformed, replies)

        failure = replies.get(timeout=1.0)

        assert isinstance(failure, InferenceFailure)
        assert failure.correlation_id == malformed.correlation_id
        assert failure.error_type == "ValueError"
        assert evaluator.batches == []
    finally:
        coordinator.stop()


def test_coordinator_accepts_a_canonical_request_payload_mapping() -> None:
    coordinator = InferenceCoordinator(
        RecordingEvaluator(),
        InferenceConfig(max_batch_size=1, max_wait_ms=500, request_queue_capacity=1),
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    try:
        coordinator.submit(_request(1).to_payload(), replies)

        response = _response_or_none(replies)

        assert isinstance(response, InferenceResponse)
        assert response.correlation_id == ("client", "request-1")
    finally:
        coordinator.stop()


def test_malformed_payload_with_recoverable_identity_returns_correlated_failure() -> None:
    coordinator = InferenceCoordinator(
        RecordingEvaluator(),
        InferenceConfig(max_batch_size=1, max_wait_ms=500, request_queue_capacity=1),
    )
    replies: Queue[object] = Queue()
    payload = _request(1).to_payload()
    payload["node_features"] = []
    coordinator.start()
    try:
        coordinator.submit(payload, replies)

        failure = _response_or_none(replies)

        assert isinstance(failure, InferenceFailure)
        assert failure.correlation_id == ("client", "request-1")
        assert failure.error_type == "ValueError"
    finally:
        coordinator.stop()


def test_uncorrelated_malformed_transport_returns_fallback_error_and_releases_capacity() -> None:
    coordinator = InferenceCoordinator(
        RecordingEvaluator(),
        InferenceConfig(max_batch_size=1, max_wait_ms=500, request_queue_capacity=1),
    )
    malformed_replies: Queue[object] = Queue()
    valid_replies: Queue[object] = Queue()
    coordinator.start()
    try:
        coordinator.submit({}, malformed_replies)

        fallback = _response_or_none(malformed_replies)
        coordinator.submit(_request(2), valid_replies)
        response = _response_or_none(valid_replies)

        assert isinstance(fallback, ValueError)
        assert "uncorrelated" in str(fallback)
        assert isinstance(response, InferenceResponse)
    finally:
        coordinator.stop()


def test_coordinator_propagates_worker_errors_to_every_request_in_the_batch() -> None:
    coordinator = InferenceCoordinator(
        RecordingEvaluator(RuntimeError("model unavailable")),
        InferenceConfig(max_batch_size=2, max_wait_ms=500, request_queue_capacity=2),
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    try:
        coordinator.submit(_request(1), replies)
        coordinator.submit(_request(2), replies)

        failures = _responses(replies, 2)

        assert all(isinstance(failure, InferenceFailure) for failure in failures)
        assert {failure.correlation_id for failure in failures} == {
            ("client", "request-1"),
            ("client", "request-2"),
        }
        assert {failure.error_type for failure in failures} == {"RuntimeError"}
    finally:
        coordinator.stop()


def test_coordinator_rejects_an_entire_batch_when_any_worker_response_is_malformed() -> None:
    class PartlyMalformedEvaluator:
        def evaluate(self, requests: tuple[InferenceRequest, ...]) -> tuple[object, ...]:
            return (
                InferenceResponse.from_eval_result(
                    requests[0], EvalResult(priors={7: 0.75, 19: 0.25}, value=0.5)
                ),
                object(),
            )

    coordinator = InferenceCoordinator(
        PartlyMalformedEvaluator(),
        InferenceConfig(max_batch_size=2, max_wait_ms=500, request_queue_capacity=2),
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    try:
        coordinator.submit(_request(1), replies)
        coordinator.submit(_request(2), replies)

        failures = _responses(replies, 2)

        assert all(isinstance(failure, InferenceFailure) for failure in failures)
        assert {failure.correlation_id for failure in failures} == {
            ("client", "request-1"),
            ("client", "request-2"),
        }
    finally:
        coordinator.stop()


def test_coordinator_stop_fails_pending_work_and_joins_its_worker() -> None:
    evaluator = RecordingEvaluator()
    coordinator = InferenceCoordinator(
        evaluator, InferenceConfig(max_batch_size=2, max_wait_ms=500, request_queue_capacity=1)
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    coordinator.submit(_request(1), replies)

    coordinator.stop()

    failure = replies.get_nowait()
    assert isinstance(failure, InferenceFailure)
    assert failure.correlation_id == ("client", "request-1")
    assert "closed" in failure.message
    assert evaluator.batches == []
    assert not coordinator.is_running


def test_stop_fails_prestart_backlog_and_rejects_later_submissions() -> None:
    coordinator = InferenceCoordinator(
        RecordingEvaluator(),
        InferenceConfig(max_batch_size=1, max_wait_ms=500, request_queue_capacity=1),
    )
    accepted_replies: Queue[object] = Queue()
    rejected_replies: Queue[object] = Queue()
    coordinator.submit(_request(1), accepted_replies)

    coordinator.stop()
    coordinator.submit(_request(2), rejected_replies)

    accepted_failure = _response_or_none(accepted_replies)
    rejected_failure = _response_or_none(rejected_replies)
    assert isinstance(accepted_failure, InferenceFailure)
    assert accepted_failure.correlation_id == ("client", "request-1")
    assert isinstance(rejected_failure, InferenceFailure)
    assert rejected_failure.correlation_id == ("client", "request-2")
    assert "closed" in rejected_failure.message


def test_stop_timeout_fails_inflight_work_without_waiting_for_a_hung_evaluator() -> None:
    evaluator = BlockingEvaluator()
    coordinator = InferenceCoordinator(
        evaluator,
        InferenceConfig(
            max_batch_size=1,
            max_wait_ms=500,
            request_queue_capacity=1,
            response_timeout_s=0.05,
        ),
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    coordinator.submit(_request(1), replies)
    assert evaluator.started.wait(timeout=1.0)

    started = monotonic()
    coordinator.stop()
    elapsed = monotonic() - started

    failure = _response_or_none(replies)
    evaluator.release.set()
    coordinator.stop()
    assert elapsed < 0.25
    assert isinstance(failure, InferenceFailure)
    assert failure.correlation_id == ("client", "request-1")
    assert "closed" in failure.message


def test_worker_exits_after_full_queue_rejects_stop_sentinel() -> None:
    class StopRejectingQueue(Queue[object]):
        def __init__(self) -> None:
            super().__init__(maxsize=1)
            self.get_started = Event()
            self.allow_get = Event()
            self.reject_timed_put = True

        def get(self, block: bool = True, timeout: float | None = None) -> object:
            self.get_started.set()
            self.allow_get.wait(timeout=1.0)
            return super().get(block=block, timeout=timeout)

        def put(
            self,
            item: object,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            if self.reject_timed_put and timeout is not None:
                raise Full
            super().put(item, block=block, timeout=timeout)

    coordinator = InferenceCoordinator(
        RecordingEvaluator(),
        InferenceConfig(
            max_batch_size=1,
            max_wait_ms=500,
            request_queue_capacity=1,
            response_timeout_s=0.02,
        ),
    )
    requests = StopRejectingQueue()
    coordinator._requests = requests
    replies: Queue[object] = Queue()
    coordinator.submit(_request(1), replies)
    coordinator.start()
    assert requests.get_started.wait(timeout=1.0)

    coordinator.stop()
    worker = coordinator._worker
    assert worker is not None
    requests.allow_get.set()
    try:
        worker.join(timeout=0.5)
        assert not worker.is_alive()
    finally:
        requests.reject_timed_put = False
        coordinator.stop()


def test_metrics_stay_bounded_under_gpu_scale_request_volume() -> None:
    """Guards the quadratic accumulation this replaced from creeping back."""
    import random

    from diamond.alphazero.inference.coordinator import InferenceMetrics
    from diamond.alphazero.inference.summary import RESERVOIR_CAPACITY

    metrics = InferenceMetrics()
    rng = random.Random(11)
    for _ in range(2_000):
        metrics = metrics.record_batch(
            batch_size=16,
            queue_to_dispatch_s=(0.001,) * 16,
            inference_duration_s=0.004,
            response_latencies_s=(0.006,) * 16,
            admission_latencies_s=(0.0,) * 16,
            rng=rng,
        )

    # Exact counts survive; retained samples do not grow with the request count.
    assert metrics.requests_completed == 32_000
    assert metrics.batches_completed == 2_000
    assert metrics.max_batch_size == 16
    for series in (
        metrics.batch_size_series,
        metrics.queue_to_dispatch_series,
        metrics.inference_latency_series,
        metrics.response_latency_series,
        metrics.admission_latency_series,
    ):
        assert len(series.reservoir) <= RESERVOIR_CAPACITY
