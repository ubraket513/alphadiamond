from __future__ import annotations

from queue import Queue
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


def _responses(reply_queue: Queue[object], count: int) -> list[object]:
    return [reply_queue.get(timeout=1.0) for _ in range(count)]


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


def test_coordinator_stop_flushes_pending_work_and_joins_its_worker() -> None:
    evaluator = RecordingEvaluator()
    coordinator = InferenceCoordinator(
        evaluator, InferenceConfig(max_batch_size=2, max_wait_ms=500, request_queue_capacity=1)
    )
    replies: Queue[object] = Queue()
    coordinator.start()
    coordinator.submit(_request(1), replies)

    coordinator.stop()

    assert isinstance(replies.get_nowait(), InferenceResponse)
    assert len(evaluator.batches) == 1
    assert not coordinator.is_running
