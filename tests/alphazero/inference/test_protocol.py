from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from diamond.alphazero.evaluator.base import EvalRequest, EvalResult
from diamond.alphazero.inference.protocol import (
    InferenceFailure,
    InferenceRequest,
    InferenceResponse,
    ModelKey,
)


def eval_request() -> EvalRequest:
    return EvalRequest(
        node_features=((0.0, 1.0, 2.0, 3.0), (4.0, 5.0, 6.0, 7.0)),
        legal_action_ids=(7, 19),
        canonical_player_ids=(1, 2),
    )


def model_key(checkpoint_sha256: str = "a" * 64) -> ModelKey:
    return ModelKey(
        model_name="Soo",
        model_version="1.2.3",
        checkpoint_sha256=checkpoint_sha256,
    )


def test_request_response_and_failure_preserve_correlation_and_eval_round_trips() -> None:
    request = InferenceRequest.from_eval_request(
        client_id="worker-7",
        request_id="request-9",
        model_key=model_key(),
        request=eval_request(),
    )

    decoded_request = InferenceRequest.from_payload(request.to_payload())
    response = InferenceResponse.from_eval_result(
        decoded_request,
        EvalResult(priors={7: 0.75, 19: 0.25}, value=0.5),
    )
    decoded_response = InferenceResponse.from_payload(response.to_payload())
    failure = InferenceFailure(
        client_id=request.client_id,
        request_id=request.request_id,
        model_key=request.model_key,
        error_type="ValueError",
        message="bad request",
    )

    assert decoded_request == request
    assert decoded_request.to_eval_request() == eval_request()
    assert decoded_response.correlation_id == ("worker-7", "request-9")
    assert decoded_response.model_key == request.model_key
    assert decoded_response.to_eval_result() == EvalResult(
        priors={7: 0.75, 19: 0.25}, value=0.5
    )
    assert InferenceFailure.from_payload(failure.to_payload()) == failure
    with pytest.raises(FrozenInstanceError):
        request.client_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model_name": "Soo", "model_version": "1.2.3", "checkpoint_sha256": "bad"},
        {
            "client_id": "worker",
            "request_id": "request",
            "model_key": model_key().to_payload(),
            "node_features": [[0.0, 1.0]],
            "legal_action_ids": [7],
            "canonical_player_ids": [1, 2],
            "unexpected": True,
        },
        {
            "client_id": "worker",
            "request_id": "request",
            "model_key": model_key().to_payload(),
            "node_features": [[0.0, 1.0]],
            "legal_action_ids": [7, 7],
            "canonical_player_ids": [1, 2],
        },
    ],
)
def test_transport_payloads_reject_malformed_or_noncanonical_data(payload: object) -> None:
    with pytest.raises(ValueError):
        if isinstance(payload, dict) and "client_id" in payload:
            InferenceRequest.from_payload(payload)
        else:
            ModelKey.from_payload(payload)


def test_model_key_distinguishes_exact_artifacts_with_the_same_version() -> None:
    first = model_key("a" * 64)
    second = model_key("b" * 64)

    assert first != second
    assert first.to_payload() != second.to_payload()
