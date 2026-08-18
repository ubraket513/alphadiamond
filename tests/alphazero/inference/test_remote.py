from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

import pytest

from diamond.alphazero.evaluator.base import EvalRequest, EvalResult, Evaluator
from diamond.alphazero.inference.coordinator import InferenceConfig, InferenceCoordinator
from diamond.alphazero.inference.protocol import (
    InferenceFailure,
    InferenceRequest,
    InferenceResponse,
    ModelKey,
)
from diamond.alphazero.inference.remote import RemoteEvaluator


def _key() -> ModelKey:
    return ModelKey("Soo", "1.2.3", "a" * 64)


def _request(number: int = 0) -> EvalRequest:
    return EvalRequest(
        node_features=((float(number), 1.0, 2.0, 3.0),),
        legal_action_ids=(7, 19),
        canonical_player_ids=(1, 2),
    )


def _response(request: InferenceRequest) -> InferenceResponse:
    return InferenceResponse.from_eval_result(
        request,
        EvalResult(
            priors={7: float(request.node_features[0][0]), 19: 1.0},
            value=0.5,
        ),
    )


class ReorderingCoordinator:
    config = InferenceConfig(max_batch_size=2, max_wait_ms=20, request_queue_capacity=4)

    def __init__(self) -> None:
        self.pending: list[tuple[InferenceRequest, Queue[object]]] = []

    def submit(self, request: InferenceRequest, reply_queue: Queue[object]) -> None:
        self.pending.append((request, reply_queue))
        if len(self.pending) % 2 == 0:
            for queued_request, queued_replies in reversed(self.pending[-2:]):
                queued_replies.put(_response(queued_request))


class SilentCoordinator:
    config = InferenceConfig(
        max_batch_size=1,
        max_wait_ms=20,
        request_queue_capacity=1,
        response_timeout_s=0.02,
    )

    def submit(self, request: InferenceRequest, reply_queue: Queue[object]) -> None:
        pass


class FailingCoordinator:
    config = InferenceConfig(max_batch_size=1, max_wait_ms=20, request_queue_capacity=1)

    def submit(self, request: InferenceRequest, reply_queue: Queue[object]) -> None:
        reply_queue.put(
            InferenceFailure(
                client_id=request.client_id,
                request_id=request.request_id,
                model_key=request.model_key,
                error_type="RuntimeError",
                message="coordinator stopped",
            )
        )


def test_remote_evaluator_obeys_the_evaluator_protocol_and_restores_response_order() -> None:
    remote = RemoteEvaluator(ReorderingCoordinator(), model_key=_key(), client_id="worker-a")

    results = remote.evaluate((_request(1), _request(2)))

    assert isinstance(remote, Evaluator)
    assert results == (
        EvalResult(priors={7: 1.0, 19: 1.0}, value=0.5),
        EvalResult(priors={7: 2.0, 19: 1.0}, value=0.5),
    )


def test_remote_evaluator_uses_deterministic_unique_client_and_request_ids() -> None:
    coordinator = ReorderingCoordinator()
    first = RemoteEvaluator(coordinator, model_key=_key(), client_id="worker-a")
    second = RemoteEvaluator(coordinator, model_key=_key(), client_id="worker-b")

    first.evaluate((_request(1), _request(2)))
    second.evaluate((_request(3), _request(4)))

    assert [request.correlation_id for request, _ in coordinator.pending] == [
        ("worker-a", "worker-a:0"),
        ("worker-a", "worker-a:1"),
        ("worker-b", "worker-b:0"),
        ("worker-b", "worker-b:1"),
    ]


def test_remote_evaluator_times_out_when_the_coordinator_does_not_reply() -> None:
    remote = RemoteEvaluator(SilentCoordinator(), model_key=_key(), client_id="worker-a")

    with pytest.raises(TimeoutError, match="timed out"):
        remote.evaluate((_request(),))


def test_remote_evaluator_raises_a_coordinator_failure() -> None:
    remote = RemoteEvaluator(FailingCoordinator(), model_key=_key(), client_id="worker-a")

    with pytest.raises(RuntimeError, match="coordinator stopped"):
        remote.evaluate((_request(),))


class EchoEvaluator:
    def evaluate(self, requests: tuple[InferenceRequest, ...]) -> tuple[InferenceResponse, ...]:
        return tuple(_response(request) for request in requests)


def test_concurrent_remote_clients_receive_only_their_own_responses() -> None:
    coordinator = InferenceCoordinator(
        EchoEvaluator(), InferenceConfig(max_batch_size=2, max_wait_ms=20, request_queue_capacity=4)
    )
    first = RemoteEvaluator(coordinator, model_key=_key(), client_id="worker-a")
    second = RemoteEvaluator(coordinator, model_key=_key(), client_id="worker-b")
    received: list[tuple[EvalResult, ...]] = []
    coordinator.start()
    try:
        threads = [
            Thread(target=lambda remote=remote: received.append(remote.evaluate((_request(1),))))
            for remote in (first, second)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert received == [
            (EvalResult(priors={7: 1.0, 19: 1.0}, value=0.5),),
            (EvalResult(priors={7: 1.0, 19: 1.0}, value=0.5),),
        ]
    finally:
        coordinator.stop()


def test_remote_matches_local_fp32_torch_evaluation_through_the_central_pool(tmp_path) -> None:
    import torch

    from diamond.alphazero.config import NetworkConfig
    from diamond.alphazero.evaluator.torch import TorchEvaluator
    from diamond.alphazero.identity import CheckpointCompatibilitySpec
    from diamond.alphazero.inference.model_pool import InferenceModelPool
    from diamond.alphazero.network import SooModel

    config = NetworkConfig(width=16, residual_blocks=1)
    spec = CheckpointCompatibilitySpec.soo(model_version="1.2.3", network_config=config)
    model = SooModel(config, model_version="1.2.3")
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "metadata": spec.to_metadata(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    pool = InferenceModelPool()
    key = pool.activate_checkpoint(checkpoint, expected=spec)
    local = TorchEvaluator(model, value_size=1)
    request = EvalRequest(
        node_features=tuple((0.0, 1.0, 2.0, 3.0) for _ in range(73)),
        legal_action_ids=(7, 19, 100),
        canonical_player_ids=(1, 2),
    )
    coordinator = InferenceCoordinator(
        pool, InferenceConfig(max_batch_size=2, max_wait_ms=20, request_queue_capacity=2)
    )
    remote = RemoteEvaluator(coordinator, model_key=key, client_id="worker-a")
    coordinator.start()
    try:
        central = remote.evaluate((request,))[0]
    finally:
        coordinator.stop()

    direct = local.evaluate((request,))[0]
    assert central.priors == pytest.approx(direct.priors)
    assert central.value == pytest.approx(direct.value)


def test_loading_mcts_modules_does_not_import_torch_rating_or_orchestration() -> None:
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    environment = os.environ | {"PYTHONPATH": source_root}
    code = """
import importlib
import sys
importlib.import_module('diamond.alphazero.mcts.search_2p')
importlib.import_module('diamond.alphazero.mcts.search_3p')
forbidden = ('torch', 'diamond.alphazero.rating', 'diamond.alphazero.orchestration')
loaded = tuple(sys.modules)
assert not any(name == prefix or name.startswith(prefix + '.') for name in loaded for prefix in forbidden), loaded
"""

    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
