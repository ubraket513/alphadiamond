from __future__ import annotations

import pytest
import torch

from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.identity import CheckpointCompatibilityError, CheckpointCompatibilitySpec
from diamond.alphazero.inference.model_pool import InferenceModelPool
from diamond.alphazero.inference.protocol import InferenceRequest
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer
from diamond.alphazero.checkpoint import save_checkpoint


def _spec(model_name: str) -> CheckpointCompatibilitySpec:
    network = NetworkConfig(width=16, residual_blocks=1)
    factory = (
        CheckpointCompatibilitySpec.soo
        if model_name == "Soo"
        else CheckpointCompatibilitySpec.min
    )
    return factory(model_version="1.2.3", network_config=network)


def _model(spec: CheckpointCompatibilitySpec):
    factory = SooModel if spec.identity.model_name == "Soo" else MinModel
    return factory(spec.network_config, model_version=spec.identity.model_version)


def _checkpoint(path, spec: CheckpointCompatibilitySpec, *, delta: float = 0.0) -> None:
    trainer = AlphaZeroTrainer(_model(spec), spec, TrainingConfig(batch_size=1))
    with torch.no_grad():
        next(trainer.model.parameters()).add_(delta)
    save_checkpoint(path, trainer)


def _inference_only_checkpoint(path, spec: CheckpointCompatibilitySpec) -> None:
    torch.save(
        {
            "format_version": 1,
            "metadata": spec.to_metadata(),
            "model_state_dict": _model(spec).state_dict(),
        },
        path,
    )


def _request(model_key, *, players: int) -> InferenceRequest:
    features = players * 2
    return InferenceRequest.from_eval_request(
        client_id="worker",
        request_id=f"request-{model_key.checkpoint_sha256[:8]}",
        model_key=model_key,
        request=EvalRequest(
            node_features=tuple((0.0,) * features for _ in range(73)),
            legal_action_ids=(7, 19, 100),
            canonical_player_ids=tuple(range(1, players + 1)),
        ),
    )


def test_activation_checks_compatibility_before_residency(tmp_path) -> None:
    soo = _spec("Soo")
    min_spec = _spec("Min")
    checkpoint = tmp_path / "soo.pt"
    _checkpoint(checkpoint, soo)
    pool = InferenceModelPool()

    with pytest.raises(CheckpointCompatibilityError, match="model_name"):
        pool.activate_checkpoint(checkpoint, expected=min_spec)

    assert pool.model_keys == ()


def test_pool_keeps_two_soo_and_three_min_artifacts_resident_and_separate(
    tmp_path, monkeypatch
) -> None:
    pool = InferenceModelPool()
    soo = _spec("Soo")
    min_spec = _spec("Min")
    soo_keys = []
    min_keys = []
    for index in range(2):
        path = tmp_path / f"soo-{index}.pt"
        _checkpoint(path, soo, delta=float(index))
        soo_keys.append(pool.activate_checkpoint(path, expected=soo))
    for index in range(3):
        path = tmp_path / f"min-{index}.pt"
        _checkpoint(path, min_spec, delta=float(index))
        min_keys.append(pool.activate_checkpoint(path, expected=min_spec))

    calls: list[tuple[int, int]] = []
    original = TorchEvaluator.evaluate

    def recording_evaluate(self, requests):
        calls.append((id(self), len(requests)))
        return original(self, requests)

    monkeypatch.setattr(TorchEvaluator, "evaluate", recording_evaluate)
    requests = tuple(
        _request(key, players=2) for key in soo_keys
    ) + tuple(_request(key, players=3) for key in min_keys)

    responses = pool.evaluate(requests)

    assert len(pool.model_keys) == 5
    assert len(set(pool.model_keys)) == 5
    assert [response.model_key for response in responses] == list(soo_keys) + list(min_keys)
    assert len(calls) == 5
    assert all(batch_size == 1 for _, batch_size in calls)
    assert len({evaluator_id for evaluator_id, _ in calls}) == 5


def test_pool_loads_inference_checkpoint_without_optimizer_or_trainer_state(tmp_path) -> None:
    spec = _spec("Soo")
    checkpoint = tmp_path / "inference-only.pt"
    _inference_only_checkpoint(checkpoint, spec)
    pool = InferenceModelPool()

    key = pool.activate_checkpoint(checkpoint, expected=spec)
    response = pool.evaluate((_request(key, players=2),))[0]

    assert response.model_key == key
    assert sum(response.to_eval_result().priors.values()) == pytest.approx(1.0)


def test_pool_rejects_unknown_key_before_any_evaluator_forward(tmp_path, monkeypatch) -> None:
    spec = _spec("Soo")
    checkpoint = tmp_path / "soo.pt"
    _checkpoint(checkpoint, spec)
    pool = InferenceModelPool()
    known = pool.activate_checkpoint(checkpoint, expected=spec)
    unknown = type(known)(
        model_name=known.model_name,
        model_version=known.model_version,
        checkpoint_sha256="f" * 64,
    )
    calls = 0
    original = TorchEvaluator.evaluate

    def recording_evaluate(self, requests):
        nonlocal calls
        calls += 1
        return original(self, requests)

    monkeypatch.setattr(TorchEvaluator, "evaluate", recording_evaluate)

    with pytest.raises(KeyError, match="unknown inference model key"):
        pool.evaluate((_request(known, players=2), _request(unknown, players=2)))

    assert calls == 0
