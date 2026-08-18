from __future__ import annotations

import pytest
import torch

from diamond.alphazero.checkpoint import (
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilityError, CheckpointCompatibilitySpec
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.replay import ReplayBatch
from diamond.alphazero.trainer import AlphaZeroTrainer


def training_batch(player_count: int) -> ReplayBatch:
    features = player_count * 2
    policy = [0.0] * 5329
    policy[73] = 1.0
    value = (1.0,) if player_count == 2 else (1.0, 0.0, -1.0)
    return ReplayBatch(
        node_features=(tuple((0.0,) * features for _ in range(73)),),
        policy_targets=(tuple(policy),),
        value_targets=(value,),
    )


@pytest.mark.parametrize("model_kind", ["Soo", "Min"])
def test_checkpoint_round_trips_model_optimizer_step_and_config(tmp_path, model_kind: str) -> None:
    network = NetworkConfig(width=16, residual_blocks=1)
    training = TrainingConfig(
        batch_size=1, learning_rate=2e-3, weight_decay=3e-4, seed=11
    )
    if model_kind == "Soo":
        model = SooModel(network, model_version="0.4.0")
        spec = CheckpointCompatibilitySpec.soo(model_version="0.4.0", network_config=network)
    else:
        model = MinModel(network, model_version="7.1.2")
        spec = CheckpointCompatibilitySpec.min(model_version="7.1.2", network_config=network)
    trainer = AlphaZeroTrainer(model, spec, training)
    trainer.train_batch(training_batch(spec.identity.player_count))
    expected_parameters = tuple(parameter.detach().clone() for parameter in model.parameters())

    path = save_checkpoint(tmp_path / f"{model_kind}.pt", trainer)

    restored_model = (
        SooModel(network, model_version="0.4.0")
        if model_kind == "Soo"
        else MinModel(network, model_version="7.1.2")
    )
    restored = AlphaZeroTrainer(restored_model, spec, TrainingConfig(batch_size=1))
    info = load_checkpoint(path, restored, expected=spec)

    assert info.training_step == 1
    assert info.training_config == training
    assert restored.training_step == 1
    assert restored.optimizer.state_dict()["state"]
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(expected_parameters, restored_model.parameters())
    )


def test_soo_checkpoint_is_rejected_by_min_before_model_state_changes(tmp_path) -> None:
    network = NetworkConfig(width=16, residual_blocks=1)
    soo_spec = CheckpointCompatibilitySpec.soo(
        model_version="0.1.0", network_config=network
    )
    path = save_checkpoint(
        tmp_path / "soo.pt",
        AlphaZeroTrainer(SooModel(network), soo_spec, TrainingConfig(batch_size=1)),
    )
    min_spec = CheckpointCompatibilitySpec.min(
        model_version="0.1.0", network_config=network
    )
    min_model = MinModel(network)
    before = tuple(parameter.detach().clone() for parameter in min_model.parameters())
    min_trainer = AlphaZeroTrainer(min_model, min_spec, TrainingConfig(batch_size=1))

    with pytest.raises(CheckpointCompatibilityError, match="model_name"):
        load_checkpoint(path, min_trainer, expected=min_spec)

    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, min_model.parameters())
    )


def test_checkpoint_rejects_changed_encoder_even_with_same_min_version(tmp_path) -> None:
    network = NetworkConfig(width=16, residual_blocks=1)
    spec = CheckpointCompatibilitySpec.min(model_version="1.4.0", network_config=network)
    path = save_checkpoint(
        tmp_path / "min.pt",
        AlphaZeroTrainer(MinModel(network, model_version="1.4.0"), spec, TrainingConfig()),
    )
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["encoder_version"] = "different-encoder"
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match="encoder_version"):
        load_checkpoint(
            path,
            AlphaZeroTrainer(
                MinModel(network, model_version="1.4.0"), spec, TrainingConfig()
            ),
            expected=spec,
        )


def test_checkpoint_rejects_missing_or_corrupted_payload(tmp_path) -> None:
    path = tmp_path / "broken.pt"
    torch.save({"format_version": 1, "metadata": {}}, path)
    network = NetworkConfig(width=16, residual_blocks=1)
    spec = CheckpointCompatibilitySpec.soo(model_version="0.1.0", network_config=network)
    trainer = AlphaZeroTrainer(SooModel(network), spec, TrainingConfig())

    with pytest.raises(CheckpointError):
        load_checkpoint(path, trainer, expected=spec)
