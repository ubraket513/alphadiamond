from __future__ import annotations

import math

import pytest
import torch

from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.replay import ReplayBatch
from diamond.alphazero.trainer import AlphaZeroTrainer


def batch(player_count: int, value_target: tuple[float, ...]) -> ReplayBatch:
    features = player_count * 2
    policy = [0.0] * 5329
    policy[12] = 0.75
    policy[44] = 0.25
    return ReplayBatch(
        node_features=(tuple((0.0,) * features for _ in range(73)),),
        policy_targets=(tuple(policy),),
        value_targets=(value_target,),
    )


@pytest.mark.parametrize(
    "model,spec,value_target",
    [
        (
            SooModel(NetworkConfig(width=16, residual_blocks=1), model_version="0.2.0"),
            CheckpointCompatibilitySpec.soo(
                model_version="0.2.0",
                network_config=NetworkConfig(width=16, residual_blocks=1),
            ),
            (1.0,),
        ),
        (
            MinModel(NetworkConfig(width=16, residual_blocks=1), model_version="4.0.1"),
            CheckpointCompatibilitySpec.min(
                model_version="4.0.1",
                network_config=NetworkConfig(width=16, residual_blocks=1),
            ),
            (1.0, 0.0, -1.0),
        ),
    ],
)
def test_trainer_updates_soo_and_min_with_finite_component_losses(
    model, spec, value_target
) -> None:
    trainer = AlphaZeroTrainer(
        model,
        spec,
        TrainingConfig(batch_size=1, learning_rate=1e-3, weight_decay=0.0),
    )
    before = tuple(parameter.detach().clone() for parameter in model.parameters())

    metrics = trainer.train_batch(batch(spec.identity.player_count, value_target))

    assert trainer.training_step == 1
    assert math.isfinite(metrics.total_loss)
    assert math.isfinite(metrics.policy_loss)
    assert math.isfinite(metrics.value_loss)
    assert metrics.total_loss == pytest.approx(metrics.policy_loss + metrics.value_loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_trainer_rejects_model_identity_mismatch() -> None:
    model = SooModel(model_version="0.1.0")
    wrong = CheckpointCompatibilitySpec.soo(
        model_version="0.2.0", network_config=NetworkConfig()
    )

    with pytest.raises(ValueError, match="identity"):
        AlphaZeroTrainer(model, wrong, TrainingConfig())


def test_trainer_rejects_wrong_value_target_shape_before_optimization() -> None:
    config = NetworkConfig(width=16, residual_blocks=1)
    model = MinModel(config)
    trainer = AlphaZeroTrainer(
        model,
        CheckpointCompatibilitySpec.min(model_version="0.1.0", network_config=config),
        TrainingConfig(batch_size=1),
    )

    with pytest.raises(ValueError, match="value target"):
        trainer.train_batch(batch(3, (1.0,)))
    assert trainer.training_step == 0
