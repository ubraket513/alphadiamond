"""`train_samples` must be the same training step as `train_batch`.

The fast path skips the dense Python policy row that `ReplayBatch` carries. It
is only a fast path if it is also the *same* path: same gradients, same loss,
same validation. This pins that, because a silently different policy target
would train a different network and nothing else would notice.
"""

from __future__ import annotations

import random

import pytest
import torch

from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network.soo import SooModel
from diamond.alphazero.replay import ReplayBuffer, TrainingSample
from diamond.alphazero.trainer import AlphaZeroTrainer

ACTION_SIZE = 73 * 73


NETWORK = NetworkConfig(width=16, residual_blocks=1)


def _compatibility() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.soo(model_version="2.0.0", network_config=NETWORK)


def _samples(count: int = 4) -> list[TrainingSample]:
    compatibility = _compatibility()
    rng = random.Random(20260824)
    samples = []
    for _ in range(count):
        actions = sorted(rng.sample(range(ACTION_SIZE), 6))
        weights = [rng.random() + 1e-3 for _ in actions]
        total = sum(weights)
        samples.append(
            TrainingSample(
                compatibility=compatibility,
                node_features=tuple(
                    tuple(float(rng.getrandbits(1)) for _ in range(4)) for _ in range(73)
                ),
                canonical_player_ids=(1, 2),
                sparse_policy=tuple(
                    (action, weight / total) for action, weight in zip(actions, weights)
                ),
                value_target=(1.0 if rng.getrandbits(1) else -1.0,),
            )
        )
    return samples


def _trainer() -> AlphaZeroTrainer:
    torch.manual_seed(7)
    compatibility = _compatibility()
    model = SooModel(NETWORK, model_version="2.0.0")
    return AlphaZeroTrainer(model, compatibility, TrainingConfig(batch_size=4))


def test_both_paths_produce_the_same_step():
    samples = _samples()
    buffer = ReplayBuffer(_compatibility(), capacity=len(samples))
    buffer.extend(samples)

    dense = _trainer().train_batch(buffer.collate(samples, action_size=ACTION_SIZE))
    sparse = _trainer().train_samples(samples, action_size=ACTION_SIZE)

    assert sparse.total_loss == pytest.approx(dense.total_loss, rel=1e-6, abs=1e-6)
    assert sparse.policy_loss == pytest.approx(dense.policy_loss, rel=1e-6, abs=1e-6)
    assert sparse.value_loss == pytest.approx(dense.value_loss, rel=1e-6, abs=1e-6)


def test_the_fast_path_keeps_the_validation():
    trainer = _trainer()
    with pytest.raises(ValueError, match="must not be empty"):
        trainer.train_samples([], action_size=ACTION_SIZE)

    # An action outside the declared space would otherwise scatter out of bounds.
    with pytest.raises(ValueError, match="action space"):
        trainer.train_samples(_samples(1), action_size=16)


def test_incompatible_samples_are_rejected():
    other = CheckpointCompatibilitySpec.soo(model_version="1.0.0", network_config=NETWORK)
    sample = _samples(1)[0]
    mismatched = TrainingSample(
        compatibility=other,
        node_features=sample.node_features,
        canonical_player_ids=sample.canonical_player_ids,
        sparse_policy=sample.sparse_policy,
        value_target=sample.value_target,
    )
    with pytest.raises(ValueError, match="compatibility"):
        _trainer().train_samples([mismatched], action_size=ACTION_SIZE)
