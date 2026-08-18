from __future__ import annotations

import pytest

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.replay import ReplayBuffer, ReplayCompatibilityError, TrainingSample


def sample(spec: CheckpointCompatibilitySpec, marker: int) -> TrainingSample:
    player_count = spec.identity.player_count
    return TrainingSample(
        compatibility=spec,
        node_features=((float(marker),) * (player_count * 2),) * 73,
        canonical_player_ids=tuple(range(1, player_count + 1)),
        sparse_policy=((marker, 0.75), (marker + 1, 0.25)),
        value_target=(1.0,) if player_count == 2 else (1.0, 0.0, -1.0),
    )


def test_replay_capacity_evicts_oldest_sample() -> None:
    spec = CheckpointCompatibilitySpec.soo(
        model_version="0.1.0", network_config=NetworkConfig()
    )
    replay = ReplayBuffer(spec, capacity=2, seed=3)
    replay.extend((sample(spec, 1), sample(spec, 2), sample(spec, 3)))

    assert len(replay) == 2
    assert {row.sparse_policy[0][0] for row in replay.samples} == {2, 3}


def test_replay_sampling_is_seeded_and_reproducible() -> None:
    spec = CheckpointCompatibilitySpec.min(
        model_version="2.0.0", network_config=NetworkConfig()
    )
    rows = tuple(sample(spec, marker) for marker in range(10))
    first = ReplayBuffer(spec, capacity=20, seed=99)
    second = ReplayBuffer(spec, capacity=20, seed=99)
    first.extend(rows)
    second.extend(rows)

    assert first.sample(4) == second.sample(4)


def test_replay_rejects_soo_min_and_version_schema_mixing() -> None:
    soo = CheckpointCompatibilitySpec.soo(
        model_version="1.4.0", network_config=NetworkConfig()
    )
    min_model = CheckpointCompatibilitySpec.min(
        model_version="1.4.0", network_config=NetworkConfig()
    )
    replay = ReplayBuffer(soo, capacity=4)

    with pytest.raises(ReplayCompatibilityError):
        replay.add(sample(min_model, 1))

    changed_encoder = CheckpointCompatibilitySpec(
        identity=soo.identity,
        network_config=soo.network_config,
        encoder_version="other-encoder",
    )
    with pytest.raises(ReplayCompatibilityError):
        replay.add(sample(changed_encoder, 1))


def test_collation_constructs_dense_policy_only_for_the_batch() -> None:
    spec = CheckpointCompatibilitySpec.soo(
        model_version="0.1.0", network_config=NetworkConfig()
    )
    replay = ReplayBuffer(spec, capacity=4)
    replay.extend((sample(spec, 4), sample(spec, 8)))

    batch = replay.collate(replay.samples, action_size=16)

    assert len(batch.node_features) == 2
    assert batch.policy_targets[0][4:6] == (0.75, 0.25)
    assert sum(batch.policy_targets[0]) == pytest.approx(1.0)
    assert batch.value_targets == ((1.0,), (1.0,))
