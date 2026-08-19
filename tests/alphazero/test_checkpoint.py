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


def training_batch(spec: CheckpointCompatibilitySpec) -> ReplayBatch:
    player_count = spec.identity.player_count
    features = player_count * 2
    policy = [0.0] * 5329
    policy[73] = 1.0
    value = (1.0,) if player_count == 2 else (1.0, 0.0, -1.0)
    return ReplayBatch(
        compatibility=spec,
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
    trainer.train_batch(training_batch(spec))
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
    assert restored.config == training
    assert restored.training_step == 1
    assert restored.optimizer.state_dict()["state"]
    assert restored.optimizer.param_groups[0]["lr"] == training.learning_rate
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


def test_malformed_optimizer_state_does_not_partially_mutate_model(tmp_path) -> None:
    network = NetworkConfig(width=16, residual_blocks=1)
    spec = CheckpointCompatibilitySpec.soo(model_version="0.1.0", network_config=network)
    source = AlphaZeroTrainer(SooModel(network), spec, TrainingConfig(batch_size=1))
    source.train_batch(training_batch(spec))
    path = save_checkpoint(tmp_path / "malformed-optimizer.pt", source)
    payload = torch.load(path, weights_only=True)
    payload["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    torch.save(payload, path)

    destination = AlphaZeroTrainer(
        SooModel(network), spec, TrainingConfig(batch_size=1, learning_rate=9e-3)
    )
    before = tuple(parameter.detach().clone() for parameter in destination.model.parameters())

    with pytest.raises(CheckpointError, match="state is incompatible"):
        load_checkpoint(path, destination, expected=spec)

    assert destination.training_step == 0
    assert destination.config.learning_rate == 9e-3
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, destination.model.parameters())
    )


def test_unpicklable_checkpoint_uses_checkpoint_error_contract(tmp_path) -> None:
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"not a torch checkpoint")
    network = NetworkConfig(width=16, residual_blocks=1)
    spec = CheckpointCompatibilitySpec.soo(model_version="0.1.0", network_config=network)
    trainer = AlphaZeroTrainer(SooModel(network), spec, TrainingConfig())

    with pytest.raises(CheckpointError, match="cannot read checkpoint"):
        load_checkpoint(path, trainer, expected=spec)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("device", "cuda", "device"),
        ("learning_rate", 9e-3, "optimizer learning_rate"),
    ],
)
def test_checkpoint_rejects_training_config_inconsistent_with_runtime_state(
    tmp_path, field: str, value: object, error: str
) -> None:
    network = NetworkConfig(width=16, residual_blocks=1)
    spec = CheckpointCompatibilitySpec.soo(model_version="0.1.0", network_config=network)
    source = AlphaZeroTrainer(SooModel(network), spec, TrainingConfig(batch_size=1))
    path = save_checkpoint(tmp_path / "inconsistent-config.pt", source)
    payload = torch.load(path, weights_only=True)
    payload["training_config"][field] = value
    torch.save(payload, path)
    destination = AlphaZeroTrainer(SooModel(network), spec, TrainingConfig(batch_size=1))
    before = tuple(parameter.detach().clone() for parameter in destination.model.parameters())

    with pytest.raises(CheckpointError, match=error):
        load_checkpoint(path, destination, expected=spec)

    assert destination.config == TrainingConfig(batch_size=1)
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, destination.model.parameters())
    )


def _soo_trainer(device: str = "cpu", *, trained: bool = True) -> AlphaZeroTrainer:
    """A small Soo trainer with real optimizer state, on the requested device."""
    network = NetworkConfig(width=16, residual_blocks=1)
    spec = CheckpointCompatibilitySpec.soo(model_version="0.4.0", network_config=network)
    trainer = AlphaZeroTrainer(
        SooModel(network, model_version="0.4.0"),
        spec,
        TrainingConfig(batch_size=1, learning_rate=2e-3, weight_decay=3e-4, device=device, seed=11),
    )
    if trained:
        # Optimizer state only exists after a step, and it is what must migrate.
        trainer.train_batch(training_batch(spec))
    return trainer


def test_device_migration_is_opt_in() -> None:
    """The strict device gate stays the default for every existing caller."""
    import inspect

    signature = inspect.signature(load_checkpoint)
    parameter = signature.parameters["allow_device_migration"]
    assert parameter.default is False
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_migration_relaxes_only_the_device_gate(tmp_path) -> None:
    """Semantic compatibility must still be enforced under migration."""
    trainer = _soo_trainer()
    path = save_checkpoint(tmp_path / "latest.pt", trainer)

    # A same-device load with the flag set behaves exactly like a normal load.
    destination = _soo_trainer(trained=False)
    info = load_checkpoint(
        path, destination, expected=destination.compatibility, allow_device_migration=True
    )
    assert info.training_step == trainer.training_step

    # A different model identity is still rejected, flag or no flag.
    other = AlphaZeroTrainer(
        MinModel(NetworkConfig(width=16, residual_blocks=1), model_version="0.7.0"),
        CheckpointCompatibilitySpec.min(
            model_version="0.7.0", network_config=NetworkConfig(width=16, residual_blocks=1)
        ),
        TrainingConfig(batch_size=2, learning_rate=1e-3, weight_decay=1e-4),
    )
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(
            path, other, expected=other.compatibility, allow_device_migration=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_a_cuda_trainer_needs_migration_to_load_a_cpu_checkpoint(tmp_path) -> None:
    cpu_trainer = _soo_trainer(device="cpu")
    path = save_checkpoint(tmp_path / "latest.pt", cpu_trainer)
    cuda_trainer = _soo_trainer(device="cuda:0", trained=False)

    with pytest.raises(CheckpointError, match="device"):
        load_checkpoint(path, cuda_trainer, expected=cuda_trainer.compatibility)

    info = load_checkpoint(
        path,
        cuda_trainer,
        expected=cuda_trainer.compatibility,
        allow_device_migration=True,
    )

    assert info.training_step == cpu_trainer.training_step
    assert all(p.device.type == "cuda" for p in cuda_trainer.model.parameters())
    for state in cuda_trainer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                assert value.device.type == "cuda"
