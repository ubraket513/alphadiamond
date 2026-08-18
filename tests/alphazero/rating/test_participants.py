from __future__ import annotations

import hashlib

import pytest
import torch

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.rating.participants import CheckpointParticipant


def _write_checkpoint(
    path,
    *,
    model_version: str = "1.2.3",
    training_step: int = 42,
) -> None:
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version=model_version,
        network_config=NetworkConfig(width=16, residual_blocks=1),
    )
    torch.save(
        {
            "format_version": 1,
            "metadata": compatibility.to_metadata(),
            "training_config": {},
            "training_step": training_step,
            "model_state_dict": {},
            "optimizer_state_dict": {},
        },
        path,
    )


def test_checkpoint_participant_binds_raw_content_hash_and_checkpoint_metadata(tmp_path) -> None:
    path = tmp_path / "soo.pt"
    _write_checkpoint(path)

    participant = CheckpointParticipant.from_checkpoint(path)

    assert participant.model_name == "Soo"
    assert participant.model_version == "1.2.3"
    assert participant.training_step == 42
    assert participant.checkpoint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert participant.compatibility_metadata == {
        "model_name": "Soo",
        "model_version": "1.2.3",
        "player_count": 2,
        "ruleset_version": "diamond-authoritative-rules-v1",
        "board_topology_version": "diamond73-v1",
        "ruleset_fingerprint": "sha256:02fff0c9c9436f247c4a2b5fb6b01903f658aae1c752377073011d0d150ba7a1",
        "encoder_version": "diamond-camp-relative-v1",
        "action_space_version": "diamond73-srcdst-v1",
        "seat_layout_version": "diamond-seat-layout-v1",
        "value_semantics_version": "current-player-scalar-winloss-v1",
        "network_config": {"width": 16, "residual_blocks": 1},
    }
    assert participant.display_name == "Soo 1.2.3 @ 42"


def test_checkpoint_participant_id_is_stable_for_the_same_artifact(tmp_path) -> None:
    path = tmp_path / "soo.pt"
    _write_checkpoint(path)

    first = CheckpointParticipant.from_checkpoint(path)
    second = CheckpointParticipant.from_checkpoint(path)

    assert first.participant_id == second.participant_id
    assert first.participant_id == (
        f"checkpoint:Soo:1.2.3:42:{first.checkpoint_sha256}"
    )


def test_checkpoint_participant_distinguishes_steps_within_one_model_version(tmp_path) -> None:
    first_path = tmp_path / "soo-step-42.pt"
    second_path = tmp_path / "soo-step-43.pt"
    _write_checkpoint(first_path, training_step=42)
    _write_checkpoint(second_path, training_step=43)

    first = CheckpointParticipant.from_checkpoint(first_path)
    second = CheckpointParticipant.from_checkpoint(second_path)

    assert first.model_version == second.model_version == "1.2.3"
    assert first.training_step != second.training_step
    assert first.participant_id != second.participant_id


def test_checkpoint_participant_compatibility_metadata_is_immutable(tmp_path) -> None:
    path = tmp_path / "soo.pt"
    _write_checkpoint(path)

    participant = CheckpointParticipant.from_checkpoint(path)

    with pytest.raises(TypeError):
        participant.compatibility_metadata["network_config"]["width"] = 32


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_version", "9.9.9"),
        ("checkpoint_sha256", "0" * 64),
    ],
)
def test_checkpoint_participant_rejects_metadata_or_hash_collisions(
    tmp_path, field: str, value: str
) -> None:
    path = tmp_path / "soo.pt"
    _write_checkpoint(path)
    participant = CheckpointParticipant.from_checkpoint(path)
    values = {
        "participant_id": participant.participant_id,
        "model_name": participant.model_name,
        "model_version": participant.model_version,
        "training_step": participant.training_step,
        "checkpoint_sha256": participant.checkpoint_sha256,
        "compatibility_metadata": participant.compatibility_metadata,
        "display_name": participant.display_name,
    }
    values[field] = value

    with pytest.raises(ValueError, match="participant_id"):
        CheckpointParticipant(**values)
