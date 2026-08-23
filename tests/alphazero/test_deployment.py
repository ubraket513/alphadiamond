import json
from pathlib import Path

import pytest

from diamond.alphazero.deployment import DeploymentArtifactError, load_metadata, validate_metadata


def _metadata() -> dict:
    return {
        "format_version": 2,
        "model_name": "Soo",
        "model_version": "2.0.0",
        "input_shape": [2, 73, 4],
        "policy_shape": [2, 5329],
        "value_shape": [2, 1],
        "dtype": "float32",
        "width": 128,
        "residual_blocks": 6,
        "board_topology_version": "diamond73-v1",
        "encoder_version": "diamond-camp-relative-v1",
        "action_space_version": "diamond73-srcdst-v1",
        "corpus_seed": 20260823,
        "model_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "checkpoint_sha256": None,
    }


def test_valid_metadata_round_trips(tmp_path: Path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(_metadata()), encoding="utf-8")
    assert load_metadata(path)["model_name"] == "Soo"


@pytest.mark.parametrize("field", ["width", "model_name", "input_shape"])
def test_metadata_mismatch_is_rejected(field: str):
    payload = _metadata()
    payload[field] = 999
    with pytest.raises(DeploymentArtifactError, match="mismatch"):
        validate_metadata(payload)


def test_missing_and_unknown_fields_are_rejected():
    missing = _metadata()
    del missing["model_sha256"]
    with pytest.raises(DeploymentArtifactError, match="missing fields"):
        validate_metadata(missing)

    extra = _metadata()
    extra["unexpected_tensor"] = "tensor.bin"
    with pytest.raises(DeploymentArtifactError, match="unknown fields"):
        validate_metadata(extra)


@pytest.mark.parametrize("field", ["model_sha256", "runtime_sha256", "checkpoint_sha256"])
def test_non_hex_digest_is_rejected(field: str):
    payload = _metadata()
    payload[field] = "z" * 64
    with pytest.raises(DeploymentArtifactError, match="SHA-256"):
        validate_metadata(payload)
