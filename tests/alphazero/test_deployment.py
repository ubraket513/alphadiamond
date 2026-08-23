import json
from pathlib import Path

import pytest

from diamond.alphazero.deployment import (
    ARTIFACT_FORMAT_VERSION,
    DeploymentArtifactError,
    load_metadata,
    validate_metadata,
)


def _metadata(family: str = "soo") -> dict:
    features, value_size = {"soo": (4, 1), "min": (6, 3)}[family]
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_family": family,
        "model_version": "2.0.0" if family == "soo" else "0.1.0",
        "architecture": {
            "type": "directional_residual",
            "width": 128,
            "residual_blocks": 6,
        },
        "game_contract": {
            "topology": "diamond73-v1",
            "encoder": "diamond-camp-relative-v1",
            "action_space": "diamond73-srcdst-v1",
        },
        "tensor_shapes": {
            "input": [2, 73, features],
            "policy": [2, 5329],
            "value": [2, value_size],
        },
        "dtype": "float32",
        "source": {
            "checkpoint_sha256": None,
            "training_commit": "c" * 40,
            "training_step": 44250,
        },
        "corpus_seed": 20260823,
        "model_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
    }


def test_valid_metadata_round_trips(tmp_path: Path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(_metadata()), encoding="utf-8")
    assert load_metadata(path)["model_family"] == "soo"


def test_both_families_are_describable():
    """The point of format 3: Min needs no relaxation of the Soo contract."""
    assert validate_metadata(_metadata("soo"))["tensor_shapes"]["value"] == [2, 1]
    assert validate_metadata(_metadata("min"))["tensor_shapes"]["value"] == [2, 3]


def test_a_family_declared_over_the_wrong_shapes_is_rejected():
    payload = _metadata("soo")
    payload["model_family"] = "min"
    with pytest.raises(DeploymentArtifactError, match="tensor_shapes"):
        validate_metadata(payload)


def test_unknown_family_is_rejected():
    payload = _metadata()
    payload["model_family"] = "moo"
    with pytest.raises(DeploymentArtifactError, match="model_family"):
        validate_metadata(payload)


def test_older_format_version_is_rejected():
    payload = _metadata()
    payload["format_version"] = 2
    with pytest.raises(DeploymentArtifactError, match="format_version"):
        validate_metadata(payload)


def test_foreign_game_contract_is_rejected():
    """A different contract is a different game, not a different model."""
    payload = _metadata()
    payload["game_contract"]["encoder"] = "diamond-absolute-v1"
    with pytest.raises(DeploymentArtifactError, match="game_contract"):
        validate_metadata(payload)


@pytest.mark.parametrize("field", ["width", "residual_blocks"])
def test_non_positive_architecture_is_rejected(field: str):
    payload = _metadata()
    payload["architecture"][field] = 0
    with pytest.raises(DeploymentArtifactError, match="architecture"):
        validate_metadata(payload)


def test_architecture_is_declared_not_fixed():
    """A narrower model is describable; only the *weights* pin the numbers."""
    payload = _metadata()
    payload["architecture"]["width"] = 64
    payload["architecture"]["residual_blocks"] = 4
    assert validate_metadata(payload)["architecture"]["width"] == 64


def test_missing_and_unknown_fields_are_rejected():
    missing = _metadata()
    del missing["model_sha256"]
    with pytest.raises(DeploymentArtifactError, match="missing fields"):
        validate_metadata(missing)

    extra = _metadata()
    extra["surprise"] = True
    with pytest.raises(DeploymentArtifactError, match="unknown fields"):
        validate_metadata(extra)

    nested = _metadata()
    nested["architecture"]["surprise"] = True
    with pytest.raises(DeploymentArtifactError, match="unknown fields"):
        validate_metadata(nested)


@pytest.mark.parametrize("field", ["model_sha256", "runtime_sha256"])
def test_digests_must_be_sha256(field: str):
    payload = _metadata()
    payload[field] = "not-a-digest"
    with pytest.raises(DeploymentArtifactError, match="SHA-256"):
        validate_metadata(payload)


def test_source_provenance_is_validated():
    payload = _metadata()
    payload["source"]["training_commit"] = "abc"
    with pytest.raises(DeploymentArtifactError, match="training_commit"):
        validate_metadata(payload)

    payload = _metadata()
    payload["source"]["training_step"] = -1
    with pytest.raises(DeploymentArtifactError, match="training_step"):
        validate_metadata(payload)

    payload = _metadata()
    payload["source"]["checkpoint_sha256"] = "d" * 63
    with pytest.raises(DeploymentArtifactError, match="checkpoint_sha256"):
        validate_metadata(payload)


def test_unexported_provenance_may_be_null():
    """An artifact exported from an untrained model has no checkpoint."""
    payload = _metadata()
    payload["source"] = {
        "checkpoint_sha256": None,
        "training_commit": None,
        "training_step": None,
    }
    assert validate_metadata(payload)["source"]["training_step"] is None


def test_malformed_json_is_rejected(tmp_path: Path):
    path = tmp_path / "metadata.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DeploymentArtifactError, match="JSON object"):
        load_metadata(path)
