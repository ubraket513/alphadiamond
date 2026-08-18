from __future__ import annotations

import pytest

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import (
    MIN_VALUE_SEMANTICS_VERSION,
    RULESET_FINGERPRINT,
    SEAT_LAYOUT_VERSION,
    SOO_VALUE_SEMANTICS_VERSION,
    CheckpointCompatibilityError,
    CheckpointCompatibilitySpec,
    ModelIdentity,
)


def test_soo_and_min_versions_are_independent() -> None:
    soo = ModelIdentity.soo("0.1.0")
    min_model = ModelIdentity.min("8.4.2")

    assert soo.model_name == "Soo"
    assert soo.model_version == "0.1.0"
    assert soo.player_count == 2
    assert soo.value_semantics_version == SOO_VALUE_SEMANTICS_VERSION
    assert min_model.model_name == "Min"
    assert min_model.model_version == "8.4.2"
    assert min_model.player_count == 3
    assert min_model.value_semantics_version == MIN_VALUE_SEMANTICS_VERSION


def test_checkpoint_metadata_contains_every_compatibility_gate() -> None:
    spec = CheckpointCompatibilitySpec.soo(
        model_version="0.1.0",
        network_config=NetworkConfig(width=64, residual_blocks=3),
    )

    assert spec.to_metadata() == {
        "model_name": "Soo",
        "model_version": "0.1.0",
        "player_count": 2,
        "ruleset_version": "diamond-authoritative-rules-v1",
        "board_topology_version": "diamond73-v1",
        "ruleset_fingerprint": RULESET_FINGERPRINT,
        "encoder_version": "diamond-camp-relative-v1",
        "action_space_version": "diamond73-srcdst-v1",
        "seat_layout_version": SEAT_LAYOUT_VERSION,
        "value_semantics_version": "current-player-scalar-winloss-v1",
        "network_config": {"width": 64, "residual_blocks": 3},
    }


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("model_name", "Min"),
        ("model_version", "0.1.1"),
        ("player_count", 3),
        ("ruleset_version", "other-rules"),
        ("board_topology_version", "diamond121-v1"),
        ("ruleset_fingerprint", "sha256:different"),
        ("encoder_version", "different-encoder"),
        ("action_space_version", "different-actions"),
        ("seat_layout_version", "different-seats"),
        ("value_semantics_version", MIN_VALUE_SEMANTICS_VERSION),
        ("network_config", {"width": 128, "residual_blocks": 3}),
    ],
)
def test_same_model_version_never_bypasses_other_compatibility_gates(
    field: str, bad_value: object
) -> None:
    expected = CheckpointCompatibilitySpec.soo(
        model_version="1.4.0",
        network_config=NetworkConfig(width=64, residual_blocks=3),
    )
    metadata = expected.to_metadata()
    metadata[field] = bad_value

    with pytest.raises(CheckpointCompatibilityError, match=field):
        expected.assert_compatible(metadata)


def test_missing_checkpoint_metadata_is_rejected() -> None:
    expected = CheckpointCompatibilitySpec.min(
        model_version="0.1.0", network_config=NetworkConfig()
    )
    metadata = expected.to_metadata()
    del metadata["encoder_version"]

    with pytest.raises(CheckpointCompatibilityError, match="encoder_version"):
        expected.assert_compatible(metadata)


def test_ruleset_fingerprint_is_a_stable_sha256_identity() -> None:
    assert RULESET_FINGERPRINT.startswith("sha256:")
    assert len(RULESET_FINGERPRINT) == len("sha256:") + 64


def test_model_version_accepts_full_semver_prerelease_and_build_metadata() -> None:
    identity = ModelIdentity.soo("1.0.0-rc.1+build.5")
    assert identity.model_version == "1.0.0-rc.1+build.5"


@pytest.mark.parametrize("version", ["1.0.0-01", "1.0.0+", "01.0.0"])
def test_model_version_rejects_invalid_semver(version: str) -> None:
    with pytest.raises(ValueError, match="semantic version"):
        ModelIdentity.min(version)
