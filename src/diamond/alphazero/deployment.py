"""Versioned Soo deployment-artifact metadata and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

ARTIFACT_FORMAT_VERSION = 2
EXPECTED_METADATA_KEYS = frozenset(
    {
        "format_version",
        "model_name",
        "model_version",
        "input_shape",
        "policy_shape",
        "value_shape",
        "dtype",
        "width",
        "residual_blocks",
        "board_topology_version",
        "encoder_version",
        "action_space_version",
        "corpus_seed",
        "model_sha256",
        "runtime_sha256",
        "checkpoint_sha256",
    }
)


class DeploymentArtifactError(ValueError):
    """The deployment artifact is malformed or incompatible."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a plain metadata mapping.

    The spike intentionally rejects both missing and unknown fields. Adding a
    field therefore requires an explicit format-version decision instead of
    silently producing an artifact an older native runtime may misread.
    """
    actual_keys = frozenset(metadata)
    missing = EXPECTED_METADATA_KEYS - actual_keys
    extra = actual_keys - EXPECTED_METADATA_KEYS
    if missing:
        raise DeploymentArtifactError(f"metadata is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise DeploymentArtifactError(f"metadata has unknown fields: {', '.join(sorted(extra))}")

    expected = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_name": "Soo",
        "model_version": "0.1.0",
        "input_shape": [2, 73, 4],
        "policy_shape": [2, 5329],
        "value_shape": [2, 1],
        "dtype": "float32",
        "width": 128,
        "residual_blocks": 6,
        "board_topology_version": "diamond73-v1",
        "encoder_version": "diamond-camp-relative-v1",
        "action_space_version": "diamond73-srcdst-v1",
    }
    for field, value in expected.items():
        if metadata[field] != value:
            raise DeploymentArtifactError(
                f"metadata {field} mismatch: expected {value!r}, got {metadata[field]!r}"
            )
    if not isinstance(metadata["corpus_seed"], int) or isinstance(metadata["corpus_seed"], bool):
        raise DeploymentArtifactError("metadata corpus_seed must be an integer")
    if not _is_sha256(metadata["model_sha256"]):
        raise DeploymentArtifactError("metadata model_sha256 must be a SHA-256 hex digest")
    if not _is_sha256(metadata["runtime_sha256"]):
        raise DeploymentArtifactError("metadata runtime_sha256 must be a SHA-256 hex digest")
    checkpoint_sha256 = metadata["checkpoint_sha256"]
    if checkpoint_sha256 is not None and not _is_sha256(checkpoint_sha256):
        raise DeploymentArtifactError("metadata checkpoint_sha256 must be null or a SHA-256 digest")
    return dict(metadata)


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentArtifactError(f"cannot read deployment metadata: {exc}") from exc
    if not isinstance(payload, dict):
        raise DeploymentArtifactError("deployment metadata must be a JSON object")
    return validate_metadata(payload)


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "DeploymentArtifactError",
    "EXPECTED_METADATA_KEYS",
    "load_metadata",
    "validate_metadata",
]
