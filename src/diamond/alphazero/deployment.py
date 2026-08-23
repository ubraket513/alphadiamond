"""Versioned deployment-artifact metadata and validation.

Format 3 differs from 2 in one way that matters: the strictness moved from
hardcoded Soo constants to a *declared* architecture. v2 asserted width 128 and
six residual blocks, so a second model family could not be described at all
without changing the validator. v3 declares `model_family` and `architecture`,
and the loader checks the weights against what the artifact declares -- which is
what makes adding Min a model port rather than a format redesign.

The contract still rejects both missing and unknown fields. Adding one is a
format-version decision, not a silent change an older native runtime may
misread.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARTIFACT_FORMAT_VERSION = 3

FAMILY_TENSOR_SHAPES = {
    # (input features per hole, value head size). Soo is the two-player model
    # with a scalar value; Min is three-player and predicts one value per seat.
    "soo": (4, 1),
    "min": (6, 3),
}

MODEL_FAMILIES = frozenset(FAMILY_TENSOR_SHAPES)

EXPECTED_METADATA_KEYS = frozenset(
    {
        "format_version",
        "model_family",
        "model_version",
        "architecture",
        "game_contract",
        "tensor_shapes",
        "dtype",
        "source",
        "corpus_seed",
        "model_sha256",
        "runtime_sha256",
    }
)

EXPECTED_ARCHITECTURE_KEYS = frozenset({"type", "width", "residual_blocks"})
EXPECTED_GAME_CONTRACT_KEYS = frozenset({"topology", "encoder", "action_space"})
EXPECTED_TENSOR_SHAPE_KEYS = frozenset({"input", "policy", "value"})
EXPECTED_SOURCE_KEYS = frozenset({"checkpoint_sha256", "training_commit", "training_step"})

GAME_CONTRACT = {
    "topology": "diamond73-v1",
    "encoder": "diamond-camp-relative-v1",
    "action_space": "diamond73-srcdst-v1",
}
"""What this binary implements. An artifact declaring anything else is for a
different game, not a different model, and must be refused."""


class DeploymentArtifactError(ValueError):
    """The deployment artifact is malformed or incompatible."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_keys(name: str, mapping: object, expected: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise DeploymentArtifactError(f"metadata {name} must be an object")
    actual = frozenset(mapping)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise DeploymentArtifactError(f"metadata {name} is missing fields: {sorted(missing)}")
    if extra:
        raise DeploymentArtifactError(f"metadata {name} has unknown fields: {sorted(extra)}")
    return mapping


def _validate_shape(name: str, value: object, suffix: list[int]) -> None:
    if not isinstance(value, list) or len(value) != len(suffix) + 1:
        raise DeploymentArtifactError(f"metadata tensor_shapes {name} has the wrong rank")
    if not all(_is_integer(entry) and entry > 0 for entry in value):
        raise DeploymentArtifactError(f"metadata tensor_shapes {name} must be positive integers")
    if list(value[1:]) != suffix:
        raise DeploymentArtifactError(
            f"metadata tensor_shapes {name} mismatch: expected [batch, *{suffix}], got {value}"
        )


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a plain metadata mapping."""
    _require_keys("metadata", metadata, EXPECTED_METADATA_KEYS)

    if metadata["format_version"] != ARTIFACT_FORMAT_VERSION:
        raise DeploymentArtifactError(
            f"metadata format_version mismatch: expected {ARTIFACT_FORMAT_VERSION}, "
            f"got {metadata['format_version']!r}"
        )
    if metadata["model_family"] not in MODEL_FAMILIES:
        raise DeploymentArtifactError(
            f"metadata model_family mismatch: expected one of {sorted(MODEL_FAMILIES)}, "
            f"got {metadata['model_family']!r}"
        )
    if not isinstance(metadata["model_version"], str) or not metadata["model_version"]:
        raise DeploymentArtifactError("metadata model_version must be a non-empty string")
    if metadata["dtype"] != "float32":
        raise DeploymentArtifactError(
            f"metadata dtype mismatch: expected 'float32', got {metadata['dtype']!r}"
        )

    architecture = _require_keys(
        "architecture", metadata["architecture"], EXPECTED_ARCHITECTURE_KEYS
    )
    if not isinstance(architecture["type"], str) or not architecture["type"]:
        raise DeploymentArtifactError("metadata architecture type must be a non-empty string")
    for field in ("width", "residual_blocks"):
        if not _is_integer(architecture[field]) or architecture[field] <= 0:
            raise DeploymentArtifactError(
                f"metadata architecture {field} must be a positive integer"
            )

    contract = _require_keys("game_contract", metadata["game_contract"], EXPECTED_GAME_CONTRACT_KEYS)
    for field, value in GAME_CONTRACT.items():
        if contract[field] != value:
            raise DeploymentArtifactError(
                f"metadata game_contract {field} mismatch: expected {value!r}, "
                f"got {contract[field]!r}"
            )

    features, value_size = FAMILY_TENSOR_SHAPES[metadata["model_family"]]
    shapes = _require_keys("tensor_shapes", metadata["tensor_shapes"], EXPECTED_TENSOR_SHAPE_KEYS)
    _validate_shape("input", shapes["input"], [73, features])
    _validate_shape("policy", shapes["policy"], [5329])
    _validate_shape("value", shapes["value"], [value_size])
    batches = {shapes[name][0] for name in ("input", "policy", "value")}
    if len(batches) != 1:
        raise DeploymentArtifactError("metadata tensor_shapes disagree about the batch size")

    source = _require_keys("source", metadata["source"], EXPECTED_SOURCE_KEYS)
    checkpoint_sha256 = source["checkpoint_sha256"]
    if checkpoint_sha256 is not None and not _is_sha256(checkpoint_sha256):
        raise DeploymentArtifactError(
            "metadata source checkpoint_sha256 must be null or a SHA-256 digest"
        )
    commit = source["training_commit"]
    if commit is not None and (not isinstance(commit, str) or len(commit) != 40):
        raise DeploymentArtifactError(
            "metadata source training_commit must be null or a 40-character commit id"
        )
    step = source["training_step"]
    if step is not None and (not _is_integer(step) or step < 0):
        raise DeploymentArtifactError(
            "metadata source training_step must be null or a non-negative integer"
        )

    if not _is_integer(metadata["corpus_seed"]):
        raise DeploymentArtifactError("metadata corpus_seed must be an integer")
    if not _is_sha256(metadata["model_sha256"]):
        raise DeploymentArtifactError("metadata model_sha256 must be a SHA-256 hex digest")
    if not _is_sha256(metadata["runtime_sha256"]):
        raise DeploymentArtifactError("metadata runtime_sha256 must be a SHA-256 hex digest")
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
    "EXPECTED_METADATA_KEYS",
    "FAMILY_TENSOR_SHAPES",
    "GAME_CONTRACT",
    "MODEL_FAMILIES",
    "DeploymentArtifactError",
    "load_metadata",
    "validate_metadata",
]
