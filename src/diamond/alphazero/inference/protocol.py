"""Strict, pickle-safe transport envelopes for centralized inference."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..evaluator.base import EvalRequest, EvalResult
from ..identity import MIN_MODEL_NAME, SOO_MODEL_NAME, ModelIdentity

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_exact_keys(payload: Mapping[str, object], expected: set[str]) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        parts = []
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(f"malformed inference payload ({'; '.join(parts)})")


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _finite_float(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _tuple(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field} must be a tuple")
    return value


def _tuple_from_payload(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ModelKey:
    """Identity for one immutable checkpoint artifact, never merely a version."""

    model_name: str
    model_version: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        _non_empty_string(self.model_name, "model_name")
        _non_empty_string(self.model_version, "model_version")
        if not _SHA256.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if self.model_name == SOO_MODEL_NAME:
            ModelIdentity.soo(self.model_version)
        elif self.model_name == MIN_MODEL_NAME:
            ModelIdentity.min(self.model_version)
        else:
            raise ValueError(f"unknown AlphaZero model name: {self.model_name}")

    @property
    def player_count(self) -> int:
        return 2 if self.model_name == SOO_MODEL_NAME else 3

    def to_payload(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ModelKey":
        if not isinstance(payload, Mapping):
            raise ValueError("model key payload must be a mapping")
        _require_exact_keys(
            payload, {"model_name", "model_version", "checkpoint_sha256"}
        )
        return cls(
            model_name=_non_empty_string(payload["model_name"], "model_name"),
            model_version=_non_empty_string(payload["model_version"], "model_version"),
            checkpoint_sha256=_non_empty_string(
                payload["checkpoint_sha256"], "checkpoint_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    client_id: str
    request_id: str
    model_key: ModelKey
    node_features: tuple[tuple[float, ...], ...]
    legal_action_ids: tuple[int, ...]
    canonical_player_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.client_id, "client_id")
        _non_empty_string(self.request_id, "request_id")
        if not isinstance(self.model_key, ModelKey):
            raise ValueError("model_key must be a ModelKey")
        features = _tuple(self.node_features, "node_features")
        if not features:
            raise ValueError("node_features must not be empty")
        feature_width: int | None = None
        for row in features:
            row_values = _tuple(row, "node_features rows")
            if not row_values:
                raise ValueError("node_features rows must not be empty")
            if feature_width is None:
                feature_width = len(row_values)
            elif len(row_values) != feature_width:
                raise ValueError("node_features rows must have the same width")
            for value in row_values:
                _finite_float(value, "node_features values")
        legal_actions = _tuple(self.legal_action_ids, "legal_action_ids")
        if not legal_actions:
            raise ValueError("legal_action_ids must not be empty")
        if len({_integer(action, "legal_action_ids values") for action in legal_actions}) != len(
            legal_actions
        ):
            raise ValueError("legal_action_ids must be unique")
        players = _tuple(self.canonical_player_ids, "canonical_player_ids")
        if len(players) != self.model_key.player_count:
            raise ValueError("canonical_player_ids does not match the model player count")
        player_values = tuple(
            _integer(player, "canonical_player_ids values") for player in players
        )
        if any(player <= 0 for player in player_values) or len(set(player_values)) != len(player_values):
            raise ValueError("canonical_player_ids must be distinct positive integers")

    @property
    def correlation_id(self) -> tuple[str, str]:
        return (self.client_id, self.request_id)

    @classmethod
    def from_eval_request(
        cls, *, client_id: str, request_id: str, model_key: ModelKey, request: EvalRequest
    ) -> "InferenceRequest":
        if not isinstance(request, EvalRequest):
            raise ValueError("request must be an EvalRequest")
        return cls(
            client_id=client_id,
            request_id=request_id,
            model_key=model_key,
            node_features=request.node_features,
            legal_action_ids=request.legal_action_ids,
            canonical_player_ids=request.canonical_player_ids,
        )

    def to_eval_request(self) -> EvalRequest:
        return EvalRequest(
            node_features=self.node_features,
            legal_action_ids=self.legal_action_ids,
            canonical_player_ids=self.canonical_player_ids,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "request_id": self.request_id,
            "model_key": self.model_key.to_payload(),
            "node_features": [list(row) for row in self.node_features],
            "legal_action_ids": list(self.legal_action_ids),
            "canonical_player_ids": list(self.canonical_player_ids),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "InferenceRequest":
        if not isinstance(payload, Mapping):
            raise ValueError("inference request payload must be a mapping")
        _require_exact_keys(
            payload,
            {
                "client_id",
                "request_id",
                "model_key",
                "node_features",
                "legal_action_ids",
                "canonical_player_ids",
            },
        )
        rows = _tuple_from_payload(payload["node_features"], "node_features")
        return cls(
            client_id=_non_empty_string(payload["client_id"], "client_id"),
            request_id=_non_empty_string(payload["request_id"], "request_id"),
            model_key=ModelKey.from_payload(payload["model_key"]),
            node_features=tuple(
                tuple(_tuple_from_payload(row, "node_features rows")) for row in rows
            ),
            legal_action_ids=tuple(
                _tuple_from_payload(payload["legal_action_ids"], "legal_action_ids")
            ),
            canonical_player_ids=tuple(
                _tuple_from_payload(payload["canonical_player_ids"], "canonical_player_ids")
            ),
        )


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    client_id: str
    request_id: str
    model_key: ModelKey
    priors: tuple[tuple[int, float], ...]
    value: tuple[float, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.client_id, "client_id")
        _non_empty_string(self.request_id, "request_id")
        if not isinstance(self.model_key, ModelKey):
            raise ValueError("model_key must be a ModelKey")
        priors = _tuple(self.priors, "priors")
        if not priors:
            raise ValueError("priors must not be empty")
        actions: set[int] = set()
        for pair in priors:
            action, probability = _tuple(pair, "prior entries")
            if len(pair) != 2:
                raise ValueError("prior entries must contain action and probability")
            action_id = _integer(action, "prior action")
            if action_id in actions:
                raise ValueError("prior actions must be unique")
            actions.add(action_id)
            if _finite_float(probability, "prior probability") < 0.0:
                raise ValueError("prior probability must be non-negative")
        values = _tuple(self.value, "value")
        expected_size = 1 if self.model_key.player_count == 2 else 3
        if len(values) != expected_size:
            raise ValueError("value does not match the model value size")
        for component in values:
            _finite_float(component, "value components")

    @property
    def correlation_id(self) -> tuple[str, str]:
        return (self.client_id, self.request_id)

    @classmethod
    def from_eval_result(cls, request: InferenceRequest, result: EvalResult) -> "InferenceResponse":
        if not isinstance(request, InferenceRequest):
            raise ValueError("request must be an InferenceRequest")
        if not isinstance(result, EvalResult):
            raise ValueError("result must be an EvalResult")
        value = (result.value,) if isinstance(result.value, float) else tuple(result.value)
        return cls(
            client_id=request.client_id,
            request_id=request.request_id,
            model_key=request.model_key,
            priors=tuple(result.priors.items()),
            value=value,
        )

    def to_eval_result(self) -> EvalResult:
        value: float | tuple[float, ...]
        value = self.value[0] if len(self.value) == 1 else self.value
        return EvalResult(priors=dict(self.priors), value=value)

    def to_payload(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "request_id": self.request_id,
            "model_key": self.model_key.to_payload(),
            "priors": [list(pair) for pair in self.priors],
            "value": list(self.value),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "InferenceResponse":
        if not isinstance(payload, Mapping):
            raise ValueError("inference response payload must be a mapping")
        _require_exact_keys(payload, {"client_id", "request_id", "model_key", "priors", "value"})
        priors = _tuple_from_payload(payload["priors"], "priors")
        return cls(
            client_id=_non_empty_string(payload["client_id"], "client_id"),
            request_id=_non_empty_string(payload["request_id"], "request_id"),
            model_key=ModelKey.from_payload(payload["model_key"]),
            priors=tuple(tuple(_tuple_from_payload(pair, "prior entries")) for pair in priors),
            value=tuple(_tuple_from_payload(payload["value"], "value")),
        )


@dataclass(frozen=True, slots=True)
class InferenceFailure:
    client_id: str
    request_id: str
    model_key: ModelKey
    error_type: str
    message: str

    def __post_init__(self) -> None:
        _non_empty_string(self.client_id, "client_id")
        _non_empty_string(self.request_id, "request_id")
        if not isinstance(self.model_key, ModelKey):
            raise ValueError("model_key must be a ModelKey")
        _non_empty_string(self.error_type, "error_type")
        _non_empty_string(self.message, "message")

    @property
    def correlation_id(self) -> tuple[str, str]:
        return (self.client_id, self.request_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "request_id": self.request_id,
            "model_key": self.model_key.to_payload(),
            "error_type": self.error_type,
            "message": self.message,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "InferenceFailure":
        if not isinstance(payload, Mapping):
            raise ValueError("inference failure payload must be a mapping")
        _require_exact_keys(
            payload,
            {"client_id", "request_id", "model_key", "error_type", "message"},
        )
        return cls(
            client_id=_non_empty_string(payload["client_id"], "client_id"),
            request_id=_non_empty_string(payload["request_id"], "request_id"),
            model_key=ModelKey.from_payload(payload["model_key"]),
            error_type=_non_empty_string(payload["error_type"], "error_type"),
            message=_non_empty_string(payload["message"], "message"),
        )


__all__ = ["InferenceFailure", "InferenceRequest", "InferenceResponse", "ModelKey"]
