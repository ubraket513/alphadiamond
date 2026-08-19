"""Atomic, JSON-only state for resumable AlphaZero training runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from ..identity import CheckpointCompatibilitySpec
from ..inference.protocol import ModelKey

_SCHEMA_VERSION = 2
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RunStateError(ValueError):
    """A training state is invalid, corrupt, stale, or incompatible."""


class RunStage(str, Enum):
    INITIALIZE = "INITIALIZE"
    SELF_PLAY = "SELF_PLAY"
    REPLAY_INGEST = "REPLAY_INGEST"
    TRAIN = "TRAIN"
    SAVE_CANDIDATE = "SAVE_CANDIDATE"
    PROMOTION_ARENA = "PROMOTION_ARENA"
    RATING_BENCHMARK = "RATING_BENCHMARK"
    PROMOTE_OR_REJECT = "PROMOTE_OR_REJECT"
    PERSIST = "PERSIST"
    COMPLETE = "COMPLETE"


def validate_run_id(value: object) -> str:
    """Return a safe run identifier before any run-relative path is resolved."""
    run_id = _non_empty_string(value, "run_id")
    if not _SAFE_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise RunStateError("run_id contains unsafe path characters")
    return run_id


_STAGES = tuple(RunStage)
_NEXT_STAGE = dict(zip(_STAGES[:-1], _STAGES[1:], strict=True))
_PROGRESS_FIELDS = {
    "champion_checkpoint",
    "champion_model_key",
    "candidate_checkpoint",
    "iteration",
    "training_step",
    "replay_manifest",
    "completed_game_ids",
    "promotion_records",
    "rating_records",
}


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RunStateError(f"training run state must contain JSON data: {error}") from error


def _freeze_json(value: object, field: str) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise RunStateError(f"{field} keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item, field) for key, item in dict(value).items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        _canonical_json(value)
        return value
    raise RunStateError(f"{field} must contain JSON data")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunStateError(f"{field} must be a non-empty string")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunStateError(f"{field} must be a non-negative integer")
    return value


def _optional_pointer(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RunStateError(f"{field} must be a JSON object")
    return value


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise RunStateError(f"{field} must be a JSON list")
    return value


def _require_exact_keys(payload: Mapping[str, object], expected: set[str]) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise RunStateError(f"malformed training run state ({'; '.join(details)})")


def _compatibility_namespace(compatibility: Mapping[str, object]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(compatibility)).hexdigest()}"


@dataclass(frozen=True, slots=True)
class TrainingRunState:
    run_id: str
    model_identity: Mapping[str, object]
    compatibility: Mapping[str, object]
    compatibility_namespace: str
    protocol_ids: Mapping[str, object]
    run_seed: int
    stage: RunStage
    generation: int
    champion_checkpoint: str | None
    champion_model_key: ModelKey | None
    candidate_checkpoint: str | None
    iteration: int
    training_step: int
    replay_manifest: str | None
    completed_game_ids: tuple[str, ...]
    promotion_records: tuple[Mapping[str, object], ...]
    rating_records: tuple[Mapping[str, object], ...]
    stage_completions: Mapping[str, object]
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if not isinstance(self.stage, RunStage):
            raise RunStateError("stage must be a RunStage")
        if self.schema_version != _SCHEMA_VERSION:
            raise RunStateError(f"unsupported training run schema version: {self.schema_version}")
        _non_negative_int(self.run_seed, "run_seed")
        _non_negative_int(self.generation, "generation")
        _non_negative_int(self.iteration, "iteration")
        _non_negative_int(self.training_step, "training_step")
        _optional_pointer(self.champion_checkpoint, "champion_checkpoint")
        if self.champion_model_key is not None:
            if not isinstance(self.champion_model_key, ModelKey):
                raise RunStateError("champion_model_key must be a ModelKey")
            if self.champion_checkpoint is None:
                raise RunStateError("champion_model_key requires champion_checkpoint")
        _optional_pointer(self.candidate_checkpoint, "candidate_checkpoint")
        _optional_pointer(self.replay_manifest, "replay_manifest")

        model_identity = _freeze_json(self.model_identity, "model_identity")
        compatibility = _freeze_json(self.compatibility, "compatibility")
        protocol_ids = _freeze_json(self.protocol_ids, "protocol_ids")
        completions = _freeze_json(self.stage_completions, "stage_completions")
        promotions = _freeze_json(self.promotion_records, "promotion_records")
        ratings = _freeze_json(self.rating_records, "rating_records")
        game_ids = tuple(
            _non_empty_string(item, "completed game ID") for item in self.completed_game_ids
        )

        if not isinstance(model_identity, Mapping) or not isinstance(compatibility, Mapping):
            raise RunStateError("model identity and compatibility must be JSON objects")
        if not isinstance(protocol_ids, Mapping) or not protocol_ids:
            raise RunStateError("protocol_ids must be a non-empty JSON object")
        if any(not isinstance(value, str) or not value for value in protocol_ids.values()):
            raise RunStateError("protocol_ids values must be non-empty strings")
        expected_identity = {
            key: compatibility.get(key)
            for key in (
                "model_name",
                "model_version",
                "player_count",
                "value_semantics_version",
            )
        }
        if dict(model_identity) != expected_identity or expected_identity["model_name"] not in {
            "Soo",
            "Min",
        }:
            raise RunStateError("model_identity does not match compatibility")
        if self.champion_model_key is not None and (
            self.champion_model_key.model_name != expected_identity["model_name"]
            or self.champion_model_key.model_version != expected_identity["model_version"]
        ):
            raise RunStateError("champion_model_key does not match model identity")
        if self.compatibility_namespace != _compatibility_namespace(compatibility):
            raise RunStateError("compatibility_namespace does not match compatibility")
        if len(set(game_ids)) != len(game_ids):
            raise RunStateError("completed_game_ids must be unique")
        if not isinstance(promotions, tuple) or any(
            not isinstance(item, Mapping) for item in promotions
        ):
            raise RunStateError("promotion_records must contain JSON objects")
        if not isinstance(ratings, tuple) or any(not isinstance(item, Mapping) for item in ratings):
            raise RunStateError("rating_records must contain JSON objects")
        if not isinstance(completions, Mapping):
            raise RunStateError("stage_completions must be a JSON object")
        stage_index = _STAGES.index(self.stage)
        expected_completed = {stage.value for stage in _STAGES[:stage_index]}
        if set(completions) != expected_completed or any(
            not isinstance(marker, str) or not marker for marker in completions.values()
        ):
            raise RunStateError("stage_completions must mark exactly the stages before stage")

        object.__setattr__(self, "model_identity", model_identity)
        object.__setattr__(self, "compatibility", compatibility)
        object.__setattr__(self, "protocol_ids", protocol_ids)
        object.__setattr__(self, "stage_completions", completions)
        object.__setattr__(self, "promotion_records", promotions)
        object.__setattr__(self, "rating_records", ratings)
        object.__setattr__(self, "completed_game_ids", game_ids)
        _canonical_json(self.to_payload())

    @property
    def model_name(self) -> str:
        return str(self.model_identity["model_name"])

    def derive_seed(self, *identities: object) -> int:
        """Derive one deterministic non-negative 63-bit seed from stable identities."""
        if not identities:
            raise RunStateError("seed derivation requires at least one stable identity")
        payload = {
            "namespace": "alphadiamond-training-run-seed-v1",
            "run_id": self.run_id,
            "model_identity": self.model_identity,
            "compatibility_namespace": self.compatibility_namespace,
            "protocol_ids": self.protocol_ids,
            "run_seed": self.run_seed,
            "identities": identities,
        }
        digest = hashlib.sha256(_canonical_json(payload)).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "run_id": self.run_id,
            "model_identity": _thaw_json(self.model_identity),
            "compatibility": _thaw_json(self.compatibility),
            "compatibility_namespace": self.compatibility_namespace,
            "protocol_ids": _thaw_json(self.protocol_ids),
            "run_seed": self.run_seed,
            "stage": self.stage.value,
            "champion_checkpoint": self.champion_checkpoint,
            "champion_model_key": (
                self.champion_model_key.to_payload()
                if self.champion_model_key is not None
                else None
            ),
            "candidate_checkpoint": self.candidate_checkpoint,
            "iteration": self.iteration,
            "training_step": self.training_step,
            "replay_manifest": self.replay_manifest,
            "completed_game_ids": list(self.completed_game_ids),
            "promotion_records": _thaw_json(self.promotion_records),
            "rating_records": _thaw_json(self.rating_records),
            "stage_completions": _thaw_json(self.stage_completions),
        }


class RunStateStore:
    """Persist authoritative run state with generation compare-and-swap."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def state_path(self, run_id: str, model_name: str) -> Path:
        validate_run_id(run_id)
        if model_name not in {"Soo", "Min"}:
            raise RunStateError("model_name must be Soo or Min")
        return self.root / model_name.lower() / run_id / "state.json"

    def initialize(
        self,
        *,
        run_id: str,
        compatibility: CheckpointCompatibilitySpec,
        run_seed: int,
        protocol_ids: Mapping[str, str],
        champion_checkpoint: str | None = None,
        champion_model_key: ModelKey | None = None,
        iteration: int = 0,
        training_step: int = 0,
    ) -> TrainingRunState:
        if not isinstance(compatibility, CheckpointCompatibilitySpec):
            raise RunStateError("compatibility must be a CheckpointCompatibilitySpec")
        metadata = compatibility.to_metadata()
        identity = {
            "model_name": compatibility.identity.model_name,
            "model_version": compatibility.identity.model_version,
            "player_count": compatibility.identity.player_count,
            "value_semantics_version": compatibility.identity.value_semantics_version,
        }
        state = TrainingRunState(
            run_id=run_id,
            model_identity=identity,
            compatibility=metadata,
            compatibility_namespace=_compatibility_namespace(metadata),
            protocol_ids=protocol_ids,
            run_seed=run_seed,
            stage=RunStage.INITIALIZE,
            generation=0,
            champion_checkpoint=champion_checkpoint,
            champion_model_key=champion_model_key,
            candidate_checkpoint=None,
            iteration=iteration,
            training_step=training_step,
            replay_manifest=None,
            completed_game_ids=(),
            promotion_records=(),
            rating_records=(),
            stage_completions={},
        )
        path = self.state_path(run_id, state.model_name)
        if path.exists():
            raise RunStateError(f"training run already exists: {run_id}")
        self._atomic_write(path, _canonical_json(state.to_payload()))
        return state

    def load(self, run_id: str, model_name: str) -> TrainingRunState:
        path = self.state_path(run_id, model_name)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise RunStateError(f"training run does not exist: {run_id} ({model_name})") from error
        except OSError as error:
            raise RunStateError(f"cannot read training run state: {path}") from error
        try:
            payload = json.loads(raw.decode("utf-8"))
            state = self._from_payload(payload)
            if state.run_id != run_id or state.model_name != model_name:
                raise RunStateError("training run path does not match its identity")
            return state
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, RunStateError):
                raise
            raise RunStateError(f"corrupt training run state: {error}") from error

    def save(self, state: TrainingRunState) -> TrainingRunState:
        if not isinstance(state, TrainingRunState):
            raise RunStateError("state must be a TrainingRunState")
        current = self._load_for_commit(state)
        if state.stage is not current.stage or state.stage_completions != current.stage_completions:
            raise RunStateError("stage changes must use transition")
        return self._commit(current, replace(state, generation=state.generation + 1))

    def transition(
        self,
        state: TrainingRunState,
        next_stage: RunStage,
        *,
        completion_marker: str,
        **changes: object,
    ) -> TrainingRunState:
        if not isinstance(state, TrainingRunState):
            raise RunStateError("state must be a TrainingRunState")
        if state.stage is RunStage.COMPLETE:
            raise RunStateError("COMPLETE is terminal")
        expected = _NEXT_STAGE[state.stage]
        if next_stage is not expected:
            raise RunStateError(
                f"invalid transition; expected {state.stage.value} -> {expected.value}"
            )
        _non_empty_string(completion_marker, "completion_marker")
        unexpected = set(changes) - _PROGRESS_FIELDS
        if unexpected:
            raise RunStateError(f"transition cannot change fields: {', '.join(sorted(unexpected))}")
        current = self._load_for_commit(state)
        completions = dict(state.stage_completions)
        completions[state.stage.value] = completion_marker
        candidate = replace(
            state,
            stage=next_stage,
            generation=state.generation + 1,
            stage_completions=completions,
            **changes,
        )
        return self._commit(current, candidate)

    def start_next_iteration(self, state: TrainingRunState) -> TrainingRunState:
        """Atomically reset completed iteration work while preserving run lineage."""
        if not isinstance(state, TrainingRunState):
            raise RunStateError("state must be a TrainingRunState")
        if state.stage is not RunStage.COMPLETE:
            raise RunStateError("only a COMPLETE run can start its next iteration")
        current = self._load_for_commit(state)
        candidate = replace(
            state,
            stage=RunStage.INITIALIZE,
            generation=state.generation + 1,
            candidate_checkpoint=None,
            completed_game_ids=(),
            stage_completions={},
        )
        return self._commit(current, candidate)

    def _load_for_commit(self, state: TrainingRunState) -> TrainingRunState:
        try:
            current = self.load(state.run_id, state.model_name)
        except RunStateError as error:
            if "does not exist" in str(error):
                raise RunStateError(
                    "immutable run identity cannot save uninitialized state"
                ) from error
            raise
        immutable = (
            "run_id",
            "model_identity",
            "compatibility",
            "compatibility_namespace",
            "protocol_ids",
            "run_seed",
            "schema_version",
        )
        if any(getattr(state, field) != getattr(current, field) for field in immutable):
            raise RunStateError("immutable training run identity changed")
        if state.generation != current.generation:
            raise RunStateError(
                f"stale training run generation: expected {current.generation}, "
                f"got {state.generation}"
            )
        return current

    def _commit(
        self, current: TrainingRunState, candidate: TrainingRunState
    ) -> TrainingRunState:
        if candidate.generation != current.generation + 1:
            raise RunStateError("new generation must increment by exactly one")
        path = self.state_path(current.run_id, current.model_name)
        self._atomic_write(path, _canonical_json(candidate.to_payload()))
        return candidate

    @staticmethod
    def _from_payload(value: object) -> TrainingRunState:
        payload = _mapping(value, "training run state")
        expected = {
            "schema_version",
            "generation",
            "run_id",
            "model_identity",
            "compatibility",
            "compatibility_namespace",
            "protocol_ids",
            "run_seed",
            "stage",
            "champion_checkpoint",
            "champion_model_key",
            "candidate_checkpoint",
            "iteration",
            "training_step",
            "replay_manifest",
            "completed_game_ids",
            "promotion_records",
            "rating_records",
            "stage_completions",
        }
        _require_exact_keys(payload, expected)
        try:
            stage = RunStage(_non_empty_string(payload["stage"], "stage"))
        except ValueError as error:
            raise RunStateError(f"unknown training run stage: {payload['stage']!r}") from error
        return TrainingRunState(
            schema_version=_non_negative_int(payload["schema_version"], "schema_version"),
            generation=_non_negative_int(payload["generation"], "generation"),
            run_id=_non_empty_string(payload["run_id"], "run_id"),
            model_identity=_mapping(payload["model_identity"], "model_identity"),
            compatibility=_mapping(payload["compatibility"], "compatibility"),
            compatibility_namespace=_non_empty_string(
                payload["compatibility_namespace"], "compatibility_namespace"
            ),
            protocol_ids=_mapping(payload["protocol_ids"], "protocol_ids"),
            run_seed=_non_negative_int(payload["run_seed"], "run_seed"),
            stage=stage,
            champion_checkpoint=_optional_pointer(
                payload["champion_checkpoint"], "champion_checkpoint"
            ),
            champion_model_key=(
                ModelKey.from_payload(payload["champion_model_key"])
                if payload["champion_model_key"] is not None
                else None
            ),
            candidate_checkpoint=_optional_pointer(
                payload["candidate_checkpoint"], "candidate_checkpoint"
            ),
            iteration=_non_negative_int(payload["iteration"], "iteration"),
            training_step=_non_negative_int(payload["training_step"], "training_step"),
            replay_manifest=_optional_pointer(payload["replay_manifest"], "replay_manifest"),
            completed_game_ids=tuple(
                _non_empty_string(item, "completed game ID")
                for item in _list(payload["completed_game_ids"], "completed_game_ids")
            ),
            promotion_records=tuple(
                _mapping(item, "promotion record")
                for item in _list(payload["promotion_records"], "promotion_records")
            ),
            rating_records=tuple(
                _mapping(item, "rating record")
                for item in _list(payload["rating_records"], "rating_records")
            ),
            stage_completions=_mapping(payload["stage_completions"], "stage_completions"),
        )

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            RunStateStore._fsync_directory(path.parent)
        except OSError as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise RunStateError(f"atomic write failed for {path.name}") from error

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        try:
            directory_fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass


__all__ = [
    "RunStage",
    "RunStateError",
    "RunStateStore",
    "TrainingRunState",
    "validate_run_id",
]
