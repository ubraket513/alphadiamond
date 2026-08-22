"""Idempotent, JSON-only persistence for bounded AlphaZero replay."""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from ..identity import CheckpointCompatibilitySpec
from ..inference.protocol import ModelKey
from ..replay import ReplayBuffer, ReplayCompatibilityError, TrainingSample
from .selfplay_workers import EpisodeResult

_MANIFEST_VERSION = 1
_CHUNK_VERSION = 1


class ReplayStoreError(ValueError):
    """Persistent replay data is incompatible, incomplete, or corrupt."""


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _json_value(value: object, field: str) -> object:
    if not isinstance(value, (dict, list, str, int, float, bool)) and value is not None:
        raise ReplayStoreError(f"{field} is not JSON data")
    return value


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReplayStoreError(f"{field} must be a JSON object")
    return value


def _as_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ReplayStoreError(f"{field} must be a JSON list")
    return value


def _as_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayStoreError(f"{field} must be a non-empty string")
    return value


def _as_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReplayStoreError(f"{field} must be an integer")
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_json(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class ReplayChunk:
    game_id: str
    sha256: str
    sample_count: int


@dataclass(frozen=True, slots=True)
class AbortedEpisode:
    game_id: str
    sha256: str
    seed: int
    retry_id: str
    model_key: Mapping[str, object]
    move_count: int
    aborted_reason: str


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    """The only authority for ingested chunks and deterministic RNG state."""

    compatibility: Mapping[str, object]
    capacity: int
    game_ids: tuple[str, ...]
    chunks: tuple[ReplayChunk, ...]
    aborted: tuple[AbortedEpisode, ...]
    rng_state: object
    schema_version: int = _MANIFEST_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "aborted": [
                {
                    "aborted_reason": row.aborted_reason,
                    "game_id": row.game_id,
                    "model_key": dict(row.model_key),
                    "move_count": row.move_count,
                    "retry_id": row.retry_id,
                    "seed": row.seed,
                    "sha256": row.sha256,
                }
                for row in self.aborted
            ],
            "capacity": self.capacity,
            "chunks": [
                {
                    "game_id": row.game_id,
                    "sample_count": row.sample_count,
                    "sha256": row.sha256,
                }
                for row in self.chunks
            ],
            "compatibility": dict(self.compatibility),
            "game_ids": list(self.game_ids),
            "rng_state": self.rng_state,
            "schema_version": self.schema_version,
        }


class PersistentReplayStore:
    """A tiny append-only store: chunks first, then an atomically replaced manifest."""

    def __init__(
        self,
        root: str | Path,
        compatibility: CheckpointCompatibilitySpec,
        *,
        capacity: int,
        seed: int = 0,
    ) -> None:
        if not isinstance(compatibility, CheckpointCompatibilitySpec):
            raise ValueError("compatibility must be a CheckpointCompatibilitySpec")  # noqa: TRY004
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be positive")
        self.compatibility = compatibility
        self._compatibility = compatibility.to_metadata()
        self.capacity = capacity
        digest = _digest(self._compatibility)
        self.namespace_path = Path(root) / "persistent-replay-v1" / compatibility.identity.model_name / digest
        self.chunks_path = self.namespace_path / "chunks"
        self.manifest_path = self.namespace_path / "manifest.json"
        self._buffer: ReplayBuffer | None = None

        if self.manifest_path.exists():
            self._manifest = self._read_manifest()
            if self._manifest.capacity != capacity:
                raise ReplayStoreError(
                    f"replay capacity mismatch: manifest has {self._manifest.capacity}, requested {capacity}"
                )
        else:
            self._manifest = ReplayManifest(
                compatibility=self._compatibility,
                capacity=capacity,
                game_ids=(),
                chunks=(),
                aborted=(),
                rng_state=random.Random(seed).getstate(),
            )
            self._write_manifest(self._manifest)

    @property
    def manifest(self) -> ReplayManifest:
        return self._manifest

    def chunk_path(self, game_id: str) -> Path:
        """Return the fixed immutable chunk location for a manifest game identity."""
        self._validate_game_id(game_id)
        return self.chunks_path / f"{hashlib.sha256(game_id.encode('utf-8')).hexdigest()}.json"

    def ingest_episode(self, episode: EpisodeResult) -> bool:
        """Ingest one episode, returning false for an identical already-recorded attempt."""
        self._validate_episode(episode)
        if episode.completed:
            body = self._completed_chunk_body(episode)
            content_hash = _digest(body)
            existing = {row.game_id: row for row in self._manifest.chunks}.get(episode.game_id)
            if existing is not None:
                if existing.sha256 != content_hash:
                    raise ReplayStoreError(f"conflicting duplicate game_id: {episode.game_id}")
                return False
            if any(row.game_id == episode.game_id for row in self._manifest.aborted):
                raise ReplayStoreError(f"conflicting duplicate game_id: {episode.game_id}")
            self._write_chunk(episode.game_id, body, content_hash)
            next_manifest = replace(
                self._manifest,
                game_ids=(*self._manifest.game_ids, episode.game_id),
                chunks=(
                    *self._manifest.chunks,
                    ReplayChunk(episode.game_id, content_hash, len(episode.samples)),
                ),
            )
            self._write_manifest(next_manifest)
            self._manifest = next_manifest
            self._buffer = None
            return True

        abort = self._aborted_record(episode)
        existing = {row.game_id: row for row in self._manifest.aborted}.get(episode.game_id)
        if existing is not None:
            if existing.sha256 != abort.sha256:
                raise ReplayStoreError(f"conflicting duplicate game_id: {episode.game_id}")
            return False
        if any(row.game_id == episode.game_id for row in self._manifest.chunks):
            raise ReplayStoreError(f"conflicting duplicate game_id: {episode.game_id}")
        next_manifest = replace(self._manifest, aborted=(*self._manifest.aborted, abort))
        self._write_manifest(next_manifest)
        self._manifest = next_manifest
        return True

    def ingest_episodes(self, episodes: Iterable[EpisodeResult]) -> int:
        """Ingest many episodes with **one** manifest write.

        ``ingest_episode`` rewrites the whole manifest per call, which is fine
        for tens of games and quadratic for a real training iteration: at 768
        games against a manifest that had grown to 1.4 MB, one iteration wrote
        about a gigabyte of manifest to say what a single write says.

        Semantics are otherwise identical, including the duplicate and conflict
        rules, and the return value counts newly ingested episodes.  Failure is
        still all-or-nothing at the manifest: chunks written before an error
        stay as uningested orphans, exactly as the single-episode path leaves
        them, and the manifest remains the only authority for what is ingested.
        """
        manifest = self._manifest
        chunk_index = {row.game_id: row for row in manifest.chunks}
        aborted_index = {row.game_id: row for row in manifest.aborted}
        new_game_ids: list[str] = []
        new_chunks: list[ReplayChunk] = []
        new_aborted: list[AbortedEpisode] = []
        ingested = 0

        for episode in episodes:
            self._validate_episode(episode)
            if episode.completed:
                body = self._completed_chunk_body(episode)
                content_hash = _digest(body)
                existing = chunk_index.get(episode.game_id)
                if existing is not None:
                    if existing.sha256 != content_hash:
                        raise ReplayStoreError(
                            f"conflicting duplicate game_id: {episode.game_id}"
                        )
                    continue
                if episode.game_id in aborted_index:
                    raise ReplayStoreError(f"conflicting duplicate game_id: {episode.game_id}")
                self._write_chunk(episode.game_id, body, content_hash)
                chunk = ReplayChunk(episode.game_id, content_hash, len(episode.samples))
                chunk_index[episode.game_id] = chunk
                new_chunks.append(chunk)
                new_game_ids.append(episode.game_id)
                ingested += 1
                continue

            abort = self._aborted_record(episode)
            existing_abort = aborted_index.get(episode.game_id)
            if existing_abort is not None:
                if existing_abort.sha256 != abort.sha256:
                    raise ReplayStoreError(f"conflicting duplicate game_id: {episode.game_id}")
                continue
            if episode.game_id in chunk_index:
                raise ReplayStoreError(f"conflicting duplicate game_id: {episode.game_id}")
            aborted_index[episode.game_id] = abort
            new_aborted.append(abort)
            ingested += 1

        if not ingested:
            return 0

        next_manifest = replace(
            manifest,
            game_ids=(*manifest.game_ids, *new_game_ids),
            chunks=(*manifest.chunks, *new_chunks),
            aborted=(*manifest.aborted, *new_aborted),
        )
        self._write_manifest(next_manifest)
        self._manifest = next_manifest
        if new_chunks:
            self._buffer = None
        return ingested

    def prune_to_capacity(self) -> int:
        """Drop chunks that ``load_buffer`` provably cannot reach, and their files.

        The store is append-only, but the buffer it rebuilds is **bounded**:
        ``load_buffer`` extends a ``ReplayBuffer(capacity=self.capacity)`` in
        manifest order, so only the newest ``capacity`` samples survive.  Every
        chunk older than that window is read from disk and immediately evicted,
        on every iteration.

        Left alone that is unbounded disk and quadratic time -- measured on the
        from-scratch run at ~79 MB and ~5,000 chunks per iteration, which fills
        a 13 GB disk in about two and a half hours and re-reads the whole
        history to build each buffer.

        Pruning is **explicit rather than automatic**: it deletes durable
        evidence, so a caller asks for it.

        ``game_ids`` is pruned in lockstep because the manifest loader requires
        exactly that -- it is the ordered chunk identity list, not a record of
        every game ever seen.  ``aborted`` is left intact: it is one small row
        per aborted game and it carries the only record that those games
        happened. The production coordinator reads ``game_ids`` to assert that
        the episodes it just ingested are present, which pruning cannot disturb
        because the newest chunks are exactly the ones kept.

        Returns the number of chunks removed.
        """
        chunks = self._manifest.chunks
        kept = 0
        total = 0
        for chunk in reversed(chunks):
            kept += 1
            total += chunk.sample_count
            if total >= self.capacity:
                break
        if kept >= len(chunks):
            return 0

        keep = chunks[len(chunks) - kept :]
        dropped = chunks[: len(chunks) - kept]
        keep_ids = tuple(chunk.game_id for chunk in keep)
        # Manifest first: a chunk file removed before the manifest stops
        # referencing it is a corrupt store, while a file left behind after the
        # manifest drops it is only an orphan, which the store already tolerates.
        next_manifest = replace(self._manifest, chunks=keep, game_ids=keep_ids)
        self._write_manifest(next_manifest)
        self._manifest = next_manifest
        for chunk in dropped:
            self.chunk_path(chunk.game_id).unlink(missing_ok=True)
        return len(dropped)

    def load_buffer(self) -> ReplayBuffer:
        """Rebuild a normal bounded ReplayBuffer from manifest-referenced JSON chunks."""
        if self._buffer is not None:
            return self._buffer
        replay = ReplayBuffer(self.compatibility, capacity=self.capacity)
        for chunk in self._manifest.chunks:
            replay.extend(self._read_chunk(chunk))
        try:
            replay._rng.setstate(_freeze_json(self._manifest.rng_state))  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ReplayStoreError("manifest has an invalid deterministic RNG state") from error
        self._buffer = replay
        return replay

    def sample(self, batch_size: int) -> tuple[TrainingSample, ...]:
        """Sample through ReplayBuffer semantics and persist the advanced RNG state."""
        replay = self.load_buffer()
        previous_rng_state = replay._rng.getstate()
        samples = replay.sample(batch_size)
        next_manifest = replace(self._manifest, rng_state=replay._rng.getstate())
        try:
            self._write_manifest(next_manifest)
        except ReplayStoreError:
            replay._rng.setstate(previous_rng_state)
            raise
        self._manifest = next_manifest
        return samples

    def compact(self) -> bool:
        """Atomically rewrite the authoritative manifest after validating referenced data.

        Chunks are immutable and retained for idempotence.  The method intentionally does
        not ingest or delete unreferenced crash orphans: they remain recoverable and cannot
        affect replay until a manifest references them.
        """
        self.load_buffer()
        self._write_manifest(self._manifest)
        return False

    def _validate_episode(self, episode: EpisodeResult) -> None:
        if not isinstance(episode, EpisodeResult):
            raise ValueError("episode must be an EpisodeResult")  # noqa: TRY004
        self._validate_game_id(episode.game_id)
        if episode.compatibility != self.compatibility:
            raise ReplayStoreError("episode compatibility does not match replay namespace")
        if (
            episode.model_key.model_name != self.compatibility.identity.model_name
            or episode.model_key.model_version != self.compatibility.identity.model_version
        ):
            raise ReplayStoreError("episode model identity does not match replay compatibility")

    @staticmethod
    def _validate_game_id(game_id: object) -> None:
        if not isinstance(game_id, str) or not game_id.strip():
            raise ReplayStoreError("game_id must be a non-empty string")

    def _completed_chunk_body(self, episode: EpisodeResult) -> dict[str, object]:
        return {
            "compatibility": self._compatibility,
            "episode": self._episode_identity(episode),
            "samples": [self._sample_payload(sample) for sample in episode.samples],
            "schema_version": _CHUNK_VERSION,
        }

    @staticmethod
    def _episode_identity(episode: EpisodeResult) -> dict[str, object]:
        return {
            "completed": episode.completed,
            "final_order": list(episode.final_order) if episode.final_order is not None else None,
            "game_id": episode.game_id,
            "model_key": episode.model_key.to_payload(),
            "move_count": episode.move_count,
            "retry_id": episode.retry_id,
            "seed": episode.seed,
        }

    @staticmethod
    def _sample_payload(sample: TrainingSample) -> dict[str, object]:
        return {
            "canonical_player_ids": list(sample.canonical_player_ids),
            "node_features": [list(row) for row in sample.node_features],
            "schema_version": sample.schema_version,
            "sparse_policy": [list(row) for row in sample.sparse_policy],
            "value_target": list(sample.value_target),
        }

    def _aborted_record(self, episode: EpisodeResult) -> AbortedEpisode:
        payload = self._episode_identity(episode) | {"aborted_reason": episode.aborted_reason}
        return AbortedEpisode(
            game_id=episode.game_id,
            sha256=_digest(payload),
            seed=episode.seed,
            retry_id=episode.retry_id,
            model_key=episode.model_key.to_payload(),
            move_count=episode.move_count,
            aborted_reason=episode.aborted_reason or "unknown abort",
        )

    def _write_chunk(self, game_id: str, body: dict[str, object], content_hash: str) -> None:
        path = self.chunk_path(game_id)
        payload = body | {"sha256": content_hash}
        encoded = _canonical_json(payload)
        if path.exists():
            try:
                current = path.read_bytes()
            except OSError as error:
                raise ReplayStoreError(f"cannot read existing replay chunk for {game_id}") from error
            if current != encoded:
                raise ReplayStoreError(f"conflicting immutable replay chunk for {game_id}")
            return
        self._atomic_write(path, encoded)

    def _read_chunk(self, reference: ReplayChunk) -> tuple[TrainingSample, ...]:
        path = self.chunk_path(reference.game_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise ReplayStoreError(f"missing replay chunk for {reference.game_id}") from error
        except OSError as error:
            raise ReplayStoreError(f"cannot read replay chunk for {reference.game_id}") from error
        try:
            payload = _as_mapping(json.loads(raw.decode("utf-8")), "replay chunk")
            digest = _as_string(payload.get("sha256"), "chunk sha256")
            body = {key: value for key, value in payload.items() if key != "sha256"}
            if digest != _digest(body) or digest != reference.sha256:
                raise ReplayStoreError(f"corrupt replay chunk hash for {reference.game_id}")
            if _as_int(body.get("schema_version"), "chunk schema_version") != _CHUNK_VERSION:
                raise ReplayStoreError(f"unsupported replay chunk for {reference.game_id}")
            if body.get("compatibility") != self._compatibility:
                raise ReplayStoreError(f"replay chunk compatibility mismatch for {reference.game_id}")
            identity = _as_mapping(body.get("episode"), "chunk episode")
            if _as_string(identity.get("game_id"), "chunk game_id") != reference.game_id:
                raise ReplayStoreError(f"corrupt replay chunk identity for {reference.game_id}")
            if identity.get("completed") is not True:
                raise ReplayStoreError(f"corrupt replay chunk completion state for {reference.game_id}")
            ModelKey.from_payload(identity.get("model_key"))
            rows = tuple(self._sample_from_payload(row) for row in _as_list(body.get("samples"), "samples"))
            if len(rows) != reference.sample_count:
                raise ReplayStoreError(f"corrupt replay chunk sample count for {reference.game_id}")
            return rows
        except (UnicodeDecodeError, json.JSONDecodeError, ReplayCompatibilityError, ValueError) as error:
            if isinstance(error, ReplayStoreError):
                raise
            raise ReplayStoreError(f"corrupt replay chunk for {reference.game_id}: {error}") from error

    def _sample_from_payload(self, payload: object) -> TrainingSample:
        row = _as_mapping(payload, "sample")
        features = tuple(
            tuple(_json_value(value, "feature value") for value in _as_list(values, "feature row"))
            for values in _as_list(row.get("node_features"), "node_features")
        )
        policy = tuple(
            tuple(_as_list(pair, "sparse policy entry"))
            for pair in _as_list(row.get("sparse_policy"), "sparse_policy")
        )
        return TrainingSample(
            compatibility=self.compatibility,
            node_features=features,  # type: ignore[arg-type]
            canonical_player_ids=tuple(_as_list(row.get("canonical_player_ids"), "player ids")),  # type: ignore[arg-type]
            sparse_policy=policy,  # type: ignore[arg-type]
            value_target=tuple(_as_list(row.get("value_target"), "value target")),  # type: ignore[arg-type]
            schema_version=_as_int(row.get("schema_version"), "sample schema_version"),
        )

    def _read_manifest(self) -> ReplayManifest:
        try:
            payload = _as_mapping(json.loads(self.manifest_path.read_text(encoding="utf-8")), "manifest")
            if _as_int(payload.get("schema_version"), "manifest schema_version") != _MANIFEST_VERSION:
                raise ReplayStoreError("unsupported replay manifest")
            compatibility = _as_mapping(payload.get("compatibility"), "manifest compatibility")
            if dict(compatibility) != self._compatibility:
                raise ReplayStoreError("replay manifest compatibility mismatch")
            chunks = tuple(
                ReplayChunk(
                    game_id=_as_string(_as_mapping(item, "chunk").get("game_id"), "chunk game_id"),
                    sha256=_as_string(_as_mapping(item, "chunk").get("sha256"), "chunk sha256"),
                    sample_count=_as_int(_as_mapping(item, "chunk").get("sample_count"), "chunk count"),
                )
                for item in _as_list(payload.get("chunks"), "manifest chunks")
            )
            game_ids = tuple(_as_string(item, "game id") for item in _as_list(payload.get("game_ids"), "game ids"))
            if game_ids != tuple(row.game_id for row in chunks) or len(set(game_ids)) != len(game_ids):
                raise ReplayStoreError("manifest game_ids do not match ordered chunks")
            aborted = tuple(self._aborted_from_payload(item) for item in _as_list(payload.get("aborted"), "aborted"))
            known = set(game_ids)
            if any(row.game_id in known for row in aborted) or len({row.game_id for row in aborted}) != len(aborted):
                raise ReplayStoreError("manifest has duplicate game identities")
            return ReplayManifest(
                compatibility=dict(compatibility),
                capacity=_as_int(payload.get("capacity"), "manifest capacity"),
                game_ids=game_ids,
                chunks=chunks,
                aborted=aborted,
                rng_state=payload.get("rng_state"),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            if isinstance(error, ReplayStoreError):
                raise
            raise ReplayStoreError(f"corrupt replay manifest: {error}") from error

    @staticmethod
    def _aborted_from_payload(payload: object) -> AbortedEpisode:
        row = _as_mapping(payload, "aborted episode")
        model_key = _as_mapping(row.get("model_key"), "aborted model key")
        ModelKey.from_payload(model_key)
        return AbortedEpisode(
            game_id=_as_string(row.get("game_id"), "aborted game_id"),
            sha256=_as_string(row.get("sha256"), "aborted sha256"),
            seed=_as_int(row.get("seed"), "aborted seed"),
            retry_id=_as_string(row.get("retry_id"), "aborted retry_id"),
            model_key=dict(model_key),
            move_count=_as_int(row.get("move_count"), "aborted move_count"),
            aborted_reason=_as_string(row.get("aborted_reason"), "aborted reason"),
        )

    def _write_manifest(self, manifest: ReplayManifest) -> None:
        self._atomic_write(self.manifest_path, _canonical_json(manifest.to_payload()))

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
        except OSError as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ReplayStoreError(f"atomic write failed for {path.name}") from error


__all__ = [
    "AbortedEpisode",
    "PersistentReplayStore",
    "ReplayChunk",
    "ReplayManifest",
    "ReplayStoreError",
]
