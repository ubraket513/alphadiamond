"""CPU-resident sparse replay with strict Soo/Min schema isolation."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .identity import CheckpointCompatibilitySpec

SAMPLE_SCHEMA_VERSION = 1


class ReplayCompatibilityError(ValueError):
    """A sample cannot be mixed into this replay schema."""


@dataclass(frozen=True, slots=True)
class TrainingSample:
    compatibility: CheckpointCompatibilitySpec
    node_features: tuple[tuple[float, ...], ...]
    canonical_player_ids: tuple[int, ...]
    sparse_policy: tuple[tuple[int, float], ...]
    value_target: tuple[float, ...]
    schema_version: int = SAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        player_count = self.compatibility.identity.player_count
        if self.schema_version != SAMPLE_SCHEMA_VERSION:
            raise ReplayCompatibilityError(f"unsupported sample schema {self.schema_version}")
        if len(self.node_features) != 73:
            raise ReplayCompatibilityError("sample must contain 73 board nodes")
        if any(len(row) != player_count * 2 for row in self.node_features):
            raise ReplayCompatibilityError("sample feature shape does not match player count")
        if len(self.canonical_player_ids) != player_count or len(set(self.canonical_player_ids)) != player_count:
            raise ReplayCompatibilityError("canonical player identity is malformed")
        expected_values = 1 if player_count == 2 else 3
        if len(self.value_target) != expected_values:
            raise ReplayCompatibilityError("value target shape does not match Soo/Min semantics")
        action_ids = [action for action, _ in self.sparse_policy]
        probabilities = [probability for _, probability in self.sparse_policy]
        if not action_ids or len(action_ids) != len(set(action_ids)):
            raise ReplayCompatibilityError("sparse policy must contain unique visited actions")
        if any(action < 0 for action in action_ids):
            raise ReplayCompatibilityError("sparse policy contains a negative action")
        if any(not math.isfinite(p) or p <= 0 for p in probabilities):
            raise ReplayCompatibilityError("sparse policy probabilities must be finite and positive")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ReplayCompatibilityError("sparse policy probabilities must sum to one")

    @property
    def model_name(self) -> str:
        return self.compatibility.identity.model_name


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    node_features: tuple[tuple[tuple[float, ...], ...], ...]
    policy_targets: tuple[tuple[float, ...], ...]
    value_targets: tuple[tuple[float, ...], ...]


class ReplayBuffer:
    def __init__(
        self,
        compatibility: CheckpointCompatibilitySpec,
        *,
        capacity: int,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.compatibility = compatibility
        self.capacity = capacity
        self._samples: deque[TrainingSample] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> tuple[TrainingSample, ...]:
        return tuple(self._samples)

    def add(self, sample: TrainingSample) -> None:
        if sample.compatibility != self.compatibility:
            raise ReplayCompatibilityError(
                "sample compatibility metadata does not match this replay buffer"
            )
        self._samples.append(sample)

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int) -> tuple[TrainingSample, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._samples):
            raise ValueError("batch_size exceeds available replay samples")
        return tuple(self._rng.sample(tuple(self._samples), batch_size))

    def collate(
        self, samples: Iterable[TrainingSample], *, action_size: int
    ) -> ReplayBatch:
        rows = tuple(samples)
        dense_policies: list[tuple[float, ...]] = []
        for sample in rows:
            if sample.compatibility != self.compatibility:
                raise ReplayCompatibilityError("cannot collate an incompatible sample")
            dense = [0.0] * action_size
            for action_id, probability in sample.sparse_policy:
                if action_id >= action_size:
                    raise ReplayCompatibilityError("sample action exceeds action space")
                dense[action_id] = probability
            dense_policies.append(tuple(dense))
        return ReplayBatch(
            node_features=tuple(sample.node_features for sample in rows),
            policy_targets=tuple(dense_policies),
            value_targets=tuple(sample.value_target for sample in rows),
        )


__all__ = [
    "ReplayBatch",
    "ReplayBuffer",
    "ReplayCompatibilityError",
    "SAMPLE_SCHEMA_VERSION",
    "TrainingSample",
]
