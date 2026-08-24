"""Deterministic, authoritative benchmark opening suites."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from ...contract.state import GameState, build_players
from ..game_adapter import AlphaZeroGameAdapter
from ..identity import RULESET_FINGERPRINT, RULESET_VERSION


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _require_non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_player_count(player_count: object) -> int:
    if (
        not isinstance(player_count, int)
        or isinstance(player_count, bool)
        or player_count not in (2, 3)
    ):
        raise ValueError("player_count must be 2 or 3")
    return player_count


def _validate_action_ids(action_ids: object) -> tuple[int, ...]:
    if not isinstance(action_ids, tuple) or any(
        not isinstance(action_id, int) or isinstance(action_id, bool) or action_id < 0
        for action_id in action_ids
    ):
        raise ValueError("action_ids must be a tuple of non-negative integers")
    return action_ids


@dataclass(frozen=True, slots=True)
class BenchmarkOpening:
    """A replayable action sequence, bound to its benchmark ruleset identity."""

    suite_version: str
    player_count: int
    ruleset_version: str
    ruleset_fingerprint: str
    action_ids: tuple[int, ...]
    opening_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty_string("suite_version", self.suite_version)
        _validate_player_count(self.player_count)
        _require_non_empty_string("ruleset_version", self.ruleset_version)
        _require_non_empty_string("ruleset_fingerprint", self.ruleset_fingerprint)
        _validate_action_ids(self.action_ids)
        object.__setattr__(
            self,
            "opening_id",
            _sha256(
                {
                    "suite_version": self.suite_version,
                    "player_count": self.player_count,
                    "ruleset_version": self.ruleset_version,
                    "ruleset_fingerprint": self.ruleset_fingerprint,
                    "action_ids": self.action_ids,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OpeningSuite:
    """A versioned set of action-only openings replayed by the game adapter."""

    version: str
    player_count: int
    ruleset_version: str
    ruleset_fingerprint: str
    openings: tuple[BenchmarkOpening, ...]
    suite_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty_string("version", self.version)
        _validate_player_count(self.player_count)
        _require_non_empty_string("ruleset_version", self.ruleset_version)
        _require_non_empty_string("ruleset_fingerprint", self.ruleset_fingerprint)
        if not isinstance(self.openings, tuple) or not self.openings:
            raise ValueError("openings must be a non-empty tuple")
        if any(not isinstance(opening, BenchmarkOpening) for opening in self.openings):
            raise ValueError("openings must contain BenchmarkOpening values")
        if self.openings[0].action_ids != ():
            raise ValueError("openings must include the standard initial opening first")
        if len({opening.opening_id for opening in self.openings}) != len(self.openings):
            raise ValueError("openings must have distinct semantic identities")
        object.__setattr__(
            self,
            "suite_hash",
            _sha256(
                {
                    "version": self.version,
                    "player_count": self.player_count,
                    "ruleset_version": self.ruleset_version,
                    "ruleset_fingerprint": self.ruleset_fingerprint,
                    "openings": [
                        {
                            "opening_id": opening.opening_id,
                            "action_ids": opening.action_ids,
                        }
                        for opening in self.openings
                    ],
                }
            ),
        )

    @classmethod
    def generate(
        cls,
        *,
        player_count: int,
        seed: int,
        opening_count: int,
        max_depth: int,
        version: str = "benchmark-openings-v1",
    ) -> OpeningSuite:
        """Generate a fixed-seed suite using only legal adapter action IDs."""
        _validate_player_count(player_count)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(opening_count, int)
            or isinstance(opening_count, bool)
            or opening_count <= 0
        ):
            raise ValueError("opening_count must be a positive integer")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if opening_count > 1 and max_depth == 0:
            raise ValueError("max_depth must be positive when generating non-empty openings")

        adapter = AlphaZeroGameAdapter(build_players(player_count))
        rng = random.Random(seed)
        openings = [
            BenchmarkOpening(
                suite_version=version,
                player_count=player_count,
                ruleset_version=RULESET_VERSION,
                ruleset_fingerprint=RULESET_FINGERPRINT,
                action_ids=(),
            )
        ]
        seen = {openings[0].opening_id}
        attempts = 0
        while len(openings) < opening_count:
            attempts += 1
            if attempts > opening_count * 100:
                raise ValueError("could not generate enough distinct openings")
            state = adapter.initial_state()
            action_ids: list[int] = []
            for _ in range(rng.randint(1, max_depth)):
                legal_action_ids = sorted(adapter.legal_action_ids(state))
                if not legal_action_ids:
                    break
                action_id = rng.choice(legal_action_ids)
                action_ids.append(action_id)
                state = adapter.apply_action(state, action_id)
            opening = BenchmarkOpening(
                suite_version=version,
                player_count=player_count,
                ruleset_version=RULESET_VERSION,
                ruleset_fingerprint=RULESET_FINGERPRINT,
                action_ids=tuple(action_ids),
            )
            if opening.opening_id not in seen:
                openings.append(opening)
                seen.add(opening.opening_id)
        return cls(
            version=version,
            player_count=player_count,
            ruleset_version=RULESET_VERSION,
            ruleset_fingerprint=RULESET_FINGERPRINT,
            openings=tuple(openings),
        )

    def reconstruct(self, opening: BenchmarkOpening) -> GameState:
        """Replay an opening through the authoritative rules implementation."""
        if self.ruleset_version != RULESET_VERSION:
            raise ValueError("ruleset_version does not match the current authoritative ruleset")
        if self.ruleset_fingerprint != RULESET_FINGERPRINT:
            raise ValueError("ruleset_fingerprint does not match the current authoritative ruleset")
        if not isinstance(opening, BenchmarkOpening):
            raise ValueError("opening must be a BenchmarkOpening")
        if opening.player_count != self.player_count:
            raise ValueError("opening player_count does not match this suite")
        if opening.suite_version != self.version:
            raise ValueError("opening suite_version does not match this suite")
        if opening.ruleset_version != self.ruleset_version:
            raise ValueError("opening ruleset_version does not match this suite")
        if opening.ruleset_fingerprint != self.ruleset_fingerprint:
            raise ValueError("opening ruleset_fingerprint does not match this suite")
        if opening.opening_id not in {candidate.opening_id for candidate in self.openings}:
            raise ValueError("opening is not a member of this suite")

        adapter = AlphaZeroGameAdapter(build_players(self.player_count))
        state = adapter.initial_state()
        for action_id in opening.action_ids:
            if action_id not in adapter.legal_action_ids(state):
                raise ValueError(f"opening {opening.opening_id} contains an illegal action")
            state = adapter.apply_action(state, action_id)
        return state


__all__ = ["BenchmarkOpening", "OpeningSuite"]
