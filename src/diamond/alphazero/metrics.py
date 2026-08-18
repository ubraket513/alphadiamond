"""Small correctness/debugging metrics for self-play runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .selfplay.common import SelfPlayEpisode


@dataclass(slots=True)
class SelfPlayMetrics:
    episodes: int = 0
    completed_episodes: int = 0
    aborted_episodes: int = 0
    completed_moves: int = 0
    aborted_moves: int = 0
    samples_generated: int = 0
    abort_reasons: Counter[str] = field(default_factory=Counter)

    def record(self, episode: SelfPlayEpisode) -> None:
        self.episodes += 1
        if episode.completed:
            self.completed_episodes += 1
            self.completed_moves += episode.move_count
            self.samples_generated += len(episode.samples)
        else:
            self.aborted_episodes += 1
            self.aborted_moves += episode.move_count
            self.abort_reasons[episode.aborted_reason or "unknown"] += 1


__all__ = ["SelfPlayMetrics"]
