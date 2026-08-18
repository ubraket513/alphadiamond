from __future__ import annotations

from diamond.alphazero.metrics import SelfPlayMetrics
from diamond.alphazero.selfplay.common import SelfPlayEpisode


def test_selfplay_metrics_distinguish_completed_and_aborted_episodes() -> None:
    metrics = SelfPlayMetrics()
    metrics.record(SelfPlayEpisode((), (1, 2), 12, True))
    metrics.record(
        SelfPlayEpisode((), None, 50, False, "max_game_moves_exceeded")
    )

    assert metrics.episodes == 2
    assert metrics.completed_episodes == 1
    assert metrics.aborted_episodes == 1
    assert metrics.completed_moves == 12
    assert metrics.aborted_moves == 50
    assert metrics.abort_reasons == {"max_game_moves_exceeded": 1}

