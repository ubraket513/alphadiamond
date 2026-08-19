from __future__ import annotations

from diamond.alphazero.milestone2_smoke import run_smoke


def test_tiny_production_smoke_runs_soo_and_min_without_fabricating_min_rating(tmp_path) -> None:
    report = run_smoke(tmp_path)

    assert report["status"] == "ok"
    assert set(report["models"]) == {"Soo", "Min"}
    assert report["models"]["Soo"]["worker_games"] == 2
    assert report["models"]["Soo"]["training_step"] == 1
    assert report["models"]["Soo"]["rating_events"] == 1
    assert report["models"]["Soo"]["state_reloaded"] is True
    assert report["models"]["Min"]["worker_games"] == 2
    assert report["models"]["Min"]["training_step"] == 1
    assert report["models"]["Min"]["rating_events"] == 0
    assert report["models"]["Min"]["rating_status"] == "insufficient_history"
    assert report["models"]["Min"]["participant_count"] == 2
    assert report["models"]["Min"]["state_reloaded"] is True
