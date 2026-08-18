from __future__ import annotations

from diamond.alphazero.smoke import (
    run_arena_smoke,
    run_checkpoint_smoke,
    run_selfplay_smoke,
    run_training_smoke,
)


def test_authoritative_selfplay_smoke_reaches_soo_and_full_min_ranking() -> None:
    result = run_selfplay_smoke()

    assert result == {
        "Soo": {"completed": True, "moves": 1, "final_order": [1, 2], "samples": 1},
        "Min": {
            "completed": True,
            "moves": 2,
            "final_order": [1, 2, 3],
            "samples": 2,
        },
    }


def test_training_smoke_updates_soo_and_min_once() -> None:
    result = run_training_smoke()
    assert result["Soo"]["training_step"] == 1
    assert result["Min"]["training_step"] == 1
    assert result["Soo"]["total_loss"] > 0
    assert result["Min"]["total_loss"] > 0


def test_checkpoint_smoke_restores_both_model_identities() -> None:
    result = run_checkpoint_smoke()
    assert result == {
        "Soo": {"model_version": "0.1.0", "training_step": 1},
        "Min": {"model_version": "0.1.0", "training_step": 1},
    }


def test_arena_smoke_reports_balanced_soo_and_min_results() -> None:
    result = run_arena_smoke()
    assert result == {
        "Soo": {"wins": 1, "losses": 1, "aborted": 0},
        "Min": {"first": 2, "second": 2, "third": 2, "aborted": 0},
    }
