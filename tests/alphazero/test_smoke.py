from __future__ import annotations

from copy import deepcopy

from diamond.alphazero.smoke import (
    run_arena_smoke,
    run_checkpoint_smoke,
    run_selfplay_smoke,
    run_training_smoke,
    smoke_succeeded,
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
        "Soo": {
            "model_version": "0.1.0",
            "training_step": 1,
            "parameters_restored": True,
            "optimizer_restored": True,
            "config_restored": True,
        },
        "Min": {
            "model_version": "0.1.0",
            "training_step": 1,
            "parameters_restored": True,
            "optimizer_restored": True,
            "config_restored": True,
        },
    }


def test_arena_smoke_reports_balanced_soo_and_min_results() -> None:
    result = run_arena_smoke()
    assert result == {
        "Soo": {"wins": 2, "losses": 2, "aborted": 0},
        "Min": {"first": 6, "second": 6, "third": 6, "aborted": 0},
    }


def test_smoke_success_requires_every_subsystem_check() -> None:
    result = {
        "selfplay": run_selfplay_smoke(),
        "training": run_training_smoke(),
        "checkpoint": run_checkpoint_smoke(),
        "arena": run_arena_smoke(),
    }
    assert smoke_succeeded(result)

    broken_training = deepcopy(result)
    broken_training["training"]["Soo"]["training_step"] = 0
    assert not smoke_succeeded(broken_training)

    broken_checkpoint = deepcopy(result)
    broken_checkpoint["checkpoint"]["Min"]["optimizer_restored"] = False
    assert not smoke_succeeded(broken_checkpoint)

    broken_arena = deepcopy(result)
    broken_arena["arena"]["Soo"]["aborted"] = 1
    assert not smoke_succeeded(broken_arena)

    missing_min = deepcopy(result)
    for section in ("selfplay", "training", "checkpoint"):
        del missing_min[section]["Min"]
    assert not smoke_succeeded(missing_min)

    empty_model_checks = deepcopy(result)
    for section in ("selfplay", "training", "checkpoint"):
        empty_model_checks[section] = {}
    assert not smoke_succeeded(empty_model_checks)
