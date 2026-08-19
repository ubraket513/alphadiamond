"""The A/B probe reports data-generation metrics without asserting thresholds."""

from __future__ import annotations

import pytest

from diamond.alphazero.bootstrap.probe import ProbeReport, format_report, run_probe
from diamond.alphazero.config import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
)
from diamond.alphazero.identity import MIN_MODEL_NAME, SOO_MODEL_NAME


def report(**overrides) -> ProbeReport:
    payload = {
        "model_name": SOO_MODEL_NAME,
        "bootstrap_prior": BOOTSTRAP_PRIOR_NONE,
        "episodes": 10,
        "completed": 5,
        "move_counts": tuple(range(10, 20)),
        "samples": 100,
        "abort_reasons": {"max_game_moves_exceeded": 5},
    }
    payload.update(overrides)
    return ProbeReport(**payload)


def test_metrics_are_derived_from_raw_counts() -> None:
    result = report()
    assert result.completion_rate == 0.5
    assert result.median_moves == 14.5
    assert result.samples_per_episode == 10.0


def test_p90_needs_enough_completions_to_rank() -> None:
    assert report(move_counts=(1, 2, 3)).p90_moves is None
    assert report(move_counts=tuple(range(1, 11))).p90_moves == 10


def test_empty_probe_reports_zero_rather_than_dividing_by_zero() -> None:
    empty = report(episodes=0, completed=0, move_counts=(), samples=0, abort_reasons={})
    assert empty.completion_rate == 0.0
    assert empty.samples_per_episode == 0.0
    assert empty.median_moves is None


def test_run_probe_rejects_a_non_positive_episode_count() -> None:
    with pytest.raises(ValueError):
        run_probe(episodes=0)


def test_run_probe_rejects_an_unsupported_model() -> None:
    with pytest.raises(ValueError):
        run_probe(model_name="Nope", episodes=1)


def test_run_probe_is_deterministic_for_a_fixed_seed() -> None:
    kwargs = {
        "bootstrap_prior": CANONICAL_TARGET_VACANCY_DISTANCE_V2,
        "episodes": 2,
        "max_moves": 400,
        "base_seed": 0,
    }
    first, second = run_probe(**kwargs), run_probe(**kwargs)
    assert first == second


def test_probe_records_abort_reasons_when_episodes_do_not_finish() -> None:
    result = run_probe(
        bootstrap_prior=BOOTSTRAP_PRIOR_NONE, episodes=1, max_moves=2, base_seed=0
    )
    assert result.completed == 0
    assert result.samples == 0
    assert result.abort_reasons == {"max_game_moves_exceeded": 1}


@pytest.mark.parametrize("model_name", [SOO_MODEL_NAME, MIN_MODEL_NAME])
def test_bootstrap_probe_generates_replay_for_both_models(model_name: str) -> None:
    """The metric that matters: completed episodes yield training samples."""
    result = run_probe(
        model_name=model_name,
        bootstrap_prior=CANONICAL_TARGET_VACANCY_DISTANCE_V2,
        episodes=2,
        max_moves=2000,
        base_seed=0,
    )
    assert result.completed > 0
    assert result.samples > 0
    assert result.abort_reasons == {}


def test_format_report_is_single_line_and_names_the_prior() -> None:
    line = format_report(report())
    assert "\n" not in line
    assert BOOTSTRAP_PRIOR_NONE in line
