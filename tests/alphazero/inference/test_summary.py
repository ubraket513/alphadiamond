"""Inference metrics must stay bounded and summarize to JSON for the ledger.

At GPU rates one iteration issues tens of thousands of requests, so keeping a
sample per request -- and rebuilding the tuple on every batch -- is both a leak
and real CPU cost inside the inference thread.
"""

from __future__ import annotations

import json
import random

import pytest

from diamond.alphazero.inference.coordinator import InferenceMetrics
from diamond.alphazero.inference.summary import (
    RESERVOIR_CAPACITY,
    StreamingSeries,
    summarize_metrics,
)


def _series(values) -> StreamingSeries:
    series = StreamingSeries()
    rng = random.Random(7)
    for value in values:
        series = series.observe(value, rng=rng)
    return series


def test_an_empty_series_reports_nothing_rather_than_raising() -> None:
    series = StreamingSeries()

    assert series.count == 0
    assert series.mean is None
    assert series.quantile(0.5) is None
    assert series.minimum is None
    assert series.maximum is None


def test_exact_statistics_are_exact() -> None:
    series = _series(range(1, 101))

    assert series.count == 100
    assert series.mean == pytest.approx(50.5)
    assert series.minimum == 1
    assert series.maximum == 100


def test_quantiles_approximate_the_true_distribution() -> None:
    series = _series(range(1, 101))

    assert series.quantile(0.5) == pytest.approx(50, abs=2)
    assert series.quantile(0.9) == pytest.approx(90, abs=2)


def test_the_reservoir_never_grows_without_bound() -> None:
    series = _series(range(100_000))

    assert len(series.reservoir) <= RESERVOIR_CAPACITY
    # Counts and extremes stay exact even though samples are capped.
    assert series.count == 100_000
    assert series.minimum == 0
    assert series.maximum == 99_999
    assert series.mean == pytest.approx(49_999.5)


def test_a_capped_reservoir_still_estimates_quantiles() -> None:
    series = _series(range(100_000))

    # Within a few percent of the true 50th/90th percentile.
    assert series.quantile(0.5) == pytest.approx(50_000, rel=0.05)
    assert series.quantile(0.9) == pytest.approx(90_000, rel=0.05)


def test_quantile_bounds_are_validated() -> None:
    series = _series((1.0, 2.0))

    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            series.quantile(bad)


def test_summary_is_json_ready_and_carries_no_raw_arrays() -> None:
    metrics = InferenceMetrics()
    rng = random.Random(3)
    for size in (1, 8, 16, 32, 32, 4):
        metrics = metrics.record_batch(
            batch_size=size,
            queue_to_dispatch_s=(0.001,) * size,
            inference_duration_s=0.004,
            response_latencies_s=(0.006,) * size,
            admission_latencies_s=(0.0,) * size,
            rng=rng,
        )

    summary = summarize_metrics(metrics, elapsed_s=10.0)

    assert summary["requests_completed"] == 93
    assert summary["batches_completed"] == 6
    assert summary["max_batch_size"] == 32
    # A batch size is a count, so report it as one.
    assert isinstance(summary["max_batch_size"], int)
    assert summary["mean_batch_size"] == pytest.approx(93 / 6)
    assert summary["evaluations_per_second"] == pytest.approx(9.3)
    assert summary["batches_per_second"] == pytest.approx(0.6)

    # Durable evidence, not a data dump: nothing long enough to bloat a ledger.
    round_tripped = json.loads(json.dumps(summary))
    assert round_tripped == summary
    for value in summary.values():
        assert not isinstance(value, (list, tuple)), value


def test_summary_of_an_idle_coordinator_is_still_serializable() -> None:
    summary = summarize_metrics(InferenceMetrics(), elapsed_s=0.0)

    assert summary["requests_completed"] == 0
    assert summary["batches_completed"] == 0
    assert summary["mean_batch_size"] is None
    assert summary["evaluations_per_second"] is None
    assert json.loads(json.dumps(summary)) == summary
