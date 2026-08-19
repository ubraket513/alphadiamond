"""Fixed-memory statistics for centralized inference.

The coordinator previously kept one latency sample per request and rebuilt the
sample tuples on every batch, which is quadratic in requests and unbounded in
memory.  At CPU rates that is invisible; at GPU rates one iteration issues tens
of thousands of requests through a single inference thread.

Counts, sums and extremes are tracked exactly.  Quantiles come from a bounded
reservoir sample, which is accurate enough to answer "is the GPU actually being
fed" without ever putting a raw array in a durable ledger.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import median

RESERVOIR_CAPACITY = 2048
"""Samples retained per series; ample for p50/p90 and bounded regardless of load."""


@dataclass(slots=True)
class StreamingSeries:
    """One observed series summarized in constant memory.

    Accumulates in place: this runs inside the inference thread on every
    request, so rebuilding an immutable record per observation cost real
    throughput.  Use :meth:`snapshot` to read a series that cannot change
    underneath the reader.
    """

    count: int = 0
    total: float = 0.0
    smallest: float | None = None
    largest: float | None = None
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float, *, rng: random.Random) -> "StreamingSeries":
        """Record ``value`` in constant time; returns self so calls can chain."""
        number = float(value)
        self.count += 1
        self.total += number
        if self.smallest is None or number < self.smallest:
            self.smallest = number
        if self.largest is None or number > self.largest:
            self.largest = number
        if len(self.samples) < RESERVOIR_CAPACITY:
            self.samples.append(number)
        else:
            # Classic reservoir sampling: every observation keeps an equal
            # chance of being retained, so the sample stays representative.
            index = rng.randrange(self.count)
            if index < RESERVOIR_CAPACITY:
                self.samples[index] = number
        return self

    def snapshot(self) -> "StreamingSeries":
        """An independent copy, safe to read while recording continues."""
        return StreamingSeries(
            count=self.count,
            total=self.total,
            smallest=self.smallest,
            largest=self.largest,
            samples=list(self.samples),
        )

    @property
    def reservoir(self) -> tuple[float, ...]:
        return tuple(self.samples)

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None

    @property
    def minimum(self) -> float | None:
        return self.smallest

    @property
    def maximum(self) -> float | None:
        return self.largest

    def quantile(self, q: float) -> float | None:
        """Approximate quantile from the retained sample, or ``None`` if empty."""
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile must be between 0 and 1")
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        if q == 0.5:
            return float(median(ordered))
        index = min(int(q * len(ordered)), len(ordered) - 1)
        return float(ordered[index])


def _milliseconds(value: float | None) -> float | None:
    return None if value is None else round(value * 1000.0, 4)


def summarize_metrics(metrics, elapsed_s: float) -> dict[str, object]:
    """Return a JSON-ready summary of one coordinator's work.

    Deliberately scalars only: the ledger is durable evidence, and raw latency
    arrays would bloat it without answering a question a summary cannot.
    """
    batch = metrics.batch_size_series
    queue = metrics.queue_to_dispatch_series
    inference = metrics.inference_latency_series
    response = metrics.response_latency_series
    rate = (lambda total: total / elapsed_s) if elapsed_s > 0 else (lambda total: None)

    return {
        "requests_completed": metrics.requests_completed,
        "batches_completed": metrics.batches_completed,
        "mean_batch_size": batch.mean,
        "median_batch_size": batch.quantile(0.5),
        "p90_batch_size": batch.quantile(0.9),
        "max_batch_size": batch.maximum,
        "queue_to_dispatch_p50_ms": _milliseconds(queue.quantile(0.5)),
        "queue_to_dispatch_p90_ms": _milliseconds(queue.quantile(0.9)),
        "inference_p50_ms": _milliseconds(inference.quantile(0.5)),
        "inference_p90_ms": _milliseconds(inference.quantile(0.9)),
        "inference_mean_ms": _milliseconds(inference.mean),
        "response_p50_ms": _milliseconds(response.quantile(0.5)),
        "response_p90_ms": _milliseconds(response.quantile(0.9)),
        "evaluations_per_second": (
            None if not metrics.requests_completed else rate(metrics.requests_completed)
        ),
        "batches_per_second": (
            None if not metrics.batches_completed else rate(metrics.batches_completed)
        ),
    }


__all__ = ["RESERVOIR_CAPACITY", "StreamingSeries", "summarize_metrics"]
