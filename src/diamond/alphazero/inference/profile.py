"""Small, measured AlphaZero inference profiling primitives.

This module deliberately reports only measurements made by the bounded profile
run.  It is not a process-wide telemetry system and it keeps optional numeric
modes isolated from the eager evaluator used by an inference pool.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from time import monotonic
from typing import Any, Callable, Mapping, Sequence


_STAGES = (
    "queue_wait",
    "inference",
    "self_play",
    "replay_collation",
    "training",
)


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Finite summary of one set of monotonic duration samples."""

    samples: int
    mean_s: float
    p50_s: float
    p95_s: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> TimingSummary:
        if not samples:
            raise ValueError("timing summary requires at least one sample")
        values = tuple(float(value) for value in samples)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("timing samples must be finite and non-negative")
        ordered = sorted(values)

        def percentile(fraction: float) -> float:
            return ordered[math.ceil((len(ordered) - 1) * fraction)]

        return cls(
            samples=len(ordered),
            mean_s=sum(ordered) / len(ordered),
            p50_s=percentile(0.5),
            p95_s=percentile(0.95),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "mean_s": self.mean_s,
            "p50_s": self.p50_s,
            "p95_s": self.p95_s,
        }


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    cpu: str
    gpu_verified: bool
    gpus: tuple[dict[str, object], ...] = ()

    @property
    def supports_bf16(self) -> bool:
        return bool(self.gpus) and all(bool(gpu["bf16_supported"]) for gpu in self.gpus)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"cpu": self.cpu, "gpu_verified": self.gpu_verified}
        if self.gpus:
            payload["gpus"] = [dict(gpu) for gpu in self.gpus]
        return payload


@dataclass(frozen=True, slots=True)
class ModeProfile:
    name: str
    states_per_second: float
    calls_per_second: float
    batch_seconds: TimingSummary
    latency_seconds: TimingSummary
    first_call_startup_s: float
    steady_state_seconds: TimingSummary
    peak_memory_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "states_per_second": self.states_per_second,
            "calls_per_second": self.calls_per_second,
            "batch_seconds": self.batch_seconds.to_dict(),
            "latency_seconds": self.latency_seconds.to_dict(),
            "first_call_startup_s": self.first_call_startup_s,
            "steady_state_seconds": self.steady_state_seconds.to_dict(),
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """JSON-safe result of one bounded profile session."""

    max_seconds: int
    hardware: HardwareProfile
    modes: tuple[ModeProfile, ...]
    stage_timings: Mapping[str, TimingSummary]
    unavailable_modes: Mapping[str, str]

    @classmethod
    def empty(cls, *, max_seconds: int) -> ProfileReport:
        return cls(
            max_seconds=max_seconds,
            hardware=detect_hardware(),
            modes=(),
            stage_timings={},
            unavailable_modes={},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_seconds": self.max_seconds,
            "hardware": self.hardware.to_dict(),
            "modes": [mode.to_dict() for mode in self.modes],
            "stage_timings": {
                name: summary.to_dict() for name, summary in self.stage_timings.items()
            },
            "unavailable_modes": dict(self.unavailable_modes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("AlphaZero profiling requires the optional torch dependency") from error
    return torch


def detect_hardware() -> HardwareProfile:
    """Return only CUDA devices whose identity was queried successfully."""
    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return HardwareProfile(cpu=cpu, gpu_verified=False)
        gpus: list[dict[str, object]] = []
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            gpus.append(
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                    "capability": [int(capability[0]), int(capability[1])],
                    "bf16_supported": bool(torch.cuda.is_bf16_supported(index)),
                }
            )
        return HardwareProfile(cpu=cpu, gpu_verified=bool(gpus), gpus=tuple(gpus))
    except Exception:
        return HardwareProfile(cpu=cpu, gpu_verified=False)


def assert_evaluation_agreement(
    reference: Sequence[Any], candidate: Sequence[Any], *, rtol: float, atol: float
) -> None:
    """Check legal normalized policies and values with explicit allclose tolerances."""
    if rtol < 0 or atol < 0:
        raise ValueError("numeric agreement tolerances must be non-negative")
    if len(reference) != len(candidate):
        raise ValueError("numeric modes returned a different result count")
    torch = _torch()
    for expected, actual in zip(reference, candidate, strict=True):
        expected_priors = expected.priors
        actual_priors = actual.priors
        if set(expected_priors) != set(actual_priors) or not expected_priors:
            raise ValueError("numeric mode changed the legal policy action set")
        for priors in (expected_priors, actual_priors):
            values = tuple(float(value) for value in priors.values())
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("numeric mode produced a non-finite legal policy")
            if not math.isclose(sum(values), 1.0, rel_tol=rtol, abs_tol=atol):
                raise ValueError("numeric mode produced a non-normalized legal policy")
        if not torch.allclose(
            torch.tensor(tuple(expected_priors.values())),
            torch.tensor(tuple(actual_priors[action] for action in expected_priors)),
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError("numeric mode policy disagreement exceeded tolerance")
        expected_value = _value_tensor(expected.value, torch)
        actual_value = _value_tensor(actual.value, torch)
        if not torch.allclose(expected_value, actual_value, rtol=rtol, atol=atol):
            raise ValueError("numeric mode value disagreement exceeded tolerance")


def _value_tensor(value: object, torch: Any) -> Any:
    values = (value,) if isinstance(value, (int, float)) else tuple(value)  # type: ignore[arg-type]
    if any(not math.isfinite(float(component)) for component in values):
        raise ValueError("numeric mode produced a non-finite value")
    return torch.tensor(values)


class _StageMeter:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    def measure(self, name: str, operation: Callable[[], Any]) -> Any:
        started = monotonic()
        try:
            return operation()
        finally:
            self.samples[name].append(monotonic() - started)

    def summaries(self) -> dict[str, TimingSummary]:
        return {name: TimingSummary.from_samples(self.samples[name]) for name in _STAGES}


def _is_cuda_evaluator(evaluator: Any) -> bool:
    return str(getattr(evaluator, "device", "cpu")).startswith("cuda")


def _synchronize(evaluator: Any) -> None:
    if _is_cuda_evaluator(evaluator):
        _torch().cuda.synchronize(getattr(evaluator, "device", None))


def _peak_memory(evaluator: Any) -> int | None:
    if not _is_cuda_evaluator(evaluator):
        return None
    torch = _torch()
    return int(torch.cuda.max_memory_allocated(getattr(evaluator, "device", None)))


def _reset_peak_memory(evaluator: Any) -> None:
    if _is_cuda_evaluator(evaluator):
        _torch().cuda.reset_peak_memory_stats(getattr(evaluator, "device", None))


def _measure_mode(
    *,
    name: str,
    evaluator: Any,
    requests: tuple[Any, ...],
    max_seconds: float,
    meter: _StageMeter | None,
) -> tuple[ModeProfile, tuple[Any, ...]]:
    """Measure a mode without mutating the evaluator or model-pool default."""
    batch_samples: list[float] = []
    latency_samples: list[float] = []
    steady_samples: list[float] = []
    first_results: tuple[Any, ...] = ()
    calls = 0
    started = monotonic()
    deadline = started + max_seconds
    _synchronize(evaluator)
    _reset_peak_memory(evaluator)

    def run_batch() -> tuple[Any, ...]:
        queue = Queue(maxsize=1)
        if meter is not None:
            meter.measure("queue_wait", lambda: (queue.put(requests), queue.get()))
        else:
            queue.put(requests)
            queue.get()
        inference_started = monotonic()

        def infer() -> Any:
            output = evaluator.evaluate(requests)
            _synchronize(evaluator)
            return output

        if meter is None:
            results = infer()
        else:
            results = meter.measure("inference", infer)
        batch_samples.append(monotonic() - inference_started)
        if meter is not None:
            meter.measure("self_play", lambda: max(results[0].priors, key=results[0].priors.get))
            meter.measure(
                "replay_collation",
                lambda: tuple((action, probability) for action, probability in results[0].priors.items()),
            )
            meter.measure("training", lambda: sum(float(value) for value in results[0].priors.values()))
        return tuple(results)

    while calls == 0 or (monotonic() < deadline and calls < 64):
        call_started = monotonic()
        results = run_batch()
        latency = monotonic() - call_started
        latency_samples.append(latency)
        if calls == 0:
            first_results = results
            first_call = latency
        else:
            steady_samples.append(latency)
        calls += 1

    elapsed = max(monotonic() - started, sys.float_info.epsilon)
    if not steady_samples:
        steady_samples.append(first_call)
    return (
        ModeProfile(
            name=name,
            states_per_second=(calls * len(requests)) / elapsed,
            calls_per_second=calls / elapsed,
            batch_seconds=TimingSummary.from_samples(batch_samples),
            latency_seconds=TimingSummary.from_samples(latency_samples),
            first_call_startup_s=first_call,
            steady_state_seconds=TimingSummary.from_samples(steady_samples),
            peak_memory_bytes=_peak_memory(evaluator),
        ),
        first_results,
    )


def profile_evaluator(
    evaluator: Any,
    requests: tuple[Any, ...],
    *,
    max_seconds: float,
    include_optional: bool = True,
) -> ProfileReport:
    """Profile eager FP32 first, then isolated BF16 and compiled candidates.

    Compiled candidates are never installed in ``evaluator`` or a model pool.
    A failed compile, recompile, agreement check, or memory measurement is
    recorded as unavailable while the eager measurement remains usable.
    """
    if max_seconds <= 0 or not math.isfinite(max_seconds):
        raise ValueError("max_seconds must be finite and positive")
    if not requests:
        raise ValueError("profiling requires at least one evaluation request")
    if getattr(evaluator, "precision", "fp32") != "fp32":
        raise ValueError("profiling requires an eager FP32 reference evaluator")
    hardware = detect_hardware()
    meter = _StageMeter()
    eager, eager_results = _measure_mode(
        name="eager-fp32",
        evaluator=evaluator,
        requests=requests,
        max_seconds=max_seconds,
        meter=meter,
    )
    modes = [eager]
    unavailable: dict[str, str] = {}
    optional_candidates: list[tuple[str, Any]] = []
    torch = _torch()

    if include_optional and hardware.gpu_verified and hardware.supports_bf16:
        try:
            from ..evaluator.torch import TorchEvaluator

            bf16 = TorchEvaluator(
                evaluator.model,
                value_size=evaluator.value_size,
                device=str(evaluator.device),
                precision="bf16",
            )
            measured, results = _measure_mode(
                name="eager-bf16",
                evaluator=bf16,
                requests=requests,
                max_seconds=max_seconds,
                meter=None,
            )
            assert_evaluation_agreement(eager_results, results, rtol=1e-2, atol=1e-3)
            modes.append(measured)
            optional_candidates.append(("bf16", bf16))
        except Exception as error:
            unavailable["eager-bf16"] = f"{type(error).__name__}: {error}"
    elif include_optional:
        unavailable["eager-bf16"] = "CUDA BF16 support is unavailable"

    if include_optional and hardware.gpu_verified:
        compile_sources = [("fp32", evaluator), *optional_candidates]
        for precision, source in compile_sources:
            mode_name = f"compiled-{precision}-reduce-overhead"
            try:
                from ..evaluator.torch import TorchEvaluator

                compiled_model = torch.compile(source.model, mode="reduce-overhead")
                compiled = TorchEvaluator(
                    compiled_model,
                    value_size=source.value_size,
                    device=str(source.device),
                    precision=source.precision,
                )
                measured, results = _measure_mode(
                    name=mode_name,
                    evaluator=compiled,
                    requests=requests,
                    max_seconds=max_seconds,
                    meter=None,
                )
                if measured.peak_memory_bytes is None:
                    raise RuntimeError("compiled CUDA peak-memory measurement is unavailable")
                assert_evaluation_agreement(eager_results, results, rtol=1e-2, atol=1e-3)
                modes.append(measured)
            except Exception as error:
                unavailable[mode_name] = f"{type(error).__name__}: {error}"
    elif include_optional:
        unavailable["compiled-fp32-reduce-overhead"] = "CUDA is unavailable"

    return ProfileReport(
        max_seconds=max(1, math.ceil(max_seconds)),
        hardware=hardware,
        modes=tuple(modes),
        stage_timings=meter.summaries(),
        unavailable_modes=unavailable,
    )


__all__ = [
    "HardwareProfile",
    "ModeProfile",
    "ProfileReport",
    "TimingSummary",
    "assert_evaluation_agreement",
    "detect_hardware",
    "profile_evaluator",
]
