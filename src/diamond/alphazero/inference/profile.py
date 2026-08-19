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
from threading import Barrier, Thread
from time import monotonic
from typing import Any, Callable, Mapping, Sequence


from .coordinator import InferenceConfig, InferenceCoordinator, InferenceMetrics
from .protocol import InferenceRequest, InferenceResponse, ModelKey
from .remote import RemoteEvaluator


_EVALUATOR_STAGES = ("admission", "queue_to_dispatch", "inference", "response")
_SUPPLIED_STAGES = (
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
    mean_batch_size: float
    max_batch_size: int

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
            "mean_batch_size": self.mean_batch_size,
            "max_batch_size": self.max_batch_size,
        }


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """JSON-safe result of one bounded profile session."""

    max_seconds: int
    hardware: HardwareProfile
    modes: tuple[ModeProfile, ...]
    stage_timings: Mapping[str, TimingSummary]
    unavailable_stages: Mapping[str, str]
    unavailable_modes: Mapping[str, str]

    @classmethod
    def empty(cls, *, max_seconds: int) -> ProfileReport:
        return cls(
            max_seconds=max_seconds,
            hardware=detect_hardware(),
            modes=(),
            stage_timings={},
            unavailable_stages={
                name: "profile operation was not run"
                for name in (*_EVALUATOR_STAGES, *_SUPPLIED_STAGES)
            },
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
            "unavailable_stages": dict(self.unavailable_stages),
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
        result = operation()
        self.samples[name].append(monotonic() - started)
        return result

    def summaries(self) -> dict[str, TimingSummary]:
        return {
            name: TimingSummary.from_samples(samples)
            for name, samples in self.samples.items()
            if samples
        }


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
    model_key: ModelKey,
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

    client_count = 4

    class _ProfileBatchEvaluator:
        def evaluate(
            self, envelopes: tuple[InferenceRequest, ...]
        ) -> tuple[InferenceResponse, ...]:
            results = evaluator.evaluate(
                tuple(envelope.to_eval_request() for envelope in envelopes)
            )
            _synchronize(evaluator)
            return tuple(
                InferenceResponse.from_eval_result(envelope, result)
                for envelope, result in zip(envelopes, results, strict=True)
            )

    coordinator = InferenceCoordinator(
        _ProfileBatchEvaluator(),
        InferenceConfig(
            max_batch_size=client_count * len(requests),
            max_wait_ms=5,
            request_queue_capacity=client_count * len(requests) * 2,
            response_timeout_s=max(5.0, max_seconds * 4),
        ),
    )
    remotes = tuple(
        RemoteEvaluator(coordinator, model_key=model_key, client_id=f"profile-client-{index}")
        for index in range(client_count)
    )
    coordinator.start()
    try:
        rounds = 0
        while rounds == 0 or (monotonic() < deadline and rounds < 64):
            barrier = Barrier(client_count + 1)
            outputs: list[tuple[Any, ...] | None] = [None] * client_count
            errors: list[BaseException] = []

            def evaluate_client(index: int) -> None:
                try:
                    barrier.wait()
                    outputs[index] = remotes[index].evaluate(requests)
                except BaseException as error:
                    errors.append(error)

            threads = tuple(
                Thread(target=evaluate_client, args=(index,), daemon=True)
                for index in range(client_count)
            )
            for thread in threads:
                thread.start()
            call_started = monotonic()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=coordinator.config.response_timeout_s)
            if any(thread.is_alive() for thread in threads):
                raise TimeoutError("profile clients did not shut down cleanly")
            if errors:
                raise errors[0]
            if any(output is None for output in outputs):
                raise RuntimeError("profile client returned no result")
            latency = monotonic() - call_started
            if rounds == 0:
                first_results = tuple(outputs[0] or ())
                first_call = latency
            else:
                steady_samples.append(latency)
            latency_samples.append(latency)
            rounds += 1
            calls += client_count
    finally:
        coordinator.stop()

    metrics: InferenceMetrics = coordinator.metrics
    batch_samples.extend(metrics.inference_latency_samples)
    if meter is not None:
        meter.samples["admission"].extend(metrics.admission_latency_samples)
        meter.samples["queue_to_dispatch"].extend(
            metrics.queue_to_dispatch_latency_samples
        )
        meter.samples["inference"].extend(metrics.inference_latency_samples)
        meter.samples["response"].extend(metrics.response_latency_samples)

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
            mean_batch_size=(
                sum(metrics.batch_sizes) / len(metrics.batch_sizes)
                if metrics.batch_sizes
                else 0.0
            ),
            max_batch_size=max(metrics.batch_sizes, default=0),
        ),
        first_results,
    )


def profile_evaluator(
    evaluator: Any,
    requests: tuple[Any, ...],
    *,
    max_seconds: float,
    include_optional: bool = True,
    stage_operations: Mapping[str, Callable[[], object]] | None = None,
    model_key: ModelKey | None = None,
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
    if model_key is None:
        model_name = "Soo" if len(requests[0].canonical_player_ids) == 2 else "Min"
        model_key = ModelKey(model_name, "0.0.0", "0" * 64)
    meter = _StageMeter()
    eager, eager_results = _measure_mode(
        name="eager-fp32",
        evaluator=evaluator,
        requests=requests,
        max_seconds=max_seconds,
        meter=meter,
        model_key=model_key,
    )
    modes = [eager]
    unavailable: dict[str, str] = {}
    unavailable_stages: dict[str, str] = {}
    optional_candidates: list[tuple[str, Any]] = []
    torch = _torch()

    operations = dict(stage_operations or {})
    unexpected_stages = set(operations) - set(_SUPPLIED_STAGES)
    if unexpected_stages:
        raise ValueError(
            f"unsupported profile stages: {', '.join(sorted(unexpected_stages))}"
        )
    for stage in _SUPPLIED_STAGES:
        operation = operations.get(stage)
        if operation is None:
            unavailable_stages[stage] = "operation was not supplied"
            continue
        try:
            meter.measure(stage, operation)
        except Exception as error:
            unavailable_stages[stage] = f"{type(error).__name__}: {error}"

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
                model_key=model_key,
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
                    model_key=model_key,
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
        unavailable_stages=unavailable_stages,
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
