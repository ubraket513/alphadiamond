from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.inference.profile import detect_hardware, profile_evaluator


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
        self.batch_sizes.append(batch)
        return torch.zeros((batch, 4)), torch.zeros((batch, 1))


def _request() -> EvalRequest:
    return EvalRequest(
        node_features=((0.0, 0.0), (0.0, 0.0)),
        legal_action_ids=(0, 2, 3),
        canonical_player_ids=(1, 2),
    )


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def test_cpu_profile_omits_unsupplied_stage_durations_and_reports_unavailable() -> None:
    model = _Model()
    report = profile_evaluator(
        TorchEvaluator(model, value_size=1),
        (_request(),),
        max_seconds=0.01,
        include_optional=False,
    )

    payload = report.to_dict()

    assert report.modes[0].name == "eager-fp32"
    assert report.modes[0].states_per_second > 0
    assert report.modes[0].calls_per_second > 0
    assert set(report.modes[0].batch_seconds.to_dict()) == {
        "samples",
        "mean_s",
        "p50_s",
        "p95_s",
    }
    assert set(report.modes[0].latency_seconds.to_dict()) == {
        "samples",
        "mean_s",
        "p50_s",
        "p95_s",
    }
    assert set(report.stage_timings) == {
        "admission",
        "queue_to_dispatch",
        "inference",
        "response",
    }
    assert "queue_wait" not in report.stage_timings
    assert all(summary.samples > 0 for summary in report.stage_timings.values())
    assert report.modes[0].max_batch_size > 1
    assert report.modes[0].mean_batch_size > 1.0
    assert max(model.batch_sizes) > 1
    assert set(report.unavailable_stages) == {
        "self_play",
        "replay_collation",
        "training",
    }
    assert all(
        stage not in payload["stage_timings"] for stage in report.unavailable_stages
    )
    assert isinstance(payload["hardware"]["gpu_verified"], bool)
    assert _all_finite(payload)
    assert json.loads(report.to_json()) == payload


def test_profile_times_each_supplied_stage_operation_once() -> None:
    invoked: list[str] = []

    def operation(name: str):
        return lambda: invoked.append(name)

    report = profile_evaluator(
        TorchEvaluator(_Model(), value_size=1),
        (_request(),),
        max_seconds=0.01,
        include_optional=False,
        stage_operations={
            "self_play": operation("self_play"),
            "replay_collation": operation("replay_collation"),
            "training": operation("training"),
        },
    )

    assert invoked == ["self_play", "replay_collation", "training"]
    assert set(report.stage_timings) == {
        "admission",
        "queue_to_dispatch",
        "inference",
        "response",
        "self_play",
        "replay_collation",
        "training",
    }
    assert report.stage_timings["self_play"].samples == 1
    assert report.stage_timings["replay_collation"].samples == 1
    assert report.stage_timings["training"].samples == 1
    assert report.unavailable_stages == {}


def test_cuda_absence_does_not_fabricate_gpu_rows(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    hardware = detect_hardware()

    assert hardware.gpu_verified is False
    assert hardware.gpus == ()
