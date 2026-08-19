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
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
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


def test_cpu_profile_reports_bounded_measured_stages_and_json() -> None:
    report = profile_evaluator(
        TorchEvaluator(_Model(), value_size=1),
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
        "queue_wait",
        "inference",
        "self_play",
        "replay_collation",
        "training",
    }
    assert all(summary.samples > 0 for summary in report.stage_timings.values())
    assert isinstance(payload["hardware"]["gpu_verified"], bool)
    assert _all_finite(payload)
    assert json.loads(report.to_json()) == payload


def test_cuda_absence_does_not_fabricate_gpu_rows(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    hardware = detect_hardware()

    assert hardware.gpu_verified is False
    assert hardware.gpus == ()
