from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.rating.protocol import (
    BenchmarkProtocol,
    EloConfig,
    TrueSkillConfig,
    benchmark_protocol_id,
)


def _compatibility() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.soo(
        model_version="1.2.3",
        network_config=NetworkConfig(width=16, residual_blocks=1),
    )


def _protocol(**changes: object) -> BenchmarkProtocol:
    values: dict[str, object] = {
        "compatibility": _compatibility(),
        "simulations": 200,
        "c_puct": 1.5,
        "dirichlet_epsilon": 0.0,
        "decision_temperature": 0.0,
        "max_game_moves": 2_000,
        "opening_suite_version": "benchmark-openings-v1",
        "opening_suite_hash": "sha256:openings-v1",
        "rating_system_version": "soo-elo-v1",
        "rating_parameters": asdict(EloConfig()),
    }
    values.update(changes)
    return BenchmarkProtocol(**values)


def test_benchmark_protocol_id_uses_deterministic_canonical_json() -> None:
    first = _protocol(rating_parameters={"k_factor": 32.0, "initial_rating": 1000.0})
    second = _protocol(rating_parameters={"initial_rating": 1000.0, "k_factor": 32.0})

    assert first.protocol_id == second.protocol_id
    assert benchmark_protocol_id(first) == first.protocol_id
    assert first.protocol_id == "sha256:1f9f04b7fff545d3e95e19d2176841de481e0f3641aa32eccaaa2978ccc9bf8d"


def test_benchmark_protocol_rejects_root_dirichlet_noise() -> None:
    with pytest.raises(ValueError, match="dirichlet_epsilon"):
        _protocol(dirichlet_epsilon=0.25)


def test_benchmark_protocol_rejects_nonzero_decision_temperature() -> None:
    with pytest.raises(ValueError, match="decision_temperature"):
        _protocol(decision_temperature=0.1)


@pytest.mark.parametrize(
    "changes",
    [
        {"simulations": 201},
        {"c_puct": 1.6},
        {"max_game_moves": 2_001},
        {"inference_numeric_mode": "bf16"},
    ],
)
def test_benchmark_protocol_id_binds_fixed_compute_fields(changes: dict[str, object]) -> None:
    assert _protocol().protocol_id != _protocol(**changes).protocol_id


def test_benchmark_protocol_id_binds_rating_parameters() -> None:
    default = _protocol(rating_parameters=asdict(EloConfig()))
    changed = _protocol(rating_parameters=asdict(replace(EloConfig(), k_factor=16.0)))

    assert default.protocol_id != changed.protocol_id
    assert default.rating_system_version == "soo-elo-v1"
    assert TrueSkillConfig().tau == 0.0
    assert TrueSkillConfig().draw_probability == 0.0


def test_benchmark_protocol_id_separates_changed_openings_and_compatibility() -> None:
    base = _protocol()
    changed_opening = _protocol(opening_suite_hash="sha256:openings-v2")
    changed_compatibility = _protocol(
        compatibility=CheckpointCompatibilitySpec.soo(
            model_version="1.2.3",
            network_config=NetworkConfig(width=32, residual_blocks=1),
        )
    )

    assert base.protocol_id != changed_opening.protocol_id
    assert base.protocol_id != changed_compatibility.protocol_id
