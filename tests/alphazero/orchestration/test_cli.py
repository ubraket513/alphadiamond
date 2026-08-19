from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from diamond.alphazero.orchestration import cli
from diamond.alphazero.checkpoint import save_checkpoint
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.profile import ProfileReport
from diamond.alphazero.network import SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer


class _Services:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def train(self, *, model_name: str, run_id: str) -> dict[str, object]:
        self.calls.append(("train", model_name, run_id))
        return {"stage": "COMPLETE"}

    def resume(self, *, model_name: str, run_id: str) -> dict[str, object]:
        self.calls.append(("resume", model_name, run_id))
        return {"stage": "COMPLETE"}

    def benchmark(self, *, model_name: str, run_id: str) -> dict[str, object]:
        self.calls.append(("benchmark", model_name, run_id))
        return {"events": 1}

    def leaderboard(self, *, model_name: str, run_id: str) -> dict[str, object]:
        self.calls.append(("leaderboard", model_name, run_id))
        return {"entries": []}

    def profile(self, *, model_name: str, max_seconds: int) -> ProfileReport:
        self.calls.append(("profile", model_name, str(max_seconds)))
        return ProfileReport.empty(max_seconds=max_seconds)


def _run(
    capsys,
    services: _Services,
    *argv: str,
) -> tuple[int, dict[str, object]]:
    arguments = [
        *argv,
        "--runtime-dir",
        "runtime",
        "--model",
        "Soo",
        "--run-id",
        "run-1",
    ]
    arguments.extend(["--config", "config.json", "--checkpoint", "checkpoint.pt"])
    exit_code = cli.main(
        arguments,
        services_factory=lambda _root, _model, _config, _checkpoint: services,
    )
    return exit_code, json.loads(capsys.readouterr().out)


def test_commands_dispatch_services_with_machine_readable_success(capsys) -> None:
    services = _Services()

    for command in ("train", "resume", "benchmark", "leaderboard"):
        exit_code, payload = _run(capsys, services, command)

        assert exit_code == cli.EXIT_OK
        assert payload["command"] == command
        assert payload["status"] == "ok"

    assert services.calls == [
        ("train", "Soo", "run-1"),
        ("resume", "Soo", "run-1"),
        ("benchmark", "Soo", "run-1"),
        ("leaderboard", "Soo", "run-1"),
    ]


def test_profile_dispatches_services_and_emits_a_bounded_report(capsys) -> None:
    services = _Services()

    exit_code = cli.main(
        [
            "profile",
            "--seconds",
            "1",
            "--runtime-dir",
            "runtime",
            "--model",
            "Soo",
            "--config",
            "config.json",
            "--checkpoint",
            "checkpoint.pt",
        ],
        services_factory=lambda _root, _model, _config, _checkpoint: services,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_OK
    assert payload["command"] == "profile"
    assert payload["max_seconds"] == 1
    assert payload["status"] == "ok"
    assert payload["modes"] == []
    assert services.calls == [("profile", "Soo", "1")]


def test_runtime_errors_have_stable_machine_readable_exit_code(capsys) -> None:
    exit_code = cli.main(
        [
            "train",
            "--runtime-dir",
            "runtime",
            "--model",
            "Soo",
            "--run-id",
            "run-1",
            "--config",
            "config.json",
            "--checkpoint",
            "checkpoint.pt",
        ],
        services_factory=lambda _root, _model, _config, _checkpoint: (
            _ for _ in ()
        ).throw(ValueError("bad runtime")),
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_RUNTIME_ERROR
    assert payload == {
        "command": "train",
        "error": "bad runtime",
        "status": "error",
    }


def test_argument_errors_have_a_distinct_stable_exit_code(capsys) -> None:
    exit_code = cli.main(["train"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_ARGUMENT_ERROR
    assert payload["command"] == "train"
    assert payload["status"] == "error"


def test_production_train_requires_explicit_config_and_checkpoint_before_writes(
    tmp_path: Path, capsys
) -> None:
    runtime = tmp_path / "runtime"

    exit_code = cli.main(
        [
            "train",
            "--runtime-dir",
            str(runtime),
            "--model",
            "Soo",
            "--run-id",
            "run-1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_ARGUMENT_ERROR
    assert payload["status"] == "error"
    assert "--config" in payload["error"]
    assert "--checkpoint" in payload["error"]
    assert not runtime.exists()


def _production_config(*, model_version: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_name": "Soo",
        "model_version": model_version,
        "network": {"width": 8, "residual_blocks": 1},
        "mcts": {
            "simulations": 1,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "seed": 7,
        },
        "self_play": {
            "max_moves": 2,
            "temperature_moves": 0,
            "temperature": 1.0,
            "seed": 7,
        },
        "replay": {"capacity": 8, "seed": 7},
        "training": {
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "device": "cpu",
            "seed": 7,
        },
        "arena": {
            "games": 4,
            "seed": 7,
            "max_moves": 2,
            "promotion_threshold": 0.5,
        },
        "workers": {
            "worker_count": 1,
            "games_per_iteration": 1,
            "retry_id": "attempt-0",
        },
        "inference": {
            "max_batch_size": 2,
            "max_wait_ms": 2,
            "request_queue_capacity": 4,
            "response_timeout_s": 5.0,
        },
        "benchmark": {
            "opening_count": 1,
            "opening_max_depth": 0,
            "opening_seed": 7,
            "opening_suite_version": "production-openings-v1",
        },
        "run_seed": 7,
    }


def test_incompatible_production_checkpoint_fails_before_runtime_writes(
    tmp_path: Path, capsys
) -> None:
    runtime = tmp_path / "runtime"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(_production_config(model_version="2.0.0")), encoding="utf-8"
    )
    checkpoint_path = tmp_path / "bootstrap.pt"
    checkpoint_spec = CheckpointCompatibilitySpec.soo(
        model_version="1.0.0",
        network_config=NetworkConfig(width=8, residual_blocks=1),
    )
    save_checkpoint(
        checkpoint_path,
        AlphaZeroTrainer(
            SooModel(checkpoint_spec.network_config, model_version="1.0.0"),
            checkpoint_spec,
            TrainingConfig(batch_size=1, weight_decay=0.0),
        ),
    )

    exit_code = cli.main(
        [
            "train",
            "--runtime-dir",
            str(runtime),
            "--model",
            "Soo",
            "--run-id",
            "run-1",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_RUNTIME_ERROR
    assert payload["status"] == "error"
    assert "model_version" in payload["error"]
    assert not runtime.exists()


@pytest.mark.parametrize("run_id", ["../escape", r"..\escape", "C:/escape", "/escape"])
def test_production_run_id_rejects_traversal_and_absolute_paths_without_writes(
    tmp_path: Path, capsys, run_id: str
) -> None:
    runtime = tmp_path / "runtime"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(_production_config(model_version="2.0.0")), encoding="utf-8"
    )
    checkpoint_path = tmp_path / "bootstrap.pt"
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="2.0.0",
        network_config=NetworkConfig(width=8, residual_blocks=1),
    )
    save_checkpoint(
        checkpoint_path,
        AlphaZeroTrainer(
            SooModel(compatibility.network_config, model_version="2.0.0"),
            compatibility,
            TrainingConfig(batch_size=1, weight_decay=0.0),
        ),
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    exit_code = cli.main(
        [
            "train",
            "--runtime-dir",
            str(runtime),
            "--model",
            "Soo",
            "--run-id",
            run_id,
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_RUNTIME_ERROR
    assert "run_id" in payload["error"]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert not runtime.exists()


def test_cli_module_import_does_not_initialize_pyside() -> None:
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import diamond.alphazero.orchestration.cli; "
            "assert not any(name.startswith('PySide6') for name in sys.modules)",
        ],
        env=os.environ | {"PYTHONPATH": source_root},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_cli_module_missing_production_inputs_fails_machine_readably_without_gui() -> None:
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "diamond.alphazero.orchestration.cli", "profile"],
        env=os.environ | {"PYTHONPATH": source_root},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == cli.EXIT_ARGUMENT_ERROR, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "--config" in payload["error"]
    assert "--checkpoint" in payload["error"]
