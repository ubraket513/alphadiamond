from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from diamond.alphazero.orchestration import cli
from diamond.alphazero.inference.profile import ProfileReport


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
    exit_code = cli.main(
        [*argv, "--runtime-dir", "runtime", "--model", "Soo", "--run-id", "run-1"],
        services_factory=lambda _root, _model: services,
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
        ["profile", "--seconds", "1"],
        services_factory=lambda _root, _model: services,
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
        ["train", "--runtime-dir", "runtime", "--model", "Soo", "--run-id", "run-1"],
        services_factory=lambda _root, _model: (_ for _ in ()).throw(ValueError("bad runtime")),
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


def test_cli_module_is_directly_executable_without_gui() -> None:
    pytest.importorskip("torch")
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "diamond.alphazero.orchestration.cli", "profile"],
        env=os.environ | {"PYTHONPATH": source_root},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == cli.EXIT_OK, result.stderr
    assert json.loads(result.stdout)["status"] == "ok"
