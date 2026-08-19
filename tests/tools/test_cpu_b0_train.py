"""The CPU session runner must preserve durable state and learning semantics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from diamond.alphazero.identity import MIN_MODEL_NAME, SOO_MODEL_NAME

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, TOOLS / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load("cpu_b0_train")


def runtime_config(name: str) -> dict:
    path = Path(__file__).resolve().parents[2] / "runtime" / "configs" / name
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("filename", "model_name", "player_count"),
    (("soo-cpu8h.json", SOO_MODEL_NAME, 2), ("min-cpu8h.json", MIN_MODEL_NAME, 3)),
)
def test_runtime_configs_keep_authoritative_identity(
    runner, filename: str, model_name: str, player_count: int
) -> None:
    config = runtime_config(filename)
    compatibility = runner.build_compatibility(config)
    assert compatibility.identity.model_name == model_name
    assert compatibility.identity.player_count == player_count
    # The CPU session may retune search and workers, never the learning target.
    assert config["training"]["device"] == "cpu"
    assert config["self_play"]["max_moves"] == 2000


def test_runtime_configs_only_retune_search_and_workers() -> None:
    """Everything the blueprint calls immutable must match the checked-in reference."""
    root = Path(__file__).resolve().parents[2]
    for model in ("soo", "min"):
        reference = json.loads(
            (root / "configs" / "alphazero" / f"{model}-bootstrap.json").read_text(
                encoding="utf-8"
            )
        )
        runtime = runtime_config(f"{model}-cpu8h.json")
        assert runtime["network"] == reference["network"]
        assert runtime["self_play"] == reference["self_play"]
        assert runtime["replay"] == reference["replay"]
        assert runtime["arena"] == reference["arena"]
        assert runtime["model_version"] == reference["model_version"]
        # Only the tuned knobs differ.
        assert runtime["mcts"]["simulations"] == 32
        assert runtime["training"]["device"] == "cpu"


def test_loop_state_round_trips_and_is_atomic(runner, tmp_path: Path) -> None:
    path = tmp_path / "loop_state.json"
    state = runner.LoopState(path)
    state.data["iteration"] = 5
    state.data["samples_generated"] = 1234
    state.save()

    reloaded = runner.LoopState(path)
    assert reloaded.data["iteration"] == 5
    assert reloaded.data["samples_generated"] == 1234
    # The temporary file must not survive an atomic replace.
    assert not path.with_suffix(".tmp").exists()


def test_ledger_appends_rather_than_overwrites(runner, tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    runner.append_ledger(path, {"event": "run_start"})
    runner.append_ledger(path, {"event": "iteration", "iteration": 0})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["run_start", "iteration"]


def test_new_model_matches_declared_compatibility(runner) -> None:
    for filename in ("soo-cpu8h.json", "min-cpu8h.json"):
        compatibility = runner.build_compatibility(runtime_config(filename))
        model = runner.new_model(compatibility)
        assert model.identity == compatibility.identity
        assert model.config == compatibility.network_config
