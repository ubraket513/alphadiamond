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


def _learning_fields(payload: dict) -> dict:
    """The self-play fields that shape a training target, and only those."""
    return {
        key: value
        for key, value in payload["self_play"].items()
        if key != "max_game_seconds"
    }


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
        assert runtime["replay"] == reference["replay"]
        assert runtime["arena"] == reference["arena"]
        assert runtime["model_version"] == reference["model_version"]
        # Every self-play field that shapes a training target is immutable.
        # max_game_seconds is an operational safety limit rather than a learning
        # knob -- it only ever turns a game into a zero-sample abort -- so it is
        # tuned per run alongside simulations and worker_count.
        assert _learning_fields(runtime) == _learning_fields(reference)
        # Only the tuned knobs differ.
        assert runtime["mcts"]["simulations"] == 32
        assert runtime["training"]["device"] == "cpu"
        # The CPU configs predate the field; omitting it must keep working,
        # which is the backward-compatibility guarantee for old configs on disk.
        assert "max_game_seconds" not in runtime["self_play"]


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


@pytest.fixture(scope="module")
def gpu_runner():
    return load("az_train")


def test_the_cpu_alias_still_exposes_the_documented_surface(runner) -> None:
    """tools/cpu_b0_train.py is a documented command; renaming must not break it."""
    for name in ("LoopState", "append_ledger", "build_compatibility", "load_config",
                 "main", "new_model"):
        assert hasattr(runner, name), name


def test_the_hardware_neutral_entry_point_shares_one_implementation(runner) -> None:
    """One execution path, two names -- never two diverging trainers.

    Asserted by provenance rather than object identity: this module loads each
    tool by file path, so the two loads are distinct module objects even though
    they share a single source file.
    """
    import inspect

    for name in ("main", "LoopState", "build_compatibility", "throughput_summary"):
        attribute = getattr(runner, name)
        assert Path(inspect.getfile(attribute)).name == "az_train.py", name


def test_the_rtx3060_config_targets_the_gpu(gpu_runner) -> None:
    config = runtime_config("soo-rtx3060.json")

    assert config["training"]["device"] == "cuda:0"
    assert config["workers"]["games_per_iteration"] == 32
    assert config["self_play"]["max_game_seconds"] == 900.0
    assert config["self_play"]["max_moves"] == 2000
    assert config["inference"]["max_batch_size"] == 32
    assert config["inference"]["max_wait_ms"] in (1, 2)
    # Inference safety is a separate limit from the per-game budget.
    assert config["inference"]["response_timeout_s"] == 600.0
    assert config["mcts"]["simulations"] == 32

    compatibility = gpu_runner.build_compatibility(config)
    assert compatibility.identity.model_name == SOO_MODEL_NAME
    assert compatibility.identity.player_count == 2


def test_the_gpu_config_opts_into_automatic_worker_resolution() -> None:
    gpu = runtime_config("soo-rtx3060.json")
    assert gpu["workers"].get("worker_count") is None

    # The CPU configs stay explicit, so CPU behaviour is unchanged.
    for name in ("soo-cpu8h.json", "min-cpu8h.json"):
        assert runtime_config(name)["workers"]["worker_count"] == 4


def test_the_gpu_config_changes_only_hardware_and_search_knobs() -> None:
    """Learning semantics must not drift just because the device changed."""
    reference = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "configs" / "alphazero" / "soo-bootstrap.json"
        ).read_text(encoding="utf-8")
    )
    cpu = runtime_config("soo-cpu8h.json")
    gpu = runtime_config("soo-rtx3060.json")

    for section in ("network", "replay", "arena"):
        assert gpu[section] == reference[section], section
    assert gpu["model_version"] == reference["model_version"]
    assert gpu["run_seed"] == cpu["run_seed"]
    # Identical self-play semantics; only the wall-clock safety limit is added.
    assert _learning_fields(gpu) == _learning_fields(cpu)
    assert gpu["self_play"]["bootstrap_prior"] == cpu["self_play"]["bootstrap_prior"]
    # Training targets are untouched: same batch size, same optimizer.
    assert gpu["training"]["batch_size"] == cpu["training"]["batch_size"]
    assert gpu["training"]["learning_rate"] == cpu["training"]["learning_rate"]
    assert gpu["training"]["weight_decay"] == cpu["training"]["weight_decay"]


def test_the_gpu_ratio_preserves_games_per_optimizer_update() -> None:
    """32 games / 8 updates is the shipped 16 / 4, not twice the data per step."""
    cpu = runtime_config("soo-cpu8h.json")
    gpu = runtime_config("soo-rtx3060.json")

    cpu_ratio = cpu["workers"]["games_per_iteration"] / 4
    gpu_ratio = gpu["workers"]["games_per_iteration"] / 8
    assert cpu_ratio == gpu_ratio == 4.0


def test_environment_metadata_reports_gpu_facts_without_new_dependencies(
    gpu_runner,
) -> None:
    environment = gpu_runner.describe_environment("cpu")

    for key in ("python", "torch", "torch_cuda", "cuda_available", "device",
                "available_cpus", "platform"):
        assert key in environment, key
    assert isinstance(environment["cuda_available"], bool)
    assert json.dumps(environment)  # ledger-serializable
    if not environment["cuda_available"]:
        assert environment["gpu_name"] is None
        assert environment["vram_total_bytes"] is None


def test_throughput_summary_reports_per_hour_rates(gpu_runner) -> None:
    summary = gpu_runner.throughput_summary(
        attempted=32, completed=31, samples=4000, train_steps=8, elapsed_s=1800.0
    )

    assert summary["games_per_hour"] == pytest.approx(64.0)
    assert summary["completed_games_per_hour"] == pytest.approx(62.0)
    assert summary["samples_per_hour"] == pytest.approx(8000.0)
    assert summary["training_steps_per_hour"] == pytest.approx(16.0)
    assert json.dumps(summary)


def test_throughput_summary_survives_a_zero_length_iteration(gpu_runner) -> None:
    summary = gpu_runner.throughput_summary(
        attempted=0, completed=0, samples=0, train_steps=0, elapsed_s=0.0
    )

    assert summary["games_per_hour"] is None
    assert json.dumps(summary)


def test_move_count_percentiles_describe_the_tail(gpu_runner) -> None:
    assert gpu_runner.percentile([], 0.5) is None
    assert gpu_runner.percentile([5], 0.9) == 5
    assert gpu_runner.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 6
    assert gpu_runner.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 10
