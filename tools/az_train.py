"""Bootstrap (B0) / normal (A0) AlphaZero training loop, on CPU or GPU.

Composes the repository's existing durable stages -- the self-play worker pool,
the persistent replay store, the AlphaZero trainer and checkpoint persistence --
and deliberately bypasses only the promotion arena and the rating benchmark.

Why the bypass: with an untrained network and ``bootstrap_prior = none`` (which
arena and rating correctly always use), evaluation games never reach a terminal
state and burn the full 2000-move cap.  One measured Soo arena costs ~0.9 h and
its rating benchmark ~0.4 h, against ~1 min of self-play per iteration, so the
full ``cli train`` path yields ~2 iterations per 8 hours instead of hundreds.
This runner therefore preserves self-play, replay, trainer, checkpoint and
resume safety, and records that it performs no promotion or rating.

Learning semantics are untouched: the bootstrap prior replaces the policy prior
only, values remain the real network values, terminal targets keep their
authoritative Soo/Min semantics, and aborted games contribute zero samples.

The device comes from the configuration, so the same execution path serves CPU
and CUDA runs; ``tools/cpu_b0_train.py`` remains as an alias for the original
CPU command.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from diamond.alphazero.checkpoint import load_checkpoint, save_checkpoint
from diamond.alphazero.hardware import available_cpu_count, resolve_worker_count
from diamond.alphazero.inference.summary import summarize_metrics
from diamond.alphazero.run_migrate import migrate_run_to_device
from diamond.alphazero.config import (
    BOOTSTRAP_PRIORS,
    MCTSConfig,
    NetworkConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from diamond.alphazero.identity import (
    MIN_MODEL_NAME,
    SOO_MODEL_NAME,
    CheckpointCompatibilitySpec,
)
from diamond.alphazero.inference.coordinator import InferenceConfig, InferenceCoordinator
from diamond.alphazero.inference.model_pool import InferenceModelPool
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.orchestration.replay_store import PersistentReplayStore
from diamond.alphazero.orchestration.selfplay_workers import (
    SelfPlayJob,
    SelfPlayWorkerPool,
)
from diamond.alphazero.trainer import AlphaZeroTrainer
from diamond.game.state import build_players, initial_state

ACTION_SIZE = 73 * 73


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_compatibility(config: dict) -> CheckpointCompatibilitySpec:
    network = NetworkConfig(**config["network"])
    model_name = config["model_name"]
    version = config["model_version"]
    if model_name == SOO_MODEL_NAME:
        return CheckpointCompatibilitySpec.soo(model_version=version, network_config=network)
    if model_name == MIN_MODEL_NAME:
        return CheckpointCompatibilitySpec.min(model_version=version, network_config=network)
    raise ValueError(f"unsupported model: {model_name}")


def new_model(compatibility: CheckpointCompatibilitySpec) -> torch.nn.Module:
    network = compatibility.network_config
    if compatibility.identity.model_name == SOO_MODEL_NAME:
        return SooModel(network, model_version=compatibility.identity.model_version)
    return MinModel(network, model_version=compatibility.identity.model_version)


# --------------------------------------------------------------------------
# durable loop state (small, JSON, atomically replaced)
# --------------------------------------------------------------------------


class LoopState:
    """Iteration counter plus cumulative counters, resumable across restarts."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "iteration": 0,
                "training_step": 0,
                "attempted": 0,
                "completed": 0,
                "aborted": 0,
                "samples_generated": 0,
                "move_counts": [],
                "abort_reasons": {},
                "phase": "B0",
                "elapsed_s": 0.0,
            }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


def append_ledger(path: Path, record: dict) -> None:
    """Append one JSON line; the ledger is evidence, never rewritten in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# run reporting
# --------------------------------------------------------------------------


def describe_environment(device: str) -> dict:
    """Record what actually ran, including GPU facts, with no new dependencies."""
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    vram_total_bytes = None
    if cuda_available:
        try:
            index = torch.device(device).index or 0
            properties = torch.cuda.get_device_properties(index)
            gpu_name = properties.name
            vram_total_bytes = int(properties.total_memory)
        except (AssertionError, RuntimeError, ValueError):
            # Never let telemetry stop a training run.
            gpu_name = None
            vram_total_bytes = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": device,
        "gpu_name": gpu_name,
        "vram_total_bytes": vram_total_bytes,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "available_cpus": available_cpu_count(),
        "platform": platform.platform(),
    }


def percentile(values, q: float):
    """Nearest-rank percentile; ``None`` when there is nothing to rank."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[index]


def throughput_summary(
    *, attempted: int, completed: int, samples: int, train_steps: int, elapsed_s: float
) -> dict:
    """Per-hour rates so CPU and GPU sessions compare directly."""
    if elapsed_s <= 0:
        per_hour = lambda total: None  # noqa: E731
    else:
        per_hour = lambda total: total * 3600.0 / elapsed_s  # noqa: E731
    return {
        "games_per_hour": per_hour(attempted),
        "completed_games_per_hour": per_hour(completed),
        "samples_per_hour": per_hour(samples),
        "training_steps_per_hour": per_hour(train_steps),
    }


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument(
        "--bootstrap-prior",
        default=None,
        help="Override the config prior; use 'none' for the A0 phase.",
    )
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--train-steps-per-iteration", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="Parent-side torch threads.")
    parser.add_argument("--phase", default=None, help="Ledger label, e.g. B0 or A0.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Explicit self-play worker count; default is available CPUs minus two.",
    )
    parser.add_argument(
        "--per-game-seconds",
        type=float,
        default=None,
        help="Override self_play.max_game_seconds for this session.",
    )
    parser.add_argument(
        "--migrate-device",
        action="store_true",
        help=(
            "Migrate this run's latest.pt onto the configured device, once. "
            "A no-op when it already records that device."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.bootstrap_prior is not None:
        if args.bootstrap_prior not in BOOTSTRAP_PRIORS:
            raise SystemExit(f"unknown bootstrap prior: {args.bootstrap_prior}")
        config["self_play"]["bootstrap_prior"] = args.bootstrap_prior
    if args.simulations is not None:
        config["mcts"]["simulations"] = args.simulations
    if args.per_game_seconds is not None:
        config["self_play"]["max_game_seconds"] = args.per_game_seconds

    model_name = config["model_name"]
    compatibility = build_compatibility(config)
    mcts_config = MCTSConfig(**config["mcts"])
    selfplay_config = SelfPlayConfig(**config["self_play"])
    training_config = TrainingConfig(**config["training"])
    workers = config["workers"]
    replay_config = config["replay"]
    run_seed = config["run_seed"]
    # An explicit count always wins; otherwise use every CPU this process may
    # actually run on, less two reserved for the parent loop and inference.
    worker_count = resolve_worker_count(
        args.workers if args.workers is not None else workers.get("worker_count")
    )

    torch.set_num_threads(args.threads)

    run_root = Path(args.runtime_dir) / model_name.lower() / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_root / "ledger.jsonl"
    state = LoopState(run_root / "loop_state.json")
    phase = args.phase or state.data.get("phase", "B0")
    state.data["phase"] = phase

    # Pin the exact runtime config for this run so the ledger is reproducible.
    config_path = run_root / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    # ---- trainer + checkpoint (resume from the latest durable checkpoint) ----
    trainer = AlphaZeroTrainer(new_model(compatibility), compatibility, training_config)
    latest = run_root / "latest.pt"
    migration = None
    if latest.exists():
        if args.migrate_device:
            # Idempotent: returns None when the checkpoint already records this
            # device, so leaving the flag in a launch script is safe.
            migration = migrate_run_to_device(
                run_root=run_root,
                trainer=trainer,
                expected=compatibility,
                operation_id=f"{args.run_id}-migrate",
            )
            if migration is not None:
                print(
                    f"[migrate] {migration.source_device} -> {migration.target_device} "
                    f"at training_step={migration.training_step}; "
                    f"backup {migration.backup_path}",
                    flush=True,
                )
        if migration is None:
            load_checkpoint(latest, trainer, expected=compatibility)
        print(f"[resume] loaded {latest} at training_step={trainer.training_step}", flush=True)
    else:
        save_checkpoint(latest, trainer, operation_id=f"{args.run_id}-init")
        print(f"[init] wrote initial checkpoint {latest}", flush=True)

    replay = PersistentReplayStore(
        run_root / "replay",
        compatibility,
        capacity=replay_config["capacity"],
        seed=replay_config["seed"],
    )

    environment = describe_environment(training_config.device) | {
        "resolved_workers": worker_count,
    }
    append_ledger(
        ledger_path,
        {
            "event": "run_start",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": args.run_id,
            "model": model_name,
            "phase": phase,
            "bootstrap_prior": selfplay_config.bootstrap_prior,
            "simulations": mcts_config.simulations,
            "worker_count": worker_count,
            "games_per_iteration": workers["games_per_iteration"],
            "batch_size": training_config.batch_size,
            "train_steps_per_iteration": args.train_steps_per_iteration,
            "hours": args.hours,
            "environment": environment,
            "promotion_and_rating": "bypassed (see module docstring)",
            "resumed_training_step": trainer.training_step,
            "max_game_seconds": selfplay_config.max_game_seconds,
            "device_migration": migration.to_payload() if migration else None,
        },
    )
    print(
        f"[start] {model_name} run={args.run_id} phase={phase} "
        f"prior={selfplay_config.bootstrap_prior} sims={mcts_config.simulations} "
        f"budget={args.hours}h",
        flush=True,
    )

    stop = {"requested": False}

    def request_stop(signum, frame):  # noqa: ARG001
        # Finish the current iteration, then exit cleanly with state persisted.
        stop["requested"] = True
        print("\n[signal] stop requested; finishing current iteration", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    model_pool = InferenceModelPool(device=training_config.device)
    inference_config = InferenceConfig(**config["inference"])
    deadline = time.perf_counter() + args.hours * 3600.0
    session_start = time.perf_counter()

    while time.perf_counter() < deadline and not stop["requested"]:
        iteration = state.data["iteration"]
        iteration_start = time.perf_counter()

        # Self-play always runs against the current durable checkpoint, so the
        # data provenance and the trained weights can never disagree.
        model_key = model_pool.activate_checkpoint(latest, expected=compatibility)
        players = build_players(compatibility.identity.player_count)
        start_state = initial_state(players)
        jobs = tuple(
            SelfPlayJob(
                run_seed=run_seed,
                iteration=iteration,
                game_index=index,
                retry_id=workers["retry_id"],
                model_key=model_key,
                compatibility=compatibility,
                players=players,
                initial_state=start_state,
                mcts_config=mcts_config,
                selfplay_config=selfplay_config,
            )
            for index in range(workers["games_per_iteration"])
        )

        coordinator = InferenceCoordinator(model_pool, inference_config)
        coordinator.start()
        try:
            episodes = SelfPlayWorkerPool(
                coordinator,
                worker_count=worker_count,
                # A game now aborts on its own budget, so the pool guard is only
                # for a dead child or broken IPC.  With no budget configured we
                # keep the previous global deadline.
                per_game_timeout_s=selfplay_config.max_game_seconds,
                worker_timeout_s=(
                    None
                    if selfplay_config.max_game_seconds is not None
                    else max(600.0, inference_config.response_timeout_s * 4)
                ),
            ).run(jobs)
        except RuntimeError as error:
            # A stop signal terminates the worker children mid-episode.  That is
            # an intended shutdown, not a defect: the previous iteration is
            # already durable, so exit cleanly instead of reporting a crash.
            if stop["requested"]:
                print(f"[stop] self-play interrupted during shutdown: {error}", flush=True)
                break
            raise
        finally:
            inference_metrics = coordinator.metrics
            coordinator.stop()
        selfplay_s = time.perf_counter() - iteration_start

        completed = aborted = new_samples = 0
        move_counts: list[int] = []
        iteration_aborts: dict[str, int] = {}
        for episode in episodes:
            # ingest_episode is idempotent on game_id, so a resumed iteration
            # cannot double-count an episode it already persisted.
            replay.ingest_episode(episode)
            if episode.completed:
                completed += 1
                new_samples += len(episode.samples)
                move_counts.append(episode.move_count)
            else:
                aborted += 1
                reason = episode.aborted_reason or "unknown"
                iteration_aborts[reason] = iteration_aborts.get(reason, 0) + 1
                state.data["abort_reasons"][reason] = (
                    state.data["abort_reasons"].get(reason, 0) + 1
                )

        state.data["attempted"] += len(episodes)
        state.data["completed"] += completed
        state.data["aborted"] += aborted
        state.data["samples_generated"] += new_samples
        state.data["move_counts"].extend(move_counts)

        # ---- training ----
        replay_size = len(replay.load_buffer())
        metrics_list = []
        train_start = time.perf_counter()
        if replay_size >= training_config.batch_size:
            buffer = replay.load_buffer()
            for _ in range(args.train_steps_per_iteration):
                samples = replay.sample(training_config.batch_size)
                metrics = trainer.train_batch(buffer.collate(samples, action_size=ACTION_SIZE))
                if not all(
                    map(
                        lambda value: value == value and abs(value) != float("inf"),
                        (metrics.total_loss, metrics.policy_loss, metrics.value_loss),
                    )
                ):
                    raise SystemExit(f"non-finite loss detected: {asdict(metrics)}")
                metrics_list.append(asdict(metrics))
        train_s = time.perf_counter() - train_start

        # ---- checkpoint every iteration; the run is never all-or-nothing ----
        if metrics_list:
            save_checkpoint(
                latest, trainer, operation_id=f"{args.run_id}-i{iteration:06d}"
            )
            archive = checkpoints_dir / (
                f"{phase}-i{iteration:06d}-step{trainer.training_step:09d}.pt"
            )
            save_checkpoint(archive, trainer, operation_id=f"{args.run_id}-i{iteration:06d}")
        else:
            archive = None

        state.data["iteration"] = iteration + 1
        state.data["training_step"] = trainer.training_step
        state.data["elapsed_s"] = time.perf_counter() - session_start
        state.save()

        elapsed = time.perf_counter() - session_start
        median_moves = (
            sorted(move_counts)[len(move_counts) // 2] if move_counts else None
        )
        record = {
            "event": "iteration",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model_name,
            "run_id": args.run_id,
            "phase": phase,
            "iteration": iteration,
            "training_step": trainer.training_step,
            "bootstrap_prior": selfplay_config.bootstrap_prior,
            "simulations": mcts_config.simulations,
            "worker_count": worker_count,
            "attempted": len(episodes),
            "completed": completed,
            "aborted": aborted,
            "abort_reasons": iteration_aborts,
            "median_moves": median_moves,
            "p90_moves": percentile(move_counts, 0.9),
            "samples_new": new_samples,
            "replay_size": len(replay.load_buffer()),
            "metrics": metrics_list[-1] if metrics_list else None,
            "selfplay_s": round(selfplay_s, 2),
            "train_s": round(train_s, 2),
            "elapsed_s": round(elapsed, 1),
            "checkpoint": str(archive) if archive else None,
            "throughput": throughput_summary(
                attempted=len(episodes),
                completed=completed,
                samples=new_samples,
                train_steps=len(metrics_list),
                elapsed_s=time.perf_counter() - iteration_start,
            ),
            "inference": summarize_metrics(inference_metrics, elapsed_s=selfplay_s),
        }
        append_ledger(ledger_path, record)
        loss = record["metrics"]["total_loss"] if record["metrics"] else None
        print(
            f"[i{iteration:04d}] {completed}/{len(episodes)} done "
            f"median_moves={median_moves} samples+{new_samples} "
            f"replay={record['replay_size']} step={trainer.training_step} "
            f"loss={loss if loss is None else round(loss, 4)} "
            f"sp={selfplay_s:.0f}s tr={train_s:.0f}s "
            f"elapsed={elapsed / 3600:.2f}h",
            flush=True,
        )

    append_ledger(
        ledger_path,
        {
            "event": "run_end",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": args.run_id,
            "model": model_name,
            "phase": phase,
            "iterations": state.data["iteration"],
            "training_step": trainer.training_step,
            "attempted": state.data["attempted"],
            "completed": state.data["completed"],
            "aborted": state.data["aborted"],
            "samples_generated": state.data["samples_generated"],
            "replay_size": len(replay.load_buffer()),
            "abort_reasons": state.data["abort_reasons"],
            "elapsed_s": round(time.perf_counter() - session_start, 1),
            "stopped_early": stop["requested"],
        },
    )
    print(
        f"[done] {model_name} iterations={state.data['iteration']} "
        f"step={trainer.training_step} replay={len(replay.load_buffer())} "
        f"elapsed={(time.perf_counter() - session_start) / 3600:.2f}h",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
