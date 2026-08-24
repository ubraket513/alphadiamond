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
import math
import os
import platform
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from diamond.alphazero.checkpoint import load_checkpoint, save_checkpoint
from diamond.alphazero.config import (
    BOOTSTRAP_PRIORS,
    MCTSConfig,
    NetworkConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from diamond.alphazero.hardware import available_cpu_count, resolve_worker_count
from diamond.alphazero.identity import (
    MIN_MODEL_NAME,
    SOO_MODEL_NAME,
    CheckpointCompatibilitySpec,
)
from diamond.alphazero.inference.coordinator import InferenceConfig, InferenceCoordinator
from diamond.alphazero.inference.model_pool import InferenceModelPool
from diamond.alphazero.inference.summary import summarize_metrics
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.orchestration.replay_store import PersistentReplayStore
from diamond.alphazero.orchestration.selfplay_workers import (
    SelfPlayJob,
    SelfPlayWorkerPool,
)
from diamond.alphazero.run_migrate import migrate_run_to_device
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
        per_hour = lambda total: None
    else:
        per_hour = lambda total: total * 3600.0 / elapsed_s
    return {
        "games_per_hour": per_hour(attempted),
        "completed_games_per_hour": per_hour(completed),
        "samples_per_hour": per_hour(samples),
        "training_steps_per_hour": per_hour(train_steps),
    }


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------



SELFPLAY_BACKENDS = ("native",)
"""Training game execution requires the native extension.

Decision 1 in docs/architecture/decisions.md: the pure-Python search and
self-play backend are no longer supported fallbacks. Keeping a second
implementation of rules, search and self-play alive for the case where the
extension is missing costs more than it returns -- both families already train
on the native pool, and every Python parity gate has been replaced by a C++ one
proven on mutation evidence.

What is required is the compiled extension, not a compiler: a wheel or prebuilt
artifact lets a user train without a toolchain.

``python`` and ``auto`` are still recognised, and both fail with that
explanation rather than silently doing something else. A config that asked for
the Python backend was asking for a specific engine, and quietly giving it
another one would break the rule this setting has always enforced -- that a
run's data is attributable to the engine that produced it."""


class NativeExtensionRequired(SystemExit):
    """Raised when a training command cannot execute games."""


def resolve_selfplay_backend(requested: str, *, max_game_seconds: float | None) -> str:
    """Check the contract, and return the only backend there is."""
    if requested in ("python", "auto"):
        raise NativeExtensionRequired(
            f"selfplay_backend={requested!r} is no longer supported: training game "
            "execution requires the native extension (decision 1 in "
            "docs/architecture/decisions.md). Set selfplay_backend to 'native', "
            "and build the extension with `python tools/build_native.py`."
        )
    if requested != "native":
        raise NativeExtensionRequired(f"unknown selfplay_backend: {requested!r}")

    from diamond.alphazero.native import is_available, native_error

    if not is_available():
        raise NativeExtensionRequired(
            "training game execution requires the native extension, which is not "
            f"importable: {native_error()}. Build it with "
            "`python tools/build_native.py`."
        )
    if max_game_seconds:
        # The native runner bounds a game by moves, not by wall clock. Dropping
        # a configured budget silently would let a run exceed a limit its own
        # config says it respects.
        raise NativeExtensionRequired(
            "self_play.max_game_seconds is not implemented by the native runner; "
            "bound the game by max_moves instead"
        )
    return "native"


def _mean_batch(metrics: dict) -> float:
    sizes = metrics.get("batch_sizes") or []
    return sum(sizes) / len(sizes) if sizes else 0.0


def run_self_play(
    *,
    backend: str,
    jobs,
    model_pool,
    model_key,
    inference_config,
    selfplay_config,
    worker_count: int,
    device: str,
    native_lanes: int,
    native_max_wait_us: int,
    native_simulations_late: int = 0,
    native_repeat_window: int = 0,
):
    """One iteration of self-play, through whichever backend is selected.

    Returns ``(episodes, inference_metrics, native_metrics)``.  The two metric
    slots are mutually exclusive by construction: the native path has no
    inference coordinator to report, because it has no inference coordinator.
    """
    if backend == "native":
        from diamond.alphazero.native.selfplay_pool import NativeSelfPlayPool

        if selfplay_config.max_game_seconds is not None:
            # The native runner bounds a game by moves, not by wall clock.
            # Silently ignoring a configured budget would let a run exceed a
            # limit its own config says it respects.
            raise ValueError(
                "selfplay_backend='native' does not implement max_game_seconds; "
                "set it to null or use the python backend"
            )
        pool = NativeSelfPlayPool(
            model_pool.evaluator(model_key).model,
            device=device,
            lanes=native_lanes,
            threads=worker_count,
            max_batch=inference_config.max_batch_size,
            # NOT inference_config.max_wait_ms.  That knob belongs to the Python
            # coordinator, where a batch takes milliseconds to assemble and
            # milliseconds are the right unit.  The native batcher faces a
            # ~0.9 ms GPU forward against lanes that can only supply ~50 requests
            # per cycle, so the batch never fills and the wait is spent in full,
            # every cycle.  Measured on the training checkpoint: 2000 us gives
            # 355 samples/s at 39 % evaluator occupancy, 50 us gives 1,077 at
            # 88 %.  Lane count and thread count change neither.
            max_wait_us=native_max_wait_us,
            # Selective deeper search on repeated positions. The aborted tail is
            # a short-cycle attractor, so spending only there reaches flat-128's
            # censoring at 1.6x its throughput, on 5 % of moves.
            simulations_late=native_simulations_late,
            repeat_window=native_repeat_window,
        )
        return pool.run(jobs), None, pool.metrics

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
    finally:
        metrics = coordinator.metrics
        coordinator.stop()
    return episodes, metrics, None


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
    parser.add_argument(
        "--network-width",
        type=int,
        default=None,
        help=(
            "Override network.width.  The shape is part of the checkpoint's "
            "compatibility identity, so this starts or resumes a run at that "
            "shape and will refuse a latest.pt of any other -- which is the "
            "intended behaviour, not an obstacle."
        ),
    )
    parser.add_argument(
        "--network-blocks",
        type=int,
        default=None,
        help=(
            "Override network.residual_blocks.  Seed a deeper run from a "
            "trained parent with tools/deepen_checkpoint.py rather than from "
            "scratch, then point --runtime-dir at the run holding it."
        ),
    )
    parser.add_argument("--train-steps-per-iteration", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="Parent-side torch threads.")
    parser.add_argument("--phase", default=None, help="Ledger label, e.g. B0 or A0.")
    parser.add_argument(
        "--selfplay-backend",
        default=None,
        choices=SELFPLAY_BACKENDS,
        help=(
            "Self-play engine.  'python' (default) is the multiprocess pool and the "
            "oracle; 'native' is the in-process batched backend, ~18x faster on the "
            "GPU host and gated against the oracle by tests/native."
        ),
    )
    parser.add_argument(
        "--native-lanes",
        type=int,
        default=0,
        help=(
            "Concurrent games for the native backend; 0 derives 2 x max_batch_size. "
            "Jobs beyond this are queued, so a long game costs its own lane only."
        ),
    )
    parser.add_argument(
        "--native-simulations-late",
        type=int,
        default=0,
        help="Boosted search budget for triggered moves; 0 disables the trigger.",
    )
    parser.add_argument(
        "--native-repeat-window",
        type=int,
        default=0,
        help=(
            "Boost a move when its physical position already occurred within "
            "this many plies of the same game. Keyed on occupancy, side to move "
            "and finish order -- not on turn_number, which never repeats, and "
            "not on the canonicalised encoder output."
        ),
    )
    parser.add_argument(
        "--actor-checkpoint",
        type=Path,
        default=None,
        help=(
            "Freeze self-play on this checkpoint while the learner keeps "
            "training. Normally the learner becomes the next iteration's actor "
            "immediately, which makes the loop's feedback gain very high: the "
            "actor lags by zero iterations and a 200k replay turns over in "
            "about three. Pinning the actor holds the state distribution still, "
            "so a learner that degrades anyway indicts the dataset rather than "
            "the actor-refresh loop. Diagnostic; leave unset for production."
        ),
    )
    parser.add_argument(
        "--native-max-wait-us",
        type=int,
        default=None,
        help=(
            "How long the native batcher waits for a batch to fill, in "
            "microseconds. Deliberately separate from inference.max_wait_ms, "
            "which is the Python coordinator's knob: the native batch does not "
            "fill, so this is spent in full every cycle and dominates lane and "
            "thread count. Default 500; 50 measured best on the RTX 5090."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Explicit self-play worker count; default is available CPUs minus two.",
    )
    parser.add_argument(
        "--archive-every",
        type=int,
        default=1,
        help=(
            "Write a numbered checkpoint archive every N iterations (default 1). "
            "latest.pt is still written every iteration either way, so resume "
            "safety does not depend on this."
        ),
    )
    parser.add_argument(
        "--keep-archives",
        type=int,
        default=0,
        help=(
            "Retain only the newest N archives; 0 (default) keeps every one. "
            "A checkpoint is ~8.7 MB, so an unbounded long run is measured in "
            "gigabytes -- set this on a small disk."
        ),
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
    if args.network_width is not None:
        config["network"]["width"] = args.network_width
    if args.network_blocks is not None:
        config["network"]["residual_blocks"] = args.network_blocks
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

    # Training game execution requires the native extension; the resolver
    # refuses anything else rather than quietly substituting an engine.
    selfplay_backend = args.selfplay_backend or workers.get("selfplay_backend", "native")
    if selfplay_backend not in SELFPLAY_BACKENDS:
        raise SystemExit(f"unknown selfplay_backend: {selfplay_backend}")
    selfplay_backend = resolve_selfplay_backend(
        selfplay_backend, max_game_seconds=selfplay_config.max_game_seconds
    )
    native_lanes = args.native_lanes
    native_max_wait_us = (
        args.native_max_wait_us
        if args.native_max_wait_us is not None
        else int(workers.get("native_max_wait_us", 500))
    )
    if native_max_wait_us < 1:
        raise SystemExit("native_max_wait_us must be positive")
    if selfplay_backend == "native":
        # Fail at startup rather than after the first iteration's self-play.
        from diamond.alphazero.native import require_native

        require_native()

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
            "selfplay_backend": selfplay_backend,
            "frozen_actor": str(args.actor_checkpoint) if args.actor_checkpoint else None,
            "native_lanes": native_lanes if selfplay_backend == "native" else None,
            "native_max_wait_us": (
                native_max_wait_us if selfplay_backend == "native" else None
            ),
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
        f"backend={selfplay_backend} budget={args.hours}h",
        flush=True,
    )

    stop = {"requested": False}

    def request_stop(signum, frame):
        # Finish the current iteration, then exit cleanly with state persisted.
        stop["requested"] = True
        print("\n[signal] stop requested; finishing current iteration", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    model_pool = InferenceModelPool(device=training_config.device)
    frozen_actor_key = None
    if args.actor_checkpoint is not None:
        # Activated once, outside the loop: the whole point is that it never
        # changes while the learner does.
        frozen_actor_key = model_pool.activate_checkpoint(
            args.actor_checkpoint, expected=compatibility
        )
        print(
            f"[frozen-actor] self-play pinned to {args.actor_checkpoint}; "
            "the learner will train but will not be deployed",
            flush=True,
        )
    inference_config = InferenceConfig(**config["inference"])
    deadline = time.perf_counter() + args.hours * 3600.0
    session_start = time.perf_counter()

    while time.perf_counter() < deadline and not stop["requested"]:
        iteration = state.data["iteration"]
        iteration_start = time.perf_counter()

        # Self-play always runs against the current durable checkpoint, so the
        # data provenance and the trained weights can never disagree -- unless
        # the actor is deliberately frozen for the diagnostic above, in which
        # case the provenance correctly records the *actor* that produced the
        # games.
        model_key = (
            frozen_actor_key
            if frozen_actor_key is not None
            else model_pool.activate_checkpoint(latest, expected=compatibility)
        )
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

        try:
            episodes, inference_metrics, native_metrics = run_self_play(
                backend=selfplay_backend,
                jobs=jobs,
                model_pool=model_pool,
                model_key=model_key,
                inference_config=inference_config,
                selfplay_config=selfplay_config,
                worker_count=worker_count,
                device=training_config.device,
                native_lanes=native_lanes,
                native_max_wait_us=native_max_wait_us,
                native_simulations_late=args.native_simulations_late,
                native_repeat_window=args.native_repeat_window,
            )
        except RuntimeError as error:
            # A stop signal terminates the worker children mid-episode.  That is
            # an intended shutdown, not a defect: the previous iteration is
            # already durable, so exit cleanly instead of reporting a crash.
            if stop["requested"]:
                print(f"[stop] self-play interrupted during shutdown: {error}", flush=True)
                break
            raise
        selfplay_s = time.perf_counter() - iteration_start
        if native_metrics:
            print(
                f"[native] evals={native_metrics['evaluations']:,} "
                f"batches={native_metrics['batches']:,} "
                f"mean_batch={_mean_batch(native_metrics):.1f} "
                f"boosted={native_metrics.get('boosted_moves', 0):,}/"
                f"{native_metrics.get('moves', 0):,} "
                f"evaluator={native_metrics['evaluator_seconds'] / max(1e-9, native_metrics['wall_seconds']) * 100:.0f}%",
                flush=True,
            )

        completed = aborted = new_samples = 0
        move_counts: list[int] = []
        iteration_aborts: dict[str, int] = {}
        # One manifest write for the whole iteration, not one per episode.
        # ingest_episodes is idempotent on game_id exactly as ingest_episode is,
        # so a resumed iteration still cannot double-count what it persisted.
        replay.ingest_episodes(episodes)
        for episode in episodes:
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

        # Chunks older than the capacity window are read and immediately evicted
        # by every load_buffer, so they cost disk and time and can never be
        # sampled.  Unpruned, this run grows ~79 MB an iteration.
        pruned = replay.prune_to_capacity()

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
                    math.isfinite(value)
                    for value in (
                        metrics.total_loss,
                        metrics.policy_loss,
                        metrics.value_loss,
                    )
                ):
                    raise SystemExit(f"non-finite loss detected: {asdict(metrics)}")
                metrics_list.append(asdict(metrics))
        train_s = time.perf_counter() - train_start

        # ---- checkpoint every iteration; the run is never all-or-nothing ----
        # latest.pt is unconditional: it is what a resume reads, and making it
        # conditional on the archive cadence would trade crash safety for disk.
        if metrics_list:
            save_checkpoint(
                latest, trainer, operation_id=f"{args.run_id}-i{iteration:06d}"
            )
            if args.archive_every > 0 and iteration % args.archive_every == 0:
                archive = checkpoints_dir / (
                    f"{phase}-i{iteration:06d}-step{trainer.training_step:09d}.pt"
                )
                save_checkpoint(
                    archive, trainer, operation_id=f"{args.run_id}-i{iteration:06d}"
                )
                if args.keep_archives > 0:
                    stale = sorted(checkpoints_dir.glob(f"{phase}-i*.pt"))[
                        : -args.keep_archives
                    ]
                    for path in stale:
                        path.unlink()
            else:
                archive = None
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
            "replay_chunks_pruned": pruned,
            "throughput": throughput_summary(
                attempted=len(episodes),
                completed=completed,
                samples=new_samples,
                train_steps=len(metrics_list),
                elapsed_s=time.perf_counter() - iteration_start,
            ),
            # Exactly one of these is populated.  The native path has no
            # inference coordinator to summarise, so recording an empty
            # coordinator summary would misreport it as one that did nothing.
            "inference": (
                summarize_metrics(inference_metrics, elapsed_s=selfplay_s)
                if inference_metrics is not None
                else None
            ),
            "native_selfplay": (
                {
                    "evaluations": native_metrics["evaluations"],
                    "batches": native_metrics["batches"],
                    "moves": native_metrics["moves"],
                    "mean_batch": _mean_batch(native_metrics),
                    "wall_seconds": native_metrics["wall_seconds"],
                    "evaluator_seconds": native_metrics["evaluator_seconds"],
                    "worker_busy_seconds": native_metrics["worker_busy_seconds"],
                }
                if native_metrics
                else None
            ),
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
