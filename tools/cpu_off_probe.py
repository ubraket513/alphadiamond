"""Heuristic-OFF gate probe for a trained CPU checkpoint (blueprint sections 10-11).

Loads an actual trained checkpoint without reinitializing it, runs fixed-seed
self-play with ``bootstrap_prior = none``, and reports raw viability metrics.
The gate is operational -- can this network generate real terminal games on its
own -- and is explicitly not a claim about playing strength.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diamond.alphazero.bootstrap.probe import run_probe
from diamond.alphazero.checkpoint import load_checkpoint
from diamond.alphazero.config import (
    BOOTSTRAP_PRIOR_NONE,
    NetworkConfig,
    TrainingConfig,
)
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.identity import (
    MIN_MODEL_NAME,
    SOO_MODEL_NAME,
    CheckpointCompatibilitySpec,
)
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer

# Blueprint section 10: at least 8 of 10 games complete, with non-empty replay.
PASS_COMPLETION = 0.8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--simulations",
        default=None,
        help="Comma-separated budgets to probe, e.g. '32,64'. Defaults to the config value.",
    )
    parser.add_argument("--max-moves", type=int, default=2000)
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prior", default=BOOTSTRAP_PRIOR_NONE)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Probe with no Dirichlet noise and no temperature sampling: can the "
            "network finish a game playing its best move every time? That is a "
            "materially easier question than production asks. The default is the "
            "configuration's own exploration, in full -- on the Soo from-scratch "
            "run a probe missing only the temperature sampling read 100%% where "
            "production self-play completed 64%%."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    network = NetworkConfig(**config["network"])
    model_name = config["model_name"]
    version = config["model_version"]

    if model_name == SOO_MODEL_NAME:
        compatibility = CheckpointCompatibilitySpec.soo(
            model_version=version, network_config=network
        )
        model = SooModel(network, model_version=version)
        value_size = 1
    elif model_name == MIN_MODEL_NAME:
        compatibility = CheckpointCompatibilitySpec.min(
            model_version=version, network_config=network
        )
        model = MinModel(network, model_version=version)
        value_size = 3
    else:
        raise SystemExit(f"unsupported model: {model_name}")

    trainer = AlphaZeroTrainer(
        model, compatibility, TrainingConfig(**config["training"])
    )
    # Load the real trained weights; never reinitialize for a probe.
    # Device migration is allowed here on purpose: a probe only reads the model,
    # so scoring a CPU-tagged archived checkpoint on the GPU box -- or a
    # GPU-trained checkpoint on a CPU box -- is a legitimate, read-only use.
    load_checkpoint(
        args.checkpoint, trainer, expected=compatibility, allow_device_migration=True
    )
    evaluator = TorchEvaluator(
        trainer.model, value_size=value_size, device=config["training"]["device"]
    )

    budgets = (
        [int(s) for s in args.simulations.split(",") if s.strip()]
        if args.simulations
        else [config["mcts"]["simulations"]]
    )

    # The gate exists to answer "is this network ready for production
    # self-play", so unless asked otherwise it probes under production's own
    # exploration rather than under greedy play.
    if args.deterministic:
        epsilon = alpha = temperature = 0.0
        temperature_moves = 0
    else:  # production's own exploration, in full
        epsilon = config["mcts"]["dirichlet_epsilon"]
        alpha = config["mcts"]["dirichlet_alpha"]
        temperature = config["self_play"]["temperature"]
        temperature_moves = config["self_play"]["temperature_moves"]

    print(
        f"[probe] {model_name} checkpoint={args.checkpoint} "
        f"training_step={trainer.training_step} prior={args.prior} "
        f"exploration={'off (greedy)' if args.deterministic else f'eps={epsilon} T={temperature}/{temperature_moves}'}",
        flush=True,
    )
    rows = []
    for simulations in budgets:
        start = time.perf_counter()
        report = run_probe(
            model_name=model_name,
            bootstrap_prior=args.prior,
            episodes=args.episodes,
            simulations=simulations,
            max_moves=args.max_moves,
            base_seed=args.base_seed,
            base_evaluator=evaluator,
            dirichlet_epsilon=epsilon,
            dirichlet_alpha=alpha,
            temperature=temperature,
            temperature_moves=temperature_moves,
        )
        elapsed = time.perf_counter() - start
        verdict = (
            "PASS"
            if report.completion_rate >= PASS_COMPLETION and report.samples > 0
            else "FAIL"
        )
        row = {
            "model": model_name,
            "checkpoint": str(args.checkpoint),
            "training_step": trainer.training_step,
            "bootstrap_prior": args.prior,
            "simulations": simulations,
            "episodes": report.episodes,
            "completed": report.completed,
            "completion_rate": round(report.completion_rate, 4),
            "median_moves": report.median_moves,
            "p90_moves": report.p90_moves,
            "samples": report.samples,
            "samples_per_episode": round(report.samples_per_episode, 2),
            "abort_reasons": report.abort_reasons,
            "sec_per_game": round(elapsed / report.episodes, 2),
            "verdict": verdict,
        }
        rows.append(row)
        print(
            f"  sims={simulations:<4d} {report.completed}/{report.episodes} complete "
            f"({report.completion_rate:.0%}) median={report.median_moves} "
            f"p90={report.p90_moves} samples/ep={report.samples_per_episode:.1f} "
            f"sec/game={row['sec_per_game']:.1f} aborts={report.abort_reasons or '{}'} "
            f"=> {verdict}",
            flush=True,
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[probe] wrote {args.out}", flush=True)

    return 0 if any(row["verdict"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
