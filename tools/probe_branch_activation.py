"""Does a transplanted block earn its place, measured in minutes rather than hours?

A depth transplant starts as an exact identity, so the question is never whether
the child *is* the parent -- it is -- but whether training then makes the new
blocks do anything.  Answering that with a full self-play arm cost hours and
mostly measured a warm-up that had not happened: the first arm's gates sat at
~0.002 from 900 steps to 2,100 while the inherited blocks held 0.8-1.6.

This runs a few hundred steps on **frozen replay** -- no self-play at all -- so
the state distribution and the targets are fixed and the only thing varying is
the architecture.  Two consequences worth stating:

*Every variant sees identical batches in identical order.*  Batches come from a
fixed index sequence built here rather than from the store's own RNG, because
the gate's gradient depends on the data and comparing initialisations on
different samples compares the data.

*Held-out loss is meaningful here, unlike in self-play.*  Pitfall 7.14 rules out
loss as a health metric for a self-play loop because the loop censors its own
dataset while the number improves.  Nothing is censored here: the targets are a
frozen set of 128-simulation searches, so "does the deeper network fit the
teacher better" is exactly the question held-out policy CE and value MSE answer.

Usage::

    python tools/probe_branch_activation.py --parent 44250.pt \\
        --replay /path/to/run/replay --steps 300 \\
        --variant shallow=@parent \\
        --variant appended=--residual-gate \\
        --variant interleaved=--residual-gate,--interleave
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.checkpoint import checkpoint_network_config, load_checkpoint
from diamond.alphazero.config import TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import SooModel
from diamond.alphazero.orchestration.replay_store import PersistentReplayStore
from diamond.alphazero.replay import ReplayBuffer, TrainingSample
from diamond.alphazero.trainer import AlphaZeroTrainer

ACTION_SIZE = 5329
PARENT_SENTINEL = "@parent"


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().pow(2).mean().sqrt())


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _build(path: Path, device: str) -> AlphaZeroTrainer:
    """Reconstruct a trainer, inferring the gate layout from the checkpoint.

    Gating is not part of the compatibility metadata, so a gated checkpoint
    loaded into an ungated model fails a strict load rather than passing
    quietly.  Reading the `alpha` keys keeps the tool working on both without a
    flag the caller has to remember to match.
    """
    network = checkpoint_network_config(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    version = payload["metadata"]["model_version"]
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version=version, network_config=network
    )
    training = TrainingConfig(**{**payload["training_config"], "device": device})
    gated = sorted(
        int(key.split(".")[2])
        for key in payload["model_state_dict"]
        if key.startswith("trunk.blocks.") and key.endswith(".alpha")
    )
    trainer = AlphaZeroTrainer(
        SooModel(network, model_version=version, gated_blocks=gated or None),
        compatibility,
        training,
    )
    load_checkpoint(path, trainer, expected=compatibility, allow_device_migration=True)
    return trainer


def _restamp(samples, compatibility) -> list[TrainingSample]:
    """Re-label samples for a different network shape; see the note in main()."""
    if not samples or samples[0].compatibility == compatibility:
        return list(samples)
    return [replace(sample, compatibility=compatibility) for sample in samples]


def _tensors(batch, device: torch.device):
    return (
        torch.tensor(batch.node_features, dtype=torch.float32, device=device),
        torch.tensor(batch.policy_targets, dtype=torch.float32, device=device),
        torch.tensor(batch.value_targets, dtype=torch.float32, device=device),
    )


def _held_out(trainer: AlphaZeroTrainer, batches) -> dict[str, float]:
    """Policy cross-entropy, KL against the search targets, and value MSE."""
    trainer.model.eval()
    totals = {"policy_ce": 0.0, "policy_kl": 0.0, "value_mse": 0.0}
    rows = 0
    with torch.no_grad():
        for batch in batches:
            features, policy, value = _tensors(batch, trainer.device)
            logits, predicted = trainer.model(features)
            log_probability = torch.log_softmax(logits, dim=-1)
            cross_entropy = -(policy * log_probability).sum(-1)
            # KL is cross-entropy less the target's own entropy, so it reads as
            # "how far from the teacher" rather than "how peaked was the teacher".
            entropy = -(policy * torch.log(policy.clamp_min(1e-12))).sum(-1)
            totals["policy_ce"] += float(cross_entropy.sum())
            totals["policy_kl"] += float((cross_entropy - entropy).sum())
            totals["value_mse"] += float((predicted - value).pow(2).sum())
            rows += features.shape[0]
    trainer.model.train()
    return {key: value / rows for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True, help="a run's replay dir")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument("--held-out", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="Steps at which to evaluate held-out fit; default 0, half, full.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=FLAG,FLAG,...",
        help=(
            "Repeat per variant; flags go to deepen_checkpoint.py.  "
            f"Use {PARENT_SENTINEL} for the untouched parent as a matched "
            "control -- give it first so later variants report a delta."
        ),
    )
    args = parser.parse_args()

    parent_network = checkpoint_network_config(args.parent)
    old_blocks = parent_network.residual_blocks
    marks = sorted({0, args.steps // 2, args.steps} if not args.checkpoints
                   else set(args.checkpoints))

    specs = []
    for raw in args.variant:
        if "=" not in raw:
            raise SystemExit(f"--variant must be NAME=FLAGS, got {raw!r}")
        name, _, flags = raw.partition("=")
        specs.append((name, [f for f in flags.split(",") if f]))

    # One fixed split and one fixed batch sequence, shared by every variant.
    # Built from indices rather than the store's RNG so the sequence cannot
    # drift when a variant happens to consume randomness differently.
    #
    # The replay is read with the *parent's* compatibility and re-stamped per
    # variant.  A store's namespace is the SHA-256 of the whole compatibility
    # spec, `network_config` included, so a 128x12 model cannot open a 128x6
    # store and `collate` refuses the samples on top of that.  The stored
    # quantities -- node features, the search's visit distribution, the terminal
    # value -- do not depend on width or depth at all, which is the point of
    # holding the teacher fixed while the architecture varies.  Re-stamping is
    # sound *here* precisely because every other compatibility field is
    # identical; it would not be if the encoder or action space differed.
    reference = _build(args.parent, args.device)
    replay = PersistentReplayStore(
        args.replay, reference.compatibility, capacity=args.capacity, seed=args.seed
    )
    source_buffer = replay.load_buffer()
    population = source_buffer.samples
    if len(population) < args.held_out + args.batch_size:
        raise SystemExit(f"replay has only {len(population)} samples")
    rng = random.Random(args.seed)
    order = list(range(len(population)))
    rng.shuffle(order)
    held_out_index, train_index = order[: args.held_out], order[args.held_out :]
    batch_rng = random.Random(args.seed + 1)
    schedule = [
        [train_index[batch_rng.randrange(len(train_index))] for _ in range(args.batch_size)]
        for _ in range(args.steps)
    ]

    print(
        f"parent {parent_network.width}x{old_blocks} -> {args.blocks}, "
        f"{args.steps} steps at batch {args.batch_size}, "
        f"{len(held_out_index)} held-out samples, identical batches per variant\n"
    )

    scratch = Path(tempfile.mkdtemp(prefix="branch-probe-"))
    control: dict[int, dict[str, float]] | None = None
    for name, flags in specs:
        if flags == [PARENT_SENTINEL]:
            target = args.parent
        else:
            target = scratch / f"{name}.pt"
            build = [
                sys.executable, str(ROOT / "tools/deepen_checkpoint.py"),
                "--source", str(args.parent), "--out", str(target),
                "--blocks", str(args.blocks), "--device", args.device,
                "--corpus", "-", *flags,
            ]
            result = subprocess.run(build, capture_output=True, text=True, check=False)
            if result.returncode:
                raise SystemExit(f"{name}: transplant failed\n{result.stdout}{result.stderr}")

        trainer = _build(target, args.device)
        stamped = _restamp(population, trainer.compatibility)
        batcher = ReplayBuffer(
            trainer.compatibility, capacity=len(stamped), seed=args.seed
        )
        held_out_batches = [
            batcher.collate(
                [stamped[i] for i in held_out_index[start : start + args.batch_size]],
                action_size=ACTION_SIZE,
            )
            for start in range(0, len(held_out_index), args.batch_size)
        ]
        blocks = trainer.model.trunk.blocks
        gate_indices = sorted(trainer.model.trunk.gated_blocks)
        gated = bool(gate_indices)
        new_blocks = gate_indices or [i for i in range(old_blocks, len(blocks))]
        gate_of = (
            (lambda index, seq=blocks: seq[index].alpha)
            if gated
            else (lambda index, seq=blocks: seq[index].norm.weight)
        )

        print(f"  {name}" + (f"   gated blocks {gate_indices}" if gated else ""))
        start = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
        gradients: list[float] = []
        trail: list[tuple[int, float]] = []
        measured: dict[int, dict[str, float]] = {}

        for step in range(args.steps + 1):
            if step in marks:
                measured[step] = _held_out(trainer, held_out_batches)
                if new_blocks and gated:
                    trail.append(
                        (step, float(gate_of(new_blocks[0]).detach().reshape(-1)[0]))
                    )
            if step == args.steps:
                break
            trainer.train_batch(
                batcher.collate(
                    [stamped[i] for i in schedule[step]], action_size=ACTION_SIZE
                )
            )
            for index in new_blocks:
                grad = gate_of(index).grad
                if grad is not None:
                    gradients.append(_rms(grad))

        share: dict[int, float] = {}
        handles = []

        def watch(index: int, sink=share):
            def hook(module, inputs, output):
                incoming = _rms(inputs[0])
                sink[index] = _rms(output - inputs[0]) / incoming if incoming else 0.0
            return hook

        for index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(watch(index)))
        with torch.no_grad():
            trainer.model(_tensors(held_out_batches[0], trainer.device)[0])
        for handle in handles:
            handle.remove()

        state = trainer.model.state_dict()
        if new_blocks and len(blocks) > old_blocks:
            inherited = [i for i in range(len(blocks)) if i not in new_blocks]
            if gated:
                values = [
                    float(state[f"trunk.blocks.{i}.alpha"].reshape(-1)[0]) for i in new_blocks
                ]
                print(f"    gate (alpha)      {', '.join(f'{v:+.4f}' for v in values)}")
                print(f"    gate mean         {_mean(values):+.5f}   "
                      f"(from 0.0; sign is the signal)")
                print(f"    gate trajectory   "
                      f"{' -> '.join(f'{v:+.4f}@{s}' for s, v in trail)}")
            else:
                print(f"    gate (gamma RMS)  "
                      f"{_mean(_rms(state[f'trunk.blocks.{i}.norm.weight']) for i in new_blocks):.5f}")
            print(f"    gate grad RMS     {_mean(gradients):.3e}")
            drift = []
            for index in new_blocks:
                prefix = f"trunk.blocks.{index}."
                keys = [k for k in state if k.startswith(prefix) and not k.endswith(".alpha")]
                moved = sum(float((state[k] - start[k]).pow(2).sum()) for k in keys) ** 0.5
                initial = sum(float(start[k].pow(2).sum()) for k in keys) ** 0.5
                drift.append(moved / initial if initial else float("nan"))
            new_share = _mean(share[i] for i in new_blocks)
            old_share = _mean(share[i] for i in inherited)
            print(f"    residual/input    {new_share:.5f}   "
                  f"(inherited {old_share:.5f}, ratio {new_share / old_share:.4f})")
            print(f"    branch drift      {_mean(drift) * 100:.1f}% of init norm")

        for step in marks:
            row = measured[step]
            delta = ""
            if control is not None:
                delta = (
                    f"   vs control CE {row['policy_ce'] - control[step]['policy_ce']:+.5f}"
                    f"  MSE {row['value_mse'] - control[step]['value_mse']:+.5f}"
                )
            print(
                f"    step {step:>4}  held-out CE {row['policy_ce']:.5f}  "
                f"KL {row['policy_kl']:.5f}  value MSE {row['value_mse']:.5f}{delta}"
            )
        if control is None:
            control = measured
        print()

    print(f"(scratch checkpoints in {scratch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
