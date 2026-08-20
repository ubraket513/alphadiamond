# RTX 3060 GPU Training Runbook

Continues the existing Soo run `cpu8h-soo-20260819` on a rented vast.ai RTX 3060,
in place, migrating its checkpoint from CPU to CUDA.

> **Status:** code complete and verified on CPU. Everything below the
> provisioning section still needs to be run on the GPU box; record the measured
> values as you go.

> **Measured results so far:** see
> [gpu_benchmark_findings.md](gpu_benchmark_findings.md) for throughput numbers,
> the latency-bound bottleneck analysis, and the open questions about
> `max_moves` and worker oversubscription.

## What continues, and what that costs

The run keeps its identity: `training_step`, `loop_state.json`, `ledger.jsonl`
and the 26,027-sample replay buffer all carry forward. A fork would have
restarted self-play from an empty replay.

The cost is that once migrated, the run is no longer a clean CPU baseline, so a
symmetric equal-wall-clock CPU-vs-GPU learning comparison is not available
against it. Benchmark from throwaway copies **before** migrating, and use the
existing ledger (18 iterations, 16 games/iter, ~400 s self-play each, 0 aborts)
as the historical CPU reference.

## Source checkpoint

| Field | Value |
|---|---|
| Path | `runtime/runs/soo/cpu8h-soo-20260819/latest.pt` |
| SHA-256 | `4b2a32ff15179e890d4266346bca178d9a255eebe16af3a6e3d0482f0ceb1320` |
| `training_step` | 72 |
| Recorded device | `cpu` |
| Model | Soo 2.0.0, width 128, 6 residual blocks |
| Replay | 26,027 samples, 18 iterations, 0 aborts |

## 1. Provision

Pick an RTX 3060 (12 GB) offer with >= 32 vCPUs and a CUDA 12.x image. Record:

```bash
nvidia-smi
nproc
```

## 2. Install a CUDA torch build

The development machine ships a **CPU-only** torch (`torch.version.cuda is None`),
so this step is mandatory, not a formality.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match nvidia-smi
pip install pytest trueskill
pip install --no-deps -e .

python -c "import torch; print(torch.__version__, torch.version.cuda, \
torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect a non-`None` CUDA version, `True`, and `NVIDIA GeForce RTX 3060`.
**If `cuda_available` is `False`, stop here** — every later step is meaningless.

## 3. Confirm the CPU allocation policy on the real box

```bash
python -c "from diamond.alphazero.hardware import available_cpu_count, resolve_worker_count; \
print(available_cpu_count(), resolve_worker_count())"
```

Expect `32 30`. If the container is cpuset-restricted the first number will be
lower and the second follows it; record what it actually prints rather than
assuming 30.

## 4. Get the run onto the box

`/runtime/` is intentionally **not** git-ignored so checkpoints travel with the
repo. Commit any untracked run artifacts first, then on the VM:

```bash
git clone <repo> alphadiamond && cd alphadiamond
sha256sum runtime/runs/soo/cpu8h-soo-20260819/latest.pt
```

The digest must match the table above. A mismatch means an incomplete clone or
binary mangling — fix it before training on it.

## 5. Prove CUDA correctness before believing any GPU result

```bash
python -m pytest tests/alphazero tests/tools -q
```

Expect the same 446 passing, with the seven CUDA-gated tests now **running
rather than skipping**. If CPU/CUDA parity fails beyond FP32 tolerances, or the
legal-action sets differ at all, stop — that is a correctness failure, not a
tuning problem.

## 6. Benchmark from copies, before migrating

While the run is still CPU-tagged, seed throwaway runs from copies so the real
run is only ever touched by real training.

```bash
for name in gpu-32 gpu-64; do
  mkdir -p /tmp/az-bench/soo/$name
  cp runtime/runs/soo/cpu8h-soo-20260819/latest.pt /tmp/az-bench/soo/$name/latest.pt
done

python tools/az_train.py --config runtime/configs/soo-rtx3060.json \
  --runtime-dir /tmp/az-bench --run-id gpu-32 --migrate-device \
  --simulations 32 --train-steps-per-iteration 8 --hours 0.5

python tools/az_train.py --config runtime/configs/soo-rtx3060.json \
  --runtime-dir /tmp/az-bench --run-id gpu-64 --migrate-device \
  --simulations 64 --train-steps-per-iteration 8 --hours 0.5
```

Watch utilization externally (no NVML dependency is added):

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
```

From each ledger record `throughput.completed_games_per_hour`,
`throughput.samples_per_hour`, `median_moves`, `p90_moves`, `abort_reasons`, and
the `inference` block.

**Expect a mean batch size well below 32.** With ~30 workers doing synchronous
leaf evaluation, the achievable batch is bounded by concurrent workers, not by
`max_batch_size`. A CPU smoke run with 4 workers measured a mean batch of 3.3.
That is the architecture working as designed, not a bug.

## 7. Back up, then migrate the real run

```bash
cp runtime/runs/soo/cpu8h-soo-20260819/latest.pt ~/soo-step72-cpu.pt   # off-box copy

python tools/az_train.py \
  --config runtime/configs/soo-rtx3060.json \
  --runtime-dir runtime/runs \
  --run-id cpu8h-soo-20260819 \
  --migrate-device \
  --train-steps-per-iteration 8 \
  --hours 0.05
```

Then confirm: `latest.pt.cpu-backup` exists with the digest from the table,
`latest.pt` now records `cuda:0` at `training_step` 72, `loop_state.json` still
reads iteration 18 with 26,027 samples, and the replay loads at its prior size.

`--migrate-device` is idempotent — it is a no-op once the checkpoint records the
target device — so it is safe to leave in a launch script.

## 8. Train

```bash
python tools/az_train.py \
  --config runtime/configs/soo-rtx3060.json \
  --runtime-dir runtime/runs --run-id cpu8h-soo-20260819 \
  --simulations 64 --train-steps-per-iteration 8 --hours 4
```

No `--migrate-device` needed after the first time.

### Switching to heuristic-off A0 is a separate, deliberate step

The run is mid-B0 with `canonical-target-vacancy-distance-v2`, and the GPU config
inherits that unchanged so resumed iterations keep generating data under the same
prior the replay was built from. Switching to `none` changes what the data
*means*, and mixing A0 and B0 episodes in one replay buffer is exactly the kind of
silent semantic change to avoid. Make it one explicit, ledger-recorded move once
the GPU path is proven:

```bash
python tools/az_train.py \
  --config runtime/configs/soo-rtx3060.json \
  --runtime-dir runtime/runs --run-id cpu8h-soo-20260819 \
  --simulations 64 --bootstrap-prior none --phase A0 \
  --train-steps-per-iteration 8 --hours 4
```

## 9. Track learning quality against the step-72 baseline

```bash
python tools/cpu_off_probe.py \
  --config runtime/configs/soo-rtx3060.json \
  --checkpoint runtime/runs/soo/cpu8h-soo-20260819/latest.pt \
  --episodes 30 --simulations 32,64 --base-seed 9000 \
  --out runtime/runs/soo/cpu8h-soo-20260819/off-probe-gpu.json
```

Write to a **new** `--out` path: the existing `off-probe.json` holds the step-72
CPU baseline this is compared against. Compare completion rate, median moves,
p90 moves and abort rate. Loss is secondary — the arms see different data
volumes, so their losses are not comparable in isolation.

To probe from a CPU-only machine, pass `--config runtime/configs/soo-cpu8h.json`
instead; the probe allows device migration for its read-only load, so it scores
checkpoints from either kind of host.

## Recovery

Training continues in place, so a bad GPU run cannot be discarded by deleting a
fork directory:

1. Stop training.
2. Restore `latest.pt` from `latest.pt.cpu-backup`, the off-box copy, or Git.
3. Roll `loop_state.json` back to the matching iteration.

Take the off-box copy **before** the first GPU iteration writes anything.

## Rollback criteria

Stop and revert if any of these hold:

- CPU/CUDA parity fails beyond FP32 tolerances, or legal-action sets differ.
- The CPU workflow regresses on `soo-cpu8h.json`, or any existing test fails.
- A game timeout crashes an iteration, or a completed sibling episode is lost.
- Aborted games contribute a non-zero sample count.
- `training_step`, `iteration` or the replay sample count changes across the
  migration, or the backup does not match the pre-migration digest.
- GPU useful throughput fails to beat the CPU baseline — keep the timeout,
  migration and metrics work, and revert only the GPU configuration.
