# RTX 5060 Ti Worker-Scaling Findings

New-host characterization for the Soo AlphaZero pipeline, run on a vast.ai
RTX 5060 Ti + Xeon E5-2686 v4 host on 2026-08-21. Follows
[gpu_profile_findings.md](gpu_profile_findings.md), which measured and fixed the
coordinator, evaluator and lane-scheduling defects on an RTX 3060.

**Status:** four measured worker-scaling points, one run each, all from one
immutable checkpoint at commit `00e1547`. The bottleneck identification is
conclusive; the choice between 48 and 64 workers rests on a single run per point.
`max_wait_ms` was **not** swept on this host — see *Open questions*.

---

## Headline

**The GPU is irrelevant on this host, and adding workers has stopped helping.**

The parent process is pegged at ~100 % of one core at *every* point measured,
including the 30-worker baseline, while the RTX 5060 Ti sits at **4.8–5.0 %
utilisation** drawing 24 W of a 180 W cap. Throughput is bounded by roughly
**1.5 ms of single-threaded Python per evaluation** in the parent.

The sharpest evidence that the host and not the device sets the pace: at the
identical setting, the previous **128-vCPU RTX 3060** box sustained **900
evals/s** where this box sustains **510**. A faster GPU behind a slower single
core is a net loss.

---

## Environment

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 5060 Ti, 16 GB** (15.48 GiB), driver 595.84, CUDA 13.2 |
| Compute capability | 12.0 (Blackwell), 36 SMs, BF16 supported |
| CPU | **Intel Xeon E5-2686 v4 @ 2.30 GHz** (Broadwell), max 3.0 GHz |
| Topology | **1 socket × 18 physical cores × 2 threads = 36 SMT threads**, 1 NUMA node |
| Affinity-visible CPUs | 36 (unrestricted) |
| Cache | L3 45 MiB shared, L2 4.5 MiB (18×) |
| RAM | 31 GiB total, 28 GiB available at rest, 23 GiB swap |
| torch | 2.13.0+cu130, CUDA runtime 13.0 |
| Python | 3.12.13, `/venv/main` |
| Precision | FP32 throughout |
| `ulimit` | 524,288 open files, unlimited user processes |

Two corrections to the provisioning brief, both material:

- The card is an RTX 5060 **Ti** with 16 GB, not an RTX 5060.
- **"36 vCPUs" is 18 physical cores plus SMT**, not 36 cores. `lscpu` reports
  `Thread(s) per core: 2`. So 64 workers is **3.6× physical-core
  oversubscription** and 96 would be 5.3× — a far more aggressive regime than
  the same integers imply under a "36 cores" reading.

No `OMP_*`, `MKL_*` or `OPENBLAS_*` variables are set. `torch.get_num_threads()`
defaults to 18, but `az_train.py` sets it to 4, and the self-play worker import
path is asserted Torch-free by a passing test — so workers cannot inherit a
multi-threaded BLAS pool. Recorded and left alone.

---

## Source and checkpoint

| Field | Value |
|---|---|
| Commit | `00e15477c808957cd16fb59855d3e2d3bc6c0174` (`main`, clean) |
| Source under test | read-only `git archive` snapshot, pinned via `PYTHONPATH` |
| `diamond.__file__` | `…/scratchpad/src-00e1547/src/diamond/__init__.py` (asserted per run) |
| Checkpoint | `runtime/runs/soo/cpu8h-soo-20260819/latest.pt` |
| SHA-256 | `1634b901e213b065c107eea734b8c172c14babb1c2565352203961e86ea165af` |
| `training_step` | 80 |
| Recorded device | `cuda:0` (so `--migrate-device` is a no-op) |

`diamond` is **not installed** on this host — there is no editable install, so
the `.pth` contamination that invalidated a previous sweep cannot occur here.
`PYTHONPATH` is authoritative. The assertion is retained anyway, because "not
installed today" is not a guarantee.

Every run logged `[resume] loaded … at training_step=80`; the driver aborts the
point on `[init] wrote initial checkpoint`.

**Correctness:** `pytest tests/alphazero tests/agents tests/tools` →
**483 passed**, including the 9 CUDA-gated tests, which *ran* rather than
skipped. CPU/CUDA FP32 evaluation parity therefore holds on Blackwell `sm_120`,
and the cu130 wheel has kernels for this architecture.

**Verified on `main` before benchmarking**, since a prompt is not evidence:
batch window anchored to batch open (`_PendingBatch.opened_at`), 64 simulations
in the shipped config, vectorized `TorchEvaluator`, shared self-play job queue,
`max_wait_ms = 2`, and adaptive wait **not** merged.

---

## Method

One complete 32/36/48/64-game self-play iteration per point, 64 simulations,
`max_wait_ms` 2, `max_batch_size` 32, `max_moves` 2000, `max_game_seconds` 900,
FP32, `run_seed` 7, same bootstrap prior, same checkpoint. Only `worker_count`
and `games_per_iteration` vary, plus `train_steps_per_iteration` held at the 4:1
games-per-optimizer-update ratio.

`games_per_iteration` **must** move with the worker count:
`lane_count = min(worker_count, len(jobs))`, so `--workers 64` against 32 games
would silently measure 32 lanes. It is also config-only — `az_train.py` has no
CLI override — so each point gets its own frozen config under `az-bench/configs/`.
The canonical `runtime/configs/soo-rtx3060.json` was never edited; the 30-worker
config is byte-identical to it.

Self-play is seed-deterministic and runs *before* the optimizer steps within an
iteration, so every row played its games against the same step-80 weights. The
30-worker row reproduced the RTX 3060's 64-sim distribution exactly — 3,064
samples, median 91 moves — confirming determinism across hosts. **Differences
between rows are pure scheduling, never different work.**

`samples/hour` is computed over the self-play window only, excluding the 1.3–2.4 s
of optimizer time, so training cannot contaminate the throughput metric.

---

## Worker scaling

```
run         wrk  gms    wall   samp/h  eval/s  meanB  p90B  maxB   qd50    qd90  rsp50  GPU%  par%   tot%    done
A-w30-g32    30   32   383.8    28743     510   7.85    16    19  13.64   16.41  28.70   5.0    98    335 32/32
B-w36-g36    36   36   367.5    33844     601  10.51    18    23  13.52   16.21  32.56   5.0   101    397 36/36
C-w48-g48    48   48   440.3    36059     640  11.63    18    23  31.76   35.98  50.53   4.8   102    438 48/48
D-w64-g64    64   64   546.0    37510     666  12.83    18    23  38.78   57.74  59.51   4.8   104    462 64/64
```

Latencies in ms; `par%`/`tot%` are percent of one core (3600 % available).

```
run         wrk   samp/h  samp/h/wrk  eval/s/wrk  speedup  par.eff  p90mv aborts
A-w30-g32    30    28743         958        17.0     1.00     1.00    114 {}
B-w36-g36    36    33844         940        16.7     1.18     0.98    126 {}
C-w48-g48    48    36059         751        13.3     1.25     0.78    114 {}
D-w64-g64    64    37510         586        10.4     1.31     0.61    108 {}
```

**Every point completed 100 % of its games with zero aborts.** No configuration
won by discarding data. Median moves 87–91, p90 108–126 — the game distribution
is stable across the sweep, so `max_moves` 2000 remains inert at 64 simulations.

### Points E (80) and F (96) were deliberately not run

Two independent stop conditions, both from the plan's own rules:

1. **The scaling rule.** 64 beat 48 by **4.0 %**, far below the "still >10–15 %
   better" threshold required to continue.
2. **Memory.** Available RAM fell 20.0 → 18.1 → 14.8 → 10.3 GB, a steady
   **~283 MB per worker**. Projecting: 80 workers ≈ 5.8 GB free, 96 workers ≈
   1.2 GB free, against 23 GB of swap standing by to silently destroy any
   measurement taken there.

---

## Scaling-curve interpretation

```
30 → 36:  +17.8 % throughput   qd50 flat (13.6 → 13.5 ms)    eff 0.98   real scaling
36 → 48:   +6.5 % throughput   qd50 +135 % (13.5 → 31.8 ms)  eff 0.78   bending
48 → 64:   +4.0 % throughput   qd90 +60 %  (36.0 → 57.7 ms)  eff 0.61   flat
```

**The knee is 36 workers.** Only the 30→36 step scales at near-unit efficiency
with flat queueing latency. Everything past it buys throughput with latency:
response p50 doubles (28.7 → 59.5 ms) across the sweep to purchase +31 % total
throughput, and parallel efficiency falls to 0.61.

### Why scaling stopped

The decisive series is **batches per second, which *decreases* monotonically**:

| workers | batches/s | mean batch | evals/s |
|---|---|---|---|
| 30 | **64.93** | 7.85 | 510 |
| 36 | 57.15 | 10.51 | 601 |
| 48 | 55.03 | 11.63 | 640 |
| 64 | **51.88** | 12.83 | 666 |

`evals/s = batches/s × mean_batch` reproduces every row exactly
(64.93 × 7.85 = 510; 51.88 × 12.83 = 666). So **all throughput growth comes from
larger batches against a decaying batch rate** — the parent's serialized dispatch
cycle is getting slower under load, not faster.

Dividing cycle time by batch size gives the per-request parent cost:

```
30 workers:  15.4 ms / 7.85  = 1.96 ms per evaluation
64 workers:  19.3 ms / 12.83 = 1.50 ms per evaluation
```

Fixed per-batch overhead amortizes as batches grow, but the marginal cost
converges to **~1.5 ms of single-threaded Python per evaluation**, which
asymptotes throughput at ~666 evals/s. That is exactly where the 64-worker point
landed. **That 1.5 ms is the wall**, and no worker count can move it.

This is also why mean batch is a trap here: it rises 7.85 → 12.83 (+63 %) while
throughput rises 31 %, and it never approaches the configured ceiling of 32.

---

## CPU diagnosis: **parent/GIL limited** (Case B)

Explicitly, with evidence:

- **Worker-side CPU limited? No.** Worker CPU *falls* as workers rise —
  11.17 → 10.67 → 9.33 → 7.64 % mean. Workers get idler, because they block on
  the parent. Aggregate tree CPU is 335–462 % of **3600 %** available, and
  system-wide CPU never exceeds 13.2 %. There is enormous CPU headroom.
- **Scheduler limited? No.** Involuntary context switches stay negligible
  (98 → 144) while voluntary ones grow 80k → 197k. That signature is IPC
  blocking, not preemption thrashing — despite 64 processes on 18 physical cores.
- **Parent/GIL limited? Yes.** Parent CPU is **97.6 → 101.2 → 102.4 → 103.6 %**
  mean (p90 ~110 %) — one saturated core at *every* point, including the
  30-worker baseline. It was already the bottleneck before the sweep began.

Parent thread count grows **80 → 114** across the sweep: one `multiprocessing.Queue`
feeder thread per response queue, plus the two bridge pumps and the coordinator
thread, all contending for one GIL. Queue→dispatch p50 of 13.6–38.8 ms against a
**2 ms** configured window — 7–19× past budget — is the backlog this produces.

**This is not the old bistable collapse.** That failure mode showed mean batch
≈ 1.0 with batches ≈ requests. Here batches/requests = 0.127 and mean batch is
7.85, so the batch-window fix is working correctly. This is genuine saturation of
a healthy coordinator.

---

## GPU diagnosis: **severely underfed**

```
utilisation   4.8–5.0 % mean, p90 6–7 %, and slightly FALLING as workers rise
power         23.4–24.5 W of a 180 W cap
VRAM          924 MB of 16,311 MB
```

Case C is excluded outright. Utilisation *declining* while worker count doubles
is the signature of a device starved by its feeder.

Answering the question as posed — how many Soo evaluations per second can this
exact pipeline sustain, not what the device advertises — the answer is **~666/s,
set entirely by the host's single-core Python speed.** Nothing about the RTX 5060
Ti's capability enters into it. Precision, CUDA graphs, `torch.compile` and larger
batch ceilings are all premature: none of them address a device idle 95 % of the
time.

---

## Coordinator diagnosis

Healthy in mechanism, saturated in capacity.

| | 30 workers | 64 workers |
|---|---|---|
| queue→dispatch p50 / p90 | 13.64 / 16.41 ms | 38.78 / 57.74 ms |
| inference p50 / p90 | 9.95 / 16.07 ms | 18.72 / 20.10 ms |
| response p50 / p90 | 28.70 / 31.92 ms | 59.51 / 77.47 ms |
| batches / requests | 0.127 | 0.078 |
| mean / p90 / max batch | 7.85 / 16 / 19 | 12.83 / 18 / 23 |

Max batch reaches only 23 against a ceiling of 32, so **`max_batch_size` is not
binding** and does not confound the sweep.

The important mechanism: batch size here is limited by **drain rate, not arrival
rate**. The coordinator pops items for `max_wait_ms`, then spends 10–19 ms inside
`evaluate()` while requests accumulate. It gathers only 8–13 items during a 2 ms
window because *popping is itself GIL-bound*. Inference p50 nearly doubles
(9.95 → 18.72 ms) even though mean batch only rises 63 %, which points at the
Python-side cost inside `TorchEvaluator.evaluate` — building tensors from Python
lists of node features — competing with 80–114 parent threads for the GIL.

Transport is not implicated, consistent with the prior CPU pass measuring process
transport at sub-millisecond round trip. **Do not rewrite IPC on this evidence.**

---

## Best worker count

**48 workers / 48 games / 12 train steps.**

| | 36 | **48** | 64 |
|---|---|---|---|
| samples/hour | 33,844 | **36,059** | 37,510 |
| vs. best measured | −9.8 % | **−3.9 %** | — |
| parallel efficiency | 0.98 | **0.78** | 0.61 |
| response p50 | 32.6 ms | **50.5 ms** | 59.5 ms |
| free RAM at floor | 18.1 GB | **14.8 GB** | 10.3 GB |

64 workers measured the highest samples/hour and that is the primary objective,
so the recommendation needs its reasoning stated plainly: 48 captures **96 %** of
the best measured throughput while using 25 % fewer processes, holding 44 % more
free RAM, and keeping response latency 15 % lower. The 48→64 gain is **+4.0 %
from a single run per point** — the prior host held repeats to 0.2–0.4 % spread,
so it is probably real, but it was not confirmed here.

If maximum throughput is wanted and the memory headroom is acceptable, 64 is
defensible. It is not where the machine wants to run.

Per §12's ratio discipline, `train_steps_per_iteration` moves with the games so
the 4:1 games-per-optimizer-update ratio is preserved: 48 games → 12 updates.
This is a throughput recommendation only; learning semantics are unchanged.

---

## Recommended RTX 5060 Ti configuration

```
workers                    48
games_per_iteration        48
train_steps_per_iteration  12      (preserves 4 games : 1 optimizer update)
simulations                64
max_wait_ms                2       (carried over -- NOT measured on this host)
precision                  fp32
max_moves                  2000    (inert: p90 108-126 moves)
max_game_seconds           900     (never hit: zero aborts across 180 games)
```

**No `runtime/configs/soo-rtx5060.json` has been created.** Three of these values
are measured on this host; `max_wait_ms` is inherited, and adopting a canonical
config that pins an unmeasured timer would misrepresent the evidence. Create it
once the timer question below is settled.

---

## Open questions

### `max_wait_ms` is unresolved, and the RTX 3060 answer may not transfer

The shipped `max_wait_ms = 2` was **not** swept here. It should not be assumed
correct, for a subtler reason than the usual latency/batching trade-off.

On the RTX 3060 host, 2 ms beat 20 ms by 2.4×, because a longer window bought
larger batches at a latency cost that was not worth paying. **The regime here is
different.** Queue→dispatch p50 is 13.6–38.8 ms against a 2 ms window at every
point, and batch size is limited by how fast the coordinator can *pop* during its
window rather than by how fast requests arrive. A longer window may therefore
gather materially larger batches per unit of parent Python work — the opposite of
the RTX 3060 result.

This is cheap to answer (~30 min: 1/2/4/8 ms at 48 workers) and is the single
highest-value remaining measurement. Until it is run, `max_wait_ms = 2` is a
carried-over default, not a validated setting.

### Also unresolved

- **Is the +4.0 % from 48→64 real?** One run per point. Repeats would settle it.
- **Where does the ~1.5 ms per evaluation actually go?** The split between
  bridge routing, coordinator bookkeeping, and Python-side tensor construction in
  `TorchEvaluator` is unmeasured. A parent-side profile would size the prize
  before any rewrite.
- **Learning quality per wall-clock** remains unmeasured on this host, as on
  every previous one. Everything here is throughput.

---

## Next bottleneck: **a dedicated inference process**

Exactly one primary engineering target.

```
CURRENT
  workers -> mp request queue -> parent _InferenceBridge -> in-process
             coordinator -> TorchEvaluator -> GPU -> parent -> response queues

PROPOSED NEXT EXPERIMENT
  workers -> shared mp request queue -> dedicated inference process
                                          - coordinator
                                          - TorchEvaluator
                                          - owns the CUDA model
             -> worker response queues

  parent:  self-play orchestration, result collection, replay/training control
           NOT per-request inference forwarding
```

Why this and not something else:

- Separate interpreter, separate GIL. The measured constraint is one saturated
  core carrying the coordinator plus 80–114 queue-feeder threads; this is the
  minimal change that removes the parent from the per-request data plane.
- It preserves synchronous MCTS semantics entirely — no async search, no virtual
  loss, no speculative leaves.
- No native code required. C++/Rust batching should only be considered if the
  dedicated inference process *itself* is later measured as a single-core ceiling.

Explicitly **not** recommended from this evidence: async MCTS, virtual loss,
multiple outstanding leaves per game, shared-memory transport, BF16, CUDA graphs,
`torch.compile`, multi-GPU, adaptive wait, or a larger `max_batch_size`. Every
one of them targets a resource that is measurably idle.

A caveat worth stating: this host's weak 2.3 GHz Broadwell core is *itself* a
large part of the ceiling — the same code did 900 evals/s on the previous box's
faster core. Moving the inference data plane off the parent is the right change,
but on this CPU the ceiling after the change will still be a single-core Python
ceiling, just a higher one.

---

## Stop / go decision

**ARCHITECTURAL BOTTLENECK CONFIRMED.**

Throughput is limited by a single saturated parent core — pegged at ~100 % from
the 30-worker baseline onward — while the GPU idles at ~5 % and 35 of 36 CPU
threads sit unused. Adding workers has stopped converting into samples/hour:
the last 33 % of process count bought 4.0 %, at doubled response latency and a
third of the free memory. No configuration change reaches the constraint.

That said, this is **not** a blocker to training. 48 workers delivers 36,059
samples/hour, **2.8× the CPU baseline** (13,046). Training can start on the
recommended configuration today.

But note where that sits against history, because it is the whole thesis in one
number: the corrected stack on the **128-vCPU RTX 3060** box measured **49,722
samples/hour**. This host, with a newer and far more capable GPU, reaches **73 %
of that**. The regression is entirely attributable to a weaker parent core, and
it is the strongest single argument for moving the inference data plane off the
parent process.

The dedicated inference process is the next engineering target whenever
throughput becomes the binding constraint — and the `max_wait_ms` sweep should be
run first regardless, because it is 30 minutes and may be worth more than it
costs.

---

## Reproducing

```bash
export BENCH_SRC=/path/to/pinned/src        # read-only git archive of the commit
./az-bench/profiles/run_point.sh A-w30-g32 \
  az-bench/configs/soo-rtx5060-w30-g32.json 30 8

python az-bench/profiles/summarize_scaling.py \
  az-bench/soo/A-w30-g32 az-bench/soo/B-w36-g36 \
  az-bench/soo/C-w48-g48 az-bench/soo/D-w64-g64 --baseline A-w30-g32
```

`run_point.sh` asserts the source path and checkpoint digest before starting and
validates the `[resume]` line afterwards, so a mis-seeded or mis-imported run
fails instead of producing a plausible row. `--hours 0.01` runs exactly one
iteration: the loop checks its deadline before starting an iteration and never
interrupts one in progress.

Ledgers, per-run configs and 2 s sampler CSVs are preserved under
`az-bench/soo/{A,B,C,D}-*`. Regenerate the config copies with
`az-bench/profiles/make_bench_configs.py`.

**Sampling cost:** a 2 s process-tree + GPU sampler ran during every point, so
its small load is common to all rows. The GPU is read by one long-lived
`nvidia-smi -l` subprocess rather than a fork per tick, so sampler cost stays
flat as worker counts rise — which matters, because the previous pass established
that a per-second sampler is enough to perturb a latency-bound run.
