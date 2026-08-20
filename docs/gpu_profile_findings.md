# RTX 3060 GPU Profiling Findings

Second GPU measurement pass on the Soo AlphaZero pipeline, run on a vast.ai
RTX 3060 (12 GB, 32 vCPU) on 2026-08-20. Follows
[gpu_benchmark_findings.md](gpu_benchmark_findings.md) (first GPU pass) and
[cpu_profile_findings.md](cpu_profile_findings.md) (CPU rounds 1 and 2), and is
the GPU-side continuation those documents asked for.

**Status:** eight measured runs plus two isolated probes, all from one immutable
trained checkpoint. The headline finding is a reproducible coordinator
scheduling defect, not a tuning result.

---

## Headline

Three things, in order of how much they are worth:

1. **The inference coordinator is bistable.** With byte-identical config and
   seeds the same workload lands in either a batching regime (mean batch 8.35,
   721 s) or a collapsed regime (mean batch 1.05, 1001 s). The batch window is
   anchored to a request's *arrival* time rather than to when its batch opened,
   so under backlog every request flushes alone. Worth **~39 % of throughput**,
   awarded at random.
2. **64 simulations beat 32 outright** — +47 % samples/hour, 32/32 games
   completed instead of 26/32, and the pathological move tail disappears.
3. **`TorchEvaluator` spends 69 % of a batch-30 evaluation on per-row GPU→CPU
   syncs**, not on the network. The forward is flat at ~2.5 ms from batch 1 to
   batch 32.

The GPU itself is never the constraint. Peak observed utilisation is ~21 %, and
the network forward for 32 positions costs the same as for one.

---

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX 3060, 12 GB, driver 590.48.01 |
| vCPU | 32 (resolved to 30 self-play workers: available minus two) |
| torch | 2.13.0+cu130, CUDA runtime 13.0 |
| Python | 3.12.13 |
| Precision | FP32 throughout |
| Source checkpoint | `sha256:1634b901…6ea165af`, `training_step` 80 |

Every run below was seeded by copying that one checkpoint into a fresh run
directory, and every run logged `[resume] loaded … at training_step=80`. The
games are seed-deterministic, so all 32-simulation runs produced *identical*
games — 3,540 samples, median 111 moves, p90 282 — and all 64-simulation runs
likewise (3,064 samples, median 91, p90 114). Differences between rows are
therefore pure scheduling and inference behaviour, never different work.

**Note on the seed.** This is the step-80 checkpoint (the output of the earlier
GPU Run 1), not the step-72 `latest.pt.cpu-backup` named in
`gpu_benchmark_findings.md`. It is a trained network and it is identical across
every row here, so the comparisons below are internally exact; only
cross-comparison to the older CPU baseline carries that caveat.

A `nvidia-smi` sampler ran at 5 s intervals during every benchmark run, so its
small load is common to all rows.

---

## The runs

One 32-game iteration each, 30 workers unless noted, `max_moves` 2000,
`max_game_seconds` 900, FP32.

| run | sims | lanes | wait | games | s/game | compl/h | **samp/h** | p90 mv | batch | q→disp | inf | evals/s | aborts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wait2 | 32 | 30 | 2 | 26/32 | 40.7 | 71.8 | 9,778 | 282 | 1.00 | 19.5 | 3.00 | 257 | 6 |
| wait5 | 32 | 30 | 5 | 26/32 | 43.3 | 67.6 | 9,200 | 282 | 1.00 | 20.0 | 3.00 | 238 | 6 |
| wait10-s32 | 32 | 30 | 10 | 26/32 | 43.1 | 67.8 | 9,234 | 282 | 1.00 | 20.6 | 2.98 | 231 | 6 |
| wait20 | 32 | 30 | 20 | 26/32 | 38.2 | 76.6 | 10,431 | 282 | 7.17 | 19.0 | 7.86 | 239 | 6 |
| s32-w60 | 32 | **32** | 20 | 26/32 | 28.2 | 103.7 | 14,115 | 282 | 1.59 | 26.2 | 2.97 | 277 | 6 |
| **s64-w30** | **64** | 30 | 20 | **32/32** | **22.5** | **159.9** | **15,309** | **114** | 8.35 | 16.8 | 5.74 | 272 | **none** |
| s64-prof | 64 | 30 | 20 | 32/32 | 31.3 | 115.1 | 11,017 | 114 | 1.05 | 87.5 | 3.01 | 195 | none |
| s64-prof2 | 64 | 30 | 20 | 32/32 | 31.3 | 115.1 | 11,022 | 114 | 1.05 | 87.0 | 2.99 | 196 | none |

CPU reference for scale: 24.9 s/game and 13,046 samples/h at 4 workers, 32 sims
(from `gpu_benchmark_findings.md`).

`s64-prof` and `s64-prof2` are repeats of `s64-w30` taken while sampling lane
occupancy. They are listed because their disagreement with `s64-w30` turned out
to be the most important result here, not an artefact.

---

## Finding 1 — the coordinator's batch window is bistable

`s64-w30`, `s64-prof` and `s64-prof2` share a config file, a checkpoint, a seed
and their game outcomes. They do not share their throughput:

| | s64-w30 | s64-prof | s64-prof2 |
|---|---|---|---|
| self-play wall | **721 s** | 1001 s | 1001 s |
| mean batch | **8.35** | 1.05 | 1.05 |
| queue→dispatch p50 | **16.75 ms** | **87.54 ms** | **87.00 ms** |
| inference p50 | 5.74 ms | 3.01 ms | 2.99 ms |
| worker response p50 | 32.8 ms | 90.5 ms | 90.1 ms |
| evals/s | 272 | 195 | 196 |
| batches dispatched | 23,446 | 186,322 | 186,642 |

Two repeats agree to three significant figures with each other and disagree by
39 % with the third. This is not noise; these are two stable operating points.

### Mechanism

`InferenceCoordinator._run` (`src/diamond/alphazero/inference/coordinator.py`)
decides when a pending batch is due:

```python
due_keys = [
    key
    for key, batch in pending.items()
    if batch[0].submitted_at + self.config.max_wait_ms / 1000 <= now
]
```

`submitted_at` is stamped in `submit()`, which runs on the bridge thread the
moment a request *arrives at the parent* — before it is queued on
`self._requests` and long before the batching thread pops it:

```python
def submit(self, request: object, reply_queue: Queue[object]) -> None:
    submitted_at = monotonic()
    ...
```

So the wait window is measured from arrival, not from batch open. Once the
request queue carries a backlog deeper than `max_wait_ms`, every request the
batching thread pops is **already past its deadline**, `due_keys` fires
immediately, and the batch flushes holding exactly one item. Dispatching one at
a time yields ~196 evals/s, which is slower than arrivals, which deepens the
backlog. The collapsed state sustains itself; so does the healthy one.

The measured queue→dispatch times are exactly what that predicts: **87 ms
against a 20 ms configured window**, i.e. requests waiting 4.4× their budget
before dispatch, with `batches ≈ requests`.

It also explains the one anomaly flagged during the run: `s32-w60` shows batch
1.59 with q→dispatch p50 26 ms but p90 95 ms — a run that spent part of its
time in each regime.

### Fix

The batch deadline should start when the batch opens, not when the request
arrived. Concretely: stamp a third timestamp when an item is appended to an
*empty* `pending[key]`, and compute `due_keys` from that. Under backlog the
batch then fills to `max_batch_size` and flushes on size rather than on a
deadline that has already passed — which is the behaviour the configuration
intends.

Note that `_QueuedRequest` already carries an `admitted_at` field, but it is
stamped inside `submit()` microseconds after `submitted_at`, so switching to it
would not help. A new batch-open timestamp is required.

**Confidence:** the two regimes and the 87 ms-vs-20 ms figure are measured; the
mechanism is read from the code and is consistent with every row in the table.
It has not yet been reproduced in an isolated unit test. That test is the first
thing to write before changing anything — see *Recommended next work*.

---

## Finding 2 — 64 simulations beat 32, decisively

Holding `max_wait_ms` at 20 and lanes at 30, changing only the simulation count:

| | wait20 (32 sims) | s64-w30 (64 sims) |
|---|---|---|
| completed | 26/32 | **32/32** |
| aborts | 6 | **none** |
| median / p90 moves | 111 / 282 | **91 / 114** |
| s/game | 38.2 | **22.5** |
| **samples/hour** | 10,431 | **15,309** (+47 %) |

Twice the search per move, and it finishes faster in wall-clock while throwing
away nothing. The historical claim in `blueprint/gpu_train.md` — that 64
simulations produce far more stable heuristic-off trajectories (p90 147 vs
1667) — reproduces here on the trained network: the p90 move count falls from
282 to 114 and the abort tail disappears entirely.

At 32 simulations, six of every 32 games (19 %) hit the 900 s per-game deadline,
contributed **zero** training samples, and set the iteration's wall-clock while
the other lanes sat idle. That is the entire reason the 32-sim rows look slow.

**32 simulations with `max_moves` 2000 should not be used on this hardware.**

---

## Finding 3 — lane double-booking costs 35 %

`selfplay_workers.py` assigns jobs to lanes round-robin, up front:

```python
lane_count = min(self.worker_count, len(jobs))
...
for index, job in enumerate(jobs):
    job_queues[index % lane_count].put(job)
```

With 32 games over 30 lanes, lanes 0 and 1 each receive **two** games. A lane's
second game cannot start until its first finishes — and if the first is one of
the pathological games that runs to the 900 s deadline, the second game starts
15 minutes late and extends the iteration by up to another 900 s.

Measured, at identical settings and identical games:

| | lanes | self-play wall | samples/h |
|---|---|---|---|
| wait20 | 30 | 1222 s | 10,431 |
| s32-w60 | **32** | **903 s** | **14,115** (+35 %) |

903 s is one 900 s deadline plus change; 1222 s is a deadline plus a partly-run
second game. Giving every job its own lane removed the serialisation and bought
35 % samples/hour with no other change.

Two consequences:

- There is no work stealing. A lane that finishes early cannot take pending
  work, so a finished lane idles while another is double-booked. A single shared
  job queue would fix both this and the tail-idling in Finding 5.
- **`--workers 60` does not create 60 lanes.** `lane_count` is capped at
  `len(jobs)`, so with `games_per_iteration = 32` the run used 32 lanes. The
  oversubscription question this run was designed to answer is therefore still
  **untested**; testing it requires raising `games_per_iteration` above the
  worker count.

---

## Finding 4 — `TorchEvaluator` is dominated by per-row syncs

Isolated CUDA benchmark, real trained weights, no coordinator and no
multiprocessing (`az-bench/profiles/bench_evaluator.py`, 73×4 features, 50 legal
actions, mean of 60 iterations after 10 warm-ups):

| batch | evaluate | forward | postproc | post/row | per request |
|---|---|---|---|---|---|
| 1 | 2.52 ms | 2.28 ms | 0.24 ms | 0.243 ms | 2.52 ms |
| 2 | 2.80 ms | 2.59 ms | 0.22 ms | 0.109 ms | 1.40 ms |
| 4 | 3.30 ms | 2.44 ms | 0.86 ms | 0.214 ms | 0.82 ms |
| 8 | 4.09 ms | 2.74 ms | 1.35 ms | 0.169 ms | 0.51 ms |
| 16 | 5.53 ms | 2.42 ms | 3.12 ms | 0.195 ms | 0.35 ms |
| 30 | 7.97 ms | 2.49 ms | **5.48 ms** | 0.183 ms | 0.27 ms |
| 32 | 8.18 ms | 2.56 ms | 5.62 ms | 0.176 ms | 0.26 ms |
| 64 | 15.87 ms | 4.28 ms | 11.59 ms | 0.181 ms | 0.25 ms |

**The network forward is flat at ~2.5 ms from batch 1 to batch 32.** Thirty-two
positions cost the GPU what one costs. Everything above that line is the
per-row Python loop in `evaluator/torch.py`, at a steady ~0.18 ms per row:

| batch | forward | per-row sync loop | sync share |
|---|---|---|---|
| 1 | 2.28 ms | 0.24 ms | 10 % |
| 8 | 2.74 ms | 1.35 ms | 33 % |
| **30** | 2.49 ms | **5.48 ms** | **69 %** |

The loop performs roughly six device synchronisations per row —
`int(legal.min())`, `int(legal.max())`, `torch.isfinite(...).all()`,
`float(probabilities.sum())`, `probabilities.cpu()`, `values[row].cpu()` — so a
batch of 30 serialises ~180 syncs. Vectorising it (one batched `index_select`,
one `.cpu()` transfer, validation on the batched tensor) should take batch-30
`evaluate()` from ~8 ms to ~2.5–3 ms.

This is the follow-up `blueprint/gpu_train.md` predicted and deliberately
deferred, now with a measurement behind it.

### What it is worth

Inference is only 5.74 ms of s64-w30's 32.8 ms round trip, so latency alone
improves ~10 %. The gain is on the coordinator's serialised dispatch cycle,
which the data models well:

```
evals/s  ~=  batch_size / (max_wait_ms + inference_time)

measured:   8.35 / (0.020 + 0.0057)  =  325/s      (observed 272/s)
batch 30:  30    / (0.020 + 0.0057)  = 1167/s
+ vectorised evaluator:
           30    / (0.020 + 0.0025)  = 1333/s      (~5x today)
```

That ~5× agrees with the headroom `gpu_benchmark_findings.md` estimated from the
other direction, and it needs **both** halves: full lanes to supply batch 30,
and a vectorised evaluator to keep the cycle short.

---

## Finding 5 — lanes are busy 52 % of the time, and the tail is timer-bound

1 Hz sampling of live lane count through a repeat s64 run
(`az-bench/profiles/probe-b-lane-occupancy.csv`), reproduced twice:

```
t+   0s  lanes 30.0  gpu 20.8%  ##############################
t+ 100s  lanes 30.0  gpu 21.2%  ##############################
t+ 200s  lanes 30.0  gpu 21.1%  ##############################
t+ 300s  lanes 30.0  gpu 21.2%  ##############################
t+ 400s  lanes 26.5  gpu 21.5%  ###########################
t+ 499s  lanes  7.5  gpu  9.1%  #######
t+ 599s  lanes  1.4  gpu  3.1%  #
t+ 699s  lanes  1.0  gpu  2.5%  #
t+ 799s  lanes  1.0  gpu  2.6%  #
t+ 899s  lanes  1.0  gpu  2.6%  #
```

| | |
|---|---|
| Lane occupancy | **52.7 %** of peak (52.3 % on the repeat) |
| Mean / median lanes | 15.8 / 19 |
| Time at 30 lanes | 40.3 % |
| **Time at exactly 1 lane** | **36.0 %** |
| GPU util, concurrent phase | ~21 % |
| GPU util, single-lane tail | ~2.6 % |

Not a gentle taper: full occupancy for ~400 s, a ~150 s cliff as games finish,
then one lane alone for over a third of the run.

**The tail is gated by the batching timer, not by compute.** With one lane
alive, no second request can ever arrive, so every evaluation waits the full
`max_wait_ms` for a batch that cannot grow:

```
tail round trip  ~=  20 ms wait + 3 ms inference + ~1 ms transport  ~=  24 ms
one ~250-move game x 64 sims  =  ~16,000 evals x 24 ms  ~=  380 s
```

which matches the observed ~300–400 s tail. At `max_wait_ms` 2 the same tail
would be roughly 4× shorter.

This is precisely the "flush when nothing more can arrive" optimisation that CPU
round 2 implemented and reverted. It lost there because on a 4-worker box the
queue is briefly empty *between* arrivals, so flushing destroyed real batching.
The GPU host supplies the regime where it wins: 36 % of wall-clock at literally
one outstanding request, where there is no batch to protect. An adaptive
wait — full `max_wait_ms` while lanes are full, immediate flush once outstanding
requests fit in one batch — targets the tail without touching the concurrent
phase.

---

## Finding 6 — `max_wait_ms` has a threshold between 10 and 20 ms

| wait | mean batch | max batch | samples/h |
|---|---|---|---|
| 2 | 1.00 | 1 | 9,778 |
| 5 | 1.00 | 2 | 9,200 |
| 10 | 1.00 | 2 | 9,234 |
| **20** | **7.17** | **30** | 10,431 |

At 2–10 ms the batch never forms — at `wait2`, `max_batch_size` was **1** across
all 334,885 requests. At 20 ms it forms properly and reaches the configured
ceiling. The CPU optimum of 5 ms does not transfer, exactly as
`cpu_profile_findings.md` warned.

But note what the fourth column does *not* do: forming batches bought only
**+7 %** throughput, because those runs are gated by the 900 s abort tail rather
than by inference. `max_wait_ms` is a real dial that was being measured through
a confound. Any future sweep of it must run on a workload without the tail —
i.e. at 64 simulations.

---

## Recommended next work, in order

1. **Fix the bistable batch window** (Finding 1). Highest value: it is worth
   ~39 % of throughput, it fires at random, and it silently disables batching
   exactly when load is highest. TDD: a unit test that submits a burst larger
   than `max_batch_size` while the batching thread is blocked, then asserts the
   next dispatch carries more than one request. It fails today.
2. **Adopt 64 simulations for GPU training** (Finding 2). A config change worth
   +47 % samples/hour and 6 fewer discarded games per iteration.
3. **Vectorise `TorchEvaluator` postprocessing** (Finding 4). ~3× on the
   inference stage; compounds with items 1 and 4.
4. **Give every job its own lane, or add work stealing** (Findings 3 and 5).
   The cheap version is `games_per_iteration <= lane_count`; the real fix is a
   single shared job queue so finished lanes take pending work. Preserve the
   4:1 games-per-optimizer-update ratio if `games_per_iteration` changes.
5. **Adaptive batching wait** (Finding 5), only after item 1 — the two interact,
   and item 1 may change what the tail looks like.

Explicitly *not* recommended from this evidence: BF16, CUDA graphs,
`torch.compile`, tree parallelism, or a shared-memory transport. CPU round 2
measured process transport at 0.87 ms round trip, and nothing here contradicts
that.

---

## Open questions

- **Does oversubscription help?** Still untested: `lane_count` is capped at
  `games_per_iteration`, so `--workers 60` gave 32 lanes. Needs
  `games_per_iteration` raised above the worker count.
- **What tips the coordinator into the collapsed regime?** Two of three repeats
  collapsed; the trigger is presumably backlog depth at start-up, when 30
  workers fire their first request simultaneously. Unconfirmed.
- **Is `max_moves` 2000 right at 64 simulations?** p90 is 114 moves and no game
  hit the move cap, so the cap is inert here — but a lower cap would not have
  been, and the 32-sim rows show one `max_game_moves_exceeded`.
- **`max_wait_ms` curve at 64 simulations.** The 2/5/10/20 sweep was run at 32
  simulations, through the abort-tail confound. It should be re-measured on the
  64-sim workload, where the concurrent phase dominates.
- **Learning quality over a fixed wall-clock** remains unmeasured. Everything
  here is throughput. The fair CPU-vs-GPU learning experiment in
  `blueprint/gpu_train.md` — same checkpoint, same wall-clock, then the same
  heuristic-off probe — has not been run.

---

## Correctness

`pytest tests/alphazero tests/agents tests/tools` — **474 passed, 0 failed** on
this host, including the 7 CUDA tests that skip on CPU-only CI. They cover
CPU/CUDA FP32 evaluation parity on the same checkpoint.

Two of those tests were failing when this session started, because
`runtime/configs/soo-rtx3060.json` had been hand-edited during the earlier GPU
session (`inference.max_wait_ms` 2→20, `arena.max_moves` 2000→200) without the
tests being updated. Both values were restored to their committed intent, which
is also the documented Run 1 baseline. Per-run `max_wait_ms` variants live in
`az-bench/configs/` so the reference config no longer has to be edited in place
to run a sweep.

---

## Reproducing

Setup and recovery are in [gpu_training.md](gpu_training.md). Each run below is
one 32-game iteration, seeded from the same immutable checkpoint:

```bash
mkdir -p az-bench/soo/<name>
cp runtime/runs/soo/cpu8h-soo-20260819/latest.pt az-bench/soo/<name>/latest.pt

python tools/az_train.py --config az-bench/configs/soo-rtx3060-wait20.json \
  --runtime-dir az-bench --run-id <name> --migrate-device \
  --workers 30 --simulations 64 --train-steps-per-iteration 8 --hours 0.01

python tools/az_report.py az-bench/soo/<name>
```

`--hours 0.01` runs exactly one iteration: the loop checks its deadline before
starting an iteration and never interrupts one in progress.

Verify every run logs `[resume] loaded … at training_step=80`. A
`[init] wrote initial checkpoint` line means the copy failed and the run is
training a random network — discard it.

The isolated evaluator benchmark needs no training run and takes ~7 seconds:

```bash
python az-bench/profiles/bench_evaluator.py
```

**Do not profile with a per-second sampler.** Two attempts to trace lane
occupancy at 1 Hz and 0.5 Hz both landed in the collapsed coordinator regime.
That turned out to be Finding 1 rather than observer effect — the samplers were
not the cause, since two different samplers produced identical results — but
until Finding 1 is fixed, any run may land in either regime, so read
`queue_to_dispatch_p50_ms` before trusting a throughput number. Roughly 17–19 ms
means the batch window is healthy; ~87 ms means it collapsed and the run's
throughput is not comparable.
