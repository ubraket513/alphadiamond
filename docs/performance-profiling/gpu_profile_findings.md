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

> **Superseded — this finding is wrong.** Its conclusion is an artefact of the
> coordinator bug in Finding 1: with the batch window anchored to arrival, a
> short `max_wait_ms` could not batch by construction. Re-measured after the
> fix the curve runs the other way — 2 ms beats 20 ms by 2.4x. Kept as
> evidence; see *Follow-up stack* below for the corrected table.

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

> **Status:** all five items below have since been actioned. See
> *Coordinator batch-window fix validation* and *Follow-up stack* for what
> shipped, what was measured, and what was deliberately not merged.

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

---

## Coordinator batch-window fix validation

Added 2026-08-21, after the fix predicted by Finding 1 was implemented. The
findings above are retained unchanged: they are the evidence that motivated this
change, and the "before" half of the comparison below.

**Result: the bistability is eliminated.** Six controlled runs — three with the
old coordinator and three with the new one, interleaved — separate completely,
with no overlap and no run landing in the other arm's regime.

### The failing mechanism

`InferenceCoordinator._run` computed a pending batch's deadline from
`batch[0].submitted_at`, which `submit()` stamps on the caller's thread the
moment a request arrives at the parent — before it is queued on `self._requests`
and long before the batching thread pops it. Once the request queue carried a
backlog deeper than `max_wait_ms`, every request the batching thread popped was
*already past its deadline*: the batch it opened flushed immediately holding one
item, one-at-a-time dispatch fell behind arrivals, and the backlog deepened. The
collapsed state sustained itself, and so did the healthy one.

### The code-level fix

`src/diamond/alphazero/inference/coordinator.py`, 29 insertions / 14 deletions.
Pending batches changed from bare `list[_QueuedRequest]` to a `_PendingBatch`
carrying `opened_at`, stamped when the first *validated* request joins an empty
batch:

```python
batch = pending.get(request.model_key)
if batch is None:
    # The window opens now, with this request -- not when it was sent.
    batch = _PendingBatch(opened_at=monotonic())
    pending[request.model_key] = batch
```

Both the due-key test and the `get()` timeout now derive from `batch.opened_at`
rather than `batch[0].submitted_at`. Under backlog the batch therefore fills to
`max_batch_size` and flushes on size, which is what the configuration intends.

The change is scheduling semantics only. `_record_batch` is untouched, so
queue-to-dispatch, admission, inference and response latencies all keep measuring
real request time — a request that genuinely waited 80 ms in the queue still
reports 80 ms. Batching by model key, bounded admission, queue capacity,
malformed-request handling, correlated `InferenceFailure`, response routing,
shutdown, outstanding-request cleanup and thread safety are all unchanged.

Note the deliberate non-changes: no adaptive wait, no `TorchEvaluator`
vectorization, no work stealing, no change to the canonical config. Bundling any
of them would have made the attribution below impossible.

### Focused test

`tests/alphazero/inference/test_coordinator.py::test_backlogged_requests_open_a_fresh_batch_window_instead_of_flushing_alone`
parks the batching thread inside a gated evaluator, queues a burst of three
behind it, ages the burst past `max_wait_ms`, then releases. It asserts the
backlog is dispatched as one batch of three.

On the pre-fix coordinator it fails with `assert 1 == 3` — the measured defect,
reproduced deterministically without relying on scheduler luck. It passes after
the fix, and passed 15 consecutive runs with no flake.

Two companion guards ship with it: `queue_to_dispatch` must still report the
request's real wait (so the new timer cannot falsify the metric), and each
`ModelKey` must keep an independent window.

### CPU test suite

`pytest tests/alphazero tests/agents tests/tools` → **477 passed**, against the
474-passing baseline plus the three new tests. CUDA-gated tests ran rather than
skipped on this host.

### Repeated GPU results

One 32-game iteration each, all six seeded from a fresh copy of the same
immutable step-80 checkpoint (`sha256:1634b901…`), all logging
`[resume] loaded … at training_step=80`. Identical config
(`az-bench/configs/soo-rtx3060-wait20.json`), 64 simulations, 30 workers,
`max_wait_ms` 20, FP32, seed 7. Arms were interleaved pre/post/pre/post/pre/post
so time-ordered drift could not favour either. Ledgers are preserved under
`az-bench/soo/batchfix-*`; regenerate the table with
`az-bench/profiles/summarize_batchfix.py`.

| run | wall | samp/h | games | mean b | max b | q→disp p50 | inf p50 | resp p50 | evals/s | batches | requests |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *hist* s64-w30 | 720.5 | 15,250 | 32/32 | 8.35 | 30 | 16.75 | 5.74 | 32.8 | 272 | 23,446 | 195,659 |
| *hist* s64-prof | 1001.2 | 10,987 | 32/32 | 1.05 | 9 | 87.54 | 3.01 | 90.5 | 195 | 186,322 | 195,659 |
| *hist* s64-prof2 | 1000.8 | 10,993 | 32/32 | 1.05 | 8 | 87.00 | 2.99 | 90.1 | 196 | 186,642 | 195,659 |
| pre-1 | 1231.4 | 8,361 | 31/32 | 1.04 | 6 | 133.41 | 4.52 | 137.9 | 156 | 185,190 | 192,512 |
| pre-2 | 1232.9 | 8,351 | 31/32 | 1.04 | 6 | 134.22 | 4.55 | 138.6 | 156 | 185,014 | 192,351 |
| pre-3 | 1233.7 | 8,345 | 31/32 | 1.04 | 6 | 132.63 | 4.50 | 137.3 | 156 | 185,276 | 192,613 |
| **post-1** | **778.2** | **14,125** | 32/32 | **8.35** | **30** | **17.31** | 8.43 | 30.5 | 251 | **23,439** | 195,659 |
| **post-2** | **781.2** | **14,071** | 32/32 | **8.35** | **30** | **17.36** | 8.41 | 30.4 | 250 | **23,439** | 195,659 |
| **post-3** | **778.3** | **14,123** | 32/32 | **8.35** | **30** | **17.36** | 8.29 | 30.2 | 251 | **23,440** | 195,659 |

### Bistability status: **eliminated**

Each arm is internally reproducible to three significant figures, and the arms do
not come close to touching:

| | pre-fix (n=3) | post-fix (n=3) |
|---|---|---|
| self-play wall | 1231.4 – 1233.7 s (0.19 % spread) | 778.2 – 781.2 s (0.39 % spread) |
| mean batch | 1.04 every run | 8.35 every run |
| max batch | 6 | 30 (the configured ceiling) |
| queue→dispatch p50 | 132.6 – 134.2 ms | 17.31 – 17.36 ms |
| batches / requests | 0.96 | 0.12 |

No post-fix run shows the pathological combination the fix targets — mean batch
≈ 1 **and** queue→dispatch ≫ `max_wait_ms` **and** batches ≈ requests. Every
pre-fix run shows all three. Post-fix queue→dispatch p50 sits *inside* the 20 ms
window for the first time; pre-fix it was 6.7× past it.

The pre-fix arm collapsed 3/3 here where the original host collapsed 2/3, and
did so harder (133 ms vs 87 ms queue→dispatch). See the caveat below.

### Throughput impact

**+69 % samples/hour**, 8,352 → 14,106 (arm means), and **+61 % evals/s**,
156 → 251. Self-play wall falls 36.9 %, 1232.7 s → 779.2 s.

The pre-fix runs were also slow enough that one game per run hit the 900 s
per-game deadline and contributed zero samples — 31/32 completed, every time.
All three post-fix runs completed 32/32, matching the historical healthy row.
That abort is a *consequence* of the collapse, not different work: median and p90
move counts are identical across arms once the aborted game is excluded, and both
arms issue ~195 k requests from the same seed-deterministic games.

Inference p50 rose 4.5 ms → 8.4 ms post-fix. That is expected and desirable: a
batch of 30 costs more per call than a batch of 1, and far less per request. It
does mean the fix has moved cost onto the per-row sync loop of Finding 4, which
raises that item's value — see below.

### Caveat: this host is not the host of the runs above

The validation ran on a **128-vCPU** box, not the 32-vCPU host that produced
every earlier row in this document. `--workers 30` was passed explicitly, since
`resolve_worker_count()` returns 126 here.

Absolute wall-clock is therefore not comparable across the horizontal rule: the
healthy post-fix runs take 778 s where the historical healthy run took 720 s.
What *is* comparable is the pre-vs-post contrast, which was measured on this box
under identical conditions and is the claim being made.

Worker-side CPU contention is essentially absent here, so the concern that the
larger box might mask the defect was real — it did not. The parent process is
pegged at ~100 % of a single core (coordinator plus 30 bridge threads under one
GIL) regardless of core count, and that is where the backlog forms. Total system
utilisation during a run is ~180 % of 12,800 % available, with the GPU at ~14 %:
this workload is latency-bound at the coordinator, exactly as Findings 1 and 4
describe.

### Effect on the recommended-work ranking

The order at the end of the previous section still holds, with item 3 promoted in
value by this measurement:

1. **Adopt 64 simulations for GPU training** (Finding 2) — unchanged, +47 %, a
   config change with the evidence already in hand.
2. **Vectorise `TorchEvaluator` postprocessing** (Finding 4) — *worth more than
   before*. Pre-fix, the per-row loop was nearly free because every batch held
   one row; post-fix, batches average 8.35 and reach 30, and inference p50 has
   risen accordingly. Applying the document's own model,
   `evals/s ≈ batch / (max_wait + inference)`: today
   `8.35 / (0.020 + 0.0084) = 294/s` against 251 observed, and cutting inference
   to ~2.5 ms gives `8.35 / 0.0225 = 371/s` (~+25 %). With full lanes at batch 30
   it is `30 / 0.0225 ≈ 1330/s`.
3. **Give every job its own lane, or add work stealing** (Findings 3 and 5).
4. **Adaptive batching wait** (Finding 5) — now unblocked, since item 1 of the
   old list is done and the tail can be re-measured against a stable batching
   regime.

---

## Follow-up stack: evaluator, lane scheduling, and the corrected `max_wait_ms`

Added 2026-08-21, after the coordinator fix above. Covers the remaining three
items from *Recommended next work*, plus a correction to Finding 6 that the
coordinator fix made visible.

### Read this first: a measurement error, and what it invalidated

`diamond` is installed into `/venv/main` as an **editable** package, so
`site-packages/*.pth` contains `/workspace/alphadiamond/src`. Every benchmark
run therefore imports the *main working tree*, no matter which directory it runs
from. A `git worktree` does **not** isolate a run.

An initial `max_wait_ms` sweep and an initial adaptive-wait A/B were taken while
that was misunderstood, with commits landing between runs. Both were discarded:
the sweep had a different code version at each of its four points, and the A/B
ran the same code in both arms. Every number in this section comes from re-runs
that pin the source with `PYTHONPATH=<snapshot>/src` and assert on
`diamond.__file__` before starting, so a wrong import fails the run instead of
silently producing a plausible row.

The tell, for next time: `wait20` measured 218 s in the contaminated sweep
against 778 s for the same setting an hour earlier. A timer knob does not move
wall-clock 3.6x. Identical config with a large unexplained delta means the code
changed, not the configuration.

### Finding 6 is wrong, and the coordinator bug is why

Finding 6 concluded that batches only form at `max_wait_ms` 20 and that "the CPU
optimum of 5 ms does not transfer". That was an artefact. With the window
anchored to arrival, a short `max_wait_ms` guaranteed instant expiry, so the
short-wait rows could not batch *by construction*.

Re-measured with the window fixed, at 64 simulations and 30 workers, one code
version, changing only the timer:

| `max_wait_ms` | self-play wall | samples/h | mean batch | q→disp p50 | evals/s |
|---|---|---|---|---|---|
| **2** | **216.8 s** | **50,259** | 8.97 | 8.75 ms | 903 |
| 5 | 261.2 s | 41,803 | 9.89 | 10.71 ms | 749 |
| 10 | 333.6 s | 32,800 | 12.85 | 7.26 ms | 587 |
| 20 | 518.5 s | 21,162 | 12.87 | 16.43 ms | 377 |

Monotonic, and it inverts the old table. Note especially that **larger batches
are worse here**: mean batch rises from 8.97 to 12.87 while throughput falls by
58%. Batch size was never the objective; it is a proxy that stops tracking
throughput once the wait dominates. The shipped `max_wait_ms = 2` in
`runtime/configs/soo-rtx3060.json` is correct and was left alone.

### The stack, one change at a time

Each row changes one thing from the row above. All from the same immutable
step-80 checkpoint, 64 simulations, 30 workers, 32 games, FP32.

| stage | wait | wall | samples/h | mean batch | evals/s | runs |
|---|---|---|---|---|---|---|
| pre-fix coordinator (collapsed) | 20 | 1232.7 s | 8,352 | 1.04 | 156 | 3 |
| + batch-window fix | 20 | 779.2 s | 14,106 | 8.35 | 251 | 3 |
| + vectorized evaluator + shared job queue | 20 | 518.5 s | 21,162 | 12.87 | 377 | 1 |
| same, at the shipped timer | **2** | **219.2 s** | **49,722** | 8.9 | 900 | 3 |
| + adaptive wait (branch only) | 2 | 218.3 s | 49,909 | 7.05 | 897 | 2 |

Isolating the middle pair at the shipped timer, which is the comparison that
matters for production:

| | wall | samples/h |
|---|---|---|
| batch-window fix alone, `wait2` | 319.2 s | 34,261 |
| + vectorized evaluator + shared job queue | 219.2 s | 49,722 |
| | **−31 %** | **+45 %** |

There is deliberately **no** measured "pre-fix at `wait2`" row. It would have
cost ~40 minutes of paid GPU to restate a result already established three runs
per arm at `wait20`, so the total is reported as its two measured legs rather
than as a single headline multiple.

### 64 simulations adopted in the canonical config

`runtime/configs/soo-rtx3060.json` now carries `mcts.simulations = 64`, on the
Finding 2 evidence: 32/32 games completed instead of 26/32, p90 moves 282 -> 114,
and +47 % samples/hour, from twice the search per move that still finishes faster
in wall-clock. The CPU configs keep 32 — this is a GPU tuning knob and the
measurement behind it was taken on the GPU.

Learning semantics are untouched: `simulations` is a search knob, not a
self-play training-target field, and the 4:1 games-per-optimizer-update ratio is
unchanged. The guard in `tests/tools/test_cpu_b0_train.py` was updated to assert
64 and to carry the reason.

Every run in this section passes `--simulations 64` explicitly, so this config
change does not affect any number reported here; it affects production runs that
take the config's default.

### `TorchEvaluator` vectorization

The per-row loop issued ~6 device synchronisations per request, so a batch of 30
serialised ~180 of them — 5.48 ms of a 7.97 ms call. Replaced by: host-side
bounds checks (the ids are already Python tuples), one padded `gather` masked to
`-inf` so padding contributes exactly zero to each row's softmax, one scalar
validity read for the whole batch, and two host transfers total.

Guarded structurally rather than by timing: a test counts `Tensor.cpu()` calls
across two batch sizes and requires the count to be constant. It measured 4 at
batch 2 against 48 at batch 24 before the change.

Semantics are asserted against one-at-a-time evaluation on both CPU and CUDA, so
narrow rows in a wide batch keep the values they have alone. One deliberate
ordering change: policy validity is now checked for the whole batch before any
row's value finiteness, so a model corrupt in both ways raises the policy error.

### Shared job queue

Jobs were pre-assigned round-robin to per-lane queues, so 32 games over 30 lanes
double-booked lanes 0 and 1, and a lane's second game could not start until its
first finished. All lanes now pull from one shared queue with one stop sentinel
each; work stealing falls out of that.

Game outcomes are unaffected — seeds derive from `run_seed`, `iteration`,
`game_index`, `model_key` and `retry_id`, never from the lane. What changes is
which lane runs which game, so `EpisodeResult.worker_id` is no longer
predictable from the job index. One existing test pinned that mapping and was
measured flaky 2-in-8 once lanes could steal; it now asserts only that the lane
ids are valid.

### Adaptive wait: measured, and deliberately not merged

Rule: flush once the pending batches already hold every outstanding request,
since no lane is left to contribute another. Keyed on requests that *cannot*
arrive, not on the queue being momentarily empty.

| | `wait2` (shipped) | `wait5` |
|---|---|---|
| without | 49,722 samples/h | 41,600 samples/h |
| with | 49,909 samples/h | 50,317 samples/h |
| | +0.4 % (noise) | **+21 %** |

It buys nothing at the setting actually shipped. What it does is make throughput
stop depending on the setting: the "with" arm lands on 216–219 s whether the
timer is 2 or 5, removing a knob that costs up to 2.4x when mis-set.

It is **not on `main`** — it lives on branch `adaptive-wait`. Two reasons. It
revokes the documented guarantee that a lone request waits its full window, for
no measured gain at the shipped setting; and the 4-worker CPU regime is
unverified, which is precisely where the analogous heuristic regressed and was
reverted in CPU round 2. Merging needs that CPU measurement first.

### Ledgers

Under `az-bench/soo/stack-*`, one directory per run, with the config each used.
Regenerate the tables with `az-bench/profiles/summarize_stack.py`.

### Where the bottleneck is now

At `wait2` with the stack applied: 219 s of self-play, ~900 evals/s, inference
p50 6.5 ms, response p50 17 ms, mean batch 8.9 against a ceiling of 32. The
coordinator's single parent process still saturates one core (coordinator plus
30 bridge threads under one GIL) while the GPU sits near 14 %, so the next real
constraint is parent-side concurrency, not the device.

Still untested from the original open questions: whether oversubscription helps
(`lane_count` was capped at `games_per_iteration`; the shared queue changes what
that experiment would mean), and whether `max_moves` 2000 is right at 64
simulations.
