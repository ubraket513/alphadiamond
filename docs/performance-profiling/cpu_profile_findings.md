# CPU Profiling Findings

Where one Soo MCTS simulation actually spends its wall-clock time, measured on
CPU before spending any RTX 3060 rental time.

**Checkpoint under test:** `sha256:4b2a32ff15179e890d4266346bca178d9a255eebe16af3a6e3d0482f0ceb1320`,
run `cpu8h-soo-20260819`, `training_step` 72 — the same trained network as the
GPU benchmark. Not a randomly initialised model.

**Host:** 8 cores, Python 3.14.6, torch 2.12.0 CPU, `torch.get_num_threads()` = 4.

---

## CPU root cause

**The pipeline was orchestration-bound, not compute-bound.** A worker-visible
inference round trip cost 43.0 ms, of which only ~5 ms was the neural network.
The other ~33 ms was the parent process waiting.

`SelfPlayWorkerPool.run()` alternated `_forward_inference()` with
`results.get(timeout=0.01)`. Forwarding was non-blocking and took 0.33 ms; the
loop then blocked up to 10 ms on the *episode-result* queue. Every inference
request and every reply crossed that loop, so each round trip absorbed the tick
twice. Measured directly: the parent looped at **93.9 ticks/s — a 10.2 ms
period, exactly the timeout value** — while 1.62 inference requests piled up per
tick.

This is Case A from the decision tree, resolving to its transport branch: the
round trip dominated the simulation, and inside it the parent bridge dominated.

The hypothesis stated in the task brief was correct, and is now measured rather
than assumed.

---

## Simulation breakdown

In-process, no IPC and no coordinator — the irreducible cost of the search
itself. 384 simulations across 12 moves.

| MCTS stage | mean | p50 | p90 | share |
|---|---|---|---|---|
| selection / traversal | 0.204 ms | 0.177 | 0.288 | 3.6% |
| state / request preparation | 0.320 ms | 0.282 | 0.446 | 5.7% |
| **CPU NN evaluation** | **4.777 ms** | 4.562 | 6.176 | **85.2%** |
| backup | 0.306 ms | 0.274 | 0.442 | 5.4% |
| **total simulation** | **5.607 ms** | 5.342 | 7.307 | 100% |

`apply_action` costs 0.120 ms per call and is already inside selection.

MCTS's own Python work is **0.83 ms per simulation**. It is not the problem, and
optimising the selection loop or game-state handling would have been wasted
effort. This is what made measuring first worthwhile.

---

## Inference breakdown

Three paths, same coordinator, same checkpoint, 4 concurrent clients. The
difference between rows one and two is exactly the multiprocessing bridge.

| Path | round trip p50 | evals/s |
|---|---|---|
| In-process, no multiprocessing | 9.6 ms | 408 |
| Through the worker pool bridge (before) | ~43.0 ms | 93 |
| Through the worker pool bridge (after) | ~20.4 ms | 196 |

Full-pipeline stage detail, 4 workers, 32 sims, `max_moves` 120:

| Inference stage | before | after |
|---|---|---|
| coordinator queue → dispatch p50 | 5.00 ms | 10.34 ms |
| coordinator queue → dispatch p90 | 21.52 ms | 13.66 ms |
| CPU NN evaluator p50 | 11.14 ms | 4.73 ms |
| CPU NN evaluator mean | 12.04 ms | 5.07 ms |
| response p50 | 16.17 ms | 14.80 ms |
| worker-visible total | 43.0 ms | 20.4 ms |

The evaluator p50 fell from 11.1 ms to 4.7 ms because batches shrank from 2.06
to 1.28 — the same per-request ~5 ms work, no longer bundled. Requests now
arrive promptly instead of accumulating behind the tick.

**Absolute milliseconds matter more than shares here.** ~15 ms of transport
remains per round trip. On CPU that sits beside ~5 ms of NN work; on the GPU,
where compute was measured at 8 ms, the same fixed overhead is proportionally
far more damaging.

---

## First optimization

One change only: `_InferenceBridge` in
[selfplay_workers.py](../../src/diamond/alphazero/orchestration/selfplay_workers.py).

Requests and responses are each pumped by a dedicated thread that **blocks on
queue arrival** rather than being drained between episode polls. The main loop's
`results.get()` no longer gates inference, so its timeout was relaxed to 50 ms —
it now only bounds liveness and deadline re-checks.

Preserved deliberately: correlation IDs, per-worker response routing, shutdown
semantics, error propagation (pump failures surface into the parent loop rather
than being swallowed), the per-game 900 s abort, and the catastrophic pool
timeout. No busy spinning, no unbounded queues. MCTS remains sequential — no
virtual loss, no tree parallelism, no speculative leaves.

---

## Before vs after

Identical checkpoint, workers, simulations, `max_moves`, bootstrap prior,
batching config and seeds.

| Metric | Before | After | Change |
|---|---|---|---|
| Wall clock | 236.6 s | 112.3 s | **2.11x** |
| sec/game | 29.58 | 14.03 | 2.11x |
| Games/hour | 121.7 | 256.5 | 2.11x |
| Completed games/hour | 106.5 | 224.5 | 2.11x |
| Samples/hour | 8,429 | 17,765 | 2.11x |
| Worker round trip | 43.0 ms | 20.4 ms | −52% |
| Evals/second | 93.0 | 196.1 | 2.11x |
| Batches/second | 45.1 | 153.0 | 3.4x |
| Mean batch size | 2.06 | 1.28 | — |

**Work performed is identical**, which is what makes this a speedup rather than
a shortcut: same 22,012 inference requests, same 554 samples, same 7/8 games
completed, same median 88 / p90 120 moves, same single
`max_game_moves_exceeded` abort.

### CPU concurrency comparison

| Workers | sec/game | queue→dispatch p50 | mean batch | round trip |
|---|---|---|---|---|
| 4 | 14.03 | 10.34 ms | 1.28 | 20.4 ms |
| 8 | 16.18 | 31.69 ms | 1.04 | 47.0 ms |

8 workers is **slower** than 4 on this 8-core box. Batch size falls rather than
rises, so this is CPU contention — 8 workers plus 4 torch threads per forward
oversubscribe the cores — not central coordination saturating. 4 workers is the
correct CPU operating point, and this ceiling is a property of this host, not of
the architecture; it should not be read as a prediction for the 30-vCPU GPU box.

---

## Correctness

- `tests/alphazero/orchestration/test_selfplay_workers.py` — 16 passed,
  including worker timeout, per-game abort, catastrophic pool timeout, request
  correlation, failure traceability and child cleanup.
- `tests/alphazero tests/agents` — **451 passed, 7 skipped** (CUDA-only),
  matching the pre-change baseline exactly.

Two tests were added ahead of the implementation: one asserting inference is
forwarded without waiting on the episode-result tick, one asserting an
uncorrelated response surfaces as a failure rather than being dropped. Both use
controlled queues and events, no wall-clock sleeps.

---

## Remaining bottlenecks

Ranked by measured cost per round trip:

1. **~15 ms of residual transport** (20.4 ms round trip − ~5 ms NN). Four
   pickle/IPC hops per evaluation remain: worker→parent, parent→coordinator,
   coordinator→parent, parent→worker. This is the next hardware-independent
   target and matters most on GPU.
2. **CPU NN forward, ~4.8 ms per request at batch ~1.3.** Genuine compute, and
   exactly what a GPU removes.
3. **Batch size ~1.3 against a configured 64.** Bounded by concurrent workers,
   not by `max_batch_size`. On CPU more workers make this worse, not better.
4. **`TorchEvaluator` per-row syncs** — ~4 device round trips per request
   (`legal.min()`, `legal.max()`, `probabilities.cpu()`, `values[row].cpu()`).
   Nearly free on CPU; real on CUDA. Not touched here, as this task excludes GPU
   optimisation.

---

## Is GPU still expected to help?

**Yes, and more than before — but the case is narrower than it looks.**

The NN forward is 4.8 ms of a now-20.4 ms round trip. GPU compute was measured
at ~8 ms per batch on the 3060, so per-request GPU work is well under the CPU's
4.8 ms once batching engages. GPU also removes the CPU contention that made 8
workers slower than 4, which is what unlocks the high worker counts the GPU box
allows.

The caution: **the ~15 ms of transport is hardware-independent and will follow
the code onto the GPU.** Before this fix it was ~33 ms, which is why the GPU
sat 83% idle behind a 97 ms round trip. Removing it on CPU is precisely the
work that should make the GPU worth renting. The remaining transport now caps
the achievable GPU speedup, so item 1 above is the highest-value next change —
and it is measurable locally, for free.

---

## Short RTX 3060 validation plan

Not executed. The GPU machine was not started for this investigation.

Purpose: confirm the ~23 ms removed on CPU also drops the worker-visible GPU
round trip from its measured 97 ms. 10–20 minutes is enough; the instrumentation
already exists, so no diagnosis time is bought at hourly rates.

```bash
# Seed from the trained checkpoint. A '[init] wrote initial checkpoint' line
# means the copy failed and the run is training a random network -- abort if so.
mkdir -p az-bench/soo/postfix-s32-w30
cp runtime/runs/soo/cpu8h-soo-20260819/latest.pt.cpu-backup \
   az-bench/soo/postfix-s32-w30/latest.pt

python tools/az_train.py --config runtime/configs/soo-rtx3060.json \
  --runtime-dir az-bench --run-id postfix-s32-w30 --migrate-device \
  --workers 30 --simulations 32 --train-steps-per-iteration 8 --hours 0.3

python tools/az_report.py az-bench/soo/postfix-s32-w30 \
  runtime/runs/soo/cpu8h-soo-20260819
```

Hold `max_moves` at 2000 and `max_wait_ms` at 2 to match GPU Run 1, the only
prior trained-model GPU point. Verify `[resume] loaded ... at training_step=72`.

Read three numbers:

| Number | GPU Run 1 | Expected |
|---|---|---|
| Worker round trip (`workers / evals_per_second`) | 97 ms | materially lower |
| Mean batch size | 1.30 | higher — requests now arrive promptly |
| Evals/second | 166 | higher |

Compare `samples_per_hour`, not `sec/game`, and check `abort_reasons` — a faster
`sec/game` bought by discarded games is not a speedup. If the round trip does
not fall, the remaining cost is in the four IPC hops rather than the parent
loop, which redirects the next optimisation.

---
---

# Round 2 — decomposing the residual 20.4 ms

Follow-up investigation into the ~15 ms sitting outside the neural network
after the `_InferenceBridge` fix. Same checkpoint
(`sha256:4b2a32ff…0ceb1320`, step 72), same 8-core host, `torch` threads 4.

**Outcome: no production change. Two candidate optimizations were implemented
and both measured slower than the current code on the real workload, so both
were reverted.** The measurements below are the deliverable.

## Pure multiprocessing transport floor

Real spawn processes, real `multiprocessing.Queue`, real `InferenceRequest` /
`InferenceResponse` objects with realistic 73x4 Soo features, real correlation
and routing. The parent echoes a valid response immediately — no NN, no
batching wait.

| | |
|---|---|
| Round trip p50 | **0.873 ms** |
| Round trip mean / p90 / p99 | 0.895 / 1.135 / 1.579 ms |
| Throughput | 3,398 echo-evals/s |
| Request payload | 3,308 bytes — encode 0.011 ms, decode 0.012 ms |
| Response payload | 1,784 bytes — encode 0.011 ms, decode 0.011 ms |

**This settles the IPC question. Process transport and pickle cost under 1 ms
of the round trip, not 15 ms.** The brief's own criterion — "if it measures
~4–6 ms, then the remaining 20 ms cannot reasonably be described as 15 ms of
IPC" — is met with room to spare. Serialization is not worth optimizing, and a
shared-memory redesign would target a cost that is not there.

## Residual latency breakdown

4 workers, 32 sims, `max_moves` 120, `max_wait_ms` 5. Worker round trip is
measured inside the child around `RemoteEvaluator.evaluate`; the rest come from
the coordinator's existing metrics.

| Component | mean | p50 | p90 |
|---|---|---|---|
| worker → parent → worker (both boundaries) | ~1.6 ms | — | — |
| coordinator queue → dispatch (batching wait) | — | 4.70 ms | 5.17 ms |
| CPU NN evaluator | 5.25 ms | 5.14 ms | — |
| **worker-visible total** | **11.77 ms** | **11.70 ms** | **13.52 ms** |

The two process boundaries are derived as worker round trip (11.77) minus the
coordinator's own submit→response span (10.15), and agree with the 0.87 ms
floor measured independently.

Note the worker-visible round trip here is 11.7 ms, better than the 20.4 ms
recorded in round 1 — that figure was derived as `workers / evals_per_second`,
which counts queueing under load rather than the latency one request sees.

## Batching wait experiment

Only `max_wait_ms` changed. Everything else identical, including seeds; all
rows produced the same 22,612 requests, 572 samples, 7/8 completed, and the
same single `max_game_moves_exceeded` abort.

| `max_wait_ms` | worker RT p50 | queue→dispatch p50 | mean batch | evals/s | samples/h | sec/game |
|---|---|---|---|---|---|---|
| 1 | 17.41 ms | 10.91 ms | 1.006 | 195.3 | 17,788 | 14.47 |
| 2 | 14.86 ms | 9.51 ms | 1.002 | 217.5 | 19,809 | 12.99 |
| **5** | **11.70 ms** | **4.70 ms** | **3.349** | **258.9** | **23,579** | **10.92** |
| 10 | 19.66 ms | 9.59 ms | 3.338 | 154.5 | 14,072 | 18.29 |
| 20 | 28.42 ms | 19.72 ms | 3.356 | 111.2 | 10,129 | 25.41 |

**A longer wait is faster, up to a point.** At 1–2 ms the batch never forms
(1.00) *and* queue→dispatch is ~10 ms — far more than the 1 ms budget, because
the coordinator wakes per request and its per-dispatch overhead contends with 4
workers plus torch threads. At 5 ms the batch reaches 3.35, so roughly 3x fewer
dispatches. Past 5 ms the batch stops growing — capped by 4 synchronously
blocked workers, not by `max_batch_size` (64) — and the wait is paid in full.

**5 ms is at or near the CPU optimum and the CPU reference config already uses
it.** There is no configuration win available here.

This is a CPU result at 4 workers. The RTX 3060 host has ~30 workers, so its
concurrent request set is ~7x larger and its optimum will differ. Do not port
this value.

## CPU NN batch scaling

Isolated, no coordinator:

| batch | total | per request | vs batch 1 |
|---|---|---|---|
| 1 | 2.37 ms | 2.37 ms | 1.00x |
| 2 | 3.47 ms | 1.74 ms | 1.36x |
| 4 | 5.55 ms | 1.39 ms | 1.70x |
| 8 | 8.17 ms | 1.02 ms | 2.32x |
| 16 | 16.18 ms | 1.01 ms | 2.34x |

Batching is genuinely worth having on CPU, and it is already working. Note
batch-1 costs 2.37 ms here versus ~5 ms in the live pipeline: the difference is
contention, not the model.

## Root cause

**The residual is batching wait plus CPU contention, not IPC**: at the 5 ms
operating point the round trip is ~4.7 ms batching wait + ~5.1 ms NN + ~1.6 ms
transport, and process transport alone has a measured floor of 0.87 ms.

## Two attempted optimizations, both reverted

Written TDD, each benchmarked against the identical workload.

**Attempt 1 — flush an underfull batch when the request queue is empty.**
Motivated by a clean isolated measurement: one in-process client with no
contention still paid 5.15 ms of queue→dispatch on a batch that could never
grow. Fixing that case cut its round trip 7.60 → 3.40 ms.

On the real pipeline it was *slower*: 93.4 s vs 87.3 s, samples/h 22,038 vs
23,579, because mean batch collapsed 3.35 → 1.05. With 4 workers the queue is
briefly empty *between* arrivals, so this flushed constantly and destroyed the
batching worth 1.7x per request.

**Attempt 2 — flush only when every outstanding request is already batched.**
Tracks `_outstanding` so a momentary gap is distinguished from "nothing more can
arrive." In-process this was the best result measured: single client 7.60 →
2.89 ms, four clients 10.46 → 8.74 ms with batch preserved and evals/s 386 →
455.

On the real pipeline it was worse still: 135.8 s, batch 1.00. The
`_state_lock` acquisition it needs sits on the batching thread's hot path and
contends with `submit()` and `_complete()`, which run on the bridge threads for
every request.

Both were reverted. The coordinator is byte-identical to `main`.

**What this rules out:** the dead batching wait is real and measurable in
isolation, but on this 4-worker CPU host it cannot be reclaimed without losing
more to batch collapse or lock contention than it saves. The 5 ms wait is
already the best of the three variants measured.

## Correctness

`tests/alphazero tests/agents tests/tools` — **474 passed, 7 skipped**
(CUDA-only). Production code unchanged from `main`.

## What not to optimize

- **MCTS** — 0.83 ms per simulation of Python work. Unchanged since round 1,
  and search semantics remain strictly sequential.
- **Serialization / shared memory** — now measured at 0.87 ms round trip and
  ~0.01 ms per pickle. There is nothing to reclaim.
- **`max_wait_ms` on CPU** — already at its measured optimum.
- **GPU-specific work** (BF16, CUDA graphs, `TorchEvaluator` per-row syncs) —
  still out of scope and still unjustified from CPU evidence.

## The real remaining constraint

With one synchronous leaf per worker, concurrent requests are capped at the
worker count, so batch cannot exceed ~4 on this host regardless of
configuration. Every remaining lever — larger batches, better GPU duty cycle —
is bounded by that. Raising it means either more workers (which this 8-core box
cannot absorb: 8 workers measured slower than 4) or multiple outstanding leaves
per tree, which is explicitly prohibited as it changes search semantics.

**That makes the GPU host the right next step rather than further CPU work.**
Its ~30 workers lift the concurrency cap ~7x, which is exactly the constraint
CPU cannot relieve.

## Next RTX 3060 experiment

Unchanged from the recipe in the round-1 section above, with one addition: the
`max_wait_ms` curve should be re-measured on the GPU host, because its optimum
is set by concurrency and 30 workers is a different regime from 4. Run the
baseline first, then two points either side:

```bash
# Baseline at the current GPU config (max_wait_ms 2, as in soo-rtx3060.json)
python tools/az_train.py --config runtime/configs/soo-rtx3060.json \
  --runtime-dir az-bench --run-id postfix-s32-w30 --migrate-device \
  --workers 30 --simulations 32 --train-steps-per-iteration 8 --hours 0.3
```

Then repeat with `max_wait_ms` 5 and 10 patched into a copy of the config,
holding `max_moves` at 2000 and everything else fixed. Read mean batch size and
`samples_per_hour`; on CPU the batch stopped growing once it hit the worker
count, and the same test on GPU says whether 30 workers is enough concurrency to
make batching pay there.
