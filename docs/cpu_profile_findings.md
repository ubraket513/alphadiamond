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
[selfplay_workers.py](../src/diamond/alphazero/orchestration/selfplay_workers.py).

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
