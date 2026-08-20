# RTX 3060 Benchmark Findings

Measurements from continuing the Soo run `cpu8h-soo-20260819` on a rented
vast.ai RTX 3060 (12 GB, 32 vCPU). Records what was actually observed, what it
means, and what is still unverified.

**Status:** early. Three GPU iterations measured, only one of them against the
trained network. Treat the throughput numbers as directional, not settled.

---

## Headline

The GPU is **latency-bound, not compute-bound**. At the best measured setting it
runs ~2.9x the CPU baseline while leaving the GPU idle roughly 83% of the time.
The remaining headroom is real but needs a code change, not a config change.

---

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX 3060, 12 GB |
| vCPU | 32 (resolved to 30 self-play workers: available minus two) |
| Host | vast.ai container, `/workspace` persistent volume |
| Python | 3.11 (image ships 3.10.12; project requires >= 3.11) |
| Precision | FP32 throughout; no BF16, no mixed precision |
| Run | `cpu8h-soo-20260819`, continued in place from `training_step` 72 |

The source checkpoint is `sha256:4b2a32ff1517…0ceb1320`, step 72, 26,027 replay
samples, 18 CPU iterations, 0 aborts. Its pre-migration copy is preserved at
`latest.pt.cpu-backup`.

---

## Baseline: CPU

Aggregated across all 18 iterations of the original run.

| Metric | Value |
|---|---|
| Games | 288 / 288 completed |
| Workers | 4 |
| Simulations | 32 |
| `max_moves` | 2000 |
| `max_wait_ms` | 5 |
| **sec/game** | **24.9** |
| Completed games/hour | 144.4 |
| Samples/hour | 13,046 |
| Median moves | 87 |
| Aborts | none |

---

## GPU runs

### Run 1 — trained model, 32 sims, `max_wait_ms: 2`, `max_moves: 2000`

The first real GPU iteration, continuing from step 72.

| Metric | Value |
|---|---|
| Games | 30 / 32 completed |
| Aborts | 2x `max_game_time_exceeded` |
| sec/game | 28.2 (30.1 per completed game) |
| Self-play wall | 902 s |
| Median / p90 moves | 85 / 96 |
| **Mean batch size** | **1.30** (configured max 32) |
| Max batch size | 13 |
| Inference p50 | 3.06 ms |
| Evals/second | 166 |
| Requests | 149,746 |

**Slower per game than the 4-worker CPU baseline.** Mean batch size of 1.30
against a configured 32 shows the coordinator was dispatching essentially one
request at a time.

The two aborts are the per-game deadline working as designed: two pathological
games hit 900 s, contributed zero samples, and **30 sibling games survived and
trained**. Before this work that iteration would have failed entirely. They were
also expensive — they gated the whole iteration's 902 s wall-clock.

### Run 2 — untrained network, 32 sims, `max_wait_ms: 20`, `max_moves: 200`

Seeded accidentally from a fresh random network (`[init] wrote initial
checkpoint`), so **not comparable to Run 1 on game distribution**. Still the
best throughput measured.

| Metric | Value |
|---|---|
| Games | 32 / 32 completed |
| Workers | 30 |
| sec/game | **8.5** |
| Completed games/hour | 423.7 |
| Samples/hour | 34,279 |
| p90 moves | 93 |
| **Mean batch size** | **14.67** |
| Inference p50 | 8.06 ms |
| Evals/second | 309 |
| Aborts | none |

---

## Where the time goes

Decomposition from Run 2's reported numbers. The model predicts 262 s of
self-play against 272 s observed, so the accounting is trustworthy.

```
evals/s reported          309
mean batch size          14.67
=> batches/s              21.1
inference p50 per batch   8.06 ms
=> GPU busy               17% of wall-clock

per worker:
  evals/s                 10.3
  => round-trip latency   97 ms   (one leaf at a time, synchronous)
     of which GPU compute  8 ms
     of which batch wait  ~20 ms  (max_wait_ms, deliberate)
     of which overhead    ~69 ms  (IPC + polling)
```

Throughput is `workers / round-trip latency` = `30 / 0.097s` ~= 309 evals/s,
matching the measurement exactly. Each worker evaluates one leaf, blocks for the
answer, and repeats 32 times per move across ~79 moves per game.

**The GPU is idle 83% of the time.** Nothing is asking it for more work.

### The ~69 ms of overhead, in likely order

1. **The parent's single-threaded forwarding loop.**
   `SelfPlayWorkerPool._forward_inference` drains the request queue, submits to
   the coordinator, then drains replies — one thread, polling
   `results.get(timeout=0.01)`. Every request and reply for all 30 workers
   funnels through it, and the 10 ms poll tick alone is significant.
2. **Multiprocessing queue hops.** Each eval crosses four pickle/IPC
   boundaries: worker→parent, parent→coordinator, coordinator→parent,
   parent→worker.
3. **CPU contention.** 30 workers on 32 cores plus the parent's poll thread.
4. **`TorchEvaluator` per-row GPU→CPU syncs.** Roughly four device syncs per
   request (`int(legal.min())`, `int(legal.max())`, `probabilities.cpu()`,
   `values[row].cpu()`), so ~60 serialized syncs inside a batch of 15. Real, but
   only ~8% of the problem at this batch size.

---

## What the data settled

**`max_wait_ms` is the batching dial, and it works.** 2 ms gave batch 1.30;
20 ms gave 14.67. An earlier prediction that raising it would not help was
wrong — it was made without knowing the value had already been raised.

The tradeoff is explicit: at 20 ms you pay ~20% of round-trip latency to get
~11x the batch size. Probably worth it, but only two points on the curve are
known.

**Batch size is bounded by concurrent workers, not by `max_batch_size`.** A CPU
run with 4 workers measured mean batch 3.3; 30 workers measured 14.67. The
configured maximum of 32 was never the constraint. This was predicted before any
GPU run and is confirmed.

**The bottleneck is not the GPU.** 17% duty cycle, 8 ms compute inside a 97 ms
round trip.

---

## Open questions

### `max_moves: 200` may be silently discarding training data

The move cap was lowered from 2000 to 200 on the box. An aborted game
contributes **zero training samples**, so a tighter cap buys seconds-per-game by
throwing away data.

Run 2 saw no move-cap aborts, but it used a *random* network with p90 93 moves.
The **trained** network historically showed p90 **1667** at 32 sims. Against that
distribution, a 200-move cap could abort a substantial fraction of games and it
would look like excellent `s/game` alongside quietly reduced `samples/hour`.

**Check `abort_reasons` for `max_game_moves_exceeded` on any trained-model run,
and compare `samp/h` rather than `s/game`.** If aborts climb, 400–600 likely
truncates the pathological tail without discarding ordinary long games.

### Does concurrency convert idle GPU into throughput?

Throughput is `workers / latency`, workers are blocked on IPC ~92% of the time,
and the GPU is 83% idle. Oversubscribing beyond 30 workers on 32 cores may pay
precisely because the workers are not compute-bound. Untested.

### Is 64 simulations cheaper in wall-clock than 32?

Historically 64 sims produced far more stable games (p90 147 vs 1667) despite
doing twice the search. Run 1 lost 1800 s to two 900 s timeouts at 32 sims. If
64 sims avoids that tail it may win on wall-clock despite the extra work. This
is a question about game quality, not raw speed. Untested.

---

## Suggested next measurements

All seeded from `latest.pt.cpu-backup` so the trained network is under test, and
all sharing one `max_moves` and `max_wait_ms` so rows stay comparable.

| Run | Purpose |
|---|---|
| `s32-w30` | Trained-model baseline at current settings |
| `s32-w60` | Does oversubscription convert idle GPU into throughput? |
| `s64-w30` | Does more search avoid the pathological move tail? |
| `s32-w30-wait10` / `wait40` | Shape of the batching-vs-latency curve |

```bash
python tools/az_report.py az-bench/soo/* runtime/runs/soo/cpu8h-soo-20260819
```

`az_report.py` shows `mvcap` and `wait` per run and warns when compared runs
disagree on a knob that changes what the comparison means.

---

## Recommendation

At ~2.9x the CPU baseline, this is a reasonable place to **start training rather
than keep tuning**. The remaining ~5x of theoretical headroom needs the parent's
forwarding loop restructured into something asynchronous — a genuine redesign,
not a tweak, and worth scoping separately if throughput becomes the binding
constraint.

Two caveats before trusting the 2.9x figure:

- Run 2 used a **random network**, not the trained one. Re-run it seeded from
  `latest.pt.cpu-backup` for a like-for-like number.
- Run 2 used `max_moves: 200` against the CPU baseline's 2000. Until the abort
  counts on a trained-model run are checked, part of that speedup may be
  discarded games rather than faster ones.

---

## Reproducing

Environment setup, migration and recovery are in
[gpu_training.md](gpu_training.md). Commands used here:

```bash
mkdir -p az-bench/soo/<name>
cp runtime/runs/soo/cpu8h-soo-20260819/latest.pt.cpu-backup \
   az-bench/soo/<name>/latest.pt

python tools/az_train.py --config runtime/configs/soo-rtx3060.json \
  --runtime-dir az-bench --run-id <name> --migrate-device \
  --workers <n> --simulations <n> --train-steps-per-iteration 8 --hours 0.4

python tools/az_report.py az-bench/soo/<name>
```

Verify each run starts with `[resume] loaded ... at training_step=72`. A
`[init] wrote initial checkpoint` line means the copy failed and the run is
training a random network — that is what happened in Run 2.
