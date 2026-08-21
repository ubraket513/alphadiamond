# RTX 5060 Ti Parent-Bottleneck Investigation

Follows [rtx5060_scaling_findings.md](rtx5060_scaling_findings.md), which
established that the parent process saturates one core while the GPU idles at
~5 %, and estimated ~1.5 ms of serialized cost per evaluation without
attributing it.

This pass attributes that cost, tests the cheap `max_wait_ms` hypothesis, and
lands one measured optimization.

**Headline:** the dominant serialized cost is **CPU kernel-launch overhead in the
model forward** — not routing, not feature construction, and not GPU execution.
The forward costs the same for batch 1 as for batch 256, and the GPU drains in
22 µs after 16 queued forwards. Vectorizing the trunk's directional loop halved
it and bought **+8.5 % samples/hour**, measured.

---

## 1. Inference call / data flow

```
WORKER PROCESS  (parallel — off the parent's critical path)
 1  DiamondSearchAdapter.evaluation_request      game_adapter.py:128
      └ CanonicalEncoder.encode → 73x4 node_features   ← feature extraction is HERE
 2  RemoteEvaluator.evaluate                     remote.py:48
      └ InferenceRequest.from_eval_request       protocol.py:159   (validates 292 floats)
 3  _ProcessRequestCoordinator.submit            selfplay_workers.py:316
      └ mp request_queue.put → worker-side feeder thread pickles

PARENT PROCESS  (serialized — the bottleneck)
 4  _InferenceBridge._pump_requests              selfplay_workers.py:459  [bridge thread]
      └ mp Queue.get → unpickles 292 floats + 73 tuples
 5  InferenceCoordinator.submit                  coordinator.py:220
 6  InferenceCoordinator._run                    coordinator.py:250       [coordinator thread]
 7  _validated_request                           coordinator.py:297
      └ REBUILDS the request, re-validating all 292 feature floats
 8  _flush → InferenceModelPool.evaluate         model_pool.py:68
 9  TorchEvaluator.evaluate                      torch.py:38
      ├ torch.tensor(nested tuples)              torch.py:51    H2D
      ├ model(features)                          ← ~200 kernel launches
      ├ gather + masked_fill + softmax           torch.py:99
      ├ isfinite/sum scalar read                 torch.py:107   first device sync
      ├ .cpu().tolist() x2                       torch.py:112   D2H
      └ per-row priors dict build                torch.py:119
10  InferenceResponse.from_eval_result           protocol.py:261  re-validates ~54 prior pairs
11  _complete → reply_queue.put                  coordinator.py:405
12  _InferenceBridge._pump_responses             selfplay_workers.py:476  [bridge thread]
      └ response_queues[worker_id].put → ONE FEEDER THREAD PER LANE pickles
13  _ProcessRequestCoordinator._pump_responses   selfplay_workers.py:341  [worker thread]
14  RemoteEvaluator._replies.get                 remote.py:74
```

Two facts fall straight out of the map:

- **Feature extraction is already in the worker.** It is not a parent cost, which
  removes most of "Case B" before any measurement.
- **One `multiprocessing.Queue` feeder thread per lane.** Confirmed by thread
  counts: `parent_threads = lanes + 50` fits every point (30→80, 36→86, 48→98,
  64→114).

---

## 2. Parent-side profile

`az-bench/profiles/bench_parent_cycle.py` replays each stage of the parent's
cycle on **real encoder output** (73x4 features, 18–94 legal actions), isolated
from the running system so it perturbs nothing. Batch 12, 200 measured batches
after 20 warm-ups.

| stage | per_batch_ms | per_eval_ms | share |
|---|---|---|---|
| **6_forward_launch** | **6.924** | **0.5770** | **68.7 %** |
| 3_validated_request | 1.091 | 0.0909 | 10.8 % |
| 11_response_envelope | 0.418 | 0.0348 | 4.1 % |
| 5_tensor_construction | 0.362 | 0.0301 | 3.6 % |
| 7_policy_gather_softmax | 0.354 | 0.0295 | 3.5 % |
| 1_unpickle_request | 0.280 | 0.0233 | 2.8 % |
| 8_validity_sync | 0.216 | 0.0180 | 2.1 % |
| 12_pickle_response | 0.204 | 0.0170 | 2.0 % |
| 10_priors_dict_build | 0.092 | 0.0076 | 0.9 % |
| 9_d2h_tolist | 0.080 | 0.0067 | 0.8 % |
| 2_submit_bookkeeping | 0.023 | 0.0019 | 0.2 % |
| 13_metrics_record | 0.018 | 0.0015 | 0.2 % |
| 4_pool_grouping | 0.017 | 0.0014 | 0.2 % |
| **TOTAL** | **10.078** | **0.8398** | 100 % |

The isolated cycle is 0.84 ms/eval against ~1.5 ms observed in the real system.
The residual is analysed in §4.

### The forward is CPU-launch-bound, not GPU-bound

CUDA timing was taken two ways: without synchronisation (CPU launch cost, which
is what the parent thread actually pays) and with CUDA events (device span). The
production path does **not** synchronise after the forward, so the sync is
confined to the benchmark.

```
 batch    cpu_ms  device_ms  cpu/device  per_row_us
     1     7.165      7.165        1.00      7165.4
    12     6.919      6.920        1.00       576.6
    32     6.666      6.666        1.00       208.3
   128     6.605      6.605        1.00        51.6
   256     6.701      6.966        0.96        27.2
```

**Batch 256 costs the same as batch 1**, and `cpu/device` is 1.00 everywhere —
the CUDA-event span is almost entirely idle gaps between launches, not work.

```
depth   1: cpu submit    6.855 ms   drain 0.021 ms
depth  16: cpu submit  110.390 ms   drain 0.022 ms
```

After queueing 16 forwards the GPU drains in **22 microseconds**. The CPU cannot
run ahead; every forward costs ~6.9 ms of pure CPU dispatch.

**Mechanism.** `DirectionalResidualBlock.forward` looped over the 6 board
directions, issuing an index, a matmul, a `Linear` and an add each — ~24
operations per block, ~144 across the 6-block trunk, ~200 for the whole model
(47 `Linear` modules). At ~34 µs of Python/ATen dispatch per operation on a
2.3 GHz Broadwell core, that is the 6.9 ms.

This explains every earlier observation at once: 5 % GPU utilisation, 24 W of a
180 W cap, why worker count stopped helping, and why the *faster-CPU* RTX 3060
host reached 900 evals/s on identical code.

---

## 3. `max_wait_ms` sweep

48 workers, 48 games, 12 train steps, 64 simulations, `max_batch_size` 32, same
checkpoint, seed and prior. Pre-optimization code.

| wait | wall | samp/h | eval/s | bat/s | meanB | p90B | maxB | cycle_ms | **ms/eval** | qd50 | rsp50 | GPU% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 506.1 | 31,370 | 557 | 72.7 | 7.65 | 11 | 13 | 13.75 | 1.796 | 43.44 | 56.41 | 5.3 |
| **2** | **429.7** | **36,943** | **656** | 56.2 | 11.67 | 18 | 23 | 17.80 | **1.525** | 31.37 | 49.95 | 5.0 |
| 4 | 431.1 | 36,824 | 654 | 43.2 | 15.11 | 26 | 32 | 23.12 | **1.530** | 19.32 | 38.76 | 3.9 |
| 8 | 485.5 | 32,700 | 580 | 36.8 | 15.79 | 32 | 32 | 27.20 | 1.723 | 21.47 | 38.33 | 3.4 |

**The shipped `max_wait_ms = 2` is correct and was left alone.**

The interesting column is `ms/eval`: **1.525 at wait 2 versus 1.530 at wait 4 —
identical.** Mean batch grows 11.67 → 15.11 (+29 %) and throughput does not move,
because batch size grows in proportion to cycle time and throughput is their
ratio. `max_wait_ms` **cannot** raise throughput in this regime; it only moves
where time is spent. It does buy latency — wait 4 cuts response p50 from 50.0 to
38.8 ms at equal throughput — which is worth knowing but is not the objective.

Wait 1 loses because batches get too small to amortize the fixed forward cost.
Wait 8 loses because `max_batch_size` 32 becomes fully binding (p90 batch = 32),
so the extra window is pure dead time.

---

## 4. Diagnosis of the dominant serialized cost

Decomposing the *real* cycle against the isolated stage model
(`cycle = fixed + per_eval x B + wait + residual`):

| run | lanes | wait | meanB | real_cycle | model | residual | **residual/eval** |
|---|---|---|---|---|---|---|---|
| W-w48-wait1 | 48 | 1 | 7.65 | 13.75 | 10.01 | 3.73 | 0.488 |
| A-w30-g32 | 30 | 2 | 7.85 | 15.40 | 11.06 | 4.34 | 0.552 |
| B-w36-g36 | 36 | 2 | 10.51 | 17.50 | 11.71 | 5.78 | 0.550 |
| C-w48-g48 | 48 | 2 | 11.63 | 18.17 | 11.99 | 6.18 | 0.532 |
| D-w64-g64 | 64 | 2 | 12.83 | 19.28 | 12.28 | 6.99 | 0.545 |
| W-w48-wait4 | 48 | 4 | 15.11 | 23.12 | 14.84 | 8.28 | 0.548 |

**The residual is flat at 0.49–0.55 ms/eval from 30 to 64 lanes** — an 11 % range
with no trend, while lane count more than doubles.

This is the load-bearing result for the architecture decision. If the residual
were GIL contention from the per-lane feeder threads, it would scale with lane
count: 64 lanes carry 2.1x the feeder threads of 30. It does not move at all.
The residual is therefore **fixed per-request IPC and lock cost** — pipe
syscalls, `_state_lock`/`_metrics_lock` acquisitions, queue signalling —
consistent with the earlier CPU pass measuring process transport at ~0.87 ms
round trip.

Per-evaluation budget at 48 workers / wait 2 (1.525 ms/eval):

| component | ms/eval | share |
|---|---|---|
| forward launch (amortized over batch) | 0.612 | 40.1 % |
| fixed per-request IPC / lock | 0.497 | 32.6 % |
| isolated per-evaluation Python | 0.245 | 16.1 % |
| `max_wait` window (amortized) | 0.171 | 11.2 % |

### What this says about a dedicated inference process

The leading hypothesis was to move the coordinator into its own process. The
measurements argue it would **move the bottleneck rather than remove it**:

- The 0.497 ms/eval residual is *per request*, not per thread. A separate process
  performs the same pipe reads, the same lock acquisitions, and the same
  per-worker `Queue.put` — it carries that cost with it.
- It would inherit the same one-feeder-thread-per-lane response fan-out.
- The forward's 6.9 ms launch cost is a property of the model and the CPU, not of
  which process runs it.

The parent would be freed (useful for overlapping training), but the inference
data plane would still be one serialized Python thread with the same cycle. This
is exactly the "remove parent routing" versus "move the bottleneck" distinction,
and the evidence lands on the second.

---

## 5. The change that shipped

### Files changed

| file | change |
|---|---|
| `src/diamond/alphazero/network/trunk.py` | `DirectionalResidualBlock.forward` vectorized; ~20 lines |
| `tests/alphazero/test_network.py` | +3 tests (5 cases) pinning parity, `state_dict`, gradients |

### What it does

The directional message passing computes

```
message = self_projection(nodes) + sum_d (adjacency[d] @ nodes) @ W_d^T
```

The sum over directions is a contraction, so it is expressed as two einsums
instead of a Python loop:

```python
neighbours = torch.einsum("dij,bjw->bdiw", adjacency, nodes)
weights = torch.stack([projection.weight for projection in self.direction_projections])
message = self.self_projection(nodes) + torch.einsum("bdiw,dvw->biv", neighbours, weights)
```

The second einsum contracts the direction axis `d` and the channel axis `w`
together, fusing the projection and the reduction into one call.

This is a **reassociation, not an approximation**. The parameters are untouched —
the same `nn.Linear` modules in the same `ModuleList` — so the `state_dict` and
every existing checkpoint stay byte-compatible. No change to simulation count,
search semantics, prior, training ratio, or termination policy.

### Correctness

| check | result |
|---|---|
| Block vs. per-direction loop (batch 1/5/17) | max abs diff **1.8e-06** |
| Trained checkpoint, 24 real positions — values | max abs diff **6.0e-08** |
| Trained checkpoint, 24 real positions — priors | max abs diff **3.9e-07** |
| Legal action sets | identical |
| `state_dict` keys | unchanged |
| Gradients reach every direction's weight | yes, all finite |
| `pytest tests/alphazero tests/agents tests/tools` | **488 passed** (483 baseline + 5 new) |

The loop is retained inside the test as an independent oracle, so the
equivalence claim stays pinned rather than asserted.

### Forward cost after the change

```
 batch    cpu_ms  device_ms  cpu/device
     1     3.317      3.318        1.00
    12     3.529      3.529        1.00
    32     3.481      3.481        1.00
   128     3.347      3.729        0.90   ← crossover to GPU-bound moved here
```

**6.92 → 3.53 ms/batch, −49 %.** Still launch-bound at production batch sizes
(drain remains 22 µs), so this halves the dominant term without leaving the
regime.

---

## 6. Before / after benchmark

48 workers, 48 games, 12 train steps, 64 simulations, same immutable step-80
checkpoint, same seed, same prior, FP32. Source pinned per run and asserted via
`diamond.__file__`.

| run | code | wait | wall | samp/h | eval/s | bat/s | meanB | ms/eval | qd50 | rsp50 | GPU% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W-w48-wait2 | before | 2 | 429.7 | 36,943 | 656 | 56.2 | 11.67 | 1.525 | 31.37 | 49.95 | 5.0 |
| W-w48-wait4 | before | 4 | 431.1 | 36,824 | 654 | 43.2 | 15.11 | 1.530 | 19.32 | 38.76 | 3.9 |
| **V-w48-wait2** | **after** | **2** | **396.2** | **40,072** | **711** | 66.4 | 10.71 | **1.406** | 26.61 | 41.47 | 3.3 |
| V-w48-wait4 | after | 4 | 409.5 | 38,772 | 688 | 46.4 | 14.83 | 1.453 | 17.23 | 36.07 | 2.7 |

**+8.5 % samples/hour** (36,943 → 40,072), self-play wall −7.8 %, response p50
−17.0 %, inference p50 −22.4 %. All rows 48/48 games completed, zero aborts,
identical game distribution (4,410 samples, median 89 moves).

`train_s` for 12 optimizer steps went 1.81 → 1.71 s, so the larger `[B, 6, 73, W]`
intermediate did **not** cost anything on the training side at batch 256.

The optimum also shifted: wait 2 and wait 4 were tied before (1.525 vs 1.530
ms/eval) and wait 2 is now clearly ahead (1.406 vs 1.453), which is the expected
consequence of a smaller fixed cost needing less amortization.

### Why +8.5 % and not the modelled +19 %

Mean batch fell 11.67 → 10.71. A faster cycle accumulates fewer requests per
batch, so part of the saving is handed back as worse amortization:

```
before: 1.525 ms/eval | fwd 0.612  py 0.245  wait 0.171  resid 0.497
after : 1.406 ms/eval | fwd 0.350  py 0.245  wait 0.187  resid 0.624
```

The forward term fell as predicted (0.612 → 0.350) but the residual rose
(0.497 → 0.624), because `batches/s` rose 18 % and the fixed per-request cost is
paid more often.

**This is the same structural trap as `max_wait_ms`:** throughput is `batch /
cycle` and batch tracks cycle, so single-thread optimizations are partly
self-limiting. It is a real gain, but it bounds how much this family of change
can ever deliver.

---

## 7. Was parent saturation eliminated, or moved?

**Neither — it is unchanged.** Parent CPU is 103 % before and 103 % after
(−0.2 %). The parent still saturates exactly one core; it now does more useful
work per second on it (`batches/s` 56.2 → 66.4).

GPU utilisation *fell*, 5.0 % → 3.3 %, while throughput rose. On this workload
GPU utilisation and throughput are uncorrelated, and optimizing the former would
have been actively misleading.

---

## 8. The system-level picture, and the new bottleneck

Per-evaluation CPU across the whole process tree:

| run | lanes | eval/s | parent % | worker % | worker ms/eval | parent ms/eval |
|---|---|---|---|---|---|---|
| A-w30-g32 | 30 | 510 | 98 | 238 | 4.66 | 1.91 |
| C-w48-g48 | 48 | 640 | 102 | 336 | 5.25 | 1.60 |
| D-w64-g64 | 64 | 666 | 104 | 358 | 5.38 | 1.56 |

Total system CPU is ~6.7 ms per evaluation, of which **~5.3 ms (79 %) is
worker-side MCTS** and ~1.4 ms (21 %) is the parent. The parent is the bottleneck
only because it is **serialized**, not because it is expensive. 18 physical
cores exist; ~4.4 are in use.

**Next bottleneck: the single serialized inference data plane.** Response p50 is
41.5 ms, of which **26.6 ms (64 %) is queue→dispatch** — time spent queueing in
front of one coordinator, not doing work.

The lever that addresses it is **sharding the data plane across K inference
processes**, which is materially different from the single dedicated inference
process considered in §4: one process inherits the serialization, K processes
divide it. The GPU uses 924 MB of 16 GB and sits at 3.3 %, so several model
replicas are affordable in both memory and compute.

Ceiling estimate: 18 physical cores ÷ 6.7 ms/eval ≈ **2,700 evals/s ≈ 150,000
samples/hour**, roughly **4x** the current 711 evals/s. At that rate the GPU
would be at ~12 %. **The GPU never becomes the constraint on this host**; the
endpoint is CPU-bound (a healthy Case D), at which point worker-side MCTS — 79 %
of system CPU per evaluation and never yet profiled — becomes the target.

Explicitly *not* levers on this evidence: a larger GPU, BF16, `max_batch_size`
(the cap is not binding at wait 2), or more workers. CUDA graphs would now
address only ~25 % of the cycle, and the varying batch size (7–32) makes static
capture awkward.

---

## 9. Reproducing

```bash
export BENCH_SRC=/path/to/pinned/src

# stage decomposition of the parent cycle
python az-bench/profiles/bench_parent_cycle.py \
  --checkpoint runtime/runs/soo/cpu8h-soo-20260819/latest.pt --batch 12 --batches 200

# CPU launch vs GPU execution
python az-bench/profiles/bench_forward_launch.py \
  --checkpoint runtime/runs/soo/cpu8h-soo-20260819/latest.pt

# prototype + parity for the vectorization
python az-bench/profiles/bench_block_vectorization.py

# one end-to-end point
./az-bench/profiles/run_point.sh V-w48-wait2 \
  az-bench/configs/soo-rtx5060-w48-wait2.json 48 12

python az-bench/profiles/summarize_scaling.py az-bench/soo/{W,V}-w48-wait2
```

Ledgers, per-run configs and 2 s sampler CSVs are under `az-bench/soo/{W,V}-*`.
