# Native Soo Self-Play — Handoff

**Start here** if you are picking this work up in a new session, especially on
the GPU training host.

Everything below is current as of the Gate D GPU-host pass. All four gates are
closed, including Gate D's throughput half, which was run on the RTX 5090
training host — see §10 of the progress document for the A/B table.

---

## 1. What this project is

Moving the Soo self-play/search hot path out of Python multiprocessing into a
native multithreaded subsystem, while training, replay, checkpoints and the
PyTorch model stay in Python.

Read in this order:

| document | what it is |
|---|---|
| [native_selfplay_phase0.md](native_selfplay_phase0.md) | **The contract.** Measurements, proposed shapes, the four gates, the risk list. A design record — *not* rewritten as work lands. |
| [native_selfplay_phase1_progress.md](native_selfplay_phase1_progress.md) | **What actually happened.** Per-gate results, every number measured, every bug found and why it was possible. |
| this file | Where to resume, how to run things, what must not break. |
| [rtx5060_bottleneck_findings.md](rtx5060_bottleneck_findings.md) | The GPU-host measurements Phase 0 rests on. |

---

## 2. Gate status

| gate | question | status |
|---|---|---|
| **A** | rules / encoding / prior parity | **pass** — 1,327-position corpus, exact |
| **B** | deterministic MCTS parity | **pass** — bit-identical q, identical request sequence |
| **C** | native + batcher throughput, no Python | **pass** — 8.2x–18x the Python ceiling at measured value-only latencies |
| **D** | end-to-end with PyTorch | **pass** — 18.5x the Python backend at cap 32, same host, same seeded work |
| **E** | stochastic MCTS: Dirichlet + temperature | **pass** — distribution parity with Python, deterministic per seed, 16 lanes → 16 games |
| **F** | native backend behind the pool contract | **pass** — episode parity with `SooSelfPlayRunner`, 18.5x end to end, trains from scratch |

Merged: #2 (A), #3 (B), #4 (gui CI fix), #5 (C), #6 (D-on-CPU).

---

## 3. What is left

### 3.0 Done on the GPU host (progress doc §10)

1. **The A/B throughput table.** Done. **18.5x** at the production cap of 32,
   against a Python baseline re-tuned on the same host (3,573 evals/s at 32
   workers, *not* the 641 in `rtx5060_bottleneck_findings.md` — that is a
   different, much weaker machine and none of its absolutes transfer).
2. **Risk 5.** Answered. The evaluator thread *is* the ceiling — 98.8–99.6 %
   busy in all 20 configurations measured — but 86 % of that is the forward
   itself and only ~1 % is the boundary. The ceiling is the GPU, which is where
   the design wanted it.

### 3.1 The actual remaining work, in order

1. ~~**Stochastic MCTS in the native backend.**~~ **Done — Gate E**, §11 of the
   progress doc. Dirichlet noise at the root and temperature sampling of moves,
   gated on distribution rather than on stream, per §9's RNG policy. 16 lanes
   now play 16 distinct games where they played 1.
2. ~~**`selfplay_backend = "native"`.**~~ **Done — Gate F**, §12. `az_train.py
   --selfplay-backend native`; `"python"` remains the default and the oracle.
3. ~~**A training loop consuming native self-play samples**~~ **Done** — §12.5.

### 3.1.0 Configuring the native backend

**`games_per_iteration` must exceed `native_lanes`**, and by a good margin.
Lanes are a fixed pool and surplus jobs are queued; with one lane per job a
handful of long games hold the whole run at batch 1 for the second half of its
dispatches (§12.2 — it cost 3x before it was found). Measured operating points:
`384 jobs / 128 lanes / cap 64 / 12 threads` gives 17.5x, and
`768 / 256 / 128 / 16` gives 18.5x.

### 3.1.1 Worth doing when PolicyValue matters

`policy_value_callback` reaches only **53 %** of its roofline where ValueOnly
reaches 89 %. The gap is the per-row Python `for` loop doing the segmented
softmax. B0 does not use this path, so it is not urgent — but it is measured
now, so it no longer falls under invariant 7.

### 3.2 Known optimisation target, deliberately untouched

The **vacancy prior is 79 % of native lane cost** (7.5 µs/eval vs 1.44 for legal
move generation). It is *not* optimised, on purpose: Gate C showed the lane side
is nowhere near binding, so tuning it would be tuning against an unmeasured
constraint. Revisit only if a GPU-host measurement says it binds.

---

## 4. Environment setup on a new host

```bash
# deps: pybind11 is build-time only and deliberately NOT in build-system.requires
python -m pip install "pybind11>=2.12" pytest
python -m pip install -e .          # works with no compiler; extension is optional

python tools/build_native.py        # -> src/diamond/alphazero/native/_diamond_native*.so
pytest tests/native -q              # gates A-D, ~40s
```

`DIAMOND_NATIVE_ARCH` overrides the `-march` flag (default `broadwell`, the
training host's baseline; `none` disables it).

The extension is **optional by construction**. If it is absent, `tests/native`
skips and everything else stays green — verify with `mv` on the `.so` if you
change anything in that area.

### Fixtures the tests need

- `tests/native/fixtures/positions.jsonl` — committed, regenerate with
  `python tools/build_native_corpus.py` (deterministic, byte-identical).
- `runtime/runs/soo/cpu8h-soo-20260819/latest.pt` — the **immutable step-80
  checkpoint**, `sha256:1634b901e213b065c107eea734b8c172c14babb1c2565352203961e86ea165af`.
  Tracked in git and hash-asserted by a test. Every measurement in the design
  rests on it; do not replace it.

---

## 5. How to run the benchmarks

```bash
# C.1 native single-search cost vs Python, same machine, full-game corpus
python az-bench/profiles/bench_native_single_search.py --repeats 3

# C.2 scheduler: lane starvation / batch cap / latency sweeps
python az-bench/profiles/bench_native_scheduler.py --mode starve
python az-bench/profiles/bench_native_scheduler.py --mode batch
python az-bench/profiles/bench_native_scheduler.py --mode latency

# D  callback boundary cost + real model through the callback
python az-bench/profiles/bench_native_callback.py --seconds 3
```

```bash
# The Gate D A/B table (GPU host).  The Python side is a full training run:
# drive it with run_point.sh first and pass its self-play seconds in.
python az-bench/profiles/bench_native_gpu_ab.py --device cuda:0 \
    --python-seconds 79 --python-samples 4410
python az-bench/profiles/bench_native_callback.py --seconds 5 --device cuda:0
```

**GPU-host baseline** (RTX 5090 / Ryzen 9 9950X3D, 16 physical cores), ValueOnly:

| measurement | value |
|---|---|
| Python backend, tuned (32 workers) | 3,573 evals/s |
| native, cap 32, diverse lanes at measured GPU latency | 66,041 evals/s (**18.5x**) |
| native, cap 256 | 147,326 evals/s (41.2x) |
| value-only forward, batch 32 | 0.401 ms |
| callback boundary | ~5 µs/crossing, ~1 % of the forward |
| evaluator thread occupancy | 98.8–99.6 %, every configuration |

**Local baseline for comparison** (8-logical-core laptop, i7-1165G7 — neither
host's absolutes transfer to the other; that is the point of re-running):

| measurement | local value |
|---|---|
| native lane stages, total | 9.43 µs/eval (Python 416.28 on the same machine) |
| whole search, dummy evaluator | 2.6 µs/eval, ~380k evals/s single lane |
| scheduler ceiling, 0 latency | 770k evals/s at 4 threads |
| scheduler at B32 / 3.784 ms | 7,775 evals/s, 92 % of roofline |
| callback boundary | 30–55 µs/batch, 0.4–1.0 µs/eval |

---

## 6. Invariants that must not break

1. **The Python implementation is the oracle and stays.** `game/rules.py`,
   `alphazero/encoder.py`, `bootstrap/heuristic.py`, `mcts/search_2p.py` and
   `game_adapter.py` are not dead code duplicated by C++ — every gate is defined
   as equality against them, and CI re-runs that comparison on every change.
   They are the only defence against risk 1 (two authoritative rule
   implementations drifting). Do not delete them.
2. **Topology is generated, never transcribed.** `native/topology.py` derives
   every board fact from `diamond.game.board` and injects it at import.
   No neighbour, camp or rotation is hard-coded in C++.
3. **Legal action ordering is exact parity, in every mode.** It decides which
   action each Dirichlet noise component lands on. Compare *sequences*, never
   sets — a set-based gate passes a backend that misassigns every noise
   component.
4. **Stochastic MCTS is unimplemented and must refuse, not approximate.**
   `dirichlet_epsilon > 0` or `temperature > 0` raises. The RNG policy is §9 of
   the design: cross-backend bit-exact RNG parity is explicitly *not* required,
   but each backend must be deterministic per seed.
5. **An evaluator must use `SearchSession::pending_state()`**, the node awaiting
   evaluation — never the lane's root. See §7.2.
6. **The batcher thread must not be starved.** See §7.5.
7. **Do not micro-optimise before measuring.** No SIMD, no bitset tricks, no
   custom allocators until a measurement names the constraint.

---

## 7. Pitfalls already paid for — do not rediscover these

### 7.1 The salt must avalanche (Gate C)
The dummy evaluator's per-lane salt was mixed as
`hash ^= salt + K + (hash << 6) + (hash >> 2)`, which moves mostly low bits —
and the value is read from `hash >> 11`, which discards them. Lane values
differed at the 11th decimal place, **32 lanes played 1 distinct game**, and
batching looked flawless *because* every lane marched in lockstep. Throughput
looked healthy the entire time. Pinned by `test_lanes_play_different_games`.

### 7.2 Root state vs node state (Gate D)
`PythonBatchEvaluator` computed the vacancy prior from the lane's **root**
instead of the node being expanded; they coincide only on the first expansion of
each move. Gate C structurally could not catch it — the dummy evaluator derives
priors from a request hash and never reads a state. It took an end-to-end
comparison against the Python oracle to surface.

### 7.3 Batch size changes the model's output
Batch size perturbs the Soo model's values by **~3e-8** — far larger than the
≤1e-12 by which the native and Python vacancy priors differ. Any parity
comparison must hold batch size fixed on both sides, or it tests the model's
numerics rather than the backend. `trunk_only` *is* bit-identical to the full
forward, so ValueOnly is a pure saving.

### 7.4 Exceptions must not escape a `std::thread`
A raising Python callback used to terminate the process. Worker and evaluator
threads capture the first failure, tear the run down so nothing waits on a dead
thread, and rethrow on the calling thread. Keep that when adding threads.

### 7.5 Oversubscription collapses throughput
8 workers + the batcher on 4 physical cores dropped the scheduler from 770k to
260k evals/s: the batcher stopped getting CPU and everything queued behind it.
Leave a core for the batcher when choosing thread counts on the 36-core host.

### 7.6 `games ≈ 2 × max_batch`, not more
512 lanes at cap 32 gave the same throughput as 64 lanes while inflating
per-eval latency 3.1 ms → 49 ms. `max_batch` is the only knob that moves
throughput. This contradicted the pre-measurement expectation.

### 7.7 Tie-breaks need tied priors to test
With distinct priors, no two PUCT keys are ever exactly equal, so the
`(-score, action)` tie-break is never consulted and a reversed tie-break passes
every test. Production ties constantly — the vacancy prior gives equal
probability to every action with the same integer progress score. Gate B runs a
**uniform-prior** evaluator specifically for this.

### 7.8 Step-beats-jump can never fire
Jump chains land on `source + 2v`, whose cube distance is always even; a step
destination is at distance 1. The sets are disjoint. Python's
`if landing not in moves` guard and its C++ mirror are both unreachable. Kept
deliberately — the mirror is exact, and the corpus cannot police a divergence
there.

### 7.9 A real evaluator has no salt (Gate D, GPU host)
§7.1 is about the dummy evaluator's salt. The mirror-image trap is that a *real*
model callback has no salt at all — it is a pure function of the position — so
in the only mode native MCTS accepts (`epsilon = 0`, `temperature = 0`) all
lanes from one opening play **one** game. Measured: 32 lanes, 1 trajectory.
Throughput does not notice, because a dense forward is indifferent to equal rows
and the evaluator is 99 % of wall. `test_lanes_play_different_games` cannot
catch it — it tests the component that has the salt. This is *why* Gate E
existed; it is now pinned from both sides, by
`test_a_real_evaluator_locks_lanes_together_without_exploration` (the cause) and
`test_exploration_makes_lanes_diverge_under_a_real_evaluator` (the fix).

### 7.10 A permissive RNG policy makes stream tests worthless (Gate E)
§9 allows the native draw sequence to differ from Python's. The consequence is
easy to miss: **no comparison of sequences can then catch a wrong sampler.** An
inverted boost exponent, a double-normalised weight vector, noise applied after
the first selection — each produces a perfectly plausible stream. Only the
distribution shows them. Gate the samplers on moments, CDFs and frequencies, and
expose whatever the search would otherwise hide (`root_priors` exists solely so
the Dirichlet mixture is observable). Related: at α = 0.3 a quantile-for-quantile
comparison is statistically hopeless — see §11.5.

---

### 7.11 A training sample is the root, not the pending node (Gate F)
§7.2 says an *evaluator* must use `pending_state()`. The mirror image bites
whoever records training data: after a completed search, `pending_features()` is
whichever leaf was expanded last, so recording it mislabels every sample's
player-to-act — and since the value target is `+1` when
`canonical_player_ids[0]` is the winner, **half the labels come out inverted**.
The games are legal, the policies are right, and the loss still goes down. Only
an end-to-end comparison against the oracle finds it. Use
`SearchSession::root_features()`.

### 7.12 A deterministic parity test may be comparing nothing (Gate F)
Cross-backend episode parity is only definable where both backends are
deterministic — and **deterministic self-play does not terminate**. Greedy play
from the opening burns the full 2000-move cap at 4, 8 and 16 simulations, so
both sides produce zero samples and every field compares `0 == 0`. The first
version of the Gate F parity test passed that way. Start such comparisons from a
near-terminal corpus position, pin it by tag (depth does not predict
termination: turns 300 and 375 never finish while 340 finishes in six moves), and
assert the comparison is non-empty.

## 8. Repo conventions

- **Branch + PR for everything**; `main` is the only long-lived branch and is
  kept clean of stale branches.
- **CI jobs**: `core` (3.11/3.12/3.13, no compiler, no pybind11 — proves the
  extension stays optional), `gui` (Qt, offscreen), `lint` (changed files only),
  `native (Gate A-F)` (builds the extension, installs CPU torch, asserts the
  checkpoint digest, runs all gates).
- **CI must not be allowed to skip silently.** Both the `gui` job and the
  `native` job assert their fixtures are present in a dedicated step, so a
  missing dependency is a red build rather than quiet skips. Follow that pattern.
- `.serena/` is **deliberately deleted and untracked**. Do not re-add it; stage
  explicit paths rather than `git add -A`.
- Benchmarks live in `az-bench/profiles/bench_*.py`, out of production code.
