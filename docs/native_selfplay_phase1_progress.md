# Native Soo Self-Play — Phase 1 Progress

Implementation log for [native_selfplay_phase0.md](native_selfplay_phase0.md) §8.
Follows the standing instruction there: **exact parity and a simple native
representation before low-level micro-optimization.**

**Status: Gates A, B and C pass. Gate D's correctness half passes on CPU;
its throughput half needs the GPU host.** The `selfplay_backend = "native"`
config switch is not written.

> Everything below was done on a **local development machine**, not the
> RTX 5060 Ti / Xeon training host. No GPU numbers, no throughput claims and no
> Gate C/D measurements are made here — those need the server. Gate A is a pure
> CPU correctness gate and is portable, so it is the whole of this phase.

---

## 1. Environment and dependency status

Probed on the development host (WSL2, Ubuntu 25.10 userland), conda env
`alphadiamond`:

| | phase 0 doc (GPU host) | this machine | note |
|---|---|---|---|
| python | 3.12 | **3.14.6** | extension builds cleanly on both |
| g++ | 13.3.0 | **15.2.0** | C++20 either way |
| cmake | 3.28.3 | 4.2.3 | unused; setuptools is the build path |
| numpy | 2.5.2 | 2.5.2 | not linked by the extension |
| torch | — | 2.12.0 | not touched by Phase 1 |
| pybind11 | absent | **3.0.4, present** | no install needed |
| `-march` | broadwell | tigerlake capable | build defaults to `broadwell` |

Changes made:

- **Nothing new was needed for the native build.** pybind11 3.0.4 was already
  in the env; the Phase 0 probe was run on the GPU host, where it is absent.
- Installed `qtawesome` — a declared project dependency that was missing
  locally, so `pytest` could not even *collect* `tests/test_icons.py`. Unrelated
  to the native work; the suite could not be shown green without it.
- Installed `ruff` to match what CI lints with.

`-march=broadwell` is kept as the default because it is the training host's
baseline and a strict subset of this machine's ISA, so one binary spec covers
both. Override with `DIAMOND_NATIVE_ARCH` (`none` disables the flag).

---

## 2. What landed

```
native/
  include/soo/{board,action,state,rules,encoder,prior}.hpp
  src/{board,rules,encoder,prior}.cpp
  bindings.cpp                       -> pybind11 module `_diamond_native`
src/diamond/alphazero/native/
  __init__.py                        import guard + capability probe
  topology.py                        authoritative table export
tests/native/
  conftest.py                        skips cleanly with no extension
  test_topology_parity.py            step 0 gate
  test_rules_parity.py               Gate A
  test_mcts_parity.py                Gate B
  reference_evaluator.py             the Gate B evaluator, in Python
  fixtures/positions.jsonl           1,327 committed positions
tools/
  build_native.py                    in-place build helper
  build_native_corpus.py             deterministic corpus generator
setup.py                             optional extension declaration
```

`tree.hpp`, `mcts.hpp` and `evaluator.hpp` arrived with Gate B (§7 below).
Against the Phase 0 file plan, `native/CMakeLists.txt` is deliberately absent
(setuptools was the stated default path); `backend.py`, `batcher.hpp`,
`selfplay.hpp` and `rng.hpp` are not written and not stubbed — they belong to
the threading work, which Gate B must not anticipate.

### Step 0 — topology export

`topology.py` derives neighbours, camp membership, the 73×73 pairwise distance
table and all six canonical rotations from `diamond.game.board` and
`diamond.alphazero.encoder`, and hands them to the extension at import time.
**No board fact is transcribed into C++.** `test_topology_parity.py` reads the
installed tables back out of the extension and compares element-wise, plus
asserts every canonical mapping is a bijection with a correct inverse.

This is the concrete mitigation for risk 1 (two authoritative rule
implementations): the *data* half of the duplication is generated, so only the
*algorithms* are ported, and those are what Gate A pins.

### Steps 1–4 — prior, rules, state, encoder

Implemented in the order §8 prescribes, which is the measured cost order:

1. **Vacancy bootstrap prior** (39.6 % of worker search CPU). Integer potential
   over the fixed 73×73 table with `std::bitset<73>` piece sets; softmax in
   `double` with the same max-shift as Python. No `frozenset` churn.
2. **Legal move generation** (21.7 %). Steps in direction order, then chained
   jumps by BFS with a fixed 73-entry queue and two bitsets. Reproduces Python's
   **ordering**, not merely its set.
3. **`State` + `apply_action`** (7.6 %). 80-byte POD state. `apply_action`
   mirrors `GameSession.commit` exactly: validate, apply, `update_ranking`,
   then hand over via `next_player_id` against the *new* podium.
4. **Canonical encoder** (4.2 %). Rotation to canonical `z+`, occupancy channels
   rotated to `self, next[, previous]`, finished flags appended.

The extension also exposes `search_current_player_id`, reproducing
`DiamondSearchAdapter`'s 2P terminal-perspective rule, because Phase 2's scalar
backup depends on it and it is cheap to pin now.

### Step 5 — Gate A

`tests/native/test_rules_parity.py` runs the committed corpus through both
implementations and asserts, for every position:

| checked | tolerance |
|---|---|
| player to act, terminal status, finish order | exact |
| `search_current_player_id` (2P terminal perspective) | exact |
| physical legal action ids, **as an ordered sequence** | exact |
| canonical legal action ids, **as an ordered sequence** | exact |
| resulting occupancy / player / turn / status / finish order for **every** legal action | exact |
| the same, applied through canonical action ids | exact |
| physical ↔ canonical action mapping and its round trip | exact |
| canonical player ids | exact |
| node features | exact float equality |
| vacancy bootstrap prior | ≤ 1e-12 |

**Result: pass.** 10 tests, ~7 s, 1,327 positions, 13,270 piece-sources and
every legal successor of each.

---

## 3. The corpus

`tests/native/fixtures/positions.jsonl`, regenerated deterministically by
`python tools/build_native_corpus.py` (byte-identical across runs).

| bucket | count | why |
|---|---|---|
| `packing` | 892 | a target camp ≥ 6 filled — where the v2 prior's potential stops being a fixed table |
| `conflict` | 293 | a piece with both step and jump moves available |
| `prior` | 92 | trajectory samples every 17th ply |
| `walk` | 48 | random legal walks at depths 1/3/8/20/50/120 |
| `opening` | 2 | the 2P and 3P standard openings |

918 two-player, 690 three-player (3P is not the Soo target but is cheap to keep
parity on, and it exercises the third occupancy channel and the mid-match
`finish_order` path that 2P never reaches).

Trajectories come from **full games driven by the production bootstrap prior**,
sampled rather than argmax'd. That satisfies risk 3 in Phase 0: per-call cost
varies ~3x between opening and full-game positions, so a corpus of openings
would both understate cost and miss the long jump chains entirely. The last 40
plies of every game are kept at full density because that is where target camps
are partially filled.

Phase 0 §7 also asks for states replayed from existing self-play ledgers. The
committed ledgers under `az-bench/` are run metadata only — replay chunks and
checkpoints are gitignored, and none are present locally — so prior-driven full
games stand in. Worth revisiting on the server, where real replay exists.

---

## 4. Findings

### The step-beats-jump rule can never actually fire

Phase 0 §7 calls for corpus positions "where a destination is reachable by both
a step and a jump chain (the step-wins rule)". Searching 13,270 piece-sources
across the corpus found **zero** such positions, and the geometry says there can
be none:

> Every jump hop moves `source + 2d` for a unit direction `d`, so any jump chain
> lands on `source + 2v` for an integer cube vector `v`. The cube distance of
> `2v` is `2·dist(v)` — always even. A step destination is at distance 1. The
> sets are disjoint.

So Python's `if landing not in moves` guard in `moves_from` is unreachable, and
so is the mirror of it in `native/src/rules.cpp`. The native guard is kept
anyway — it mirrors the oracle line for line, and dropping it would be an
optimization that the corpus cannot police. Both are commented to say so.

Practical consequence: a mutation that removes that guard passes Gate A, and
that is correct rather than a coverage hole. The two mutations that *do* matter
— reversing the step direction order, and letting the BFS re-emit an already
seen landing — both fail Gate A loudly, which is what was checked.

### Ordering parity is the real content of Gate A

Reversing the direction order in native step generation produces the identical
*set* of legal actions and fails only `test_legal_actions_match_as_ordered_
sequences` and the prior comparison. Had Gate A compared sets, it would have
passed a backend that lands every Dirichlet noise component on a different
action (Phase 0 §9, risk 2). Sequence comparison is doing real work.

---

## 5. Build, test, and the optional-extension guarantee

```bash
python tools/build_native.py            # -> src/diamond/alphazero/native/_diamond_native*.so
pytest tests/native -q                  # Gate A
```

The extension is **optional and off the default path**:

- `pybind11` is in the `native` extra, deliberately *not* in
  `build-system.requires`, so `pip install -e .` still works with no compiler.
- `setup.py` returns no `ext_modules` when pybind11 is absent, and marks the
  extension `optional=True` when it is present.
- `diamond.alphazero.native.is_available()` / `native_error()` are the probe;
  nothing imports the extension eagerly.
- Verified by deleting the built `.so`: `tests/native` skips (10 skipped) and
  the full suite stays green.

CI gains a `native (Gate A parity)` job that installs pybind11, builds the
extension and runs Gate A. It is separate from `core` on purpose — `core` must
keep proving the tree is green *without* a compiler.

---

## 6. Not done, and explicitly out of this phase

- **Phase 2 MCTS** (`tree.hpp`, `mcts.hpp`) and Gate B deterministic parity.
- **Batch callback ABI** (`ValueOnly` / `PolicyValue`), the batcher, lane
  threads, the GIL policy of §5 and its three required tests.
- **`native/backend.py`** and the `selfplay_backend = "native"` config switch.
- **Gate C** (native throughput, no Python) and **Gate D** (end-to-end).
  Both need the GPU host; neither is attempted or estimated here.
- **No native cost-per-evaluation figure is claimed.** Phase 0 §0.5 forbids it
  until Gate C measures it, and nothing in this phase measured it.


---

## 7. Gate B — deterministic MCTS parity

**Result: pass.** 21 tests total across the native suite, ~12 s.

Scope was held deliberately narrow: **single-threaded, single-game, one
deterministic evaluator, `dirichlet_epsilon = 0`, `temperature = 0`.** No
batcher, no thread pool, no Python callback, and no RNG. The native search
*refuses* rather than approximates when either stochastic knob is non-zero, so
the unimplemented RNG policy of §9 cannot be reached by accident; that refusal
is itself a test.

### The evaluator is the thing both sides share

Gate B compares two searches, so the one component they have in common has to be
pinned first. `tests/native/reference_evaluator.py` and
`native/src/evaluator.cpp` are two implementations of one spec: FNV-1a over a
canonical byte serialization of the request, then an exact 53-bit mantissa
division. Everything is integer arithmetic until a single division by 2^53, so
the two agree **bit for bit**, not to a tolerance —
`test_reference_evaluator_is_bit_identical_across_backends` asserts that over
all 722 two-player corpus positions before any search is compared.

It is a pure function of the request, never a constant. §7 of the design
insists on that, and the reason showed up under mutation: a constant makes every
PUCT key tie, so selection collapses to the action-id tie-break and a genuinely
divergent traversal can still produce matching visit counts.

### What is compared

| checked | tolerance |
|---|---|
| selected action | exact |
| root actions, **as an ordered sequence** | exact |
| visit count per root action | exact |
| q per root action | **bit-identical** (§7 asks for 1e-6) |
| policy per root action | 1e-6 |
| evaluator call count | exact |
| evaluator request sequence: `(request hash, legal actions)` per expansion, in order | exact |
| expanded legal-action sequence, per node, at every expansion | exact |
| simulations run | exact |
| terminal-leaf simulation count | exact |

Simulation counts 1, 2, 8, 33, 64 across 16 sampled positions, plus 400
simulations on the three narrowest-branching positions, plus every corpus
position that reaches a terminal leaf.

**q is asserted bit-identical, not within 1e-6.** The PUCT key is a double
comparison, so a one-ulp drift in q is not a rounding curiosity — it can flip a
selection and change the entire descent. The looser bound is kept as the
documented contract; the strict one is asserted while it holds, and will fail
loudly if it ever stops.

`exploration_bonus` is written in Python's operation order —
`((c_puct * prior) * sqrt(max(1, parent))) / (1 + edge_visits)` — for exactly
that reason.

### The request sequence is the load-bearing comparison

Comparing root visit counts alone is necessary and nowhere near sufficient: two
searches can reach identical root statistics through different traversals. The
request-hash sequence pins *which* leaves were evaluated and *in what order*.
Under mutation it is consistently the assertion that fires first.

### Native tree representation

Per the Phase 0 sketch and not a transliteration of the Python object model:
`std::vector<Node>` + `std::vector<Edge>` arenas, dicts replaced by contiguous
index ranges, and `parent_visits` served from a cached `total_visits` aggregate
rather than re-summing the children on every selection. The aggregate is a sum
over integers, so caching it is exact; Gate B pins the behaviour either way.

Edge blocks are contiguous and append-only, which holds only because a node is
always expanded in the same simulation that creates it. That is an invariant a
future deferred-expansion change could break silently, so `Arena::open_edges`
checks it rather than assuming it.

### Mutation results

Gate B was probed with five mutations, which is also how two coverage gaps were
found and closed:

| mutation | caught by |
|---|---|
| parent-visit aggregate off by one | request sequence, expanded sequences, deep trees |
| backup sign flipped once too few | request sequence, expanded sequences, deep trees |
| child node uses `state.current_player_id` instead of the search perspective | deep trees → **now** the terminal-leaf test |
| PUCT tie-break reversed to the largest action id | **nothing** → **now** the tied-prior test |
| root node uses `state.current_player_id` | nothing, correctly — the root is never terminal, so the two agree by construction |

Two gaps this exposed, both now closed:

1. **The PUCT tie-break was untested.** With distinct priors no two selection
   keys are ever exactly equal, so the `(-score, action)` ordering was never
   consulted. That is not a theoretical case: the production vacancy prior
   assigns equal probability to every action sharing an integer progress score,
   which ties the keys exactly on an unvisited node. Gate B now also runs every
   comparison against a **uniform-prior** evaluator, which keeps the value
   request-dependent while tying every prior. It catches the reversed tie-break
   immediately.
2. **The terminal-value path was covered only by luck.** Only 20 of 722
   two-player corpus positions reach a terminal leaf within 64 simulations, so
   `terminal_scalar_value` and the search adapter's `finish_order[1]` terminal
   perspective rested entirely on the deep-tree case. There is now a dedicated
   test over exactly those positions, with a floor assertion so it cannot
   quietly become a no-op if the corpus changes.

Terminal-leaf counts need no new instrumentation: every simulation either
expands one leaf or bottoms out on a terminal node, and the root costs one
expansion, so `simulations - (evaluator_calls - 1)` gives it on both sides.

### Still not measured

No timing was taken, and none should be read into this. Gate B ran the Python
oracle and the native backend against each other for correctness only. §0.5
stands: native cost per evaluation is a Gate C question, on the GPU host, and
Gate C should separate *native single-search cost* from *scheduler + dummy
evaluator throughput* rather than reporting one blended number.


---

## 8. Gate C — throughput, no Python

Run on the **local development machine** (i7-1165G7, 4 physical / 8 logical
cores), not the 36-core training host. Absolute ceilings will differ there;
the *shape* — what binds and what does not — is what this section claims.

Split into the three questions the design says to keep apart, so that a
disappointing number identifies a culprit instead of prompting a guess.

### 8.0 A prerequisite: the search had to become resumable

Synchronous MCTS allows one outstanding request per lane. A worker that blocks
on its own evaluation therefore pins a lane to a thread, caps the achievable
batch at the thread count, and makes "many logical games" buy nothing. So
`MCTS2P` was refactored into a `SearchSession` that **suspends at its
evaluation points** and is driven from outside; `MCTS2P` is now a thin
synchronous driver over it, which is the Gate B shape.

Gate B was the regression test for that refactor, and it is why the refactor
was safe to attempt: the restructured search still produces bit-identical q
values and the identical evaluator request sequence.

### 8.1 Native single-search cost

`az-bench/profiles/bench_native_single_search.py`. Per-call stage cost over the
722 full-game two-player corpus positions, with Python re-measured **on the same
machine** — quoting against the §0.1 table would compare two different hosts.

| stage | python µs | native µs | ratio |
|---|---|---|---|
| legal move generation | 142.18 | 1.44 | 98.9x |
| vacancy prior | 208.45 | 7.48 | 27.9x |
| canonical encoder | 27.66 | 0.38 | 73.7x |
| state + apply_action | 37.99 | 0.14 | 274.4x |
| **total** | **416.28** | **9.43** | **44.1x** |

Stable to ±5 % across runs. `steady_clock::now()` costs 14 ns, so the two clock
reads per stage inflate the fastest native stages by up to ~20 % — the error
runs *against* native, making these conservative.

Whole search, 64 simulations, dummy evaluator: **2.6 µs/eval**, 0.166 ms/search,
**~380,000 evals/s on one lane**. That is below the stage total because the
search does not compute the vacancy prior — the dummy evaluator stands in for it.
A production-shaped lane is roughly `2.6 + 7.5 ≈ 10 µs/eval`.

**The vacancy prior is now the dominant native lane cost** — 79 % of the
four-stage total, having been 50 % in Python. It is the obvious optimisation
target, and it is deliberately *not* optimised here: §8.2 shows the lane side is
nowhere near binding, so tuning it now would be tuning against an unmeasured
constraint.

### 8.2 Scheduler with a dummy evaluator

`az-bench/profiles/bench_native_scheduler.py`. Many logical games over a fixed
worker pool and one global batcher, per the agreed architecture.

**Batch formation is never the problem.** At every combination tested with at
least 2x cap lanes in flight, batches were **100 % full at the cap** — mean, p50
and p90 all equal to `max_batch`. Lane starvation did not occur anywhere in the
sweep.

Throughput is exactly `max_batch / evaluator_latency`, at 87–93 % of that
roofline. At a fixed 3 ms per batch:

| batch cap | evals/s | roofline | achieved |
|---|---|---|---|
| 16 | 4,872 | 5,333 | 91 % |
| 32 | 9,540 | 10,667 | 89 % |
| 64 | 18,922 | 21,333 | 89 % |
| 128 | 37,300 | 42,667 | 87 % |

The missing 7–13 % is the batcher thread's own serial work between dispatches.

Three conclusions, one of which contradicts the expectation going in:

1. **`max_batch` is the only knob that moves throughput.** Exactly risk 4 in the
   design; now measured rather than anticipated. It should be promoted to a
   first-class knob.
2. **More games do not help beyond ~2x the batch cap.** Going from 64 to 512
   lanes at cap 32 left throughput flat at ~9,600 evals/s while per-eval latency
   grew from 3.1 ms to 49 ms. **512 logical games is not a natural operating
   point** at these caps — it is pure queueing. The right rule is
   `games ≈ 2 x max_batch`: one batch being evaluated while the next is
   assembled.
3. **CPU utilisation is 1–5 %** at realistic latency. The lane work is nearly
   free relative to inference, which is the good outcome, not a wasted machine.

**The scheduler's own ceiling**, with zero evaluator latency:

| threads | evals/s | cpu |
|---|---|---|
| 1 | 178,609 | 52 % |
| 2 | 447,211 | 78 % |
| 4 | **770,157** | 90 % |
| 8 | 259,956 | 14 % |

Scaling is near-linear to 4 threads and then **collapses**. Eight workers plus
the batcher oversubscribe four physical cores, the batcher thread stops getting
CPU, and everything queues behind it (`runnable` falls to 2.5, `waiting` rises to
253). **The batcher thread must not be starved**: worker count should leave a
core for it. That is a scheduling constraint the design did not anticipate, and
it will matter when choosing thread counts on the 36-core host.

At 770k evals/s the scheduler is ~90x faster than needed to feed a 3.8 ms
value-only batch of 32. **The scheduler is not the bottleneck. The evaluator is** —
which is exactly where the design wanted the constraint to sit.

### 8.3 Evaluator latency sweep

The question that decides Gate D's odds: does batch formation survive at real
inference latency? Measured at §0.4's *own* value-only figures rather than round
numbers, with `games = 2 x cap`:

| cap | latency (§0.4 path D) | evals/s | batch | roofline | achieved | vs ~950 |
|---|---|---|---|---|---|---|
| 32 | 3.784 ms | 7,775 | 32.0 | 8,457 | 92 % | **8.2x** |
| 64 | 4.586 ms | 12,815 | 64.0 | 13,956 | 92 % | **13.5x** |
| 128 | 6.944 ms | 17,130 | 128.0 | 18,433 | 93 % | **18.0x** |

Batches stay **completely full** at every latency from 0 to 5 ms. The scheduler
does not need a slow evaluator to fill batches, and it does not degrade when it
gets one.

### 8.4 Verdict against the Gate C stop condition

> **C** — native + batcher throughput, no Python. *Stop if: no substantial
> headroom over 950 evals/s.*

**Gate C passes.** At the measured value-only latency for a batch of 32, the
native scheduler sustains 7,775 evals/s against a Python production ceiling of
~950 — **8.2x**, rising to 13.5x at B64 and 18.0x at B128, and it reaches 92 % of
the single-evaluator-thread roofline in every case.

The caveat that keeps this honest: the evaluator here is a dummy that *sleeps*.
It holds no GIL, allocates nothing, and never touches Python. Gate D replaces it
with a real callback, and risk 5 says plainly that one evaluator thread holding
the GIL is the next ceiling. What Gate C establishes is narrower and still
worth having: **everything on the native side of that callback has ceased to be
the constraint.**

### 8.5 Correctness, because a fast wrong scheduler is worthless

`tests/native/test_scheduler.py`, six tests, all under a watchdog so a hang fails
rather than wedging CI.

The strongest is **thread-count invariance**: a lane's evaluations depend only on
its own request and its own salt, so its entire game must be identical whether
one worker ran it or eight, however batches happened to form. It holds exactly
across 1/2/4/8 threads. Also covered: the single-lane tail (a solitary lane
dispatches alone at the full `max_wait_us`), batch-scale dispatch counts, workers
staying fed with more lanes than threads, and termination across degenerate
shapes including `games=1, max_batch=32` and `max_batch=1, threads=8`.

### 8.6 A bug that would have made this whole section a lie

The dummy evaluator salts each lane so trajectories diverge. The first
implementation mixed the salt in as
`hash ^= salt + K + (hash << 6) + (hash >> 2)` — which moves mostly low bits,
and the value is read from `hash >> 11`, which discards them.

Lane values then differed at the **11th decimal place**. Every lane played
identical moves: 32 lanes, **1 distinct trajectory**. Batching would have looked
flawless because all lanes marched in lockstep, submitting in perfect step —
the error flattered every number in this section.

Caught by checking the assumption rather than the throughput, which looked
entirely healthy throughout. Fixed with splitmix64's finaliser, giving 32/32
distinct trajectories; `test_lanes_play_different_games` now asserts it
permanently.

### 8.7 Not done, and still out of scope

- **No micro-optimisation.** No SIMD, no bitset tricks, no custom allocators.
  The vacancy prior is the identified target and stays untouched until there is
  a measurement that says it binds — §8.2 says it does not.
- **Gate D**: the real PyTorch callback, the `ValueOnly` ABI, GIL behaviour under
  a real evaluator, and the `selfplay_backend = "native"` switch.
- **All numbers here are from an 8-logical-core laptop.** The 36-core host will
  differ, and the 4-thread collapse point certainly will.


---

## 9. Gate D — the Python callback, on CPU

Gate D as specified needs the GPU host: *"same immutable checkpoint, seed and
config as the Python backend; report the full A/B table."* That table is not in
this document and no throughput claim is made for it.

What does **not** need a GPU is everything else Gate D asks: the batch callback
ABI, the GIL discipline, and whether the native backend actually plays the same
game as the Python one. All of that is done, against the same immutable
checkpoint — `sha256:1634b901…`, `training_step` 80 — which is committed to this
repository and hash-checked by a test.

### 9.1 The callback ABI

`ValueOnly` — the B0 production path:

```
C++  -> Python     float32 features[B][73][4]
Python -> C++      float32 values[B]
```

Nothing else crosses. The vacancy prior is native (Gate A), so the entire policy
tail is absent from the boundary: no bounds check, no padded index build, no
gather, no mask, no softmax, no validity sync, no `[B, 5329]` device-to-host
copy, no priors dict, no response envelope.

`PolicyValue` — the reference path — additionally passes ragged legal sets as a
flat array plus offsets rather than padded, so the Python side gathers without
materialising `[B, max_legal]`. Both modes stay compiled and both are tested.

The feature array is a **view onto a buffer the native side reuses**, not a copy.
It is valid for the duration of the call only; the contract is documented on
`backend.py` and the buffer is refilled on the next batch.

### 9.2 GIL discipline

The entry point releases the GIL for the whole run. The single evaluator thread
is the only thread that ever reacquires it, around the callback and nothing
else. Lane threads never touch Python.

Section 5 names the deadlock this design must not have — an entry point that
kept the GIL would leave the evaluator thread blocked on it forever. Its three
required tests are implemented, all under a watchdog so a hang fails rather than
wedging CI:

- **Deadlock guard** — a callback that imports, allocates and runs `gc.collect()`,
  under four workers.
- **Single-lane tail** — one lane dispatches alone at the full `max_wait_us`.
- **Callback frequency** — crossings equal *batches*, not evaluations; 32
  evaluations per GIL acquisition at cap 32.

Two failure modes not in the design's list, both real:

- **A raising callback used to terminate the process.** An exception escaping a
  `std::thread` is fatal. Worker and evaluator threads now capture the first
  failure, tear the run down so nothing waits on a dead thread, and rethrow on
  the calling thread — the original Python exception type survives, and the
  scheduler is reusable afterwards.
- **A callback returning the wrong number of values** is rejected rather than
  read out of bounds.

### 9.3 End-to-end: the native backend plays the same game

The correctness claim Gate D exists for, on the immutable checkpoint:

| simulations | moves | result |
|---|---|---|
| 32 | 20 | **identical move for move** |
| 64 | 12 | **identical move for move** |

Python runs `MCTS2P` + `VacancyPriorEvaluator` over `TorchEvaluator`; native
computes the vacancy prior itself and takes only values across the boundary.
Same weights, `epsilon = 0`, `temperature = 0`.

A single lane is used deliberately, so both sides evaluate at batch 1. Batch
size perturbs the model's own output by **~3e-8** (measured), which dwarfs the
≤1e-12 by which the native and Python priors differ — comparing across different
batch shapes would test the model's numerics rather than the backend.

`trunk_only` is **bit-identical** to the full forward, so skipping the policy
head is a pure saving and not a different evaluator. That is asserted, because
every parity claim above depends on it.

### 9.4 A bug only an end-to-end test could find

`PythonBatchEvaluator` computed the vacancy prior from **the lane's root state**
instead of the state of the node being expanded. The two coincide only on the
first expansion of each move, so every deeper node got priors for the wrong
position.

It was invisible to Gate C: the dummy evaluator derives priors from the request
hash and never looks at a state. It showed up on the first real comparison
against the Python oracle — 9 of 10 moves matched and move 7 diverged, and
crucially it was **not a near-tie**: Python gave the chosen action 14 of 32
visits while native picked a 1-visit action. That gap is what ruled out the
≤1e-12 prior difference as an explanation and forced the real diagnosis.

`SearchSession` now exposes `pending_state()` — the node awaiting evaluation, not
the search root — and the comment says why an evaluator must use it.

### 9.5 What the boundary costs

The CPU forward here is ~30x the GPU host's (87 ms vs 3.1 ms at batch 32), so
absolute throughput with a real model says nothing about Gate D and is not
quoted. The transferable quantity is the cost of the boundary itself.

Measured with a null callback, ValueOnly, per **evaluator-thread** microsecond:

| batch cap | µs/crossing | µs/eval |
|---|---|---|
| 32 | 30.4 | 0.95 |
| 64 | 38.8 | 0.61 |
| 128 | 55.5 | 0.43 |

Marshalling plus one GIL acquisition costs **30–55 µs per batch**. Against §0.4's
measured 3.784 ms value-only forward at batch 32, that is **0.8 %**. The ABI is
not the ceiling.

### 9.6 Moving the prior off the critical thread

Those numbers are *after* a structural fix the measurement prompted.

The vacancy prior is a pure function of the request, but it was being computed
inside `evaluate()` — on the single evaluator thread. At 7.5 µs/eval that is
**241 µs of serial work per batch of 32**, charged to precisely the thread §8.2
showed must never be starved, while the search workers sat at ~15 % utilisation.

`BatchEvaluator` now has a `prepare()` hook that runs **on a search worker**
before the lane is handed over. Request-only work belongs there.

| | evaluator µs/crossing | evaluator µs/eval | worker cpu |
|---|---|---|---|
| prior in `evaluate()` | 240.9 | 7.53 | 15 % |
| prior in `prepare()` | **30.4** | **0.95** | **78 %** |

Same total work, moved from the scarce serial resource to the idle one — 7.9x
less time on the evaluator thread, and 6 % of a real value-only batch handed
back.

This is not the micro-optimisation the plan defers. It is a scheduling fix that
only became visible once the boundary was measured, and it changes which thread
does the work rather than how fast the work is.

### 9.7 What Gate D still needs from the GPU host

- The **A/B throughput table** against the Python backend.
- Whether the callback recreates the ceiling under a *real* forward — risk 5's
  actual question. §9.5 shows the marshalling does not; the forward is untested
  at GPU speed from the native side.
- `selfplay_backend = "native"`, and a full training loop consuming native
  self-play samples.

---

## 10. Gate D on the GPU host

Everything in §9 was measured on the laptop and said so. This section is the
other half: the same code on the training host, with the forward running at GPU
speed.

### 10.0 The host

| | |
|---|---|
| GPU | RTX 5090, 32 GB, driver 580.173.02, compute capability 12.0 (Blackwell) |
| CPU | AMD Ryzen 9 9950X3D, **16 physical / 32 logical** cores |
| torch | 2.13.0+cu130 |
| checkpoint | the immutable step-80 `latest.pt`, digest asserted `1634b901…` |

The extension builds unchanged (`-march=broadwell`) and **all 38 gate tests pass
on the first run**. No port work was needed; §4 of the handoff was accurate.

Note this is **not** the host `rtx5060_bottleneck_findings.md` was written on.
Every absolute number in that document is from an RTX 5060 with 18 physical
cores, and none of them transfer. The Python baseline below was therefore
re-measured here rather than quoted.

### 10.1 The Python backend, re-measured on this host

`run_point.sh` with the production `C-w48-g48` configuration, same seed, same
checkpoint. The work is byte-identical to the reference run — 48 games, 4,410
samples, median 89 moves — so this is a clean same-work comparison:

| run | workers | self-play s | evals/s |
|---|---|---|---|
| reference (RTX 5060) | 48 | 440 | 641 |
| P5090-w16-g48 | 16 | 91 | 3,102 |
| P5090-w24-g48 | 24 | 87 | 3,244 |
| **P5090-w32-g48** | **32** | **79** | **3,573** |
| P5090-w48-g48 | 48 | 85 | 3,320 |
| P5090-w64-g48 | 64 | 86 | 3,282 |

`evals/s = samples x simulations / self-play seconds`, the same formula the
reference row uses.

Two things worth recording:

1. **The Python backend is 5.6x faster here than on the reference host** — 3,573
   against 641. The A/B is run against this, not against the number in the
   design document. Quoting the 641 would have inflated the native result by
   5.6x for free.
2. **Worker count barely matters**: 16 workers and 64 workers are within 15 % of
   each other, on a 16-core machine. That is §7.3 of the findings reproducing
   exactly — the backend is **parent-bound**, serialized behind one coordinator,
   not bound by worker parallelism. GPU utilisation sat at **7 %** throughout.

### 10.2 Risk 5 — does a real GPU forward behind the GIL recreate the ceiling?

**Yes, and it is the right ceiling.** The evaluator thread is **98.8–99.6 % busy
in every configuration measured** — 20 combinations of thread count and batch
cap. Worker threads change nothing (2 threads and 24 threads land within 4 %);
`max_batch` is the only knob that moves throughput, exactly as risk 4 predicted
and §8.2 measured with the dummy evaluator.

But the *composition* of that busy time is the answer risk 5 actually wanted.
At cap 32 the crossing costs 468 µs against an isolated 401 µs forward:

| component | µs/crossing | share |
|---|---|---|
| the forward itself | ~401 | 86 % |
| boundary: marshalling + one GIL acquisition | ~5 | 1 % |
| H2D / D2H transfer and sync | ~62 | 13 % |

The boundary is **5 µs here, down from the laptop's 30 µs**. The evaluator
thread is saturated doing *GPU work*, not waiting on Python. Risk 5 named a real
constraint and the constraint is the model, which is where the design wanted it.

Risk 5 projected **8,456 evals/s** in ValueOnly. Measured: **70,720**. The
projection was arithmetically sound and anchored to the 5060's 3.784 ms forward;
this card does the same batch in 0.401 ms.

### 10.3 The A/B throughput table

`python az-bench/profiles/bench_native_gpu_ab.py --device cuda:0 --python-seconds 79 --python-samples 4410`

Three quantities, and the distance between them is the content:

- **roofline** — `max_batch / forward_seconds`; free, always-full batching.
- **real** — the actual Soo checkpoint through the callback on the GPU.
- **diverse** — the Gate C salted dummy evaluator, provably `2 x cap` distinct
  games, with `eval_latency_ms` pinned to *this host's measured forward*.

**ValueOnly — the B0 production path:**

| cap | fwd ms | roofline | real | diverse | % roof | calls/s | evals/call | evaluator % | vs Python |
|---|---|---|---|---|---|---|---|---|---|
| **32** | 0.401 | 79,743 | 70,720 | **66,041** | 83 % | 2,210 | 32.0 | 98.8 % | **18.5x** |
| 64 | 0.558 | 114,679 | 99,176 | 95,684 | 83 % | 1,550 | 64.0 | 98.7 % | 26.8x |
| 128 | 0.875 | 146,341 | 133,703 | 122,964 | 84 % | 1,045 | 128.0 | 98.8 % | 34.4x |
| 256 | 1.463 | 174,933 | 159,187 | 147,326 | 84 % | 622 | 256.0 | 98.9 % | 41.2x |

**PolicyValue — the reference path:**

| cap | fwd ms | roofline | real | diverse | % roof | calls/s | evals/call | evaluator % |
|---|---|---|---|---|---|---|---|---|
| 32 | 0.430 | 74,444 | 39,218 | 62,333 | 84 % | 1,226 | 32.0 | 99.4 % |
| 64 | 0.605 | 105,857 | 54,316 | 89,162 | 84 % | 849 | 64.0 | 99.4 % |
| 128 | 0.923 | 138,695 | 65,896 | 118,286 | 85 % | 515 | 128.0 | 99.4 % |
| 256 | 1.527 | 167,658 | 73,221 | 140,674 | 84 % | 286 | 256.0 | 99.6 % |

**Batches are completely full at every cap**, up to 256 — the §8.3 result holds
at real GPU latency.

**The quotable number is 18.5x**, at cap 32, against the *diverse* column,
against this host's own tuned Python baseline. Everything larger requires
raising `max_batch_size` above its production value of 32, which risk 4 already
said should be promoted to a first-class knob and which this table now justifies
with GPU-host evidence.

**PolicyValue reaches only 53 % of its roofline** while ValueOnly reaches 89 %.
The gap is the per-row Python `for` loop doing the segmented softmax in
`policy_value_callback`. That is a genuine, now-measured constraint on the
reference path — and it is a Python-side one, so it costs nothing to fix when
PolicyValue matters. It does not affect B0, which does not use it.

### 10.4 A caveat that had to be measured, not assumed

The `real` and `diverse` columns differ by 4–18 %, and the reason is not noise.

**With a real model callback, every lane plays the same game.** Lane diversity
in Gate C came entirely from the dummy evaluator's per-lane salt (§8.6). A real
model has no salt: it is a pure function of the position. With
`dirichlet_epsilon = 0` and `temperature = 0` — the only mode native MCTS
accepts — 32 lanes from one opening produce **1 distinct trajectory**. Measured
directly, not inferred.

This is §8.6's bug wearing different clothes, and none of the existing tests
could catch it: `test_lanes_play_different_games` asserts diversity *of the
dummy evaluator*, which is precisely the component that has the salt.

Throughput barely notices — a dense forward does not care that its rows are
equal, and the evaluator is 99 % of wall — which is exactly what makes it
dangerous. It is now pinned by
`test_a_real_evaluator_makes_every_lane_play_the_same_game`, written to **fail**
when stochastic MCTS lands so the assertion gets replaced deliberately rather
than quietly.

The `diverse` column exists because of this: it is the same scheduler at the
same measured GPU latency with lane diversity that is *proved* rather than
assumed, and it is the pessimistic figure. Quote it.

### 10.5 What this means for `selfplay_backend = "native"`

Items 3 and 4 of the handoff — the config switch, then a training loop on native
samples — read like wiring. **They are blocked, and not by wiring.**

A native self-play pool today would hand the trainer *N copies of one game* per
iteration. Production self-play depends on Dirichlet noise at the root
(`dirichlet_epsilon = 0.25`) and temperature sampling for the first 20 moves
(`temperature = 1.0`) for its trajectory diversity, and native MCTS refuses both
by design — invariant 4, which says it must refuse rather than approximate, and
is correct to.

So the remaining path is:

1. **Stochastic MCTS in the native backend** — Dirichlet noise and temperature
   sampling, to §9 of the design's RNG policy (distribution and semantics must
   match Python; the draw sequence need not). This is a **new gate**, with its
   own parity question, not a step inside Gate D.
2. Then `selfplay_backend = "native"` behind the existing `SelfPlayWorkerPool`
   contract — `run(tuple[SelfPlayJob, ...]) -> tuple[EpisodeResult, ...]`.
3. Then the training loop.

This was not visible before the GPU host, because it takes a real model to
expose it. It does not change Gate D's verdict.

### 10.6 Verdict

Gate D's throughput half **passes**, decisively. 18.5x the Python backend at the
production batch cap, on the same machine, same checkpoint, same seeded work,
against a Python baseline tuned on this host rather than quoted from another.
The evaluator thread is the ceiling and the ceiling is the GPU forward, which is
the constraint the design chose to have.
