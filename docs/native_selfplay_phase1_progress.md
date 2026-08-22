# Native Soo Self-Play — Phase 1 Progress

Implementation log for [native_selfplay_phase0.md](native_selfplay_phase0.md) §8.
Follows the standing instruction there: **exact parity and a simple native
representation before low-level micro-optimization.**

**Status: Gate A passes. Gate B passes.**
The batch callback ABI, the batcher, threading and the backend switch are not
started, deliberately — Gate B is single-threaded, single-game, and never calls
into Python.

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
