# Native Soo Self-Play — Phase 1 Progress

Implementation log for [native_selfplay_phase0.md](native_selfplay_phase0.md) §8.
Follows the standing instruction there: **exact parity and a simple native
representation before low-level micro-optimization.**

**Status: Phase 1 steps 0–5 complete. Gate A passes.**
Phase 2 (MCTS), the batch callback ABI and the lane runner are not started.

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
  fixtures/positions.jsonl           1,327 committed positions
tools/
  build_native.py                    in-place build helper
  build_native_corpus.py             deterministic corpus generator
setup.py                             optional extension declaration
```

Against the Phase 0 file plan, `native/CMakeLists.txt` is deliberately absent
(setuptools was the stated default path) and `native/backend.py` is not written
yet — it belongs to Phase 2, when there is a search to put behind the pool
contract. `mcts.hpp`, `tree.hpp`, `batcher.hpp`, `selfplay.hpp` and `rng.hpp`
are Phase 2 and are not stubbed.

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
