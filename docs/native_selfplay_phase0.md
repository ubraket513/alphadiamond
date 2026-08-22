# Native Soo Self-Play — Phase 0 Design

Design document for moving the Soo self-play/search hot path out of Python
multiprocessing into a native multithreaded subsystem, while training, replay,
checkpoints and the PyTorch model stay in Python.

This is the evidence, the proposed shapes, and the parity plan that Phase 1 is
judged against. It is a design record and is **not** rewritten as work lands.

**Status: Phase 1 (§8 steps 0-5) is implemented and Gate A passes.** Gates B, C
and D are open. See
[native_selfplay_phase1_progress.md](native_selfplay_phase1_progress.md).

Follows [rtx5060_bottleneck_findings.md](rtx5060_bottleneck_findings.md).

---

## 0. Measurements this design rests on

Everything below was measured on the RTX 5060 Ti / Xeon E5-2686 v4 host with the
immutable step-80 checkpoint (`sha256:1634b901…`). The stage profile is from
commit `d907970`; the transport and value-only numbers from `9557a98`, which
does not affect them.

### 0.1 Where the system's CPU goes

Production-scale run: 48 workers, 48 games, real trained network, 64 simulations,
`max_wait_ms` 2. Reproduces the ledgered production point (694 evals/s here vs
640 for `C-w48-g48`; 281,779 evaluations and 4,410 moves in both).

```
cpu ms/eval:  parent 1.589   workers 3.697   total 5.286
throughput    694 evals/s    evals/move 63.9
batches/s     63.2           mean batch 10.99
resp p50      44.75 ms       q->dispatch p50 29.08 ms
```

Worker-side stage CPU, `time.thread_time_ns`, exclusive (nested calls
subtracted), aggregated over all 48 workers. **This is the Python oracle
baseline the native backend will be compared against**, taken *after* the
duplicate legal-generation fix (risk 6 below):

| stage | ms/eval | share | calls/eval | µs/call |
|---|---|---|---|---|
| **bootstrap_prior** | **1.2241** | **39.6 %** | 1.00 | 1224.1 |
| **legal_action_ids** | **0.6704** | **21.7 %** | 1.02 | 660.0 |
| remote_cpu | 0.4281 | 13.8 % | 1.00 | 428.2 |
| apply_action | 0.2359 | 7.6 % | 1.01 | 234.6 |
| select | 0.1928 | 6.2 % | 1.67 | 115.4 |
| expand own | 0.1307 | 4.2 % | 1.00 | 130.7 |
| encode | 0.1294 | 4.2 % | 1.02 | 127.4 |
| MCTS residual (tree/backup) | 0.0680 | 2.2 % | — | — |
| is_terminal | 0.0105 | 0.3 % | 2.70 | 3.9 |
| current_player_id | 0.0048 | 0.2 % | 1.01 | 4.8 |
| **search_total (main thread)** | **3.0948** | 100 % | | |

Cross-validated: main-thread CPU 3.161 + other worker threads (response pump,
queue feeder) 0.540 = **3.701 ms/eval**, against 3.697 measured independently
from `/proc` by the parent's poller.

Before/after the duplicate-generation fix, same run configuration:

| | before | after | delta |
|---|---|---|---|
| legal_action_ids calls/eval | 2.02 | **1.02** | — |
| legal_action_ids ms/eval | 1.3452 | 0.6704 | −50 % |
| search_total ms/eval | 3.7946 | 3.0948 | −18.4 % |
| worker cpu ms/eval | 4.405 | 3.697 | −16.1 % |
| total cpu ms/eval | 6.050 | 5.286 | −12.6 % |
| throughput | 669 evals/s | 694 evals/s | +3.7 % |

Throughput moved only +3.7 % because the worker side was never the binding
constraint — the parent is (§0.2). The CPU reduction is what matters here: it
is the baseline a native implementation has to beat, and it removes work that
would otherwise have been ported.

### 0.2 What the multiprocessing architecture costs

Identical Soo self-play work, evaluator answering instantly in both arms (no
GPU, no model), full-length games:

```
LOCAL  1 process, no pool            1.154 ms cpu/eval    867 evals/s
POOL   workers=12   888 evals/s   parent 1.127  workers 4.196  total 5.324
POOL   workers=24  1030 evals/s   parent 1.073  workers 4.476  total 5.549
POOL   workers=48   900 evals/s   parent 1.070  workers 4.468  total 5.538
```

Two facts follow, and they are the whole case for this project:

1. **~79 % of all CPU is transport.** The same search costs 1.15 ms/eval
   single-process and 5.5 ms/eval through the pool.
2. **The parent is a hard serialized ceiling at ~950 evals/s, with no GPU
   involved at all.** Parent CPU sits at 96–110 % of one core in every row and
   throughput refuses to scale from 12 to 48 workers. Production's 669–711
   evals/s is already ~75 % of a ceiling that a free GPU could not lift.

An earlier version of this benchmark reported transport as *free*. That was a
measurement bug: child CPU was sampled after `SelfPlayWorkerPool.run` had
already joined and closed the workers, so it read zero and the "tree total" was
parent-only. Child CPU is now accumulated by a poller while the children live.

### 0.3 The search trajectory is not distorted by a constant evaluator

A constant value could in principle change PUCT's Q, hence which branches are
re-descended, hence calls per evaluation. Measured, it does not:

| | instant evaluator | real trained NN |
|---|---|---|
| evaluations / move | 64.0 | 63.9 |
| select calls / eval | — | 1.67 |
| apply_action / eval | — | 1.01 |

The trees are shallow by construction: with 64 simulations against ~54 legal
root actions, most simulations expand a fresh root child. What *does* vary is
per-call cost — `legal_action_ids` costs 171 µs/call in opening positions and
660 µs/call across full games, because jump chains lengthen as pieces disperse.
Any Phase 1 microbenchmark must therefore use full-game position corpora.

### 0.4 Value-only inference (B0 ABI question)

`VacancyPriorEvaluator` keeps the network's value and discards its policy
entirely, so in B0 every policy logit is computed, gathered, softmaxed, copied
to host, dictified, validated and pickled for nothing.

| batch | A full fwd | B trunk+value | C TorchEvaluator | D value-only path |
|---|---|---|---|---|
| 1 | 3.027 | 2.793 | 3.569 | 2.907 |
| 32 | 3.104 | 2.890 | 5.019 | 3.784 |
| 64 | 3.130 | 2.913 | 6.390 | 4.586 |
| 128 | 3.253 | 3.110 | 9.073 | 6.944 |

CPU ms per batch. aten operations per call: A 61, B 55, C 83.

**The saving is in the postprocessing, not the model.** Dropping the policy head
saves only 7 % of the forward (6 of 61 operations; the trunk is 49). Dropping
the whole policy *tail* saves 25 % of the evaluator path at batch 32 and 28 % at
batch 64.

Single-evaluator-thread roofline:

| batch | current path (C) | value-only (D) | gain |
|---|---|---|---|
| 32 | 6,375 evals/s | 8,456 evals/s | 1.33x |
| 64 | 10,016 evals/s | 13,955 evals/s | 1.39x |

Device span exceeds CPU only at batch 128, so the boundary stays launch-bound
through the batch sizes this design targets.

### 0.5 What is *not* yet known

- **Native cost per evaluation is not assumed anywhere in this document, and
  must not be.** The 1.15 ms/eval of useful Python work is measured; what the
  same work costs in C++ is a Gate C microbenchmark. Any figure for native
  throughput, native ms/eval or a speedup multiple is **speculative upside**
  until Gate C measures it, and no design decision here depends on one.
- Whether native lanes can fill batches to 32–64. That is Gate C.
- Whether the per-request IPC cost parallelizes across shards or processes.

---

## 1. Current Soo self-play hot path

```
WORKER PROCESS
 1  SooSelfPlayRunner.run                       selfplay/runner_2p.py:44
      per move: fresh MCTS2P(seed = selfplay.seed + move_count)   <- no tree reuse
 2  MCTS2P.run                                  mcts/search_2p.py:50
      _expand(root, root_noise=True)
      for simulation in range(simulations):
          descend: _select -> apply_action -> new ScalarNode
          leaf: terminal_scalar_value  or  _expand
          backup: value = -value per traversed edge
      select_from_visits(temperature)
 3  MCTS2P._expand                              mcts/search_2p.py:104
      game.evaluation_request(state)            -> encoder.encode + legal_action_ids
      evaluator.evaluate((request,))            -> VacancyPriorEvaluator
          base.evaluate  -> RemoteEvaluator     -> IPC round trip
          _priors(...)   -> heuristic replaces the neural prior
      set(game.legal_action_ids(state))         <- SECOND full legal generation
      add_dirichlet_noise(rng) if root
 4  RemoteEvaluator.evaluate                    inference/remote.py:48
      InferenceRequest.from_eval_request        validates 292 feature floats
      coordinator.submit -> mp request queue    (shared, one per pool)
      self._replies.get(timeout)                blocks ~49 ms/eval
 5  _ProcessRequestCoordinator._pump_responses  orchestration/selfplay_workers.py:341

PARENT PROCESS
 6  _InferenceBridge._pump_requests             orchestration/selfplay_workers.py:459
 7  InferenceCoordinator.submit / _run          inference/coordinator.py:220,250
 8  _validated_request                          inference/coordinator.py:297  (rebuild + revalidate)
 9  InferenceModelPool.evaluate                 inference/model_pool.py:68
10  TorchEvaluator.evaluate                     evaluator/torch.py:38
11  InferenceResponse.from_eval_result          inference/protocol.py:261  (revalidates priors)
12  _InferenceBridge._pump_responses            -> one mp.Queue per lane, one feeder thread each
```

Structures the native design removes: spawn worker processes, pickle/unpickle on
every request and response, a shared mp request queue, one response queue and
feeder thread per lane (`parent_threads = lanes + 50`, fitted across 30/36/48/64
lanes), correlation-id routing, per-request Python protocol objects, and the
GIL-serialized coordinator.

### Semantics that must be preserved

| concern | authority |
|---|---|
| board topology, 73 positions, 6 directions | `game/board.py`, `game/coordinates.py` |
| step + chained jump generation, BFS canonical path | `game/rules.py:51 moves_from` |
| move application, ranking, turn rotation | `game/session.py`, `game/state.py` |
| physical action id = `source * 73 + destination` | `alphazero/action_codec.py:46` |
| canonical rotation (home camp -> `z+`) | `alphazero/encoder.py:61 _canonical_cube` |
| node features = occupancy channels + finished flags | `alphazero/encoder.py:82 encode` |
| PUCT, tie-break by action id, unvisited q = 0 | `mcts/search_2p.py:124`, `mcts/puct.py:9` |
| scalar backup, one sign flip per edge | `mcts/search_2p.py:88` |
| root Dirichlet, gammavariate over `priors.items()` | `mcts/puct.py:13` |
| temperature selection | `mcts/puct.py:31` |
| vacancy bootstrap prior | `bootstrap/heuristic.py:120` |
| training sample construction | `selfplay/runner_2p.py:76` |

---

## 2. Proposed native module layout

Contained in one directory; no C++ scattered through the Python package.

```
native/
  CMakeLists.txt              (optional; setuptools is the default path)
  include/soo/
    board.hpp                 73-position topology, neighbour[73][6], camps
    action.hpp                source/destination codec
    state.hpp                 State, occupancy, status, finish order
    rules.hpp                 step + chained-jump generation (BFS)
    encoder.hpp               canonical mapping, node features
    prior.hpp                 vacancy bootstrap prior
    tree.hpp                  arena Node/Edge
    mcts.hpp                  synchronous Soo PUCT
    batcher.hpp               global inference batcher + tickets
    selfplay.hpp              lane runner, episode accumulation
    rng.hpp                   RNG policy (see §9)
  src/*.cpp
  bindings.cpp                pybind11 module `diamond_native`
src/diamond/alphazero/native/
  __init__.py                 import guard + capability probe
  backend.py                  native backend behind the existing pool contract
  topology.py                 exports authoritative tables to the extension
tests/native/
  test_rules_parity.py        Phase 1 gate
  test_mcts_parity.py         Phase 2 gate
  fixtures/                   frozen position corpus
az-bench/profiles/            benchmarks stay out of production code
```

Backend selection is explicit and additive:

```
selfplay_backend = "python"   # default, reference oracle, unchanged
selfplay_backend = "native"   # opt-in, only after gates pass
```

---

## 3. C++ state and tree representations

### State

`GameState` carries `occupancy` (73 ints), `current_player_id`, `turn_number`,
`status`, `finish_order`. Compactly:

```cpp
struct State {
    std::array<uint8_t, 73> occupancy;   // 0 = EMPTY, else player id
    uint8_t  current_player;
    uint8_t  status;                     // mirrors GameStatus
    uint16_t turn_number;
    std::array<uint8_t, 3> finish_order; // Soo uses 2; 3 keeps Min portable
    uint8_t  finished_count;
};                                        // 80 bytes, trivially copyable
```

Topology is fixed and precomputed once:

```cpp
int8_t  neighbour[73][6];      // -1 for off-board
uint8_t camp_of[73];
uint8_t camp_positions[6][10];
uint8_t pairwise_distance[73][73];   // vacancy prior
uint8_t physical_to_canonical[6][73];
uint8_t canonical_to_physical[6][73];
```

These are **exported from Python at build or init time**, never hand-transcribed
(§9, risk 1).

Jump generation uses fixed-size storage, not `dict`/`set`/`deque`:

```cpp
uint8_t  queue[73]; uint8_t head, tail;
std::bitset<73> visited;
int16_t  best_destination[73];   // -1 = unreached; BFS keeps the first path
```

### Tree arena

```cpp
struct Edge {
    int32_t  action;        // canonical action id, ascending within a node
    int32_t  child;         // -1 until materialised
    float    prior;
    float    value_sum;
    uint32_t visits;
};

struct Node {
    State    state;
    uint32_t edge_begin;
    uint16_t edge_count;
    uint8_t  player_id;
    bool     expanded;
};

std::vector<Node> nodes;   // reserved to simulations + 2
std::vector<Edge> edges;   // reserved to simulations * mean_legal
```

One arena per lane, cleared between moves (no tree reuse — matching Python,
which constructs a fresh `MCTS2P` per move).

Deliberately kept batch-friendly for a possible future GPU-native experiment:
flat arrays, fixed topology tables, POD state, no pointer chasing.

---

## 4. Batch callback ABI

Two modes, because B0 discards the neural policy but later phases will not.

```cpp
enum class InferenceMode { ValueOnly, PolicyValue };
```

### ValueOnly — the B0 production path

```
C++  -> Python     float32 features[B][73][4]
Python -> C++      float32 values[B]
```

Nothing else crosses. Legal actions, the canonical encoding and the vacancy
prior are all native, so `legal_action_ids`, `legal_offsets` and `priors` are
absent from the boundary, and the entire policy tail (bounds check, padded index
build, gather, mask, softmax, validity sync, `[B, 5329]` D2H, priors dict,
response envelope) disappears. Measured worth: 1.33x at batch 32 (§0.4).

### PolicyValue — reference and post-bootstrap

```
C++  -> Python     float32 features[B][73][4]
                   int32   legal_action_ids[total_legal]
                   int32   legal_offsets[B + 1]
Python -> C++      float32 priors[total_legal]
                   float32 values[B]
```

Ragged legal sets travel as a flat array plus offsets rather than padded, so the
Python side can gather without materialising `[B, max_legal]`.

Buffers are caller-owned and reused per batch; the first prototype may copy.
Zero-copy (DLPack / buffer protocol views onto pinned staging buffers) is a
follow-up, not a prerequisite.

The mode is a property of the run, not of the build: both paths stay compiled and
tested, and the Python reference implementation continues to exercise
`PolicyValue`.

---

## 5. GIL and threading design

```
Python main thread
  native.run_selfplay(...)      -> releases the GIL for the whole run

C++ lane threads  (N, one per game)
  fully native; never touch Python; no GIL

C++ evaluator thread  (exactly one)
  collect batch -> acquire GIL -> one Python callback -> release GIL -> wake lanes
```

- The outer entry point holds `py::gil_scoped_release` for its entire body. This
  is the deadlock the design must not have: if the caller kept the GIL while the
  evaluator thread waited for it, the run would hang immediately.
- The evaluator thread uses `py::gil_scoped_acquire` around the callback only.
- Lane threads block on a per-ticket condition variable; no spinning.

Ticket model:

```cpp
Ticket t = batcher.submit(features_ptr, legal_ptr, legal_count);
const Result& r = t.wait();      // blocks this lane only
```

### Batching policy

Work-conserving, and the timer bounds only how long we wait for *future*
arrivals:

```
on first request: open window, deadline = now + max_wait_us
loop:
    drain everything already queued, up to max_batch_size
    if batch full -> dispatch
    if now >= deadline -> dispatch
    else wait on condvar until next arrival or deadline
before dispatch: drain the ready backlog again, up to max_batch_size
```

No hard minimum batch size: synchronous MCTS means at most one outstanding
request per lane, so a target batch larger than the live lane count would
deadlock. Exposed knobs: `max_batch_size`, `max_wait_us`, `lane_count`.

### Required tests

- **Deadlock guard.** A test that runs `run_selfplay` with a callback that
  itself touches Python objects, under a lane count > 1, with a watchdog that
  fails the test rather than hanging CI.
- **Single-lane tail.** One lane must still make progress at the full
  `max_wait_us`, since no second request can ever arrive.
- **Callback frequency.** Assert Python crossings are batch-scale: with N lanes
  the callback count must be close to `evaluations / mean_batch`, not to
  `evaluations`.

---

## 6. Build system

Environment probed on this host:

```
g++ 13.3.0 (C++20)    cmake 3.28.3    ninja 1.11.1
Python.h at /venv/main/include/python3.12/     numpy 2.5.2
-march=broadwell (AVX2)
pybind11 / nanobind / Cython : absent, installable
```

**Proposal: pybind11 + the existing setuptools backend.** The project already
builds with `setuptools.build_meta`; pybind11 is header-only and plugs in as a
normal `Extension`, so `pip install -e .` keeps working with no new build
ecosystem. nanobind produces smaller binaries and has nicer `ndarray` support,
but expects CMake/scikit-build-core, which would replace the project's build
backend — more disruption than the binding-layer difference is worth here.

Flags: `-O3 -march=broadwell -std=c++20 -fvisibility=hidden`. The extension is
**optional**: if it is absent or fails to import, `selfplay_backend = "python"`
must still work, and the test suite must pass without a compiler.

---

## 7. Parity test plan

Parity is the gate, so the corpus is a committed fixture and a permanent
regression test, not a one-off script.

### Corpus

Frozen to `tests/native/fixtures/positions.jsonl`, built from:

- standard opening positions
- states replayed from existing self-play ledgers (real trajectories)
- random legal trajectories at several depths
- jump-heavy positions (long chains, branch points)
- near-terminal and terminal positions, including partially-filled camps
- positions where a destination is reachable by both a step and a jump chain
  (the step-wins rule)

### Gate A — rules, encoding, prior

For every corpus state, native must equal Python exactly:

```
current player
terminal status, finish order
legal canonical action ids   -- AS AN ORDERED SEQUENCE, not a set   (see below)
for every legal action: resulting occupancy, current player, status, finish order
canonical player ids
node features                 (exact float equality; values are 0.0/1.0)
physical <-> canonical position and action mappings
vacancy bootstrap prior       (within 1e-12; it is a softmax over integer scores)
```

**Order matters.** `add_dirichlet_noise` draws one `gammavariate` per entry in
`priors.items()`, whose order is the order of `legal_action_ids`. A native
implementation that produces the same *set* in a different order changes the
noise assignment. Selection itself is order-independent (the PUCT key
tie-breaks on action id), but the corpus test must still compare sequences.

### Gate B — MCTS

Deterministic first: `dirichlet_epsilon = 0`, `temperature = 0`, a reproducible
evaluator that is a pure function of the request (not a constant — a constant
makes every edge tie and hides selection bugs). Compare:

```
selected action
visit count for every root action
q value for every root action        (within 1e-6)
the ordered sequence of evaluator requests
expanded legal action sets per node
```

Then re-enable noise and temperature under the RNG policy chosen in §9.

### Gate C — native throughput, no Python

Dummy native evaluator with configurable artificial latency; sweep lanes
1/2/4/8/16/32/64. Report searches/s, evaluations/s, batch distribution,
batches/s, per-lane wait, CPU utilisation. This decides whether the architecture
is worth integrating before PyTorch enters.

### Gate D — end-to-end

Same immutable checkpoint, seed, config as the Python backend. Report the full
A/B table from `rtx5060_bottleneck_findings.md` plus Python callback calls/s and
evaluations per callback.

---

## 8. Phase 1 implementation plan

Ordered by the measured stage profile (§0.1), which differs from the intuition
that MCTS selection dominates — it is 6.2 %. Note the order changed once the
duplicate legal generation was removed: the bootstrap prior, not legal move
generation, is now the largest single cost.

**Standing instruction for Phase 1: exact parity and a simple native
representation come before low-level micro-optimization.** The lane side has
large headroom against the batch-callback ceiling (§0.4), so a straightforward
native implementation is expected to suffice; optimizing it before Gate C would
be tuning against an unmeasured constraint.

0. **Topology export + native tables** (prerequisite, not a cost item).
   `topology.py` emits neighbours, camps, pairwise distances and canonical
   mappings; the extension consumes them. A test asserts the native tables equal
   the Python ones element-wise.
1. **Vacancy bootstrap prior** (**39.6 %**, the single largest cost once the
   duplicate legal generation is gone). Integer potential over a fixed 73×73
   distance table with at most 10 pieces and ~54 actions — no `frozenset` churn.
   Softmax in double, compared within 1e-12.
2. **Legal move generation** (**21.7 %**). Step generation, then chained-jump
   BFS with fixed-size queue/visited/best arrays, preserving direction order,
   shortest-path-wins and step-beats-jump. Must reproduce Python's **ordering**,
   not merely its set.
3. **`State` + `apply_action`** (7.6 %). Cheaper than the two above, but Gate A
   cannot close without it: the corpus test applies every legal action and
   compares the resulting occupancy, player and status.
4. **Canonical encoder** (4.2 %). Cheap, but needed for the ABI's feature
   buffer.
5. **Gate A corpus test.** Stop and report.

MCTS (Phase 2) deliberately comes after all of the above.

Commits stay reviewable: tables, then prior, then rules, then state/apply, then
encoder, then the parity gate — each with its own test.

---

## 9. Risks to preserving semantics

**1. Two authoritative rule implementations.** The main long-term risk, not a
technical one. Mitigations: topology tables generated from Python rather than
transcribed; the parity corpus committed and run in CI on every change; the
Python backend retained as the default and the oracle. If the jump-BFS canonical
path policy cannot be reproduced without fragile duplication, that is Gate A
failing and the project should stop and report.

**2. RNG parity.** `add_dirichlet_noise` calls `random.Random.gammavariate(0.3,
1.0)` once per legal action; for `alpha < 1` CPython uses a rejection algorithm
that consumes a *variable* number of `random()` draws, so bit-exact reproduction
means reimplementing that algorithm on MT19937, not merely seeding MT19937.
`select_from_visits` shares the same generator, so call ordering is coupled.

One simplification helps: `SooSelfPlayRunner` builds a **fresh `MCTS2P` per move**
with `seed = selfplay.seed + move_count`, so each move's stream is short and
independently seeded rather than one long coupled sequence.

**Decided (2026-08-21): cross-backend bit-exact RNG parity is NOT required.**
The policy is:

| mode | requirement |
|---|---|
| `dirichlet_epsilon = 0`, `temperature = 0` | **exact** Python/native MCTS parity — selected action, per-action visit counts, q values, evaluator request sequence |
| stochastic (`epsilon > 0` or `temperature > 0`) | each backend deterministic for a given seed; Dirichlet and temperature **distribution and semantics** must match; the *draw sequence* need not match Python |
| legal action ordering | **exact** parity, at Gate A, in every mode |

Legal action ordering stays exact because it is not an RNG question: it decides
which action each noise component lands on, and it is observable in the
deterministic mode as well.

Self-play exploration noise does not require a specific draw sequence for
learning validity, so reimplementing CPython's `gammavariate` rejection
algorithm on MT19937 buys reproducibility no experiment needs. If bit-exactness
is ever wanted, it stays achievable and should be scoped as its own task.

**3. Trajectory-dependent cost.** Per-call costs vary ~3x between opening and
full-game positions. Native microbenchmarks must use the full-game corpus or
they will overstate the port's benefit.

**4. `max_batch_size` will bind.** Native threads make hundreds of lanes cheap,
and the forward is flat in batch size, so the batch ceiling of 32 becomes the
constraint rather than the lane count. It is currently on the "do not touch"
list; Phase 3/4 benchmarks are expected to identify it, at which point it should
be promoted to a first-class knob.

**5. The batch callback becomes the next ceiling.** One evaluator thread holding
the GIL for the callback gives 6,375 evals/s at batch 32 today, 8,456 in
ValueOnly. A second evaluator thread does not help while the callback holds the
GIL; reducing kernel launches (CUDA graphs) would, and remains explicitly out of
scope for this project.

**6. ~~Duplicate legal generation~~ — fixed before Phase 1.** `_expand`
regenerated the authoritative legal set to validate the evaluator's prior keys,
so `legal_action_ids` ran **2.02 times per evaluation** at 667 µs/call — ~18 %
of worker search CPU. Both 2P and 3P now compare against
`request.legal_action_ids`, which is the same authoritative set the request was
built from. The invariant is preserved and arguably tighter: it asserts the
evaluator answered for exactly the actions it was asked about.

This landed as its own Python commit **before** any native work, so the Python
oracle baseline the native backend will be compared against does not carry the
duplicate. Re-measured: calls/eval 2.02 -> 1.02, worker CPU 4.405 -> 3.697
ms/eval (-16.1 %), throughput 669 -> 694 evals/s. Full table in §0.1.

It also reordered Phase 1: with the duplicate gone, the vacancy prior (39.6 %)
overtakes legal move generation (21.7 %) as the largest native target.

---

## 10. Go / no-go summary

| gate | question | stop condition |
|---|---|---|
| A | rules/encoding/prior parity | exact parity needs fragile duplication |
| B | deterministic MCTS parity | native cannot reproduce reference semantics |
| C | native + batcher throughput, no Python | no substantial headroom over 950 evals/s |
| D | end-to-end with PyTorch | callback recreates the ceiling, little gain |

Success is measured only as **correct self-play samples per wall-clock hour**.
Faster native microbenchmarks, lower parent CPU and higher GPU utilisation are
all explicitly insufficient — GPU utilisation in particular has already moved
*against* throughput once in this codebase.
