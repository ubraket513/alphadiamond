# Migration progress

Tracks the brief in [`migrate-and-cleanup.md`](../../migrate-and-cleanup.md) at
the repository root. That file is the approved plan; this one records what has
been done against it, what is deliberately different, and what is left — so
that someone else can pick it up without reading the whole history.

Last updated 2026-08-24.

## Status by milestone

| Milestone | State |
|---|---|
| 0 — contracts and boundaries | **done** |
| 1 — native build and CI | **done** |
| 2 — storage and cleanup | **mostly done** (fixture fetch and history rewrite left) |
| 3 — release-grade model artifacts | **done** |
| 4 — remove duplicate runtime Python | **in progress** |
| 5 — measured additional ports | **first measurement done** |

## Milestone 0 — contracts and boundaries

* [`products.md`](products.md) records the three products, the golden contract
  and the rule for retiring a parity gate.
* [`model_artifact_v3.md`](model_artifact_v3.md) fixes the artifact schema.
* The golden corpus and its generator version are frozen in `tests/golden/`,
  regenerated and checked on every CI run by `tests/test_golden_is_current.py`.

## Milestone 1 — native build and CI

* `CMakePresets.json`: `native-ci`, `native-release`, `native-asan`,
  `native-qt`, `pybind`. CMake remains the build system; the Makefile is a
  façade holding no source lists, compiler flags or platform branches.
* CI lanes: `native-core` (Linux/macOS/Windows, no Python), `native-sanitizers`
  (ASan/UBSan), `native-qt`, `core`, `bridge-parity`, `lint`. The `gui` job is
  gone — its PySide suite was deleted upstream and it collected nothing.
* Native test groups, against the brief's list:

  | Brief's name | State |
  |---|---|
  | `board_test` | folded into `topology_test` — the board *is* the topology tables here |
  | `topology_test` | done |
  | `action_codec_test` | done |
  | `rules_golden_test` | done |
  | `state_transition_golden_test` | folded into `rules_golden_test` (successor of every legal action) |
  | `encoder_golden_test` | folded into `rules_golden_test` (encoding digest per position) |
  | `prior_golden_test` | folded into `rules_golden_test` (prior max and dot product) |
  | `mcts_deterministic_golden_test` | done as `mcts_golden_test` |
  | `mcts_stochastic_test` | done |
  | `batcher_test` | done |
  | `scheduler_test` | folded into `selfplay_test` (lane trajectories across worker counts) |
  | `selfplay_test` | done |
  | `deployment_artifact_test` | done as `soo_artifact_contract`, plus `model_parity_test` and `model_index_test` |

  Folding was deliberate: a golden record per corpus position already carries
  the successors, the encoding and the prior, and splitting one reader across
  four executables would have bought file names rather than coverage.

* No Python parity test has been deleted. `bridge-parity` runs them all.

## Milestone 2 — storage and cleanup

Done:

* `build-pybind/` (59 tracked files of object files, PDBs and Ninja state) and
  two duplicate 9.1 MB checkpoints untracked.
* `.gitignore` covers build directories, run output and weights by extension.
* `TrainAlphaDiamond/` is the ignored bucket sync root; only its README and
  `manifests/` are tracked. Directional `make data-push` / `data-pull` with
  `-dry-run` variants and no `--delete` anywhere.
* `tests/test_repo_hygiene.py` blocks generated payloads, undocumented weights
  and unexpected large files from returning.
* `manifests/ci-fixtures.json` names each large fixture by digest, size and
  bucket path; the hygiene test derives its allow-list from it.

Left:

* **CI does not yet fetch fixtures from the bucket.** The step-80 checkpoint is
  still tracked, as the one documented exception. When the bucket is populated:
  fetch by path and digest, drop `tracked_in_git` from the manifest entry,
  `git rm` the file.
* **History still contains the removed payloads.** A `git filter-repo` rewrite
  is a separate, announced operation because it changes every commit hash.

## Milestone 3 — release-grade model artifacts

* Artifact format v3: declared `model_family`, `architecture`, `game_contract`,
  `tensor_shapes` and `source`; the weight manifest is validated against the
  declaration rather than against constants.
* `tools/export_deployment.py --family soo|min` replaces the Soo-only exporter
  and records the training commit and step.
* Min native inference: `DiamondModel` takes input features and value width
  from the artifact. `model_parity_test` runs each artifact's own corpus
  through the native model and compares with PyTorch's output — both families
  match exactly.
* Packaging: `tools/build_model_index.py` writes `models/index.json`; the Qt
  runtime prefers the packaged index; CPack ZIP with a SHA-256 checksum.

Left:

* **One canonical runtime representation.** v3 still ships both a TorchScript
  graph and raw weights, so every model is stored twice. Decide which the
  runtime consumes and keep the other only as a conversion fixture.
* **Promotion policy is documented, not implemented.** Nothing yet marks a
  checkpoint `candidate` or `promoted`, or drives conversion from that.

## Milestone 4 — remove duplicate runtime Python

The C++ core is the authority. Progress and the remaining order live in
[`retiring_the_python_engine.md`](retiring_the_python_engine.md):

```text
native self-play default            done (selfplay_backend defaults to auto)
dependent ratchet                   done (tests/test_engine_retirement.py)
arena on the native core            done (Soo and Min)
GUI agent on native                 done (both seat counts)
three-player native search          done (SearchSession3P)
native wall-clock deadline          done (set_budget on both sessions)
self-play runners on the pool       next: batching, not the per-node bridge
freeze the golden corpus            then
retire the Python parity gates      one at a time, each with its C++ replacement
delete src/diamond/game             last
```

`tests/test_engine_retirement.py` freezes the dependent set: adding a module
fails, and removing one requires deleting its line. That list is the work queue
for the rest of this milestone. It grew by one -- `search_factory.py`, whose
whole job is to choose between the engines -- and everything that searches now
goes through it.

`tests/native/test_search_factory.py` proves the choice is real rather than
nominal, including end to end: the agent's two-seat move actually constructs
`NativeSearch2P`. `arena.py` has already left the list -- both its searches now
go through the selector.

The three-player search is `SearchSession3P`
(`native/src/mcts3p.cpp`). It is not the 2P tree with a wider value: Min backs a
utility vector -- one component per seat, placement-ordered 1/0/-1 at a terminal
-- through every ancestor unchanged, where Soo negates a scalar once per edge.
`tests/native/test_search_3p_parity.py` compares it with `MCTS3P` per position
on the selected action, the visit distribution, the evaluator request sequence
and every q vector *by seat*, with an evaluator that is deliberately asymmetric
between seats: symmetric values would hide a search maximising the wrong
component.

Nothing here rewrites the training loop, the optimizer tooling, the experiment
orchestration or the rating code. The brief is explicit that those stay in
Python, and they do.

## Milestone 5 — measured additional ports

* The per-node bridge measured before extending it
  ([`bridge_search_findings.md`](../performance-profiling/bridge_search_findings.md)):
  native + callback beats the Python search by 1.7-2.2x even against an
  evaluator that returns a constant. The first unwarmed measurement said the
  opposite -- 3.5x *slower* -- because the first native search in a process pays
  the extension's one-time setup. That number would have justified abandoning
  the bridge; warming up first is the difference between a finding and a wrong
  decision.
* Replay path measured before touching it
  ([`replay_pipeline_findings.md`](../performance-profiling/replay_pipeline_findings.md)):
  ingestion is 2.98 µs/sample and needs nothing; the batch path cost ~310 ms
  per training step. `AlphaZeroTrainer.train_samples` scatters the sparse
  policy straight into a tensor: 13.6 ms, identical values, 22.8×. **The port
  target was torch, not C++** — a C++ rewrite of the same dense strategy would
  have optimised the wrong thing.

Left, and deliberately unmeasured until someone needs them:

* `PersistentReplayStore.load_buffer()` is called once per training step; on a
  large store that is a re-read per step.
* Arena statistics, dataset transformations: no measurement, so no port.

## Known gaps and open questions

1. **The Python stochastic gate is not fully replaceable.**
   `native/tests/mcts_stochastic_test.cpp` now covers the distribution half
   with no interpreter -- gamma moments (the boost branch production actually
   uses), weighted-choice frequencies, the Dirichlet mixture, temperature
   behaviour, and reproducibility from a seed. Verified by mutation: inverting
   the gamma boost exponent moves the mean from 0.30 to 1.00 and the test goes
   red.

   What stays in `tests/native/test_stochastic_parity.py` is the part that
   needs CPython by definition -- the gamma CDF compared against
   `random.Random` itself. That one is a bridge test and retires with the
   bridge, not before.
2. **The self-play runners belong on the pool, not the bridge.** Both native
   sessions now take a wall-clock budget, so the deadline is no longer what
   holds them back. What does is batching: self-play wants many games in flight
   so one evaluator call answers a batch, and the per-node bridge gives that up
   to gain about 2x on the tree
   ([measured](../performance-profiling/bridge_search_findings.md)). The native
   pool already batches; that is where they go.
3. **Min self-play still runs the Python pool.** `SearchSession3P` is the
   single-search shape; the batched scheduler and `play_episodes` are 2P only,
   so native Min self-play needs the vector value threaded through the pool as
   well.
4. **The `apps/` / `python/` restructure from section 4 of the brief has not
   been done.** It rewrites hundreds of documented command paths and deserves
   its own change; the boundaries it would express are already established in
   `products.md`.
5. **`src/diamond/qml` and `src/diamond/assets` belong to the native
   application**, not to the Python package that excludes them. They are the
   obvious first move if the restructure happens.
6. **Branch protection** is on `main` with the CI checks required,
   `enforce_admins: false` and no required reviews. Remove with
   `gh api -X DELETE repos/ubraket513/alphadiamond/branches/main/protection`.

## How to pick this up

```bash
make test-native      # C++ core and its gates; no Python at all
make test-python      # engine, AlphaZero, hygiene, ratchet
make test-parity      # bridge: Python <-> C++ (needs the pybind build)
make golden           # regenerate tests/golden from the Python oracle
```

The next concrete task is a three-player native search (`MCTS3P`'s
counterpart). It is what unblocks `MinArena`, `runner_3p` and the agent's
three-seat path in one go -- and after it, a wall-clock deadline in
`SearchSession` unblocks `runner_2p`. Each one ends with a line struck from
`ALLOWED` in `tests/test_engine_retirement.py`; that deletion is the unit of
progress.
