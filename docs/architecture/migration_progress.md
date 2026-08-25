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
| 4 — remove duplicate runtime Python | **done** — the engine, the Python MCTS, the Python board and the oracle are deleted |
| 5 — measured additional ports | **first measurement done** |

## Milestone 0 — contracts and boundaries

* [`products.md`](products.md) records the three products, the golden contract
  and the rule for retiring a parity gate.
* [`model_artifact_v3.md`](model_artifact_v3.md) fixes the artifact schema.
* The golden corpus is frozen in `tests/golden/` and checked for provenance by
  `tests/test_golden_contract.py`. It is not regenerated: the generator was
  deleted with the Python engine (decision 3).

## Milestone 1 — native build and CI

* `CMakePresets.json`: `native-ci`, `native-release`, `native-asan`,
  `native-qt`, `pybind`. CMake remains the build system; the Makefile is a
  façade holding no source lists, compiler flags or platform branches.
* CI lanes: `native-core` (Linux/macOS/Windows, no Python), `native-sanitizers`
  (ASan/UBSan), `native-qt`, `core` (3.12), `bridge`, `lint`. The `gui` job is
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
  | 3P search gate (not in the brief's list) | done as `mcts3p_golden_test` |
  | `batcher_test` | done |
  | `scheduler_test` | folded into `selfplay_test` (lane trajectories across worker counts) |
  | `selfplay_test` | done |
  | `deployment_artifact_test` | done as `soo_artifact_contract`, plus `model_parity_test` and `model_index_test` |

  Folding was deliberate: a golden record per corpus position already carries
  the successors, the encoding and the prior, and splitting one reader across
  four executables would have bought file names rather than coverage.

* **All five Python parity gates are retired**, each on mutation evidence
  (`tools/mutation_check.py`, thirteen mutations, every one caught by the
  gate named for it). `bridge` now runs boundary tests only: the callback ABI,
  the pool, and the search paths Python still drives.

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

* **One canonical runtime representation, in the package.** The release ships
  `metadata.json`, the topology tables and `weights/` -- what the runtime loads,
  and exactly what `runtime_sha256` covers. The TorchScript graph and the parity
  corpus stay in the development artifact. Package: 11.5 MB to 6.0 MB.

* **Promotion is implemented.** `tools/promote_checkpoint.py` moves a bucket
  checkpoint archival -> candidate -> promoted, one step at a time, and
  promotion *converts*: it exports the deployment artifact and records its
  digests back into the manifest. It refuses a drifted checkpoint digest (an
  immutable path that was overwritten invalidates every measurement against
  it), refuses to skip the candidate gate, and refuses a `promoted` state whose
  artifact was never exported. Verified end to end on the step-80 checkpoint:
  promoted, converted, and the native model loads the result with zero error
  against PyTorch.

Left:

* Nothing outstanding in this milestone.

## Milestone 4 — remove duplicate runtime Python

**Done.** The C++ core is the only implementation of rules, encoding, search,
self-play and geometry. `src/diamond/game`, `src/diamond/alphazero/mcts`, the
Python board, `src/diamond/agents` and the two corpus generators are deleted;
[`retiring_the_python_engine.md`](retiring_the_python_engine.md) records the
order it happened in, the evidence collected before each deletion, and what can
no longer be done as a result.

What Python still owns, and always was going to: the training loop and
autograd, optimizer and checkpoint tooling, experiment orchestration, arena
statistics and rating, release tooling, and the Hugging Face synchronisation.

Both retirement ledgers reached zero, so `tests/test_engine_retirement.py` was
deleted with the thing it counted.

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

1. **Resolved.** The stochastic gate is fully replaced. Comparing the sampler
   against CPython's gamma was the weaker reference anyway: the C++ gate now
   runs a KS test against the *analytic* gamma CDF, which is what CPython is
   itself an implementation of. Proven by a mutation moments alone cannot see
   -- an exponential draw with the right mean, which the KS check rejects.
2. **The self-play runners belong on the pool, not the bridge.** Both native
   sessions now take a wall-clock budget, so the deadline is no longer what
   holds them back. What does is batching: self-play wants many games in flight
   so one evaluator call answers a batch, and the per-node bridge gives that up
   to gain about 2x on the tree
   ([measured](../performance-profiling/bridge_search_findings.md)). The native
   pool already batches; that is where they go.
3. **Resolved.** Min self-play runs on the native pool. `EpisodeSearch` holds
   either session behind one handle -- the lanes, batching, job queue and move
   recording were always seat-agnostic, and naming `SearchSession` directly was
   the only thing that made the pool two-player. The callback ABI carries a
   value per seat, in both `value_only` and `policy_value` modes, and the pool
   builds Min's placement target ordered by `canonical_player_ids` rather than
   by seat id -- the order the value head is trained in.
4. **The `apps/` / `python/` restructure from section 4 of the brief has not
   been done.** It rewrites hundreds of documented command paths and deserves
   its own change; the boundaries it would express are already established in
   `products.md`.
5. **`native/qt/qml` and `native/qt/assets` belong to the native
   application**, not to the Python package that excludes them. They are the
   obvious first move if the restructure happens.
6. **Branch protection** is on `main` with the CI checks required,
   `enforce_admins: false` and no required reviews. Remove with
   `gh api -X DELETE repos/ubraket513/alphadiamond/branches/main/protection`.

## How to pick this up

```bash
make test-native      # the game: rules, search, self-play, geometry. No Python.
make test-python      # the trainer's Python: training, orchestration, hygiene
make test-parity      # the pybind boundary (needs the pybind build)
make golden-freeze    # re-record tests/golden provenance (contract change only)
```

The migration's own work is finished. What is left is the ordinary backlog:

1. **CI does not fetch fixtures from the bucket** (milestone 2). The step-80
   checkpoint is still tracked as the one documented exception.
2. **History still contains the removed payloads.** A `git filter-repo` rewrite
   is a separate, announced operation because it changes every commit hash.
3. **The `apps/` / `python/` restructure** from section 4 of the brief. It
   rewrites hundreds of documented command paths and deserves its own change;
   `native/qt/qml` and `native/qt/assets` belong to the native application
   and are the obvious first move.
4. **Measured ports, when something needs them.**
   `PersistentReplayStore.load_buffer()` is called once per training step; on a
   large store that is a re-read per step. Nothing else has a measurement, so
   nothing else has a case.
