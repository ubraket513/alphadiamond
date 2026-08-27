# alphadiamond

A 3-player Diamond (Chinese-checkers-family) engine, a Qt console, and an
AlphaZero training stack. The two-player configuration is called **Soo**; the
three-player one is **Min**.

**This is a C++ project.** There is no Python in it: no engine, no trainer, no
bridge, no tests, no tooling. Rules, encoding, search, self-play, training,
checkpointing, arena and release are all native. Anything you read elsewhere
about a Python oracle, a `_diamond_native` pybind extension, `src/diamond/`, a
`selfplay_backend` setting or a `make test-python` / `make test-parity` /
`pytest` step describes a state this repository has left. See
[docs/architecture/decisions.md](docs/architecture/decisions.md) and
[docs/architecture/retiring_the_python_engine.md](docs/architecture/retiring_the_python_engine.md)
for how and why.

## Rules that are expensive to rediscover

- **The frozen fixtures are the oracle, except for replay.** `tests/golden/`
  and `tests/native/fixtures/` are normative and **cannot be regenerated** —
  the implementation that produced them is gone. Settle contract questions by
  reading them, never by reasoning about what the C++ currently does. The one
  exception is `tests/golden/replay-v1/`: `ReplayStore` is schema-4-only and no
  longer reads schema 1/2/3 at all, so that corpus is unreadable by this tree
  and the replay tests build their stores natively instead. It is kept only as
  a historical record.
- **Game geometry has exactly one authority.** Seats come from
  `soo::standard_soo_match()` and `soo::standard_min_match()` in
  `native/src/board.cpp`, never from triples written at a call site. Five
  hand-written copies drifted once: the trainer played a different game from the
  Qt application, and Soo self-play completed 77 % of games instead of 98 % for
  as long as it lasted. `match_geometry_test` pins the factory to
  `tests/golden/rules-v1.txt`. See
  [docs/model-training/baseline_768_completion_regression.md](docs/model-training/baseline_768_completion_regression.md).
- **Do not micro-optimise before measuring.** A simple native representation and
  exact fixture parity come first.
- **Measure at a scale that can answer the question.** Self-play completion
  varies enough run to run that sixteen games cannot distinguish two
  configurations; 768 games is the harness the findings in `docs/model-training/`
  use, and it costs about three minutes on one RTX 5090.

## Layout

| path | what |
|---|---|
| `native/include/soo/`, `native/src/` | the core: board, rules, encoder, MCTS, self-play, training, checkpoint, arena |
| `native/qt/` | the Qt application |
| `native/tests/` | the test suite, run by CTest |
| `native/benchmarks/` | benchmarks, kept out of production code |
| `tests/golden/` | frozen normative fixtures: rules, MCTS, topology |
| `tests/native/fixtures/` | frozen normative fixtures from the retired parity gates |
| `configs/alphazero/` | training configurations |
| `models/` | shipped deployment artifacts (format v3) |
| `docs/` | design records and measured findings |

## Build and test

CMake is the build system and CTest runs the tests; the Makefile is a command
facade over the presets in `CMakePresets.json` (`native-ci`, `native-training`,
`native-release`, `native-qt`, `native-asan`, `native-package`).

```bash
make test-native                 # core contract tests, no LibTorch needed
make test-training               # LibTorch training tests
make test-qt                     # Qt + LibTorch application tests
make package                     # CPack ZIP, models bundled beside the binary
```

The training preset needs LibTorch. Point CMake at it and build directly when
you need a specific target:

```bash
tools/native_training.sh cmake --preset native-training -G Ninja \
    -DCMAKE_PREFIX_PATH=<libtorch-root>
tools/native_training.sh cmake --build --preset native-training --parallel
tools/native_training.sh ctest --preset native-training --output-on-failure
```

`tools/native_training.sh` wraps `cmake`/`ctest` so the Visual Studio
environment is picked up on Windows; on Linux and macOS it is a passthrough.

## Commands

```bash
alphadiamond-train train  --run-dir <dir> --config <config.json> \
                          (--scratch | --checkpoint <dir> | --warm-start <artifact>)
alphadiamond-train resume --run-dir <dir>
alphadiamond-train report --run-dir <dir>
alphadiamond-checkpoint   inspect|validate|migrate <checkpoint-root>
alphadiamond-release      ...
```

A run directory is durable and idempotent: every stage is ledgered by operation
id, `resume` replays exactly the work that was not completed, and re-running a
finished stage is a no-op. `--run-dir` must sit under a `soo/` or `min/` parent.

Each iteration also writes two diagnostic sidecars that nothing reads back and
no gate depends on: `selfplay.metrics.json` (search budget actually in force,
boosted-move count, throughput, batch distribution, abort reasons,
completed-game move percentiles) and `aborted-games.json` (the final position of
every aborted game, with target-camp occupancy and blocker mobility).

## Shipping a model

```bash
make package                     # CPack ZIP, models bundled beside the binary
```

Artifact format v3 declares the model family and architecture and is validated
against the weights on both sides; see
[docs/architecture/model_artifact_v3.md](docs/architecture/model_artifact_v3.md).
A training checkpoint carries optimizer, scheduler and RNG state, is not a
deployment artifact, and is never bundled.

**There is currently no way in this tree to turn a training checkpoint into a
deployment artifact.** That exporter was Python and went with the rest of it;
`alphadiamond-checkpoint` accepts only native checkpoint-v2/v3 roots and refuses
a raw `.pt`. Converting an archival `latest.pt` — such as the assets on the
`soo-v2.0.0-rc.1` release — currently requires restoring the deleted exporter
from history outside this tree. Rebuilding that path natively is open work.

## Training data

Checkpoints, replay and logs live in the Hugging Face bucket, synchronised
through the ignored `TrainAlphaDiamond/` root; only its README and manifests are
tracked. See [TrainAlphaDiamond/README.md](TrainAlphaDiamond/README.md). Nothing
generated — build directories, checkpoints, run output — belongs in Git.
