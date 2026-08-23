# Three products, one core

Written 2026-08-24. This is the boundary the native migration is organised
around. It is not a plan to rewrite Python in C++.

| Product | What it is | Language |
|---|---|---|
| **Native application** | The shipped Qt program: board, search, inference. Runs with no Python installed. | C++ |
| **Python trainer** | Training loop, experiments, arena statistics, reports, release tooling. | Python |
| **Bridge** | The pybind extension and the cross-language parity gates. Exists for the migration and for feeding the native core to the trainer. | both |

The core rules, encoder, search and self-play are one implementation in C++.
Python reaches them through the bridge. The Python `src/diamond/game/`
implementation stays as the oracle until the native golden tests have taken
over each contract it currently proves, and is deleted only then.

```text
          C++ core (rules, encoder, MCTS, self-play, inference)
                     |                         |
        native Qt application        pybind module (bridge)
                                               |
                                     Python training / research
```

## What "the C++ tests do not need Python" means

`make test-native` configures with CMake, builds and runs CTest on a clean
checkout with no interpreter, no pytest and no pybind11. The tests get their
board topology and their expected answers from committed fixtures under
`tests/golden/`, not from a running Python process.

## The golden contract

A C++ test that calls one C++ function and compares it with another C++
function proves nothing about the port. So the Python oracle's answers are
frozen into a language-neutral file:

* `tools/build_golden.py` regenerates `tests/golden/` from the Python oracle
  and the frozen corpus in `tests/native/fixtures/positions.jsonl`.
* `tests/golden/topology/*` are the exported board tables — the same five files
  the deployment artifact ships. Nothing in `native/` transcribes them.
* `tests/golden/rules-v1.txt` holds, per corpus position, the legal actions
  (ordered), every successor state, the canonical encoding and the bootstrap
  prior — folded into FNV-1a 64 digests, except the prior, which is floating
  point and is compared as max and an order-sensitive dot product.
* `tests/test_golden_is_current.py` regenerates the file in CI and fails if it
  no longer matches the oracle. That is what keeps the frozen answers honest.

### What each native test covers

| Test | Proves |
|---|---|
| `action_codec_test` | every source/destination pair round-trips; out-of-range ids throw |
| `topology_test` | loader rejects malformed input; neighbour symmetry, camp structure, distance metric, canonical rotation is a distance-preserving bijection |
| `rules_golden_test` | Gate A: ordered legal actions, every successor, the encoding, the prior |
| `mcts_golden_test` | Gate B: root statistics *and* the evaluator request sequence, q values bit-exact |
| `batcher_test` | no minimum batch, batch cap, arrival order, `stop()` drains |
| `selfplay_test` | lane trajectories do not depend on the worker count; episodes reproduce from a job list |
| `model_index_test` | a packaged `models/index.json` is refused when a default names no bundled model, a path escapes the package, or a digest is malformed |

With LibTorch (`-DDIAMOND_BUILD_LIBTORCH_PROBE=ON`, artifacts exported first):

| Test | Proves |
|---|---|
| `soo_artifact_contract` | v3 metadata validation: missing/unknown fields at every nesting level, a family declared over the wrong shapes, an architecture that no longer matches the tensors, a foreign game contract, corrupt or missing weights |
| `model_parity_test` | native inference reproduces PyTorch's own outputs on the artifact's corpus, per family |

Ordering is part of the contract, not cosmetic: `add_dirichlet_noise` draws one
`gammavariate` per prior entry, so the same set in a different order is a
different search.

## Retiring a Python parity gate

The Python gates in `tests/native/` are not deleted when a C++ test covering
the same ground is written. A gate is retired only after its C++ replacement
has been shown to catch the mutations that gate catches — the minimum
demonstration is: mutate the golden expectation (or the native code) and watch
the C++ test go red. Until then, `bridge-parity` keeps running in CI.

## CI lanes

| Lane | Needs Python? | What it proves |
|---|---|---|
| `native-core` (Linux/macOS/Windows) | no | the shipped core builds and its tests pass |
| `native-sanitizers` (Linux) | no | the same tests under ASan/UBSan |
| `core` | yes | Python engine, AlphaZero stack, repository hygiene |
| `bridge-parity` | yes | Python and C++ still play the same game |
| `gui` | yes | the PySide app |
| `lint` | yes | ruff over changed files |

The native Qt lane is deliberately absent: `diamond_qt_controller_contract`
fails on its telemetry assertion at `main`, and a job that is red on arrival
trains people to ignore CI. Add the lane in the same change that fixes it.
