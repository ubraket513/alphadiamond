# alphadiamond

3-player Diamond (Chinese-checkers-family) engine, Qt console, and an AlphaZero
training stack. The two-player configuration is called **Soo**.

## Working on the native self-play backend

Active project. **Read [docs/native_selfplay_handoff.md](docs/native_selfplay_handoff.md) first** —
it covers gate status, environment setup, how to run the benchmarks, the
invariants that must not break, and the pitfalls already paid for.

Two rules that are easy to get wrong and expensive to rediscover:

- **The Python implementation is the oracle and must not be deleted.** The C++
  in `native/` duplicates it on purpose; every gate is defined as equality
  against the Python side and CI re-runs that comparison on every change.
- **Do not micro-optimise before measuring.** The plan is explicit that a
  simple native representation and exact parity come first.

## Layout

| path | what |
|---|---|
| `src/diamond/game/` | authoritative rules, board, state — the oracle |
| `src/diamond/alphazero/` | encoder, MCTS, evaluators, orchestration |
| `src/diamond/alphazero/native/` | optional native backend: import guard, topology export, callback |
| `native/` | the C++ extension (`_diamond_native`) |
| `tests/native/` | parity gates A–D |
| `az-bench/profiles/` | benchmarks, out of production code |
| `docs/` | design records and measured findings |

## Three products

The native application, the Python trainer and the Python<->C++ bridge are
separate products with separate build and test commands. Read
[docs/architecture/products.md](docs/architecture/products.md) before moving
code between them -- it defines what the golden fixtures are for and the rule
for retiring a Python parity gate.

## Build and test

CMake is the build system and CTest runs the native tests; the Makefile is only
a command facade over the presets in `CMakePresets.json`.

```bash
make test-native                 # C++ core + tests; needs NO Python at all
make test-python                 # engine, AlphaZero, repository hygiene
make test-parity                 # bridge: Python <-> C++ gates (needs pybind)
make golden                      # regenerate tests/golden from the Python oracle
```

```bash
python tools/build_native.py     # optional extension; absence must stay harmless
pytest -m "not gui"              # engine + AlphaZero + native gates
```

## Shipping a model

```bash
python tools/export_deployment.py artifacts/soo-spike --family soo \
    --checkpoint runtime/runs/soo/<run>/latest.pt
python tools/build_model_index.py dist/models --artifact soo=artifacts/soo-spike
make package                     # CPack ZIP, models bundled beside the binary
```

Artifact format v3 declares the model family and architecture and is validated
against the weights on both sides; see
[docs/architecture/model_artifact_v3.md](docs/architecture/model_artifact_v3.md).
A training checkpoint (optimizer, scheduler, RNG) is not a deployment artifact
and is never bundled.

## Training data

Checkpoints, replay and logs live in the Hugging Face bucket, synchronised
through the ignored `TrainAlphaDiamond/` root; only its README and manifests
are tracked. See [TrainAlphaDiamond/README.md](TrainAlphaDiamond/README.md).
Nothing generated -- build directories, checkpoints, run output -- belongs in
Git; `tests/test_repo_hygiene.py` enforces that.
