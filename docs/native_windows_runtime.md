# Native Windows runtime migration

Status: Gate Q2 model-deployment spike in progress, 2026-08-23

This document is the first implementation artifact for `blueprint/GUI_renewal.md`.
It records the current boundaries before any C++ GUI translation or model hand-copying.

## Repository state

- HEAD: `9c9454d`
- Working tree at audit start: `blueprint/GUI_renewal.md` is an untracked user-provided file.
- The current application is a Python package (`diamond-console`) using PySide6/QML.
- There is no repository CMake build yet.
- The native engine is compiled as an optional pybind11 extension by `setup.py`.
- Existing training and native self-play code remains in scope-protected Python/C++ paths.

## Current GUI → game → AI → model flow

```text
diamond.main
  ├─ QGuiApplication / QQmlApplicationEngine
  ├─ bundled fonts, icon provider, Windows chrome helpers
  ├─ GameController (context property: controller)
  └─ Main.qml and child QML components

GameController (Qt-facing state machine)
  ├─ GameSession
  │    ├─ GameState / Board / History
  │    └─ rules.validate_move / move application
  ├─ BoardModel, PieceModel, MoveHistoryModel, PlayerModel, BoardGeometry
  └─ AiWorker (one QThreadPool task)
       └─ Agent.choose_move(MoveRequest)
            ├─ RandomAgent, when DIAMOND_AGENT=random
            └─ AlphaZeroAgent
                 ├─ 2 players → MCTS2P / Soo semantics
                 ├─ 3 players → MCTS3P / Min semantics
                 ├─ DummyEvaluator + bootstrap prior if no learned evaluator
                 └─ Torch evaluator when a learned checkpoint is configured
                      └─ SooModel or MinModel
```

The QML layer does not calculate legality or mutate game state. It calls
`GameController` slots and renders controller properties and Qt list-model roles.
`AiWorker` carries a generation token; the controller drops stale proposals after
commit, undo, load, or new game. This generation/cancellation behavior is a
required parity contract for the native worker.

### QML/backend surface inventory

The current context properties are `controller`, `appFontFamily`, and
`nativeChrome`. The controller exposes:

- models: `boardModel`, `pieceModel`, `historyModel`, `playerModel`, `geometry`;
- turn/status: phase, turn number, current player identity/color, game label,
  status/error text, game-over/winner state;
- AI/proposal state: thinking state, agent name, proposal summary/details/path,
  rejected proposals, and confirmation controls;
- settings/session state: player count, turn order, AI seats, sound state and
  volume, save/load/new-game/undo controls;
- interaction slots: select position, confirm proposal, think again, undo,
  new match, save/load, and shutdown-related lifecycle behavior.

List-model roles are defined in `src/diamond/app/models.py`; notably the board
model publishes 73 holes with geometry, occupancy, selection, legal-target,
path, last-move, and proposal flags. The native models must preserve these roles
before QML is changed.

## Current native C++ dependency structure

The current native source of truth is under `native/`:

```text
diamond.alphazero.native._diamond_native (optional pybind11 module)
  ├─ bindings.cpp
  ├─ soo::board / rules / state / action
  ├─ soo::encoder
  ├─ soo::prior
  ├─ soo::mcts / tree
  ├─ soo::evaluator
  ├─ soo::batcher / selfplay
  └─ soo::profile / random
```

`setup.py` compiles the ten native translation units as one optional extension,
with C++20 and an AVX2-compatible default target (`broadwell`). There is no
Qt or LibTorch dependency in this native target. The C++ engine already owns
board topology, canonical encoding, rules, priors, and native MCTS/self-play;
the binding currently exposes these to Python and includes Python callback
bridges for evaluator work. Q1 must factor these same files into reusable CMake
libraries without creating a second engine or changing the setuptools path.

## Current Soo model/checkpoint structure

`src/diamond/alphazero/network/soo.py` defines the authoritative training model:

- input: `[B, 73, 4]` float features;
- graph trunk: input projection `4 → width`, six directional residual blocks by
  default, then output `LayerNorm`;
- each residual block: self projection, six direction-specific projections,
  directional adjacency aggregation, `LayerNorm`, `GELU`, residual add;
- policy: source and destination linear projections, scaled dot product, flatten
  to `73 * 73 = 5329` logits;
- value: mean node embedding, `Linear → GELU → Linear → Tanh`, output `[B, 1]`.

The default `NetworkConfig` is width 128 and six residual blocks. Checkpoints are
strictly validated by `CheckpointCompatibilitySpec`; compatibility includes
model identity/version, ruleset and fingerprint, topology, encoder, action-space
and seat-layout versions, value semantics, and network config. The checkpoint
stores a PyTorch `model_state_dict` plus training/optimizer metadata. This is a
training artifact, not yet a Windows runtime artifact.

## Risks and unknowns

1. The current default GUI is 3-player, while the first native human-play target
   is 2-player Soo. A native shell must not silently claim Soo parity for the
   existing default flow.
2. No LibTorch deployment path has been proven. A `.pt` checkpoint must not be
   assumed loadable by Windows LibTorch until the round-trip spike succeeds.
3. The C++ native MCTS currently has a reference evaluator interface, but no
   verified LibTorch-backed Soo evaluator.
4. Windows Qt/LibTorch availability and DLL packaging are unverified in this
   environment. Linux/WSL success cannot be reported as Windows success.
5. Native QML model registration, resource packaging, Windows non-client chrome,
   sound, and font loading have not been implemented.
6. The existing training path is a hard invariant: action indexing, canonical
   encoding, MCTS semantics, and checkpoint format must not change during GUI
   migration.

## Proposed CMake target graph

The first CMake refactor should preserve the current sources and expose narrow
libraries:

```text
soo_core
  ├─ board, state, action, rules, encoder, prior, random
soo_search
  └─ tree, mcts, evaluator, batcher, profile, selfplay
soo_pybind (optional MODULE)
  └─ links soo_search + soo_core
diamond_model (later, optional LibTorch)
  └─ model artifact loader + Soo forward/evaluator adapter
diamond_game (later)
  └─ Qt-independent session/controller domain types
diamond_qt (later, Qt6)
  └─ QML-facing models, controller, AI worker, resources, Windows adapters
alphadiamond (later, Windows executable)
  └─ diamond_qt + diamond_game + soo_search + diamond_model
```

`soo_core` and `soo_search` must remain buildable without Qt or LibTorch. The
pybind target must continue to be optional and must use the same C++ sources.

## Proposed model deployment spike (Q2)

Use a versioned portable data bundle, generated from the authoritative Python
checkpoint. Initial bundle metadata:

```json
{
  "format_version": 1,
  "model_name": "Soo",
  "model_version": "0.1.0",
  "input_features": 4,
  "board_size": 73,
  "policy_size": 5329,
  "width": 128,
  "residual_blocks": 6,
  "ruleset_fingerprint": "sha256:...",
  "encoder_version": "diamond-camp-relative-v1",
  "action_space_version": "diamond73-srcdst-v1",
  "training_step": 0,
  "tensors": "..."
}
```

The exact tensor container is deliberately undecided. The spike must compare
one deterministic corpus through:

```text
Python checkpoint → export bundle → C++/LibTorch loader → one forward pass
```

It must fail on missing/extra tensors, metadata mismatch, wrong shapes, or
unsupported versions, and compare encoded features, raw value, raw policy
logits, legal logits, and normalized legal priors with explicit tolerances.

## Staged implementation plan and gates

### Q0 — audit (this change)

- inventory the Python QML/backend contract;
- record current native ownership and model/checkpoint facts;
- identify scope contradictions and unverified Windows assumptions;
- define the CMake target graph and model spike contract.

Exit evidence: this document plus baseline tests.

### Q1 — CMake/native libraries

- add root/native CMake configuration;
- factor the existing native source list into `soo_core` and `soo_search`;
- keep `setup.py` compiling the same sources;
- add portable C++ tests/build checks.

Exit tests: existing Python tests, native parity tests, pybind import/build,
and CMake core/search build.

### Q2 — model deployment compatibility spike

- export a versioned Soo bundle from Python;
- load it in a small C++ LibTorch probe;
- compare the deterministic parity corpus;
- document any Windows-only step as unverified until run on Windows.

Exit tests: artifact schema/negative tests and Python/C++ numerical parity.

### Q3–Q6 — native Qt shell, controller parity, native human-vs-Soo, retirement

Preserve QML and migrate the controller surface incrementally. Keep the Python
GUI as an oracle until behavior, stale-result rejection, undo/load/new-game,
resources, and 2-player native Soo search have parity evidence. Only then make
the native executable primary and remove the Python GUI runtime dependency.

## Gate Q0 baseline

The current repository has no CMake or native Qt target to execute yet. The
baseline verification command is:

```text
python -m pytest
```

The next implementation step is Q1: factor the existing native engine into
reusable CMake targets while leaving Python training and the optional pybind
extension unchanged.

## Midpoint status — Q1 implementation

Completed in this work session:

- installed/verified CMake 4.4.2, Ninja 1.13.2, the conda-forge C++ compiler
  toolchain, and pybind11 in the existing `alphadiamond` environment at
  `C:\ProgramData\miniforge3\envs\alphadiamond`;
- added root and native CMake files with `soo_core` and `soo_search` static
  libraries;
- kept the existing `setup.py` source list untouched;
- added the `soo_native_smoke` CTest target for action-codec round trips;
- configured and built the native libraries with MSVC on Windows;
- configured and built the optional CMake pybind module from the same native
  sources;
- ran the Python suite successfully: **627 passed, 88 skipped**;
- ran CTest successfully: **1/1 native smoke test passed**.

The conda compiler activation emits a stale Visual Studio 2022 toolset probe
warning before falling back to the installed Visual Studio 2026 toolset
(`cl.exe` 19.51); compilation and linking complete successfully. This should
be cleaned up in environment/developer documentation, but it does not block
the Q1 build.

Q1 is functionally validated; the remaining pybind import-shell cleanup is a
developer-experience follow-up and does not block the Q2 spike. Q2 adds the
Python-checkpoint to portable-model artifact parity path.

## Q2 model-deployment spike status

Implemented:

- `tools/export_soo_deployment.py`, which constructs the authoritative
  `SooModel`, optionally loads a compatibility-checked checkpoint, emits a
  traced `model.ts`, strict `metadata.json`, and deterministic float32 parity
  corpus files;
- `diamond.alphazero.deployment`, a shared strict metadata validator with
  negative tests for missing, unknown, and mismatched fields;
- optional `soo_libtorch_probe` CMake target, linked against the LibTorch
  package shipped in the active environment;
- metadata checks for artifact format, model identity, dtype, width, and trunk
  depth before loading the graph;
- native forward comparison against Python-generated policy and value outputs.

Verified on Windows:

```text
Python torch: 2.13.0+cpu
LibTorch CMake package: found
Soo LibTorch parity passed
max_policy_error=0
max_value_error=0
```

Reproduce the spike from an activated `alphadiamond` environment:

```text
python tools/export_soo_deployment.py artifacts/soo-spike
cmake -S . -B build-libtorch -G Ninja ^
  -DDIAMOND_BUILD_LIBTORCH_PROBE=ON ^
  -DCMAKE_PREFIX_PATH=%CONDA_PREFIX%/Lib/site-packages/torch/share/cmake
cmake --build build-libtorch
build-libtorch/native/soo_libtorch_probe.exe artifacts/soo-spike
```

This proves the Windows LibTorch execution path for a platform-neutral
TorchScript artifact and a deterministic forward pass. The hand-authored
model and evaluator probes below extend the same artifact contract to native
MCTS.

The metadata validator tests currently pass **5/5**. The generated artifact is
ignored by Git and should be regenerated for each parity run.

## Hand-authored C++ Soo model parity

The LibTorch spike now includes `diamond_model::SooModel`, a C++ implementation
of the authoritative Python topology:

- 4-feature input projection over 73 nodes;
- six directional projections per residual block;
- six residual blocks with LayerNorm and GELU;
- source/destination scaled policy head with 5,329 logits;
- mean-node value head with Linear/GELU/Linear/Tanh;
- explicit float32 raw-weight loading with shape checks and a no-grad guard.

`soo_native_model_probe` loads the raw weights emitted by the Python exporter
and compares the hand-authored C++ forward pass with the Python corpus. The
Windows probe passed with:

```text
Soo native model parity passed
max_policy_error=0
max_value_error=0
```

This completes the model-parity task and supplies the evaluator implementation
used by the native MCTS integration below.

## Native evaluator and MCTS integration status

Implemented:

- `diamond_model::SooEvaluator`, which implements the existing
  `soo::Evaluator` contract and applies the authoritative legal-only softmax
  over canonical action ids;
- deterministic Python-exported topology and two-player initial-state fixture
  data for native search verification;
- `soo_evaluator_probe`, proving native legal-prior extraction against the
  Python corpus;
- `soo_mcts_probe`, which configures the native topology, runs `soo::MCTS2P`
  with the hand-authored evaluator, and checks the root legal-action sequence
  against Python's `DiamondSearchAdapter`.

Verified on Windows:

```text
Soo legal-prior parity passed; max_error=0
Soo native MCTS evaluator integration passed; root_actions=18, evaluator_calls=3
```

This closes the evaluator/MCTS parity task. The remaining implementation work
is the native Qt shell and controller/resource parity described in Q3–Q6.

## Q3 native Qt/QML shell status

Implemented the first Q3 slice:

- optional `DIAMOND_BUILD_QT` CMake target `diamond_qt`;
- existing QML bundled with `qt_add_resources` rather than loaded through
  Python package resources;
- native `QGuiApplication`/`QQmlApplicationEngine` entry point;
- typed C++ placeholder controller, geometry object, and list-model contract
  sufficient to launch the existing `Main.qml` without Python or PySide6;
- native `qta` image-provider compatibility shim, so preserved icon URLs no
  longer produce missing-provider diagnostics while the real icon renderer is
  ported;
- offscreen `diamond_qt --smoke` mode for resource/QML loading;
- Qt remains isolated from the portable `soo_core` and `soo_search` targets.

Verified on Windows with Qt 6 from the `alphadiamond` environment:

```text
diamond_qt.exe --smoke   # passed with QT_QPA_PLATFORM=offscreen
```

This is intentionally a shell/contract gate, not controller parity: the
native controller now owns the initial two-player state, legal move selection,
undo, match setup, save/load JSON, and model refreshes from the existing Soo
rules engine. The direct smoke executable passes; CTest currently exposes a Qt
teardown crash in the harness context and is deliberately not registered until
that harness issue is isolated.

## Q4 native controller and AI lifecycle

The QML contract is now backed by `NativeController` and `ContractListModel`:

- two- and three-seat match setup uses the native `soo::Match`/`soo::State`
  structures and the selected QML turn order/AI seats;
- legal source/destination selection and undo call the native rules engine;
- board, piece, and history models are rebuilt from native state;
- save/load uses a versioned JSON document containing match order, AI seats,
  occupancy, turn, and current player;
- AI results are generation-checked and legal-action-checked; new matches,
  undo, load, and controller destruction invalidate pending results.

The visual source remains the existing QML bundle. Native geometry mirrors the
PySide6 `BoardGeometry` contract (the same cube-coordinate projection, 73
holes, lattice edges, bounds, camp corners, and seat colors), and the native
`qta` provider supplies the existing icon names used by `Icon.qml`, so the
window chrome and board composition do not fork at the native boundary.

The worker cancellation mechanism is cooperative at the result boundary: a
running LibTorch/MCTS call is allowed to finish, but its result is discarded
after cancellation. This avoids use-after-free and stale-state commits while
preserving the current synchronous evaluator API.

## Q5 native Soo search and CPU configuration

The optional `DIAMOND_BUILD_QT_SOO=ON` target links the Qt shell to the
hand-authored LibTorch model/evaluator and `soo::MCTS2P`. It loads the exported
raw weights and runs the same native search path used after a human move.
Torch CPU thread counts are bounded to one intra-op and one inter-op thread by
default; set `DIAMOND_TORCH_THREADS` to override intra-op parallelism.

Verified on Windows:

```text
diamond_qt.exe --game-smoke   # passed, native shell with LibTorch
diamond_qt.exe --soo-smoke    # passed, model load + two MCTS simulations
```

The Qt/LibTorch build defines `QT_NO_KEYWORDS` and uses `Q_SIGNALS`/`Q_EMIT` to
avoid Qt's `slots` macro colliding with LibTorch's `IValue::slots()` API.

## Q6 final verification and compatibility notes

The implementation remains Python-free at runtime. The shell-only build works
without LibTorch; the Soo-enabled build requires the exported
`artifacts/soo-spike` weights/topology and the CPU LibTorch package. The native
controller resolves `artifacts/soo-spike` beside the executable first, then
falls back to the current working directory for developer builds.

Known non-blocking environment diagnostics are the missing optional Vulkan
headers, the Torch CMake warning about the optional `kineto` library, and MSVC
third-party-header narrowing warnings. The mamba activation script also probes
for an unavailable VS 2022 toolset before successfully falling back to the
installed VS 2026 14.51 toolset. These do not affect the passing native smoke
checks. Final packaging should run `windeployqt` and ship the Qt DLLs, the
LibTorch CPU DLLs when Soo search is enabled, and the exported artifact
directory together.

Use the included deployment script after activating `alphadiamond`. It copies
the Qt runtime explicitly because the conda-forge `windeployqt` executable can
resolve its prefix incorrectly on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\deploy_native_qt.ps1
powershell -ExecutionPolicy Bypass -File .\tools\deploy_native_qt.ps1 `
  -BuildDir build-qt-soo-clean -OutputDir dist\diamond-qt-soo -WithSoo
```

The second command also copies the CPU LibTorch DLLs and Soo artifacts.

For interactive launching, use `tools\run_native_qt.ps1`; it overrides any
inherited `QT_QPA_PLATFORM=offscreen` value left by smoke tests:

```powershell
Get-Process diamond_qt -ErrorAction SilentlyContinue | Stop-Process -Force
powershell -ExecutionPolicy Bypass -File .\tools\run_native_qt.ps1
# Soo build:
powershell -ExecutionPolicy Bypass -File .\tools\run_native_qt.ps1 -Soo
```
