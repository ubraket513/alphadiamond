# Native Windows Qt runtime

Status: GUI renewal gates Q0–Q6 complete, 2026-08-23.

The primary Windows GUI is now a native C++/Qt 6 application. It embeds and
loads the existing QML, runs the authoritative native rules and MCTS, and uses
LibTorch CPU inference for the two-player Soo seat. The deployed executable
does not import Python or start a Python/WSL process. Python remains the
authoritative training/export environment. The old PySide6 host, its GUI tests,
dependencies, package data, and command entry point were removed after native
controller/package parity was established; shared QML and assets remain the
native application's visual source.

## Gate record

| Gate | Result | Evidence |
|---|---|---|
| Q0 inventory | Pass | GUI/QML/native/model path inventoried in the blueprint and this document |
| Q1 reusable native libraries | Pass | `soo_core`, `soo_search`, `diamond_model`, and `diamond_qt_backend`; app/test do not copy controller sources |
| Q2 deployment spike | Pass | versioned artifact, strict schema/hash/tensor-manifest validation, Windows LibTorch load, policy/value and legal-prior parity probes |
| Q3 native Qt shell | Pass | existing QML and fonts embedded in resources; native icon/chrome/Snap Layout integration; no Python runtime |
| Q4 controller parity | Pass | proposal/confirm/cancel, path, per-hop animation/sound, history, undo, save/load, new game, terminal state, model roles, illegal-action rejection |
| Q5 human-vs-Soo | Pass | native `MCTS2P` + LibTorch evaluator on worker thread; deterministic settings; canonical action converted to physical seat coordinates |
| Q6 native primary | Pass | PySide6/QtAwesome dependencies, Python GUI host, GUI-only tests, Python GUI package data, and legacy entry point removed; training/export dependencies preserved |

## Runtime architecture

```text
existing QML (embedded Qt resources)
  -> NativeController / Qt list models
  -> soo_core rules and state
  -> NativeAiWorker
  -> MCTS2P
  -> SooEvaluator / SooModel
  -> LibTorch CPU
```

`diamond_qt_backend` is the shared Qt backend library linked by both
`diamond_qt.exe` and `diamond_qt_controller_contract`. `native_chrome.cpp`
stays in the executable host because it owns the HWND/DWM boundary.

## Platform boundary

The released GUI targets Windows 10/11 x64 and CPU LibTorch. It assumes neither
CUDA nor Intel XPU support; XPU/OpenVINO remains an optional future backend, not
a runtime dependency. Python/PyTorch may train and export on Windows or WSL,
but only the versioned deployment artifact crosses into the Windows package.
The executable never launches Python or WSL.

## Build and package

Activate the requested environment and enter a Visual Studio developer shell:

```powershell
mamba activate C:\ProgramData\miniforge3\envs\alphadiamond
& 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1' `
  -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
```

Configure and build the release Soo application:

```powershell
cmake -S . -B build-qt-soo-clean -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DDIAMOND_BUILD_QT_SOO=ON `
  -DDIAMOND_BUILD_LIBTORCH_PROBE=ON
cmake --build build-qt-soo-clean --parallel 1
```

The shell-only build, useful for controller work without LibTorch, is:

```powershell
cmake -S . -B build-qt-clean -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DDIAMOND_BUILD_QT=ON
cmake --build build-qt-clean --parallel 1
```

Export the model artifact the application loads. This step is not optional for
a Soo build: `deploy_native_qt.ps1 -WithSoo` fails outright if
`artifacts/soo-spike` is absent, and the directory is generated rather than
tracked, so it does not arrive with a clone.

```powershell
python tools\export_deployment.py artifacts\soo-spike `
  --checkpoint runtime\runs\soo\soo-scratch-20260822\latest.pt
```

`soo-scratch-20260822/latest.pt` is the A0 checkpoint at training step 44,250 --
the strongest network the project has produced, and the one the GUI should play
humans with. It is not the same file as
`runtime/runs/soo/cpu8h-soo-20260819/latest.pt`, which stays in the repository
untouched: that one is the immutable step-80 checkpoint every native gate
measurement is defined against, and a test asserts its SHA-256. Exporting from
it would produce a working artifact that plays like an untrained network.

Omitting `--checkpoint` also produces a valid artifact, from *random* weights.
That is useful for shell and packaging work and useless for playing, so pass
the flag whenever the build is meant to play.

Verify the artifact against the native contract before packaging:

```powershell
ctest --test-dir .\build-qt-soo-clean -R "artifact_contract|model_parity"
```

`model_parity_test` replaced `soo_native_model_probe.exe`: same comparison,
registered as a test and covering every bundled family rather than Soo alone.

Create a self-contained package:

```powershell
.\tools\deploy_native_qt.ps1 `
  -BuildDir build-qt-soo-clean `
  -OutputDir dist\diamond-qt-soo `
  -WithSoo `
  -EnvironmentRoot $env:CONDA_PREFIX
```

The deployment step copies Qt DLLs, plugins and QML imports; Visual C++
runtimes; LibTorch and its protobuf/UTF-8/Abseil dependency closure; the Soo
artifact; and `assets/sounds/move.m4a`. It then runs package-local shell,
engine, worker, persistent-failure, media-decode, and Soo smokes and fails if a
DLL/plugin/artifact is missing. The shell-only package contains topology data
but does not carry the unused model or weight files. Smokes run with conda,
Python, and external Qt/QML paths removed; only the package and Windows system
directories are available for DLL resolution.
Only runtime plugin families are copied; Designer/QML tooling plugins are
excluded, and deployment fails if a Python DLL, PySide plugin, or
`site-packages` path appears in the native bundle.

Run the application:

```powershell
.\tools\run_native_qt.ps1 -Soo
.\tools\run_native_qt.ps1 -Soo -Simulations 256
```

The launcher forces the Windows QPA plugin so a stale
`QT_QPA_PLATFORM=offscreen` cannot make an interactive launch invisible.

## Search configuration and measured latency

Human play uses temperature zero, no root Dirichlet noise, deterministic visit
selection, 128 simulations by default, one Torch intra-op thread, and one
interop thread. Override with:

```powershell
$env:DIAMOND_MCTS_SIMULATIONS = '256'
$env:DIAMOND_TORCH_THREADS = '2'
```

Benchmark command:

```powershell
.\build-qt-soo-clean\native\soo_mcts_probe.exe `
  .\artifacts\soo-spike 128 30 1
```

Measured on this 8-logical-CPU Windows host, Release build, B1 inference,
sequential runs, one Torch thread, and 30 untraced timed moves per setting. A
separate traced search is run before timing to prove root-action/prior parity.
`raw forward` is the network call alone, `evaluator` adds feature cloning and
legal-action prior extraction, `MCTS` uses a preloaded evaluator, and `GUI
proposal` matches the current warm-cache GUI path by validating the artifact,
constructing/loading the model, and completing MCTS for every request.

| simulations | raw forward p50/p95 | evaluator p50/p95 | MCTS p50/p95 | GUI proposal p50/p95 |
|---:|---:|---:|---:|---:|
| 32 | 3.10 / 4.37 ms | 3.32 / 5.09 ms | 102.23 / 125.15 ms | 382.78 / 421.46 ms |
| 64 | 3.53 / 5.63 ms | 2.96 / 4.18 ms | 208.48 / 242.65 ms | 503.44 / 523.12 ms |
| 128 | 3.12 / 4.18 ms | 3.15 / 4.07 ms | 417.11 / 447.72 ms | 714.16 / 765.85 ms |
| 256 | 3.24 / 4.50 ms | 3.11 / 4.33 ms | 863.85 / 1005.77 ms | 1160.05 / 1232.68 ms |

At 128 simulations, the same 30-run methodology measured MCTS p50/p95 of
417.11/447.72 ms (1 thread), 400.19/453.59 ms (2), 451.91/532.65 ms (4),
529.29/557.51 ms (6), and 564.46/682.47 ms (8). Normalized whole-machine CPU
was 12.4%, 24.8%, 48.0%, 70.3%, and 87.0%, respectively. One thread is both
the fastest measured setting and the conservative default, leaving headroom
for the GUI and other desktop work. Higher thread counts add contention and
worsen latency. GUI responsiveness was also verified while the worker was
searching.

## QML backend compatibility manifest

All shared-QML consumers retain their original names and Qt-compatible types.

| Contract | Former Python type | Native type | QML consumers | Status |
|---|---|---|---|---|
| `boardModel`, `pieceModel`, `historyModel`, `playerModel` | `QAbstractListModel` | `QAbstractListModel` | Board, history, setup | Parity |
| `geometry` | `QObject` invokables | `QObject` invokables | Board | Parity |
| game/current-player properties | Qt `Property` | `Q_PROPERTY` | GamePanel, TitleBar | Parity |
| proposal/path/capability properties | Qt `Property` | `Q_PROPERTY` | Board, AiPanel, menus, shortcuts | Parity |
| AI name/status/details | Qt `Property` | `Q_PROPERTY` | AiPanel | Parity |
| sound available/enabled/status/volume | Qt `Property` | `Q_PROPERTY` | SoundDialog | Parity |
| standings/order/AI seats | Qt `Property` | `Q_PROPERTY` | setup/result dialogs | Parity |
| `changed`, `errorRaised`, `gameFinished`, `playerFinished` | Qt signals | Qt signals | bindings/dialog lifecycle | Parity |
| selection/confirm/cancel/undo/Think Again | Qt slots | `Q_INVOKABLE` | Board, AiPanel, TitleBar | Parity |
| new/start/save/load/request-AI/shutdown | Qt slots | `Q_INVOKABLE` | Main and lifecycle | Parity |
| sound preview/mute/volume | Qt slots | `Q_INVOKABLE` | SoundDialog | Parity |

Model role manifest:

| Model | Native roles consumed/preserved |
|---|---|
| Board | `positionId`, `unitX`, `unitY`, `campKey`, `occupant`, `isSelected`, `isLegalStep`, `isLegalJump`, `isPathNode`, `pathIndex`, `isLastMoveSource`, `isLastMoveDest`, `isProposalSource`, `isProposalDest` |
| Piece | `pieceId`, `positionId`, `playerId`, `color`, `unitX`, `unitY`, `isMoving`, `isSelected` |
| History | `turnNumber`, `playerId`, `playerLabel`, `playerColor`, `moveText`, `pathText`, `hopCount`, `isAi` |
| Player | `playerId`, `name`, `kindLabel`, `color`, `isCurrent`, `isAi`, `homeCount`, `campSize`, `hasFinished`, `turnIndex`, `place`, `placeLabel` |

The contract executable checks the meta-object API and every role above, then
drives a complete controller sequence including real LibTorch AI when built
with `DIAMOND_BUILD_QT_SOO=ON`.

The deployment bundle uses artifact format v2. Metadata is an exact schema;
the native loader rejects missing/unknown fields, version/model/shape/topology
mismatches, the wrong tensor set or byte dimensions, a bad `model.ts` hash,
and a deterministic aggregate hash mismatch across every raw weight and
topology file actually consumed by the GUI runtime.

## Sound, animation, path, and confirmation behavior

* `QMediaPlayer` owns an explicit `QAudioOutput`, defaults to 60%, queues a
  play request while media loads, and reports decode/backend errors to the
  existing Sound dialog.
* The package contract waits for `move.m4a` to reach loaded media status, not
  merely for the file to exist.
* Mute state is applied directly to `QAudioOutput`, including audio already in
  flight; raising volume above zero unmutes both controller and output state.
* Piece rows retain stable logical IDs. A multi-hop move advances one landing
  every 140 ms while the existing `Piece.qml` 130 ms Behaviors animate lattice
  coordinates; resize does not animate or detach pieces from holes.
* One sound request is emitted per landing.
* The canonical path is displayed before commit with the existing blue line,
  ghost destination, and numbered intermediate landings.
* Human proposals commit only on Confirm/Enter and cancel on Cancel/Escape.
  AI results are proposals; Confirm commits, while Think Again rejects that
  physical action and searches without mutating game state.

## Tests and final checks

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
ctest --test-dir build-qt-clean --output-on-failure
ctest --test-dir build-qt-soo-clean --output-on-failure
python -m pytest
```

Package smokes can also be run manually with `--smoke`, `--game-smoke`,
`--worker-smoke`, `--failure-smoke`, `--sound-smoke`, and `--soo-smoke`. The
final desktop pass is performed on the exact packaged executable, not the
build-tree binary.

## Compatibility notes

* Release target: two-player human-vs-Soo. Three-player native game/controller
  play is retained; native Min/MCTS3P search is deliberately not a Q5
  dependency and currently uses the deterministic legal fallback.
* Worker cancellation is result-boundary cooperative because the existing
  synchronous MCTS evaluator is not interruptible. Results from superseded
  generations are discarded, so stale moves cannot commit; process close may
  wait for the current short search to return.
* Search failures are latched and shown once. They never auto-relaunch; an
  explicit retry or a state-changing operation clears the latch. A requested
  restart while cancellation is draining runs only after the old worker is
  idle.
* Save schema v2 is Python-compatible and restores history/undo by replay and
  state cross-check. The earlier native schema v1 is accepted for migration.
* Tree reuse was inspected and not implemented: undo/load/new-game/settings
  invalidation would add risk before evidence of a human-play latency need.
* Optional Vulkan headers, optional Torch kineto, Visual Studio 2022 probing
  before the installed VS 2026 fallback, and third-party LibTorch narrowing
  warnings are non-blocking environment diagnostics.
* A prior exact-package desktop-control pass covered selection/path display,
  Confirm, AI proposal, Think Again, AI Confirm, and the Sounds dialog. The
  final post-cleanup package could not repeat that input-driven pass because
  Windows denied the automation host's `GetCursorPos` call with `0x80070005`.
  The exact final package still passed QML-load, game, worker, failure, media,
  and Soo clean-environment smokes plus the offscreen controller contract.
