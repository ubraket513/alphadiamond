# Native Windows Qt runtime

Status: GUI renewal gates Q0–Q6 complete, 2026-08-23.

The primary Windows GUI is now a native C++/Qt 6 application. It embeds and
loads the existing QML, runs the authoritative native rules and MCTS, and uses
LibTorch CPU inference for the two-player Soo seat. The deployed executable
does not import Python or start a Python/WSL process. Python remains the
authoritative training/export environment and the old PySide6 host is retained
only as the optional `legacy-gui` oracle.

## Gate record

| Gate | Result | Evidence |
|---|---|---|
| Q0 inventory | Pass | GUI/QML/native/model path inventoried in the blueprint and this document |
| Q1 reusable native libraries | Pass | `soo_core`, `soo_search`, `diamond_model`, and `diamond_qt_backend`; app/test do not copy controller sources |
| Q2 deployment spike | Pass | versioned artifact, Windows LibTorch load, policy/value and legal-prior parity probes |
| Q3 native Qt shell | Pass | existing QML and fonts embedded in resources; native icon/chrome/Snap Layout integration; no Python runtime |
| Q4 controller parity | Pass | proposal/confirm/cancel, path, per-hop animation/sound, history, undo, save/load, new game, terminal state, model roles, illegal-action rejection |
| Q5 human-vs-Soo | Pass | native `MCTS2P` + LibTorch evaluator on worker thread; deterministic settings; canonical action converted to physical seat coordinates |
| Q6 native primary | Pass | PySide6/QtAwesome removed from base dependencies and old entry point renamed `diamond-legacy` under `legacy-gui` |

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
engine, worker, and Soo smokes and fails if a DLL/plugin/artifact is missing.

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
  .\artifacts\soo-spike 128 7 1
```

Measured on this 8-logical-CPU Windows host, Release build, B1 inference,
sequential runs, one Torch thread, seven complete moves per setting:

| simulations | forward p50/p95 | complete move p50/p95 |
|---:|---:|---:|
| 32 | 5.56 / 5.98 ms | 151.29 / 162.20 ms |
| 64 | 4.15 / 5.97 ms | 256.47 / 276.79 ms |
| 128 | 3.91 / 4.39 ms | 505.19 / 519.20 ms |
| 256 | 3.94 / 4.56 ms | 1000.51 / 1022.98 ms |

At 128 simulations, the thread sweep measured move p50/p95 of 569.69/623.78
ms (1 thread), 453.69/463.65 ms (2), 530.01/540.51 ms (4),
653.56/665.24 ms (6), and 898.38/1009.00 ms (8). One thread remains the
conservative default so the GUI and other desktop work retain CPU headroom;
two threads are an available measured tuning option. A final seven-move sample
measured about 12.2% whole-machine CPU at one thread and 23.4% at two threads
(process CPU time normalized across eight logical CPUs). GUI responsiveness
was also verified while the worker was searching.

## QML backend compatibility manifest

All shared-QML consumers retain their original names and Qt-compatible types.

| Contract | Python type | Native type | QML consumers | Status |
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

## Sound, animation, path, and confirmation behavior

* `QMediaPlayer` owns an explicit `QAudioOutput`, defaults to 60%, queues a
  play request while media loads, and reports decode/backend errors to the
  existing Sound dialog.
* The package contract waits for `move.m4a` to reach loaded media status, not
  merely for the file to exist.
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
python -m pytest -m 'not gui'
```

Package smokes can also be run manually with `--smoke`, `--game-smoke`,
`--worker-smoke`, and `--soo-smoke`. The final desktop pass is performed on the
exact packaged executable, not the build-tree binary.

## Compatibility notes

* Release target: two-player human-vs-Soo. Three-player native game/controller
  play is retained; native Min/MCTS3P search is deliberately not a Q5
  dependency and currently uses the deterministic legal fallback.
* Worker cancellation is result-boundary cooperative because the existing
  synchronous MCTS evaluator is not interruptible. Results from superseded
  generations are discarded, so stale moves cannot commit; process close may
  wait for the current short search to return.
* Save schema v2 is Python-compatible and restores history/undo by replay and
  state cross-check. The earlier native schema v1 is accepted for migration.
* Tree reuse was inspected and not implemented: undo/load/new-game/settings
  invalidation would add risk before evidence of a human-play latency need.
* Optional Vulkan headers, optional Torch kineto, Visual Studio 2022 probing
  before the installed VS 2026 fallback, and third-party LibTorch narrowing
  warnings are non-blocking environment diagnostics.
