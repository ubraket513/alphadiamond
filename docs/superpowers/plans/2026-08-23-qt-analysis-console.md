# Qt AlphaZero Analysis Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live, fixed-perspective AlphaZero telemetry console to the native Qt game without changing search or move semantics.

**Architecture:** Extend the Qt-independent Soo search result with root values and measured evaluator time, translate one completed search into a Qt-independent worker result, and let `NativeController` own ephemeral per-ply telemetry models. Reuse the AI move-selection search, serialize human background analysis through the existing single worker, and render three two-series charts plus a compute split in reusable QML components.

**Tech Stack:** C++20, Qt 6 Core/QML/Quick, LibTorch CPU evaluator, QML Canvas, CMake/CTest.

**Spec:** `C:/Users/dzk55/.codex/attachments/602b0ccb-12c0-4cc1-9207-cc89887e813b/pasted-text.txt`

## Global Constraints

- Preserve Soo rules, PUCT, simulations, temperature, Dirichlet behavior, weights, and move selection.
- Never run an extra root neural evaluation solely for telemetry.
- Keep MCTS independent of Qt and expose no GPU telemetry.
- Normalize stored Soo values to a fixed P1 perspective; P2 display is the exact negation/complement.
- Human moves remain immediately interactive; stale background results are discarded by generation and position identity.
- Telemetry is deliberately ephemeral on save/load to preserve schema-v2 compatibility.
- Preserve all pre-existing worktree changes.

---

### Task 1: Search telemetry semantics and timing

**Files:**
- Modify: `native/include/soo/mcts.hpp`
- Modify: `native/src/mcts.cpp`
- Modify: `native/tests/smoke.cpp`

**Interfaces:**
- Produces: `SearchResult::root_network_value`, `root_mean_value`, and `neural_evaluation_ms`, all in canonical root-player-to-act perspective.

- [ ] Add failing native smoke assertions using a deterministic counting evaluator: the captured root NN value equals the first root evaluation, the root mean equals the visit-weighted root-edge Q, evaluator timing is finite/non-negative, and aligned selected action metrics remain correct.
- [ ] Run `cmake --build build --target soo_native_smoke` and `ctest --test-dir build -R soo_native_smoke --output-on-failure`; confirm the new assertions fail to compile because the fields do not exist.
- [ ] Capture the already-performed root evaluation in `complete_expansion`, compute the visit-weighted root mean in `finalize`, and accumulate `steady_clock` time around each real evaluator call in `MCTS2P::run`.
- [ ] Rebuild and rerun the native smoke test; confirm it passes without changing search outputs.

### Task 2: Structured worker result and telemetry math

**Files:**
- Create: `native/qt/search_telemetry.hpp`
- Create: `native/qt/search_telemetry.cpp`
- Modify: `native/qt/ai_worker.hpp`
- Modify: `native/qt/ai_worker.cpp`
- Modify: `native/qt/CMakeLists.txt`
- Modify: `native/qt/tests/controller_contract.cpp`

**Interfaces:**
- Produces: `ActionTelemetry`, `SearchTelemetry`, `AiSearchResult`, safe compute-rate helpers, and P1/P2 normalization helpers.
- Produces: `NativeAiWorker::resultReady(quint64, AiSearchResult)` and `start(quint64, std::function<AiSearchResult()>)`.

- [ ] Add failing controller-contract assertions for zero-safe timing/rates, exact P1/P2 negation and estimate complements, action/prior/Q/visit alignment, and typed worker delivery.
- [ ] Build the contract target and confirm failure because the telemetry API is absent.
- [ ] Implement the native telemetry types and pure normalization/accounting helpers; register the result metatype and refactor the worker boundary.
- [ ] Rebuild and run the targeted contract test; confirm the math and worker contract pass.

### Task 3: Persistent Soo runtime and controller lifecycle

**Files:**
- Create: `native/qt/soo_search_runtime.hpp`
- Create: `native/qt/soo_search_runtime.cpp`
- Modify: `native/qt/native_controller.hpp`
- Modify: `native/qt/native_controller.cpp`
- Modify: `native/qt/CMakeLists.txt`
- Modify: `native/qt/tests/controller_contract.cpp`

**Interfaces:**
- Produces: one lazily initialized worker-owned `SooSearchRuntime::search(...)` that validates and loads the artifact once and returns `AiSearchResult`.
- Produces: `positionTelemetryModel`, `decisionTelemetryModel`, `latestSearchCompute`, `analysisAvailable`, and `perspectivePlayerId` controller properties.

- [ ] Add failing contract assertions for AI telemetry publication/commit, stale-result rejection, non-blocking human commit with missing data, undo truncation independent of history, and new-game/load clearing.
- [ ] Run the contract test and confirm the lifecycle assertions fail because telemetry models are absent.
- [ ] Move artifact/model/evaluator construction into the serialized runtime; reuse it for AI selection and human analysis.
- [ ] Store pending search telemetry by analyzed state/generation, attach selected-action metrics only when the matching move commits, append one per committed ply, and clear/truncate on load/new game/undo.
- [ ] Start background analysis only on supported human Soo turns, never set `aiThinking` for it, never lock selection, and discard late results by generation/state signature.
- [ ] Rebuild and rerun the controller contract until all lifecycle assertions pass.

### Task 4: Analysis console and relocated history

**Files:**
- Create: `src/diamond/qml/TelemetryChart.qml`
- Create: `src/diamond/qml/PositionOutlookPanel.qml`
- Create: `src/diamond/qml/DecisionValuePanel.qml`
- Create: `src/diamond/qml/MovePreferencePanel.qml`
- Create: `src/diamond/qml/SearchComputePanel.qml`
- Create: `src/diamond/qml/HistoryDrawer.qml`
- Modify: `src/diamond/qml/SidePanel.qml`
- Modify: `src/diamond/qml/TitleBar.qml`
- Modify: `src/diamond/qml/Main.qml`
- Modify: `src/diamond/qml/AiPanel.qml`
- Modify: `native/qt/CMakeLists.txt`
- Modify: `native/qt/tests/controller_contract.cpp`

**Interfaces:**
- Consumes: controller telemetry models/properties from Task 3.
- Produces: three fixed-range two-series Canvas charts, latest-search two-segment compute bar, P1/P2 selector, and overlay history drawer.

- [ ] Add failing runtime/QML contract assertions that the analysis objects instantiate, permanent side panel has no history child, and the title-bar history action opens a separate overlay without resizing the board row.
- [ ] Build/run the Qt contract/smoke and confirm the new QML surface is absent.
- [ ] Implement the reusable Canvas chart and four compact panels, including unsupported messaging for Min.
- [ ] Replace permanent history with the analysis console, keep AI controls intact, and wire a title-bar action to an overlay `HistoryPanel`.
- [ ] Add all QML files to resources, rebuild, and rerun QML/controller smokes.

### Task 5: Regression and release verification

**Files:**
- Modify only files required by failures attributable to this feature.

- [ ] Run `cmake --build build --parallel 1` and `ctest --test-dir build --output-on-failure` for portable native parity/smoke coverage.
- [ ] Run `cmake --build build-qt-soo-clean --parallel 1` and `ctest --test-dir build-qt-soo-clean --output-on-failure` for the Soo-enabled Qt build and contracts.
- [ ] Run the repository Python tests relevant to native MCTS parity and Qt contracts.
- [ ] Inspect `git diff --check`, the final diff, and the specification checklist; report exact commands, results, semantics, timing convention, and deferred Min analysis.
