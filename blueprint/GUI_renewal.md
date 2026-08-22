You are working in the existing repository:

`ubraket513/alphadiamond`

Your task is to migrate the human-play application runtime from PySide6/Python to a fully native Windows C++ application while preserving the existing Python/PyTorch training system.

This is an architectural migration. Do not perform a one-shot rewrite. Inspect the current repository first, establish exact interfaces and parity gates, then migrate incrementally.

## End goal

The final human-vs-AI Windows application must have this runtime architecture:

```text
Windows native executable
┌────────────────────────────────────────────┐
│ Qt 6 Quick / existing QML                  │
│              ↕                             │
│ C++ GameController + Qt models             │
│              ↕                             │
│ native Diamond/Soo rules and state         │
│              ↕                             │
│ native C++ MCTS                            │
│              ↕                             │
│ C++ PyTorch / LibTorch model runtime       │
│              ↕                             │
│ CPU inference                              │
└────────────────────────────────────────────┘
```

Runtime requirements:

- no Python interpreter
- no PySide6
- no pybind11 dependency in the shipped Windows executable
- no WSL service/process during gameplay
- no IPC to Python
- no CUDA requirement
- CPU inference is the authoritative initial backend
- Intel Iris Xe acceleration must NOT be required
- optional XPU support may be added later only if runtime capability and performance are proven

The development/training architecture remains:

```text
WSL/Linux
Python + PyTorch
training
checkpointing
model verification/export/preprocessing
        ↓
portable deployment model/weight artifact
        ↓
Windows native application
```

The Windows target machine has:

- 8 CPU cores
- Intel Iris Xe Graphics
- no CUDA-capable NVIDIA GPU

Do not assume Iris Xe is supported by PyTorch XPU. CPU must work well independently.

---

# Non-goals

Do NOT:

- rewrite the existing QML UI unless a specific binding incompatibility requires a small change
- visually redesign the GUI
- rewrite the training system in C++
- delete Python training code
- delete the current Python GUI before the C++ GUI reaches behavioral parity
- duplicate the native game rules unnecessarily
- replace the native MCTS with a new algorithm
- introduce a network service between WSL and Windows
- depend on Linux AOTInductor binaries from the Windows executable
- assume a `.pt2` compiled under WSL can be loaded by native Windows
- make XPU mandatory
- optimize speculative bottlenecks before measuring them
- break the existing native Python extension or training CI while restructuring C++

---

# Current repository facts you must verify

Do not trust this prompt blindly; inspect current `main`.

At the time this task was written, the repository already contains:

```text
src/diamond/
    main.py
    app/
        controller.py
        ai_worker.py
        models.py
        fonts.py
        icons.py
        sounds.py
        native_chrome.py
        window_chrome.py
    qml/
    alphazero/
    game/

native/
    bindings.cpp
    include/
    src/
        board.cpp
        rules.cpp
        encoder.cpp
        prior.cpp
        mcts.cpp
        evaluator.cpp
        batcher.cpp
        selfplay.cpp
        profile.cpp
```

The C++ native engine is currently built mainly as an optional pybind11 Python extension through `setup.py`.

The existing PySide entry point creates:

- `QGuiApplication`
- `QQmlApplicationEngine`
- the existing `GameController`
- existing QML context properties
- bundled fonts/icons
- Windows shell integration/native chrome
- a QML `Main.qml`

The QML frontend is already separated enough that it should preferably remain unchanged.

The native search/rules implementation is already performance-critical and extensively parity-tested. Reuse it.

Before changing anything:

1. inspect `CLAUDE.md`, `README.md`, build files, relevant docs, current CI
2. inspect the latest commits
3. inspect current PySide/QML API contracts
4. inspect native C++ public/private boundaries
5. inspect the current Soo model definition exactly
6. inspect the current checkpoint format
7. inspect current Python MCTS agent/runtime paths
8. identify whether the GUI currently supports Soo 2-player, Min 3-player, or both
9. report any facts that contradict this prompt

Do not begin the migration until you understand the current implementation.

---

# Architectural target

Refactor native C++ into reusable targets with boundaries conceptually equivalent to:

```text
diamond_core
diamond_search
diamond_model
diamond_qt
diamond_python
```

Exact names may differ if repository conventions suggest something better.

Responsibilities should be:

## `diamond_core`

Pure C++.

Contains reusable game-domain functionality such as:

- board representation
- game state
- actions
- legal move generation
- state transitions
- terminal detection
- encoder/topology utilities required independently of Python

Must depend on neither:

- Python
- pybind11
- Qt
- LibTorch, unless an extremely small tensor-free abstraction makes that impossible

Prefer no Torch dependency at this layer.

## `diamond_search`

Pure C++ search layer.

Contains:

- MCTS
- search node/arena structures
- search configuration
- deterministic and stochastic search logic as currently implemented
- evaluator interface abstraction

Depends on `diamond_core`.

Must not depend on:

- Python
- pybind11
- Qt

The existing training native backend must continue to use this code.

## `diamond_model`

C++ inference abstraction.

Expose an interface similar in spirit to:

```cpp
class ModelEvaluator {
public:
    virtual ~ModelEvaluator() = default;

    virtual Evaluation evaluate(
        const EncodedState& state,
        std::span<const ActionId> legal_actions
    ) = 0;
};
```

Choose the exact interface based on current native evaluator contracts.

Initial concrete backend:

```text
LibTorch CPU Soo evaluator
```

This layer must be replaceable later by another backend without changing:

- MCTS
- Qt controller
- game logic

Potential future backends may include:

- Windows XPU if proven supported
- future Windows-compatible AOTInductor
- another compiled PyTorch deployment mechanism

Do not architect the rest of the application around LibTorch-specific tensor types.

## `diamond_qt`

Native Qt 6 application/backend.

Contains:

- `main.cpp`
- C++ `GameController`
- Qt list models currently implemented in `models.py`
- async AI worker/lifecycle
- font resource registration
- icon/image provider equivalent
- sound support
- Windows chrome/shell integration
- QML wiring

Depends on:

- Qt 6
- `diamond_core`
- `diamond_search`
- `diamond_model`

Must not depend on Python or pybind11.

## `diamond_python`

Keep the existing Python binding surface for:

- training
- parity tests
- Python oracle
- development tooling

It may use pybind11.

It should link against `diamond_core` and `diamond_search` rather than compiling independent duplicate implementations.

The training backend must not regress.

---

# Build system

Introduce a proper CMake build for the native libraries and Windows Qt executable.

Requirements:

- C++20 unless current native code requires otherwise
- reusable static/shared libraries rather than copying `.cpp` files into several targets
- Qt 6 Quick/QML support
- LibTorch integration
- Windows native executable
- reasonable Debug and Release configurations
- explicit warnings
- no `-march=broadwell` assumption for Windows
- architecture flags must be platform-aware

The existing `setup.py` Python extension must continue working during migration.

Preferred direction:

```text
CMake native libraries
       ├── Windows Qt executable
       └── pybind11 extension
```

If making setuptools consume CMake immediately creates unnecessary complexity, an intermediate state is acceptable, but source ownership must remain singular and there must be no forked duplicate C++ engine.

Document how to build:

1. Python native extension on WSL/Linux
2. deployment/export tooling on WSL
3. Windows Qt/LibTorch executable

Do not assume the same binary dependencies are used on Linux and Windows.

---

# Model migration: critical requirement

The Python `SooModel` remains the authoritative TRAINING definition.

The Windows runtime must eventually execute an equivalent model without Python.

Do not casually hand-copy the model and declare success.

First create a model-deployment compatibility spike and parity gate.

## Step M0 — inspect the model

Determine exactly:

- input shape
- dtype
- graph topology
- directional/GNN operations
- residual structure
- layer norm behavior
- policy head
- value head
- source/destination policy indexing
- all tensor reshapes/permutations
- initialization-independent architecture metadata
- checkpoint keys
- model width/depth metadata stored in checkpoint/config

Pin these findings in documentation/tests.

## Step M1 — determine a portable Windows loading strategy

WSL is allowed to preprocess/export the checkpoint.

However:

- do NOT produce a Linux machine-code artifact and expect Windows to load it
- do NOT require Python at Windows runtime

Prove the actual artifact path with a minimal round-trip before implementing the full GUI.

Preferred initial runtime is LibTorch C++ CPU.

Investigate the safest portable way to transfer the authoritative Python weights into the C++ model.

Possible approaches may include:

1. a portable tensor/weight archive explicitly designed for Python → C++ interoperability
2. an officially supported PyTorch serialization path proven to load correctly in LibTorch
3. a small custom deployment bundle containing tensors plus architecture metadata
4. TorchScript only as a fallback if it proves materially simpler and robust for this exact model

Do not pick a format merely because it “should work.”

Write a tiny proof first:

```text
Python checkpoint
→ deployment export
→ C++ LibTorch loader
→ one forward pass
→ compare with Python
```

The final artifact must be platform-neutral data, not Linux-native code.

## Step M2 — implement the C++ model

Mirror the exact Soo network using LibTorch C++ APIs only if the selected deployment path requires this.

Keep the model implementation isolated inside `diamond_model`.

Do not leak `torch::Tensor` across the general game/search/UI API more than necessary.

## Step M3 — mandatory numerical parity

Create a deterministic corpus of representative states.

For every state compare Python and C++:

- encoded input
- raw value output
- raw policy logits
- legal-action extracted logits
- normalized legal priors where applicable

Use sensible explicit tolerances derived from dtype/device.

Test:

- opening states
- middle-game states
- near-terminal states
- states with few legal actions
- states with many legal actions
- both current-player canonicalizations

A model implementation is NOT accepted merely because legal moves look reasonable.

Fail loudly on:

- missing keys
- extra keys
- wrong width/depth metadata
- unsupported checkpoint version
- incompatible topology
- invalid input/output shapes

Create a versioned deployment artifact format.

---

# CPU-first performance policy

The target system has 8 CPU cores.

Do not let LibTorch blindly consume all cores.

Implement or expose configurable CPU threading.

Benchmark combinations such as:

```text
LibTorch intra-op threads:
1
2
4
6
8

LibTorch inter-op:
prefer 1 initially
```

The GUI thread must remain responsive.

AI computation must run away from the Qt GUI thread.

Do not hard-code “8 CPU cores = 8 inference threads.”

Record:

- model B1 forward p50/p95
- complete MCTS move p50/p95
- CPU utilization
- GUI responsiveness

Initial human-play workloads should include:

```text
32 simulations
64 simulations
128 simulations
256 simulations
```

Batch-size-throughput measurements from RTX 5090 training are NOT relevant enough to substitute for single-game CPU latency measurements.

---

# Human-play search requirements

The human-play path should use the existing native C++ MCTS, not the Python MCTS.

Initial rollout target:

```text
2-player human vs Soo
```

If native 3-player search parity is not already available, do not make the first release depend on it.

Preserve current MCTS semantics.

For human play:

- no root Dirichlet noise by default
- no training temperature sampling
- choose the strongest deterministic action appropriate for evaluation/play
- expose search simulation count as configuration
- support cancellation when user starts a new game, loads, closes, or undoes while the AI is thinking

Do not reuse the training exploration settings blindly.

## Tree reuse

Inspect whether safe MCTS tree reuse across alternating human/AI moves is feasible.

Do NOT implement it until the basic native path is correct and measured.

If implemented:

- reuse only when the actual played move matches an existing child
- invalidate on undo
- invalidate on load
- invalidate on new game
- invalidate on settings/model change
- never allow stale game-state/search-state mismatch

Benchmark before and after.

---

# QML migration

Preserve the current QML wherever practical.

The goal is to replace:

```text
PySide6 backend
```

with:

```text
Qt C++ backend
```

not to redesign the frontend.

Before implementing C++ types, inventory every QML dependency on the Python backend:

- context properties
- QObject properties
- NOTIFY signals
- invokable methods/slots
- list-model roles
- enums
- image provider URLs
- window/nativeChrome methods
- sound triggers
- undo/redo behavior
- selected-piece state
- legal-target state
- player status
- game-over state
- history
- settings
- AI-thinking state

Create a compatibility matrix.

Then implement equivalent C++ APIs.

Prefer typed C++ Qt properties and models.

If possible, expose backend classes using modern Qt 6 registration mechanisms rather than excessive untyped context properties, but do not force widespread QML churn just for stylistic purity.

---

# C++ controller

Port the existing Python controller behavior, not just its public method names.

The existing `controller.py` is large. Before translating it:

- identify its responsibilities
- separate pure game orchestration from Qt presentation state
- avoid producing a single 30k-line-equivalent C++ god object

Suggested separation:

```text
GameSession / MatchState
    pure orchestration

GameController
    Qt-facing adapter

AIController / AIWorker
    search lifecycle + cancellation

BoardModel
PieceModel
PlayerModel
HistoryModel
    QAbstractListModel subclasses
```

Use current repository naming where sensible.

Do not refactor unrelated behavior.

---

# AI worker / threading

The current Python GUI uses worker threads so AI thinking does not block QML.

The C++ application must maintain that property.

Design requirements:

- Qt GUI thread never executes MCTS/model inference
- search runs in a worker
- clear lifetime ownership
- cooperative cancellation
- no use-after-free when window closes during search
- no stale AI result may be applied after:
  - undo
  - load
  - reset/new game
  - player configuration change
  - model change

Use an operation/generation ID or equivalent mechanism so obsolete AI results are discarded deterministically.

Remove any artificial “thinking delay” from performance measurements.

A cosmetic thinking delay may remain as an optional UX setting, but default benchmarking must use zero.

---

# Windows-native GUI behavior

Port current Windows-specific behavior from:

```text
native_chrome.py
window_chrome.py
```

to C++/Qt/Win32 where needed.

Preserve, where currently supported:

- application icon
- AppUserModelID
- taskbar behavior
- minimize/restore behavior
- resize animation
- Snap Layout behavior
- rounded corners
- frameless/custom chrome behavior

Do not assume Qt's default frameless window reproduces these.

Use `QAbstractNativeEventFilter`/Win32 APIs where appropriate.

Keep Windows-specific code isolated behind platform guards.

Non-Windows builds of core/search/model should remain possible even if the Qt app itself targets Windows first.

---

# Assets

Move GUI runtime assets to a native Qt resource strategy where appropriate:

- QML
- fonts
- icons
- images
- sounds

Prefer `.qrc` / `qt_add_resources` / Qt 6 QML module facilities.

Do not require Python package-resource APIs at runtime.

Verify resource paths in Release builds, not only from source-tree runs.

---

# Migration safety strategy

The Python GUI must remain temporarily available as an oracle.

Do this in stages.

## Gate Q0 — inventory

Produce a document describing:

- current Python GUI surface
- QML/backend contract
- current native engine structure
- model deployment path proposal
- exact migration phases

No functional change yet.

## Gate Q1 — CMake/native libraries

Refactor existing native C++ into reusable CMake libraries.

Requirements:

- existing native tests still pass
- Python extension still builds
- training path unchanged
- no Qt yet required for training

## Gate Q2 — model deployment spike

Prove:

```text
Python checkpoint
→ WSL export/preprocess
→ Windows-compatible portable artifact
→ C++ LibTorch load
→ numerical parity
```

Do not proceed to full GUI model integration without this.

If actual Windows execution is unavailable in the current Codex environment, structure the code and tests so that the Windows step can be run later, and clearly report what remains unverified.

Do not fake a Windows success from WSL.

## Gate Q3 — native Qt shell

Build a minimal native Qt executable that:

- launches existing QML
- loads resources
- provides placeholder/native controller objects
- reproduces window chrome basics
- contains no Python runtime

No AI yet.

## Gate Q4 — game/controller parity

Port game/controller/models.

Compare behavior with Python GUI.

Test:

- legal selection/movement
- turn advancement
- history
- undo
- redo if supported
- load/save if supported
- new game
- terminal state
- player model data
- QML model roles
- illegal actions rejected

## Gate Q5 — native human-vs-Soo

Wire:

```text
Qt C++
→ native MCTS
→ LibTorch CPU model
```

Use deterministic human-play search settings.

Measure real move latency.

## Gate Q6 — remove Python GUI runtime

Only after parity:

- make native Qt executable the primary GUI
- remove PySide6 from GUI/runtime dependency path
- leave Python training dependencies intact
- optionally retain old Python GUI temporarily under an explicit legacy/dev target if useful
- otherwise delete it only after tests prove replacement coverage

---

# Behavioral parity testing

Do not rely only on GUI screenshots.

Add tests around C++ domain/controller behavior.

Where possible, generate shared fixtures so Python and C++ consume identical command sequences, for example:

```text
new game
select A
move A→B
select C
move C→D
undo
...
```

Compare resulting:

- board occupancy
- current player
- legal moves
- history
- terminal state
- selected state if semantically relevant

The existing Python implementation should remain the oracle during migration.

For QML-visible APIs, add a contract test or documented manifest enumerating:

```text
property / signal / method / model role
Python old type
C++ new type
QML consumer
parity status
```

---

# Training compatibility is a hard invariant

This migration must NOT destabilize the training work.

Current training infrastructure includes a mature native self-play backend with correctness/performance gates.

Therefore:

- do not change MCTS semantics casually
- do not alter action indexing
- do not alter canonical encoding
- do not alter policy indexing
- do not change stochastic training behavior
- do not change Python checkpoint format without an explicit compatibility layer
- do not change replay/training configuration as part of this GUI migration

Run the existing tests after every relevant structural change.

If a CMake refactor causes the pybind extension or native training path to diverge, fix that before continuing.

---

# Performance methodology

The project goal includes eliminating runtime overhead, but optimize using measurements.

Establish these baselines:

```text
A. existing Python GUI + Python agent path
B. PySide/QML + native C++ MCTS + Python PyTorch model
   if convenient to measure
C. native Qt + native MCTS + LibTorch CPU model
```

Most important metric:

```text
human-play AI move latency
```

Measure separately:

```text
encode
search/tree operations
number of NN evaluations
LibTorch forward
policy legal-action processing
total move latency
```

Record:

- p50
- p90/p95
- CPU utilization
- inference thread count
- simulations
- model version

Do not claim the Qt migration itself caused inference speedup unless measured.

---

# Single-game inference considerations

The production training benchmark uses large batches on RTX 5090. Human play on this Windows laptop will normally run one MCTS tree and B1 inference.

Expect a pattern like:

```text
simulation
→ one leaf evaluation
→ simulation
→ one leaf evaluation
...
```

Therefore benchmark B1 latency explicitly.

After correctness is complete, investigate in this order:

1. CPU thread tuning
2. native MCTS baseline
3. tree reuse
4. reducing avoidable tensor allocations/copies
5. fixed-buffer reuse
6. CPU inference compilation/fusion options supported on Windows
7. only then more invasive single-tree batching or parallel-leaf search

Do not change MCTS semantics merely to increase GPU-style batch size on a CPU laptop.

---

# Optional Iris Xe/XPU support

This is explicitly NOT part of the first correctness milestone.

After CPU runtime works:

1. detect whether the installed LibTorch build exposes XPU
2. detect whether the actual Iris Xe device is supported
3. run the exact Soo model
4. verify numerical parity
5. benchmark B1 inference and complete move latency

Only retain an XPU backend if:

- it is actually supported on this machine
- it is reliable
- it materially improves latency

CPU remains fallback.

Do not make packaging depend on Intel GPU drivers beyond what is necessary for optional acceleration.

---

# Model artifact/versioning

The deployment artifact must contain enough metadata to reject incompatible models.

At minimum consider:

```text
format_version
game/model = Soo
input feature version
topology/action-index version
width
residual block count
policy shape/version
value shape/version
training/checkpoint step
parameter tensors
optional model hash
```

The Windows app should fail with a useful message instead of crashing or silently loading incompatible weights.

Keep the training checkpoint and deployment artifact conceptually separate.

The deployment artifact may be regenerated from an authoritative checkpoint.

---

# Packaging target

Final deliverable should be a native Windows application package containing only required runtime components such as:

```text
alphadiamond.exe
Qt runtime DLLs/plugins
QML/resources
LibTorch CPU runtime DLLs
model deployment artifact
other native dependencies
```

There must be no required:

```text
python.exe
python3.dll
PySide6
site-packages
WSL launch
pip environment
```

Provide a packaging/deployment script or documented CMake install path.

If using `windeployqt`, integrate/document it.

---

# CI

Do not require Windows-only CI before the repository has a Windows runner, but structure tests into:

```text
portable C++ core tests
Python/native parity tests
model export tests
Qt backend tests where supported
Windows integration tests when runner exists
```

If GitHub Actions can reasonably add a Windows build job without large proprietary dependencies, add one.

Do not download huge LibTorch packages on every unrelated CI job unnecessarily.

Use caching or a separate opt-in/integration job as appropriate.

---

# Documentation

Create a migration design document before major implementation.

Suggested location:

```text
docs/native_windows_runtime.md
```

Document:

- architecture
- Linux/WSL vs Windows responsibility boundary
- build instructions
- model deployment artifact
- parity guarantees
- supported hardware
- CPU thread configuration
- optional XPU status
- performance baseline
- remaining limitations

Also update README only when user-facing instructions actually change.

---

# Commit/PR strategy

Do not make one enormous commit.

Prefer logically isolated commits/PRs such as:

```text
1. build: factor native engine into reusable CMake targets
2. model: add portable deployment export and LibTorch parity probe
3. qt: add native Qt/QML application shell
4. qt: port controller and list models
5. qt: port Windows chrome/assets/sounds
6. ai: connect native Soo MCTS to LibTorch CPU evaluator
7. perf: benchmark and tune human-play CPU inference
8. cleanup: retire PySide GUI runtime dependency
```

Exact boundaries may change after repository inspection.

Each commit should keep the repository buildable and tests meaningful.

Do not mix unrelated training changes into these commits.

---

# Before writing code

Your first response/work product must contain:

1. repository state and current HEAD
2. files/components inspected
3. exact current GUI → game → AI → model data flow
4. exact current native C++ dependency structure
5. exact current Soo model/checkpoint structure
6. risks/unknowns
7. proposed CMake target graph
8. proposed model deployment format/spike
9. staged implementation plan
10. explicit tests/gates for each stage

If you discover a material contradiction—for example the current native search only supports a subset of GUI modes—call it out before implementation.

Then proceed stage by stage.

---

# Key decision rules

When uncertain, follow these rules:

- Preserve QML.
- Preserve Python training.
- Reuse existing native rules/search.
- Make C++ libraries independent of Python.
- CPU first.
- No WSL runtime dependency.
- No CUDA assumption.
- No mandatory Iris Xe/XPU assumption.
- Measure B1 human-play latency.
- Model parity before GUI integration.
- Behavioral parity before deleting PySide.
- Do not weaken existing native/training correctness gates.
- Do not claim Windows functionality unless it was actually built/tested on Windows.
- Prefer a clear portable model artifact over clever cross-platform binary reuse.
- Keep abstractions narrow enough that the model backend can change later.

The intended end result is not merely “the GUI was rewritten in C++.”

It is:

> A standalone Windows Qt/QML application in which the complete human-vs-Soo runtime—GUI controller, game logic, MCTS, and PyTorch model inference—runs natively in C++ with no Python or WSL process, while WSL/Python remains the authoritative environment for training and producing versioned deployment model artifacts.

Start by auditing the current repository and presenting the migration design and exact first gate. Do not begin with a bulk translation of Python files.