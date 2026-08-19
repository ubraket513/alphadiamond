# RTX 3060 AlphaZero Training Implementation Plan

**Goal:** Run Soo AlphaZero self-play and training efficiently on an RTX 3060 while using available CPU parallelism, preserving learning semantics and CPU compatibility, and making pathological games abort individually instead of terminating an iteration.

**Architecture:** Keep independent multiprocessing self-play workers and the existing centralized inference coordinator. Resolve worker count from available CPUs minus two, run policy/value inference and training on CUDA, batch inference across games, propagate a 900-second per-game monotonic deadline through MCTS, and keep the parent pool timeout only as a catastrophic safety guard.

**Tech Stack:** Python 3.11+, PyTorch, multiprocessing spawn, pytest, NVIDIA CUDA / RTX 3060.

**Spec:** `blueprint/gpu_train.md`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

---

## Verified Repository Baseline

Every assumption in `blueprint/gpu_train.md` was checked against the code on branch `main` at commit `27b06e8`. Results:

| Prompt assumption | Verdict | Evidence |
|---|---|---|
| Workers are separate `spawn` processes | **True** | `selfplay_workers.py:426` `multiprocessing.get_context("spawn")` |
| Workers send NN requests to the parent | **True** | `_ProcessRequestCoordinator` + `SelfPlayWorkerPool._forward_inference` |
| Parent owns a central `InferenceCoordinator` | **True** | `cpu_b0_train.py:275-282` |
| Coordinator batches on `max_batch_size` / `max_wait_ms` | **True** | `coordinator.py:159-204` |
| `InferenceModelPool` accepts a device | **True** | `model_pool.py:19-25` |
| `TorchEvaluator` does real batched inference, supports CUDA | **True** | `evaluator/torch.py:51-62` |
| `TorchEvaluator` has FP32/BF16 support | **True** | `evaluator/torch.py:29-35`, `56-60` |
| `AlphaZeroTrainer` honours `TrainingConfig.device` | **True** | `trainer.py:38-41`, `62-68` |
| Soo MCTS evaluates one leaf synchronously | **True** | `search_2p.py:91` `self.evaluator.evaluate((request,))[0]` |
| `lane_count = min(worker_count, len(jobs))` | **True** | `selfplay_workers.py:427` |
| Pool timeout is `max(600.0, response_timeout * 4)` | **True** | `cpu_b0_train.py:281` → 2400 s with the shipped config |
| `self_play.max_moves = 2000`, `response_timeout_s = 600` | **True** | `runtime/configs/soo-cpu8h.json` |
| Source checkpoint is "around step 72" | **True, now pinned** | see below |

### Pinned source-checkpoint facts (do not re-derive; do not hard-code `72` in code)

Measured from `runtime/runs/soo/cpu8h-soo-20260819/latest.pt`:

```text
training_step   : 72
sha256          : 4b2a32ff15179e890d4266346bca178d9a255eebe16af3a6e3d0482f0ceb1320
size_bytes      : 9107147
operation_id    : cpu8h-soo-20260819-i000017
format_version  : 1
training_config : {"batch_size": 256, "learning_rate": 0.001,
                   "weight_decay": 0.0001, "device": "cpu", "seed": 7}
metadata        : Soo 2.0.0, player_count 2, width 128, residual_blocks 6,
                  value_semantics_version "current-player-scalar-winloss-v1"
loop_state      : iteration 18, attempted 288, completed 288, aborted 0
```

These values belong in the new run's provenance record, discovered at runtime. Tests must never assert `training_step == 72`.

### Findings that change the design (discovered, not in the prompt)

1. **`load_checkpoint` rejects a cross-device checkpoint.** `checkpoint.py:126-133` compares the checkpoint's `training_config.device` against `trainer.device` and raises `CheckpointError`. The source checkpoint records `device: "cpu"`; a `cuda:0` trainer therefore **cannot** load it through `load_checkpoint`. This is the central reason the fork needs its own explicit code path (Task 5), not just a CLI flag. `load_inference_checkpoint` has **no** such check and already accepts `device=`, so self-play inference on CUDA works today.

2. **`ProductionConfig._exact_keys` forbids unknown keys.** `production.py:60-67` and `_dataclass_from_payload` require the payload key set to equal the dataclass field set exactly. Adding `max_game_seconds` to `SelfPlayConfig` **invalidates all four** `configs/alphazero/*.json` files until they are updated, and `tests/alphazero/orchestration/test_reference_configs.py` turns that into a build failure. Task 2 updates those four files in the same commit.

3. **`runtime/configs/*.json` never go through `ProductionConfig`.** `cpu_b0_train.py:66-67` loads them as raw dicts and splats them into dataclasses. So a *new* `runtime/configs/soo-rtx3060.json` is validated only by the dataclass constructors plus the tests we write.

4. **The worker module must stay torch-free.** `test_selfplay_workers.py:394` spawns a subprocess asserting `torch` is absent from `sys.modules` after importing `selfplay_workers`. Any clock/deadline abstraction it imports must be pure standard library.

5. **`InferenceMetrics` accumulation is O(n²) and unbounded.** `coordinator.py:362-385` rebuilds four tuples by unpacking the previous tuples on *every* batch, appending one element per request. At CPU rates this is invisible; at GPU rates (tens of thousands of requests per iteration) it becomes both a memory leak and a measurable CPU cost inside the inference thread. Task 8 replaces the accumulation with streaming summaries — this is a correctness/scaling fix, not premature optimization.

6. **`TorchEvaluator` performs per-row GPU→CPU syncs.** `evaluator/torch.py:74-84`: for each row of a batch it builds a `legal` tensor on-device, calls `int(legal.min())`/`int(legal.max())` (two syncs), then `.cpu()` on probabilities and on the value row (two more). That is ~4 device syncs *per request*, serializing the batch it just gathered. Profiled as a follow-up in Task 11; deliberately **not** folded into the first correctness patch.

7. **This development machine is not the target.** It reports 8 CPUs, no `nvidia-smi`, and `torch 2.12.0` with `torch.version.cuda is None` (CPU-only build). The RTX 3060 / 32-CPU target is a **rented vast.ai instance reached over SSH**. Consequently: all logic must be verifiable on this CPU-only box, CUDA tests must skip cleanly, and GPU provisioning is an explicit rollout step (Task 10), not an assumption.

8. **`/runtime/` is commented out in `.gitignore`** (line 13), so run artifacts currently show as untracked. The new GPU run directory must not be committed; Task 6 re-enables the ignore rule for the new run path only, without retroactively removing the existing CPU run from the index.

### Baseline test state

`pytest tests/alphazero tests/tools` → **377 passed, 1 skipped** in ~10 s. The full suite (`-m "not gui"`) currently fails collection on `tests/test_icons.py` because `qtawesome` is absent from this environment; that is a pre-existing GUI-dependency gap, unrelated to this work, and no task here should try to fix it. Use the scoped command above as the regression gate.

---

## Global Constraints

- Learning semantics are frozen. No change to policy/value targets, terminal semantics, temperature schedule, replay sampling, or the bootstrap prior. Aborted games contribute exactly zero samples.
- The existing CPU run `runtime/runs/soo/cpu8h-soo-20260819` is immutable. Nothing in this plan writes to it.
- `tools/cpu_b0_train.py` keeps working with its current CLI and its current config files. Backward compatibility is a test, not an intention.
- CPU-only CI must stay green. Every CUDA test carries `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
- `src/diamond/alphazero/orchestration/selfplay_workers.py` must not import torch.
- FP32 only. No mixed-precision training, no BF16 in the first GPU path.
- No intra-tree parallel MCTS, no virtual loss, no `torch.compile`, no CUDA graphs, no NVML dependency.
- Deadlines use `time.monotonic` through an injectable clock. No `sleep()` watchdogs. No test waits 15 minutes.
- Every behavior change is red-green-refactor TDD, with the failing test committed first in the working tree.

---

## File/Component Map

**New source files**

- `src/diamond/alphazero/hardware.py` — affinity-aware CPU/worker resolution. Pure stdlib, torch-free.
- `src/diamond/alphazero/deadline.py` — `Deadline` value object and `DEADLINE_EXCEEDED` reason constant. Pure stdlib, torch-free.
- `src/diamond/alphazero/inference/summary.py` — streaming batch/latency summarization for the ledger.
- `src/diamond/alphazero/run_fork.py` — fresh-run initialization from an external checkpoint.

**Modified source files**

- `src/diamond/alphazero/config.py` — add `SelfPlayConfig.max_game_seconds: float | None = None`.
- `src/diamond/alphazero/mcts/search_2p.py` — accept an optional `deadline`; check between simulations.
- `src/diamond/alphazero/mcts/search_3p.py` — same, for parity.
- `src/diamond/alphazero/selfplay/runner_2p.py` — own the game deadline, emit the timeout abort.
- `src/diamond/alphazero/selfplay/runner_3p.py` — same.
- `src/diamond/alphazero/selfplay/common.py` — export the new abort reason.
- `src/diamond/alphazero/orchestration/selfplay_workers.py` — lane-aware pool timeout.
- `src/diamond/alphazero/inference/coordinator.py` — streaming metrics.
- `src/diamond/alphazero/checkpoint.py` — `load_checkpoint(..., allow_device_migration=False)`.
- `tools/cpu_b0_train.py` — worker auto-resolution, `--init-checkpoint`, throughput/GPU ledger fields.

**New config**

- `runtime/configs/soo-rtx3060.json`

**Modified configs** (forced by constraint 2 above)

- `configs/alphazero/soo-bootstrap.json`, `soo-production.json`, `min-bootstrap.json`, `min-production.json`

**New test files**

- `tests/alphazero/test_hardware.py`
- `tests/alphazero/test_deadline.py`
- `tests/alphazero/test_cuda_parity.py`
- `tests/alphazero/inference/test_summary.py`
- `tests/alphazero/test_run_fork.py`

**Modified test files**

- `tests/alphazero/test_mcts_2p.py`, `tests/alphazero/test_selfplay.py`
- `tests/alphazero/orchestration/test_selfplay_workers.py`
- `tests/alphazero/test_checkpoint.py`
- `tests/tools/test_cpu_b0_train.py`
- `tests/alphazero/orchestration/test_reference_configs.py`

---

## Implementation Tasks

### Task 1: Affinity-aware worker resolution

**Files:**
- Create: `src/diamond/alphazero/hardware.py`
- Create: `tests/alphazero/test_hardware.py`

**Interfaces:**

```python
RESERVED_CPUS = 2

def available_cpu_count() -> int:
    """CPUs actually usable by this process: sched_getaffinity, else cpu_count, else 1."""

def resolve_worker_count(
    configured: int | None = None,
    *,
    available: int | None = None,
    reserved: int = RESERVED_CPUS,
) -> int:
    """Explicit override wins; otherwise available - reserved, floored at 1."""
```

`available_cpu_count` prefers `os.sched_getaffinity(0)` when the attribute exists (Linux), falls back to `os.cpu_count()`, and finally to `1`. It must not raise on platforms lacking affinity support.

- [ ] **Step 1: Write failing tests** in `tests/alphazero/test_hardware.py` covering: `resolve_worker_count(available=32) == 30`; `resolve_worker_count(available=8) == 6`; `resolve_worker_count(available=1) == 1` and `available=2` → `1` (floor, never zero or negative); `resolve_worker_count(4, available=32) == 4` (explicit override wins); `resolve_worker_count(0, ...)` and a negative value raise `ValueError`; `available_cpu_count()` returns `len(os.sched_getaffinity(0))` when present (monkeypatch `os.sched_getaffinity` to return a 12-element set → expect `12`); with `sched_getaffinity` deleted via `monkeypatch.delattr(os, "sched_getaffinity", raising=False)` and `os.cpu_count` patched to `9`, expect `9`; with both unavailable (`cpu_count` → `None`), expect `1`.
- [ ] **Step 2: Run `python -m pytest tests/alphazero/test_hardware.py -v`.** Expected failure: `ModuleNotFoundError: No module named 'diamond.alphazero.hardware'`.
- [ ] **Step 3: Implement `hardware.py`** with the interface above. Standard library only — no torch import, so the module is safe for the spawn-worker import graph.
- [ ] **Step 4: Run `python -m pytest tests/alphazero/test_hardware.py -v`.** Expected: all pass.
- [ ] **Step 5: Run `python -m pytest tests/alphazero tests/tools -q`.** Expected: 377 passed + the new tests, 1 skipped.
- [ ] **Step 6: Commit** `feat(alphazero): resolve self-play worker count from available CPUs`.

---

### Task 2: `max_game_seconds` configuration surface

Adding a `SelfPlayConfig` field breaks strict config validation repo-wide, so the field, the four strict config files, and their tests move together in one atomic commit.

**Files:**
- Modify: `src/diamond/alphazero/config.py`
- Modify: `configs/alphazero/soo-bootstrap.json`, `soo-production.json`, `min-bootstrap.json`, `min-production.json`
- Modify: `tests/alphazero/orchestration/test_reference_configs.py`

**Interface:**

```python
@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    max_moves: int = 2000
    temperature_moves: int = 20
    temperature: float = 1.0
    seed: int = 0
    bootstrap_prior: str = BOOTSTRAP_PRIOR_NONE
    max_game_seconds: float | None = None
    """Per-game wall-clock budget; ``None`` disables the deadline entirely."""
```

`__post_init__` gains: if `max_game_seconds is not None` and (not a real number, or `<= 0`), raise `ValueError("max_game_seconds must be a positive number or None")`. `None` is the default so that **old configs and old checkpoints keep behaving exactly as before** — an omitted key means "no wall-clock limit", which is today's semantics.

- [ ] **Step 1: Write failing tests.** In `test_reference_configs.py` add `test_reference_configs_declare_a_game_time_budget`, asserting every file in `configs/alphazero/` has `max_game_seconds` present in its `self_play` object and that it round-trips through `ProductionConfig.from_payload`. Add to `tests/alphazero/test_selfplay.py` (or a new focused block in `test_hardware.py`'s sibling) direct dataclass tests: `SelfPlayConfig()` has `max_game_seconds is None`; `SelfPlayConfig(max_game_seconds=900.0)` is accepted; `max_game_seconds=0` and `-1.0` each raise `ValueError`.
- [ ] **Step 2: Run `python -m pytest tests/alphazero/orchestration/test_reference_configs.py tests/alphazero/test_selfplay.py -v`.** Expected failure: `KeyError`/assertion on the missing `max_game_seconds` key, and `TypeError: SelfPlayConfig.__init__() got an unexpected keyword argument`.
- [ ] **Step 3: Add the field** to `SelfPlayConfig` with its validation, then add `"max_game_seconds": null` to the `self_play` block of all four `configs/alphazero/*.json` files. Leave `runtime/configs/soo-cpu8h.json` and `min-cpu8h.json` **unchanged** — they are loaded as raw dicts and omitting the key must keep working, which is exactly the backward-compatibility property under test.
- [ ] **Step 4: Run `python -m pytest tests/alphazero/orchestration/test_reference_configs.py tests/alphazero/test_selfplay.py -v`.** Expected: all pass.
- [ ] **Step 5: Run `python -m pytest tests/alphazero tests/tools -q`.** Expected: green, confirming the untouched runtime configs still load.
- [ ] **Step 6: Commit** `feat(alphazero): add optional per-game wall-clock budget to SelfPlayConfig`.

---

### Task 3: Deadline abstraction

**Files:**
- Create: `src/diamond/alphazero/deadline.py`
- Create: `tests/alphazero/test_deadline.py`

**Interfaces:**

```python
MAX_GAME_TIME_EXCEEDED = "max_game_time_exceeded"

@dataclass(frozen=True, slots=True)
class Deadline:
    started_at: float
    budget_s: float
    clock: Callable[[], float] = monotonic

    @classmethod
    def start(cls, budget_s: float | None, *, clock: Callable[[], float] = monotonic) -> "Deadline | None":
        """Return a started deadline, or ``None`` when no budget is configured."""

    @property
    def expired(self) -> bool: ...

    @property
    def remaining_s(self) -> float: ...
```

`Deadline.start(None)` returns `None`, so "no budget" is representable without a sentinel object and callers use a plain `if deadline is not None and deadline.expired`. The injected `clock` is what makes every timeout test instant.

- [ ] **Step 1: Write failing tests** in `tests/alphazero/test_deadline.py`: `Deadline.start(None)` returns `None`; `Deadline.start(0)` and a negative budget raise `ValueError`; with a fake clock list-driven callable, a 900 s budget is not expired at `t+899.9`, is expired at exactly `t+900.0` (boundary is inclusive), and is expired at `t+901`; `remaining_s` is `900.0` at start, clamps to `0.0` (never negative) once past the budget; the dataclass is frozen (`FrozenInstanceError` on assignment); the module imports without torch (assert `"torch" not in sys.modules` in a subprocess, mirroring the existing convention in `test_selfplay_workers.py:394`).
- [ ] **Step 2: Run `python -m pytest tests/alphazero/test_deadline.py -v`.** Expected failure: `ModuleNotFoundError: No module named 'diamond.alphazero.deadline'`.
- [ ] **Step 3: Implement `deadline.py`.** Standard library only.
- [ ] **Step 4: Run `python -m pytest tests/alphazero/test_deadline.py -v`.** Expected: all pass.
- [ ] **Step 5: Commit** `feat(alphazero): add monotonic Deadline value object`.

---

### Task 4: MCTS deadline propagation and per-game timeout abort

The runner owns the deadline; MCTS only observes it. A game therefore stops between simulations rather than only after a whole move, which matters because a 64-simulation move on a pathological state is exactly the case that overruns.

**Files:**
- Modify: `src/diamond/alphazero/mcts/search_2p.py`, `src/diamond/alphazero/mcts/search_3p.py`
- Modify: `src/diamond/alphazero/selfplay/runner_2p.py`, `src/diamond/alphazero/selfplay/runner_3p.py`
- Modify: `src/diamond/alphazero/selfplay/common.py`
- Modify: `tests/alphazero/test_mcts_2p.py`, `tests/alphazero/test_selfplay.py`

**Interfaces:**

`MCTS2P.__init__` and `MCTS3P.__init__` gain a keyword-only `deadline: Deadline | None = None`. Inside `run`, the simulation loop becomes:

```python
for _ in range(self.config.simulations):
    if self.deadline is not None and self.deadline.expired:
        break
    ...
```

The check sits at the top of the loop, before descent and before `_expand` (the expensive, inference-bound step). The root expansion at `search_2p.py:45` is **not** guarded: a search must always return a usable action distribution, and the root is expanded exactly once. Breaking early leaves the visit counts accumulated so far, so `run` still returns a valid `SearchResult2P` — the runner, not MCTS, decides the game is over.

Both runners gain the deadline as an optional constructor argument and start one from `selfplay_config.max_game_seconds`:

```python
def __init__(self, game, evaluator, mcts_config, selfplay_config, compatibility,
             *, clock: Callable[[], float] = monotonic) -> None:
    ...
    self.clock = clock

def run(self) -> SelfPlayEpisode:
    deadline = Deadline.start(self.selfplay_config.max_game_seconds, clock=self.clock)
    ...
    while not self.game.is_terminal(state) and move_count < self.selfplay_config.max_moves:
        if deadline is not None and deadline.expired:
            return SelfPlayEpisode((), None, move_count, False, MAX_GAME_TIME_EXCEEDED)
        ...
```

Ordering is deliberate: the deadline is checked at the top of the move loop, so an expired game returns `max_game_time_exceeded` with zero samples and `final_order=None`, satisfying `EpisodeResult.__post_init__`'s abort invariant. `max_game_moves_exceeded` remains the separate terminal-check branch at `runner_2p.py:54`, unchanged and still distinct.

- [ ] **Step 1: Write failing MCTS tests** in `tests/alphazero/test_mcts_2p.py`: with a fake clock that jumps past the budget after N calls, `MCTS2P(..., deadline=expired_deadline).run(...)` returns a valid `SearchResult2P` whose total visit count is strictly less than `config.simulations`; with `deadline=None` the visit total is exactly `simulations` (proving the default path is byte-for-byte the old behavior); an already-expired deadline still returns a result with a legal `selected_action` (root expansion is not skipped).
- [ ] **Step 2: Write failing runner tests** in `tests/alphazero/test_selfplay.py`: a `SooSelfPlayRunner` with `max_game_seconds=900.0` and an injected clock that advances 901 s after the first move returns `completed is False`, `aborted_reason == "max_game_time_exceeded"`, `samples == ()`, `final_order is None`, and a `move_count` equal to the number of moves actually played; a runner with `max_game_seconds=None` and a clock that advances an hour still completes normally; a runner hitting `max_moves` first still reports `max_game_moves_exceeded`, proving the two reasons stay distinct. No test advances real time.
- [ ] **Step 3: Run `python -m pytest tests/alphazero/test_mcts_2p.py tests/alphazero/test_selfplay.py -v`.** Expected failure: `TypeError: __init__() got an unexpected keyword argument 'deadline'` / `'clock'`.
- [ ] **Step 4: Implement** the deadline parameter in both MCTS classes, the clock/deadline in both runners, and export `MAX_GAME_TIME_EXCEEDED` from `selfplay/common.py`.
- [ ] **Step 5: Run `python -m pytest tests/alphazero/test_mcts_2p.py tests/alphazero/test_mcts_3p.py tests/alphazero/test_selfplay.py -v`.** Expected: all pass.
- [ ] **Step 6: Run `python -m pytest tests/alphazero tests/tools -q`.** Expected: green — in particular `tests/alphazero/bootstrap/` and `test_milestone2_smoke.py`, which construct runners positionally.
- [ ] **Step 7: Commit** `feat(alphazero): abort a self-play game on its wall-clock deadline`.

---

### Task 5: One timed-out game must not fail its siblings

The abort produced in Task 4 is an ordinary `EpisodeResult` travelling the existing result queue, so a timed-out game already returns rather than raising. This task proves that end-to-end and fixes the pool-level deadline, which is still a global 2400 s guillotine.

**Files:**
- Modify: `src/diamond/alphazero/orchestration/selfplay_workers.py`
- Modify: `tests/alphazero/orchestration/test_selfplay_workers.py`

**Interface:**

```python
class SelfPlayWorkerPool:
    def __init__(self, coordinator, *, worker_count: int,
                 worker_timeout_s: float | None = None,
                 per_game_timeout_s: float | None = None,
                 grace_s: float = 300.0,
                 join_timeout_s: float = 2.0) -> None: ...

    @staticmethod
    def derive_pool_timeout(*, job_count: int, lane_count: int,
                            per_game_timeout_s: float, grace_s: float) -> float:
        """max jobs on any one lane x per-game budget + grace."""
        jobs_per_lane = math.ceil(job_count / lane_count)
        return jobs_per_lane * per_game_timeout_s + grace_s
```

With 32 games over 30 lanes, `ceil(32/30) == 2`, so the guard is `2 * 900 + 300 = 2100 s` — comfortably above the worst legitimate lane, and no longer a fixed number that a slow-but-healthy iteration can trip. `worker_timeout_s` stays supported and explicit-wins, so today's callers are unaffected; when it is `None` and `per_game_timeout_s` is given, the derived value is used; when both are `None`, the previous default of `60.0` is retained.

The guard's meaning changes in the error message too: it now names a catastrophic condition (`"self-play pool exceeded its catastrophic safety deadline"`) and keeps listing pending game IDs, so the four failure modes — game wall-clock, inference response, worker crash, pool catastrophe — remain textually distinguishable in logs.

- [ ] **Step 1: Write failing tests** in `test_selfplay_workers.py`:
  - `test_pool_timeout_accounts_for_multiple_jobs_per_lane` — `derive_pool_timeout(job_count=32, lane_count=30, per_game_timeout_s=900.0, grace_s=300.0) == 2100.0`; `job_count=32, lane_count=32` → `1200.0`; `job_count=64, lane_count=30` → `ceil(64/30)==3` → `3000.0`.
  - `test_a_single_timed_out_game_does_not_discard_completed_siblings` — build three jobs; give one a `selfplay_config` with `max_game_seconds` set so small it expires on the first clock read inside the child, and the other two normal near-terminal setups from the existing `_job` helper. Assert all three results return, exactly one has `completed is False` with `aborted_reason == "max_game_time_exceeded"` and `samples == ()`, and the other two are `completed` with non-empty samples. Because the budget expiry is driven by a genuinely tiny `max_game_seconds` (e.g. `1e-9`) rather than by waiting, the test is instant.
  - `test_pool_completes_and_leaves_no_children_after_a_game_timeout` — reuse the `active_children()` before/after assertion already used at `test_selfplay_workers.py:320`.
  - Keep `test_timeout_terminates_worker_and_leaves_no_children` passing unchanged, proving the catastrophic guard still cleans up every child.
- [ ] **Step 2: Run `python -m pytest tests/alphazero/orchestration/test_selfplay_workers.py -v`.** Expected failure: `AttributeError: type object 'SelfPlayWorkerPool' has no attribute 'derive_pool_timeout'`.
- [ ] **Step 3: Implement** `derive_pool_timeout`, the new constructor parameters, and the reworded catastrophic-timeout error. `import math` is stdlib; the module stays torch-free.
- [ ] **Step 4: Run `python -m pytest tests/alphazero/orchestration/test_selfplay_workers.py -v`.** Expected: all pass, including the pre-existing torch-free import test.
- [ ] **Step 5: Run `python -m pytest tests/alphazero tests/tools -q`.** Expected: green.
- [ ] **Step 6: Commit** `feat(alphazero): derive the pool guard from per-lane job load`.

---

### Task 6: Fresh-run fork from an external checkpoint

This is where finding 1 bites: the CPU checkpoint records `device: "cpu"` and `load_checkpoint` refuses to load it into a CUDA trainer. Rather than weaken that guard for everyone, add an explicit opt-in used only by the fork path.

**Files:**
- Modify: `src/diamond/alphazero/checkpoint.py`
- Create: `src/diamond/alphazero/run_fork.py`
- Modify: `tests/alphazero/test_checkpoint.py`
- Create: `tests/alphazero/test_run_fork.py`
- Modify: `.gitignore`

**Interfaces:**

```python
def load_checkpoint(path, trainer, *, expected, allow_device_migration: bool = False) -> CheckpointInfo:
    ...
```

When `allow_device_migration` is `True`, the device equality gate at `checkpoint.py:129-133` is skipped; every other gate (format version, semantic metadata, optimizer param-group lr/weight_decay, staged load) still runs unchanged. `torch.load` already uses `map_location=trainer.device`, so tensors land on the right device; AdamW's exponential-average states move with them. The returned `CheckpointInfo.training_config` still reports the *source* config — callers that care use `trainer.config`, which the loader overwrites at line 166. The fork therefore re-applies the destination `TrainingConfig` after loading (see below), so the new run's checkpoints record `cuda:0` and subsequent same-run resumes need no migration flag at all.

```python
@dataclass(frozen=True, slots=True)
class ForkProvenance:
    source_path: str
    source_sha256: str
    source_training_step: int
    source_device: str
    target_device: str

def initialize_run_from_checkpoint(
    *, run_root: Path, source: Path, trainer: AlphaZeroTrainer,
    expected: CheckpointCompatibilitySpec, operation_id: str,
) -> ForkProvenance:
    """Seed an uninitialized run directory from an external checkpoint.

    Raises FileExistsError if run_root/latest.pt already exists.
    """
```

Semantics, matching the spec's required table exactly:

- `run_root/latest.pt` exists → raise `FileExistsError`. An initialized run is never silently re-forked.
- otherwise → `load_checkpoint(source, trainer, expected=expected, allow_device_migration=True)`, restore `trainer.config` to the destination `TrainingConfig` (so the new `latest.pt` records the GPU device), then `save_checkpoint(run_root/"latest.pt", trainer, operation_id=operation_id)`.
- The source file is opened read-only and hashed; it is never written, moved, or truncated.
- Loop state, ledger, and replay store are **not** copied — the new run starts at iteration 0 with an empty replay, which is the point of a fork.
- `trainer.training_step` is inherited from the source, so training-step provenance stays continuous across the fork.

- [ ] **Step 1: Write failing checkpoint tests** in `tests/alphazero/test_checkpoint.py`: a checkpoint saved by a `cpu` trainer, loaded into another `cpu` trainer with `allow_device_migration=True`, still succeeds and still rejects mismatched *semantic* metadata (proving the flag relaxes only the device gate); loading a `device: "cpu"` checkpoint into a trainer whose `TrainingConfig.device` differs raises `CheckpointError` when the flag is absent and succeeds when it is set. Express the second case without needing CUDA by constructing the destination `TrainingConfig` with a device string that differs textually from the source — e.g. save with `device="cpu"` and load into a trainer built with `device="cpu:0"`, which `torch.device` normalizes differently enough to exercise the comparison on CPU-only hosts. Add a CUDA-gated variant asserting real `cpu` → `cuda:0` migration, skipped when `torch.cuda.is_available()` is false.
- [ ] **Step 2: Run `python -m pytest tests/alphazero/test_checkpoint.py -v`.** Expected failure: `TypeError: load_checkpoint() got an unexpected keyword argument 'allow_device_migration'`.
- [ ] **Step 3: Implement** the `allow_device_migration` parameter.
- [ ] **Step 4: Run `python -m pytest tests/alphazero/test_checkpoint.py -v`.** Expected: all pass.
- [ ] **Step 5: Write failing fork tests** in `tests/alphazero/test_run_fork.py`: forking into an empty `tmp_path` run creates `latest.pt`, preserves `trainer.training_step` from the source, and returns a `ForkProvenance` whose `source_sha256` equals the SHA-256 of the source bytes and whose `source_training_step` is read from the file (never hard-coded); the source file's bytes and mtime are unchanged after the fork; forking a second time into the same directory raises `FileExistsError` and leaves the existing `latest.pt` byte-identical; a run forked then loaded normally (no migration flag) resumes cleanly, proving the new run's checkpoint records the destination device; the new run's `loop_state.json`, `ledger.jsonl`, and `replay/` do not exist yet, i.e. the fork does not inherit the source run's history.
- [ ] **Step 6: Run `python -m pytest tests/alphazero/test_run_fork.py -v`.** Expected failure: `ModuleNotFoundError: No module named 'diamond.alphazero.run_fork'`.
- [ ] **Step 7: Implement** `run_fork.py`. Also uncomment `/runtime/` in `.gitignore` so new GPU run artifacts stay out of Git; do **not** `git rm --cached` the already-tracked CPU run in this task, so the existing run's history is untouched.
- [ ] **Step 8: Run `python -m pytest tests/alphazero/test_run_fork.py tests/alphazero/test_checkpoint.py -v`.** Expected: all pass.
- [ ] **Step 9: Run `python -m pytest tests/alphazero tests/tools -q`.** Expected: green.
- [ ] **Step 10: Commit** `feat(alphazero): fork a fresh run from an external checkpoint`.

---

### Task 7: CUDA FP32 parity tests

**Files:**
- Create: `tests/alphazero/test_cuda_parity.py`

No production code changes. This task exists to prove the CUDA path is numerically trustworthy before any GPU run is believed, and to give CPU-only CI a clean skip.

```python
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)
```

- [ ] **Step 1: Write the parity tests.** Build one `SooModel` with a fixed seed, save it through `save_checkpoint`, then load the *same* artifact twice via `load_inference_checkpoint` — once with `device="cpu"`, once with `device="cuda:0"` — and wrap each in a `TorchEvaluator`. Over a batch of at least 8 distinct `EvalRequest` values covering different legal-action counts, assert: legal-action key sets are **identical** (`set` equality, exact, no tolerance); prior probabilities agree within `rtol=1e-4, atol=1e-5`; each prior distribution sums to 1.0 within `1e-5`; scalar values agree within `rtol=1e-4, atol=1e-5`; a checkpoint saved from the CUDA-resident model reloads onto CPU and still matches, proving device-portable save/load. Reuse `assert_evaluation_agreement` from `diamond.alphazero.inference.profile`, which already exists for exactly this purpose (`test_numeric_modes.py:10`). Add one batching-invariance check: evaluating 8 requests as a single batch gives the same results as evaluating them one at a time, within the same tolerance — this is the property that centralized batching depends on and it is where a real GPU bug would surface.
- [ ] **Step 2: Run `python -m pytest tests/alphazero/test_cuda_parity.py -v` on this CPU-only host.** Expected: all tests **skipped**, zero failures, zero errors.
- [ ] **Step 3: Run the same command on the vast.ai RTX 3060 box** once Task 10 has provisioned it. Expected: all pass. If priors or values disagree beyond tolerance, stop the rollout — that is a correctness failure, not a tuning problem.
- [ ] **Step 4: Run `python -m pytest tests/alphazero tests/tools -q`.** Expected: green with the new tests skipped.
- [ ] **Step 5: Commit** `test(alphazero): assert CPU/CUDA FP32 evaluation parity`.

---

### Task 8: Streaming inference metrics and summaries

Fixes finding 5 (unbounded O(n²) accumulation) and gives the ledger the batch statistics the benchmark needs, without a second metrics system and without giant raw arrays in durable JSON.

**Files:**
- Create: `src/diamond/alphazero/inference/summary.py`
- Modify: `src/diamond/alphazero/inference/coordinator.py`
- Create: `tests/alphazero/inference/test_summary.py`
- Modify: `tests/alphazero/inference/test_coordinator.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class StreamingQuantile:
    """Fixed-memory reservoir over one latency/size series."""
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    reservoir: tuple[float, ...] = ()   # capped at RESERVOIR_CAPACITY = 2048

    def observe(self, value: float, *, rng: random.Random) -> "StreamingQuantile": ...
    @property
    def mean(self) -> float: ...
    def quantile(self, q: float) -> float | None: ...

def summarize_metrics(metrics: InferenceMetrics) -> dict[str, object]:
    """JSON-ready summary: counts, rates, batch p50/p90/mean/max, latency p50/p90."""
```

`InferenceMetrics` keeps its existing field names so no current consumer breaks, but the four unbounded `*_samples` tuples plus `batch_sizes` become `StreamingQuantile` instances behind properties that still expose `mean`/quantiles. Reservoir sampling with a fixed 2048 cap bounds memory at roughly one iteration's worth of samples regardless of request volume, and `_record_batch` stops rebuilding tuples, so its cost becomes O(batch) instead of O(total requests).

`summarize_metrics` emits exactly the fields the spec asks for: `requests_completed`, `batches_completed`, `mean_batch_size`, `median_batch_size`, `p90_batch_size`, `max_batch_size`, `queue_to_dispatch_p50_ms`/`_p90_ms`, `inference_p50_ms`/`_p90_ms`, `response_p50_ms`/`_p90_ms`, `evaluations_per_second`, `batches_per_second`.

- [ ] **Step 1: Write failing tests** in `tests/alphazero/inference/test_summary.py`: observing 1..100 gives `mean == 50.5`, `quantile(0.5) ≈ 50`, `quantile(0.9) ≈ 90` within a documented tolerance; the reservoir never exceeds `RESERVOIR_CAPACITY` after 100_000 observations while `count` is exact and `mean` stays within 1% of truth; `quantile` on an empty series returns `None` rather than raising; `summarize_metrics` on a coordinator that ran known batches returns exactly the documented key set, all values JSON-serializable (`json.dumps` round-trips), and **no key whose value is a list longer than 16**, enforcing the "no giant raw arrays in the ledger" constraint.
- [ ] **Step 2: Write failing coordinator tests** in `test_coordinator.py`: after submitting 5_000 requests through a stub evaluator, `coordinator.metrics` retains exact `requests_completed` and `batches_completed`, and the object graph holds no tuple longer than `RESERVOIR_CAPACITY` (guards against the O(n²) regression returning).
- [ ] **Step 3: Run `python -m pytest tests/alphazero/inference/ -v`.** Expected failure: `ModuleNotFoundError: No module named 'diamond.alphazero.inference.summary'`.
- [ ] **Step 4: Implement** `summary.py` and rework `_record_batch`. Keep `InferenceMetrics` frozen and keep `metrics` returning a snapshot copy.
- [ ] **Step 5: Run `python -m pytest tests/alphazero/inference/ -v`.** Expected: all pass, including the existing coordinator/profile tests unchanged.
- [ ] **Step 6: Run `python -m pytest tests/alphazero tests/tools -q`.** Expected: green.
- [ ] **Step 7: Commit** `perf(alphazero): summarize inference metrics in bounded memory`.

---

### Task 9: Hardware-neutral trainer entry point

`cpu_b0_train.py` is now misleading, but renaming it would break the documented CPU command and the existing test module. Resolution: move the loop into a shared module, keep `tools/cpu_b0_train.py` as a thin, still-working wrapper, and add `tools/az_train.py` as the hardware-neutral name. No duplicated implementation, no broken CPU workflow.

**Files:**
- Create: `tools/az_train.py`
- Modify: `tools/cpu_b0_train.py`
- Modify: `tests/tools/test_cpu_b0_train.py`
- Create: `runtime/configs/soo-rtx3060.json`

**Design:** `tools/az_train.py` holds `main()`, `LoopState`, `append_ledger`, `load_config`, `build_compatibility`, and `new_model` — moved verbatim from `cpu_b0_train.py` except for the changes below. `tools/cpu_b0_train.py` becomes:

```python
"""Backward-compatible alias for tools/az_train.py (see that module's docstring)."""
from az_train import (  # noqa: F401
    LoopState, append_ledger, build_compatibility, load_config, main, new_model,
)

if __name__ == "__main__":
    raise SystemExit(main())
```

Because `tests/tools/test_cpu_b0_train.py:17-22` loads the module by file path with `importlib`, the re-export keeps `runner.LoopState`, `runner.build_compatibility`, and `runner.new_model` resolving exactly as today; the loader must add the `tools` directory to `sys.path` so the `from az_train import ...` line resolves under that import style.

**New CLI arguments on `az_train.py`:**

```text
--init-checkpoint PATH   Seed an uninitialized run from an external checkpoint.
--workers N              Explicit worker count; default is available CPUs - 2.
--per-game-seconds S     Override self_play.max_game_seconds.
```

`--init-checkpoint` routes to `initialize_run_from_checkpoint` and is refused when `latest.pt` already exists; the resulting `ForkProvenance` is written into the `run_start` ledger record. Worker count resolves via `resolve_worker_count(config["workers"].get("worker_count"))` — note the existing configs set `worker_count: 4` explicitly, so **CPU behavior is unchanged**; the GPU config omits the key (or sets it to `null`) to opt into automatic resolution.

**`run_start` ledger record gains:**

```json
{"environment": {"torch": "...", "torch_cuda": "12.4", "cuda_available": true,
                 "cuda_device": "cuda:0", "gpu_name": "NVIDIA GeForce RTX 3060",
                 "vram_total_bytes": 12884901888, "available_cpus": 32,
                 "resolved_workers": 30},
 "fork_source": {"source_path": "...", "source_sha256": "...",
                 "source_training_step": 72, "source_device": "cpu",
                 "target_device": "cuda:0"}}
```

GPU fields come from `torch.cuda.get_device_properties(0)` and `torch.version.cuda` — no NVML, no new dependency. All are `null` when CUDA is unavailable.

**`iteration` ledger record gains:** `games_per_hour`, `completed_games_per_hour`, `samples_per_hour`, `training_steps_per_hour`, `p90_moves`, `abort_reasons` (per-iteration, alongside the existing cumulative counter), and `inference` (the `summarize_metrics` dict from Task 8). Existing fields keep their names and meanings so `tools/cpu_report.py` and the current ledger stay readable.

**`runtime/configs/soo-rtx3060.json`** — copied from `soo-cpu8h.json` with only these differences:

```json
{"training": {"device": "cuda:0", "batch_size": 256, ...},
 "inference": {"max_batch_size": 32, "max_wait_ms": 2,
               "request_queue_capacity": 1024, "response_timeout_s": 600.0},
 "self_play": {"max_moves": 2000, "max_game_seconds": 900.0, ...},
 "workers": {"worker_count": null, "games_per_iteration": 32, "retry_id": "attempt-0"}}
```

`network`, `replay`, `arena`, `model_version`, `run_seed`, and the rest of `self_play` are byte-identical to the CPU config, preserving the immutability the existing test asserts. `training.batch_size` stays 256 as instructed. Ratio preservation: 32 games/iteration with `--train-steps-per-iteration 8` keeps the shipped 16:4 = 4:1 games-per-update ratio exactly.

- [ ] **Step 1: Write failing tests** in `tests/tools/test_cpu_b0_train.py`: the existing module-level `runner` fixture still exposes `LoopState`, `append_ledger`, `build_compatibility`, `new_model` (regression guard on the alias); a new `az_runner` fixture loading `az_train` exposes the same names; `runtime_config("soo-rtx3060.json")` has `training.device == "cuda:0"`, `workers.games_per_iteration == 32`, `self_play.max_game_seconds == 900.0`, `inference.max_batch_size == 32`, and `mcts.simulations == 32`; the GPU config's `network`, `replay`, `arena`, and `model_version` equal `configs/alphazero/soo-bootstrap.json`'s, and its `self_play` equals the CPU config's `self_play` plus exactly the `max_game_seconds` key; `build_compatibility(runtime_config("soo-rtx3060.json"))` yields Soo/2 players; the GPU config's `worker_count` is absent-or-null so automatic resolution applies, while both CPU configs keep an explicit `worker_count == 4`. Extend `test_runtime_configs_keep_authoritative_identity` to skip the `device == "cpu"` assertion for the GPU config rather than weakening it for the CPU ones.
- [ ] **Step 2: Run `python -m pytest tests/tools/test_cpu_b0_train.py -v`.** Expected failure: `FileNotFoundError` on `soo-rtx3060.json` and `ModuleNotFoundError: No module named 'az_train'`.
- [ ] **Step 3: Implement** the module move, the alias, the three new CLI arguments, the ledger fields, and the GPU config file.
- [ ] **Step 4: Run `python -m pytest tests/tools/test_cpu_b0_train.py -v`.** Expected: all pass.
- [ ] **Step 5: Run `python -m pytest tests/alphazero tests/tools -q`.** Expected: green.
- [ ] **Step 6: CPU regression smoke** — `python tools/cpu_b0_train.py --config runtime/configs/soo-cpu8h.json --runtime-dir /tmp/az-regression --run-id cpu-smoke --hours 0.02 --phase B0 --train-steps-per-iteration 1 --simulations 4`. Expected: at least one iteration completes, `latest.pt` and `ledger.jsonl` are written, exit code 0, and the original `runtime/runs/soo/cpu8h-soo-20260819` is untouched (`git status` shows no new modification to it).
- [ ] **Step 7: Commit** `feat(alphazero): add hardware-neutral trainer entry point and RTX 3060 config`.

---

### Task 10: vast.ai provisioning and GPU smoke verification

The target is a rented vast.ai instance reached over SSH, and this box's torch is CPU-only (`torch.version.cuda is None`). Nothing GPU-related can be believed until this task passes on the real hardware.

**Files:**
- Create: `docs/gpu_training.md`

- [ ] **Step 1: Provision the instance.** Choose an RTX 3060 (12 GB) offer with ≥32 vCPUs and a CUDA 12.x image. Record the instance ID, image tag, and `nvidia-smi` output verbatim in `docs/gpu_training.md`.
- [ ] **Step 2: Install a CUDA torch build** — `pip install torch --index-url https://download.pytorch.org/whl/cu124` (match the driver reported by `nvidia-smi`), then `pip install --no-deps -e .` plus `pytest` and `trueskill`. Verify:
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  Expected: a non-`None` CUDA version, `True`, and `NVIDIA GeForce RTX 3060`. **If `cuda_available` is `False`, stop** — every later step is meaningless.
- [ ] **Step 3: Confirm the CPU allocation policy on the real box** — `python -c "from diamond.alphazero.hardware import available_cpu_count, resolve_worker_count; print(available_cpu_count(), resolve_worker_count())"`. Expected on a 32-CPU instance: `32 30`. If the container is cpuset-restricted, this is where affinity-awareness proves itself; record the actual numbers rather than assuming 30.
- [ ] **Step 4: Transfer the source checkpoint** by SHA-256, not by trust:
  ```bash
  scp runtime/runs/soo/cpu8h-soo-20260819/latest.pt vast:~/alphadiamond/seed/soo-step72.pt
  ssh vast 'sha256sum ~/alphadiamond/seed/soo-step72.pt'
  ```
  Expected digest: `4b2a32ff15179e890d4266346bca178d9a255eebe16af3a6e3d0482f0ceb1320`. A mismatch means a corrupt transfer; re-copy.
- [ ] **Step 5: Run the full suite on the GPU box** — `python -m pytest tests/alphazero tests/tools -q`. Expected: the same 377+ passing, and the Task 7 CUDA parity tests now **run rather than skip**, and pass.
- [ ] **Step 6: Fork the GPU run** (see CLI examples below) and confirm the new run directory has its own `latest.pt`, that `ledger.jsonl`'s `run_start` carries the recorded `fork_source` and GPU environment, and that the source seed file's digest is unchanged.
- [ ] **Step 7: 32-sim and 64-sim smoke runs** — a few minutes each, per the CLI examples. Expected: iterations complete, `nvidia-smi` shows non-trivial GPU utilization during self-play, mean batch size is reported, and no `max_game_time_exceeded` aborts appear at these short budgets.
- [ ] **Step 8: Write `docs/gpu_training.md`** with the provisioning steps, the verified environment block, and the exact commands used.
- [ ] **Step 9: Commit** `docs: record RTX 3060 provisioning and GPU smoke procedure`.

---

### Task 11: Benchmark execution and the CPU-vs-GPU learning comparison

- [ ] **Step 1: Throughput sweep.** From the same immutable seed checkpoint, run each configuration for an identical wall-clock budget (30 min suggested) into a *separate* run directory, so no configuration inherits another's replay: CPU baseline / 32 sims; RTX 3060 FP32 / 32 sims; RTX 3060 FP32 / 64 sims. Record from each ledger: `completed_games_per_hour`, `samples_per_hour`, `training_steps_per_hour`, median and p90 moves, abort counts by reason, and the `inference` batch/latency summary.
- [ ] **Step 2: Narrow secondary sweep**, only on the better simulation count from Step 1, and only if Step 1 shows the GPU is under-fed (mean batch size well below 32 or low GPU utilization): `max_batch_size` ∈ {16, 32} × `max_wait_ms` ∈ {1, 2, 5}. Six short runs, not a combinatorial explosion. Note that with ~30 workers doing synchronous leaf evaluation the achievable batch is bounded by concurrent workers — a mean batch far below 32 is expected, not a bug.
- [ ] **Step 3: Profile the `TorchEvaluator` transfer path** (finding 6). Time a fixed batch of 32 requests on CUDA, then compare against a variant that hoists the per-row `.cpu()` calls into one batched transfer. Only if this shows a material share of inference wall-clock, implement the batched transfer as its own TDD task guarded by the Task 7 parity tests. Otherwise record the measurement and stop.
- [ ] **Step 4: Fair learning experiment.** From the *same* seed checkpoint, train CPU and GPU for the same wall-clock interval (4 h suggested) with the same games-per-update ratio (4:1), then run the identical heuristic-off probe against both resulting checkpoints with identical `--episodes`, `--base-seed`, and simulation budgets. Compare off-probe completion rate, median moves, p90 moves, abort rate, and samples/training-steps generated. Treat loss as secondary information only — the spec is explicit that loss alone is not the success criterion, and the two runs see different data volumes, which makes their losses non-comparable in isolation.
- [ ] **Step 5: Record all results** in `docs/gpu_training.md` with the exact commands, run IDs, and checkpoint digests, so the comparison is reproducible.

---

## Test Matrix

| Requirement | Test | File | CPU-only |
|---|---|---|---|
| available − 2 resolution | `test_resolve_worker_count_reserves_two` | `test_hardware.py` | yes |
| minimum worker count ≥ 1 | `test_worker_count_floors_at_one` | `test_hardware.py` | yes |
| affinity-aware detection | `test_uses_sched_getaffinity_when_available` | `test_hardware.py` | yes |
| fallback detection | `test_falls_back_to_cpu_count_then_one` | `test_hardware.py` | yes |
| explicit override | `test_explicit_worker_count_wins` | `test_hardware.py` | yes |
| 32 games use ~30 lanes | `test_pool_timeout_accounts_for_multiple_jobs_per_lane` | `test_selfplay_workers.py` | yes |
| deadline value semantics | `test_deadline_boundary_and_remaining` | `test_deadline.py` | yes |
| deadline module is torch-free | `test_deadline_module_does_not_import_torch` | `test_deadline.py` | yes |
| MCTS stops between simulations | `test_expired_deadline_truncates_simulations` | `test_mcts_2p.py` | yes |
| no deadline ⇒ unchanged search | `test_absent_deadline_runs_every_simulation` | `test_mcts_2p.py` | yes |
| per-game 900 s abort | `test_game_deadline_aborts_the_episode` | `test_selfplay.py` | yes |
| timeout abort has zero samples | same test's `samples == ()` assertion | `test_selfplay.py` | yes |
| reason is `max_game_time_exceeded` | same test's `aborted_reason` assertion | `test_selfplay.py` | yes |
| `max_game_moves_exceeded` distinct | `test_move_cap_reason_remains_distinct` | `test_selfplay.py` | yes |
| one timeout keeps siblings | `test_a_single_timed_out_game_does_not_discard_completed_siblings` | `test_selfplay_workers.py` | yes |
| pool still cleans up children | `test_timeout_terminates_worker_and_leaves_no_children` (existing) | `test_selfplay_workers.py` | yes |
| lane-aware pool timeout | `test_pool_timeout_accounts_for_multiple_jobs_per_lane` | `test_selfplay_workers.py` | yes |
| fork from external checkpoint | `test_fork_creates_new_run_from_source` | `test_run_fork.py` | yes |
| source run untouched | `test_fork_never_modifies_the_source` | `test_run_fork.py` | yes |
| independent loop/replay/ledger | `test_forked_run_starts_with_no_history` | `test_run_fork.py` | yes |
| no accidental re-init | `test_reforking_an_initialized_run_raises` | `test_run_fork.py` | yes |
| normal resume still works | `test_forked_run_resumes_without_migration` | `test_run_fork.py` | yes |
| device-migration gate | `test_load_checkpoint_rejects_device_mismatch_by_default` | `test_checkpoint.py` | yes |
| CUDA FP32 parity | `test_cpu_and_cuda_agree_on_priors_and_values` | `test_cuda_parity.py` | **skips** |
| batching invariance | `test_batched_matches_single_request_evaluation` | `test_cuda_parity.py` | **skips** |
| device-portable save/load | `test_cuda_checkpoint_reloads_on_cpu` | `test_cuda_parity.py` | **skips** |
| metric summarization | `test_summarize_metrics_emits_json_ready_summary` | `test_summary.py` | yes |
| bounded metrics memory | `test_reservoir_never_exceeds_capacity` | `test_summary.py` | yes |
| ledger has throughput metrics | `test_iteration_record_contains_throughput_fields` | `test_cpu_b0_train.py` | yes |
| old configs stay valid | `test_runtime_configs_keep_authoritative_identity` (existing) | `test_cpu_b0_train.py` | yes |
| new GPU config valid | `test_rtx3060_config_is_valid_and_gpu_targeted` | `test_cpu_b0_train.py` | yes |
| reference configs still load | `test_reference_config_round_trips` (existing) | `test_reference_configs.py` | yes |

Every row is CPU-runnable except the three CUDA parity rows, which skip cleanly.

---

## Benchmark Procedure

All runs start from the identical seed checkpoint (`sha256:4b2a32ff…`) and write to separate run directories.

| Run | Device | Sims | Workers | Batch | Wait | Budget |
|---|---|---|---|---|---|---|
| `bench-cpu-32` | cpu | 32 | 4 | 64 | 5 ms | 30 min |
| `bench-gpu-32` | cuda:0 | 32 | auto (30) | 32 | 2 ms | 30 min |
| `bench-gpu-64` | cuda:0 | 64 | auto (30) | 32 | 2 ms | 30 min |

Primary metric is **useful throughput**, not `sec/game`: completed games/hour, non-aborted samples/hour, move-count p90, and off-probe completion. The historical heuristic-off context (32 sims: 29/30 complete, median ≈144, p90 ≈1667, one move-cap abort; 64 sims: 30/30, median ≈105, p90 ≈147) is orientation only — no test asserts these numbers. The open question the benchmark answers is whether 64 sims' more stable trajectories beat 32 sims' cheaper games on *useful* throughput.

Fairness rules: identical seed checkpoint; identical wall-clock budget; identical 4:1 games-per-update ratio; separate replay stores; identical off-probe seeds and episode counts; GPU utilization observed externally with `nvidia-smi` (no NVML dependency).

---

## Rollout Procedure

1. Tasks 1–9 land on a feature branch, each with its own commit, `pytest tests/alphazero tests/tools` green after every one.
2. Task 9 Step 6 proves the CPU workflow is unbroken on this machine.
3. Task 10 provisions the vast.ai box and proves CUDA parity on real hardware.
4. Fork the GPU run; confirm `runtime/runs/soo/cpu8h-soo-20260819` is byte-identical (compare `latest.pt` digests before and after).
5. Smoke runs at 32 and 64 sims.
6. Task 11 benchmark and learning comparison.
7. Merge only after CPU regression, GPU smoke, and CUDA parity all pass.

## Rollback Criteria

Stop and revert if any of these hold:

- CPU/CUDA parity fails beyond the stated FP32 tolerances, or legal-action sets differ at all.
- The CPU workflow regresses: `tools/cpu_b0_train.py` changes behavior on `soo-cpu8h.json`, or any existing test fails.
- A game timeout crashes an iteration, or a completed sibling episode is lost.
- Aborted games contribute a non-zero sample count.
- The forked run mutates the source run or the seed checkpoint's digest.
- GPU useful throughput fails to beat the CPU baseline — in that case keep the timeout/fork/metrics work (independently valuable) and revert only the GPU configuration.

Every task is an independent commit, so rollback is `git revert` of a contiguous range; the deadline, fork, and metrics work has no dependency on CUDA being used.

---

## Expected CLI Examples

**Fork a new RTX 3060 run from the existing CPU checkpoint**

```bash
python tools/az_train.py \
  --config runtime/configs/soo-rtx3060.json \
  --runtime-dir runtime/runs \
  --run-id rtx3060-soo-20260820 \
  --init-checkpoint runtime/runs/soo/cpu8h-soo-20260819/latest.pt \
  --phase A0 --bootstrap-prior none \
  --train-steps-per-iteration 8 \
  --hours 0.05
```

Creates `runtime/runs/soo/rtx3060-soo-20260820/` with its own `latest.pt`, `loop_state.json`, `ledger.jsonl`, `replay/`, `checkpoints/`, and `config.json`. Re-running with `--init-checkpoint` against the now-initialized run exits with an error rather than overwriting it.

**Short 32-simulation GPU smoke test**

```bash
python tools/az_train.py \
  --config runtime/configs/soo-rtx3060.json \
  --runtime-dir runtime/runs --run-id rtx3060-smoke-32 \
  --init-checkpoint runtime/runs/soo/cpu8h-soo-20260819/latest.pt \
  --simulations 32 --phase A0 --bootstrap-prior none \
  --train-steps-per-iteration 8 --hours 0.1
```

**Short 64-simulation GPU smoke test**

```bash
python tools/az_train.py \
  --config runtime/configs/soo-rtx3060.json \
  --runtime-dir runtime/runs --run-id rtx3060-smoke-64 \
  --init-checkpoint runtime/runs/soo/cpu8h-soo-20260819/latest.pt \
  --simulations 64 --phase A0 --bootstrap-prior none \
  --train-steps-per-iteration 8 --hours 0.1
```

**Fixed-duration GPU training session**

```bash
python tools/az_train.py \
  --config runtime/configs/soo-rtx3060.json \
  --runtime-dir runtime/runs --run-id rtx3060-soo-20260820 \
  --simulations 64 --phase A0 --bootstrap-prior none \
  --train-steps-per-iteration 8 --hours 4
```

No `--init-checkpoint`: the run already exists, so this is a normal resume of its own `latest.pt`.

**Heuristic-off probe against the resulting checkpoint**

```bash
python tools/cpu_off_probe.py \
  --config runtime/configs/soo-rtx3060.json \
  --checkpoint runtime/runs/soo/rtx3060-soo-20260820/latest.pt \
  --episodes 30 --simulations 32,64 --base-seed 9000 \
  --out runtime/runs/soo/rtx3060-soo-20260820/off-probe.json
```

The probe reads `training.device` from the config (`cpu_off_probe.py:82`), so this runs the probe on CUDA. To probe the GPU checkpoint on a CPU-only machine, point `--config` at `runtime/configs/soo-cpu8h.json` instead — same network and identity, `device: "cpu"`. Note the probe's `load_checkpoint` call enforces the device gate, so the config's device must match the checkpoint's recorded device; a GPU-trained checkpoint records `cuda:0`, which is why the CPU-side probe needs the CPU config **and** the `allow_device_migration` path from Task 6. Wire `cpu_off_probe.py` to pass `allow_device_migration=True` as part of Task 6 Step 7 so both directions work.

---

## Self-Review

**Requirements coverage.** All ten concerns in the prompt's recommended decomposition are covered: hardware resolution (1), deadline abstraction (3), MCTS propagation (4), pool timeout behavior (5), fork semantics (6), RTX 3060 config (9), CUDA FP32 correctness (7), inference metrics (8), benchmark tooling (10–11), and end-to-end CPU regression plus GPU smoke (9 Step 6, 10 Steps 5–7). Config schema (Task 2) was split out of the config task because it forces a repo-wide atomic change the prompt did not anticipate.

**Missing tests.** The initial draft lacked a batching-invariance test; without it, CPU/CUDA parity on single requests could pass while batched GPU inference silently corrupted results — the exact path this architecture depends on. Added to Task 7. Also added a bounded-memory regression test in Task 8, since the O(n²) fix is invisible to functional tests.

**Backward compatibility.** `max_game_seconds` defaults to `None` (old behavior exactly). `worker_timeout_s` still works and still wins. `cpu_b0_train.py` keeps its name, CLI, and importable symbols. Existing configs keep `worker_count: 4`, so CPU worker resolution is unchanged. `allow_device_migration` defaults to `False`, so the existing device gate is unweakened for every current caller.

**File/function names.** Every path, class, function, and line reference was read from the repository, not inferred. `SooSelfPlayRunner`, `MCTS2P._expand`, `SelfPlayWorkerPool.run`, `InferenceCoordinator._record_batch`, `load_inference_checkpoint`, and `assert_evaluation_agreement` all exist as cited.

**CPU-only CI.** Three tests are CUDA-gated and skip; everything else runs on the CPU box. The plan does not touch the pre-existing `qtawesome` collection failure, and says so explicitly so an implementer does not mistake it for regression.

**Timeout edge cases.** Boundary is inclusive at exactly the budget. `remaining_s` clamps at zero. Root expansion is never skipped, so a deadline-truncated search still returns a legal action. An expired game returns before appending a sample, satisfying `EpisodeResult`'s zero-samples invariant. The four failure modes stay textually distinct. No test waits on real time — expiry is driven by injected clocks or by an infinitesimal budget.

**Resume/fork safety.** Re-forking raises rather than overwriting. The source is opened read-only and digest-verified. The forked run inherits `training_step` but no replay, loop state, or ledger. After the fork, the new run's checkpoint records the destination device, so ordinary resume needs no migration flag.

**Benchmark fairness.** Same seed checkpoint, same wall-clock budget, same 4:1 games-per-update ratio, separate replay stores, identical probe seeds. Loss is explicitly demoted to secondary evidence because the two runs consume different data volumes.

**Placeholder language.** No `TODO`, `TBD`, "add tests", "handle errors", or "etc." appears in any task step.
