# CUDA Device Runtime and Checkpoint Portability Implementation Plan

> **For Codex:** Execute after the native contract plan. Use TDD. CPU-host work must leave real CUDA cases pending, not mocked green.

**Goal:** Introduce one enforced LibTorch device runtime, correct batched inference/training on CPU or CUDA, separate actor/learner/candidate ownership, and portable model/AdamW archives.

**Architecture:** Torch-free parsing feeds `ResolvedDevice`; model roles own separate modules. Batches cross the host/device boundary once in each direction. Checkpoint loading is staged and explicitly migrates optimizer tensors.

**Tech stack:** C++20, LibTorch C++ API, CMake/CTest, CUDA-enabled LibTorch on the acceptance host.

**Spec:** `docs/superpowers/specs/2026-08-25-native-gpu-training.md`

## Task 1: Add the shared device resolver

**Files:**

- Add: `native/include/diamond_training/device.hpp`
- Add: `native/src/device.cpp`
- Add: `native/tests/device_test.cpp`
- Modify: `native/CMakeLists.txt`
- Modify: `native/src/train_main.cpp`

**Steps:**

- [ ] Write CPU tests for canonical `cpu`, grammar handoff, unavailable CUDA, and no run-directory mutation.
- [ ] Implement `ResolvedDevice` and one resolver using LibTorch CUDA availability/device count.
- [ ] Resolve once before constructing/loading models or touching a run directory.
- [ ] Report requested and canonical device separately.
- [ ] Add real-CUDA cases under the `cuda` CTest label.

**Acceptance:** build/run `device_test`; on this host CUDA cases must be explicitly not run by the CPU preset.

**Commit:** `feat(cuda): add explicit device resolution`

## Task 2: Establish deep model snapshots and weight identities

**Files:**

- Modify: `native/include/diamond_pipeline/model_pool.hpp`
- Modify: `native/src/model_pool.cpp`
- Modify: `native/include/diamond_training/trainer.hpp`
- Modify: `native/src/trainer.cpp`
- Modify: `native/tests/inference_coordinator_test.cpp`
- Modify: `native/tests/training_step_parity_test.cpp`

**Steps:**

- [ ] Prove with a failing test that copied `ModuleHolder` instances currently share parameters.
- [ ] Add a compatibility-aware deep snapshot helper for parameters and buffers.
- [ ] Construct immutable actor/champion and distinct mutable learner modules; set eval/train modes explicitly.
- [ ] Train the learner and prove actor bytes/weight digest do not change.
- [ ] Serialize a candidate snapshot and prove its key equals its canonical weight digest and differs after an update.

**Acceptance:** `inference_coordinator_test` and `training_step_parity_test`.

**Commit:** `feat(training): separate actor learner and candidate models`

## Task 3: Make ModelPool inference device-safe and batched

**Files:**

- Modify: `native/include/diamond_pipeline/model_pool.hpp`
- Modify: `native/src/model_pool.cpp`
- Modify: `native/tests/inference_coordinator_test.cpp`

**Steps:**

- [ ] Add representative multi-row Soo and Min CPU fixtures with ragged legal actions and exact order checks.
- [ ] Add non-finite, empty-legal-set, duplicate/range, and compatibility validation cases before forward.
- [ ] Collate contiguous features, padded legal indices, and mask buffers.
- [ ] Transfer buffers once, gather/mask/softmax on device, and make one compact result transfer.
- [ ] Remove host dereferences/per-row `.cpu()`/`.item()` from the batch loop.
- [ ] Add device-observation hooks used only by tests/metrics; add CUDA parity batches 1/17/64/256 under `cuda`.

**Acceptance:** `inference_coordinator_test` on CPU; CUDA matrix on the hardware host.

**Commit:** `feat(cuda): batch inference on the resolved device`

## Task 4: Move learner, targets, loss, gradients, and AdamW to the device

**Files:**

- Modify: `native/include/diamond_training/trainer.hpp`
- Modify: `native/src/trainer.cpp`
- Modify: `native/tests/training_step_parity_test.cpp`
- Modify: `native/benchmarks/training_step_benchmark.cpp`

**Steps:**

- [ ] Make device an explicit constructor dependency and move the learner before constructing AdamW.
- [ ] Transfer the three collated host buffers to the resolved device once per step.
- [ ] Keep loss math unchanged and return scalar metrics only after the complete step.
- [ ] Record collation/H2D/forward/backward/optimizer/total timings and peak CUDA memory when available.
- [ ] Add GPU residency and CPU/GPU one-step parity cases, including optimizer state, under `cuda`.
- [ ] Extend the benchmark with device, artifact/checkpoint, batch, warmups, repetitions, threads, and canonical JSON.

**Acceptance:** CPU parity test and one CUDA forward/AdamW test on hardware.

**Commit:** `feat(cuda): train entirely on the resolved device`

## Task 5: Make checkpoint archives device-portable

**Files:**

- Modify: `native/include/diamond_training/checkpoint.hpp`
- Modify: `native/src/checkpoint.cpp`
- Modify: `native/tests/checkpoint_v2_roundtrip_test.cpp`
- Modify: `native/tests/checkpoint_v2_transaction_test.cpp`
- Modify: `native/tests/checkpoint_v2_reject_test.cpp`

**Steps:**

- [ ] Stage all loads into a newly constructed model/optimizer before live mutation.
- [ ] Add explicit target device and migrate every AdamW tensor/state entry.
- [ ] Preserve CPU round-trip bytes/contracts and corruption rejection.
- [ ] Add the CPU→CUDA, CUDA→CUDA, and CUDA→CPU matrix under `cuda`.
- [ ] Prove interrupted generation/CURRENT activation cannot expose a partial checkpoint.

**Acceptance:** all checkpoint-v2 tests on CPU; portability matrix on CUDA.

**Commit:** `feat(checkpoint): support explicit device migration`

## Task 6: CUDA-runtime boundary verification

- [ ] Run the full CPU `native-training` preset once.
- [ ] Produce a CUDA-host command list with exact labelled tests and expected residency/digest/tolerance evidence.
- [ ] Do not claim this PR GPU-accepted until the JSON from a real NVIDIA run is attached.
