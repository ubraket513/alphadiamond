# Binary Replay Segments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace new replay JSON chunks with deterministic, transactional binary-v1 segments while retaining schema 4 read compatibility.

**Architecture:** A focused binary codec owns byte layout and validation. `ReplayStore` selects a codec from each manifest descriptor, writes schema 5 manifests, and continues reading mixed old/new stores. Existing `TrainingSample` ownership remains unchanged in phase 1.

**Tech Stack:** C++20, SHA-256 support library, canonical JSON manifests, CMake/CTest

**Spec:** `docs/superpowers/specs/2026-08-31-binary-replay-segments-design.md`

## Global Constraints

- New segment integers are explicitly little-endian and floats are IEEE-754 float32.
- Readers accept schema 4 JSON and schema 5 mixed descriptors; writers emit binary-v1 only.
- Segment activation precedes atomic manifest replacement.
- The replay capacity remains 1,000,000 samples and failures do not copy the resident replay pool.
- No new third-party serialization dependency is introduced.

---

### Task 1: Deterministic binary-v1 codec

**Files:**
- Create: `native/include/diamond_pipeline/replay_segment.hpp`
- Create: `native/src/replay_segment.cpp`
- Create: `native/tests/replay_segment_test.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Consumes: `Episode`, `Compatibility`, `TrainingSample`, `diamond_support::sha256`
- Produces: `std::vector<std::byte> encode_replay_segment(const Episode&, const Compatibility&)` and `Episode decode_replay_segment(std::span<const std::byte>, const Compatibility&)`

- [ ] **Step 1: Write the failing codec tests** for deterministic bytes, full field round trip, truncation at every structural boundary, corrupt offsets, checksum mismatch, non-finite floats, and compatibility mismatch.
- [ ] **Step 2: Run** `cmake --build build/native-training --target replay_segment_test -j2` and verify the target or symbols are absent.
- [ ] **Step 3: Implement the codec** with checked little-endian append/read helpers, fixed header/index records, length-prefixed arrays, SHA-256 footer, and validation before allocation.
- [ ] **Step 4: Run** `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./build/native-training/native/replay_segment_test` and verify PASS.

### Task 2: Schema 5 mixed-format replay store

**Files:**
- Modify: `native/src/replay_store.cpp`
- Modify: `native/tests/replay_store_test.cpp`

**Interfaces:**
- Consumes: Task 1 codec functions
- Produces: schema 5 descriptors with `encoding`, extension-aware load/cleanup, and binary-only new writes

- [ ] **Step 1: Add failing integration tests** that reopen the existing schema 4 fixture, add a binary episode, reopen the mixed store, verify deterministic samples, reject a corrupt binary segment, and preserve reachable files during cleanup.
- [ ] **Step 2: Run** `ctest --test-dir build/native-training -R replay_store_test --output-on-failure` and verify schema 5 expectations fail.
- [ ] **Step 3: Add descriptor encoding state** beside each digest, accept manifest versions 4 and 5, dispatch JSON or binary decoding, emit `.bin` for new episodes, and make duplicate checks and cleanup encoding-aware.
- [ ] **Step 4: Preserve transactions** by activating the binary segment before manifest replacement and rolling appended state back by vector sizes on failure.
- [ ] **Step 5: Run** the replay segment/store tests and verify PASS.

### Task 3: Timed ingest and reopen comparison

**Files:**
- Modify: `native/benchmarks/replay_benchmark.cpp`
- Modify: `native/tests/benchmark_schema_test.cmake`

**Interfaces:**
- Consumes: schema 4 fixture mode and schema 5 default writer
- Produces: JSON fields `ingest_seconds`, `reopen_seconds`, `encoding`, `bytes_written`, and sample timing summary

- [ ] **Step 1: Extend the benchmark contract test** to require the new timing and encoding fields and reject negative/non-finite values.
- [ ] **Step 2: Run** `ctest --test-dir build/native-training -R benchmark_schema_test --output-on-failure` and verify the missing fields fail.
- [ ] **Step 3: Time ingest and reopen** with `steady_clock`, measure reachable chunk bytes, and retain the existing sampling fields.
- [ ] **Step 4: Run equal-pool JSON and binary trials** and save the raw JSON outputs with the Min optimization report.

### Task 4: Boundary deployment and verification

**Files:**
- Modify: `docs/superpowers/reports/2026-08-31-min-training-optimization.md`

**Interfaces:**
- Consumes: durable Min checkpoint, Task 3 timings, current run reports
- Produces: resumed Min process and before/after iteration metrics

- [ ] **Step 1: Wait for a durable iteration boundary** and record PID, iteration, checkpoint digest, replay size, and training step.
- [ ] **Step 2: Run** `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ctest --test-dir build/native-training --output-on-failure` and require 38/38 or the updated total to pass.
- [ ] **Step 3: Stop only the current trainer process**, rebuild the production target, and resume the same run directory/config without deleting state.
- [ ] **Step 4: Compare the first completed schema 5 iteration** against iterations 0 and 1 for self-play, ingest, replay-open, training, total core time, and samples/sec.
- [ ] **Step 5: Record rollback instructions:** resume the same checkpoint with the prior binary; schema 5 remains readable by the new binary and no scratch restart is performed.

## Self-review

- Spec coverage: codec, compatibility, transactions, cleanup, corruption, benchmarks, and boundary rollout each map to a task.
- Placeholder scan: no deferred implementation choices remain.
- Type consistency: Task 2 consumes the exact encode/decode signatures produced by Task 1; Task 3 observes Task 2's `encoding` value.
