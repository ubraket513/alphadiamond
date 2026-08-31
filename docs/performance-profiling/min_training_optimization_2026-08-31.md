# Min training optimisation record — 2026-08-31

This note records the optimisations actually adopted by the first production-shaped Min B0 run, the evidence used to keep them, and the experiments deliberately deferred until the legacy JSON replay has retired. It is an operational record, not a list of every idea considered.

Run under measurement:

```text
/workspace/alphadiamond-training/runs/min/min-b0-6h-20260831
```

The run always resumed the same transactional checkpoint. None of the measurements below restarted Min from scratch.

## Fixed training invariants

These were treated as quality/stability constraints rather than throughput knobs:

| setting | adopted value | reason |
|---|---:|---|
| replay capacity | 1,000,000 samples | reduce the risk of replay collapsing into a short-lived attractor |
| MCTS simulations | 128 per move | measured as more stable than shallower search |
| replay passes | approximately 2.9 | 1,408 updates × batch 256 over roughly 124k–126k new samples per iteration |
| checkpoint cadence | every iteration | exact resume and release provenance |
| Min scratch value head | exactly zero | removes arbitrary seat preference and improved the move/repetition tail |
| bootstrap prior | `canonical-target-vacancy-distance-v2` | the Soo policy-head transfer probe was rejected; the native Min prior completed reliably |

The active six-hour config is [min-production-6h.json](../../configs/alphazero/min-production-6h.json). Its production operating point is 1,024 games, 512 logical lanes, 16 search threads, batch cap 256, 100 µs maximum wait, and a 1,024-entry request queue on CUDA FP32.

## Adopted optimisations

### Production-shaped self-play scheduling

Min inherited the measured Soo scheduler shape: jobs outnumber lanes, finished lanes immediately take queued games, and inference requests are centrally batched. This avoids the end-of-run collapse produced by assigning one lane permanently to each game. Pinned, reused inference staging buffers and the directional trunk implementation were already part of the native baseline and remain enabled.

Recent steady-state self-play measurements were 203.8–249.0 seconds for about 91k–95k moves. Evaluator busy fraction was approximately 96%–97%; therefore the remaining throughput variation tracks batch occupancy and game-length tails rather than an idle evaluator.

### Arena removal from the B0 production loop

`arena.enabled=false` bypasses promotion play and rating calculation for this B0 run, promotes each newly trained candidate directly, and persists a durable promotion record. This preserves checkpoint progression while removing a stage that previously cost minutes and did not answer the B0 training question.

The batched arena scheduler and seeded repetition escape remain implemented and tested for runs that enable the arena. They are documented separately in [min_arena_throughput.md](../model-training/min_arena_throughput.md).

### One replay object per trainer process

`TrainingRunResources` lazily constructs one full `ReplayStore` and reuses it across replay ingest and training stages and across iterations. Metadata-only opens remain available for paths that only need identity or manifest data. In a steady process, `replay_open_seconds` is effectively zero instead of rehydrating the full capacity multiple times per iteration.

This optimisation does not hide restart cost: an intentional process restart produces a measurable cold open, recorded separately below.

### Binary replay segments

New completed games are written as deterministic, content-addressed `binary-v1` segments. Schema 5 manifests can reference both legacy `json-v1` chunks and binary chunks, so the active run migrated without discarding its replay or checkpoint.

The binary codec provides:

- explicit little-endian integer and float32 encoding;
- a compatibility digest and per-segment SHA-256 integrity checks;
- a fixed sample offset/length index;
- atomic segment activation before manifest replacement;
- strict bounds, finite-value, compatibility, checksum, and truncation validation.

The average segment size in the first measured binary iteration was about 253 KB, versus about 308 KB for the existing JSON chunks, an approximately 18% reduction.

### No full-pool transaction copies

Replay ingest holds pointers to accepted episodes instead of copying their sample vectors. Failure rollback records vector sizes and restores them with `resize`; it no longer snapshots the resident replay pool. Segment bytes and digests are constructed once per accepted episode.

Capacity pruning follows the same rule. The next retained episode prefix is computed before the atomic manifest commit. Only after that commit succeeds are old in-memory descriptors erased and unreachable chunk files deleted. The transaction therefore never copies the 1M-sample pool.

### Automatic capacity pruning and progressive JSON retirement

Before this fix, the in-memory pool was truncated to 1M samples but the manifest and chunk directory grew without bound because `ReplayStore::prune()` had no caller. Ingest now commits only the newest capacity window and removes unreachable files after commit.

The first production prune at iteration 14 changed the durable replay as follows:

| measure | before | after |
|---|---:|---:|
| manifest samples | 1,298,196 | 1,000,002 |
| JSON samples | 927,813 | 535,401 |
| JSON chunks | 7,680 | 4,412 |
| binary samples | 370,383 | 464,601 |

Each subsequent iteration adds roughly 92k–95k binary samples and evicts the oldest JSON prefix. No eager migration rewrites old training data; normal training retires it.

### Trainer warning and log-I/O removal

The trainer used deprecated `torch::optim::Optimizer::size()` once per training step, producing 1,024 warnings per iteration. State coverage now sums `param_groups()[].params().size()` directly. `training_step_parity_test` passes without emitting the deprecated warning. The warning-free binary is staged for activation only after the JSON-retirement gate, so the active run is not restarted merely to remove logging noise.

## Measured replay result

The fair comparison is warm-cache ingest before and after binary segments plus copy-free transactions:

| iteration | format/path | replay ingest | replay open | accepted samples |
|---:|---|---:|---:|---:|
| 8 | JSON, warm cache | 49.28 s | ~0 s | 94,279 |
| 9 | JSON, warm cache | 48.15 s | ~0 s | 92,076 |
| 10 | first binary write, cold legacy open | 290.94 s | 285.18 s | 93,816 |
| 11 | binary, warm cache | 6.10 s | ~0 s | 91,140 |
| 12 | binary, warm cache | 5.72 s | ~0 s | 92,423 |
| 13 | binary, warm cache | 5.83 s | ~0 s | 93,004 |
| 14 | pruning binary, cold mixed open | 312.72 s | 306.31 s | 94,218 |
| 15 | pruning binary, warm cache | 5.99 s | ~0 s | 94,688 |

Against iteration 9, iteration 11 reduced warm replay ingest from 48.15 seconds to 6.10 seconds: 7.89× faster, or an 87.3% reduction. The cold iterations are not regressions in ingest; almost all of their wall time is the explicit reconstruction of the mixed legacy replay after a process restart.

## Current steady-state budget

Excluding a cold-open iteration, recent core stage times were:

| iteration | self-play | replay ingest | train | core total |
|---:|---:|---:|---:|---:|
| 11 | 211.48 s | 6.10 s | 26.47 s | 244.05 s |
| 12 | 203.79 s | 5.72 s | 26.51 s | 236.02 s |
| 13 | 237.02 s | 5.83 s | 26.35 s | 269.20 s |
| 15 | 249.03 s | 5.99 s | 26.35 s | 281.36 s |

Replay is now about 2%–3% of a normal iteration. Self-play is the dominant remaining cost; replay sampling itself is only about 0.8 seconds across all 1,024 training steps.

## Post-JSON operating-point experiments

The JSON-retirement gate was reached before iteration 25, after which the approved production-shaped sweep was run:

1. Games increased to 1,024 while retaining 512 lanes, and training increased to 1,408 steps to preserve approximately 2.9 passes.
2. Sixteen search threads beat 24 and 28 threads. At batch 256, increasing maximum wait from 50 to 100 µs improved throughput by 3.32%; batch 384 regressed at both wait settings and was rejected.
3. FP16 initially improved production-shaped self-play from 257.686 s to 236.852 s (8.58%). BF16 took 286.217 s and was rejected. After two full training iterations, the iteration-24 model deterministically overflowed an intermediate FP16 activation. Stored weights remained finite and within FP16 range, while the same checkpoint completed 128/128 FP32 control games with zero aborts. The active run therefore transitioned back to FP32 at iteration 25.
4. CUDA Graph capture was rejected for this release because batches have dynamic row counts, legal-action widths, and tensor addresses. Safe capture requires fixed staging/output buffers plus shape buckets and remains a later architectural experiment.

The iteration-25 release checkpoint was trained at the adopted scheduling point with FP32 inference. Precision transitions are recorded without rewriting the original resolved config.

## Release gate

The approved release target is GitHub release `Min v1.0.1` at completed iteration 25. Release requires all of the following:

- JSON replay count is zero;
- iteration 25 has completed `persist`;
- the iteration-25 checkpoint validates and its training step and SHA-256 are recorded;
- the full native suite passes in the runtime environment;
- accepted feature work is merged into `main` without rewriting history;
- release archives are independently hashed and extraction-tested.

The detailed integration and publication sequence is in [2026-08-31-min-v1.0.1-release.md](../superpowers/plans/2026-08-31-min-v1.0.1-release.md).

## Validation coverage

The adopted replay path is covered by schema compatibility, binary codec, replay store, coordinator resume, CLI contract, native pipeline smoke, and training parity tests. The complete native suite passed 39/39 with the required runtime preload:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  ctest --test-dir build/native-training --output-on-failure
```

The preload is an environment requirement of the installed LibTorch wheel; without it, its bundled libstdc++ can interpose incompatible `std::filesystem` symbols and make checkpoint/smoke tests fail with SIGSEGV.
