# Min 10-hour autopilot log — 2026-08-31

This is the durable hand-off log for the Min optimization, training, A0
transition, and v1.0.1 release session. Times are UTC.

## Starting state (18:00 UTC)

- Run: `/workspace/alphadiamond-training/runs/min/min-b0-6h-20260831`
- Durable state: iteration 23, `SELF_PLAY`; iteration 22 candidate is the active
  checkpoint (`training_step=23552`). Training was stopped at this durable
  boundary to run isolated GPU benchmarks.
- Active checkpoint model SHA-256:
  `f4fd7f351dce7e313d0c042e8f5e43fda75857ea2a85edae0ab78e526540a047`
- Replay: binary-only, about 1,000,027 samples; JSON chunks/samples are zero.
- Immutable production shape before the pending transition: 768 games, 512
  lanes, 16 search threads, batch cap 256, 50 us wait, 128 simulations, 500
  max moves, 1M replay, 1024 training steps, Arena disabled.
- Most recent complete iteration (22): self-play 240.322 s, replay 5.846 s,
  train 26.393 s, approximately 93,195 new samples.
- First binary-only cold-open (iteration 21): replay stage 36.447 s, of which
  replay open was 30.445 s. The previous mixed JSON cold-open was about 306 s.

## Implemented before benchmark sweep

- Binary replay schema v5 with mixed-format migration and automatic capacity
  pruning; all legacy JSON chunks have now aged out.
- Replay cache retained for the lifetime of the training process.
- Accepted episodes are moved through the pipeline without the former deep
  copy, with resize-based rollback.
- Arena can be disabled and a candidate can be activated directly.
- Deprecated trainer parameter-count API and associated warning removed.
- `selfplay_benchmark` now accepts `--checkpoint DIR --config FILE`, constructs
  the correct Min network/match, restores native checkpoint weights, and
  preserves the configured bootstrap prior. The benchmark schema contract and
  a CUDA smoke run against iteration 22 passed.

## Storage backup and cleanup (18:09–18:16 UTC)

- Bucket: `hf://buckets/ubraket513/AlphaDiamond`
- Uploaded completed iterations 0–21 to:
  `checkpoints/min/min-b0-6h-20260831/iterations/`
- Exact post-upload comparison passed:
  - files: 462 local = 462 remote
  - bytes: 417,448,499 local = 417,448,499 remote
  - `CURRENT`: 44 local = 44 remote
  - manifests: 44 local = 44 remote
  - remote iteration prefixes: 22
- Only after this comparison, local candidate/trained checkpoints for
  iterations 0–21 were deleted. Iteration 22, run metadata, and the active
  replay remain local. Disk improved from 69% used / 5.0 GB free to 67% used /
  5.4 GB free. The deleted checkpoint copies are recoverable from the bucket.

## Active benchmark sweep

- Production-shaped baseline completed: 1024 games, 512 lanes, 16 threads,
  batch 256, wait 50 us, 128 simulations, max 500 moves, iteration 22 Min
  checkpoint. Wall time 266.247 s; 1,024/1,024 games completed; 122,864
  samples; 1.661M samples/hour; mean batch 167.272 (p50/p90/max 256);
  58,779 evaluations/s; evaluator busy 96.99%; search-worker busy 16.32%.
  Evaluator time: forward 181.229 s, policy postprocess 23.758 s, D2H 19.385 s,
  H2D 6.394 s, collation 5.440 s, scatter 4.463 s.
- 24-thread comparison (all other fields and seed unchanged): 277.679 s,
  1.593M samples/hour, mean batch 167.148, evaluator busy 97.05%, worker busy
  13.08%, aborts 0. This is 4.12% slower in wall time / 4.12% lower sample
  throughput than 16 threads, so 24 threads is rejected unless a later
  batch/wait interaction reverses the result.
- 28-thread comparison: 298.517 s, 1.482M samples/hour, mean batch 169.956,
  evaluator busy 96.38%, worker busy 11.25%, aborts 0. It is 12.12% slower in
  wall time and 10.81% lower in sample throughput than 16 threads. The tiny
  batch increase does not offset contention and longer forward time. Thread
  winner: **16**. Increasing host load by adding search workers is rejected.
- Batch 256 / wait 100 us with 16 threads: 257.686 s, 1.716M samples/hour,
  mean batch 174.468, 60,733 evaluations/s, evaluator busy 96.21%, aborts 0.
  The sample stream is identical (122,864); versus 256/50 this is 3.32%
  faster in wall time and 3.32% higher throughput. Wait 100 us is the current
  winner pending the batch-384 interaction.
- Batch 384 / wait 100 us: 283.797 s, 1.559M samples/hour, mean batch 175.891,
  p50/p90/max 197/333/384, evaluator busy 94.76%, aborts 0. D2H rose from
  23.030 s at batch 256/100 to 33.928 s. It is 10.13% slower than 256/100 and
  is rejected. Completed samples differ by four (122,860), consistent with a
  batch-shape-dependent floating-point search path; this is an additional
  reason not to treat the larger cap as a drop-in parity-preserving win.
- Batch 384 / wait 50 us: 285.444 s, 1.550M samples/hour, mean batch 167.469,
  p50/p90/max 190/327/384, D2H 36.374 s, aborts 0. It is 7.21% slower than
  the original 256/50 baseline and 10.81% slower than 256/100. Batch cap 384
  is rejected in both wait configurations. Grid winner: **16 threads, batch
  256, wait 100 us**.
- Added inference-only FP16/BF16 actor support under TDD. The CUDA parity test
  compares real Min outputs against FP32 for 64 ragged rows (policy absolute
  error <0.02, each value component <0.05) and passed for both formats. Actor
  snapshots may be reduced precision; learner, optimizer, checkpoint, and
  replay remain FP32/native.
- FP16 at the grid winner (16/256/100): 236.852 s, 1.864M samples/hour,
  65,944 evaluations/s, mean batch 174.442, evaluator busy 95.85%, aborts 0.
  This is 8.80% faster in wall time and 8.58% higher sample throughput than
  FP32 16/256/100. The search path produced 122,616 samples versus 122,864 in
  FP32, so adoption still requires the BF16 comparison and a bounded gameplay
  stability gate.
- BF16 at 16/256/100: 286.217 s, 1.542M samples/hour, mean batch 147.141,
  54,574 evaluations/s, aborts 0. It is slower than both FP16 and FP32 and is
  rejected.
- CUDA Graph feasibility was rejected for this release. Current batch rows,
  legal-action width, output allocation, and tensor addresses vary per call;
  capture would first require fixed device buffers and shape bucketing. The
  256/100 distribution still contains a variable tail even though p50/p90 are
  full. Implementing that larger scheduler change before iteration 25 would
  add parity/reproducibility risk without measured evidence of a win.

## Config transition and real training resume

- Targeted tests passed: CPU/CUDA inference coordinator including FP16/BF16
  parity, config transition allow-list, CLI optional resume config, and all
  benchmark schemas (5/5 selected tests).
- Applied transition at durable iteration-23 `SELF_PLAY`, before any new
  self-play report existed. Original `resolved-config.json` remains untouched.
- Audit record: `config-transitions/23.json`.
  - from: `6998f0dbd2e44dcff19ac5993df9fedc26c42495c39fd9998ef6dd1c0c8df098`
  - to: `c5b7874c49e72fd72232c21126bea7dd5831bce91f02b7b2fa6d514008afe78a`
  - changed: `runtime.precision`, `workers.games_per_iteration`,
    `inference.max_wait_us`, `training.train_steps_per_iteration`
- Active production config: FP16 self-play actor; 1,024 games; 512 lanes; 16
  search threads; batch cap 256; 100 us wait; 128 simulations; 1M replay;
  1,408 FP32 learner steps (approximately 2.9 replay passes for the expected
  new sample count). Arena remains disabled during this optimized bootstrap
  continuation.
- `alphadiamond-train resume` is live from the iteration-22 checkpoint with
  optimizer state and replay preserved; this is not a scratch restart.
- Pending comparisons: threads 24/28; wait 100 us; batch cap 384; then
  FP16/BF16 parity and performance gates; then CUDA Graph feasibility/profile.
- A setting is adopted only if it improves completed-sample throughput without
  unacceptable abort, numerical-parity, or stability regressions.

## Additional disk cleanup (about 18:20 UTC)

- Removed disposable `/tmp` replay benchmark stores and package staging cache,
  plus the unused debug build. These were not training inputs or checkpoints.
- Disk moved from 67% used / 5.4 GB free to 63% used / 6.0 GB free.
- The apparent 11 GB container usage includes immutable environment layers:
  `/venv` is about 7 GB and NVIDIA tooling about 1.2 GB. Active local run/replay
  is 2.1 GB and the required release build is 1.2 GB.

## Periodic backup 1

- Synced iteration 22 and current run-state metadata to the HF bucket.
- Iteration 22 exact comparison: 21 files, 18,983,487 bytes, two `CURRENT`
  files, and two manifests on both local and remote sides.
- Remote run-state contains `state.json` (17,070 bytes) and
  `resolved-config.json` (1,319 bytes). No local deletion was needed.

## Remaining gates

1. Finish the throughput/precision/graph sweep and record all measurements.
2. Introduce an explicit, allow-listed config transition so immutable source
   checkpoint provenance is never rewritten. Planned production shape is 1024
   games and 1408 training steps (about 2.9 passes), plus measured winning
   throughput fields.
3. Resume from iteration 23 and persist through iteration 25, backing up and
   checking storage approximately every 30 minutes.
4. Run a bounded Arena evaluation at a checkpoint boundary. Remove the
   bootstrap prior and enter genuine A0 only when the candidate demonstrates
   stable non-attractor play; document the quantitative gate and preserve the
   pre-transition checkpoint.
5. Verify checkpoint/replay/run state and the full native test suite, merge
   accepted work to `main`, audit and remove obsolete branches, tag
   `min-v1.0.1`, publish the GitHub release, and upload release checkpoint,
   replay, and run-state archives.
## Iteration 24 completion and FP16 safety rollback

- Iteration 24 completed and persisted at training step 26,368. Its self-play produced 124,364 new samples with zero aborted games.
- Iteration 25 initially stopped at its durable `SELF_PLAY` boundary with `native model produced a non-finite inference row`; no replay or checkpoint state was lost.
- The failure was reproduced deterministically from the iteration 24 checkpoint in FP16. All stored FP32 parameters were finite and had maximum absolute value 2.366, ruling out checkpoint corruption or direct parameter-cast overflow.
- The same checkpoint completed an FP32 control workload (128/128 games, 15,440 samples, zero aborts), identifying FP16 intermediate-activation overflow as the failure mode.
- BF16 was not selected because the earlier production-shape benchmark was slower than FP32 (286.217 s versus 257.686 s). Runtime precision was therefore transitioned from FP16 back to FP32 at iteration 25. The provenance record is `config-transitions/25.json` (`c5b7874...` to `524b888a...`).
- Training resumed from the iteration 24 champion and the existing 1M-capacity binary replay at iteration 25; it did not restart from scratch.

## Iteration 25 release gate

- Iteration 25 completed 1,024/1,024 games with zero aborts in 280.169 s and produced 125,542 samples.
- Replay ingest accepted all 125,542 samples, retained exactly 1,000,000 samples, and took 38.717 s, of which 30.762 s was the explicit process-restart cold open. The following 1,408-step training stage reused the live replay object (`replay_cache_hit=true`) and took 34.906 s.
- The candidate persisted at training step 27,776 with model digest `847007a72b0a283a1789c7cb160f9fd93864117ad1cdbf5bbf3541c92f5ef59a`. Transactional checkpoint validation reported format v3, optimizer restored, and `valid=true`.
- Release digests: checkpoint tree `51579b816bfed34378ebf89233717eecdfb713ee294ce9e85bef927017d79c1e`; state `d32d88e8a28d6295efb76c6402d9bdae8e435392250e46a1ed68e9e73e3fb776`; replay manifest `dc0ce2efea60fc4b4b3ce1a21fe36f60bf7ff5b0fb916489a127e1988ae3ee59`.
- The replay manifest contained 8,223 `binary-v1` chunks and zero `json-v1` descriptors. No legacy JSON chunk files remained.
- Iterations 24 and 25 were uploaded to `hf://buckets/ubraket513/AlphaDiamond` and verified exactly before cleanup: 22 files / 277,072,477 bytes and 22 files / 279,381,277 bytes respectively. Their recoverable `selfplay.episodes` files were then removed locally, reducing the run to 2.1 GB and disk use to 63%.

## Iteration 25 incremental Arena

- A complete 36-game scheduled Arena was attempted with iteration 25 as candidate and iteration 24 as champion, using the active FP32 config.
- Three games aborted, leaving one incomplete opening block; promotion statistics were therefore correctly marked invalid rather than extrapolated from partial data.
- Across the 33 completed candidate seats, iteration 25 placed first 13 times, second 11 times, and third 9 times. This is mildly positive descriptive evidence but not a valid promotion result and is insufficient by itself to authorize removal of the B0 bootstrap prior.

## A0 transition gate after release

- A bounded no-prior probe started from the released iteration-25 checkpoint with 256 games, 128 lanes, 16 search threads, 128 simulations, and an 800-move cap.
- The probe had not completed after more than ten minutes, despite being only one quarter of a production iteration's game count. GPU utilization remained roughly 29%–35%, compared with about 79%–82% during B0 self-play. This is operational evidence of long no-prior game tails and collapsing batch occupancy.
- The probe was terminated rather than spending further training time on an already failed throughput/stability gate. A0 was not adopted. B0 training resumed at iteration 26 from the released iteration-25 champion, step 27,776, and the same 1M binary replay.

## End-of-session learning assessment

- Extending the unchanged B0 loop is unlikely to produce a large strength gain. From iterations 22 through 25, policy loss stayed effectively fixed at approximately `6.063785` across more than 5,000 optimizer steps. Value loss fluctuated around `0.58`–`0.61` without a sustained downward trend.
- The iteration-25 Arena placements (13 first, 11 second, 9 third among 33 completed games) are mildly positive but too small and incomplete to establish a meaningful improvement over iteration 24.
- B0 remains useful as a stable replay generator and may yield marginal gains, but additional iterations alone do not address the apparent policy-learning plateau. Before another long run, measure MCTS visit-target entropy, policy-head gradient norms, and policy KL divergence between adjacent checkpoints. These distinguish an uninformative target distribution from a disconnected/saturated policy head or simply tiny updates.
- The session stopped iteration 26 during `SELF_PLAY` before ingest or training. Durable state remains at the released iteration-25 champion, training step 27,776; iteration 26 contains only `initialize.json`. Resuming the run will restart iteration-26 self-play without losing checkpoint or replay state.
