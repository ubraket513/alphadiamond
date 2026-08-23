# The replay path: measured before porting anything

Measured 2026-08-24 on the development machine (Windows, CPython 3.12, CPU
tensors) with `az-bench/profiles/bench_replay_pipeline.py`, 20 000 samples,
batch 512, action space 5329.

Milestone 5 of the native migration says to port only what is measured to be
expensive. So this is the measurement, taken before deciding.

## What it costs

| Step | Cost |
|---|---|
| Ingest into the replay buffer | 2.98 µs/sample |
| Draw a batch of 512 | 7 ms |
| `ReplayBuffer.collate` (dense policy rows) | 41.5 ms |
| `torch.tensor(batch.policy_targets)` in the trainer | 268.2 ms |
| **Total per training step** | **≈310 ms** |

Ingestion is not a bottleneck and does not need porting: three microseconds per
sample is invisible next to a search that spends milliseconds per move.

The batch path is a different story. A 512-sample batch materialises
512 × 5329 = 2.7 million Python floats -- once when `collate` builds a dense
tuple per sample, and again when `torch.tensor` walks those tuples. The policy
itself is sparse: about 24 visited actions per sample, or 0.45 % of the row.

## What the fix is -- and is not

Scattering the sparse policy straight into a zeroed tensor:

```
current path (collate + torch.tensor):  309.7 ms per batch
scatter straight to a tensor:            13.6 ms per batch
speedup:                                 22.8x, values identical (max diff 0.0)
```

**The port target here is torch, not C++.** A C++ implementation of the same
dense-materialisation strategy would have optimised the wrong thing; the cost
was never the language, it was building 2.7 million Python objects to describe
12 thousand non-zero probabilities. This is the case the "profile first" rule
exists to catch.

Implemented as `AlphaZeroTrainer.train_samples`, which the production training
stage now uses. `train_batch` and `ReplayBatch` stay: they are the readable
reference, and `tests/alphazero/test_trainer_sample_path.py` asserts both paths
produce the same loss on the same samples, so the fast path cannot quietly
become a different training step.

## Still unmeasured

* `PersistentReplayStore.load_buffer()` is called once per training step in the
  production stage. On a large store that is a re-read per step; it was out of
  scope here and is the next thing worth timing.
* Arena game execution and dataset transformations: no measurement yet, so no
  port.
