# Python-zero benchmark disposition

PR11 keeps five decision-useful native measurements. They run explicitly and
emit JSON; none is a correctness gate or an elapsed-time pass/fail test.

```bash
cmake -S . -B build/native-bench -DDIAMOND_BUILD_LIBTORCH=ON -DDIAMOND_BUILD_BENCHMARKS=ON
cmake --build build/native-bench --target training_step_benchmark checkpoint_benchmark replay_benchmark selfplay_benchmark end_to_end_benchmark
build/native-bench/native/training_step_benchmark --repetitions 5
build/native-bench/native/checkpoint_benchmark --repetitions 5 --scratch build/bench-checkpoint
build/native-bench/native/replay_benchmark --repetitions 5 --scratch build/bench-replay
build/native-bench/native/selfplay_benchmark --repetitions 5
build/native-bench/native/end_to_end_benchmark --repetitions 5 --scratch build/bench-e2e
```

## Tool disposition

| Retired Python tool | Native replacement or disposition |
|---|---|
| `bench_block_vectorization.py` | Retired compiler micro-probe; production work is measured by `selfplay_benchmark`. |
| `bench_bridge_search.py` | Retired with the Python/native bridge. |
| `bench_evaluator.py` | `selfplay_benchmark` |
| `bench_forward_launch.py` | `selfplay_benchmark` |
| `bench_native_callback.py` | Retired with callback transport. |
| `bench_native_gpu_ab.py` | `selfplay_benchmark` |
| `bench_native_scheduler.py` | `selfplay_benchmark` |
| `bench_native_single_search.py` | `selfplay_benchmark` |
| `bench_parent_cycle.py` | `end_to_end_benchmark` |
| `bench_policy_value_callback.py` | Retired with callback transport. |
| `bench_production_stages.py` | `end_to_end_benchmark` |
| `bench_replay_pipeline.py` | `replay_benchmark` |
| `bench_transport_cost.py` | Retired with the Python/native bridge. |
| `bench_values_only.py` | `selfplay_benchmark` |
| `bench_worker_gap.py` | `selfplay_benchmark` |
| `bench_worker_mcts.py` | `selfplay_benchmark` |
| `make_bench_configs.py` | Retired; benchmark arguments are explicit. |
| `sample_run.py` | Retired raw sampling helper. |
| `summarize_batchfix.py` | Retired historical report generator. |
| `summarize_scaling.py` | Retired historical report generator. |

## Comparison evidence

| Contract | Python Phase 0 | Native command | Status |
|---|---|---|---|
| training step | No controlled run recorded | `training_step_benchmark` | Ready to measure |
| checkpoint save/validate | No controlled run recorded | `checkpoint_benchmark` | Ready to measure |
| replay ingest/sample | Semantic baseline only | `replay_benchmark` | Ready to measure |
| self-play | Historical runs use different workloads | `selfplay_benchmark` | Ready to measure |
| end-to-end iteration | No controlled run recorded | `end_to_end_benchmark` | Ready to measure |

No numeric comparison is fabricated from incompatible historical runs. PR13
records final medians/ranges using one pinned machine and workload.

The replay-store decision remains unchanged: descriptors are authoritative on
disk and the loaded sample window is bounded by persisted capacity.
