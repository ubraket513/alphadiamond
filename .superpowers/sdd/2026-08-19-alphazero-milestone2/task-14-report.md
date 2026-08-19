# AlphaDiamond Milestone 2 — Task 14 Report

Date: 2026-08-19. Start commit: `5500d753e03b36b0a2cc5a874a97d75cbac3ad94`.
Documentation commit before this report amendment:
`d26ec4417de9efaa087592acc5f372fae7699338`
(subject `docs: complete AlphaZero Milestone 2 runbook`). The final amended
HEAD is reported in the task handoff because a committed file cannot contain
the hash of the commit that creates its own final contents.

## Hardware

`nvidia-smi` was not found. AlphaZero environment CUDA detection exited 0 and
reported `torch_version=2.13.0+cpu`, `cuda_available=false`,
`cuda_version=null`, and `device_count=0`. CUDA/A30 integration, BF16, and
compiled GPU modes were not attempted.

## Verification

| Command | Exit | Result |
| --- | ---: | --- |
| `C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m pytest tests\alphazero -o addopts= -q` | 0 | `287 passed, 1 skipped in 17.99s` |
| `C:\Python314\python.exe -m pytest tests --ignore=tests/alphazero -o addopts= -q` | 0 | `134 passed, 5 skipped in 4.43s` |
| `C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m diamond.alphazero.smoke` | 0 | Soo/Min self-play, training step 1, checkpoint restore, and balanced arena JSON succeeded. |
| `C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m diamond.alphazero.milestone2_smoke` | 0 | `status=ok`; Soo: 2 workers, 4 rating events, eligible; Min: 2 workers, 0 events, insufficient history. |
| `C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m diamond.alphazero.orchestration.cli profile --runtime-dir .superpowers\sdd\2026-08-19-alphazero-milestone2\task-14-profile-runtime --model Soo --seconds 1` | 0 | CPU eager FP32 profile below. |
| `C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m compileall src\diamond\alphazero` | 0 | All AlphaZero package directories compiled. |
| `C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m pytest tests\alphazero\inference\test_remote.py::test_loading_mcts_modules_does_not_import_torch_rating_or_orchestration -o addopts= -q` | 0 | `1 passed in 0.18s` |
| `git diff --check` | 0 | No whitespace errors. |

The first attempted Qt-suite command used the current `python`, which resolved
to `C:\ProgramData\miniforge3\python.exe` and exited 1 because it has no
pytest. The verified Qt-working interpreter was `C:\Python314\python.exe`;
the requested command then passed as listed above.

## CPU profile evidence

Hardware: `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel`;
`gpu_verified=false`. One-second eager-FP32 profile: 1015.873 states/s and
calls/s; first call 0.016 s; 64 inference and queue samples; self-play 0.265
s, replay collation 0 s, training 0.016 s. `eager-bf16` was unavailable
(`CUDA BF16 support is unavailable`) and `compiled-fp32-reduce-overhead` was
unavailable (`CUDA is unavailable`); no GPU rows were fabricated.

This is a bounded tiny-service CPU profile, not production A30 evidence.
With no CUDA/A30 and no production A30 stage percentages, do not implement
C++ yet. Reconsider only after production A30 profiling shows a reproducible
CPU search/game/tree bottleneck dominating end-to-end time after centralized
eager inference.

## Changed documentation

- `docs/alphazero.md`: Milestone 2 command/env runbook, ratings/protocols,
  persistence, profile interpretation, and deferrals.
- `blueprint/milestone2.md`: implementation status/evidence section only.
- `blueprint/design.md`: implementation-progress status update required by the
  user.
- This report.
