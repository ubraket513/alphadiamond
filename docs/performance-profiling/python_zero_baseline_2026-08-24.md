# Python-Zero Migration — PR 00 Baseline

**Status:** BLOCKED — first prescribed correctness command could not be started.

**Recorded at:** 2026-08-24 (Windows PowerShell host)

## Baseline identity (captured before any tracked edit)

| Field | Recorded value |
| --- | --- |
| Worktree | `C:\tmp\codex-alphadiamond-pr00` |
| Branch | `codex/python-zero-pr00-baseline` |
| HEAD | `f0f4a8a7a3928a3770afe16d5e1e246001961759` |
| Tracked Python paths | `183` |
| Initial `git status --short --branch` | `## codex/python-zero-pr00-baseline` |

The identity command emitted two non-fatal Git warnings because the sandbox denied
access to `C:\Users\dzk55\.config\git\ignore`.  Its attempt to write the
specified temporary ledger at `C:\tmp\alphadiamond-tracked-python.txt` was also
denied by the sandbox; the sorted in-memory `git ls-files '*.py'` count was `183`.
No repository file was changed during identity capture.

## Parity inputs and manifest preflight

| Input | Value | Status |
| --- | --- | --- |
| Fixed config | `configs/alphazero/soo-bootstrap.json` | SHA-256 `c33153cc47352d74e00bc38272313269d4853b51eaeeb7f474fe0e84807d5a02` |
| Proposed checkpoint | `C:\Users\dzk55\alphadiamond\runtime\runs\soo\soo-scratch-20260822\latest.pt` | Exists, but its SHA-256 is not recorded in the approved planning documents; it was intentionally not read, copied, modified, or used. |
| Short-run benchmark manifest | No tracked approved manifest found | **Manifest blocker:** no safe, supported exact commands for short-run training, checkpoint/resume, self-play, or end-to-end measurement can be derived. No commands were invented. |

The fixed configuration's digest was calculated without modifying it.  The
checkpoint is excluded from this baseline until an approved manifest records its
digest and exact disposable-run commands.

## Correctness baseline

Only the first required command was run, once. Per the task stop rule, no later
correctness command was run or diagnosed.

| Command | Status | Duration | Raw result |
| --- | --- | --- | --- |
| `cmake --preset native-ci` | **FAIL (launch)** | 2.4 s | See exact PowerShell output below. |
| `cmake --build --preset native-ci --parallel` | NOT RUN | — | Stop rule after preceding failure. |
| `ctest --preset native-ci --output-on-failure` | NOT RUN | — | Stop rule after preceding failure. |
| `python -m pytest --ignore=tests/native --durations=10` | NOT RUN | — | Stop rule after preceding failure. |
| `python -m pytest tests/native -v --durations=10` | NOT RUN | — | Stop rule after preceding failure. |

No pass/fail/skip totals exist because CMake could not launch.

```text
cmake:
Line |
   2 |  cmake --preset native-ci
     |  ~~~~~
     | The term 'cmake' is not recognized as a name of a cmdlet, function,
     | script file, or executable program. Check the spelling of the name, or
     | if a path was included, verify that the path is correct and try again.
```

## Benchmark baseline

No benchmark subject was run. The correctness stop rule prevents warm-up and
measurement repetitions. In addition, this Windows PowerShell environment has no
POSIX `/usr/bin/time -v`; no substitute RSS result is fabricated.

| Subject | Status |
| --- | --- |
| Incremental native build (`/usr/bin/time -v ...`) | NOT RUN — correctness blocked; POSIX timing tool unavailable. |
| Native CTest (`/usr/bin/time -v ...`) | NOT RUN — correctness blocked; POSIX timing tool unavailable. |
| Replay pipeline profile | NOT RUN — correctness blocked. |
| Native scheduler profile | NOT RUN — correctness blocked. |
| Training / checkpoint-resume / self-play / end-to-end | NOT RUN — correctness blocked and approved short-run manifest absent. |

## Environment, artifact, and protection recording

The prescribed parity-environment commands (`python`/Torch, CMake version,
C++ version, and clean-diff check) were not run after the first correctness
failure. Their values, CMake generator, CPU/GPU, artifact sizes, and peak RSS are
therefore **not captured**, rather than inferred.

The approved branch-protection PATCH and its readback were not run: the task
requires stopping after a failed baseline command. No GitHub settings were
changed. The required eight-context payload remains ready for a rerun on a host
where the correctness baseline can start.

## Required unblock conditions

1. Run PR 00 on a host with `cmake` available on `PATH`, then restart the
   prescribed baseline from its first correctness command.
2. Add an approved tracked short-run benchmark manifest that pins the checkpoint
   SHA-256 and safe disposable commands for training, checkpoint/resume,
   self-play, and end-to-end measures.
