# Python-Zero Migration — PR 00 Baseline

**Status:** BLOCKED — the fourth prescribed correctness command could not start
because the selected Python interpreter has no `pytest` module.

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

The initial configure launch failed because `cmake` was absent from `PATH`. The
operator then confirmed the Visual Studio CMake location and explicitly authorized
one re-attempt with a **process-local** environment repair: CMake's Visual Studio
bin directory was prepended to `PATH` and `vcvars64.bat` was imported. No system
or user environment value was persisted. The re-attempt and the next two commands
passed; the fourth command then failed, so no later correctness command was run
or diagnosed.

| Command | Status | Duration | Raw result |
| --- | --- | --- | --- |
| `cmake --preset native-ci` (initial) | **FAIL (launch)** | 2.4 s | See exact PowerShell output below. |
| `cmake --preset native-ci` (authorized repaired re-attempt) | PASS | 15.4 s | Visual Studio 18 2026 generator; MSVC 19.51.36248.0; generated `build/native-ci`. |
| `cmake --build --preset native-ci --parallel` | PASS | 31.2 s | Native targets and test executables built successfully. |
| `ctest --preset native-ci --output-on-failure` | PASS | 11.2 s | 9/9 passed, 0 failed; CTest real time 3.85 s. |
| `python -m pytest --ignore=tests/native --durations=10` | **FAIL (launch)** | 6.9 s | See exact Python output below. |
| `python -m pytest tests/native -v --durations=10` | NOT RUN | — | Stop rule after preceding failure. |

The native CTest baseline has 9 passed, 0 failed. Pytest produced no collection,
pass, fail, or skip count because its module could not launch.

```text
cmake:
Line |
   2 |  cmake --preset native-ci
     |  ~~~~~
     | The term 'cmake' is not recognized as a name of a cmdlet, function,
     | script file, or executable program. Check the spelling of the name, or
     | if a path was included, verify that the path is correct and try again.
```

Successful repaired configure output:

```text
-- Building for: Visual Studio 18 2026
-- Selecting Windows SDK version 10.0.26100.0 to target Windows 10.0.26200.
-- The CXX compiler identification is MSVC 19.51.36248.0
-- Detecting CXX compiler ABI info - done
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread - not found
-- Found Threads: TRUE
-- Configuring done (6.2s)
-- Generating done (0.2s)
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_BUILD_TYPE

-- Build files have been written to: C:/tmp/codex-alphadiamond-pr00/build/native-ci
```

CTest output:

```text
1/9 Test #1: action_codec_test ................   Passed    0.04 sec
2/9 Test #2: topology_test ....................   Passed    0.07 sec
3/9 Test #3: rules_golden_test ................   Passed    0.11 sec
4/9 Test #4: mcts_golden_test .................   Passed    0.11 sec
5/9 Test #5: mcts_stochastic_test .............   Passed    0.30 sec
6/9 Test #6: mcts3p_golden_test ...............   Passed    0.07 sec
7/9 Test #7: budget_test ......................   Passed    0.11 sec
8/9 Test #8: batcher_test .....................   Passed    0.12 sec
9/9 Test #9: selfplay_test ....................   Passed    2.66 sec

100% tests passed, 0 tests failed out of 9

Total Test time (real) =   3.85 sec
```

```text
C:\ProgramData\miniforge3\python.exe: No module named pytest
```

## Benchmark baseline

No benchmark subject was run. The correctness stop rule after the pytest launch
failure prevents warm-up and measurement repetitions. In addition, this Windows
PowerShell environment has no POSIX `/usr/bin/time -v`; no substitute RSS result
is fabricated.

| Subject | Status |
| --- | --- |
| Incremental native build (`/usr/bin/time -v ...`) | NOT RUN — correctness blocked; POSIX timing tool unavailable. |
| Native CTest (`/usr/bin/time -v ...`) | NOT RUN — correctness blocked; POSIX timing tool unavailable. |
| Replay pipeline profile | NOT RUN — correctness blocked. |
| Native scheduler profile | NOT RUN — correctness blocked. |
| Training / checkpoint-resume / self-play / end-to-end | NOT RUN — correctness blocked and approved short-run manifest absent. |

## Environment, artifact, and protection recording

The prescribed parity-environment commands (`python`/Torch, CMake version,
C++ version, and clean-diff check) were not run after the pytest launch failure.
Their values, CPU/GPU, artifact sizes, and peak RSS are therefore **not captured**,
rather than inferred. The repaired configure did establish the Visual Studio 18
2026 generator and MSVC 19.51.36248.0 only.

The approved branch-protection PATCH and its readback were not run: the task
requires stopping after a failed baseline command. No GitHub settings were
changed. The required eight-context payload remains ready for a rerun on a host
where the pytest baseline can start.

## Required unblock conditions

1. Provide `pytest` to the selected `C:\ProgramData\miniforge3\python.exe`
   environment, then restart the prescribed baseline from its first pytest
   command; do not rerun the successful native configure/build/CTest commands.
2. Add an approved tracked short-run benchmark manifest that pins the checkpoint
   SHA-256 and safe disposable commands for training, checkpoint/resume,
   self-play, and end-to-end measures.
