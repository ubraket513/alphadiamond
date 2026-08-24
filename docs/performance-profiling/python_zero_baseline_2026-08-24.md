# Python-Zero Migration — PR 00 Baseline

**Status:** INCOMPLETE — the pre-repair mamba pytest run failed during collection
because `_diamond_native` was unavailable. That extension was subsequently built
and imported successfully, but the two post-repair pytest process completions
lost their exit statuses and final summaries; neither is claimed PASS.

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
passed; the fourth command then failed to launch under the original interpreter. The
operator then provided a worktree virtual environment and explicitly authorized
one retry using its exact interpreter. It also failed to launch with the same
missing-module condition. The controller classified both as environment-repair
launch failures, then authorized one required-suite run using only the verified
mamba interpreter. That suite started and failed during collection when the native
extension import failed, so no later correctness command was run or diagnosed.

| Command | Status | Duration | Raw result |
| --- | --- | --- | --- |
| `cmake --preset native-ci` (initial) | **FAIL (launch)** | 2.4 s | See exact PowerShell output below. |
| `cmake --preset native-ci` (authorized repaired re-attempt) | PASS | 15.4 s | Visual Studio 18 2026 generator; MSVC 19.51.36248.0; generated `build/native-ci`. |
| `cmake --build --preset native-ci --parallel` | PASS | 31.2 s | Native targets and test executables built successfully. |
| `ctest --preset native-ci --output-on-failure` | PASS | 11.2 s | 9/9 passed, 0 failed; CTest real time 3.85 s. |
| `python -m pytest --ignore=tests/native --durations=10` | **FAIL (launch)** | 6.9 s | See exact Python output below. |
| `C:\tmp\codex-alphadiamond-pr00\.venv\Scripts\python.exe -m pytest --ignore=tests/native --durations=10` (authorized repaired retry) | **FAIL (launch)** | 9.0 s | See exact worktree-venv output below. |
| `C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m pytest --ignore=tests/native --durations=10` (controller-authorized suite run) | **FAIL (collection)** | 14.0 s | 1 collection error in 4.82 s: `_diamond_native` import unavailable. |
| `python -m pytest tests/native -v --durations=10` | NOT RUN | — | Stop rule after preceding failure. |

The native CTest baseline has 9 passed, 0 failed. The first two pytest invocations
were environment-repair launch failures with no suite counts. The verified mamba
run collected one module error and ran no tests.

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

The operator-provided environment was invoked exactly as
`C:\tmp\codex-alphadiamond-pr00\.venv\Scripts\python.exe`; no substitute Python
was selected. Its exact retry output was:

```text
C:\tmp\codex-alphadiamond-pr00\.venv\Scripts\python.exe: No module named pytest
```

The controller verified the required mamba interpreter before authorizing its
single suite run: `C:\ProgramData\miniforge3\envs\alphadiamond\python.exe`
(Python 3.12.13; pytest 9.1.1; Torch 2.13.0+cpu; TrueSkill 0.4.5). Its exact
result was:

```text
=================================== ERRORS ====================================
______________ ERROR collecting tests/alphazero/test_trainer.py _______________
tests\alphazero\test_trainer.py:34: in <module>
    SooModel(NetworkConfig(width=16, residual_blocks=1), model_version="0.2.0"),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
src\diamond\alphazero\network\soo.py:31: in __init__
    self.trunk = DiamondGraphTrunk(
src\diamond\alphazero\network\trunk.py:123: in __init__
    self.register_buffer("adjacency", directional_adjacency())
                                      ^^^^^^^^^^^^^^^^^^^^^^^
src\diamond\alphazero\network\trunk.py:24: in directional_adjacency
    neighbours = neighbour_table()
                 ^^^^^^^^^^^^^^^^^
src\diamond\alphazero\native\topology.py:62: in neighbour_table
    return topology_tables()["neighbour"]
           ^^^^^^^^^^^^^^^^^
src\diamond\alphazero\native\topology.py:41: in topology_tables
    tables = require_native().export_tables()
             ^^^^^^^^^^^^^^^^
src\diamond\alphazero\native\__init__.py:73: in require_native
    raise RuntimeError(native_error() or "native extension unavailable")
E   RuntimeError: native extension unavailable: cannot import name '_diamond_native' from 'diamond.alphazero.native' (C:\tmp\codex-alphadiamond-pr00\src\diamond\alphazero\native\__init__.py)
=========================== short test summary info ===========================
ERROR tests/alphazero/test_trainer.py - RuntimeError: native extension unavai...
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 4.82s
```

## Native-extension repair and diagnostic

The collection failure was traced to the mamba environment's editable
`diamond-console` target being the user worktree rather than this PR worktree,
and no `_diamond_native` extension being importable from the PR source. The user
authorized the following repair using only the verified mamba interpreter and a
process-local Visual Studio/MSVC environment:

| Action | Status | Duration | Result |
| --- | --- | --- | --- |
| `python -m pip install -e '.[native]'` in this PR worktree | PASS | 15.7 s | Built and installed editable `diamond-console` from `C:\tmp\codex-alphadiamond-pr00`; no tracked source, test, or CI file changed. |
| Minimal import: `from diamond.alphazero.native import _diamond_native` with `PYTHONPATH=C:\tmp\codex-alphadiamond-pr00\src` | **FAIL** | 7.9 s | `ImportError: cannot import name '_diamond_native' from 'diamond.alphazero.native'`. |
| Restore original editable source: `python -m pip install -e C:\Users\dzk55\alphadiamond` | PASS | 12.6 s | `pip` exited successfully and reinstalled editable `diamond-console` from the original user worktree. |

The import diagnostic was run once and was not retried. Because it failed, no
post-repair pytest rerun was started at that point. No test or import diagnostic
was run after restoration; the recorded successful `pip` exit is the only restore
verification. A later manual build diagnostic and its resulting test evidence are
recorded below.

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name '_diamond_native' from 'diamond.alphazero.native' (C:\tmp\codex-alphadiamond-pr00\src\diamond\alphazero\native\__init__.py)
```

### Root-cause build diagnostic and post-repair test evidence

The editable `.[native]` installation did **not** make an extension importable.
One manually authorized, process-local build command,
`python setup.py build_ext --inplace --verbose`, then completed successfully.
It emitted MSVC warnings about ignored GCC-style options and pybind11 code-page
warnings, but no build error. One subsequent minimal import diagnostic passed and
loaded:

```text
C:\tmp\codex-alphadiamond-pr00\src\diamond\alphazero\native\_diamond_native.cp312-win_amd64.pyd
```

With that in-place extension and the required mamba interpreter, worktree
`PYTHONPATH`, and process-local Visual Studio/MSVC environment, each prescribed
pytest process was launched exactly once post-repair:

| Process | Observed progress | Final status |
| --- | --- | --- |
| `pytest --ignore=tests/native --durations=10` | Progress reached 69%, with dots and skips shown. | **Indeterminate:** the tool lost the exit status and final summary. |
| `pytest tests/native -v --durations=10` | Collected 41 items; output reached `tests\\native\\test_selfplay_pool.py ..`. | **Indeterminate:** the tool lost the exit status and final summary. |

The native pytest process later exited, and its supplied
`.pytest_cache/v/cache/lastfailed` value was `{}`. This is recorded as limited
evidence only; it does not establish a suite pass. No benchmark, parity-
environment, or branch-protection command was authorized after these
indeterminate results.

## Benchmark baseline

No benchmark subject was run. The post-repair pytest results are indeterminate,
so no later baseline stage was authorized. In addition, this Windows PowerShell
environment has no POSIX `/usr/bin/time -v`; no substitute RSS result is
fabricated.

| Subject | Status |
| --- | --- |
| Incremental native build (`/usr/bin/time -v ...`) | NOT RUN — correctness blocked; POSIX timing tool unavailable. |
| Native CTest (`/usr/bin/time -v ...`) | NOT RUN — correctness blocked; POSIX timing tool unavailable. |
| Replay pipeline profile | NOT RUN — correctness blocked. |
| Native scheduler profile | NOT RUN — correctness blocked. |
| Training / checkpoint-resume / self-play / end-to-end | NOT RUN — correctness blocked and approved short-run manifest absent. |

## Environment, artifact, and protection recording

The prescribed parity-environment commands (`python`/Torch, CMake version,
C++ version, and clean-diff check) were not run after the indeterminate
post-repair pytest results. CPU/GPU, artifact sizes, and peak RSS are therefore
**not captured**, rather than inferred. The controller verified Python 3.12.13,
pytest 9.1.1, Torch 2.13.0+cpu, and TrueSkill 0.4.5 for the mamba interpreter;
the repaired configure established the Visual Studio 18 2026 generator and MSVC
19.51.36248.0.

The approved branch-protection PATCH and its readback were not run: the task
requires stopping after a failed baseline command. No GitHub settings were
changed. The required eight-context payload remains ready for a rerun on a host
where the pytest baseline can start.

## Required unblock conditions

1. Recover authoritative exit statuses/final summaries for the already-launched
   post-repair pytest processes before authorizing benchmark, environment, or
   branch-protection stages.
2. Add an approved tracked short-run benchmark manifest that pins the checkpoint
   SHA-256 and safe disposable commands for training, checkpoint/resume,
   self-play, and end-to-end measures.
