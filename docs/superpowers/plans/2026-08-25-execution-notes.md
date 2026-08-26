# Execution Notes

- Use `tools/native_training.sh` for every native CMake/CTest invocation; it initializes the Windows MSVC environment explicitly.
- Before diagnosing a native test failure, rebuild its exact target through that wrapper so stale executables cannot mislead the diagnosis.
- On Windows, close all files in a staged directory before promoting that directory. Capture `GetLastError()` immediately after a failed Win32 call, before cleanup.
- Root directly handles environment, build/link, DLL/PATH, integration, and small acceptance fixes. It also takes over any path after two failed diagnosis/fix exchanges.
- Sol owns inventory/design/dependency/risk/PR planning; Terra owns an approved independent implementation slice; Luna owns only independent mechanical file/scaffolding work; Sol max owns the final native-trainer/Python-zero gate.
- Spawn roles with an explicit model and reasoning effort: `gpt-5.6-sol`/`xhigh` for inventory and design, `gpt-5.6-terra`/`high` for approved implementation and focused acceptance, `gpt-5.6-luna`/`medium` for mechanical inventory/scaffolding/documentation, and `gpt-5.6-sol`/`max` only for final native-trainer, CUDA, release, or Python-zero gates. Do not use `ultra` by default and do not rely on inherited full-history model settings.
- Give subagents a bounded recent-turn fork so explicit model/effort overrides and current context coexist. Every subagent must read and apply `caveman` and `ponytail` at `full` intensity, read Serena `initial_instructions`, activate this project, and report whether Serena was actually used; do not equate tool availability with use.
- Do not repeat independent review agents or bounce plan, implementation, and review between agents. Parallelize only independent file sets, with at most three subagents because root occupies the fourth slot.
- A subagent brief must name repository/branch/base SHA, owned and prohibited files, POSIX/Git-Bash script policy, the `alphadiamond` environment path, build directory/target, one acceptance command, no-worktree/current-branch rule, and completed/stale work that must not be revisited.
- Work on the current PR branch rather than creating an independent worktree. Preserve user-owned untracked files and stage explicit paths instead of `git add .`.
- For Windows discovery, inspect existing CMake caches, presets, and logs before deciding a tool is absent. Load both `VsDevCmd.bat` and the conda environment DLL paths.
- Repository scripts remain POSIX `.sh`. Use Git Bash when the Codex launcher permits it; PowerShell may prepare and pass the native environment but must not become a new production `.ps1` workflow.
- For normal tasks, build the changed target and run the nearest contract/smoke test once. Reserve full suites, release audits, GUI/device runs, and final Python-zero checks for subsystem or final boundaries.
- After tested implementation, commit, then push/create/watch a PR as separate visible operations. Merge and branch deletion require a separate decision.
