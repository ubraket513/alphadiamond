# Execution Notes

- Use `tools/native_training.ps1` for every native CMake/CTest invocation; do not rely on inherited `PATH`.
- Before diagnosing a native test failure, rebuild its exact target through that wrapper so stale executables cannot mislead the diagnosis.
- On Windows, close all files in a staged directory before promoting that directory. Capture `GetLastError()` immediately after a failed Win32 call, before cleanup.
- If an implementation subagent requires more than two diagnosis/fix exchanges, the primary agent takes over the exact failing path directly.
- Keep one Sol checklist per PR, then let Terra own implementation and acceptance; send mechanical fixture/scaffolding work to Luna only when it can run independently.
