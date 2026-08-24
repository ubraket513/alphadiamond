# Python-zero final acceptance

Date: 2026-08-25  
Branch: `codex/python-zero-pr13-final-gates`

## Release build

- Configured the `native-package` preset with MSVC, Ninja, Qt 6, and the CPU LibTorch installation from the `alphadiamond` environment.
- Built `diamond_qt` and `alphadiamond-train` in Release mode.
- Fixed the Git Bash MSVC launcher so the Visual Studio environment preserves the active Qt, conda runtime, and LibTorch paths.

## CPU training

Executed one real native LibTorch forward/backward/optimizer step on CPU:

```json
{"name":"training_step","repetitions":1,"elapsed_seconds":0.0055034,"seconds_per_repetition":0.0055034,"training_step":2}
```

The process exited successfully.

## Windows package

- Created `dist/diamond-qt-final` from the Release executable.
- Bundled Qt plugins and QML modules, the Soo artifact, and the LibTorch runtime.
- Verified during deployment that both `c10.dll` and `torch_cpu.dll` are present.
- Ensured Torch-provided DLLs win over same-named conda fallback DLLs.
- Excluded `python*.dll`, `torch_python*.dll`, PySide, qtawesome, and `site-packages` paths from the package.

The default deployment path still runs all package-local smoke modes. The final Codex run used the explicit `--skip-runtime-smoke` option because Qt's `offscreen` QPA constructor blocks inside the restricted Codex Windows shell session; the actual desktop application was tested instead. This exception was reported rather than treated as a passing smoke run.

## GUI and model connection

- Launched the packaged Windows executable through the real Explorer desktop session.
- Computer control selected the exact final process path and confirmed the `Diamond — Controller Console` window plus its QML accessibility tree.
- The computer-control window capture returned a black Qt Quick surface even for a standard-frame validation build. A temporary, uncommitted `QQuickWindow::grabWindow()` acceptance hook therefore captured the actual rendered frame after `latestSearchCompute` became non-empty.
- Computer control independently observed the acceptance window title change to `Diamond — Controller Console [Acceptance Ready]` only after the Soo search result reached the GUI controller.
- Vision inspection confirmed the menu, board, both player camps and pieces, analysis panels, rotation control, fonts, colors, and layout rendered without an error dialog or missing resource.
- The rendered Search Compute panel showed an actual connected search: 128 simulations, 1091.8 ms total, 1089.8 ms neural evaluation, 2.1 ms MCTS/rules, and a 99.8%/0.2% compute split.

All validation-only capture, title, software-rendering, and standard-frame changes were removed before the final Release rebuild. The committed application remains frameless and contains no acceptance instrumentation.
