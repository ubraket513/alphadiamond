# Min local CPU audit — 2026-09-11

## Finding

The saved human game exposes a large sensitivity to opponent turn order in both
step 20064 and step 47520 (Min 2.0.1). This is the strongest issue found in this
audit. It is not evidence that a larger network or a different reward scale is
the first necessary intervention.

The game uses P1 → P3 → P2, with P2 controlled by the AI. Training self-play uses
P1 → P2 → P3 with the same player camps. The encoder orders opponent occupancy
and finished-flag channels by turn order, while rotating the board by the acting
player's home camp. Reversing the opponents therefore changes their relationship
to the spatial input the model learned. The GUI accepts this order without a
model compatibility distinction.

Earlier weight and runtime checks established that the intended weights were
loaded and output values were mapped to the correct players. Those checks did
not establish that the model was trained for the GUI's alternate turn order.

## Controlled evidence

Replayed all 94 saved moves through the native rules, checking legality and final
occupancy and finish order. P2 finished third. Evaluated all 31 AI positions with
both checkpoints. A second pass keeps each board and acting player unchanged but
sets the match order to P1 → P2 → P3.

| Checkpoint | Last-position NN, actual order | NN, training order | Actual-order MCTS, 1024 | Actual-order MCTS, 4096 |
|---|---:|---:|---:|---:|
| 20064 | +0.936428 | −0.999997 | −0.740293 | −0.516386 |
| 47520 | +0.771095 | −0.999999 | −0.511506 | −0.409361 |

The sign change is not confined to the final position. At ply 63, the older
checkpoint changes from +0.945047 to −0.961479; the latest changes from +0.980227
to −0.986143. Both checkpoints show strongly positive actual-order estimates
through much of the late game and strongly negative training-order estimates.

This order substitution is a **counterfactual**, not an inference fix: changing
turn order also changes future game dynamics. The actual third-place result is
not a ground-truth value label for the counterfactual game. Likewise, one game
does not establish either checkpoint's overall playing strength or calibration.
The large systematic sensitivity plus fixed-order training supports a serious
distribution mismatch; it does not isolate every cause of poor play.

Search coverage: 24 actual-order positions per checkpoint at 128 simulations;
four positions (plies 3, 33, 63, 93) per checkpoint at 1024 and 4096; the same four
counterfactual positions at 128 and 1024. Together with the 31-position passes,
this is 204 searches. Extra simulations do not reliably close the NN/MCTS gap.
Neither a value shift nor a changed selected move demonstrates a win-rate gain.

## Training evidence recovered

Verified the downloaded final backup's SHA-256:
`029230818711f1bb29835e549fc731129389bbf73b85d577fd93f76081bc46e6`.
Indexed the archive and selectively extracted configuration and small reports;
did not deserialize arbitrary checkpoint code or resume training.

- Active configuration: six residual blocks, width 128, self-play 128 simulations,
  1024 games/iteration, batch size 1024, 352 learner steps/iteration, no bootstrap
  prior, repetition window 0, replay capacity 1,000,000, arena disabled.
- Final iteration 134: step 47520, 1022 completed games and two move-limit aborts,
  median completed length 100 plies. This does not support widespread self-play
  noncompletion as the primary explanation for this human game.
- The prior 40128-versus-20064 arena completed 432 games. Mean utility +0.04398,
  95% interval [−0.05093, +0.14126]: improvement was not established. This is not
  an evaluation of final step 47520. Arena used balanced orders and repetition
  sampling, so its conditions differ from fixed-order self-play and GUI play.
- Loss values across changing self-play distributions are not a fixed held-out
  quality measurement. The extracted summaries do not constitute a full replay
  target audit; individual replay records were not sampled in this phase.

## Implications for the proposed changes

1. **Fix coverage first.** Establish a fresh human-game baseline with the actual
   training order. For full GUI order support, train/evaluate all allowed orders
   and represent player goals/order explicitly if needed. Do not silently sort
   channels while searching the original game: that would misrepresent dynamics.
2. **Keep reward experiments separate.** The current value head uses tanh, so
   raw +10/−5/−10 targets cannot be represented. A bounded +1/−0.5/−1 experiment
   would express those relative placement preferences, but all training targets,
   terminal search utilities, and evaluation objectives must agree. This changes
   preferences, not merely learning strength.
3. **Long jumps exist.** The legal-action generator breadth-first enumerates
   chained jumps without a small hop limit. Network message-passing depth is a
   different limitation; the current evidence does not justify widening or
   deepening before correcting coverage.
4. **Do not raise simulations globally yet.** On this laptop, four-position
   4096-simulation searches took roughly 5–19 seconds each. The result is useful
   for diagnosis, but not evidence of consistently better decisions. Timings are
   observational wall times on an interactive machine, not a controlled benchmark.

## Reproduction and files

Added `native/qt/min_position_audit.cpp`, built as
`build/native-package/native/qt/diamond_qt_min_position_audit.exe` when Qt Soo and
testing are enabled. It calls the same `SooSearchRuntime` as the GUI. It refuses
to overwrite an existing output file. No production inference behavior changed.

```powershell
$env:PATH=(Resolve-Path dist/diamond-qt-soo).Path+';'+$env:PATH
$env:QT_PLUGIN_PATH=(Resolve-Path dist/diamond-qt-soo/plugins).Path
$env:DIAMOND_TORCH_THREADS='2'
& build/native-package/native/qt/diamond_qt_min_position_audit.exe `
  artifacts/min-local-audit/human-game.json models/min/2.0.1 `
  128,1024 4 artifacts/min-local-audit/new-run.jsonl actual
python tools/summarize_min_local_audit.py artifacts/min-local-audit
```

Evidence resides in `artifacts/min-local-audit/`: saved game, per-position JSONL
(values, selected/recorded actions, priors, visits, timing), `summary.json`,
`archive-index.json`, and selected backup reports under `recovered/`.
The old artifact is under
`artifacts/min-step20064-verified/models/min/2.0.0-min-test.20064`.

Relevant source: `native/src/encoder.cpp` (channel order),
`native/src/train_main.cpp` (`iteration_job` fixed match versus arena's
`ordered_match`), `native/src/board.cpp` (standard camps/order),
`native/qt/native_controller.cpp` (`startMatch` preserves chosen order),
`native/src/rules.cpp` (jump enumeration), `native/src/soo_model.cpp` (tanh),
and `native/src/pipeline.cpp` (placement targets).
