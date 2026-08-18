# Diamond AlphaZero Milestone 1

`diamond.alphazero` is the Python/PyTorch correctness reference for the
authoritative Diamond engine. It does not replace or duplicate the rules in
`diamond.game`; every search transition resolves and commits through that
engine.

## Model identities

- **Soo** is the independently versioned 2-player model. Its value is a scalar
  current-player win/loss utility in `[-1, +1]`.
- **Min** is the independently versioned 3-player model. Its value is the
  placement utility vector `[self, next, previous]`, where first is `+1`,
  second is `0`, and third is `-1`.

Soo and Min versions are unrelated. A Soo `0.4.0` release does not imply that
Min has or needs a `0.4.0` release.

Checkpoint loading exact-matches `model_name`, `model_version`, player count,
ruleset, board topology, a deterministic ruleset fingerprint, encoder, action
space, seat layout, value semantics, and network configuration. A matching
model version never overrides another mismatch.

## Package boundary

- `action_codec.py`, `encoder.py`, and `game_adapter.py`: versioned action and
  canonical-state bridge to the authoritative game.
- `network/`: neutral directional graph trunk and named `SooModel`/`MinModel`.
- `evaluator/`: batched framework-neutral contract, deterministic dummy, and
  eager PyTorch evaluator.
- `mcts/`: framework-neutral scalar Soo and vector Min PUCT.
- `selfplay/`, `replay.py`, and `trainer.py`: single-process data generation,
  sparse CPU replay, compatibility-bearing batches, exact Soo/Min target
  validation, and FP32 AdamW training.
- `checkpoint.py`: strict atomic checkpoint persistence.
- `arena.py`: deterministic candidate/baseline evaluation with balanced
  seating and turn orders. A complete balance cycle is 4 games for Soo and 18
  games for Min; partial cycles are rejected, and the shared default of 36
  games completes both cycles.

The GUI remains separate and continues to use the existing `Agent` protocol.
Milestone 1 does not route Soo or Min into the GUI.

## Install

```powershell
mamba env update -n alphadiamond -f environment.yml
mamba activate alphadiamond
pip install -e .
```

PyTorch is also available through the `alphazero` Python extra:

```powershell
pip install -e ".[alphazero,dev]"
```

## Verification

Run only the AlphaZero tests:

```powershell
python -m pytest tests/alphazero
```

Run the complete game, GUI-controller, and AlphaZero regression suite:

```powershell
python -m pytest
```

Run all CPU smoke checks:

```powershell
python -m diamond.alphazero.smoke
```

That command performs:

- a one-move authoritative Soo self-play finish;
- a two-move authoritative Min finish through full three-place ranking;
- one real FP32 AdamW update for Soo and Min;
- exact checkpoint model/optimizer/config/training-step round trips;
- deterministic balanced Soo and Min arena orchestration.

Its exit status covers every check above; success is not based on self-play
alone.

Checkpoint resume also rejects a serialized training device that differs from
the destination trainer and optimizer hyperparameters that disagree with the
serialized `TrainingConfig`.

`TrainingConfig.seed` controls PyTorch randomness after the model is supplied
to the trainer. For reproducible initial weights, set the PyTorch seed before
constructing `SooModel` or `MinModel`; the trainer deliberately does not reset
or reinitialize a caller-supplied model.

Individual smoke functions can also be run directly:

```powershell
python -c "from diamond.alphazero.smoke import run_selfplay_smoke; print(run_selfplay_smoke())"
python -c "from diamond.alphazero.smoke import run_training_smoke; print(run_training_smoke())"
python -c "from diamond.alphazero.smoke import run_checkpoint_smoke; print(run_checkpoint_smoke())"
python -c "from diamond.alphazero.smoke import run_arena_smoke; print(run_arena_smoke())"
```

## Deferred work

C++ search/game hot paths, centralized batched inference, distributed
self-play, BF16/`torch.compile` profiling, OpenVINO deployment, and GUI
`SooAgent`/`MinAgent` routing are later milestones. None are implemented here.
