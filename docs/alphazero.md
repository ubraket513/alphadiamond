# Diamond AlphaZero Milestones 1–2

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

## Milestone 2 operator runbook

Milestone 2 adds headless, resumable training orchestration, historical
rating, and bounded inference profiling. It remains separate from the GUI.
Use the AlphaZero mamba environment for Torch/AlphaZero commands:

```powershell
C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m pytest tests\alphazero -o addopts= -q
C:\ProgramData\miniforge3\envs\alphadiamond\python.exe -m diamond.alphazero.milestone2_smoke
```

On this Windows host, Qt tests use the system Python with PySide6 instead:

```powershell
C:\Python314\python.exe -m pytest tests --ignore=tests/alphazero -o addopts= -q
```

The headless CLI emits one JSON object and an exit status. Its commands are
`train`, `resume`, `benchmark`, `leaderboard`, and `profile`; `Soo` and `Min`
are always separate model namespaces.

Every command runs the real production services, so each one requires an
explicit `--config` (a strict production config JSON) and `--checkpoint` (a
bootstrap or champion checkpoint whose compatibility spec and
`training_config` match that config). Reference configs live in
`configs/alphazero/`.

```powershell
$py = 'C:\ProgramData\miniforge3\envs\alphadiamond\python.exe'
$cfg = 'configs\alphazero\soo-production.json'
$ckpt = 'runtime\soo\soo-001\bootstrap.pt'
& $py -m diamond.alphazero.orchestration.cli train --runtime-dir runtime --model Soo --run-id soo-001 --config $cfg --checkpoint $ckpt
& $py -m diamond.alphazero.orchestration.cli resume --runtime-dir runtime --model Soo --run-id soo-001 --config $cfg --checkpoint $ckpt
& $py -m diamond.alphazero.orchestration.cli benchmark --runtime-dir runtime --model Soo --run-id soo-001 --config $cfg --checkpoint $ckpt
& $py -m diamond.alphazero.orchestration.cli leaderboard --runtime-dir runtime --model Soo --run-id soo-001 --config $cfg --checkpoint $ckpt
& $py -m diamond.alphazero.orchestration.cli profile --runtime-dir runtime --model Soo --seconds 1 --config $cfg --checkpoint $ckpt
```

`--run-id` is validated before any runtime path is resolved, so traversal or
absolute identifiers fail with an argument error and write nothing.

The reference configs target a CUDA training host. Self-play only contributes
replay samples from games that actually finish, so `self_play.max_moves` must
be large enough for real Diamond games to reach a podium; a truncated game is
recorded as `max_game_moves_exceeded` and yields zero samples. Likewise
`inference.response_timeout_s` bounds the self-play worker timeout
(`max(60, response_timeout_s * 4)`), so short timeouts abort long searches.

The current default services place a run under the supplied runtime root by
model and run ID (for example, `runtime/soo/soo-001/`), with state, replay,
checkpoint, rating, and stage-artifact data kept there rather than in Git:

```text
runtime/
  soo|min/<run-id>/
    state.json
    bootstrap.pt
    replay/
    ratings/registry.json
    artifacts/<stage>/<operation-sha256>.json|.pt
```

Replay records are keyed by stable episode/game identity, and stage artifacts
and run state use atomic writes. Resume reloads completed identities instead
of repeating a completed side effect; conflicting artifact identities and
checkpoint/protocol compatibility changes are rejected.

### Rating and benchmark protocol

- Soo historical strength uses Elo. Soo promotion is a separate
  candidate-versus-champion decision with its own balanced arena.
- Min historical strength uses the official TrueSkill environment. `tau=0`
  because a checkpoint is immutable, so its rating does not drift between
  matches. The leaderboard sorts by the environment's conservative exposure,
  and displays `mu`, `sigma`, exposure, and rated games.
- A Min TrueSkill event requires **three distinct checkpoint artifacts**.
  Candidate/champion/champion can be useful for promotion, but is not a
  TrueSkill event. A fresh two-artifact Min run therefore reports
  `insufficient_history` rather than fabricating a rating.
- Min promotion uses **18 games**. Historical Min league rating uses **36
  games**, covering the six turn orders and rotations; it is not the promotion
  arena.

Every rated event is bound to a deterministic benchmark protocol identity:
model and compatibility namespace, fixed MCTS compute, disabled root noise,
deterministic decision policy, ruleset/encoder/action/seat/value identities,
rating parameters, and a versioned, hashed opening suite. The suite contains
the standard initial state and authoritative legal action sequences. Ratings
from different protocol IDs never mix; strength ratings and throughput
profiles are different measurements.

Checkpoint artifacts are content-addressed by SHA-256 and carry independent
Soo/Min model versions plus strict compatibility gates. A version number is
neither a checkpoint artifact identity nor a strength rating.

### Profile interpretation and current limitation

Run the bounded CPU profile through the CLI above. It reports only work it
actually executed: evaluator calls/states per second, latency and batch
quantiles, and supplied self-play, replay-collation, and training timings.
`gpu_verified=false`, no `gpus` field, or an unavailable CUDA mode means no
GPU evidence was collected—not a zero-throughput GPU result. Do not add rows
for unavailable CUDA/BF16/compiled modes.

The verified host has no CUDA device or NVIDIA A30, and no production A30
stage-percentage profile exists. Therefore no C++ implementation is
recommended yet. Reconsider only after a production A30 profile shows a
reproducible CPU-side search/game/tree stage as the dominant end-to-end share
after centralized eager inference is in place; then define a narrow boundary
and differential-test it against the Python oracle before seeking separate
approval.

## Deferred work

C++ search/game hot paths, OpenVINO deployment, GUI `SooAgent`/`MinAgent`
routing, and distributed/multi-node training remain deferred. CUDA BF16 and
`torch.compile` modes are benchmark-only and require an available, measured
CUDA environment; no unavailable-mode result is treated as a performance
claim.
