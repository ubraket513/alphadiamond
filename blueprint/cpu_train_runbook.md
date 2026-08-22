# CPU 8-hour training runbook (Soo and Min)

Operator guide for the CPU-only bootstrap-to-normal AlphaZero session described
in `blueprint/cpu_train.md`.

## Environment

Training requires the `alphadiamond` conda environment (PySide6 is not needed):

```
PY=/home/dzk55/miniforge3/envs/alphadiamond/bin/python
```

Measured on this machine: Python 3.14.6, PyTorch 2.12.0 (CPU), 8 logical / 4
physical cores (i7-1165G7), 7 GB RAM, CUDA unavailable.

Every command below is run from the repository root with `PYTHONPATH=src`.

## Why this session does not use `alphadiamond-train`

The full `cli train` / `cli resume` pipeline runs a fixed nine-stage iteration
that always includes `PROMOTION_ARENA` and `RATING_BENCHMARK`. Both correctly
evaluate with `bootstrap_prior = none`, and an untrained network without the
heuristic never reaches a terminal state, so every evaluation game burns the
full 2000-move cap.

Measured on this CPU:

| stage | cost per iteration |
| --- | --- |
| self-play (16 games, 4 workers) | ~1-3 min |
| Soo promotion arena (40 games) | ~0.9 h |
| Soo rating benchmark (16 matches) | ~0.4 h |

That path yields roughly two iterations per eight hours. Blueprint section 8
explicitly permits using "the narrowest existing training path that preserves
self-play, replay, trainer, checkpoint and resume safety", so `tools/cpu_b0_train.py`
composes exactly those existing durable components and performs no promotion and
no rating. Rating semantics are unchanged; they are simply not exercised here.

## Selected search budget: 32 simulations

Fixed-seed calibration with the real Torch evaluator, v2 prior, `max_moves=2000`:

| model | sims | completion | median moves | sec/game | samples/s |
| --- | --- | --- | --- | --- | --- |
| Soo | 8 | 100% | 134 | 5.9 | 21.8 |
| Soo | 16 | 100% | 191 | 16.3 | 12.5 |
| Soo | 32 | 100% | 97 | 14.3 | 6.8 |
| Min | 8 | 100% | 131 | 5.8 | 22.7 |
| Min | 16 | 100% | 123 | 15.3 | 12.2 |
| Min | 32 | 100% | 109 | 20.7 | 5.2 |

Against an untrained network 8 simulations looks cheapest, but that ranking does
not survive training. After only four optimizer steps the Soo checkpoint behaved
very differently at each budget:

| sims | completion | median moves | sec/game |
| --- | --- | --- | --- |
| 8 | 17% | 248 | 78.3 |
| 16 | 100% | 369 | 24.6 |
| 32 | 100% | 87 | 11.7 |

A partially trained value head is uniformly optimistic (mean value +0.15 versus
+0.04 untrained), and with a shallow search that bias dominates the heuristic
prior, so games wander. Thirty-two simulations is therefore both faster and
stable once training starts, and the same effect is confirmed for Min
(untrained median 429 moves at 8 sims versus 123 at 32).

Only `mcts.simulations`, `workers` and `training.device` differ from the
checked-in bootstrap reference configs; `tests/tools/test_cpu_b0_train.py`
enforces that the network, self-play, replay and arena blocks are untouched.

## Runtime layout

```
runtime/configs/soo-cpu8h.json      run configuration (not the canonical config)
runtime/configs/min-cpu8h.json
runtime/runs/<model>/<run-id>/
    config.json      exact pinned config for the run
    latest.pt        durable checkpoint, resumed automatically
    checkpoints/     per-iteration archive checkpoints
    replay/          persistent replay store
    ledger.jsonl     append-only evidence
    loop_state.json  resumable counters
```

`runtime/` is git-ignored: checkpoints and replay never enter version control.

## Commands

All commands assume:

```
cd /home/dzk55/alphadiamond
export PY=/home/dzk55/miniforge3/envs/alphadiamond/bin/python
export PYTHONPATH=src
```

### 0. Preflight (once)

```
$PY -m pytest tests/alphazero tests/agents tests/tools -o addopts=""
$PY tools/cpu_make_configs.py
```

### 1. Soo bootstrap (B0), about 3h15m

```
$PY tools/cpu_b0_train.py \
  --config runtime/configs/soo-cpu8h.json \
  --runtime-dir runtime/runs --run-id cpu8h-soo-20260819 \
  --hours 3.25 --phase B0
```

Interrupt with Ctrl-C at any time: the current iteration finishes, state is
persisted, and re-running the identical command resumes from `latest.pt` with the
replay intact.

### 2. Soo heuristic-OFF gate

```
$PY tools/cpu_off_probe.py \
  --config runtime/configs/soo-cpu8h.json \
  --checkpoint runtime/runs/soo/cpu8h-soo-20260819/latest.pt \
  --episodes 10 --simulations 32,64 \
  --out runtime/runs/soo/cpu8h-soo-20260819/off-probe.json
```

PASS requires at least 8 of 10 completions with non-empty replay. If 32 fails and
64 passes, run phase A0 at 64 simulations; if both fail, return to step 1 for
another bootstrap block.

### 3. Soo normal AlphaZero (A0), only after a PASS

```
$PY tools/cpu_b0_train.py \
  --config runtime/configs/soo-cpu8h.json \
  --runtime-dir runtime/runs --run-id cpu8h-soo-20260819 \
  --hours 1.0 --bootstrap-prior none --phase A0
```

The same run id continues from the same trained checkpoint and keeps the existing
bootstrap replay, exactly as blueprint section 12 requires.

### 4-6. Min

Identical, with `min-cpu8h.json`, run id `cpu8h-min-20260819` and about 4 hours.

### 7. Reports

```
$PY tools/cpu_report.py --run-dir runtime/runs/soo/cpu8h-soo-20260819 \
  --probe runtime/runs/soo/cpu8h-soo-20260819/off-probe.json
$PY tools/cpu_report.py --run-dir runtime/runs/min/cpu8h-min-20260819 \
  --probe runtime/runs/min/cpu8h-min-20260819/off-probe.json
```

## Monitoring a live run

```
tail -f runtime/runs/soo/cpu8h-soo-20260819/ledger.jsonl | \
  $PY -c "import sys,json; [print(json.loads(l).get('event'), json.loads(l).get('iteration'), json.loads(l).get('metrics')) for l in sys.stdin]"
```

Progress lines are printed per iteration:

```
[i0003] 16/16 done median_moves=97 samples+1580 replay=6210 step=16 loss=6.7 sp=58s tr=9s elapsed=0.41h
```

Watch for: completion below 16/16, `median_moves` climbing toward 2000, a
non-finite loss (the runner exits on one), or replay that stops growing.
