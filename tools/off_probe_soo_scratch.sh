#!/usr/bin/env bash
# The heuristics-OFF gate for the from-scratch Soo run.
#
# Blueprint sections 10-11: can this network generate real *terminal* games on
# its own, with bootstrap_prior = none?  The pass criterion is the blueprint's
# -- at least 8 of 10 games complete -- and it is operational, not a claim about
# playing strength.  Until it passes, turning the heuristic off produces games
# that never end and therefore no training data at all.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ROOT="${TRAIN_ROOT:-/workspace/alphadiamond-training}"
RUN_ID="${RUN_ID:-soo-scratch-20260822}"
CONFIG="${CONFIG:-$ROOT/runtime/configs/soo-rtx5090-native.json}"

export PYTHONPATH="$TRAIN_ROOT/pinned-src"
RUN="$TRAIN_ROOT/runs/soo/$RUN_ID"
STAMP="$(date +%Y%m%dT%H%M%S)"
mkdir -p "$TRAIN_ROOT/probes"

python "$ROOT/tools/cpu_off_probe.py" \
  --config "$CONFIG" \
  --checkpoint "$RUN/latest.pt" \
  --episodes "${EPISODES:-10}" \
  --max-moves "${MAX_MOVES:-500}" \
  --threads "${THREADS:-8}" \
  --out "$TRAIN_ROOT/probes/off-probe-$STAMP.json"
