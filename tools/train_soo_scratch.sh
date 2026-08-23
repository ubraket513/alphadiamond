#!/usr/bin/env bash
# One resumable block of the from-scratch Soo run on the native backend.
#
# Re-running continues from latest.pt rather than restarting, so a block is a
# unit of supervision, not a unit of training.  SIGTERM finishes the current
# iteration and exits with state durable.
#
#   tools/train_soo_scratch.sh [hours] [bootstrap-prior] [phase] [az_train args...]
#     tools/train_soo_scratch.sh 6                    # B0, heuristics on
#     tools/train_soo_scratch.sh 6 none A0            # after the OFF gate passes
#     RUN_ID=soo-deep12 tools/train_soo_scratch.sh 6 none A0 \
#       --network-blocks 12 --simulations 64          # an architecture arm
#
# Arguments after the third are forwarded verbatim, so a sweep does not need a
# config file or a launcher script per shape.
#
# TRAIN_ROOT is deliberately outside the repository.  Checkpoints and replay
# chunks are large and regenerable; the previous run put 213 MB of them into
# git and it had to be taken back out again.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ROOT="${TRAIN_ROOT:-/workspace/alphadiamond-training}"
RUN_ID="${RUN_ID:-soo-scratch-20260822}"
CONFIG="${CONFIG:-$ROOT/runtime/configs/soo-rtx5090-native.json}"

HOURS="${1:-6}"
PRIOR="${2:-canonical-target-vacancy-distance-v2}"
PHASE="${3:-B0}"
shift 3 2>/dev/null || shift $# # anything further is forwarded to az_train.py

# A0 needs 128 simulations and the config default is 64.
#
# The config's 64 is the B0 setting and always has been; A0 was only ever run at
# 128 because the operator passed the flag, and the handoff records that as a
# comment.  A comment is not a guard: resuming A0 without it starts training on
# 64-simulation targets, which §6.8 measured as not being policy-improvement
# targets at all -- the loop degrades while the loss improves, so nothing in the
# output says anything is wrong.  Paid for once, on a production checkpoint.
case " $* " in
  *" --simulations "*) ;;
  *) if [ "$PHASE" = "A0" ]; then
       echo "[guard] phase A0 with no --simulations; using 128 (config default 64 is the B0 setting)" >&2
       set -- --simulations 128 "$@"
     fi ;;
esac

# The run imports from a pinned snapshot, not the working tree: repo work
# continues while training runs, and a rebuilt extension under the working tree
# must not reach a run already in flight.
export PYTHONPATH="$TRAIN_ROOT/pinned-src"
if [ ! -d "$PYTHONPATH" ]; then
  echo "[setup] pinning $ROOT/src -> $PYTHONPATH"
  mkdir -p "$TRAIN_ROOT"
  cp -r "$ROOT/src" "$PYTHONPATH"
fi
ACTUAL="$(python -c 'import diamond, pathlib; print(pathlib.Path(diamond.__file__).resolve())')"
case "$ACTUAL" in
  "$PYTHONPATH"/*) ;;
  *) echo "[abort] diamond imported from $ACTUAL, not the pinned snapshot" >&2; exit 1 ;;
esac

exec python "$ROOT/tools/az_train.py" \
  --config "$CONFIG" \
  --runtime-dir "$TRAIN_ROOT/runs" \
  --run-id "$RUN_ID" \
  --phase "$PHASE" \
  --bootstrap-prior "$PRIOR" \
  --hours "$HOURS" \
  --workers 12 \
  --native-lanes 256 \
  --train-steps-per-iteration 300 \
  --native-max-wait-us "${NATIVE_MAX_WAIT_US:-50}" \
  --archive-every 25 \
  --keep-archives 20 \
  "$@"
