#!/usr/bin/env bash
# Run one benchmark point under a pinned source snapshot, with sampling.
#
# Usage:
#   run_point.sh <run-id> <config-path> <workers> <train-steps> [extra az_train args...]
#
# Guarantees, in order, and every one of them aborts the point rather than
# producing a row that looks plausible and is not:
#
#   1. The source under test is the pinned snapshot in $BENCH_SRC, asserted via
#      diamond.__file__ before anything starts.  The previous GPU pass lost a
#      whole sweep to an editable install silently importing the working tree;
#      `diamond` is not installed on this host, so PYTHONPATH is authoritative,
#      but the assertion stays because "not installed today" is not a guarantee.
#   2. The run starts from a fresh copy of the immutable step-80 checkpoint,
#      verified by SHA-256 before the run and by the '[resume] ... training_step'
#      line after it.  An '[init] wrote initial checkpoint' line means the copy
#      failed and the run trained a random network -- that row is discarded.
#   3. A process-tree + GPU sampler runs for exactly the life of the run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_SRC="${BENCH_SRC:?set BENCH_SRC to the pinned source snapshot (…/src)}"
CHECKPOINT="${CHECKPOINT:-$REPO/runtime/runs/soo/cpu8h-soo-20260819/latest.pt}"
EXPECTED_SHA="${EXPECTED_SHA:-1634b901e213b065c107eea734b8c172c14babb1c2565352203961e86ea165af}"
EXPECTED_STEP="${EXPECTED_STEP:-80}"
INTERVAL="${INTERVAL:-2}"

RUN_ID="${1:?run-id}"
CONFIG="${2:?config path}"
WORKERS="${3:?workers}"
TRAIN_STEPS="${4:?train steps per iteration}"
shift 4

RUN_DIR="$REPO/az-bench/soo/$RUN_ID"
LOG="$RUN_DIR/run.log"
SAMPLES="$RUN_DIR/samples.csv"

export PYTHONPATH="$BENCH_SRC"

# ---- 1. source isolation -------------------------------------------------
ACTUAL_SRC="$(python -c 'import diamond, pathlib; print(pathlib.Path(diamond.__file__).resolve())')"
case "$ACTUAL_SRC" in
  "$BENCH_SRC"/*) ;;
  *) echo "[abort] diamond imported from $ACTUAL_SRC, expected under $BENCH_SRC" >&2; exit 1 ;;
esac

# ---- 2. immutable checkpoint --------------------------------------------
ACTUAL_SHA="$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "[abort] checkpoint digest $ACTUAL_SHA != expected $EXPECTED_SHA" >&2
  exit 1
fi

if [ -e "$RUN_DIR" ]; then
  echo "[abort] $RUN_DIR already exists; pick a fresh run-id" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cp "$CHECKPOINT" "$RUN_DIR/latest.pt"

# Provenance goes into the run directory as well as stdout.  Keeping it only on
# the driver's stdout scattered the evidence across sweep logs, and a commit
# landing mid-sweep made "which source did this row actually run?" a question
# that had to be reconstructed after the fact rather than read off the run.
PROVENANCE="$RUN_DIR/provenance.txt"
{
  echo "run=$RUN_ID"
  echo "workers=$WORKERS"
  echo "train_steps=$TRAIN_STEPS"
  echo "config=$CONFIG"
  echo "source=$ACTUAL_SRC"
  echo "checkpoint=$CHECKPOINT"
  echo "checkpoint_sha256=$ACTUAL_SHA"
  echo "git_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty=$(git -C "$REPO" status --porcelain -- src tests 2>/dev/null | wc -l)"
  echo "started_at=$(date -Is)"
} | tee "$PROVENANCE" | sed 's/^/[point] /'

# ---- 3. run, sampled -----------------------------------------------------
python "$REPO/tools/az_train.py" \
  --config "$CONFIG" \
  --runtime-dir "$REPO/az-bench" \
  --run-id "$RUN_ID" \
  --workers "$WORKERS" \
  --simulations 64 \
  --train-steps-per-iteration "$TRAIN_STEPS" \
  --hours 0.01 \
  "$@" > "$LOG" 2>&1 &
TRAIN_PID=$!

python "$REPO/az-bench/profiles/sample_run.py" \
  --pid "$TRAIN_PID" --out "$SAMPLES" --interval "$INTERVAL" &
SAMPLER_PID=$!

set +e
wait "$TRAIN_PID"
STATUS=$?
set -e
wait "$SAMPLER_PID" 2>/dev/null || true

# ---- 4. validate the row -------------------------------------------------
if [ "$STATUS" -ne 0 ]; then
  echo "[abort] az_train exited $STATUS; last lines:" >&2
  tail -20 "$LOG" >&2
  exit "$STATUS"
fi
if grep -q '\[init\] wrote initial checkpoint' "$LOG"; then
  echo "[abort] run trained a random network -- INVALID, discard $RUN_ID" >&2
  exit 1
fi
if ! grep -q "\[resume\] loaded .* training_step=$EXPECTED_STEP" "$LOG"; then
  echo "[abort] run did not resume at training_step=$EXPECTED_STEP:" >&2
  grep -E '\[resume\]|\[init\]' "$LOG" >&2 || true
  exit 1
fi

grep -E '^\[(resume|start|i[0-9]+|done)\]' "$LOG" || true
echo "[point] ok: $RUN_ID"
