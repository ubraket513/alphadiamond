#!/usr/bin/env bash
set -euo pipefail

source "${MIN_A0_ENV:-/workspace/alphadiamond-experiments/min-v1.0.1/env.sh}"
BIN=${MIN_A0_SELFPLAY_BIN:-/workspace/alphadiamond-min-a0/build/native-training/native/selfplay_benchmark}
OUT=${MIN_A0_SWEEP_OUT:-$MIN_A0_EXP/results/serial-search}
PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
scale=${MIN_A0_SWEEP_SCALE:-smoke}
games=${MIN_A0_GAMES:-32}
lanes=${MIN_A0_LANES:-32}
if (( lanes >= games )); then
  lanes=$(( games / 2 ))
  (( lanes > 0 )) || lanes=1
  printf 'adjusted lanes to %d because selfplay_benchmark requires games > lanes\n' "$lanes" >&2
fi
filter=${MIN_A0_ARM_FILTER:-}
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
  dry_run=true
  scale=${2:-smoke}
fi
mkdir -p "$OUT/$scale"

common=(
  --checkpoint "$MIN_A0_CHECKPOINT"
  --config "$MIN_A0_CONFIG"
  --device cuda
  --precision fp32
  --threads 16
  --max-batch 256
  --max-wait-us 100
  --max-moves 800
  --max-game-seconds 180
  --temperature 1.0
  --temperature-moves 20
  --dirichlet-epsilon 0.25
  --diagnostic-roots 4096
  --diagnostic-batch 256
  --warmups 0
  --repetitions 1
  --seed 20260901
)

selected() {
  [[ -z $filter || ,$filter, == *,$1,* ]]
}

run_arm() {
  local name=$1 arm_games=$2 arm_lanes=$3 prior=$4 simulations=$5 late=$6 repeat=$7
  selected "$name" || return 0
  local destination="$OUT/$scale/$name.json"
  local command=("$BIN" "${common[@]}" --games "$arm_games" --lanes "$arm_lanes"
    --bootstrap-prior "$prior" --simulations "$simulations"
    --simulations-late "$late" --repeat-window "$repeat")
  if $dry_run; then
    printf '%q ' "${command[@]}"
    printf '> %q\n' "$destination"
    return 0
  fi
  local temporary="$destination.tmp"
  rm -f -- "$temporary"
  if LD_PRELOAD="$PRELOAD" "${command[@]}" >"$temporary"; then
    mv -- "$temporary" "$destination"
  else
    local status=$?
    rm -f -- "$temporary"
    return "$status"
  fi
}

run_arm b0-128 "$games" "$lanes" vacancy 128 0 0
run_arm b0-256 "$games" "$lanes" vacancy 256 0 0
run_arm b0-400 "$games" "$lanes" vacancy 400 0 0
run_arm a0-128 "$games" "$lanes" none 128 0 0
run_arm a0-256 "$games" "$lanes" none 256 0 0
run_arm a0-400 "$games" "$lanes" none 400 0 0
run_arm a0-adaptive-256 "$games" "$lanes" none 128 256 8
run_arm a0-adaptive-400 "$games" "$lanes" none 128 400 8
