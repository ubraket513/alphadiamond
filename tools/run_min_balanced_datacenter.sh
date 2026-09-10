#!/usr/bin/env bash
# Linux, single-GPU, bounded Min turn-order experiment.
set -euo pipefail

mode=${1:-smoke}
case "$mode" in
  smoke|pilot|control) ;;
  -h|--help)
    echo 'LIBTORCH_ROOT=/path/to/torch [CXX=g++-14] bash tools/run_min_balanced_datacenter.sh smoke|pilot|control'
    echo 'Optional: OUTPUT_ROOT, BUILD_DIR, BUILD_JOBS, TORCH_CUDA_ARCH_LIST, RUN_ID'
    exit 0 ;;
  *) echo 'Expected smoke, pilot, or control' >&2; exit 2 ;;
esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
: "${LIBTORCH_ROOT:?Set LIBTORCH_ROOT to a CUDA-enabled LibTorch root (or the installed torch package directory)}"
[[ $(uname -s) == Linux ]] || { echo 'This launcher requires Linux.' >&2; exit 2; }
for tool in cmake ctest ninja hf unzip sha256sum nvidia-smi; do
  command -v "$tool" >/dev/null || { echo "Missing command: $tool" >&2; exit 2; }
done
[[ -f "$LIBTORCH_ROOT/share/cmake/Torch/TorchConfig.cmake" && -f "$LIBTORCH_ROOT/lib/libtorch_cuda.so" ]] || {
  echo 'LIBTORCH_ROOT must contain TorchConfig.cmake and CUDA LibTorch libraries.' >&2; exit 2;
}
nvidia-smi -L
hf auth whoami >/dev/null
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u | paste -sd ';' -)}
[[ -n "$TORCH_CUDA_ARCH_LIST" ]] || { echo 'Cannot determine GPU compute capability.' >&2; exit 2; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export LD_LIBRARY_PATH="$LIBTORCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
output=${OUTPUT_ROOT:-$repo/artifacts/min-balanced-datacenter}
mkdir -p "$output"
output=$(cd "$output" && pwd)
assets="$output/assets"
mkdir -p "$assets"
archive="$assets/min-step20064.zip"
bucket='hf://buckets/ubraket513/alphadiamond-min-private-backups/min-balanced-orders-20260911/v1'
if [[ ! -f "$archive" ]]; then
  hf buckets cp "$bucket/min-step20064.zip" "$archive"
fi
printf '%s  %s\n' 'a656bbd683a14048defe17d80365d3d3877580a86c5a19be123ee2c606f6f965' "$archive" | sha256sum --check --status
artifact="$assets/step20064/models/min/2.0.0-min-test.20064"
if [[ ! -d "$assets/step20064" ]]; then
  mkdir "$assets/step20064"
  unzip -q "$archive" -d "$assets/step20064"
fi
[[ -f "$artifact/metadata.json" ]] || { echo 'Starting artifact is incomplete.' >&2; exit 2; }

build=${BUILD_DIR:-$repo/build/min-balanced-datacenter}
cmake -S "$repo" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT" \
  -DDIAMOND_BUILD_LIBTORCH=ON -DDIAMOND_BUILD_QT=OFF -DBUILD_TESTING=ON
cmake --build "$build" --target alphadiamond-train selfplay_test config_test --parallel "${BUILD_JOBS:-4}"
ctest --test-dir "$build" -R '^(selfplay_test|config_test)$' --output-on-failure

config="$repo/configs/alphazero/min-balanced-orders-cuda-$mode.json"
run_id=${RUN_ID:-min-orders-$mode-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid RUN_ID.' >&2; exit 2; }
run="$output/runs/min/$run_id"
[[ ! -e "$run" ]] || { echo "Run already exists: $run (use native resume explicitly)" >&2; exit 2; }
mkdir -p "$run"
{
  printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'mode=%s\narchitecture=%s\n' "$mode" "$TORCH_CUDA_ARCH_LIST"
  git status --short
  nvidia-smi
} > "$run/host.txt"
"$build/native/alphadiamond-train" train --run-dir "$run" --config "$config" \
  --warm-start "$artifact" | tee "$run/launch.json"
printf '\nRun: %s\nInspect iterations/0/selfplay.metrics.json for per-order completion and retained samples.\n' "$run"
printf 'Do not interpret the CLI processed-game total as the number of non-aborted games.\n'
