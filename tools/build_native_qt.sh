#!/usr/bin/env sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
: "${CONDA_PREFIX:?activate the alphadiamond mamba environment first}"

cd -- "$repo"
tools/native_training.sh cmake --preset native-package
tools/native_training.sh cmake --build --preset native-package --target diamond_qt --parallel 1
tools/deploy_native_qt.sh \
    --build-dir build/native-package \
    --output-dir dist/diamond-qt-soo \
    --with-soo \
    --environment-root "$CONDA_PREFIX"
