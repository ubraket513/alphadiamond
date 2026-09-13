#!/usr/bin/env sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
: "${CONDA_PREFIX:?activate the alphadiamond mamba environment first}"
: "${DIAMOND_QT_ROOT:=$HOME/Qt/6.12.0/msvc2022_64}"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        DIAMOND_QT_ROOT=$(cygpath -m "$DIAMOND_QT_ROOT")
        CONDA_PREFIX=$(cygpath -m "$CONDA_PREFIX")
        export QT_QPA_PLATFORM=windows
        ;;
esac
export DIAMOND_QT_ROOT CONDA_PREFIX
export QMLPREVIEW_HOTRELOAD=1

cd -- "$repo"
jobs=${CMAKE_BUILD_PARALLEL_LEVEL:-${NUMBER_OF_PROCESSORS:-$(getconf _NPROCESSORS_ONLN)}}
tools/native_training.sh cmake --preset native-qt-preview -G Ninja
tools/native_training.sh cmake --build --preset native-qt-preview --target diamond_qt --parallel "$jobs"
# qmlpreview starts the app over a local debugging socket, maps its resources
# back to the source files, and watches saves. Qt 6.12 enables hot reload.
tools/native_training.sh cmake --build --preset native-qt-preview --target diamond_qt_preview
