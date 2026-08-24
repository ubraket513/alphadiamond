#!/usr/bin/env sh
set -eu

build_dir='build/native-package'
output_dir='dist/diamond-qt'
with_soo=0
environment_root=${CONDA_PREFIX:-}

usage() {
    echo "usage: tools/deploy_native_qt.sh [--build-dir DIR] [--output-dir DIR] [--with-soo] [--environment-root DIR]" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-dir) [ "$#" -ge 2 ] || usage; build_dir=$2; shift 2 ;;
        --output-dir) [ "$#" -ge 2 ] || usage; output_dir=$2; shift 2 ;;
        --environment-root) [ "$#" -ge 2 ] || usage; environment_root=$2; shift 2 ;;
        --with-soo) with_soo=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) ;;
    *) echo "native Qt deployment is currently supported from Windows Git Bash" >&2; exit 1 ;;
esac

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
case "$build_dir" in /*) ;; *) build_dir=$repo/$build_dir ;; esac
case "$output_dir" in /*) ;; *) output_dir=$repo/$output_dir ;; esac
case "$environment_root" in
    [A-Za-z]:\\*|[A-Za-z]:/*) environment_root=$(cygpath -u "$environment_root") ;;
esac

[ -n "$environment_root" ] && [ -d "$environment_root" ] || {
    echo "activate alphadiamond or pass --environment-root" >&2
    exit 1
}

exe=''
for candidate in \
    "$build_dir/native/qt/diamond_qt.exe" \
    "$build_dir/native/qt/Release/diamond_qt.exe"
do
    if [ -f "$candidate" ]; then exe=$candidate; break; fi
done
[ -n "$exe" ] || { echo "Qt executable not found under $build_dir" >&2; exit 1; }

qt_bin=$environment_root/Library/lib/qt6/bin
qt_root=$environment_root/Library/lib/qt6
env_bin=$environment_root/Library/bin
[ -d "$qt_bin" ] || { echo "Qt runtime directory not found: $qt_bin" >&2; exit 1; }

dist_root=$repo/dist
mkdir -p -- "$dist_root"
dist_root=$(CDPATH= cd -- "$dist_root" && pwd -P)
destination_parent=$(dirname -- "$output_dir")
[ -d "$destination_parent" ] || {
    echo "output parent does not exist: $destination_parent" >&2
    exit 1
}
destination_parent=$(CDPATH= cd -- "$destination_parent" && pwd -P)
[ "$destination_parent" = "$dist_root" ] || {
    echo "--output-dir must be a direct child of $dist_root" >&2
    exit 1
}
destination=$dist_root/$(basename -- "$output_dir")

is_reparse_point() {
    path=$1
    [ -L "$path" ] && return 0
    [ -e "$path" ] || return 1
    windows_path=$(cygpath -w "$path")
    cmd.exe //d //s //c "fsutil reparsepoint query \"$windows_path\" >nul 2>nul" >/dev/null 2>&1
}

if is_reparse_point "$dist_root"; then
    echo "deployment root must not be a reparse point: $dist_root" >&2
    exit 1
fi
if [ -e "$destination" ]; then
    if is_reparse_point "$destination"; then
        echo "refusing to replace deployment reparse point: $destination" >&2
        exit 1
    fi
    rm -rf -- "$destination"
fi
mkdir -p -- "$destination/assets/sounds"
cp -f -- "$exe" "$destination/diamond_qt.exe"
cp -f -- "$repo/native/qt/assets/sounds/move.m4a" "$destination/assets/sounds/move.m4a"

copy_glob() {
    target=$1
    shift
    for pattern in "$@"; do
        for file in $pattern; do
            [ -f "$file" ] || continue
            cp -f -- "$file" "$target/"
        done
    done
}

copy_glob "$destination" "$qt_bin"/Qt6*.dll
mkdir -p -- "$destination/plugins"
for plugin_type in generic iconengines imageformats multimedia networkinformation platforms styles tls; do
    if [ -d "$qt_root/plugins/$plugin_type" ]; then
        cp -R -- "$qt_root/plugins/$plugin_type" "$destination/plugins/"
    fi
done
cp -R -- "$qt_root/qml" "$destination/qml"

copy_glob "$destination" \
    "$env_bin"/msvcp140*.dll "$env_bin"/vcruntime140*.dll "$env_bin"/concrt140*.dll \
    "$env_bin"/icu*.dll "$env_bin"/pcre2*.dll "$env_bin"/zlib*.dll \
    "$env_bin"/zstd*.dll "$env_bin"/double-conversion*.dll "$env_bin"/freetype*.dll \
    "$env_bin"/harfbuzz*.dll "$env_bin"/libpng*.dll "$env_bin"/brotli*.dll \
    "$env_bin"/md4c*.dll "$env_bin"/libcrypto*.dll "$env_bin"/libssl*.dll

printf '%s\n' '[Paths]' 'Plugins = plugins' 'Qml2Imports = qml' >"$destination/qt.conf"

artifact_source=$repo/artifacts/soo-spike
[ -d "$artifact_source" ] || { echo "Soo artifacts not found: $artifact_source" >&2; exit 1; }
mkdir -p -- "$destination/artifacts"
if [ "$with_soo" -eq 1 ]; then
    cp -R -- "$artifact_source" "$destination/artifacts/"
else
    shell_artifact=$destination/artifacts/soo-spike
    mkdir -p -- "$shell_artifact"
    for topology_file in \
        topology_neighbour.i8 topology_camp_positions.i32 topology_pairwise_distance.i32 \
        topology_physical_to_canonical.i32 topology_canonical_to_physical.i32
    do
        cp -f -- "$artifact_source/$topology_file" "$shell_artifact/$topology_file"
    done
fi

if [ "$with_soo" -eq 1 ]; then
    torch_lib=''
    for candidate in \
        "$environment_root/Lib/site-packages/torch/lib" \
        "$environment_root/lib/site-packages/torch/lib"
    do
        if [ -d "$candidate" ]; then torch_lib=$candidate; break; fi
    done
    if [ -z "$torch_lib" ]; then
        for candidate in "$environment_root"/lib/python*/site-packages/torch/lib; do
            if [ -d "$candidate" ]; then torch_lib=$candidate; break; fi
        done
    fi
    [ -n "$torch_lib" ] || { echo "LibTorch runtime directory not found under $environment_root" >&2; exit 1; }
    copy_glob "$destination" "$torch_lib"/*.dll
    copy_glob "$destination" \
        "$env_bin"/fbgemm*.dll "$env_bin"/asmjit*.dll "$env_bin"/mkl*.dll \
        "$env_bin"/libiomp*.dll "$env_bin"/vcomp*.dll "$env_bin"/tbb*.dll \
        "$env_bin"/sleef*.dll "$env_bin"/uv*.dll "$env_bin"/libomp*.dll \
        "$env_bin"/libprotobuf*.dll "$env_bin"/utf8_validity*.dll "$env_bin"/abseil_dll*.dll
    [ -f "$destination/c10.dll" ] || { echo "c10.dll was not bundled" >&2; exit 1; }
    [ -f "$destination/torch_cpu.dll" ] || { echo "torch_cpu.dll was not bundled" >&2; exit 1; }
fi

forbidden=$(find "$destination" -type f \( \
    -iname 'python*.dll' -o -iname 'PySide*' -o -iname 'qtawesome*' \
    -o -path '*/site-packages/*' \) -print -quit)
[ -z "$forbidden" ] || { echo "Python/PySide runtime leaked into deployment: $forbidden" >&2; exit 1; }

run_smoke() {
    argument=$1
    windows_destination=$(cygpath -w "$destination")
    windows_root=${SystemRoot:-C:\\Windows}
    env -i \
        MSYS2_ARG_CONV_EXCL='*' \
        SystemRoot="$windows_root" \
        WINDIR="$windows_root" \
        PATH="$windows_destination;$windows_root\\System32;$windows_root" \
        QT_QPA_PLATFORM=offscreen \
        "$destination/diamond_qt.exe" "$argument"
}

for argument in --smoke --game-smoke --worker-smoke --rotation-smoke \
    --analysis-smoke --failure-smoke --sound-smoke
do
    run_smoke "$argument"
done
if [ "$with_soo" -eq 1 ]; then run_smoke --soo-smoke; fi

echo "Native Qt deployment created at: $destination"
echo "Packaged runtime smoke checks passed."
