#!/usr/bin/env sh
set -eu

usage() {
    echo "usage: tools/native_training.sh cmake|ctest [args...]" >&2
    exit 2
}

[ "$#" -ge 1 ] || usage
tool=$1
shift
case "$tool" in
    cmake|ctest) ;;
    *) usage ;;
esac

find_tool() {
    if [ -n "${DIAMOND_CMAKE_BIN:-}" ] && [ -x "$DIAMOND_CMAKE_BIN/$tool.exe" ]; then
        printf '%s\n' "$DIAMOND_CMAKE_BIN/$tool.exe"
        return
    fi
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/Library/bin/$tool.exe" ]; then
        printf '%s\n' "$CONDA_PREFIX/Library/bin/$tool.exe"
        return
    fi
    command -v "$tool" || return 1
}

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        tool_path=$(find_tool) || {
            echo "$tool is unavailable; activate the alphadiamond environment or set DIAMOND_CMAKE_BIN" >&2
            exit 1
        }
        inherited_path=$PATH
        environment_root=${CONDA_PREFIX:-}
        case "$environment_root" in
            [A-Za-z]:\\*|[A-Za-z]:/*) environment_root=$(cygpath -u "$environment_root") ;;
        esac

        vsdevcmd=''
        for candidate in \
            '/c/Program Files/Microsoft Visual Studio/18/Community/Common7/Tools/VsDevCmd.bat' \
            '/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/Tools/VsDevCmd.bat' \
            '/c/Program Files/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat'
        do
            if [ -f "$candidate" ]; then
                vsdevcmd=$candidate
                break
            fi
        done
        if [ -z "$vsdevcmd" ]; then
            echo "Visual Studio x64 developer environment is unavailable" >&2
            exit 1
        fi

        environment_dump=$(mktemp "${TMPDIR:-/tmp}/alphadiamond-vs-env.XXXXXX")
        trap 'rm -f -- "$environment_dump"' EXIT HUP INT TERM
        vsdevcmd_windows=$(cygpath -d "$vsdevcmd")
        if ! cmd.exe //d //s //c "call $vsdevcmd_windows -arch=x64 -host_arch=x64 >nul && set" >"$environment_dump"; then
            echo "failed to initialize the Visual Studio x64 environment" >&2
            exit 1
        fi
        sed -i 's/\r$//' "$environment_dump"

        while IFS='=' read -r name value; do
            case "$name" in
                ''|[0-9]*|*[!A-Za-z0-9_]*) continue ;;
                PATH|Path)
                    PATH=$(cygpath -p "$value")
                    export PATH
                    ;;
                *) export "$name=$value" ;;
            esac
        done <"$environment_dump"

        if [ -n "$environment_root" ]; then
            for runtime_dir in \
                "$environment_root/Library/lib/qt6/bin" \
                "$environment_root/Library/bin" \
                "$environment_root/Lib/site-packages/torch/lib"
            do
                if [ -d "$runtime_dir" ]; then PATH=$PATH:$runtime_dir; fi
            done
        fi
        PATH=$PATH:$inherited_path
        export PATH

        exec "$tool_path" "$@"
        ;;
    *)
        exec "$tool" "$@"
        ;;
esac
