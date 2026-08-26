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
        vswhere='/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
        if [ -x "$vswhere" ]; then
            installation_path=$("$vswhere" \
                -latest \
                -products '*' \
                -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
                -property installationPath | tr -d '\r' | sed -n '1p')
            if [ -n "$installation_path" ]; then
                candidate=$(cygpath -u "$installation_path")/Common7/Tools/VsDevCmd.bat
                if [ -f "$candidate" ]; then vsdevcmd=$candidate; fi
            fi
        fi

        if [ -z "$vsdevcmd" ]; then
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
        fi
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

        developer_path=''
        fallback_path=''
        while IFS='=' read -r name value; do
            case "$name" in
                ''|[0-9]*|*[!A-Za-z0-9_]*) continue ;;
                PATH|Path)
                    converted_path=$(cygpath -p "$value")
                    case "$value" in
                        *\\VC\\Tools\\MSVC\\*) developer_path=$converted_path ;;
                        *)
                            if [ -z "$fallback_path" ]; then fallback_path=$converted_path; fi
                            ;;
                    esac
                    ;;
                *) export "$name=$value" ;;
            esac
        done <"$environment_dump"

        PATH=${developer_path:-$fallback_path}
        export PATH

        runtime_path=''
        append_runtime_dir() {
            runtime_dir=$1
            if [ -d "$runtime_dir" ]; then
                if [ -n "$runtime_path" ]; then
                    runtime_path=$runtime_path:$runtime_dir
                else
                    runtime_path=$runtime_dir
                fi
            fi
        }

        explicit_torch_runtime=${DIAMOND_TORCH_RUNTIME_DIR:-}
        case "$explicit_torch_runtime" in
            [A-Za-z]:\\*|[A-Za-z]:/*) explicit_torch_runtime=$(cygpath -u "$explicit_torch_runtime") ;;
        esac
        if [ -n "$explicit_torch_runtime" ]; then
            if [ ! -d "$explicit_torch_runtime" ]; then
                echo "DIAMOND_TORCH_RUNTIME_DIR does not exist: $explicit_torch_runtime" >&2
                exit 1
            fi
            append_runtime_dir "$explicit_torch_runtime"
        fi

        if [ -n "$environment_root" ]; then
            for runtime_dir in \
                "$environment_root/Library/bin" \
                "$environment_root/Scripts" \
                "$environment_root" \
                "$environment_root/Lib/site-packages/torch/lib" \
                "$environment_root/Library/lib/qt6/bin"
            do
                append_runtime_dir "$runtime_dir"
            done
        fi
        if [ -n "$runtime_path" ]; then PATH=$runtime_path:$PATH; fi
        PATH=$PATH:$inherited_path
        export PATH

        exec "$tool_path" "$@"
        ;;
    *)
        exec "$tool" "$@"
        ;;
esac
