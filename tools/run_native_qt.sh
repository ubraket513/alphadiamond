#!/usr/bin/env sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
deploy_dir='dist/diamond-qt'
soo=0
simulations=128

usage() {
    echo "usage: tools/run_native_qt.sh [--deploy-dir DIR] [--soo] [--simulations N]" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --deploy-dir) [ "$#" -ge 2 ] || usage; deploy_dir=$2; shift 2 ;;
        --soo) soo=1; shift ;;
        --simulations) [ "$#" -ge 2 ] || usage; simulations=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

case "$simulations" in
    ''|*[!0-9]*) echo "--simulations must be an integer from 1 to 4096" >&2; exit 2 ;;
esac
[ "$simulations" -ge 1 ] && [ "$simulations" -le 4096 ] || {
    echo "--simulations must be an integer from 1 to 4096" >&2
    exit 2
}

if [ "$soo" -eq 1 ] && [ "$deploy_dir" = 'dist/diamond-qt' ]; then
    deploy_dir='dist/diamond-qt-soo'
fi
case "$deploy_dir" in /*) directory=$deploy_dir ;; *) directory=$repo/$deploy_dir ;; esac
[ -d "$directory" ] || { echo "deployment directory not found: $directory" >&2; exit 1; }
directory=$(CDPATH= cd -- "$directory" && pwd -P)
exe=$directory/diamond_qt.exe
[ -f "$exe" ] || { echo "executable not found: $exe" >&2; exit 1; }

export QT_QPA_PLATFORM=windows
export DIAMOND_MCTS_SIMULATIONS=$simulations
unset QT_DEBUG_PLUGINS || true

CDPATH= cd -- "$directory"
./diamond_qt.exe >/dev/null 2>&1 &
pid=$!
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    set +e
    wait "$pid"
    status=$?
    set -e
    echo "diamond_qt.exe exited immediately with code $status" >&2
    exit 1
fi

echo "Diamond GUI started (PID $pid)."
