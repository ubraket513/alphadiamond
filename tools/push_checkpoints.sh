#!/bin/sh
# Push candidate checkpoints from a run directory to the Hugging Face bucket.
#
# The bucket is mutable, non-versioned object storage, so this script only ever
# creates: --ignore-existing means an already-pushed generation is never
# rewritten, and no delete operation is used.  Only candidate checkpoints are
# pushed; replay is regenerable and stays local.
set -eu

usage() {
    echo "usage: $0 --run-dir <run-dir> --bucket <namespace/bucket> [--family soo|min] [--iteration <n>] [--dry-run]" >&2
    exit 2
}

run_dir= bucket= family=soo iteration= dry_run=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-dir|--bucket|--family|--iteration)
            [ "$#" -ge 2 ] || usage
            case "$1" in
                --run-dir) run_dir=$2 ;;
                --bucket) bucket=$2 ;;
                --family) family=$2 ;;
                --iteration) iteration=$2 ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) usage ;;
    esac
done
[ -n "$run_dir" ] && [ -n "$bucket" ] || usage
[ -d "$run_dir/iterations" ] || { echo "push-checkpoints: no iterations under $run_dir" >&2; exit 1; }
case "$family" in soo|min) ;; *) echo "push-checkpoints: --family must be soo or min" >&2; exit 2 ;; esac
case "$iteration" in ''|*[0-9]) ;; *) echo "push-checkpoints: --iteration must be a number" >&2; exit 2 ;; esac

hf auth whoami >/dev/null 2>&1 || {
    echo "push-checkpoints: Hugging Face authentication failed; run 'hf auth login'" >&2
    exit 1
}

run_id=$(basename "$run_dir")
remote_root="hf://buckets/$bucket/checkpoints/$family/$run_id"
pushed=0

for candidate in "$run_dir"/iterations/*/candidate-checkpoint; do
    [ -d "$candidate" ] || continue
    # A checkpoint without CURRENT is mid-write; never publish a partial one.
    [ -f "$candidate/CURRENT" ] || continue
    n=$(basename "$(dirname "$candidate")")
    [ -n "$iteration" ] && [ "$n" != "$iteration" ] && continue

    remote="$remote_root/iteration-$n"
    if [ "$dry_run" -eq 1 ]; then
        echo "dry-run sync $candidate -> $remote"
        hf sync "$candidate" "$remote" --ignore-existing --dry-run || {
            echo "push-checkpoints: dry-run failed for iteration $n" >&2; exit 1; }
    else
        hf sync "$candidate" "$remote" --ignore-existing >/dev/null || {
            echo "push-checkpoints: upload failed for iteration $n" >&2; exit 1; }
        echo "pushed iteration $n -> $remote"
    fi
    pushed=$((pushed + 1))
done

[ "$pushed" -gt 0 ] || { echo "push-checkpoints: no candidate checkpoint found to push" >&2; exit 1; }
