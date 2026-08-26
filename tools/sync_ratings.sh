#!/bin/sh
# Immutable Hugging Face Bucket sync for rating v2 events.  No delete operation is used.
set -eu

usage() {
    echo "usage: $0 --bucket <namespace/bucket> --outbox <rating-outbox> --protocol <protocol-v2.json> --binary <alphadiamond-rating-sync> [--retries <n>] [--dry-run]" >&2
    exit 2
}

bucket= outbox= protocol= binary= retries=3 dry_run=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --bucket|--outbox|--protocol|--binary|--retries)
            [ "$#" -ge 2 ] || usage
            case "$1" in
                --bucket) bucket=$2 ;;
                --outbox) outbox=$2 ;;
                --protocol) protocol=$2 ;;
                --binary) binary=$2 ;;
                --retries) retries=$2 ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) usage ;;
    esac
done
[ -n "$bucket" ] && [ -n "$outbox" ] && [ -n "$protocol" ] && [ -n "$binary" ] || usage
[ -x "$binary" ] || { echo "rating sync: executable is not runnable: $binary" >&2; exit 1; }
[ -f "$protocol" ] || { echo "rating sync: missing protocol: $protocol" >&2; exit 1; }
[ -d "$outbox/events" ] || { echo "rating sync: missing outbox events: $outbox/events" >&2; exit 1; }
case "$retries" in ''|*[!0-9]*|0) echo "rating sync: --retries must be positive" >&2; exit 2;; esac

hf auth whoami >/dev/null || { echo "rating sync: Hugging Face authentication failed" >&2; exit 1; }

work_parent=${TMPDIR:-/tmp}
work=$(mktemp -d "$work_parent/alphadiamond-ratings.XXXXXX") || exit 1
trap 'rm -rf "$work"' EXIT HUP INT TERM
remote_uri="hf://buckets/$bucket"
list_remote() {
    hf buckets list "$bucket" -R -q >"$work/list.raw" || {
        echo "rating sync: remote event listing failed" >&2; exit 1;
    }
    sed -n 's#^\(ratings/events/\)\{0,1\}\([0-9a-fA-F]\{64\}\.json\)$#ratings/events/\2#p' "$work/list.raw" | LC_ALL=C sort -u
}

remote_contains() {
    list_remote >"$work/current-events"
    grep -Fx "$1" "$work/current-events" >/dev/null 2>&1
}

check_event_name() {
    case "$1" in *.json) digest=${1%.json} ;; *) return 1 ;; esac
    [ "${#digest}" -eq 64 ] && printf '%s' "$digest" | LC_ALL=C grep -Eq '^[0-9a-fA-F]{64}$'
}

for event in "$outbox/events"/*.json; do
    [ -f "$event" ] || continue
    name=$(basename "$event")
    check_event_name "$name" || { echo "rating sync: outbox event filename is not a SHA-256 JSON name: $name" >&2; exit 1; }
    remote="ratings/events/$name"
    if remote_contains "$remote"; then
        hf buckets cp "$remote_uri/$remote" "$work/remote-$name" >/dev/null
        cmp -s "$event" "$work/remote-$name" || { echo "rating sync: remote event conflicts: $remote" >&2; exit 1; }
    elif [ "$dry_run" -eq 1 ]; then
        echo "dry-run upload $remote"
    else
        hf buckets cp "$event" "$remote_uri/$remote" >/dev/null || { echo "rating sync: upload failed: $name" >&2; exit 1; }
    fi
done

attempt=1
while [ "$attempt" -le "$retries" ]; do
    events="$work/events-$attempt"
    mkdir "$events"
    before="$work/before-$attempt"
    list_remote >"$before"
    while IFS= read -r remote; do
        [ -n "$remote" ] || continue
        name=${remote##*/}
        hf buckets cp "$remote_uri/$remote" "$events/$name" >/dev/null || { echo "rating sync: download failed: $remote" >&2; exit 1; }
    done <"$before"
    "$binary" --protocol "$protocol" --events-dir "$events" --output "$work/ratings-$attempt.json"
    after="$work/after-$attempt"
    list_remote >"$after"
    if ! cmp -s "$before" "$after"; then
        attempt=$((attempt + 1)); continue
    fi
    if [ "$dry_run" -eq 1 ]; then
        echo "dry-run upload ratings/ratings.json"
        exit 0
    fi
    hf buckets cp "$work/ratings-$attempt.json" "$remote_uri/ratings/ratings.json" >/dev/null || { echo "rating sync: ratings upload failed" >&2; exit 1; }
    final="$work/final-$attempt"
    list_remote >"$final"
    if ! cmp -s "$after" "$final"; then
        attempt=$((attempt + 1)); continue
    fi
    mkdir -p "$outbox/receipts"
    for event in "$outbox/events"/*.json; do
        [ -f "$event" ] || continue
        name=$(basename "$event")
        receipt="$outbox/receipts/$name"
        [ -e "$receipt" ] && continue
        tmp="$outbox/receipts/.${name}.tmp.$$"
        printf '{"event_id":"sha256:%s","remote_path":"ratings/events/%s"}\n' "${name%.json}" "$name" >"$tmp"
        mv "$tmp" "$receipt"
    done
    exit 0
done

echo "rating sync: remote event set changed during materialization after $retries attempts" >&2
exit 1
