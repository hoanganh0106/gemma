#!/usr/bin/env bash
set -u

export PATH=/home/hoanganh/.cargo/bin:$PATH
repo=/mnt/d/Project/furiosa-opt-gemma4-12B-main
cd "$repo" || exit 1

for n in $(seq 2 20); do
    out=$(moa-submitter submit --source "$repo" 2>&1)
    id=$(printf '%s\n' "$out" | sed -n 's/^Submission:[[:space:]]*//p' | head -1)
    echo "SUBMISSION $n ID $id"
    if [ -z "$id" ]; then
        printf '%s\n' "$out"
        continue
    fi

    while :; do
        sleep 300
        st=$(moa-submitter status "$id" 2>&1)
        printf '%s\n' "$st" | grep -E '^(Submission:|Status:|Stage:|Finished:|Cycles:|  ops::|Score:|Error:|Failure:)' || true
        status=$(printf '%s\n' "$st" | sed -n 's/^Status:[[:space:]]*//p' | head -1)
        case "$status" in
            completed|failed|rejected|cancelled)
                moa-submitter log "$id" 2>&1 | tail -35
                break
                ;;
        esac
    done
done
