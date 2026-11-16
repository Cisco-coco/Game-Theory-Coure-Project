#!/usr/bin/env bash
# Run a serial, fixed-iteration width comparison using the active Python.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
OUT="${1:?Usage: bash run_width_study.sh OUTPUT [training options]}"
shift
for ARG in "$@"; do
    case "$ARG" in
        --hidden|--hidden=*|--output|--output=*|--resume|--resume=*)
            echo "The width study manages its own width/output and starts fresh: $ARG" >&2
            exit 1
            ;;
    esac
done
if [[ -e "$OUT" ]]; then
    echo "Output already exists: $OUT" >&2
    exit 1
fi
for WIDTH in 64 128; do
    "$PYTHON" train_deep_cfr.py --game leduc_poker --iterations 100 \
        --eval-every 10 --seeds 7 42 99 123 2026 --hidden "$WIDTH" \
        --traversals 400 --advantage-steps 500 --strategy-steps 1000 \
        --capacity 50000 --output "$OUT/width-$WIDTH" "$@"
done
