#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/mnt/fakeyou/openpi_LIP"
TOTAL_RUNS=5
START_BATCH_ID=0

usage() {
  echo "Usage: $0 [--runs N] [--start-batch-id N] [-- python_args_for_main_verify_chunk]"
  echo
  echo "Examples:"
  echo "  $0 --runs 20"
  echo "  $0 --runs 5 -- --args.task-id 2 --args.episode-idx 1 --args.rtc True"
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      TOTAL_RUNS="$2"
      shift 2
      ;;
    --start-batch-id)
      START_BATCH_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$TOTAL_RUNS" =~ ^[0-9]+$ ]] || [[ "$TOTAL_RUNS" -le 0 ]]; then
  echo "Error: --runs must be a positive integer. Got: $TOTAL_RUNS" >&2
  exit 1
fi

if ! [[ "$START_BATCH_ID" =~ ^[0-9]+$ ]]; then
  echo "Error: --start-batch-id must be a non-negative integer. Got: $START_BATCH_ID" >&2
  exit 1
fi

cd "$ROOT_DIR"

source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"

echo "Running main_verify_chunk.py for $TOTAL_RUNS times (start batch_id=$START_BATCH_ID)"

for ((run_idx=0; run_idx<TOTAL_RUNS; run_idx++)); do
  batch_id=$((START_BATCH_ID + run_idx))
  echo "[$((run_idx + 1))/$TOTAL_RUNS] batch_id=$batch_id"
  python examples/libero/main_verify_chunk.py --args.batch-id "$batch_id" "${EXTRA_ARGS[@]}"
done

echo "All runs completed."
