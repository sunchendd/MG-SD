#!/usr/bin/env bash
set -euo pipefail

cd /home/scd/MG-SD

mode="${1:-pilot}"

case "$mode" in
  pilot)
    exec python3 run_entity_eval_pilot.py \
      --sample-count 12 \
      --max-tokens 256 \
      --temperature 0.9
    ;;
  large300)
    exec python3 run_entity_eval_pilot.py \
      --sample-count 300 \
      --max-tokens 256 \
      --temperature 0.9 \
      --logprobs 2 \
      --modes ears,mgsd-d0.10,mgsd-d0.05 \
      --output-dir /home/scd/MG-SD/entity_eval_large_300
    ;;
  *)
    echo "Usage: $0 [pilot|large300]" >&2
    exit 1
    ;;
esac
