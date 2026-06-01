#!/usr/bin/env bash
set -euo pipefail

cd /home/scd/MG-SD

sampleCount="${SAMPLE_COUNT:-12}"
maxTokens="${MAX_TOKENS:-256}"
temperature="${TEMPERATURE:-0.9}"
modes="${MODES:-baseline,ears,mgsd-d0.10}"
outputDir="${OUTPUT_DIR:-/home/scd/MG-SD/entity_eval}"
logprobs="${LOGPROBS:-}"

args=(
  --sample-count "$sampleCount"
  --max-tokens "$maxTokens"
  --temperature "$temperature"
  --modes "$modes"
  --output-dir "$outputDir"
)

if [[ -n "$logprobs" ]]; then
  args+=(--logprobs "$logprobs")
fi

exec python3 run_entity_eval_pilot.py "${args[@]}"
