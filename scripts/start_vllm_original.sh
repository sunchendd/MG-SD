#!/usr/bin/env bash
set -euo pipefail

mode="${1:-mgsd}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

case "$mode" in
  baseline)
    unset VLLM_EARS_BASE_TOLERANCE VLLM_MGSD_ENABLED VLLM_MGSD_MARGIN_DELTA
    ;;
  ears)
    export VLLM_EARS_BASE_TOLERANCE=0.1
    unset VLLM_MGSD_ENABLED VLLM_MGSD_MARGIN_DELTA
    ;;
  mgsd)
    export VLLM_EARS_BASE_TOLERANCE=0.1
    export VLLM_MGSD_ENABLED=1
    export VLLM_MGSD_MARGIN_DELTA=0.10
    ;;
  *)
    echo "Usage: $0 [baseline|ears|mgsd]" >&2
    exit 1
    ;;
esac

if [[ -x /home/scd/ai-infra-tools/tools/kill_vllm.sh ]]; then
  /home/scd/ai-infra-tools/tools/kill_vllm.sh
fi

exec vllm serve /data/models/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --speculative-config '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}'
