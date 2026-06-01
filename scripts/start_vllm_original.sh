#!/usr/bin/env bash
set -euo pipefail

mode="${1:-mgsd}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

host="${HOST:-127.0.0.1}"
port="${PORT:-8000}"
targetModel="${TARGET_MODEL:-/data/models/Qwen3-32B}"
draftModel="${DRAFT_MODEL:-/data/models/Qwen3-0.6B}"
tensorParallelSize="${TP_SIZE:-2}"
maxModelLen="${MAX_MODEL_LEN:-4096}"
gpuMemoryUtilization="${GPU_MEMORY_UTILIZATION:-0.85}"
baseTolerance="${BASE_TOLERANCE:-0.1}"
mgsdMarginDelta="${MGSD_MARGIN_DELTA:-0.10}"
numSpeculativeTokens="${NUM_SPECULATIVE_TOKENS:-5}"
parallelDrafting="${PARALLEL_DRAFTING:-false}"

case "$mode" in
  baseline)
    unset VLLM_EARS_BASE_TOLERANCE VLLM_MGSD_ENABLED VLLM_MGSD_MARGIN_DELTA
    ;;
  ears)
    export VLLM_EARS_BASE_TOLERANCE="$baseTolerance"
    unset VLLM_MGSD_ENABLED VLLM_MGSD_MARGIN_DELTA
    ;;
  mgsd)
    export VLLM_EARS_BASE_TOLERANCE="$baseTolerance"
    export VLLM_MGSD_ENABLED=1
    export VLLM_MGSD_MARGIN_DELTA="$mgsdMarginDelta"
    ;;
  *)
    echo "Usage: $0 [baseline|ears|mgsd]" >&2
    exit 1
    ;;
esac

if [[ -x /home/scd/ai-infra-tools/tools/kill_vllm.sh ]]; then
  /home/scd/ai-infra-tools/tools/kill_vllm.sh
fi

exec vllm serve "$targetModel" \
  --host "$host" \
  --port "$port" \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size "$tensorParallelSize" \
  --max-model-len "$maxModelLen" \
  --gpu-memory-utilization "$gpuMemoryUtilization" \
  --trust-remote-code \
  --speculative-config "{\"model\":\"${draftModel}\",\"method\":\"draft_model\",\"num_speculative_tokens\":${numSpeculativeTokens},\"parallel_drafting\":${parallelDrafting}}"

