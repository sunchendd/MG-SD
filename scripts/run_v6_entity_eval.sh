#!/usr/bin/env bash
set -euo pipefail

: "${PRESET:?Set PRESET to v5 or v6-default}"

PROJECT_DIR="${PROJECT_DIR:-/home/scd/MG-SD}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_DIR}/datasets/v6_pilot20_seed123.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/entity_eval_v6_rebuild/${PRESET}_pilot20}"
CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST:-0,1,4,5}"
SAMPLE_COUNT="${SAMPLE_COUNT:-20}"
TARGET_MODEL="${TARGET_MODEL:-/data/models/Qwen3-32B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/models/Qwen3-8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
BASE_TOLERANCE="${BASE_TOLERANCE:-0.2}"

cd "${PROJECT_DIR}"

echo "[1/4] Run local logic tests"
python3 -m unittest \
  test_run_entity_eval_v6.py \
  test_v6_eval_runtime.py \
  test_entity_eval.py -v

echo "[2/4] Verify fixed dataset and current GPU usage"
wc -l "${DATASET_PATH}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo "[3/4] Verify sampler exposes V6 hooks"
python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm.v1.sample.rejection_sampler")
if not spec or not spec.origin:
    raise SystemExit("vLLM rejection sampler not found")
text = Path(spec.origin).read_text(encoding="utf-8", errors="replace")
required = (
    "VLLM_MGSD_V6_ENABLED",
    "VLLM_MGSD_V6_SAFE_FLOOR",
    "VLLM_MGSD_V6_RHO_SAFE",
    "VLLM_MGSD_V6_RHO_RISK",
)
missing = [name for name in required if name not in text]
print("sampler =", spec.origin)
if missing:
    raise SystemExit(f"sampler missing V6 hooks: {missing}")
PY

echo "[4/4] Start one reproducible preset: ${PRESET}"
python3 run_entity_eval_v6.py \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --preset "${PRESET}" \
  --target-model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_LIST}" \
  --base-tolerance "${BASE_TOLERANCE}" \
  --sample-count "${SAMPLE_COUNT}"

echo "Completed preset=${PRESET}; output=${OUTPUT_DIR}"
