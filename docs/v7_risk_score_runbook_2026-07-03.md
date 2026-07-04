# V7-risk-score 运行手册（2026-07-03）

V7 是在 V6 conditional gate 之后新增的连续风险分数版本。它仍然是不训练、不微调模型的 inference-time decoding 方法，只改变 speculative decoding 的 relaxed tolerance multiplier。

## 运行前检查

必须在容器内运行：

```bash
docker start scd-vllm-openai-v0.21.0-lunwen
docker exec -it scd-vllm-openai-v0.21.0-lunwen bash
cd /home/scd/MG-SD
```

确认模型路径：

```bash
ls -lah /data/weight/Qwen3-32B /data/weight/Qwen3-8B
```

确认 sampler 已包含 V7 hooks：

```bash
grep -nE 'VLLM_MGSD_V7_ENABLED|VLLM_MGSD_V7_LAMBDA|VLLM_MGSD_V7_MIN_GATE' \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py
```

如果没有输出，需要先在已打 V6 patch 的基础上继续打：

```bash
patch -p1 -d /usr/local/lib/python3.12/dist-packages \
  < /home/scd/MG-SD/patch/versions/v7/0007-MG-SD-v7-risk-score-gate.patch
```

## 20 条 smoke

```bash
PRESET=v7-risk-score \
CUDA_VISIBLE_DEVICES_LIST=0,1,2,3 \
TARGET_MODEL=/data/weight/Qwen3-32B \
DRAFT_MODEL=/data/weight/Qwen3-8B \
OUTPUT_DIR=/home/scd/MG-SD/entity_eval_v6_rebuild/v7_riskscore_pilot20_g0123 \
bash scripts/run_v6_entity_eval.sh
```

## 对比 baseline / V5 / V6 / V7

```bash
python3 scripts/compare_entity_eval_runs.py \
  entity_eval_v6_rebuild/baseline_pilot20_timing_weight_g0123/baseline_summary.json \
  entity_eval_v6_rebuild/v5_pilot20_timing_weight_g0123/v5_summary.json \
  entity_eval_v6_rebuild/v6_default_pilot20_timing_weight_g0123/v6-default_summary.json \
  entity_eval_v6_rebuild/v6_strict_freq0_med05_num02_unit02_rho095_floor085_pilot20_g0123/v6-default_summary.json \
  entity_eval_v6_rebuild/v7_riskscore_pilot20_g0123/v7-risk-score_summary.json \
  --csv-output entity_eval_v6_rebuild/pilot20_method_comparison.csv
```

## 分析 gate debug

```bash
python3 scripts/analyze_gate_debug.py \
  entity_eval_v6_rebuild/v7_riskscore_pilot20_g0123/gate_debug.jsonl \
  --json-output entity_eval_v6_rebuild/v7_riskscore_pilot20_g0123/gate_debug_summary.json
```

## 结果判断

- 20 条只看链路和初步趋势，不能作为论文主结果。
- 如果 `CEER <= baseline + 0.02`，优先选 tokens/s 最高的方法。
- 如果没有方法满足，则选 CEER 最低且 tokens/s 不低于 baseline 的方法。
- 如果 V7 不优于 V6-strict，就把 V6-strict 作为主方法，V7 作为探索性版本或附录结果。
