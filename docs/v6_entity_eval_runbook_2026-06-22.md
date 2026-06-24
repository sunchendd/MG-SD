# V6 实体评测运行手册（2026-06-22）

## 目的

本手册用于在固定20条 MIMIC-IV-Note 派生样本上，以统一的 Qwen3-32B target、Qwen3-8B draft、TP4 和 base tolerance 0.2 依次验证 V5 与重新固化的 `v6-default`。该实验不能冒充历史 `v6_b`/`v6_c`。

## 数据与代码边界

- 固定数据：`/home/scd/MG-SD/datasets/v6_pilot20_seed123.jsonl`
- 数据只保留在受控服务器，不进入 Git。
- 历史入口 `run_entity_eval_pilot.py` 不改动。
- 新入口：`run_entity_eval_v6.py`
- 每次运行只执行一个 preset，不自动重试，也不覆盖已有输出。

## 同步开发分支

在代码尚未合并到 `main` 时：

```bash
cd /home/scd/MG-SD
git fetch origin
git switch feature/v6-repro-eval
git pull --ff-only origin feature/v6-repro-eval
```

合并到 `main` 后：

```bash
cd /home/scd/MG-SD
git switch main
git pull --ff-only
```

## 启动前检查

必须在 Docker 容器内运行：

```bash
cd /home/scd/MG-SD
wc -l datasets/v6_pilot20_seed123.jsonl
nvidia-smi --query-gpu=index,name,memory.used --format=csv
```

只有计划使用的四张卡空闲时才启动。GPU 编号变化时，通过 `CUDA_VISIBLE_DEVICES_LIST` 显式覆盖。

## 先运行 V5 控制组

```bash
PRESET=v5 \
CUDA_VISIBLE_DEVICES_LIST=0,1,4,5 \
OUTPUT_DIR=/home/scd/MG-SD/entity_eval_v6_rebuild/v5_pilot20 \
bash scripts/run_v6_entity_eval.sh
```

完成后检查：

```bash
grep -m 4 "MG-SD enabled" \
  entity_eval_v6_rebuild/v5_pilot20/v5_server.log
cat entity_eval_v6_rebuild/v5_pilot20/v5_summary.json
```

日志应显示 `base_tolerance=0.200`、`draft_min_ratio=0.850` 和 `v6=False`。

## 再运行 V6-default

```bash
PRESET=v6-default \
CUDA_VISIBLE_DEVICES_LIST=0,1,4,5 \
OUTPUT_DIR=/home/scd/MG-SD/entity_eval_v6_rebuild/v6_default_pilot20 \
bash scripts/run_v6_entity_eval.sh
```

完成后检查：

```bash
grep -m 4 "MG-SD enabled" \
  entity_eval_v6_rebuild/v6_default_pilot20/v6-default_server.log
cat entity_eval_v6_rebuild/v6_default_pilot20/v6-default_summary.json
```

日志必须显示 `v6=True`；`run_manifest.json` 中五类风险 token ID 计数必须非零。

## 结果解释边界

20条 smoke 只回答链路是否正确、V6 是否真正启用、配置是否可回溯。它不能证明 V6 显著改善临床安全，也不能直接写成 AAAI 主结果。通过 smoke 后，再根据 V5/V6 的实体指标和 gate debug 设计 V6-safe、V6-balanced 与 V6-fast，并在固定100/300条上比较。
