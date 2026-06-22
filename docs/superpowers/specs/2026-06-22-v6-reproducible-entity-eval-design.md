# V6 可复现实体评测设计

本设计建立独立于历史 `run_entity_eval_pilot.py` 的固定数据评测入口。它使用显式的模型、GPU、TP、tolerance、风险 token ID 和输出目录配置，并在每次运行中保存 Git commit、数据哈希、软件版本、sampler 路径、完整环境变量和启动命令。

本地单测不加载模型；服务器使用 Qwen3-32B target、Qwen3-8B draft、TP4 和固定20条受保护样本。首轮仅比较 V5 与重新固化的 `v6-default`，不冒充配置缺失的历史 `v6_b`/`v6_c`。MIMIC 派生数据永不进入 Git。
