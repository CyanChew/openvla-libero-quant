Day 1–2（3.24 → 3.25）
🔵 跑通 &采数据
 跑 baseline（FP16）
 跑 int8 / int4
 固定 sample set（20–50条）
 每条记录：
latency（mean/p95）
memory（peak）
success_rate （optioinal)

👉 目标：
≥ 1000 条有效数据

✅ Day 3–4
🔵 数据处理 + feature
 JSONL → DataFrame
 skip_modules → summary
num_quantized_layers
skip_vision / projector / action
 构建训练表
✅ Day 4–5
🔵 训练 MLP
 预测：
latency
memory
 train/test split
 relative error loss

👉 目标：

latency error < 20%
memory error < 15%
✅ Day 5–6（汇报前）
🔵 画 3 张图（必须有🔥）
 quant vs latency（柱状图）
 quant vs memory（柱状图）
 prediction vs GT（散点图）
