## 🚀 时间规划（汇报前）

---

### 📅 Day 1–2（03/24 → 03/25）

#### 🔵 跑通流程 & 数据采集

- 跑 baseline（FP16）
- 跑 INT8 / INT4
- 固定 sample set（20–50条）

- 每条实验记录：
  - `latency`（mean / p95）
  - `memory`（peak）
  - `success_rate`（可选）

👉 **目标：**
- ≥ 1000 条有效数据

---

### 📅 Day 3–4

#### 🔵 数据处理 & 特征工程

- JSONL → DataFrame
- 解析 `skip_modules` → 结构化特征：
  - `num_quantized_layers`
  - `skip_vision`
  - `skip_projector`
  - `skip_action_head`

- 构建训练数据集

---

### 📅 Day 4–5

#### 🔵 训练 MLP 性能预测模型

- 预测目标：
  - `latency`
  - `memory`

- 划分 train / test
- 使用 **relative error loss**

👉 **目标：**
- latency error < **20%**
- memory error < **15%**

---

### 📅 Day 5–6（汇报前）

#### 🔵 可视化（必须完成 🔥）

- 量化方式 vs latency（柱状图）
- 量化方式 vs memory（柱状图）
- prediction vs ground truth（散点图）
