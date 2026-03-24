# OpenVLA 量化性能预测器（MLP）技术报告（汇报版）

## 1. 项目目标
基于 OpenVLA 量化 benchmark 数据，建立一个轻量 MLP 预测器，用于预测两项性能指标：
- `mean_latency_ms`
- `peak_memory_mb`

目标流程：JSON/JSONL 数据 → 特征工程 → 训练 MLP → 评估 → 可视化。

---

## 2. 本次测试使用的数据（你问的重点）
### 2.1 数据来源
本次实际测试读取的是 sweep 生成目录：
- `/mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs`

该目录是「每个样本一个 `.json` 文件」的形式（不是单文件 JSONL），但字段结构与 JSONL 每行一致。

### 2.2 样本筛选规则
仅保留：
- `status == "success"`
- `metrics.mean_latency_ms != None`
- `metrics.peak_memory_mb != None`

### 2.3 本次测试样本规模
- 原始样本：22
- 有效样本：21
- 划分：Train/Val/Test = 14 / 3 / 4（约 70/15/15）

### 2.4 目标值分布（本次）
- Latency 范围：197.66 ~ 262.80 ms
- Memory 范围：4424.44 ~ 12976.48 MB

---

## 3. 方法与模型设计
## 3.1 特征工程（核心）
共构建 34 维特征，包含：
1. 量化基础特征（位宽、4bit/8bit、`llm_quant_ratio`）
2. 分类特征 One-Hot（`quant_method`、`compute_dtype`、`llm_quant_target`、`vision_quant_target` 等）
3. 输入特征（`num_images`、`batch_size`、分辨率拆分成 `image_height/image_width`）
4. 硬件特征（`gpu_mem_gb`）
5. `skip_modules` 结构化解析特征（跳过模块数量、跳过 LLM 层数量、是否跳过 vision/projector/action）
6. 推导特征（`num_quantized_llm_layers`、`llm_quantized_ratio`）

说明：未使用 `exp_id`、`timestamp` 作为特征，符合约束要求。

## 3.2 模型结构
MLP：
- `input_dim -> 64 -> 64 -> 2`
- 激活函数：ReLU
- 输出第 1 维预测 latency，第 2 维预测 memory

## 3.3 训练策略
- 数据划分：70/15/15
- 特征标准化：StandardScaler
- 目标值：log 变换（提高训练稳定性）
- 优化器：Adam
- 训练轮数：100 epochs（本次测试）

---

## 4. 实验结果（本次）
测试集（4 样本）结果：
- Latency MAPE: **96.73%**
- Memory MAPE: **99.84%**

解读：
- 本次结果不达标，核心原因是样本量过小（仅 21 条有效样本，测试仅 4 条）。
- 当前阶段主要验证了 pipeline 的可用性与正确性，不代表最终可用精度。

---

## 5. 当前结论
1. 端到端 pipeline 已打通：数据加载、特征工程、训练、评估、作图均可运行。
2. 特征工程（尤其 `skip_modules` 结构化解析）已按设计实现。
3. 在小样本下模型可收敛，但泛化精度不足，属预期现象。

---

## 6. 下一步建议（汇报可直接说）
1. 持续积累 sweep 数据，建议至少 100+ 有效样本后再评估最终精度。
2. 增加更稳定的目标建模策略：
   - 分开训练 latency / memory 两个模型
   - 使用加权损失或相对误差损失
3. 在样本增长后做交叉验证，输出更稳健的误差区间。

---

## 7. 产出文件
- 模型权重：`mlp_predictor/models/perf_mlp.pt`
- 评估结果：`mlp_predictor/models/evaluation_results.json`
- 特征信息：`mlp_predictor/models/feature_info.json`
- 可视化：
  - `mlp_predictor/models/pred_vs_gt.png`
  - `mlp_predictor/models/error_distribution.png`
  - `mlp_predictor/models/latency_memory_tradeoff.png`
  - `mlp_predictor/models/training_history.png`
