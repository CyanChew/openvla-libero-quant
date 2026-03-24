# OpenVLA 量化性能预测 MLP Pipeline

完整的机器学习 pipeline，用于预测 OpenVLA 模型在不同量化配置下的性能指标。

## 项目结构

```
mlp_predictor/
├── dataset.py           # 数据加载和清洗
├── features.py          # 特征工程（核心）
├── model.py             # MLP 网络定义
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── plot.py              # 可视化脚本
├── pipeline.py          # 完整 pipeline 入口
├── README.md            # 本文件
└── models/              # 输出目录（自动创建）
    ├── perf_mlp.pt                   # 训练好的模型
    ├── training_history.json         # 训练历史
    ├── evaluation_results.json       # 评估结果
    ├── feature_info.json             # 特征信息
    ├── data_stats.json               # 数据统计
    ├── pred_vs_gt.png                # 预测 vs 真实值
    ├── error_distribution.png        # 误差分布
    ├── latency_memory_tradeoff.png   # latency-memory 权衡
    └── training_history.png          # 训练曲线
```

## 快速开始

### 1. 前置条件

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn
```

### 2. 运行完整 Pipeline

```bash
cd mlp_predictor
python pipeline.py --data-dir ../outputs --output-dir models --epochs 100 --batch-size 32
```

### 可选参数

- `--data-dir`: 输入数据目录（包含 JSON 文件）
- `--output-dir`: 输出目录
- `--epochs`: 训练轮数（默认 100）
- `--batch-size`: 批大小（默认 32）
- `--lr`: 学习率（默认 1e-3）
- `--verbose`: 显示详细信息

### 3. 分步运行

如果需要分步执行，可以单独调用各个模块：

```bash
# 只训练
python train.py --data-dir ../outputs --output-dir models --epochs 100

# 只评估
python evaluate.py --data-dir ../outputs --model-path models/perf_mlp.pt

# 只绘图
python plot.py --eval-results models/evaluation_results.json --output-dir models
```

## 特征工程详解

### 提取的特征（共 38 个）

#### A. 量化基础特征
- `weight_bits`: 量化位宽 (4 或 8)
- `load_in_4bit`: 是否使用 4-bit 量化 (0/1)
- `load_in_8bit`: 是否使用 8-bit 量化 (0/1)
- `llm_quant_ratio`: 量化比例 (0.0-1.0)

#### B. One-Hot 编码分类特征

**quant_method** (2 个特征)：
- `quant_method_bnb_int4`
- `quant_method_bnb_int8`

**compute_dtype** (2 个特征)：
- `compute_dtype_bf16`
- `compute_dtype_fp32`

**llm_quant_target** (4 个特征)：
- `llm_target_ffn_only`
- `llm_target_all`
- `llm_target_attn_only`
- `llm_target_none`

**llm_layer_selection** (3 个特征)：
- `llm_selection_prefix`
- `llm_selection_suffix`
- `llm_selection_uniform`

**vision_quant_target** (4 个特征)：
- `vision_target_none`
- `vision_target_projector_only`
- `vision_target_tower_only`
- `vision_target_all`

**action_quant_target** (2 个特征)：
- `action_target_none`
- `action_target_all`

#### C. 输入特征
- `num_images`: 输入图像数量
- `batch_size`: 批大小
- `image_height`: 图像高度
- `image_width`: 图像宽度

#### D. 硬件特征
- `gpu_mem_gb`: GPU 显存大小 (GB)

#### E. skip_modules 解析特征
- `num_skip_modules`: 跳过的模块总数
- `num_skip_llm_layers`: 跳过的 LLM 层数
- `num_skip_self_attn`: 跳过的 self_attn 模块数
- `skip_vision_tower`: 是否跳过 vision_tower (0/1)
- `skip_projector`: 是否跳过 projector (0/1)
- `skip_action_head`: 是否跳过 action_head (0/1)

#### F. 推导特征
- `num_quantized_llm_layers`: 量化的 LLM 层数（32 - skip）
- `llm_quantized_ratio`: 实际量化比例（quantized_layers / 32）

### skip_modules 解析算法

从 `skip_modules` 列表解析特征：

```
"language_model.model.layers.0.self_attn" → 层 0，self_attn skip
"language_model.model.layers.10" → 层 10 完全 skip
"vision_tower" → 标记 skip_vision_tower=1
```

**假设** LLM 总层数 = 32 (OpenVLA-7B)

## 数据格式要求

输入 JSON 文件需包含以下结构：

```json
{
  "quant_config": {
    "weight_bits": 4,
    "load_in_4bit": true,
    "load_in_8bit": false,
    "llm_quant_ratio": 0.1,
    "llm_quant_target": "ffn_only",
    "llm_layer_selection": "prefix",
    "vision_quant_target": "projector_only",
    "action_quant_target": "all",
    "skip_modules": ["language_model.model.layers.0", ...]
  },
  "input_config": {
    "num_images": 1,
    "batch_size": 1,
    "image_resolution": "224x224"
  },
  "hardware": {
    "gpu_mem_gb": 47.4
  },
  "metrics": {
    "mean_latency_ms": 206.41,
    "peak_memory_mb": 12976.48
  },
  "status": "success"
}
```

## 模型架构

### PerformanceMLP

```
Input (D features)
    ↓
Linear(D → 64) + BatchNorm + ReLU + Dropout
    ↓
Linear(64 → 64) + BatchNorm + ReLU + Dropout
    ↓
Linear(64 → 2)
    ↓
Output (latency, memory)
```

**设计特点**：
- 使用 BatchNorm 加速收敛
- Dropout 防止过拟合
- Xavier 权重初始化
- 目标值使用 log 变换提升稳定性

## 评估指标

### 预测精度指标

| 指标 | 说明 |
|------|------|
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| MAPE | 平均绝对百分比误差（%） |
| Median APE | 中位数绝对百分比误差 |
| P95 APE | 95 分位绝对百分比误差 |

### 性能目标

```
✓ Latency MAPE < 20%
✓ Memory MAPE < 15%
```

## 输出文件说明

### 模型文件

- **perf_mlp.pt**: 训练好的模型权重（PyTorch）

### 数据文件

- **evaluation_results.json**: 测试集上的预测结果和指标
- **training_history.json**: 每个 epoch 的 loss 和验证指标
- **feature_info.json**: 特征名称、索引和类型
- **data_stats.json**: 数据集统计信息

### 可视化图表

- **pred_vs_gt.png**: 预测值 vs 真实值散点图（含 R² 和 MAPE）
- **error_distribution.png**: 绝对误差和相对误差的直方图
- **latency_memory_tradeoff.png**: latency-memory 相关性分析
- **training_history.png**: 训练过程中的 loss 和 MAE 曲线

## 典型工作流

### 1. 初始运行（小数据集测试）

```bash
python pipeline.py \
    --data-dir ../outputs \
    --output-dir models_v1 \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3
```

### 2. 查看结果

```bash
# 查看评估报告
cat models_v1/evaluation_results.json

# 查看特征列表
cat models_v1/feature_info.json

# 查看训练曲线
open models_v1/training_history.png
```

### 3. 调优（如果指标不满足）

- 增加 `--epochs`（更多训练轮数）
- 调整 `--lr`（学习率）
- 增加数据（更多 JSON 文件）
- 修改 `features.py` 中的特征定义

## 常见问题

### Q: 如何处理缺失值？
A: 所有可选字段默认填 0，确保 feature matrix 完整。

### Q: 是否需要 GPU？
A: 不是必需，但推荐使用（CUDA）加快训练。

### Q: 如何改变目标值预处理？
A: 编辑 `train.py` 中的 `prepare_data()` 函数，修改 log 变换代码。

### Q: 如何添加新特征？
A: 在 `features.py` 的 `extract_features()` 函数中添加新的特征提取逻辑。

## 依赖项

```
numpy        >= 1.20
torch        >= 1.10
scikit-learn >= 0.24
matplotlib   >= 3.4
seaborn      >= 0.11
```

## 联系

如有问题或建议，欢迎反馈。

---

**Version**: 1.0.0  
**Last Updated**: 2026-03-24
