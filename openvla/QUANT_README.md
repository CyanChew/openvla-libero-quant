# OpenVLA 量化评测与数据采集指南

本文档说明了如何使用 `run_libero_eval.py` 在 LIBERO benchmark 上进行 OpenVLA 模型的各项指标（推理耗时、显存占用、任务成功率等）的采集与量化配置。

## 1. 基础环境
确保你已激活相应的环境（如 `libero` conda环境），并且已经安装了相关的依赖，尤其是用于模型量化的 `bitsandbytes`：
```bash
pip install bitsandbytes
```

## 2. 核心量化配置采集方式

我们在 `experiments/robot/libero/run_libero_eval.py` 中注入了量化（INT4/INT8）的支持和性能探测点。当你运行该脚本完成指定任务时，除原有的 WandB 与文本日志外，会在当前目录生成一个完整的 JSON 文件 `libero_quant_metrics.json`，其中包含所有冷热延时、峰值内存等数据。

> 💡 **提示**: 运行前记得指定用哪张卡以免冲突。比如你想在卡 3 上运行：`export CUDA_VISIBLE_DEVICES=3`

### 模式 A: 采集 FP16 (BFloat16) Baseline
不对模型进行任何低精度量化，默认使用 bfloat16。
```bash
PYTHONPATH=. python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --run_id_note "bf16_baseline"
```
**JSON 指标中呈现**：`quant_method` = "fp16", `weight_bits` = 16。

### 模式 B: 采集 LLM INT8 (8-bit) 数据
使用 bitsandbytes 对 OpenVLA 的 LLM 部分进行 8-bit 量化，视觉特征层和 Action输出头保持 BF16。
```bash
PYTHONPATH=. python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_8bit True \
  --run_id_note "int8_evaluation"
```
**JSON 指标中呈现**：`quant_method` = "bnb_int8", `weight_bits` = 8。

### 模式 C: 采集 LLM INT4 (4-bit) NF4 数据 (强烈推荐)
使用 bitsandbytes 对 OpenVLA 的 LLM 部分进行 4-bit 量化（底层使用 normal-float4），同样保留视觉特征层为高精度。
```bash
PYTHONPATH=. python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_4bit True \
  --run_id_note "int4_evaluation"
```
**JSON 指标中呈现**：`quant_method` = "bnb_int4", `weight_bits` = 4。

## 3. 面向“几百条配置”的模块级组织方案

你现在关心的是三个区域：
- `FFN / MLP`（主压缩来源）
- `Vision`（只量化一部分投影层，不做全量 aggressive）
- `Action`（有选择地量化）

我们已经把 `run_libero_eval.py` 扩展成可直接按模块分组配置：

```bash
PYTHONPATH=. python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --load_in_4bit True \
  --llm_quant_target ffn_only \
  --llm_quant_ratio 0.5 \
  --llm_layer_selection uniform \
  --vision_quant_target projector_only \
  --action_quant_target none \
  --run_id_note "int4_ffn50_visproj"
```

关键参数说明：
- `llm_quant_target`: `all | ffn_only | attn_only | none`
- `llm_quant_ratio`: 量化多少比例的 decoder layers（0~1）
- `llm_layer_selection`: `prefix | suffix | uniform`（量化哪一段层）
- `vision_quant_target`: `none | projector_only | tower_only | all`
- `action_quant_target`: `none | all`
- `quant_skip_modules`: 可选 CSV，手工覆盖自动策略（最高优先级）

> 推荐做法：先固定 `action_quant_target=none`，主扫 FFN；再在高价值 FFN 组合上加少量 `vision/projector` 与 `action` 组合。

## 4. 一键批量 Sweep（自动生成几百组实验）

新增脚本：`experiments/robot/libero/run_libero_quant_sweep.py`

示例（默认就是 200+ 组合）：

```bash
PYTHONPATH=. python experiments/robot/libero/run_libero_quant_sweep.py \
  --repo_root . \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 3 \
  --quant_modes int4,int8 \
  --llm_targets ffn_only,all \
  --llm_ratios 0.25,0.5,0.75,1.0 \
  --llm_layer_selections prefix,suffix,uniform \
  --vision_targets none,projector_only \
  --action_targets none,all \
  --seeds 7,17,27 \
  --output_dir experiments/robot/libero/sweep_results
```

默认组合规模：
- $2$(quant) × $2$(llm target) × $4$(ratio) × $3$(layer pick) × $2$(vision) × $2$(action) × $3$(seed) = **576** 组

批量结果：
- 每组一个 JSON：`sweep_results/*.json`
- 一个总索引：`sweep_results/manifest.jsonl`

## 5. 在代码中调整细粒度的量化参数（进阶）
我们对 `AutoModelForVision2Seq` 的装载机制进行了拦截。如果你需要调整具体的量化白名单或模块，你可以前往 `experiments/robot/openvla_utils.py` 的 `get_vla()` 函数：

```python
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", # 这里可以改成 "fp4"
            bnb_4bit_use_double_quant=True, # 嵌套量化，可调 False
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "action_head"] # 指定不量化的模块
        )
```
修改这里的配置可以精细化地测试哪一层对 action 指标的波动影响最大。

## 6. 输出解析 (`libero_quant_metrics.json`)
评测跑完（或者是你按下 `Ctrl+C` 提前中断）后生成的 metrics 文件结构如下：
* **quant_config**: 当前使用的量化手法、位数与过滤的模块名。
* **metrics/cold_latency_ms**: 记录在第一帧环境画面传给模型预测 action 时的耗时（包括了CUDA冷启动与显存加载等开销）。
* **metrics/{mean, p50, p95}_latency_ms**: 记录后续推流状态下生成 action 的统计学耗时。
* **metrics/peak_memory_mb**: 监控全测试周期内的最高显存分配阈值。
* **input_config/success_rate**: 在真实物理任务中的成功率表现，能最直观反映因量化引入导致的严重掉发性能现象。