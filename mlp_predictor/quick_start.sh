#!/bin/bash
# ============================================================================
# OpenVLA 性能预测 MLP Pipeline - 快速启动脚本
# ============================================================================
# 
# 用法：
#   bash quick_start.sh              # 使用默认参数
#   bash quick_start.sh 50 32 1e-3   # 自定义 epochs、batch_size、learning_rate
#
# ============================================================================

# 配置
LIBERO_ENV="/home/zilai/anaconda3/envs/libero/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"  # 数据输入目录
MODEL_DIR="${SCRIPT_DIR}/models"    # 模型输出目录

# 参数
EPOCHS=${1:-100}
BATCH_SIZE=${2:-16}
LR=${3:-1e-3}

echo "================================================================================"
echo " OpenVLA 性能预测 MLP Pipeline"
echo "================================================================================"
echo ""
echo "配置："
echo "  EPOCHS:      $EPOCHS"
echo "  BATCH_SIZE:  $BATCH_SIZE"
echo "  LEARNING_RATE: $LR"
echo "  DATA_DIR:    $OUTPUT_DIR"
echo "  MODEL_DIR:   $MODEL_DIR"
echo ""

# 检查输入数据
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ 错误：数据目录不存在 -> $OUTPUT_DIR"
    exit 1
fi

JSON_COUNT=$(find "$OUTPUT_DIR" -name "*.json" -type f | wc -l)
if [ "$JSON_COUNT" -lt 5 ]; then
    echo "⚠️  警告：数据文件数目较少 ($JSON_COUNT 个)"
    echo "   建议至少 5+ 个 JSON 文件以获得可靠的预测"
fi

echo "找到 $JSON_COUNT 个 JSON 文件"
echo ""

# 清理旧输出
if [ -d "$MODEL_DIR" ]; then
    echo "清理旧结果..."
    rm -rf "$MODEL_DIR"
fi

# 运行 Pipeline
echo "启动 Pipeline..."
echo "================================================================================"
echo ""

cd "$SCRIPT_DIR"

$LIBERO_ENV pipeline.py \
    --data-dir "$OUTPUT_DIR" \
    --output-dir "$MODEL_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline 执行成功！"
    echo ""
    echo "输出文件："
    echo "  模型:       $MODEL_DIR/perf_mlp.pt"
    echo "  评估结果:   $MODEL_DIR/evaluation_results.json"
    echo "  可视化:"
    echo "    - $MODEL_DIR/pred_vs_gt.png"
    echo "    - $MODEL_DIR/error_distribution.png"
    echo "    - $MODEL_DIR/latency_memory_tradeoff.png"
    echo "    - $MODEL_DIR/training_history.png"
    echo ""
    echo "查看结果："
    echo "  cat $MODEL_DIR/evaluation_results.json"
    echo "  ls -lh $MODEL_DIR/"
else
    echo "❌ Pipeline 执行失败！(exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi

echo "================================================================================"
