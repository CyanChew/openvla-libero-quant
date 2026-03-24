"""
可视化模块
"""
import json
import numpy as np
from typing import Dict
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def plot_predictions_vs_ground_truth(
    pred_lat: np.ndarray,
    gt_lat: np.ndarray,
    pred_mem: np.ndarray,
    gt_mem: np.ndarray,
    output_dir: str = 'models'
):
    """
    绘制预测值 vs 真实值散点图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Latency
    ax = axes[0]
    ax.scatter(gt_lat, pred_lat, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # 理想线
    min_val, max_val = min(gt_lat.min(), pred_lat.min()), max(gt_lat.max(), pred_lat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Ground Truth Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Prediction', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 计算 R² 和 MAPE
    lat_mape = np.mean(np.abs(pred_lat - gt_lat) / gt_lat) * 100
    lat_r2 = 1 - np.sum((pred_lat - gt_lat) ** 2) / np.sum((gt_lat - gt_lat.mean()) ** 2)
    ax.text(0.05, 0.95, f'MAPE: {lat_mape:.2f}%\nR²: {lat_r2:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Memory
    ax = axes[1]
    ax.scatter(gt_mem, pred_mem, alpha=0.6, s=50, color='green', edgecolors='k', linewidth=0.5)
    
    min_val, max_val = min(gt_mem.min(), pred_mem.min()), max(gt_mem.max(), pred_mem.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Ground Truth Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Prediction', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    mem_mape = np.mean(np.abs(pred_mem - gt_mem) / gt_mem) * 100
    mem_r2 = 1 - np.sum((pred_mem - gt_mem) ** 2) / np.sum((gt_mem - gt_mem.mean()) ** 2)
    ax.text(0.05, 0.95, f'MAPE: {mem_mape:.2f}%\nR²: {mem_r2:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'pred_vs_gt.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_distribution(
    pred_lat: np.ndarray,
    gt_lat: np.ndarray,
    pred_mem: np.ndarray,
    gt_mem: np.ndarray,
    output_dir: str = 'models'
):
    """
    绘制预测误差分布
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Latency 绝对误差
    lat_abs_err = np.abs(pred_lat - gt_lat)
    ax = axes[0, 0]
    ax.hist(lat_abs_err, bins=20, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(np.mean(lat_abs_err), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lat_abs_err):.2f}')
    ax.set_xlabel('Absolute Error (ms)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Latency Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Latency 相对误差
    lat_rel_err = np.abs(pred_lat - gt_lat) / gt_lat * 100
    ax = axes[0, 1]
    ax.hist(lat_rel_err, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(lat_rel_err), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lat_rel_err):.2f}%')
    ax.set_xlabel('Relative Error (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Latency Relative Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Memory 绝对误差
    mem_abs_err = np.abs(pred_mem - gt_mem)
    ax = axes[1, 0]
    ax.hist(mem_abs_err, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(mem_abs_err), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mem_abs_err):.2f}')
    ax.set_xlabel('Absolute Error (MB)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Memory Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Memory 相对误差
    mem_rel_err = np.abs(pred_mem - gt_mem) / gt_mem * 100
    ax = axes[1, 1]
    ax.hist(mem_rel_err, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(mem_rel_err), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mem_rel_err):.2f}%')
    ax.set_xlabel('Relative Error (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Memory Relative Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'error_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_latency_vs_memory_tradeoff(
    gt_lat: np.ndarray,
    gt_mem: np.ndarray,
    output_dir: str = 'models'
):
    """
    绘制 latency vs memory 的 tradeoff 关系
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scatter = ax.scatter(gt_lat, gt_mem, c=range(len(gt_lat)), cmap='viridis', 
                        s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Latency vs Memory Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sample Index', fontsize=11)
    
    # 计算相关系数
    correlation = np.corrcoef(gt_lat, gt_mem)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'latency_memory_tradeoff.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_training_history(history: Dict, output_dir: str = 'models'):
    """
    绘制训练历史
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲线
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE 曲线
    ax = axes[1]
    ax.plot(history['val_lat_mae'], label='Latency MAE', linewidth=2)
    ax.plot(history['val_mem_mae'], label='Memory MAE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE (log scale)', fontsize=11)
    ax.set_title('Validation MAE', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """主绘图流程"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-results', type=str, default='models/evaluation_results.json',
                        help='评估结果路径')
    parser.add_argument('--training-history', type=str, default='models/training_history.json',
                        help='训练历史路径')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='输出目录')
    args = parser.parse_args()
    
    # 加载评估结果
    with open(args.eval_results, 'r') as f:
        eval_data = json.load(f)
    
    pred_lat = np.array(eval_data['predictions']['predicted_latencies_ms'])
    gt_lat = np.array(eval_data['predictions']['ground_truth_latencies_ms'])
    pred_mem = np.array(eval_data['predictions']['predicted_memories_mb'])
    gt_mem = np.array(eval_data['predictions']['ground_truth_memories_mb'])
    
    # 生成图表
    print("Generating plots...")
    plot_predictions_vs_ground_truth(pred_lat, gt_lat, pred_mem, gt_mem, args.output_dir)
    plot_error_distribution(pred_lat, gt_lat, pred_mem, gt_mem, args.output_dir)
    plot_latency_vs_memory_tradeoff(gt_lat, gt_mem, args.output_dir)
    
    # 加载并绘制训练历史
    if Path(args.training_history).exists():
        with open(args.training_history, 'r') as f:
            history = json.load(f)
        plot_training_history(history, args.output_dir)
    
    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
