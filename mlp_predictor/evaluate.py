"""
模型评估脚本
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict

from dataset import load_json_files_from_dir, filter_valid_samples
from features import build_feature_matrix
from model import PerformanceMLP
from train import prepare_data


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_lat_test_log: np.ndarray,
    y_mem_test_log: np.ndarray,
    original_latencies: np.ndarray,
    original_memories: np.ndarray,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_lat_test_log, y_mem_test_log: log 转换后的标签
        original_latencies, original_memories: 原始值（用于相对误差计算）
        device: 计算设备
        
    Returns:
        dict: 评估指标
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # 转换为 tensor
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_lat_test_tensor = torch.from_numpy(y_lat_test_log.astype(np.float32)).to(device)
    y_mem_test_tensor = torch.from_numpy(y_mem_test_log.astype(np.float32)).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)  # [N, 2]
        
        pred_lat_log = predictions[:, 0].cpu().numpy()
        pred_mem_log = predictions[:, 1].cpu().numpy()
    
    # 从 log 空间转换回原始空间
    pred_latencies = np.exp(pred_lat_log)
    pred_memories = np.exp(pred_mem_log)
    
    # 计算相对误差
    lat_relative_errors = np.abs(pred_latencies - original_latencies) / original_latencies
    mem_relative_errors = np.abs(pred_memories - original_memories) / original_memories
    
    # 计算各种统计
    metrics = {
        'num_test_samples': int(len(X_test)),
        
        # Latency 指标
        'latency_mae_ms': float(np.mean(np.abs(pred_latencies - original_latencies))),
        'latency_rmse_ms': float(np.sqrt(np.mean((pred_latencies - original_latencies) ** 2))),
        'latency_mape': float(np.mean(lat_relative_errors) * 100),  # Mean Absolute Percentage Error
        'latency_median_ape': float(np.median(lat_relative_errors) * 100),
        'latency_max_ape': float(np.max(lat_relative_errors) * 100),
        'latency_p95_ape': float(np.percentile(lat_relative_errors, 95) * 100),
        
        # Memory 指标
        'memory_mae_mb': float(np.mean(np.abs(pred_memories - original_memories))),
        'memory_rmse_mb': float(np.sqrt(np.mean((pred_memories - original_memories) ** 2))),
        'memory_mape': float(np.mean(mem_relative_errors) * 100),
        'memory_median_ape': float(np.median(mem_relative_errors) * 100),
        'memory_max_ape': float(np.max(mem_relative_errors) * 100),
        'memory_p95_ape': float(np.percentile(mem_relative_errors, 95) * 100),
    }
    
    return metrics, pred_latencies, pred_memories, original_latencies, original_memories


def print_evaluation_report(metrics: Dict[str, float]):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nTest Samples: {metrics['num_test_samples']}")
    
    print("\n--- Latency Prediction ---")
    print(f"  MAE:        {metrics['latency_mae_ms']:8.2f} ms")
    print(f"  RMSE:       {metrics['latency_rmse_ms']:8.2f} ms")
    print(f"  MAPE:       {metrics['latency_mape']:8.2f} %")
    print(f"  Median APE: {metrics['latency_median_ape']:8.2f} %")
    print(f"  Max APE:    {metrics['latency_max_ape']:8.2f} %")
    print(f"  P95 APE:    {metrics['latency_p95_ape']:8.2f} %")
    
    print("\n--- Memory Prediction ---")
    print(f"  MAE:        {metrics['memory_mae_mb']:8.2f} MB")
    print(f"  RMSE:       {metrics['memory_rmse_mb']:8.2f} MB")
    print(f"  MAPE:       {metrics['memory_mape']:8.2f} %")
    print(f"  Median APE: {metrics['memory_median_ape']:8.2f} %")
    print(f"  Max APE:    {metrics['memory_max_ape']:8.2f} %")
    print(f"  P95 APE:    {metrics['memory_p95_ape']:8.2f} %")
    
    print("\n" + "=" * 70)
    
    # 检查是否满足目标
    print("\n--- Performance Goals ---")
    lat_target = 20.0  # % MAPE
    mem_target = 15.0  # % MAPE
    
    lat_pass = metrics['latency_mape'] < lat_target
    mem_pass = metrics['memory_mape'] < mem_target
    
    print(f"  Latency MAPE < {lat_target}%: {'✓ PASS' if lat_pass else '✗ FAIL'} ({metrics['latency_mape']:.2f}%)")
    print(f"  Memory MAPE < {mem_target}%:  {'✓ PASS' if mem_pass else '✗ FAIL'} ({metrics['memory_mape']:.2f}%)")


def main():
    """主评估流程"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../outputs',
                        help='JSON 文件目录')
    parser.add_argument('--model-path', type=str, default='models/perf_mlp.pt',
                        help='模型路径')
    parser.add_argument('--feature-info', type=str, default='models/feature_info.json',
                        help='特征信息路径')
    args = parser.parse_args()
    
    # 加载数据
    print("Loading data...")
    data = load_json_files_from_dir(args.data_dir)
    data = filter_valid_samples(data)
    
    # 提取特征
    print("Extracting features...")
    X, feature_indices, feature_names = build_feature_matrix(data)
    latencies = np.array([s['metrics']['mean_latency_ms'] for s in data], dtype=np.float32)
    memories = np.array([s['metrics']['peak_memory_mb'] for s in data], dtype=np.float32)
    
    # 准备数据
    data_dict = prepare_data(X, latencies, memories)
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    input_dim = X.shape[1]
    model = PerformanceMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 评估
    print("Evaluating on test set...")
    metrics, pred_lat, pred_mem, gt_lat, gt_mem = evaluate_model(
        model,
        data_dict['X_test'],
        data_dict['y_lat_test'],
        data_dict['y_mem_test'],
        data_dict['original_latencies'],
        data_dict['original_memories'],
        device,
    )
    
    # 保存预测结果
    results = {
        'metrics': metrics,
        'predictions': {
            'predicted_latencies_ms': pred_lat.tolist(),
            'predicted_memories_mb': pred_mem.tolist(),
            'ground_truth_latencies_ms': gt_lat.tolist(),
            'ground_truth_memories_mb': gt_mem.tolist(),
        }
    }
    
    results_path = Path(args.model_path).parent / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # 打印报告
    print_evaluation_report(metrics)


if __name__ == '__main__':
    main()
