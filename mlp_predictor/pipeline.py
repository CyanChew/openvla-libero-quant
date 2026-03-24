#!/usr/bin/env python3
"""
完整的性能预测 MLP Pipeline
执行流程：
1. 加载 JSONL 数据
2. 特征工程
3. 划分并标准化
4. 训练 MLP
5. 评估
6. 生成可视化
"""
import sys
import os
from pathlib import Path
import argparse

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_json_files_from_dir, filter_valid_samples, get_data_statistics
from features import build_feature_matrix, get_feature_names_and_types
from train import train_model, prepare_data
from evaluate import evaluate_model, print_evaluation_report
from plot import (
    plot_predictions_vs_ground_truth,
    plot_error_distribution,
    plot_latency_vs_memory_tradeoff,
    plot_training_history
)

import numpy as np
import json
import torch


def run_full_pipeline(
    data_dir: str = '../outputs',
    output_dir: str = 'models',
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    verbose: bool = True,
):
    """
    运行完整的 pipeline
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("性能预测 MLP Pipeline")
    print("="*80)
    
    # ========== 步骤 1: 加载数据 ==========
    print("\n[Step 1] Loading data...")
    data = load_json_files_from_dir(data_dir)
    
    if len(data) == 0:
        print(f"ERROR: No data found in {data_dir}")
        return False
    
    # ========== 步骤 2: 数据清洗 ==========
    print("\n[Step 2] Filtering valid samples...")
    data = filter_valid_samples(data)
    
    if len(data) < 10:
        print(f"ERROR: Not enough valid samples ({len(data)} < 10)")
        return False
    
    # ========== 步骤 3: 数据统计 ==========
    print("\n[Step 3] Data statistics...")
    stats = get_data_statistics(data)
    print("\nData Summary:")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Latency: {stats['latency']['min']:.2f} - {stats['latency']['max']:.2f} ms")
    print(f"           mean = {stats['latency']['mean']:.2f} ms")
    print(f"  Memory:  {stats['memory']['min']:.2f} - {stats['memory']['max']:.2f} MB")
    print(f"           mean = {stats['memory']['mean']:.2f} MB")
    if 'inference_memory' in stats:
        print(f"  Inference Memory: {stats['inference_memory']['min']:.2f} - {stats['inference_memory']['max']:.2f} MB")
        print(f"                    mean = {stats['inference_memory']['mean']:.2f} MB")
    
    # ========== 步骤 4: 特征工程 ==========
    print("\n[Step 4] Feature engineering...")
    X, feature_indices, feature_names = build_feature_matrix(data)
    print(f"\nExtracted {len(feature_names)} features from {len(data)} samples")
    print(f"Feature matrix shape: {X.shape}")
    
    if verbose:
        print(f"\nFeature names:")
        for i, name in enumerate(feature_names, 1):
            print(f"  {i:2d}. {name}")
    
    # 保存特征信息
    feature_info = {
        'timestamp': str(Path(data_dir).resolve()),
        'num_samples': len(data),
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'feature_indices': feature_indices,
        'feature_types': get_feature_names_and_types(),
    }
    feature_info_path = output_dir / 'feature_info.json'
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"Feature info saved to {feature_info_path}")
    
    # ========== 步骤 5: 提取目标值 ==========
    print("\n[Step 5] Extracting targets...")
    latencies = np.array([s['metrics']['mean_latency_ms'] for s in data], dtype=np.float32)
    memories = np.array([s['metrics']['peak_memory_mb'] for s in data], dtype=np.float32)
    
    print(f"Latency range:  {latencies.min():.2f} - {latencies.max():.2f} ms")
    print(f"Memory range:   {memories.min():.2f} - {memories.max():.2f} MB")
    
    # ========== 步骤 6: 数据准备 ==========
    print("\n[Step 6] Preparing data (train/val/test split)...")
    data_dict = prepare_data(X, latencies, memories, test_size=0.15, val_size=0.15)
    
    # ========== 步骤 7: 模型训练 ==========
    print("\n[Step 7] Training MLP model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = train_model(
        data_dict['X_train'],
        data_dict['y_lat_train'],
        data_dict['y_mem_train'],
        data_dict['X_val'],
        data_dict['y_lat_val'],
        data_dict['y_mem_val'],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    
    # 保存模型和训练历史
    model_path = output_dir / 'perf_mlp.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # ========== 步骤 8: 模型评估 ==========
    print("\n[Step 8] Evaluating model on test set...")
    metrics, pred_lat, pred_mem, gt_lat, gt_mem = evaluate_model(
        model,
        data_dict['X_test'],
        data_dict['y_lat_test'],
        data_dict['y_mem_test'],
        data_dict['original_latencies'],
        data_dict['original_memories'],
        device,
    )
    
    # 保存评估结果
    eval_results = {
        'metrics': metrics,
        'predictions': {
            'predicted_latencies_ms': pred_lat.astype(float).tolist(),
            'predicted_memories_mb': pred_mem.astype(float).tolist(),
            'ground_truth_latencies_ms': gt_lat.astype(float).tolist(),
            'ground_truth_memories_mb': gt_mem.astype(float).tolist(),
        }
    }
    
    eval_path = output_dir / 'evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {eval_path}")
    
    # 打印评估报告
    print_evaluation_report(metrics)
    
    # ========== 步骤 9: 生成可视化 ==========
    print("\n[Step 9] Generating visualizations...")
    plot_predictions_vs_ground_truth(pred_lat, gt_lat, pred_mem, gt_mem, str(output_dir))
    plot_error_distribution(pred_lat, gt_lat, pred_mem, gt_mem, str(output_dir))
    plot_latency_vs_memory_tradeoff(gt_lat, gt_mem, str(output_dir))
    plot_training_history(history, str(output_dir))
    
    # ========== 总结 ==========
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    print(f"\nOutput files:")
    for f in output_dir.glob("*"):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name:40s} ({size:,} bytes)")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance Prediction MLP Pipeline')
    parser.add_argument('--data-dir', type=str, default='../outputs',
                        help='JSONL or JSON data directory')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models and results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    success = run_full_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        verbose=args.verbose,
    )
    
    sys.exit(0 if success else 1)
