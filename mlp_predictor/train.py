"""
MLP 模型训练脚本
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Dict, Any

from dataset import load_json_files_from_dir, filter_valid_samples, get_data_statistics
from features import build_feature_matrix
from model import PerformanceMLP, PerformanceMLPSeparate


def prepare_data(
    features_matrix: np.ndarray,
    latencies: np.ndarray,
    memories: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    划分数据集，使用 StandardScaler 标准化数值特征
    
    Args:
        features_matrix: [N, D] 特征矩阵
        latencies: [N] 延迟标签
        memories: [N] 内存标签
        test_size: 测试集比例
        val_size: 验证集比例
        random_seed: 随机种子
        
    Returns:
        dict: 包含 train/val/test 的数据和 scaler
    """
    # 第一次拆分：train+val vs test
    X_temp, X_test, y_lat_temp, y_lat_test, y_mem_temp, y_mem_test = train_test_split(
        features_matrix,
        latencies,
        memories,
        test_size=test_size,
        random_state=random_seed,
    )
    
    # 第二次拆分：train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_lat_train, y_lat_val, y_mem_train, y_mem_val = train_test_split(
        X_temp,
        y_lat_temp,
        y_mem_temp,
        test_size=val_size_adjusted,
        random_state=random_seed,
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples ({100*len(X_train)/len(features_matrix):.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({100*len(X_val)/len(features_matrix):.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({100*len(X_test)/len(features_matrix):.1f}%)")
    
    # 标准化特征（使用训练集的统计）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 目标值使用 log 变换提升稳定性
    y_lat_train_log = np.log(y_lat_train)
    y_lat_val_log = np.log(y_lat_val)
    y_lat_test_log = np.log(y_lat_test)
    
    y_mem_train_log = np.log(y_mem_train)
    y_mem_val_log = np.log(y_mem_val)
    y_mem_test_log = np.log(y_mem_test)
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_lat_train': y_lat_train_log,
        'y_lat_val': y_lat_val_log,
        'y_lat_test': y_lat_test_log,
        'y_mem_train': y_mem_train_log,
        'y_mem_val': y_mem_val_log,
        'y_mem_test': y_mem_test_log,
        'scaler': scaler,
        'original_latencies': y_lat_test,  # 用于评估
        'original_memories': y_mem_test,
    }


def relative_error_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    相对误差 loss: |pred - target| / |target|
    """
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-6))


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # 前向传播
        pred = model(X_batch)
        
        # 计算 loss（两个输出的 MSE）
        loss = nn.MSELoss()(pred, y_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    验证模型
    
    Returns:
        (latency_mae, memory_mae, total_loss)
    """
    model.eval()
    total_loss = 0.0
    lat_errors = []
    mem_errors = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(X_batch)
            loss = nn.MSELoss()(pred, y_batch)
            total_loss += loss.item()
            
            # 记录每个输出的误差
            lat_err = torch.abs(pred[:, 0] - y_batch[:, 0]).mean().item()
            mem_err = torch.abs(pred[:, 1] - y_batch[:, 1]).mean().item()
            lat_errors.append(lat_err)
            mem_errors.append(mem_err)
    
    return np.mean(lat_errors), np.mean(mem_errors), total_loss / len(val_loader)


def train_model(
    X_train: np.ndarray,
    y_lat_train: np.ndarray,
    y_mem_train: np.ndarray,
    X_val: np.ndarray,
    y_lat_val: np.ndarray,
    y_mem_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: torch.device = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    训练 MLP 模型
    
    Args:
        X_train, y_lat_train, y_mem_train: 训练数据
        X_val, y_lat_val, y_mem_val: 验证数据
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        device: 计算设备
        
    Returns:
        (model, training_history)
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nUsing device: {device}")
    
    # 准备数据
    y_train = np.column_stack([y_lat_train, y_mem_train]).astype(np.float32)
    y_val = np.column_stack([y_lat_val, y_mem_val]).astype(np.float32)
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_dim = X_train.shape[1]
    model = PerformanceMLP(input_dim=input_dim, hidden_dim=64, dropout=0.1)
    model = model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 训练循环
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_lat_mae': [],
        'val_mem_mae': [],
    }
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print(f"\nTraining MLP for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_lat_mae, val_mem_mae, val_loss = validate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_lat_mae'].append(val_lat_mae)
        history['val_mem_mae'].append(val_mem_mae)
        
        scheduler.step(val_loss)
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Lat MAE: {val_lat_mae:.4f} | "
                  f"Mem MAE: {val_mem_mae:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, history


def main():
    """主训练流程"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../outputs',
                        help='JSONL 或 JSON 文件目录')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {args.data_dir}...")
    data = load_json_files_from_dir(args.data_dir)
    
    # 清洗数据
    data = filter_valid_samples(data)
    
    if len(data) < 10:
        print(f"Error: 数据太少 ({len(data)} samples)")
        return
    
    # 打印统计信息
    stats = get_data_statistics(data)
    print("\nData statistics:")
    for key, val in stats.items():
        if isinstance(val, dict):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {val}")
    
    # 提取特征
    print(f"\nExtracting features from {len(data)} samples...")
    X, feature_indices, feature_names = build_feature_matrix(data)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature names ({len(feature_names)}): {feature_names}")
    
    # 提取目标值
    latencies = np.array([s['metrics']['mean_latency_ms'] for s in data], dtype=np.float32)
    memories = np.array([s['metrics']['peak_memory_mb'] for s in data], dtype=np.float32)
    
    print(f"Targets:")
    print(f"  Latency: {latencies.min():.2f} - {latencies.max():.2f} ms (mean: {latencies.mean():.2f})")
    print(f"  Memory:  {memories.min():.2f} - {memories.max():.2f} MB (mean: {memories.mean():.2f})")
    
    # 准备数据
    data_dict = prepare_data(X, latencies, memories, test_size=0.15, val_size=0.15)
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = train_model(
        data_dict['X_train'],
        data_dict['y_lat_train'],
        data_dict['y_mem_train'],
        data_dict['X_val'],
        data_dict['y_lat_val'],
        data_dict['y_mem_val'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )
    
    # 保存模型
    model_path = Path(args.output_dir) / 'perf_mlp.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # 保存训练历史
    history_path = Path(args.output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # 保存特征索引
    feature_info = {
        'feature_names': feature_names,
        'feature_indices': feature_indices,
        'num_features': len(feature_names),
    }
    feature_info_path = Path(args.output_dir) / 'feature_info.json'
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"Feature info saved to {feature_info_path}")
    
    # 保存数据统计
    data_stats = {
        'num_samples': len(data),
        'num_train': len(data_dict['X_train']),
        'num_val': len(data_dict['X_val']),
        'num_test': len(data_dict['X_test']),
    }
    stats_path = Path(args.output_dir) / 'data_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(data_stats, f, indent=2)
    
    print("\nTraining completed successfully!")
    return model, data_dict


if __name__ == '__main__':
    main()
