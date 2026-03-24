"""
MLP 模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerformanceMLP(nn.Module):
    """
    性能预测 MLP 网络
    
    结构:
    input_dim → 64 → ReLU → 64 → ReLU → 2
    
    输出:
    - output[:,0] = latency prediction
    - output[:,1] = memory prediction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度（默认 64）
            dropout: dropout 比率（默认 0.1）
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # 输出层：2 个输出（latency, memory）
        self.fc_out = nn.Linear(hidden_dim, 2)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """使用 Xavier 初始化"""
        for module in [self.fc1, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, input_dim]
            
        Returns:
            out: [batch_size, 2] - (latency, memory)
        """
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # 输出层
        out = self.fc_out(x)
        
        return out


class PerformanceMLPSeparate(nn.Module):
    """
    分离式 MLP - 为 latency 和 memory 各训练一个模型
    （可选的增强版本）
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        
        # Latency 预测器
        self.latency_fc1 = nn.Linear(input_dim, hidden_dim)
        self.latency_bn1 = nn.BatchNorm1d(hidden_dim)
        self.latency_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.latency_bn2 = nn.BatchNorm1d(hidden_dim)
        self.latency_out = nn.Linear(hidden_dim, 1)
        
        # Memory 预测器
        self.memory_fc1 = nn.Linear(input_dim, hidden_dim)
        self.memory_bn1 = nn.BatchNorm1d(hidden_dim)
        self.memory_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.memory_bn2 = nn.BatchNorm1d(hidden_dim)
        self.memory_out = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
            
        Returns:
            out: [batch_size, 2] - (latency, memory)
        """
        # Latency 分支
        lat = self.relu(self.latency_bn1(self.latency_fc1(x)))
        lat = self.dropout(lat)
        lat = self.relu(self.latency_bn2(self.latency_fc2(lat)))
        lat = self.dropout(lat)
        lat = self.latency_out(lat)
        
        # Memory 分支
        mem = self.relu(self.memory_bn1(self.memory_fc1(x)))
        mem = self.dropout(mem)
        mem = self.relu(self.memory_bn2(self.memory_fc2(mem)))
        mem = self.dropout(mem)
        mem = self.memory_out(mem)
        
        return torch.cat([lat, mem], dim=1)
