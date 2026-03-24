"""
数据加载和清洗模块
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    读取 JSONL 文件
    
    Args:
        path: JSONL 文件路径
        
    Returns:
        list[dict]: 对象列表
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


def load_json_files_from_dir(directory: str) -> List[Dict[str, Any]]:
    """
    从目录加载所有 JSON 文件（单独的 JSON 文件，非 JSONL）
    
    Args:
        directory: 包含 JSON 文件的目录
        
    Returns:
        list[dict]: 对象列表
    """
    data = []
    path = Path(directory)
    json_files = sorted(path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in {directory}")
    for json_file in json_files:
        with open(json_file, 'r') as f:
            obj = json.load(f)
            # 添加源文件信息便于调试
            obj['_source_file'] = str(json_file.name)
            data.append(obj)
    
    print(f"Loaded {len(data)} samples from directory")
    return data


def filter_valid_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤有效样本：
    - status == "success"
    - metrics.mean_latency_ms 不为 None
    - metrics.peak_memory_mb 不为 None
    
    Args:
        data: 原始数据列表
        
    Returns:
        list[dict]: 清洗后的数据
    """
    valid = []
    
    for sample in data:
        # 检查 status
        if sample.get('status') != 'success':
            continue
            
        # 检查 metrics
        metrics = sample.get('metrics', {})
        mean_latency = metrics.get('mean_latency_ms')
        peak_memory = metrics.get('peak_memory_mb')
        
        if mean_latency is None or peak_memory is None:
            continue
            
        # 检查数值有效性
        try:
            mean_latency_val = float(mean_latency)
            peak_memory_val = float(peak_memory)
            
            # 排除异常值
            if mean_latency_val <= 0 or peak_memory_val <= 0:
                continue
                
            valid.append(sample)
        except (ValueError, TypeError):
            continue
    
    print(f"Filtered: {len(valid)} valid samples out of {len(data)} total")
    print(f"Removed {len(data) - len(valid)} invalid samples")
    
    return valid


def get_data_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    获取数据统计信息
    """
    if not data:
        return {}
    
    latencies = [s['metrics']['mean_latency_ms'] for s in data]
    memories = [s['metrics']['peak_memory_mb'] for s in data]
    inferences = [s['metrics'].get('inference_peak_mb', None) for s in data 
                  if s['metrics'].get('inference_peak_mb') is not None]
    
    stats = {
        'num_samples': len(data),
        'latency': {
            'min': min(latencies),
            'max': max(latencies),
            'mean': sum(latencies) / len(latencies),
        },
        'memory': {
            'min': min(memories),
            'max': max(memories),
            'mean': sum(memories) / len(memories),
        }
    }
    
    if inferences:
        stats['inference_memory'] = {
            'min': min(inferences),
            'max': max(inferences),
            'mean': sum(inferences) / len(inferences),
        }
    
    return stats
