"""
特征工程模块 - 从原始 JSON 数据提取和转换特征
"""
import re
from typing import Dict, List, Any, Set, Tuple
import numpy as np


# 常数定义
LLM_TOTAL_LAYERS = 32  # OpenVLA LLM 层数


def parse_skip_modules(skip_modules: List[str]) -> Dict[str, Any]:
    """
    从 skip_modules 列表解析特征
    
    规则：
    - "language_model.model.layers.X" 或 "language_model.model.layers.X.self_attn" → 提取层号 X
    - 计数 self_attn skips
    - 检查 vision_tower, multi_modal_projector, action_head
    
    Args:
        skip_modules: skip_modules 列表
        
    Returns:
        dict: 解析结果
    """
    if not skip_modules:
        return {
            'num_skip_modules': 0,
            'num_skip_llm_layers': 0,
            'num_skip_self_attn': 0,
            'skip_vision_tower': 0,
            'skip_projector': 0,
            'skip_action_head': 0,
            'skip_llm_layer_indices': set(),
        }
    
    skip_llm_layers: Set[int] = set()
    skip_self_attn_count = 0
    skip_vision_tower = 0
    skip_projector = 0
    skip_action_head = 0
    
    for module in skip_modules:
        # 检查 vision_tower
        if 'vision_tower' in module:
            skip_vision_tower = 1
        
        # 检查 projector
        if 'multi_modal_projector' in module or 'projector' in module:
            skip_projector = 1
        
        # 检查 action_head
        if 'action_head' in module:
            skip_action_head = 1
        
        # 提取 LLM 层号 - 匹配 language_model.model.layers.X 或 language_model.model.layers.X.self_attn
        match = re.search(r'language_model\.model\.layers\.(\d+)', module)
        if match:
            layer_idx = int(match.group(1))
            skip_llm_layers.add(layer_idx)
            
            # 统计 self_attn skips
            if 'self_attn' in module:
                skip_self_attn_count += 1
    
    return {
        'num_skip_modules': len(skip_modules),
        'num_skip_llm_layers': len(skip_llm_layers),
        'num_skip_self_attn': skip_self_attn_count,
        'skip_vision_tower': skip_vision_tower,
        'skip_projector': skip_projector,
        'skip_action_head': skip_action_head,
        'skip_llm_layer_indices': skip_llm_layers,
    }


def extract_features(sample: Dict[str, Any]) -> Dict[str, float]:
    """
    从一条 JSON 样本提取特征
    
    Args:
        sample: 一条 JSON 对象
        
    Returns:
        dict: 扁平化的特征字典
    """
    features = {}
    
    # ====== A. 量化基础特征 ======
    quant_config = sample.get('quant_config', {})
    
    features['weight_bits'] = float(quant_config.get('weight_bits', 0))
    features['load_in_4bit'] = float(quant_config.get('load_in_4bit', False))
    features['load_in_8bit'] = float(quant_config.get('load_in_8bit', False))
    features['llm_quant_ratio'] = float(quant_config.get('llm_quant_ratio', 0.0))
    
    # ====== B. Categorical → one-hot 编码 ======
    
    # quant_method (int4, int8)
    quant_method = quant_config.get('quant_method', 'unknown')
    features['quant_method_bnb_int4'] = float(quant_method == 'bnb_int4')
    features['quant_method_bnb_int8'] = float(quant_method == 'bnb_int8')
    
    # compute_dtype (bf16, fp32)
    compute_dtype = quant_config.get('compute_dtype', 'unknown')
    features['compute_dtype_bf16'] = float(compute_dtype == 'bf16')
    features['compute_dtype_fp32'] = float(compute_dtype == 'fp32')
    
    # llm_quant_target (ffn_only, all, attn_only, none)
    llm_target = quant_config.get('llm_quant_target', 'unknown')
    features['llm_target_ffn_only'] = float(llm_target == 'ffn_only')
    features['llm_target_all'] = float(llm_target == 'all')
    features['llm_target_attn_only'] = float(llm_target == 'attn_only')
    features['llm_target_none'] = float(llm_target == 'none')
    
    # llm_layer_selection (prefix, suffix, uniform)
    llm_selection = quant_config.get('llm_layer_selection', 'unknown')
    features['llm_selection_prefix'] = float(llm_selection == 'prefix')
    features['llm_selection_suffix'] = float(llm_selection == 'suffix')
    features['llm_selection_uniform'] = float(llm_selection == 'uniform')
    
    # vision_quant_target (none, projector_only, tower_only, all)
    vision_target = quant_config.get('vision_quant_target', 'unknown')
    features['vision_target_none'] = float(vision_target == 'none')
    features['vision_target_projector_only'] = float(vision_target == 'projector_only')
    features['vision_target_tower_only'] = float(vision_target == 'tower_only')
    features['vision_target_all'] = float(vision_target == 'all')
    
    # action_quant_target (none, all)
    action_target = quant_config.get('action_quant_target', 'unknown')
    features['action_target_none'] = float(action_target == 'none')
    features['action_target_all'] = float(action_target == 'all')
    
    # ====== C. 输入特征 ======
    input_config = sample.get('input_config', {})
    
    features['num_images'] = float(input_config.get('num_images', 0))
    features['batch_size'] = float(input_config.get('batch_size', 0))
    
    # 从分辨率字符串解析图像尺寸
    image_resolution = input_config.get('image_resolution', '224x224')
    try:
        h, w = map(int, image_resolution.split('x'))
        features['image_height'] = float(h)
        features['image_width'] = float(w)
    except (ValueError, AttributeError):
        features['image_height'] = 224.0
        features['image_width'] = 224.0
    
    # ====== D. 硬件特征 ======
    hardware = sample.get('hardware', {})
    features['gpu_mem_gb'] = float(hardware.get('gpu_mem_gb', 40.0))
    
    # ====== E. skip_modules 解析 ======
    skip_modules = quant_config.get('skip_modules', [])
    skip_info = parse_skip_modules(skip_modules)
    
    features['num_skip_modules'] = float(skip_info['num_skip_modules'])
    features['num_skip_llm_layers'] = float(skip_info['num_skip_llm_layers'])
    features['num_skip_self_attn'] = float(skip_info['num_skip_self_attn'])
    features['skip_vision_tower'] = float(skip_info['skip_vision_tower'])
    features['skip_projector'] = float(skip_info['skip_projector'])
    features['skip_action_head'] = float(skip_info['skip_action_head'])
    
    # ====== F. 推导特征 ======
    num_quantized_llm_layers = LLM_TOTAL_LAYERS - skip_info['num_skip_llm_layers']
    features['num_quantized_llm_layers'] = float(num_quantized_llm_layers)
    features['llm_quantized_ratio'] = float(num_quantized_llm_layers / LLM_TOTAL_LAYERS)
    
    return features


def build_feature_matrix(data: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    从样本列表构建特征矩阵
    
    Args:
        data: 清洗后的样本列表
        
    Returns:
        tuple: (特征矩阵 [N, D], 特征名→索引映射, 原始样本ID列表)
    """
    all_features = []
    feature_names = None
    
    for sample in data:
        feat_dict = extract_features(sample)
        all_features.append(feat_dict)
        
        if feature_names is None:
            feature_names = sorted(feat_dict.keys())
    
    # 构建矩阵
    X = np.zeros((len(all_features), len(feature_names)), dtype=np.float32)
    for i, feat_dict in enumerate(all_features):
        for j, fname in enumerate(feature_names):
            X[i, j] = feat_dict.get(fname, 0.0)
    
    # 创建特征名→索引映射
    feature_indices = {name: i for i, name in enumerate(feature_names)}
    
    return X, feature_indices, feature_names


def get_feature_names_and_types() -> Dict[str, str]:
    """
    返回所有特征及其类型说明
    """
    return {
        # 基础特征
        'weight_bits': 'numeric',
        'load_in_4bit': 'binary',
        'load_in_8bit': 'binary',
        'llm_quant_ratio': 'numeric',
        
        # quant_method one-hot
        'quant_method_bnb_int4': 'categorical_onehot',
        'quant_method_bnb_int8': 'categorical_onehot',
        
        # compute_dtype one-hot
        'compute_dtype_bf16': 'categorical_onehot',
        'compute_dtype_fp32': 'categorical_onehot',
        
        # llm_target one-hot
        'llm_target_ffn_only': 'categorical_onehot',
        'llm_target_all': 'categorical_onehot',
        'llm_target_attn_only': 'categorical_onehot',
        'llm_target_none': 'categorical_onehot',
        
        # llm_selection one-hot
        'llm_selection_prefix': 'categorical_onehot',
        'llm_selection_suffix': 'categorical_onehot',
        'llm_selection_uniform': 'categorical_onehot',
        
        # vision_target one-hot
        'vision_target_none': 'categorical_onehot',
        'vision_target_projector_only': 'categorical_onehot',
        'vision_target_tower_only': 'categorical_onehot',
        'vision_target_all': 'categorical_onehot',
        
        # action_target one-hot
        'action_target_none': 'categorical_onehot',
        'action_target_all': 'categorical_onehot',
        
        # 输入特征
        'num_images': 'numeric',
        'batch_size': 'numeric',
        'image_height': 'numeric',
        'image_width': 'numeric',
        
        # 硬件特征
        'gpu_mem_gb': 'numeric',
        
        # skip_modules 特征
        'num_skip_modules': 'numeric',
        'num_skip_llm_layers': 'numeric',
        'num_skip_self_attn': 'numeric',
        'skip_vision_tower': 'binary',
        'skip_projector': 'binary',
        'skip_action_head': 'binary',
        
        # 推导特征
        'num_quantized_llm_layers': 'numeric',
        'llm_quantized_ratio': 'numeric',
    }
