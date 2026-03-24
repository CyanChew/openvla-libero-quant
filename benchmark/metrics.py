import torch
import time
import numpy as np
from contextlib import contextmanager

@contextmanager
def measure_memory_and_time():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    return (end_time - start_time) * 1000.0, peak_memory # return ms, MB
    
def calculate_action_deviation(actions_pred: np.ndarray, actions_ref: np.ndarray):
    """
    actions_pred: [seq_len, action_dim] or [batch, seq_len, action_dim]
    actions_ref: [seq_len, action_dim] or [batch, seq_len, action_dim]
    """
    diff = actions_pred - actions_ref
    l1 = np.mean(np.abs(diff))
    l2 = np.mean(np.sqrt(np.sum(diff**2, axis=-1)))
    return l1, l2
