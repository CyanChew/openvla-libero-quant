import numpy as np
import torch
import random
from typing import List

def get_latency_stats(latencies: List[float]):
    if not latencies:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95))
    }

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
