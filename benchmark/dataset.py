import os
import torch
from PIL import Image
import numpy as np
from typing import List, Dict

def load_libero_samples(path: str, num_samples: int = 5) -> List[Dict]:
    if os.path.exists(path) and path.endswith('.pt'):
        try:
            samples = torch.load(path)
            if len(samples) > 0: return samples
        except Exception as e:
            print(f"Warning: Failed to load {path} ({e})")
            
    print(f"Generating {num_samples} dummy samples for benchmarking...")
    samples = []
    for i in range(num_samples):
        # Dummy image: 224x224 RGB
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        samples.append({
            "sample_id": i,
            "image": Image.fromarray(img_array),
            "instruction": "What action should the robot take to grasp the red block?"
        })
    return samples
