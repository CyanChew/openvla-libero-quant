from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class QuantConfig:
    precision: str = "fp16"  # "fp16", "bf16", "bnb_int8", "bnb_int4", "gptq_int4"
    quant_scope: str = "llm_only" # "llm_only" skips vision, projected, and action heads
    llm_int8_skip_modules: List[str] = field(default_factory=lambda: ["vision_tower", "multi_modal_projector", "action_head"])

@dataclass
class BenchmarkResult:
    experiment_id: str
    checkpoint: str
    precision: str
    quant_scope: str
    cold_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    action_l1_vs_fp16: Optional[float] = None
    action_l2_vs_fp16: Optional[float] = None
    status: str = "success"
    error_message: Optional[str] = None
