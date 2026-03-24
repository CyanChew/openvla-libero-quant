import sys
import os
import argparse
import json
import numpy as np
import torch
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.schema import QuantConfig, BenchmarkResult
from benchmark.model_loader import load_openvla_model
from benchmark.dataset import load_libero_samples
from benchmark.utils import get_latency_stats, set_seed
from benchmark.metrics import calculate_action_deviation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--quant_method", type=str, default="fp16", choices=["bf16", "fp16", "bnb_int8", "bnb_int4"])
    parser.add_argument("--output_jsonl", type=str, default="outputs/results.jsonl")
    parser.add_argument("--samples_path", type=str, default="configs/sample_set.pt")
    parser.add_argument("--ref_actions_path", type=str, default="outputs/fp16_reference.pt", help="Path to reference fp16 actions .pt")
    
    parser.add_argument("--num_warmups", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    
    quant_config = QuantConfig(precision=args.quant_method, quant_scope="llm_only")
    
    res = BenchmarkResult(
        experiment_id=f"exp_{args.quant_method}_{os.path.basename(args.checkpoint.strip('/'))}",
        checkpoint=args.checkpoint,
        precision=args.quant_method,
        quant_scope=quant_config.quant_scope
    )
    
    try:
        fp16_reference = {}
        if os.path.exists(args.ref_actions_path):
            try:
                fp16_reference = torch.load(args.ref_actions_path)
            except Exception:
                pass

        print(f"Loading {args.quant_method} model...")
        processor, model = load_openvla_model(args.checkpoint, quant_config)
        model.eval()

        samples = load_libero_samples(args.samples_path)
        if not samples:
            raise ValueError("No samples loaded!")
            
        sample = samples[0] # use the first sample for latency & memory benchmark
        sample_id = sample["sample_id"]
        
        dtype = torch.bfloat16 if args.quant_method == "bf16" or "bf16" in args.quant_method else torch.float16
        
        def run_inference(samp):
            with torch.no_grad():
                inputs = processor(samp["instruction"], samp["image"]).to("cuda", dtype=dtype)
                if hasattr(model, "predict_action"):
                    return model.predict_action(**inputs, unnorm_key=None)
                else: # Fallback dummy
                    return model.generate(**inputs, max_new_tokens=256)
        
        print("Measuring Cold Latency...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        pred_action = run_inference(sample)
        torch.cuda.synchronize()
        res.cold_latency_ms = (time.perf_counter() - start) * 1000.0
        
        print(f"Warming up ({args.num_warmups} runs)...")
        for _ in range(args.num_warmups):
            run_inference(sample)
            
        print(f"Measuring Latency ({args.num_runs} runs)...")
        latencies = []
        for _ in range(args.num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_inference(sample)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
        stats = get_latency_stats(latencies)
        res.mean_latency_ms = stats["mean"]
        res.p95_latency_ms = stats["p95"]
        
        print("Measuring Peak Memory...")
        torch.cuda.reset_peak_memory_stats()
        run_inference(sample)
        res.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Calculate Deviation
        if sample_id in fp16_reference:
            ref_act = fp16_reference[sample_id]
            pred_act_np = pred_action.detach().cpu().numpy() if isinstance(pred_action, torch.Tensor) else pred_action
            ref_act_np = ref_act.numpy() if isinstance(ref_act, torch.Tensor) else ref_act
            
            l1, l2 = calculate_action_deviation(pred_act_np, ref_act_np)
            res.action_l1_vs_fp16 = float(l1)
            res.action_l2_vs_fp16 = float(l2)
            
        res.status = "success"

    except torch.cuda.OutOfMemoryError as e:
        res.status = "oom"
        res.error_message = "OOM: " + str(e)
        print(f"OOM during {args.quant_method} benchmark!")
    except Exception as e:
        if "out of memory" in str(e).lower():
            res.status = "oom"
            res.error_message = "OOM: " + str(e)
            print(f"OOM during {args.quant_method} benchmark!")
        else:
            res.status = "failed"
            res.error_message = str(e)
            print(f"Error during {args.quant_method} benchmark: {e}")

    # Append to JSONL
    with open(args.output_jsonl, "a") as f:
        f.write(json.dumps(res.__dict__) + "\n")
        
    print(f"[{res.status}] Exp: {res.experiment_id} saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()
