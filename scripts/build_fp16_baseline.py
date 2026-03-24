import sys
import os
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.schema import QuantConfig
from benchmark.model_loader import load_openvla_model
from benchmark.dataset import load_libero_samples
from benchmark.utils import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="HF model checkpoint")
    parser.add_argument("--samples_path", type=str, default="configs/sample_set.pt")
    parser.add_argument("--output_path", type=str, default="outputs/fp16_reference.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Loading FP16 baseline model from {args.checkpoint}...")
    quant_config = QuantConfig(precision="fp16", quant_scope="llm_only")
    processor, model = load_openvla_model(args.checkpoint, quant_config)
    model.eval()

    samples = load_libero_samples(args.samples_path)
    
    fp16_reference = {}
    print(f"Generating FP16 reference actions for {len(samples)} samples...")
    
    with torch.no_grad():
        for sample in samples:
            sample_id = sample["sample_id"]
            
            inputs = processor(sample["instruction"], sample["image"]).to("cuda", dtype=torch.float16)
            
            # Predict action (fallback to generate if predict_action is missing via config mapping)
            if hasattr(model, "predict_action"):
                action = model.predict_action(**inputs, unnorm_key=None)
            else:
                inputs = {
                    "input_ids": inputs.get("input_ids", inputs.get("qformer_input_ids")), 
                    "pixel_values": inputs.get("pixel_values")
                }
                action = model.generate(**inputs, max_new_tokens=256) # Fallback fallback
             
            # Convert to CPU tensor for storage
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu()
            elif isinstance(action, np.ndarray):
                action = torch.from_numpy(action).cpu()
                
            fp16_reference[sample_id] = action

    torch.save(fp16_reference, args.output_path)
    print(f"Successfully saved {len(fp16_reference)} actions to {args.output_path}")

if __name__ == "__main__":
    main()
