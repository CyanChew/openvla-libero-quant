import torch
import transformers
from transformers import BitsAndBytesConfig
from benchmark.schema import QuantConfig

def get_quantization_config(config: QuantConfig, compute_dtype: torch.dtype = torch.float16) -> BitsAndBytesConfig | None:
    if config.precision in ["fp16", "bf16"]:
        return None

    if config.precision == "bnb_int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=config.llm_int8_skip_modules if config.quant_scope == "llm_only" else None,
        )
    
    if config.precision == "bnb_int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Note: bnb_4bit does not formally accept llm_int8_skip_modules, but OpenVLA/HF supports passing it via HF loading config or manually handling ignoring 4bit
            llm_int8_skip_modules=config.llm_int8_skip_modules if config.quant_scope == "llm_only" else None,
        )

    # placeholder for GPTQ
    if config.precision == "gptq_int4":
        raise NotImplementedError("GPTQ support is reserved for phase 2 extensions.")
    
    raise ValueError(f"Unknown precision: {config.precision}")

def load_openvla_model(checkpoint_path: str, quant_config: QuantConfig, device: str = "cuda"):
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        torch_dtype = torch.bfloat16 if quant_config.precision == "bf16" or "bf16" in quant_config.precision else torch.float16

        config_kwargs = {}
        hf_quant_config = get_quantization_config(quant_config, compute_dtype=torch_dtype)
        
        if hf_quant_config is not None:
            config_kwargs["quantization_config"] = hf_quant_config
            config_kwargs["device_map"] = {"": device} # required so HF automatically loads directly into device
        else:
            config_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **config_kwargs
        )
        
        # VERY IMPORTANT: NEVER call .to() for bitsandbytes quantized models
        if hf_quant_config is None:
            model = model.to(device)
            
        return processor, model
    except ValueError as ve:
        if '.to(' in str(ve).lower() or 'move' in str(ve).lower():
             print(f"[FATAL] Mistakenly called .to() on a quantized model: {ve}")
        raise
    except Exception as e:
        print(f"[FATAL] Error loading {checkpoint_path} with {quant_config.precision}: {e}")
        raise
