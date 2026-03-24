"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _parse_csv_modules(raw_modules):
    if raw_modules is None:
        return []
    if isinstance(raw_modules, (list, tuple)):
        return [str(item).strip() for item in raw_modules if str(item).strip()]
    return [item.strip() for item in str(raw_modules).split(",") if item.strip()]


def _select_layer_indices(num_layers, quant_ratio, strategy="prefix"):
    if num_layers <= 0:
        return set()

    ratio = float(max(0.0, min(1.0, quant_ratio)))
    if ratio == 0.0:
        return set()
    if ratio == 1.0:
        return set(range(num_layers))

    num_to_quantize = max(1, int(round(num_layers * ratio)))
    if strategy == "suffix":
        return set(range(num_layers - num_to_quantize, num_layers))

    if strategy == "uniform":
        raw_points = np.linspace(0, num_layers - 1, num=num_to_quantize)
        return set(int(round(point)) for point in raw_points)

    return set(range(num_to_quantize))


def _build_quant_skip_modules(cfg):
    custom_skip_modules = _parse_csv_modules(getattr(cfg, "quant_skip_modules", None))
    if custom_skip_modules:
        return sorted(set(custom_skip_modules))

    llm_quant_target = str(getattr(cfg, "llm_quant_target", "all")).lower().strip()
    llm_quant_ratio = float(getattr(cfg, "llm_quant_ratio", 1.0))
    llm_layer_selection = str(getattr(cfg, "llm_layer_selection", "prefix")).lower().strip()

    vision_quant_target = str(getattr(cfg, "vision_quant_target", "none")).lower().strip()
    action_quant_target = str(getattr(cfg, "action_quant_target", "none")).lower().strip()

    skip_modules = []

    num_hidden_layers = 0
    try:
        model_config = AutoConfig.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
        text_config = getattr(model_config, "text_config", None)
        if text_config is not None and hasattr(text_config, "num_hidden_layers"):
            num_hidden_layers = int(text_config.num_hidden_layers)
    except Exception as err:
        print(f"[WARN] Unable to infer LLM layer count from config: {err}")

    if llm_quant_target == "none":
        skip_modules.append("language_model")
    else:
        quantize_attn = llm_quant_target in {"all", "attn_only"}
        quantize_mlp = llm_quant_target in {"all", "ffn_only"}

        quantized_layer_indices = _select_layer_indices(
            num_layers=num_hidden_layers,
            quant_ratio=llm_quant_ratio,
            strategy=llm_layer_selection,
        )

        if num_hidden_layers > 0:
            for layer_idx in range(num_hidden_layers):
                layer_prefix = f"language_model.model.layers.{layer_idx}"
                if layer_idx not in quantized_layer_indices:
                    skip_modules.append(layer_prefix)
                    continue
                if not quantize_attn:
                    skip_modules.append(f"{layer_prefix}.self_attn")
                if not quantize_mlp:
                    skip_modules.append(f"{layer_prefix}.mlp")
        else:
            if not quantize_attn and not quantize_mlp:
                skip_modules.append("language_model")

    if vision_quant_target == "none":
        skip_modules.extend(["vision_tower", "multi_modal_projector"])
    elif vision_quant_target == "projector_only":
        skip_modules.append("vision_tower")
    elif vision_quant_target == "tower_only":
        skip_modules.append("multi_modal_projector")
    elif vision_quant_target == "all":
        pass
    else:
        raise ValueError(
            "Unsupported vision_quant_target. Use one of: none, projector_only, tower_only, all"
        )

    if action_quant_target == "none":
        skip_modules.append("action_head")
    elif action_quant_target == "all":
        pass
    else:
        raise ValueError("Unsupported action_quant_target. Use one of: none, all")

    return sorted(set(skip_modules))


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    quantization_config = None
    if cfg.load_in_4bit or cfg.load_in_8bit:
        from transformers import BitsAndBytesConfig
        skip_modules = _build_quant_skip_modules(cfg)
        cfg.effective_skip_modules = skip_modules
        if cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip_modules,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=skip_modules,
            )
    else:
        cfg.effective_skip_modules = []

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,  # Handled by quantization_config explicitly
        load_in_4bit=False,  # Handled by quantization_config explicitly
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if (cfg.load_in_8bit or cfg.load_in_4bit) else None,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
