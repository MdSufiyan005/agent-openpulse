"""
setup.py
========
Loads SmolVLM-Instruct model + processor, discovers linear layers,
loads sample images. Returns a context object used by run.py.

Model:   HuggingFaceTB/SmolVLM-Instruct
Decoder: model.model.text_model.layers.* (language decoder only)
Vision:  model.model.vision_model + model.model.connector — always F16, never quantized
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import subprocess
from transformers import AutoProcessor, AutoModelForImageTextToText

IMAGE_DIR = os.environ.get("IMAGE_DIR", "images")
MODEL_ID  = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolVLM-Instruct")
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))

@dataclass
class RunContext:
    model:        torch.nn.Module
    processor:    object
    images:       List[str]
    IMAGE_DIR:    str
    linear_layers: List[str]
    device:       torch.device
    BLEU_FLOOR:   float = 0.8
    clean_state:  dict  = field(default_factory=dict)
    embedding_fn: Optional[object] = None
    target_fn:    Optional[object] = None


def _get_linear_layers(model: torch.nn.Module, decoder_only: bool = True) -> List[str]:
    """
    Returns names of all nn.Linear layers in the language decoder.
    Excludes vision encoder and connector (always F16).
    """
    layers = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if decoder_only:
            # Only text decoder blocks
            if "text_model.layers" not in name:
                continue
        layers.append(name)
    return layers


# def export_to_gguf_if_needed(hf_model_id: str, output_gguf: str):
#     if os.path.exists(output_gguf):
#         print(f"[setup] Found existing GGUF: {output_gguf}")
#         return

#     print("[setup] Downloading HF model locally...")

#     from huggingface_hub import snapshot_download

#     local_dir = snapshot_download(
#         repo_id=hf_model_id,
#         local_dir="./artifacts/hf_model",
#         local_dir_use_symlinks=False
#     )

#     print(f"[setup] Model downloaded to: {local_dir}")

#     print("[setup] Converting to GGUF (F16)...")

#     convert_script = "./llama.cpp/convert_hf_to_gguf.py"

#     cmd = [
#         "python", convert_script,
#         local_dir,  # ✅ NOW this is a REAL directory
#         "--outfile", output_gguf,
#         "--outtype", "f16"
#     ]

#     result = subprocess.run(cmd, capture_output=True, text=True)

#     if result.returncode != 0:
#         print(result.stderr)
#         raise RuntimeError("GGUF conversion failed")

#     print(f"[setup] ✅ GGUF created: {output_gguf}")

def export_to_gguf_if_needed(hf_model_id: str, output_gguf: str):
    if os.path.exists(output_gguf):
        print(f"[setup] Found existing GGUF: {output_gguf}")
        return

    print("[setup] Downloading MINIMAL HF model...")

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=hf_model_id,
        local_dir="./artifacts/hf_model",
        local_dir_use_symlinks=False,

        # 🔥 ONLY download required files
        allow_patterns=[
            "*.json",
            "*.safetensors",   # main weights
            "*.model",         # tokenizer
            "*.py",
            "tokenizer.*",
            "preprocessor_config.json"
        ],

        # ❌ explicitly ignore garbage
        ignore_patterns=[
            "*.bin",
            "*.pt",
            "*.h5",
            "*.msgpack"
        ]
    )

    print(f"[setup] Model downloaded to: {local_dir}")

    print("[setup] Converting to GGUF (F16)...")

    convert_script = "./llama.cpp/convert_hf_to_gguf.py"

    cmd = [
        "python", convert_script,
        local_dir,
        "--outfile", output_gguf,
        "--outtype", "f16"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("GGUF conversion failed")

    print(f"[setup] ✅ GGUF created: {output_gguf}")

def build_inputs(processor, model, image_path: str) -> dict:
    """
    Builds model-ready inputs from an image.
    Shared across tools (KL, quant, etc.)

    Returns:
        dict of tensors on correct device + dtype
    """
    from PIL import Image

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    img = Image.open(image_path).convert("RGB")

    prompt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }],
        add_generation_prompt=True,
    )

    raw = processor(text=prompt, images=[img], return_tensors="pt")

    return {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in raw.items()
    }


def build_context() -> RunContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[setup] Loading model {MODEL_ID} on {device} ({dtype})")
    # processor = AutoProcessor.from_pretrained(MODEL_ID)
    # model     = AutoModelForImageTextToText.from_pretrained(
    #     MODEL_ID,
    #     torch_dtype=dtype,
    #     device_map="auto" if torch.cuda.is_available() else None,
    #     low_cpu_mem_usage=True,
    # )
    CACHE_DIR = "./artifacts/hf_cache"

    os.makedirs(CACHE_DIR, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        cache_dir=CACHE_DIR,
        torch_dtype=dtype,
        device_map="auto"   # 🔥 IMPORTANT
    )

    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        cache_dir=CACHE_DIR
    )

    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    # Discover decoder linear layers
    linear_layers = _get_linear_layers(model, decoder_only=True)
    print(f"[setup] Found {len(linear_layers)} decoder linear layers")

    # Load image filenames
    print("CWD:", os.getcwd())
    print("IMAGE_DIR:", IMAGE_DIR)
    print("Exists?", Path(IMAGE_DIR).exists())
    image_dir_path = Path(IMAGE_DIR)
    if image_dir_path.exists():
        all_images = sorted([
            f for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:MAX_IMAGES]
    else:
        print(f"[setup] WARNING: IMAGE_DIR={IMAGE_DIR} not found — using dummy images")
        all_images = []

    if not all_images:
        print("[setup] No images found. Quality probes will be skipped.")

    print(f"[setup] Using {len(all_images)} images: {all_images}")
    # ✅ Export GGUF once for quantization pipeline
    GGUF_PATH = "./artifacts/input-f16.gguf"

    export_to_gguf_if_needed(
    hf_model_id=MODEL_ID,
    output_gguf="data/input-f16.gguf"
    )

    return RunContext(
        model         = model,
        processor     = processor,
        images        = all_images,
        IMAGE_DIR     = IMAGE_DIR,
        linear_layers = linear_layers,
        device        = device,
    )
