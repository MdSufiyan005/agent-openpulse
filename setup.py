# """
# setup.py
# ========
# Loads SmolVLM-Instruct model + processor, discovers linear layers,
# loads sample images. Returns a context object used by run.py.

# Model:   HuggingFaceTB/SmolVLM-Instruct
# Decoder: model.model.text_model.layers.* (language decoder only)
# Vision:  model.model.vision_model + model.model.connector — always F16, never quantized
# """

# from __future__ import annotations

# import os
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import List, Optional

# import torch
# import subprocess
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from dotenv import load_dotenv
# load_dotenv()

# IMAGE_DIR = os.environ.get("IMAGE_DIR", "images")
# MODEL_ID  = os.getenv("MODEL_ID", "unsloth/Qwen3-VL-2B-Thinking-1M-GGUF")
# MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))

# @dataclass
# class RunContext:
#     model:        torch.nn.Module
#     processor:    object
#     images:       List[str]
#     IMAGE_DIR:    str
#     linear_layers: List[str]
#     device:       torch.device
#     BLEU_FLOOR:   float = 0.8
#     clean_state:  dict  = field(default_factory=dict)
#     embedding_fn: Optional[object] = None
#     target_fn:    Optional[object] = None


# def _get_linear_layers(model: torch.nn.Module, decoder_only: bool = True) -> List[str]:
#     """
#     Returns names of all nn.Linear layers in the language decoder.
#     Excludes vision encoder and connector (always F16).
#     """
#     layers = []
#     for name, module in model.named_modules():
#         if not isinstance(module, torch.nn.Linear):
#             continue
#         if decoder_only:
#             # Only text decoder blocks
#             if "text_model.layers" not in name:
#                 continue
#         layers.append(name)
#     return layers


# # def export_to_gguf_if_needed(hf_model_id: str, output_gguf: str):
# #     if os.path.exists(output_gguf):
# #         print(f"[setup] Found existing GGUF: {output_gguf}")
# #         return

# #     print("[setup] Downloading HF model locally...")

# #     from huggingface_hub import snapshot_download

# #     local_dir = snapshot_download(
# #         repo_id=hf_model_id,
# #         local_dir="./artifacts/hf_model",
# #         local_dir_use_symlinks=False
# #     )

# #     print(f"[setup] Model downloaded to: {local_dir}")

# #     print("[setup] Converting to GGUF (F16)...")

# #     convert_script = "./llama.cpp/convert_hf_to_gguf.py"

# #     cmd = [
# #         "python", convert_script,
# #         local_dir,  # ✅ NOW this is a REAL directory
# #         "--outfile", output_gguf,
# #         "--outtype", "f16"
# #     ]

# #     result = subprocess.run(cmd, capture_output=True, text=True)

# #     if result.returncode != 0:
# #         print(result.stderr)
# #         raise RuntimeError("GGUF conversion failed")

# #     print(f"[setup] ✅ GGUF created: {output_gguf}")

# def export_to_gguf_if_needed(hf_model_id: str, output_gguf: str):
#     if os.path.exists(output_gguf):
#         print(f"[setup] Found existing GGUF: {output_gguf}")
#         return

#     print("[setup] Downloading MINIMAL HF model...")

#     from huggingface_hub import snapshot_download

#     local_dir = snapshot_download(
#         repo_id=hf_model_id,
#         local_dir="./artifacts/hf_model",
#         local_dir_use_symlinks=False,

#         # 🔥 ONLY download required files
#         allow_patterns=[
#             "*.json",
#             "*.safetensors",   # main weights
#             "*.model",         # tokenizer
#             "*.py",
#             "tokenizer.*",
#             "preprocessor_config.json"
#         ],

#         # ❌ explicitly ignore garbage
#         ignore_patterns=[
#             "*.bin",
#             "*.pt",
#             "*.h5",
#             "*.msgpack"
#         ]
#     )

#     print(f"[setup] Model downloaded to: {local_dir}")

#     print("[setup] Converting to GGUF (F16)...")

#     convert_script = "./llama.cpp/convert_hf_to_gguf.py"

#     cmd = [
#         "python", convert_script,
#         local_dir,
#         "--outfile", output_gguf,
#         "--outtype", "f16"
#     ]

#     result = subprocess.run(cmd, capture_output=True, text=True)

#     if result.returncode != 0:
#         print(result.stderr)
#         raise RuntimeError("GGUF conversion failed")

#     print(f"[setup] ✅ GGUF created: {output_gguf}")

# def build_inputs(processor, model, image_path: str) -> dict:
#     """
#     Builds model-ready inputs from an image.
#     Shared across tools (KL, quant, etc.)

#     Returns:
#         dict of tensors on correct device + dtype
#     """
#     from PIL import Image

#     device = next(model.parameters()).device
#     dtype  = next(model.parameters()).dtype

#     img = Image.open(image_path).convert("RGB")

#     prompt = processor.apply_chat_template(
#         [{
#             "role": "user",
#             "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": "Describe this image briefly."},
#             ],
#         }],
#         add_generation_prompt=True,
#     )

#     raw = processor(text=prompt, images=[img], return_tensors="pt")

#     return {
#         k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
#         for k, v in raw.items()
#     }


# def build_context() -> RunContext:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
#     print(f"Model id : {MODEL_ID}")
#     print(f"[setup] Loading model {MODEL_ID} on {device} ({dtype})")
#     # processor = AutoProcessor.from_pretrained(MODEL_ID)
#     # model     = AutoModelForImageTextToText.from_pretrained(
#     #     MODEL_ID,
#     #     torch_dtype=dtype,
#     #     device_map="auto" if torch.cuda.is_available() else None,
#     #     low_cpu_mem_usage=True,
#     # )
#     CACHE_DIR = "./artifacts/hf_cache"

#     os.makedirs(CACHE_DIR, exist_ok=True)

#     model = AutoModelForImageTextToText.from_pretrained(
#         MODEL_ID,
#         cache_dir=CACHE_DIR,
#         torch_dtype=dtype,
#         device_map="auto"   # 🔥 IMPORTANT
#     )

#     processor = AutoProcessor.from_pretrained(
#         MODEL_ID,
#         cache_dir=CACHE_DIR
#     )

#     if not torch.cuda.is_available():
#         model = model.to(device)
#     model.eval()

#     # Discover decoder linear layers
#     linear_layers = _get_linear_layers(model, decoder_only=True)
#     print(f"[setup] Found {len(linear_layers)} decoder linear layers")

#     # Load image filenames
#     print("CWD:", os.getcwd())
#     print("IMAGE_DIR:", IMAGE_DIR)
#     print("Exists?", Path(IMAGE_DIR).exists())
#     image_dir_path = Path(IMAGE_DIR)
#     if image_dir_path.exists():
#         all_images = sorted([
#             f for f in os.listdir(IMAGE_DIR)
#             if f.lower().endswith((".jpg", ".jpeg", ".png"))
#         ])[:MAX_IMAGES]
#     else:
#         print(f"[setup] WARNING: IMAGE_DIR={IMAGE_DIR} not found — using dummy images")
#         all_images = []

#     if not all_images:
#         print("[setup] No images found. Quality probes will be skipped.")

#     print(f"[setup] Using {len(all_images)} images: {all_images}")
#     # ✅ Export GGUF once for quantization pipeline
#     GGUF_PATH = "./artifacts/input-f16.gguf"

#     export_to_gguf_if_needed(
#     hf_model_id=MODEL_ID,
#     output_gguf="data/input-f16.gguf"
#     )

#     return RunContext(
#         model         = model,
#         processor     = processor,
#         images        = all_images,
#         IMAGE_DIR     = IMAGE_DIR,
#         linear_layers = linear_layers,
#         device        = device,
#     )


"""
setup.py
========
Loads the HF model + processor for KL/NNI analysis.
Separately handles GGUF export for the quantization pipeline.

KEY DISTINCTION:
  MODEL_ID      — base HuggingFace model (has config.json, loadable by transformers)
  GGUF_BASE     — path to the F16 GGUF used by llama-quantize (generated from MODEL_ID)

If you want to use a GGUF-only repo (e.g. unsloth/Qwen3-VL-2B-Thinking-1M-GGUF),
set MODEL_ID to the BASE model (Qwen/Qwen2.5-VL-2B-Instruct) and download the
GGUF separately — transformers cannot load GGUF repos directly.

Decoder layer discovery:
  Qwen2-VL:    model.layers.*
  SmolVLM:     model.model.text_model.layers.*
  Auto-detect: tries both, falls back to all Linear layers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import subprocess
from transformers import AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv

load_dotenv()

IMAGE_DIR  = os.environ.get("IMAGE_DIR",  "images")
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))

# ── Model ID (must be a standard HF repo with config.json) ───────────────────
# For Qwen3-VL use the base model; the GGUF is used only by llama-quantize.
# Examples:
#   "HuggingFaceTB/SmolVLM-Instruct"   — original project model
#   "Qwen/Qwen2.5-VL-2B-Instruct"      — Qwen2.5-VL base (loads fine)
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-2B-Instruct")

# Path to the F16 GGUF (generated once from MODEL_ID via llama.cpp converter)
GGUF_PATH = os.environ.get("BASE_GGUF", os.path.join("artifacts", "input-f16.gguf"))


@dataclass
class RunContext:
    model:         torch.nn.Module
    processor:     object
    images:        List[str]
    IMAGE_DIR:     str
    linear_layers: List[str]
    device:        torch.device
    BLEU_FLOOR:    float = 0.8
    clean_state:   dict  = field(default_factory=dict)
    embedding_fn:  Optional[object] = None
    target_fn:     Optional[object] = None


# ── Layer discovery ───────────────────────────────────────────────────────────

def _get_linear_layers(model: torch.nn.Module) -> List[str]:
    """
    Auto-detects decoder layer naming convention and returns all
    nn.Linear layer names within the text decoder.

    Supports:
      - Qwen2-VL:  model.layers.N.*
      - SmolVLM:   model.model.text_model.layers.N.*
      - Fallback:  all nn.Linear layers (excludes vision by name heuristic)
    """
    all_linear = [
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]

    # Try Qwen2-VL pattern first
    qwen_layers = [n for n in all_linear if "model.layers." in n]
    if qwen_layers:
        print(f"[setup] Layer pattern: Qwen2-VL (model.layers.*) — {len(qwen_layers)} layers")
        return qwen_layers

    # Try SmolVLM pattern
    smol_layers = [n for n in all_linear if "text_model.layers" in n]
    if smol_layers:
        print(f"[setup] Layer pattern: SmolVLM (text_model.layers.*) — {len(smol_layers)} layers")
        return smol_layers

    # Fallback: exclude obvious vision layers
    vision_keywords = ["vision", "visual", "patch_embed", "img", "image_encoder"]
    filtered = [
        n for n in all_linear
        if not any(kw in n.lower() for kw in vision_keywords)
    ]
    print(f"[setup] Layer pattern: fallback (all non-vision Linear) — {len(filtered)} layers")
    return filtered


# ── GGUF export ───────────────────────────────────────────────────────────────

def export_to_gguf_if_needed(hf_model_id: str, output_gguf: str):
    """
    Download the HF model and convert to F16 GGUF via llama.cpp.
    Skips if output_gguf already exists.
    """
    if os.path.exists(output_gguf):
        print(f"[setup] Found existing GGUF: {output_gguf}")
        return

    print(f"[setup] Downloading HF model: {hf_model_id}")
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=hf_model_id,
        local_dir="./artifacts/hf_model",
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json", "*.safetensors", "*.model",
            "*.py", "tokenizer.*", "preprocessor_config.json"
        ],
        ignore_patterns=["*.bin", "*.pt", "*.h5", "*.msgpack"],
    )
    print(f"[setup] Downloaded to: {local_dir}")

    convert_script = os.path.join("llama.cpp", "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        raise RuntimeError(
            f"[setup] convert_hf_to_gguf.py not found at {convert_script}. "
            "Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        )

    os.makedirs(os.path.dirname(output_gguf), exist_ok=True)

    cmd = ["python", convert_script, local_dir, "--outfile", output_gguf, "--outtype", "f16"]
    print(f"[setup] Converting to GGUF F16: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("GGUF conversion failed")

    print(f"[setup] GGUF created: {output_gguf}")


# ── Input builder (shared by KL + sparsity tools) ────────────────────────────

def build_inputs(processor, model, image_path: str) -> dict:
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


# ── Main context builder ──────────────────────────────────────────────────────

def build_context() -> RunContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[setup] MODEL_ID : {MODEL_ID}")
    print(f"[setup] Device   : {device} ({dtype})")

    CACHE_DIR = "./artifacts/hf_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load model — MODEL_ID must be a standard HF repo (not a GGUF repo)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=dtype,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    except ValueError as e:
        if "GGUF" in MODEL_ID or "gguf" in MODEL_ID.lower():
            raise ValueError(
                f"[setup] ERROR: MODEL_ID='{MODEL_ID}' appears to be a GGUF repo. "
                "Transformers cannot load GGUF repos directly.\n"
                "Set MODEL_ID to the base HF model, e.g.:\n"
                "  Qwen/Qwen2.5-VL-2B-Instruct\n"
                "  HuggingFaceTB/SmolVLM-Instruct\n"
                "The GGUF file for quantization is handled separately via BASE_GGUF env var."
            ) from e
        raise

    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    # Auto-detect decoder linear layers
    linear_layers = _get_linear_layers(model)
    print(f"[setup] Found {len(linear_layers)} decoder linear layers")

    # Load images
    image_dir_path = Path(IMAGE_DIR)
    if image_dir_path.exists():
        all_images = sorted([
            f for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:MAX_IMAGES]
    else:
        print(f"[setup] WARNING: IMAGE_DIR={IMAGE_DIR} not found — no calibration images")
        all_images = []

    print(f"[setup] Images: {all_images}")

    # Export GGUF if needed (uses MODEL_ID — the base HF model, not the GGUF repo)
    export_to_gguf_if_needed(hf_model_id=MODEL_ID, output_gguf=GGUF_PATH)

    return RunContext(
        model          = model,
        processor      = processor,
        images         = all_images,
        IMAGE_DIR      = IMAGE_DIR,
        linear_layers  = linear_layers,
        device         = device,
    )


# import hashlib, subprocess, os
# from pathlib import Path

# ARTIFACTS = Path("data")
# IMATRIX_PATH = ARTIFACTS / "imatrix.dat"
# IMATRIX_HASH_FILE = ARTIFACTS / "imatrix.hash"
# CALIBRATION_TXT = ARTIFACTS / "calibration.txt"

# def _model_hash(model_id: str) -> str:
#     return hashlib.md5(model_id.encode()).hexdigest()

# def _build_calibration_data(max_tokens: int = 512, num_samples: int = 200):
#     """Pull from NeelNanda/pile-10k, write plain text."""
#     if CALIBRATION_TXT.exists():
#         return
#     print("[imatrix] Building calibration dataset from NeelNanda/pile-10k ...")
#     from datasets import load_dataset
#     ds = load_dataset("NeelNanda/pile-10k", split="train")
#     lines = []
#     total = 0
#     for row in ds:
#         text = row["text"].strip().replace("\n", " ")
#         if len(text) < 50:
#             continue
#         lines.append(text[:1000])   # cap per-sample length
#         total += 1
#         if total >= num_samples:
#             break
#     CALIBRATION_TXT.write_text("\n".join(lines), encoding="utf-8")
#     print(f"[imatrix] Wrote {total} samples → {CALIBRATION_TXT}")

# def ensure_imatrix(model_id: str, f16_gguf: Path, llama_imatrix_bin: str = "llama-imatrix"):
#     """
#     Regenerate imatrix if model_id changed or imatrix.dat is missing.
#     Returns Path to imatrix.dat.
#     """
#     current_hash = _model_hash(model_id)
#     stored_hash = IMATRIX_HASH_FILE.read_text().strip() if IMATRIX_HASH_FILE.exists() else ""

#     if IMATRIX_PATH.exists() and current_hash == stored_hash:
#         print(f"[imatrix] Cache hit for {model_id}, skipping regeneration.")
#         return IMATRIX_PATH

#     print(f"[imatrix] Model changed or imatrix missing — regenerating for {model_id}")
#     _build_calibration_data()

#     cmd = [
#         llama_imatrix_bin,
#         "-m", str(f16_gguf),
#         "-f", str(CALIBRATION_TXT),
#         "-o", str(IMATRIX_PATH),
#         "--chunks", "200",     # matches num_samples above
#     ]
#     print(f"[imatrix] Running: {' '.join(cmd)}")
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"imatrix generation failed:\n{result.stderr}")

#     # Update hash only on success
#     IMATRIX_HASH_FILE.write_text(current_hash)
#     print(f"[imatrix] Done → {IMATRIX_PATH}")
#     return IMATRIX_PATH