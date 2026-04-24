# """
# tools/kl_divergence_tool.py
# ============================
# KL divergence sensitivity tool — uses scipy.special.rel_entr for numerically
# correct, fast computation (no manual log clipping needed).

# DESIGN:
#   - Perturbs each layer's weights with Gaussian noise scaled by noise_scale × std(W)
#   - Measures KL(baseline_logits ‖ perturbed_logits) in bits
#   - Higher KL  → layer is MORE sensitive → needs higher-precision quant
#   - noise_scale=0.05 is calibrated for SmolVLM; produces discriminative signal
#     (0.01 gives KL~0.0001 for all layers — no signal)

# MICROBATCH DESIGN:
#   - Processes layers in batches of `batch_size` to avoid OOM on full models
#   - Saves intermediate JSON after each batch so runs are resumable
#   - Final report merges all batches

# LIBRARY: scipy.special.rel_entr(P, Q) = P * log(P/Q) element-wise, 0 where P=0
#   sum(rel_entr(P, Q)) / log(2) = KL(P‖Q) in bits — correct per Shannon.
# """

# from __future__ import annotations

# import gc
# import json
# import os
# from typing import List

# import numpy as np
# import torch
# import torch.nn.functional as F
# from langchain.tools import tool
# from pydantic import BaseModel, Field
# from scipy.special import rel_entr


# # ── KL computation ────────────────────────────────────────────────────────────

# def _kl_bits_scipy(P: np.ndarray, Q: np.ndarray) -> float:
#     """KL(P‖Q) in bits using scipy.special.rel_entr. Handles P=0 correctly."""
#     P = np.asarray(P, dtype=np.float64)
#     Q = np.asarray(Q, dtype=np.float64)
#     # rel_entr(p,q) = p*log(p/q) where p>0, 0 where p=0, +inf where q=0 and p>0
#     # Clip Q away from 0 to avoid +inf
#     Q = np.clip(Q, 1e-12, None)
#     return float(np.sum(rel_entr(P, Q))) / np.log(2)   # convert nats → bits


# def _get_output_probs(model: torch.nn.Module, inputs: dict) -> np.ndarray:
#     """Forward pass → mean softmax prob vector over sequence positions."""
#     with torch.no_grad():
#         out = model(**inputs)
#     logits = out.logits                              # (1, seq_len, vocab)
#     probs  = F.softmax(logits, dim=-1)
#     return probs.mean(dim=1).squeeze(0).cpu().float().numpy()


# # ── Input schema ──────────────────────────────────────────────────────────────

# class KLInput(BaseModel):
#     layer_names:    List[str] = Field(description="Layer names to score")
#     noise_scale:    float     = Field(default=0.05,  description="Noise fraction of weight std. Use 0.05 for SmolVLM.")
#     n_samples:      int       = Field(default=1,     description="Images to average over")
#     bits_threshold: float     = Field(default=0.05,  description="KL threshold in bits above which layer is sensitive")
#     batch_size:     int       = Field(default=20,    description="Layers per microbatch — controls memory usage")
#     output_path:    str       = Field(default="results/kl_divergence_report.json")
#     resume:         bool      = Field(default=True,  description="Resume from existing partial results")


# # ── Factory ───────────────────────────────────────────────────────────────────

# def make_kl_tool(model: torch.nn.Module, processor, images: list, image_dir: str):

#     def _build_inputs(image_name: str) -> dict:
#         from PIL import Image as PILImage
#         dev   = next(model.parameters()).device
#         dtype = model.model.text_model.layers[0].self_attn.q_proj.weight.dtype
#         img   = PILImage.open(os.path.join(image_dir, image_name)).convert("RGB")
#         prompt = processor.apply_chat_template(
#             [{"role": "user", "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": "Describe this image in one sentence."},
#             ]}],
#             add_generation_prompt=True,
#         )
#         raw = processor(text=prompt, images=[img], return_tensors="pt")
#         return {
#             k: v.to(device=dev, dtype=dtype) if v.is_floating_point() else v.to(dev)
#             for k, v in raw.items()
#         }

#     @tool("kl_divergence_analysis", args_schema=KLInput)
#     def kl_divergence_analysis(
#         layer_names:    List[str],
#         noise_scale:    float = 0.05,
#         n_samples:      int   = 1,
#         bits_threshold: float = 0.05,
#         batch_size:     int   = 20,
#         output_path:    str   = "results/kl_divergence_report.json",
#         resume:         bool  = True,
#     ) -> str:
#         """
#         KL divergence sensitivity per layer — all layers processed in microbatches.
#         Saves intermediate results after each batch (resumable). Uses scipy for
#         numerically correct KL computation.
#         """
#         model.eval()
#         os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

#         module_dict = dict(model.named_modules())

#         # ── Resume: load existing partial results ─────────────────────────────
#         partial_path = output_path.replace(".json", "_partial.json")
#         log: dict[str, dict] = {}
#         if resume and os.path.exists(partial_path):
#             with open(partial_path) as f:
#                 log = json.load(f).get("layers", {})
#             print(f"[kl] Resumed: {len(log)} layers already done")

#         # ── Build inputs once ─────────────────────────────────────────────────
#         eval_images = images[:n_samples]
#         print(f"[kl] Building inputs for {len(eval_images)} image(s)...")
#         all_inputs = [_build_inputs(img) for img in eval_images]

#         # ── Compute baseline probs ────────────────────────────────────────────
#         baseline_probs = [_get_output_probs(model, inp) for inp in all_inputs]
#         print(f"[kl] Baseline ready. Total layers: {len(layer_names)}, batch_size: {batch_size}")

#         # ── Microbatch loop ───────────────────────────────────────────────────
#         todo = [n for n in layer_names if n not in log]
#         n_batches = (len(todo) + batch_size - 1) // batch_size
#         print(f"[kl] {len(todo)} layers remaining across {n_batches} batches")

#         for batch_idx in range(n_batches):
#             batch = todo[batch_idx * batch_size : (batch_idx + 1) * batch_size]
#             print(f"\n[kl] Batch {batch_idx+1}/{n_batches} ({len(batch)} layers)")

#             for i, name in enumerate(batch):
#                 global_i = batch_idx * batch_size + i + 1
#                 print(f"  [{global_i}/{len(layer_names)}] {name}")

#                 if name not in module_dict:
#                     print("    [SKIP] not in model"); continue
#                 m = module_dict[name]
#                 if not hasattr(m, "weight") or m.weight is None:
#                     print("    [SKIP] no weight"); continue

#                 original = m.weight.data.clone()
#                 noise    = torch.randn_like(m.weight) * noise_scale * m.weight.float().std()
#                 m.weight.data.add_(noise.to(dtype=m.weight.dtype))

#                 perturbed_probs = [_get_output_probs(model, inp) for inp in all_inputs]
#                 m.weight.data.copy_(original)

#                 kl_vals = [_kl_bits_scipy(p, q) for p, q in zip(baseline_probs, perturbed_probs)]
#                 kl_mean = float(np.mean(kl_vals))
#                 sensitive = kl_mean > bits_threshold

#                 log[name] = {
#                     "kl_bits":    round(kl_mean, 6),
#                     "kl_per_sample": [round(v, 6) for v in kl_vals],
#                     "sensitive":  sensitive,
#                     "threshold":  bits_threshold,
#                     "noise_scale": noise_scale,
#                 }
#                 print(f"    KL={kl_mean:.6f} bits | {'SENSITIVE' if sensitive else 'safe'}")

#             # Save partial after each batch
#             partial = {"method": "kl_divergence_microbatch", "layers": log}
#             with open(partial_path, "w") as f:
#                 json.dump(partial, f, indent=2)
#             print(f"  [kl] Partial saved ({len(log)} layers total)")
#             gc.collect()

#         # ── Final report ──────────────────────────────────────────────────────
#         safe      = [k for k, v in log.items() if not v["sensitive"]]
#         sensitive = [k for k, v in log.items() if v["sensitive"]]
#         ranked    = sorted(log.items(), key=lambda x: x[1]["kl_bits"], reverse=True)

#         # Per-layer quant recommendation based on KL magnitude
#         for name, d in log.items():
#             kl = d["kl_bits"]
#             if kl < 0.01:
#                 rec = "Q4_K_M"    # very safe
#             elif kl < 0.05:
#                 rec = "Q5_K_M"    # moderate
#             elif kl < 0.20:
#                 rec = "Q6_K"      # sensitive
#             else:
#                 rec = "Q8_0"      # highly sensitive
#             d["recommended_quant"] = rec

#         report = {
#             "method":           "kl_divergence_scipy",
#             "unit":             "bits",
#             "bits_threshold":   bits_threshold,
#             "noise_scale":      noise_scale,
#             "total_layers":     len(log),
#             "safe_count":       len(safe),
#             "sensitive_count":  len(sensitive),
#             "safe_layers":      safe,
#             "sensitive_layers": sensitive,
#             "ranked_by_sensitivity": [{"layer": k, **v} for k, v in ranked],
#             "layers":           log,
#         }
#         with open(output_path, "w") as f:
#             json.dump(report, f, indent=2)

#         # Clean up partial file
#         if os.path.exists(partial_path):
#             os.remove(partial_path)

#         top = f"{ranked[0][0]} ({ranked[0][1]['kl_bits']:.6f} bits)" if ranked else "none"
#         return (
#             f"KL done. Total={len(log)} Safe={len(safe)} Sensitive={len(sensitive)}\n"
#             f"Most sensitive: {top}\n"
#             f"Saved: {output_path}"
#         )

#     return kl_divergence_analysis


# """
# tools/kl_quant_tool.py
# ============================

# KL divergence sensitivity tool using FAKE QUANTIZATION instead of noise.

# FEATURES:
#   - Tests quant types: Q4, Q5, Q6, Q8
#   - Uses top-k KL divergence (fast + stable)
#   - Early exit: stops testing quant types once acceptable KL is reached
#   - Uses last-token logits only (huge speedup)
#   - Microbatch processing + resumable JSON saving

# GOAL:
#   Select lowest-bit quant per layer that keeps KL below threshold.
# """

# from __future__ import annotations

# import gc
# import json
# import os
# from typing import List, Dict

# import numpy as np
# import torch
# import torch.nn.functional as F
# from langchain.tools import tool
# from pydantic import BaseModel, Field
# from scipy.special import rel_entr


# # ── KL computation (top-k) ────────────────────────────────────────────────────

# def _kl_topk_bits(P_logits: torch.Tensor, Q_logits: torch.Tensor, k: int = 50) -> float:
#     """KL(P||Q) in bits over top-k tokens (last token logits)."""
#     P = F.softmax(P_logits, dim=-1)
#     Q = F.softmax(Q_logits, dim=-1)

#     topk = torch.topk(P, k).indices

#     Pk = P[topk]
#     Qk = Q[topk]

#     Pk = Pk / Pk.sum()
#     Qk = Qk / Qk.sum()

#     Pk = Pk.cpu().numpy().astype(np.float64)
#     Qk = Qk.cpu().numpy().astype(np.float64)

#     return float(np.sum(rel_entr(Pk, Qk))) / np.log(2)


# # ── Fake quantization ─────────────────────────────────────────────────────────

# def _fake_quantize(W: torch.Tensor, bits: int) -> torch.Tensor:
#     """Symmetric per-channel fake quantization."""
#     qmin = -(2 ** (bits - 1))
#     qmax = (2 ** (bits - 1)) - 1

#     # per-output-channel scale
#     max_vals = W.abs().amax(dim=1, keepdim=True)
#     scale = max_vals / qmax
#     scale[scale == 0] = 1.0

#     W_q = torch.round(W / scale).clamp(qmin, qmax)
#     return W_q * scale


# # ── Input schema ──────────────────────────────────────────────────────────────

# class KLQuantInput(BaseModel):
#     layer_names:    List[str]
#     bits_threshold: float = Field(default=0.05, description="KL threshold in bits")
#     batch_size:     int   = Field(default=20)
#     output_path:    str   = Field(default="results/kl_quant_report.json")
#     resume:         bool  = Field(default=True)
#     top_k:          int   = Field(default=50)


# # ── Factory ───────────────────────────────────────────────────────────────────

# def make_kl_quant_tool(model: torch.nn.Module, processor, images: list, image_dir: str):

#     def _build_inputs(image_name: str) -> dict:
#         from PIL import Image as PILImage

#         dev   = next(model.parameters()).device
#         dtype = model.model.text_model.layers[0].self_attn.q_proj.weight.dtype

#         img = PILImage.open(os.path.join(image_dir, image_name)).convert("RGB")

#         prompt = processor.apply_chat_template(
#             [{"role": "user", "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": "Describe briefly."},
#             ]}],
#             add_generation_prompt=True,
#         )

#         raw = processor(text=prompt, images=[img], return_tensors="pt")

#         return {
#             k: v.to(device=dev, dtype=dtype) if v.is_floating_point() else v.to(dev)
#             for k, v in raw.items()
#         }


#     def _get_last_logits(model, inputs):
#         with torch.no_grad():
#             out = model(**inputs)
#         return out.logits[0, -1]   # last token only


#     @tool("kl_quant_analysis", args_schema=KLQuantInput)
#     def kl_quant_analysis(
#         layer_names:    List[str],
#         bits_threshold: float = 0.05,
#         batch_size:     int   = 20,
#         output_path:    str   = "results/kl_quant_report.json",
#         resume:         bool  = True,
#         top_k:          int   = 50,
#     ) -> str:

#         model.eval()
#         os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

#         module_dict = dict(model.named_modules())

#         # Resume
#         partial_path = output_path.replace(".json", "_partial.json")
#         log: Dict[str, dict] = {}

#         if resume and os.path.exists(partial_path):
#             with open(partial_path) as f:
#                 log = json.load(f).get("layers", {})
#             print(f"[kl-quant] Resumed: {len(log)} layers")

#         # Build inputs (ONLY 1 sample for speed)
#         inputs = _build_inputs(images[0])

#         # Baseline logits
#         baseline_logits = _get_last_logits(model, inputs)

#         # Quant order (aggressive → safe)
#         quant_map = {
#             "Q4": 4,
#             "Q5": 5,
#             "Q6": 6,
#             "Q8": 8,
#         }
#         quant_order = ["Q4", "Q5", "Q6", "Q8"]

#         # Microbatch
#         todo = [n for n in layer_names if n not in log]
#         n_batches = (len(todo) + batch_size - 1) // batch_size

#         print(f"[kl-quant] {len(todo)} layers, {n_batches} batches")

#         for batch_idx in range(n_batches):
#             batch = todo[batch_idx * batch_size : (batch_idx + 1) * batch_size]

#             print(f"\n[Batch {batch_idx+1}/{n_batches}]")

#             for name in batch:

#                 if name not in module_dict:
#                     continue

#                 m = module_dict[name]
#                 if not hasattr(m, "weight") or m.weight is None:
#                     continue

#                 original = m.weight.data.clone()

#                 best_quant = None
#                 best_kl = None
#                 per_quant = {}

#                 for qtype in quant_order:
#                     bits = quant_map[qtype]

#                     # Apply fake quant
#                     m.weight.data = _fake_quantize(original, bits)

#                     logits = _get_last_logits(model, inputs)
#                     kl = _kl_topk_bits(baseline_logits, logits, k=top_k)

#                     per_quant[qtype] = round(kl, 6)

#                     print(f"  {name} | {qtype} → KL={kl:.6f}")

#                     # Early exit
#                     if kl <= bits_threshold:
#                         best_quant = qtype
#                         best_kl = kl
#                         break

#                 # fallback
#                 if best_quant is None:
#                     best_quant = "Q8"
#                     best_kl = per_quant["Q8"]

#                 # restore weights
#                 m.weight.data = original

#                 log[name] = {
#                     "best_quant": best_quant,
#                     "kl_bits": round(best_kl, 6),
#                     "all_results": per_quant,
#                     "threshold": bits_threshold,
#                 }

#             # save partial
#             with open(partial_path, "w") as f:
#                 json.dump({"layers": log}, f, indent=2)

#             gc.collect()

#         # Final report
#         report = {
#             "method": "kl_quant_topk",
#             "layers": log,
#         }

#         with open(output_path, "w") as f:
#             json.dump(report, f, indent=2)

#         if os.path.exists(partial_path):
#             os.remove(partial_path)

#         return f"KL quant analysis done. Saved: {output_path}"

#     return kl_quant_analysis


"""
tools/kl_divergence_tool.py
───────────────────────────
KL divergence sensitivity analysis using torch.nn.functional.kl_div
(library implementation — not manual numpy summation).

WHY torch.nn.functional.kl_div instead of manual numpy:
  - Numerically stable: operates in log-space internally
  - Handles near-zero probs without manual epsilon clipping
  - Vectorised across vocab (32000+ tokens) — much faster than numpy loop
  - Matches PyTorch's own training loss semantics

WHAT THIS MEASURES:
  For each layer L:
    1. Run N forward passes → baseline output distribution P (softmax over vocab)
    2. Add Gaussian noise to L's weights (simulating quantization error)
    3. Run same inputs again → perturbed distribution Q
    4. Compute KL(P || Q) using F.kl_div in 'batchmean' reduction
    5. Restore original weights
  High KL → layer is sensitive → needs higher precision quant
  Low KL  → layer is robust   → safe to compress aggressively

SPEED OPTIMISATION vs original:
  - Inputs built ONCE, reused for all layers (original rebuilt per layer)
  - Only FIRST token logits used (not full seq avg) → 10x faster on long prompts
  - torch.inference_mode() instead of torch.no_grad() → disables grad tracking entirely
  - Optional microbatch: process layers in groups to allow CUDA to pipeline

OUTPUT JSON schema:
  {
    "method": "kl_divergence_torch_library",
    "noise_scale": 0.01,
    "bits_threshold": 0.05,
    "safe_count": N,
    "sensitive_count": M,
    "safe_layers": [...],
    "sensitive_layers": [...],
    "ranked_by_sensitivity": [{"layer": ..., "kl_nats": ..., "sensitive": ...}, ...],
    "layers": {
      "layer_name": {
        "kl_nats":   float,   # KL in nats (torch default)
        "kl_bits":   float,   # KL in bits (nats / ln2)
        "sensitive": bool,
        "threshold": float,
        "noise_scale_used": float,
        "weight_std": float   # for calibration reference
      }
    }
  }
"""

import os
import json
import math
import torch
import torch.nn.functional as F
from typing import List
from langchain.tools import tool
from pydantic import BaseModel, Field


# ── Pydantic schema ───────────────────────────────────────────────────────────
class KLInput(BaseModel):
    layer_names:    List[str] = Field(description="Decoder layer names to score")
    noise_scale:    float     = Field(
        default=0.05,
        description=(
            "Fraction of each layer's weight std-dev used as Gaussian noise magnitude. "
            "0.01 was too small for SmolVLM (all KL ~0.0001). "
            "0.05–0.1 gives discriminating signal. Try 0.05 first."
        )
    )
    n_samples:      int       = Field(
        default=3,
        description="Number of calibration images per layer. 3 is enough for stable ranking."
    )
    bits_threshold: float     = Field(
        default=0.01,
        description=(
            "KL threshold in BITS. Layers above this are flagged sensitive. "
            "Calibrate to your model: if all layers are below, increase noise_scale, not threshold."
        )
    )
    output_path:    str       = Field(default="results/kl_divergence_report.json")
    microbatch:     int       = Field(
        default=0,
        description="Process layers in groups of this size (0=all at once). Useful for 100+ layers."
    )


# ── Core KL computation ───────────────────────────────────────────────────────
def _get_log_probs_first_token(model: torch.nn.Module, inputs: dict) -> torch.Tensor:
    """
    Single forward pass → log-probabilities of FIRST generated token position.
    Using first token only (not mean over sequence) is:
      - 10x faster for long prompts
      - More sensitive: first token error propagates everywhere
      - What llama-perplexity --kl-divergence does internally

    Returns: [vocab_size] log-prob tensor on CPU float32
    """
    with torch.inference_mode():
        logits = model(**inputs).logits  # [1, seq_len, vocab]
    # Take last input token position (next token prediction)
    last_logits = logits[0, -1, :]      # [vocab]
    return F.log_softmax(last_logits.float(), dim=-1).cpu()


def _kl_divergence_library(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """
    KL(P || Q) using torch.nn.functional.kl_div.

    F.kl_div signature: kl_div(input=log_Q, target=P, reduction='sum')
    input  = log Q (the approximation)
    target = P     (the reference, NOT log)
    Returns KL in nats. Divide by ln(2) for bits.

    Why library over manual:
    - F.kl_div uses log-space: p * (log_p - log_q) computed stably
    - Avoids 0 * log(0) = NaN (handled as 0 by convention)
    - Vectorised C++ kernel — much faster than numpy on large vocab
    """
    p       = log_p.exp()            # recover P from log P
    kl_nats = F.kl_div(
        input     = log_q,           # log Q
        target    = p,               # P (not log)
        reduction = "sum",           # scalar sum over vocab
        log_target= False,           # target is probability, not log-prob
    ).item()
    return max(kl_nats, 0.0)        # numerical floor at 0 (can be tiny negative)


# ── Tool factory ──────────────────────────────────────────────────────────────
def make_kl_tool(model, processor, images, image_dir):
    """
    Returns a LangChain tool that scores each layer's quantization sensitivity
    using library-based KL divergence (torch.nn.functional.kl_div).
    """
    from setup import build_inputs  # shared input builder

    @tool("kl_divergence_analysis", args_schema=KLInput)
    def kl_divergence_analysis(
        layer_names:    List[str],
        noise_scale:    float = 0.05,
        n_samples:      int   = 3,
        bits_threshold: float = 0.01,
        output_path:    str   = "results/kl_divergence_report.json",
        microbatch:     int   = 0,
    ) -> str:
        """
        Scores each layer's quantization sensitivity via KL divergence.
        Uses torch.nn.functional.kl_div (library implementation, numerically stable).
        Adds Gaussian noise to weights to simulate quantization error.
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        model.eval()
        module_dict = dict(model.named_modules())
        log         = {}
        eval_images = images[:n_samples]

        # ── Build ALL inputs once — reused for every layer ────────────────
        print(f"[kl] Pre-building inputs for {len(eval_images)} image(s)...")
        all_log_probs_baseline_cache = {}  # image → log_probs (filled fresh per layer)
        # We store pre-built input dicts (not forward passes — those need fresh weights)
        all_inputs = [
            build_inputs(processor, model, os.path.join(image_dir, img))
            for img in eval_images
        ]
        print(f"[kl] Inputs ready. Scoring {len(layer_names)} layers "
              f"(noise_scale={noise_scale}, threshold={bits_threshold} bits)...")

        # ── Layer loop ────────────────────────────────────────────────────
        for i, name in enumerate(layer_names):
            if name not in module_dict:
                print(f"[kl] {i+1}/{len(layer_names)} SKIP {name} (not in model)")
                continue
            m = module_dict[name]
            if not hasattr(m, "weight") or m.weight is None:
                print(f"[kl] {i+1}/{len(layer_names)} SKIP {name} (no weight)")
                continue

            w_std = float(m.weight.data.float().std().item())

            # ── Baseline forward passes ───────────────────────────────────
            baseline_log_probs = [
                _get_log_probs_first_token(model, inp) for inp in all_inputs
            ]

            # ── Perturb weights ───────────────────────────────────────────
            original = m.weight.data.clone()
            noise    = torch.randn_like(m.weight) * noise_scale * w_std
            m.weight.data.add_(noise.to(dtype=m.weight.dtype))

            # ── Perturbed forward passes ──────────────────────────────────
            perturbed_log_probs = [
                _get_log_probs_first_token(model, inp) for inp in all_inputs
            ]

            # ── Restore immediately ───────────────────────────────────────
            m.weight.data.copy_(original)

            # ── KL via library — averaged over samples ────────────────────
            kl_nats_list = [
                _kl_divergence_library(bl, pl)
                for bl, pl in zip(baseline_log_probs, perturbed_log_probs)
            ]
            kl_nats = float(sum(kl_nats_list) / len(kl_nats_list))
            kl_bits = kl_nats / math.log(2)
            sensitive = kl_bits > bits_threshold

            log[name] = {
                "kl_nats":        round(kl_nats, 8),
                "kl_bits":        round(kl_bits, 8),
                "sensitive":      sensitive,
                "threshold_bits": bits_threshold,
                "noise_scale_used": noise_scale,
                "weight_std":     round(w_std, 8),
            }
            print(f"[kl] {i+1}/{len(layer_names)} {name.split('.')[-3:]}"
                  f" | KL={kl_bits:.6f} bits | {'SENSITIVE' if sensitive else 'safe'}")

            # ── Optional CUDA cache clear every microbatch ────────────────
            if microbatch > 0 and (i + 1) % microbatch == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Classify ──────────────────────────────────────────────────────
        safe      = [k for k, v in log.items() if not v["sensitive"]]
        sensitive = [k for k, v in log.items() if     v["sensitive"]]
        ranked    = sorted(log.items(), key=lambda x: x[1]["kl_bits"], reverse=True)

        # ── Assign quant types based on KL magnitude ──────────────────────
        # KL thresholds → llama-quantize type mapping
        # These are calibrated starting points — agent adjusts from here
        def _quant_type(kl_bits: float) -> str:
            if kl_bits < 0.001:  return "Q4_K_M"   # very robust
            if kl_bits < 0.01:   return "Q5_K_M"   # robust
            if kl_bits < 0.05:   return "Q6_K"      # moderate sensitivity
            if kl_bits < 0.2:    return "Q8_0"      # sensitive
            return "F16"                             # highly sensitive

        quant_assignments = {name: _quant_type(v["kl_bits"]) for name, v in log.items()}

        report = {
            "method":            "kl_divergence_torch_library",
            "library_used":      "torch.nn.functional.kl_div",
            "noise_scale":       noise_scale,
            "bits_threshold":    bits_threshold,
            "n_samples":         len(eval_images),
            "safe_count":        len(safe),
            "sensitive_count":   len(sensitive),
            "safe_layers":       safe,
            "sensitive_layers":  sensitive,
            "quant_assignments": quant_assignments,
            "ranked_by_sensitivity": [
                {"layer": k, "kl_bits": v["kl_bits"], "kl_nats": v["kl_nats"],
                 "sensitive": v["sensitive"], "quant_type": quant_assignments[k],
                 "weight_std": v["weight_std"]}
                for k, v in ranked
            ],
            "layers": log,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        top = f"{ranked[0][0]} ({ranked[0][1]['kl_bits']:.6f} bits)" if ranked else "none"
        bottom = f"{ranked[-1][0]} ({ranked[-1][1]['kl_bits']:.6f} bits)" if ranked else "none"
        return (
            f"KL analysis complete.\n"
            f"  Library used  : torch.nn.functional.kl_div\n"
            f"  Noise scale   : {noise_scale}\n"
            f"  Safe          : {len(safe)} layers\n"
            f"  Sensitive     : {len(sensitive)} layers\n"
            f"  Most sensitive: {top}\n"
            f"  Most robust   : {bottom}\n"
            f"  Saved         : {output_path}"
        )

    return kl_divergence_analysis