# """
# tools/sparsity_tool.py
# ======================
# Sparsity sweep via NNI (Neural Network Intelligence) magnitude pruning.

# PURPOSE:
#   For layers where KL AND conductance are both ambiguous (disputed after both),
#   test how much weight sparsity the layer can tolerate before accuracy drops.
#   More tolerant → safer to quantize aggressively.

# WHEN AGENT USES THIS:
#   ONLY for genuinely disputed layers after KL + conductance both ran.
#   "Disputed" = KL within ±20% of threshold AND conductance rank is middle-tier.
#   Typically <5 layers per run. Do NOT use as a primary signal.

# MECHANISM:
#   1. For each sparsity level [0.3, 0.5, 0.7], apply L1-magnitude pruning mask
#   2. Run forward pass, measure KL from clean baseline
#   3. Find max sparsity where KL stays below 2× bits_threshold
#   4. Higher max_safe_sparsity → more compressible → can use lower quant type

# LIBRARY: nni.compression.pruning (pip install nni)
#   Falls back to manual magnitude masking if NNI unavailable (no hard dependency).
# """

# from __future__ import annotations

# import json
# import os
# from typing import List

# import torch
# import torch.nn.functional as F
# from langchain.tools import tool
# from pydantic import BaseModel, Field


# class SparsityInput(BaseModel):
#     layer_names: List[str] = Field(
#         description="Disputed layers to sweep. Keep to <5 layers. "
#                     "Only call after KL + conductance are both ambiguous."
#     )
#     sparsity_levels: List[float] = Field(
#         default=[0.3, 0.5, 0.7],
#         description="Sparsity ratios to test (fraction of weights zeroed)."
#     )
#     bits_threshold: float = Field(
#         default=0.05,
#         description="KL threshold used in KL analysis (for 2× safety margin)."
#     )
#     output_path: str = Field(default="results/sparsity_report.json")


# def make_sparsity_tool(model, processor, images: list, image_dir: str):
#     """
#     Returns a sparsity tolerance sweep tool.
#     Only invoke for genuinely disputed layers — this is expensive.
#     """
#     from setup import build_inputs

#     def _get_log_probs(mdl, inp):
#         with torch.no_grad():
#             out = mdl(**inp)
#         logits = out.logits[:, -1, :]  # keep batch dimension
#         logits = logits.squeeze(0)
#         return F.log_softmax(logits.float(), dim=-1).cpu()

#     def _kl_bits(log_p, log_q):
#         p = log_p.exp()
#         kl_nats = F.kl_div(log_q, p, reduction="sum", log_target=False).item()
#         import math
#         return max(kl_nats, 0.0) / math.log(2)

#     def _apply_magnitude_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
#         """Zero out the lowest |sparsity| fraction of weights by L1 magnitude."""
#         flat = weight.abs().flatten()
#         k = int(len(flat) * sparsity)
#         if k == 0:
#             return torch.ones_like(weight)
#         threshold = flat.kthvalue(k).values.item()
#         mask = (weight.abs() > threshold).to(weight.dtype)
#         return mask

#     @tool("sparsity_sweep", args_schema=SparsityInput)
#     def sparsity_sweep(
#         layer_names: List[str],
#         sparsity_levels: List[float] = [0.3, 0.5, 0.7],
#         bits_threshold: float = 0.05,
#         output_path: str = "results/sparsity_report.json",
#     ) -> str:
#         """
#         Sparsity tolerance sweep for genuinely disputed layers.
#         Tests how much pruning each layer tolerates before quality degrades.
#         More tolerant layers can use lower quant types safely.
#         Call ONLY after KL + conductance are both ambiguous for a layer.
#         """
#         os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

#         if not images:
#             return "ERROR: No images available."

#         model.eval()
#         module_dict = dict(model.named_modules())

#         img_path = os.path.join(image_dir, images[0])
#         try:
#             print(f"[sparsity] Building inputs from: {img_path}")
#             inp = build_inputs(processor, model, img_path)
#             print(f"[sparsity] Inputs ready")
#         except Exception as e:
#             return f"ERROR building inputs: {e}"

#         # Baseline log probs
#         print("[sparsity] Computing baseline forward pass...")
#         baseline_lp = _get_log_probs(model, inp)
#         print("[sparsity] Baseline ready")
#         tolerance_kl = bits_threshold * 2.0  # 2× threshold = acceptable degradation

#         log = {}

#         for name in layer_names:
#             print(f"\n[sparsity] Processing layer: {name}")
#             if name not in module_dict:
#                 log[name] = {"error": "not in model"}
#                 continue

#             m = module_dict[name]
#             if not hasattr(m, "weight") or m.weight is None:
#                 log[name] = {"error": "no weight"}
#                 continue

#             original = m.weight.data.clone()
#             layer_results = {}
#             max_safe_sparsity = 0.0

#             for sparsity in sorted(sparsity_levels):
#                 print(f"  → Applying sparsity {sparsity:.0%}")
#                 mask = _apply_magnitude_mask(original, sparsity)
#                 m.weight.data.copy_((original * mask).to(dtype=m.weight.dtype))

#                 try:
#                     pruned_lp = _get_log_probs(model, inp)
#                     kl = _kl_bits(baseline_lp, pruned_lp)
#                     safe = kl < tolerance_kl
#                     layer_results[str(sparsity)] = {
#                         "kl_bits": round(kl, 6),
#                         "safe": safe,
#                     }
#                     if safe:
#                         max_safe_sparsity = sparsity
#                     print(f"[sparsity] {name} @ {sparsity:.0%}: KL={kl:.4f}b {'✓' if safe else '✗'}")
#                 except Exception as e:
#                     layer_results[str(sparsity)] = {"error": str(e)}

#                 # Restore after each test
#                 m.weight.data.copy_(original)

#             # Recommend quant based on sparsity tolerance
#             # High tolerance → aggressively compressible
#             if max_safe_sparsity >= 0.7:
#                 recommendation = "COMPRESS_AGGRESSIVE"  # Q4_K_M safe
#             elif max_safe_sparsity >= 0.5:
#                 recommendation = "COMPRESS"             # Q5_K_M safe
#             elif max_safe_sparsity >= 0.3:
#                 recommendation = "COMPRESS_LIGHT"       # Q6_K safe
#             else:
#                 recommendation = "KEEP"                 # Q8_0 minimum

#             log[name] = {
#                 "max_safe_sparsity": max_safe_sparsity,
#                 "tolerance_kl_threshold": tolerance_kl,
#                 "sparsity_results": layer_results,
#                 "recommendation": recommendation,
#             }

#         report = {
#             "method": "sparsity_sweep_magnitude",
#             "sparsity_levels_tested": sparsity_levels,
#             "tolerance_kl": tolerance_kl,
#             "layers": log,
#         }

#         with open(output_path, "w") as f:
#             json.dump(report, f, indent=2)

#         summary = []
#         for n, d in log.items():
#             if "error" not in d:
#                 summary.append(f"  {n.split('.')[-3:]}: max_safe={d['max_safe_sparsity']:.0%} → {d['recommendation']}")

#         return (
#             f"Sparsity sweep done. {len(summary)} layers evaluated.\n"
#             + "\n".join(summary) + "\n"
#             f"  Saved: {output_path}\n"
#             f"  NOTE: Use recommendation to break ties only. KL is still primary signal."
#         )

#     return sparsity_sweep

"""
Tool: nni_sparsity_tool.py

WHAT THIS MEASURES:
    For each layer at each sparsity level:
        1. Apply L1 mask (zero out smallest weights)
        2. Run image through model → get output probabilities
        3. Restore weights
        4. Compare output probabilities: baseline vs pruned
        5. JS distance tells us how much the output changed

    High JS after pruning = layer is sensitive, pruning hurts output.
    Low JS after pruning  = layer is safe to prune at that sparsity.
"""

import torch
import torch.nn.functional as F
import gc
import json
import numpy as np
from typing import List
from scipy.spatial.distance import jensenshannon
from langchain.tools import tool
from pydantic import BaseModel, Field



def _js(P, Q):
    """JS distance between two probability distributions. Range [0, 1]."""
    eps = 1e-12
    P   = np.clip(P, eps, 1)
    Q   = np.clip(Q, eps, 1)
    return float(jensenshannon(P, Q, base=2))


def _get_output_probs(model, inputs):
    """One forward pass → output probability distribution [vocab_size]."""
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits                 # [1, seq_len, vocab]
    probs  = F.softmax(logits, dim=-1)      # → probabilities
    probs  = probs.mean(dim=1).squeeze(0)   # avg over seq → [vocab]
    return probs.cpu().float().numpy()

def _apply_l1_mask(model, layer_names, sparsity):
    """Zero out the smallest `sparsity` fraction of weights by L1 norm."""
    module_dict = dict(model.named_modules())
    applied     = {}

    for name in layer_names:
        if name not in module_dict:
            continue
        m = module_dict[name]
        if not hasattr(m, "weight") or m.weight is None:
            continue

        w      = m.weight.data
        scores = w.float().abs().flatten()
        k      = int(scores.numel() * sparsity)
        if k == 0:
            continue

        threshold        = scores.kthvalue(k).values.item()
        mask             = (w.float().abs() > threshold).to(dtype=w.dtype)
        m._orig_weight   = w.clone()
        m.weight.data    = (w.float() * mask.float()).to(dtype=w.dtype)
        applied[name]    = mask

        zeros = int((mask == 0).sum().item())
        print(f"    {name}: {zeros}/{mask.numel()} ({zeros/mask.numel()*100:.1f}% masked)")

    return applied


def _restore_weights(model, applied):
    """Restore weights that were masked."""
    module_dict = dict(model.named_modules())
    for name in applied:
        m = module_dict.get(name)
        if m and hasattr(m, "_orig_weight"):
            m.weight.data = m._orig_weight
            del m._orig_weight



class SparsityInput(BaseModel):
    layer_names:     List[str]   = Field(description="Layer names to sweep")
    sparsity_levels: List[float] = Field(default=[0.1, 0.3, 0.5])
    n_samples:       int         = Field(default=1)
    js_threshold:    float       = Field(default=0.1, description="JS distance above which pruning is considered harmful")
    bleu_floor:      float       = Field(default=0.8)
    n_eval_samples:  int         = Field(default=1)
    output_path:     str         = Field(default="sparsity_sensitivity_report.json")



def make_sparsity_tool(model, processor, images, image_dir, clean_state, device):
    def _build_inputs(image_name):
        from PIL import Image as PILImage
        import os

        model_device = next(model.parameters()).device
        target_dtype = model.model.text_model.layers[0].self_attn.q_proj.weight.dtype

        image  = PILImage.open(os.path.join(image_dir, image_name)).convert("RGB")
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in one small sentence."}
            ]}],
            add_generation_prompt=True
        )
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        return {
            k: v.to(device=model_device, dtype=target_dtype) if v.is_floating_point()
               else v.to(model_device)
            for k, v in inputs.items()
        }

    from nni.compression.pruning import L1NormPruner


    @tool("sparsity_sensitivity_sweep", args_schema=SparsityInput)
    def sparsity_sensitivity_sweep(
        layer_names,
        sparsity_levels=[0.1, 0.3, 0.5],
        n_samples=1,
        js_threshold=0.1,
        bleu_floor=0.8,
        n_eval_samples=1,
        output_path="sparsity_sensitivity_report.json",
    ) -> str:

        model.eval()
        log = {}
        eval_images = images[:n_samples]

        print(f"[sparsity] Building inputs...")
        all_inputs = [_build_inputs(img) for img in eval_images]

        print(f"[sparsity] Getting baseline...")
        baseline_probs = [_get_output_probs(model, inp) for inp in all_inputs]

        print(f"[sparsity] Starting NNI sweep...")

        for i, name in enumerate(layer_names):
            print(f"\n[sparsity] Layer {i+1}/{len(layer_names)}: {name}")
            log[name] = {}

            for sparsity in sparsity_levels:
                print(f"  → Sparsity {sparsity:.0%}")

                # 🔹 NNI CONFIG
                config_list = [{
                    "sparsity": sparsity,
                    "op_names": [name],
                }]

                try:
                    # 🔹 APPLY NNI PRUNING
                    pruner = L1NormPruner(model, config_list)
                    pruner.compress()

                    # 🔹 FORWARD PASS
                    pruned_probs = [
                        _get_output_probs(model, inp)
                        for inp in all_inputs
                    ]

                    # 🔹 CLEANUP (IMPORTANT)
                    pruner.unwrap_model()
                    del pruner
                    gc.collect()

                    # 🔹 JS COMPUTE
                    js = float(np.mean([
                        _js(p, q)
                        for p, q in zip(baseline_probs, pruned_probs)
                    ]))

                    safe = js < js_threshold

                    log[name][str(sparsity)] = {
                        "js_distance": round(js, 6),
                        "safe": safe,
                    }

                    print(f"    JS={js:.5f} | {'SAFE' if safe else 'SENSITIVE'}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    log[name][str(sparsity)] = {"error": str(e)}

        # 🔹 ANALYSIS
        results = []
        safe_layers = []
        keep_layers = []

        for name in layer_names:
            if name not in log:
                continue

            scores = log[name]
            safe_up_to = 0.0

            for sp in sorted(sparsity_levels):
                entry = scores.get(str(sp), {})
                if entry.get("safe", False):
                    safe_up_to = sp
                else:
                    break

            recommendation = (
                "COMPRESS" if safe_up_to >= 0.3 else
                "CAUTION" if safe_up_to >= 0.1 else
                "KEEP"
            )

            results.append({
                "layer": name,
                "safe_up_to": safe_up_to,
                "recommendation": recommendation,
                "scores": scores,
            })

            (safe_layers if recommendation != "KEEP" else keep_layers).append(name)

            print(f"  → {name}: safe_up_to={safe_up_to:.0%} | {recommendation}")

        report = {
            "method": "nni_sparsity_js",
            "sparsity_levels": sparsity_levels,
            "compress_layers": safe_layers,
            "keep_layers": keep_layers,
            "ranked": sorted(results, key=lambda x: x["safe_up_to"], reverse=True),
            "layers": log,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return f"Sparsity done. Saved: {output_path}"