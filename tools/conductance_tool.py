"""
tools/conductance_tool.py
=========================
Layer Conductance via Captum — secondary signal for tiebreaking KL.

PURPOSE:
  KL divergence measures output shift from weight noise.
  Conductance measures how much a layer's *activations* contribute to the
  final prediction (gradient × activation, integrated). These are orthogonal:
  - A layer can have low KL (robust to noise) but high conductance (critical path)
  - High conductance + low KL → safe to quantize but keep it Q5_K_M minimum
  - High conductance + high KL → definitely Q8_0 or F16

WHEN AGENT USES THIS:
  Only for "disputed" layers — those where KL is borderline (within ±20% of threshold).
  Do NOT run on all layers (too slow). Agent calls with a short list.

LIBRARY: captum.attr.LayerConductance
  - Requires: pip install captum
  - Computes: integrated gradients of output score w.r.t. layer activations
  - Returns: absolute mean conductance (scalar per layer)
"""

from __future__ import annotations

import json
import os
from typing import List, Optional
import gc

import torch
import torch.nn.functional as F
from langchain.tools import tool
from pydantic import BaseModel, Field


class ConductanceInput(BaseModel):
    layer_names: List[str] = Field(
        description="Layers to score with conductance. Keep this list SHORT (<20 layers). "
                    "Only use for layers where KL is ambiguous/borderline."
    )
    image_name: Optional[str] = Field(
        default=None,
        description="Single image filename to use (from image_dir). If None, uses first available."
    )
    n_steps: int = Field(
        default=20,
        description="Integrated gradients steps. 20 is fast; 50 is more accurate."
    )
    output_path: str = Field(default="results/conductance_report.json")


def make_conductance_tool(model, processor, images: list, image_dir: str):
    """
    Returns a LangChain tool for layer conductance scoring.
    Only scores layers provided — keep list short for speed.
    """
    from setup import build_inputs

    @tool("conductance_analysis", args_schema=ConductanceInput)
    def conductance_analysis(
        layer_names: List[str],
        image_name: Optional[str] = None,
        n_steps: int = 20,
        output_path: str = "results/conductance_report.json",
    ) -> str:
        """
        Computes layer conductance for a SHORT list of disputed layers.
        Use only as a tiebreaker when KL is borderline. Do NOT call on all layers.
        Returns a conductance score per layer — higher = more critical to predictions.
        """
        try:
            from captum.attr import LayerGradientXActivation
        except ImportError:
            return "ERROR: captum not installed. Run: pip install captum"

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if not images:
            return "ERROR: No images available for conductance analysis."

        img_file = image_name or images[0]
        img_path = os.path.join(image_dir, img_file)

        try:
            inputs = build_inputs(processor, model, img_path)
        except Exception as e:
            return f"ERROR building inputs: {e}"

        module_dict = dict(model.named_modules())
        log = {}

        # We need input_ids as the baseline anchor for integrated gradients
        # Captum LayerConductance works on embedding-level inputs
        # We use a forward wrapper that takes embeddings
        try:
            embed_layer = model.model.text_model.embed_tokens
        except AttributeError:
            return "ERROR: Cannot find embed_tokens — adjust model path in conductance_tool.py"

        input_ids = inputs.get("input_ids")
        
        if input_ids is None:
            return "ERROR: input_ids not found in model inputs."

        # Build baseline (zero embeddings = neutral reference)
        with torch.no_grad():
            embeddings = embed_layer(input_ids)  # (1, seq, hidden)
        baseline_embeds = torch.zeros_like(embeddings)

        model.eval()

        for name in layer_names:
            if name not in module_dict:
                log[name] = {"error": "not in model", "conductance": 0.0, "rank": "unknown"}
                continue

            layer = module_dict[name]

            try:
                # Forward wrapper: takes embeddings, returns scalar logit score
                def forward_fn(embeds):
                    # Replace input_ids with pre-computed embeddings via inputs_embeds
                    fwd_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
                    fwd_inputs["inputs_embeds"] = embeds
                    out = model(**fwd_inputs)
                    # Use mean of max logits as scalar target
                    return out.logits[:, -1, :].max(dim=-1).values

                lgxa = LayerGradientXActivation(forward_fn, layer)

                print(f"[conductance] Running {name}...")

                attr = lgxa.attribute(
                    inputs=embeddings,
                    target=None
                )
                # attr shape varies by layer type; take abs mean
                conductance_score = float(attr.abs().mean().item())

                # Normalize relative to weight std for comparability
                if hasattr(layer, "weight") and layer.weight is not None:
                    w_std = float(layer.weight.data.float().std().item())
                    normalized = conductance_score / (w_std + 1e-8)
                else:
                    normalized = conductance_score

                log[name] = {
                    "conductance_raw": round(conductance_score, 8),
                    "conductance_normalized": round(normalized, 8),
                    "rank": "unknown",  # ranked after all layers scored
                }
                print(f"[conductance] {name}: {conductance_score:.6f} (norm: {normalized:.6f})")

            except Exception as e:
                log[name] = {"error": str(e), "conductance": 0.0, "conductance_normalized": 0.0}
                print(f"[conductance] ERROR {name}: {e}")
                # ✅ ADD THIS
            del lgxa
            del attr
            torch.cuda.empty_cache()
            gc.collect()

        # Rank by normalized conductance
        scored = [(n, d.get("conductance_normalized", 0.0)) for n, d in log.items() if "error" not in d]
        scored.sort(key=lambda x: x[1], reverse=True)
        for rank_idx, (n, _) in enumerate(scored):
            log[n]["rank"] = rank_idx + 1

        report = {
            "method": "layer_conductance_captum",
            "n_steps": n_steps,
            "image_used": img_file,
            "layers_scored": len(scored),
            "ranked": [{"layer": n, "conductance_normalized": s} for n, s in scored],
            "layers": log,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        top3 = scored[:3]
        top_str = ", ".join(f"{n.split('.')[-3:]}({s:.4f})" for n, s in top3)
        return (
            f"Conductance analysis done. {len(scored)} layers scored.\n"
            f"  Top 3 most critical: {top_str}\n"
            f"  Saved: {output_path}\n"
            f"  NOTE: Use these scores only as KL tiebreaker. "
            f"High conductance + borderline KL → upgrade quant type by one tier."
        )

    return conductance_analysis