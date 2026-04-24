# """
# tools/merge_report_tool.py
# ==========================
# Merges KL divergence report into a unified compression plan.

# Simplified from original: KL-only (JS and Rényi removed).
# Quant type assigned purely from KL magnitude buckets — no cross-tool voting needed.

# Output: unified_compression_report.json with:
#   - per-layer recommendation (COMPRESS / KEEP)
#   - quant_assignments dict  → tensor_name: quant_type_string
#   - llama_quantize_cmd  → ready-to-run command string
# """

# from __future__ import annotations

# import json
# import os
# from typing import Optional

# from langchain.tools import tool
# from pydantic import BaseModel, Field


# # ── KL → quant type mapping ───────────────────────────────────────────────────
# # Calibrated for SmolVLM / general decoder layers.
# # These match what the KL tool's recommended_quant field already assigns,
# # but the merge tool re-derives them from raw kl_bits so it's self-contained.

# def _kl_to_quant(kl_bits: float, layer_name: str) -> tuple[str, str]:
#     """
#     Returns (recommendation, quant_type) from KL magnitude.
#     Special cases: output/embedding layers always get Q6_K or above.
#     """
#     is_boundary = any(tok in layer_name for tok in
#                       ["embed", "lm_head", "output", "norm", ".0.", ".1."])

#     if kl_bits < 0.01:
#         return ("COMPRESS", "Q4_K_M")
#     elif kl_bits < 0.05:
#         rec = "COMPRESS" if not is_boundary else "KEEP"
#         return (rec, "Q5_K_M")
#     elif kl_bits < 0.20:
#         return ("KEEP", "Q6_K")
#     else:
#         return ("KEEP", "Q8_0")


# # ── Schema ────────────────────────────────────────────────────────────────────

# class MergeInput(BaseModel):
#     kl_report_path: str = Field(default="results/kl_divergence_report.json")
#     output_path:    str = Field(default="results/unified_compression_report.json")
#     default_quant:  str = Field(default="Q4_K_M",
#                                 description="Fallback quant for unlisted tensors in llama-quantize command")
#     model_gguf_path: Optional[str] = Field(default=None,
#                                            description="Path to input F16 GGUF for command generation")
#     out_gguf_path:   Optional[str] = Field(default=None,
#                                            description="Path for output quantized GGUF")


# # ── Tool ──────────────────────────────────────────────────────────────────────

# @tool("merge_analysis_reports", args_schema=MergeInput)
# def merge_analysis_reports(
#     kl_report_path:  str           = "results/kl_divergence_report.json",
#     output_path:     str           = "results/unified_compression_report.json",
#     default_quant:   str           = "Q4_K_M",
#     model_gguf_path: Optional[str] = None,
#     out_gguf_path:   Optional[str] = None,
# ) -> str:
#     """
#     Reads the KL divergence report and generates:
#     1. Per-layer compression plan (COMPRESS / KEEP + quant type)
#     2. quant_assignments dict for LLM review
#     3. llama-quantize --tensor-type command string
#     """
#     os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

#     if not os.path.exists(kl_report_path):
#         return f"ERROR: KL report not found at {kl_report_path}"

#     with open(kl_report_path) as f:
#         kl = json.load(f)

#     layers_kl: dict = kl.get("layers", {})
#     if not layers_kl:
#         return "ERROR: KL report has no 'layers' key."

#     # ── Build per-layer plan ──────────────────────────────────────────────────
#     layer_details:    dict[str, dict] = {}
#     quant_assignments: dict[str, str] = {}
#     compress_layers:  list[str] = []
#     keep_layers:      list[str] = []

#     for name, d in layers_kl.items():
#         kl_bits = d.get("kl_bits", 0.0)
#         rec, qt  = _kl_to_quant(kl_bits, name)

#         layer_details[name] = {
#             "kl_bits":          kl_bits,
#             "sensitive":        d.get("sensitive", False),
#             "recommendation":   rec,
#             "quant_type":       qt,
#             "recommended_quant": d.get("recommended_quant", qt),
#         }
#         quant_assignments[name] = qt

#         if rec == "COMPRESS":
#             compress_layers.append(name)
#         else:
#             keep_layers.append(name)

#     # ── Build llama-quantize command ──────────────────────────────────────────
#     # Group layers by quant type to minimize --tensor-type flags
#     by_quant: dict[str, list[str]] = {}
#     for name, qt in quant_assignments.items():
#         by_quant.setdefault(qt, []).append(name)

#     cmd_parts = ["./llama-quantize \\"]
#     if model_gguf_path:
#         in_gguf  = model_gguf_path
#         out_gguf = out_gguf_path or model_gguf_path.replace(".gguf", "_layerwise.gguf")
#     else:
#         in_gguf  = "input-f16.gguf"
#         out_gguf = "output-layerwise.gguf"

#     # Emit non-default quant types only (default handled by global quant arg)
#     for qt, names in sorted(by_quant.items()):
#         if qt == default_quant:
#             continue  # covered by the global quant type argument
#         # Build a regex pattern matching these layer names
#         # Use exact names since llama-quantize supports full tensor name matching
#         for n in names:
#             # Convert python module name to gguf tensor name format:
#             # model.model.text_model.layers.5.self_attn.q_proj → blk.5.attn_q
#             tensor_alias = _to_gguf_tensor_name(n)
#             cmd_parts.append(f"  --tensor-type \"{tensor_alias}={qt}\" \\")

#     cmd_parts.append(f"  {in_gguf} {out_gguf} {default_quant}")
#     llama_cmd = "\n".join(cmd_parts)

#     # ── Stats ─────────────────────────────────────────────────────────────────
#     compress_count = len(compress_layers)
#     keep_count     = len(keep_layers)
#     total          = len(layer_details)

#     report = {
#         "method":          "kl_only_merge",
#         "kl_report":       kl_report_path,
#         "total_layers":    total,
#         "compress_count":  compress_count,
#         "keep_count":      keep_count,
#         "compress_layers": compress_layers,
#         "keep_layers":     keep_layers,
#         "quant_assignments": quant_assignments,
#         "llama_quantize_cmd": llama_cmd,
#         "quant_distribution": {qt: len(ns) for qt, ns in by_quant.items()},
#         "layer_details":   layer_details,
#     }

#     with open(output_path, "w") as f:
#         json.dump(report, f, indent=2)

#     # Validation check: at least 20% should be COMPRESS
#     compress_pct = compress_count / max(total, 1) * 100
#     warn = ""
#     if compress_pct < 20:
#         warn = (
#             f"\n⚠  Only {compress_pct:.1f}% layers marked COMPRESS. "
#             "Consider lowering bits_threshold or increasing noise_scale in KL tool."
#         )

#     return (
#         f"Merge done. Total={total} COMPRESS={compress_count} ({compress_pct:.1f}%) KEEP={keep_count}\n"
#         f"Quant distribution: {report['quant_distribution']}\n"
#         f"Command written to report.\n"
#         f"Saved: {output_path}"
#         f"{warn}"
#     )


# def _to_gguf_tensor_name(python_name: str) -> str:
#     """
#     Best-effort conversion of HuggingFace module path to llama.cpp tensor name.
#     e.g. model.model.text_model.layers.5.self_attn.q_proj → blk.5.attn_q
#     These are approximate — the agent should verify against actual GGUF tensor names.
#     """
#     import re

#     # Extract layer index
#     m = re.search(r"layers?[.\[](\d+)", python_name)
#     layer_idx = m.group(1) if m else "?"

#     # Map sub-module to tensor suffix
#     suffix_map = {
#         "q_proj":     "attn_q",
#         "k_proj":     "attn_k",
#         "v_proj":     "attn_v",
#         "o_proj":     "attn_output",
#         "gate_proj":  "ffn_gate",
#         "up_proj":    "ffn_up",
#         "down_proj":  "ffn_down",
#         "fc1":        "ffn_up",
#         "fc2":        "ffn_down",
#     }
#     for py_suffix, gguf_suffix in suffix_map.items():
#         if python_name.endswith(py_suffix):
#             return f"blk.{layer_idx}.{gguf_suffix}"

#     # Fallback: use the raw name (llama-quantize accepts full tensor names too)
#     return python_name


"""
tools/merge_report_tool.py
==========================
Merges KL + optional Conductance + optional Sparsity into a unified plan.

Signal priority (hard-coded, not agent-configurable):
  1. KL divergence   — PRIMARY: always used, sets baseline quant
  2. Conductance     — SECONDARY: upgrades quant if layer is on critical activation path
  3. Sparsity        — TERTIARY: only for genuinely disputed layers; can downgrade or confirm

Logic:
  - Start from KL-assigned quant type
  - If conductance available for this layer AND conductance rank ≤ 30th percentile
    (top-tier critical): upgrade quant by one tier (Q4→Q5, Q5→Q6, Q6→Q8)
  - If sparsity available for this layer AND recommendation == COMPRESS_AGGRESSIVE:
    downgrade allowed if KL was borderline (between Q5 and Q6 threshold)
  - Critical layers (first/last 2, lm_head, embed) → minimum Q6_K always

Output: unified_compression_report.json
"""

from __future__ import annotations

import json
import os
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field


# ── Quant tier ordering ───────────────────────────────────────────────────────
QUANT_TIERS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]


def _tier(q: str) -> int:
    try:
        return QUANT_TIERS.index(q)
    except ValueError:
        return len(QUANT_TIERS) - 1


def _upgrade(q: str, steps: int = 1) -> str:
    idx = min(_tier(q) + steps, len(QUANT_TIERS) - 1)
    return QUANT_TIERS[idx]


def _is_critical(name: str) -> bool:
    """Returns True if this layer must be at least Q6_K."""
    critical_tokens = [
        "embed", "lm_head", "output",
        "layers.0.", "layers.1.",
        "layers.28.", "layers.29.",  # last 2 of 30-layer SmolVLM
    ]
    return any(tok in name for tok in critical_tokens)


def _kl_to_quant(kl_bits: float) -> str:
    """KL magnitude → quant type (matches KL tool thresholds)."""
    if kl_bits < 0.001:  return "Q4_K_M"
    if kl_bits < 0.01:   return "Q5_K_M"
    if kl_bits < 0.05:   return "Q6_K"
    if kl_bits < 0.2:    return "Q8_0"
    return "F16"


def _to_gguf_tensor_name(python_name: str) -> str:
    """Convert HuggingFace module path → llama.cpp GGUF tensor name."""
    import re
    m = re.search(r"layers?[.\[](\d+)", python_name)
    layer_idx = m.group(1) if m else "?"
    suffix_map = {
        "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
        "o_proj": "attn_output", "gate_proj": "ffn_gate",
        "up_proj": "ffn_up", "down_proj": "ffn_down",
        "fc1": "ffn_up", "fc2": "ffn_down",
    }
    for py_suffix, gguf_suffix in suffix_map.items():
        if python_name.endswith(py_suffix):
            return f"blk.{layer_idx}.{gguf_suffix}"
    if "lm_head" in python_name:
        return "output.weight"
    return python_name


# ── Schema ────────────────────────────────────────────────────────────────────

class MergeInput(BaseModel):
    kl_report_path:          str           = Field(default="results/kl_divergence_report.json")
    conductance_report_path: Optional[str] = Field(default=None,
        description="Optional. Only provide if conductance_analysis was run.")
    sparsity_report_path:    Optional[str] = Field(default=None,
        description="Optional. Only provide if sparsity_sweep was run.")
    output_path:             str           = Field(default="results/unified_compression_report.json")
    default_quant:           str           = Field(default="Q4_K_M")
    model_gguf_path:         Optional[str] = Field(default=None)
    out_gguf_path:           Optional[str] = Field(default=None)


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool("merge_analysis_reports", args_schema=MergeInput)
def merge_analysis_reports(
    kl_report_path:          str           = "results/kl_divergence_report.json",
    conductance_report_path: Optional[str] = None,
    sparsity_report_path:    Optional[str] = None,
    output_path:             str           = "results/unified_compression_report.json",
    default_quant:           str           = "Q4_K_M",
    model_gguf_path:         Optional[str] = None,
    out_gguf_path:           Optional[str] = None,
) -> str:
    """
    Merges KL divergence + optional conductance + optional sparsity reports
    into a unified compression plan with llama-quantize command.

    Signal priority: KL (primary) → Conductance (upgrade tiebreaker) → Sparsity (disputed only).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if not os.path.exists(kl_report_path):
        return f"ERROR: KL report not found at {kl_report_path}"

    with open(kl_report_path) as f:
        kl = json.load(f)

    # Load optional reports
    conductance_data = {}
    if conductance_report_path and os.path.exists(conductance_report_path):
        with open(conductance_report_path) as f:
            c = json.load(f)
        conductance_data = c.get("layers", {})
        c_ranked = c.get("ranked", [])
        # Build conductance percentile lookup: rank 1 = most critical
        n_c = len(c_ranked)
        c_percentile = {
            item["layer"]: (item["rank"] if "rank" in item else idx + 1) / max(n_c, 1)
            for idx, item in enumerate(c_ranked)
        }
        print(f"[merge] Conductance data loaded: {n_c} layers")
    else:
        c_percentile = {}

    sparsity_data = {}
    if sparsity_report_path and os.path.exists(sparsity_report_path):
        with open(sparsity_report_path) as f:
            s = json.load(f)
        sparsity_data = s.get("layers", {})
        print(f"[merge] Sparsity data loaded: {len(sparsity_data)} layers")

    layers_kl = kl.get("layers", {})
    if not layers_kl:
        return "ERROR: KL report has no 'layers' key."

    # ── Build per-layer plan ──────────────────────────────────────────────────
    layer_details:     dict[str, dict] = {}
    quant_assignments: dict[str, str]  = {}
    compress_layers:   list[str]       = []
    keep_layers:       list[str]       = []
    adjustments:       list[str]       = []

    for name, d in layers_kl.items():
        kl_bits  = d.get("kl_bits", 0.0)
        kl_quant = _kl_to_quant(kl_bits)
        final_quant = kl_quant
        reasons = [f"KL={kl_bits:.6f}b→{kl_quant}"]

        # ── Critical layer floor ──────────────────────────────────────────────
        if _is_critical(name) and _tier(final_quant) < _tier("Q6_K"):
            final_quant = "Q6_K"
            reasons.append("critical_floor→Q6_K")

        # ── Conductance upgrade (secondary) ───────────────────────────────────
        if name in conductance_data and name in c_percentile:
            pct = c_percentile[name]
            # Top 30% of conductance-scored layers = upgrade by 1 tier
            if pct <= 0.30:
                upgraded = _upgrade(final_quant, 1)
                if upgraded != final_quant:
                    reasons.append(f"conductance_top30%→{upgraded}")
                    final_quant = upgraded

        # ── Sparsity downgrade/confirm (tertiary, disputed only) ─────────────
        if name in sparsity_data:
            sp_rec = sparsity_data[name].get("recommendation", "")
            # Only allow downgrade if KL is truly borderline (was Q5 or Q6 from KL)
            if sp_rec == "COMPRESS_AGGRESSIVE" and kl_quant in ("Q5_K_M", "Q6_K"):
                downgraded = "Q4_K_M" if kl_quant == "Q5_K_M" else "Q5_K_M"
                # Never downgrade below Q6_K for critical layers
                if not _is_critical(name):
                    reasons.append(f"sparsity_tolerant→{downgraded}")
                    final_quant = downgraded

        # ── Record ────────────────────────────────────────────────────────────
        rec = "COMPRESS" if _tier(final_quant) <= _tier("Q5_K_M") else "KEEP"
        quant_assignments[name] = final_quant
        layer_details[name] = {
            "kl_bits": kl_bits,
            "kl_quant": kl_quant,
            "final_quant": final_quant,
            "recommendation": rec,
            "reasons": reasons,
            "sensitive": d.get("sensitive", False),
        }
        if rec == "COMPRESS":
            compress_layers.append(name)
        else:
            keep_layers.append(name)

        if len(reasons) > 1:
            adjustments.append(f"{name}: {' | '.join(reasons)}")

    # ── Build llama-quantize command ──────────────────────────────────────────
    by_quant: dict[str, list[str]] = {}
    for name, qt in quant_assignments.items():
        by_quant.setdefault(qt, []).append(name)

    in_gguf  = model_gguf_path  or "input-f16.gguf"
    out_gguf = out_gguf_path    or "output-layerwise.gguf"

    cmd_parts = ["./llama-quantize \\"]
    for qt, names in sorted(by_quant.items()):
        if qt == default_quant:
            continue
        for n in names:
            tensor_name = _to_gguf_tensor_name(n)
            cmd_parts.append(f'  --tensor-type "{tensor_name}={qt}" \\')

    cmd_parts.append(f"  {in_gguf} {out_gguf} {default_quant}")
    llama_cmd = "\n".join(cmd_parts)

    # ── Stats ─────────────────────────────────────────────────────────────────
    total = len(layer_details)
    compress_count = len(compress_layers)
    keep_count = len(keep_layers)
    quant_dist = {qt: len(ns) for qt, ns in by_quant.items()}

    report = {
        "method": "kl_conductance_sparsity_merge",
        "signals_used": {
            "kl": True,
            "conductance": bool(conductance_data),
            "sparsity": bool(sparsity_data),
        },
        "kl_report": kl_report_path,
        "conductance_report": conductance_report_path,
        "sparsity_report": sparsity_report_path,
        "total_layers": total,
        "compress_count": compress_count,
        "keep_count": keep_count,
        "compress_layers": compress_layers,
        "keep_layers": keep_layers,
        "quant_assignments": quant_assignments,
        "llama_quantize_cmd": llama_cmd,
        "quant_distribution": quant_dist,
        "layer_details": layer_details,
        "adjustments_applied": adjustments,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    compress_pct = compress_count / max(total, 1) * 100
    warn = ""
    if compress_pct < 20:
        warn = (
            f"\n⚠  Only {compress_pct:.1f}% COMPRESS — noise_scale may be too low. "
            "Re-run KL with noise_scale=0.1."
        )

    return (
        f"Merge done. Total={total} COMPRESS={compress_count} ({compress_pct:.1f}%) KEEP={keep_count}\n"
        f"Signals: KL=✓  Conductance={'✓' if conductance_data else '✗'}  Sparsity={'✓' if sparsity_data else '✗'}\n"
        f"Quant distribution: {quant_dist}\n"
        f"Adjustments applied: {len(adjustments)}\n"
        f"Saved: {output_path}"
        f"{warn}"
    )