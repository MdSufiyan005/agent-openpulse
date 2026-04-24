# compressing_context.py
# ======================
# Context compressor for KL + NNI reports
# Reduces token load while preserving decision signal

import json


# ─────────────────────────────────────────────
# KL CONTEXT COMPRESSOR
# ─────────────────────────────────────────────
def compress_kl_report(kl_report: dict, top_k: int = 20):
    """
    Keeps only:
    - top sensitive layers
    - borderline layers
    - minimal stats
    """

    ranked = kl_report.get("ranked_by_sensitivity", [])[:top_k]

    compressed_ranked = [
        {
            "layer": r["layer"],
            "kl_bits": r.get("kl_bits", 0),
            "quant_hint": r.get("quant_type", "unknown"),
        }
        for r in ranked
    ]

    return {
        "noise_scale": kl_report.get("noise_scale"),
        "total_layers": kl_report.get("total_layers"),
        # "top_sensitive_layers": compressed_ranked,
        "top_sensitive_layers": [
            {
                "layer": r["layer"].split(".")[-1],  # shorten name
                "kl": r["kl_bits"]
            }
            for r in ranked
        ],
        "borderline_layers": kl_report.get("borderline_layers", [])[:20],
    }


# ─────────────────────────────────────────────
# NNI CONTEXT COMPRESSOR
# ─────────────────────────────────────────────
def compress_nni_report(nni_report: dict, top_k: int = 50):
    """
    Converts huge layer lists into:
    - compress candidates
    - keep candidates
    - summary stats only
    """

    compress_layers = nni_report.get("compress_layers", [])[:top_k]
    keep_layers = nni_report.get("keep_layers", [])[:top_k]

    ranked = nni_report.get("ranked", [])[:top_k]

    simplified_ranked = []
    for r in ranked:
        simplified_ranked.append({
            "layer": r.get("layer"),
            "decision": r.get("recommendation"),
            "safe_up_to": r.get("safe_up_to"),
        })

    return {
        "method": nni_report.get("method"),
        "sparsity_levels": nni_report.get("sparsity_levels"),
        "compress_layers": compress_layers,
        "keep_layers": keep_layers,
        "ranked_summary": simplified_ranked,
    }


# ─────────────────────────────────────────────
# FINAL PLANNER CONTEXT BUILDER
# ─────────────────────────────────────────────
def build_planner_context(kl_report: dict, nni_report: dict, metrics: dict = None):
    """
    Unified compressed context for planner agent.
    Keeps KL + NNI + optional runtime metrics.
    """

    return {
        "kl": compress_kl_report(kl_report),
        "nni": compress_nni_report(nni_report),
        "metrics": metrics or {}
    }