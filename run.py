"""
run.py
======
KL-only layerwise quantization agent — SmolVLM-Instruct.

Pipeline:
  Phase 1: KL divergence on ALL decoder layers (microbatched, resumable)
  Phase 2: Merge → quant_assignments + llama-quantize command
  Phase 3: Agent synthesis → reviews assignments, adjusts, flags outliers
  Phase 4: Memory update → saves run + distils insights

Key design decisions:
  - noise_scale=0.05  (0.01 produces no signal for SmolVLM)
  - batch_size=20     (process 20 layers per microbatch to control memory)
  - resume=True       (restart-safe: skips already-scored layers)
  - Agent gets compact synthesis message (avoids 4096-token context overflow)
  - Vision projector excluded from analysis (always F16 in GGUF)
"""

from __future__ import annotations

import json
import os

from setup   import build_context
from planner import build_tools, create_ablation_agent
from memory  import save_run, update_agent_memory
from tools.kl_divergence_tool import make_kl_tool
from coding_agent import run_coding_agent
from tools.merge_report_tool  import merge_analysis_reports

# ── Config ────────────────────────────────────────────────────────────────────
NOISE_SCALE    = 0.05      # calibrated for SmolVLM — produces discriminative KL signal
KL_THRESHOLD   = 0.05      # bits above this = sensitive layer
BATCH_SIZE     = 20        # layers per microbatch
N_SAMPLES      = 3         # images to average KL over
SAVE_DIR       = "."
IMAGE_DIR      = os.environ.get("IMAGE_DIR", "/images")
SKIP_KL = True   # 🔥 toggle this

os.makedirs("results", exist_ok=True)
KL_PATH     = os.path.join(SAVE_DIR, "results/kl_divergence_report.json")
MERGED_PATH = os.path.join(SAVE_DIR, "results/unified_compression_report.json")

# ── Startup ───────────────────────────────────────────────────────────────────
print("=" * 64)
print("  QUANTIZATION AGENT — KL-ONLY PIPELINE")
print("=" * 64)

ctx           = build_context()
target_layers = ctx.linear_layers   # ALL decoder linear layers

print(f"\nTotal decoder layers to analyse: {len(target_layers)}")
print(f"KL config: noise_scale={NOISE_SCALE}, threshold={KL_THRESHOLD}, batch_size={BATCH_SIZE}")

# ── Phase 1: KL divergence (all layers, microbatched) ─────────────────────────
print("\n" + "=" * 64)
print("  PHASE 1 — KL DIVERGENCE (microbatched, resumable)")
print("=" * 64)

kl_tool = make_kl_tool(ctx.model, ctx.processor, ctx.images, ctx.IMAGE_DIR)
kl_result = kl_tool.invoke({
    "layer_names":    target_layers,
    "noise_scale":    NOISE_SCALE,
    "n_samples":      N_SAMPLES,
    "bits_threshold": KL_THRESHOLD,
    "batch_size":     BATCH_SIZE,
    "output_path":    KL_PATH,
    "resume":         True,
})
print(kl_result)
assert os.path.exists(KL_PATH), f"KL report not written to {KL_PATH} — aborting"
# ── Phase 1: KL divergence (optional) ─────────────────────────

# print("\n" + "=" * 64)
# print("  PHASE 1 — KL DIVERGENCE")
# print("=" * 64)

# if not SKIP_KL:
#     kl_tool = make_kl_tool(ctx.model, ctx.processor, ctx.images, ctx.IMAGE_DIR)
#     kl_result = kl_tool.invoke({
#         "layer_names":    target_layers,
#         "noise_scale":    NOISE_SCALE,
#         "n_samples":      N_SAMPLES,
#         "bits_threshold": KL_THRESHOLD,
#         "batch_size":     BATCH_SIZE,
#         "output_path":    KL_PATH,
#         "resume":         True,
#     })
#     print(kl_result)
#     assert os.path.exists(KL_PATH), f"KL report not written to {KL_PATH} — aborting"

# else:
#     print("⚡ SKIPPING KL — using existing report")
#     assert os.path.exists(KL_PATH), \
#         "KL report missing. Run once with SKIP_KL=False first."


# ── Phase 2: Merge → quant plan ───────────────────────────────────────────────
print("\n" + "=" * 64)
print("  PHASE 2 — MERGE → QUANT PLAN")
print("=" * 64)

merge_result = merge_analysis_reports.invoke({
    "kl_report_path": KL_PATH,
    "output_path":    MERGED_PATH,
    "default_quant":  "Q4_K_M",
})
print(merge_result)
assert os.path.exists(MERGED_PATH), f"Merged report not written — aborting"

# ── Phase 3: Agent synthesis ──────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  PHASE 3 — AGENT SYNTHESIS")
print("=" * 64)

with open(MERGED_PATH) as f:
    merged = json.load(f)

with open(KL_PATH) as f:
    kl_report = json.load(f)


def _build_synthesis_message(merged: dict, kl_report: dict, max_outliers: int = 20) -> str:
    """
    Compact message for the agent — stays well within 4096 token context.
    Includes: global stats, quant_assignments, top sensitive layers, llama-quantize cmd.
    """
    qa = merged.get("quant_assignments", {})
    layer_details = merged.get("layer_details", {})

    # Top sensitive layers (highest KL)
    ranked = kl_report.get("ranked_by_sensitivity", [])[:max_outliers]

    # Layers where quant is not Q4_K_M (these need the most review)
    non_default = {k: v for k, v in qa.items() if v != "Q4_K_M"}

    lines = [
        "KL divergence analysis is complete. Here is the summary for your review:",
        "",
        "GLOBAL STATS:",
        f"  Total decoder layers scored: {merged['total_layers']}",
        f"  COMPRESS (≤Q5_K_M): {merged['compress_count']}",
        f"  KEEP (≥Q6_K): {merged['keep_count']}",
        f"  Quant distribution: {merged.get('quant_distribution', {})}",
        f"  noise_scale used: {kl_report.get('noise_scale', NOISE_SCALE)}",
        "",
        f"TOP {len(ranked)} MOST SENSITIVE LAYERS (by KL bits):",
    ]
    for entry in ranked:
        name = entry["layer"]
        lines.append(
            f"  {name}: KL={entry['kl_bits']:.6f}b → {layer_details.get(name, {}).get('quant_type', '?')}"
        )

    lines += [
        "",
        f"NON-DEFAULT QUANT ASSIGNMENTS ({len(non_default)} layers above Q4_K_M):",
        json.dumps(non_default, indent=2),
        "",
        "PROPOSED LLAMA-QUANTIZE COMMAND:",
        merged.get("llama_quantize_cmd", "(not generated)"),
        "",
        "YOUR TASKS:",
        "1. Validate the quant_assignments — especially the KEEP layers.",
        "2. Adjust any layers where the KL signal seems wrong.",
        "3. Check: are first/last transformer blocks getting ≥Q6_K?",
        "4. Output final JSON in this format:",
        "",
        "```json",
        "{",
        '  "final_quant_assignments": {"tensor_name": "QUANT_TYPE", ...},',
        '  "llama_quantize_cmd": "full corrected command string",',
        '  "flags": ["layer: reason for special handling"],',
        '  "summary": "one paragraph explaining the strategy"',
        "}",
        "```",
    ]
    return "\n".join(lines)


synthesis_msg = _build_synthesis_message(merged, kl_report)
print(f"\n[run] Synthesis message: {len(synthesis_msg)} chars")

tools = build_tools(
    model     = ctx.model,
    processor = ctx.processor,
    images    = ctx.images,
    image_dir = ctx.IMAGE_DIR,
)
agent, llm = create_ablation_agent(tools, md_path="reference_docs/agent_reference.md")

config         = {"configurable": {"thread_id": "kl-synthesis-1"}}
final_response = ""

print("\n── AGENT RUNNING ────────────────────────────────────────────")
for step in agent.stream(
    {"messages": [("user", synthesis_msg)]},
    config=config,
    stream_mode="values",
):
    msgs = step.get("messages", [])
    if not msgs:
        continue

    last = msgs[-1]

    if hasattr(last, "content") and isinstance(last.content, str) and last.content.strip():
        # Ignore tool calls
        if hasattr(last, "tool_calls") and last.tool_calls:
            continue

        final_response = last.content

        print("\n── AGENT SYNTHESIS ──────────────────────────────────────")
        print(final_response)

        break   # ✅ CRITICAL: stop streaming once final answer is received

# ── Phase 4: Memory update ────────────────────────────────────────────────────
print("\n── SAVING MEMORY ────────────────────────────────────────────")
report_paths = {"kl": KL_PATH, "merged": MERGED_PATH}
save_run(
    target_layers  = target_layers,
    report_paths   = report_paths,
    agent_response = final_response,
)
new_insights = update_agent_memory(
    agent_response = final_response,
    report_paths   = report_paths,
    llm            = llm,
)
print(f"[memory] New insights: {new_insights}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  FINAL REPORT SUMMARY")
print("=" * 64)
for path, label in [(KL_PATH, "KL Divergence"), (MERGED_PATH, "Unified Plan")]:
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"\n── {label} ({path})")
        for k, v in d.items():
            if not isinstance(v, (dict, list)):
                print(f"   {k}: {v}")
    else:
        print(f"\n── {label} — NOT FOUND")

print("\n✓ Done. Load results/unified_compression_report.json for the full plan.")


# ── Phase 5: Coding Agent (Quantization Execution) ───────────────────────────
print("\n" + "=" * 64)
print("  PHASE 5 — CODING AGENT (GGUF GENERATION)")
print("=" * 64)

coding_result = run_coding_agent(
    merged_report_path = MERGED_PATH,
    input_gguf         = os.environ.get("INPUT_GGUF", "input-f16.gguf"),
    output_gguf        = os.environ.get("OUTPUT_GGUF", "output-layerwise.gguf"),
    imatrix_path       = os.environ.get("IMATRIX_PATH", ""),
    dry_run            = False   # ⚠️ MUST be False
)

print("\n── CODING AGENT RESULT ───────────────────────────────────")
print(coding_result)