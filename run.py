from __future__ import annotations

import json
import os
import time
import sys

from setup import build_context
from agents.planner import build_tools, create_ablation_agent
from agents.memory import save_run, update_agent_memory
from agents.summarizer_agent import run_metrics_agent
from agents.coding_agent import run_coding_agent
from tools.kl_divergence_tool import make_kl_tool
from tools.sparsity_tool import make_sparsity_tool
from tools.deployment.modelpulse_tool import (
    start_modelpulse_server,
    stop_modelpulse_server,
    upload_model_to_server,
    shard_gguf,
    METRICS_JSONL_PATH,
)
from tools.compress_context import build_planner_context
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ─────────────────────────────────────────────
# PATHS  (single source of truth)
# ─────────────────────────────────────────────
ARTIFACTS_DIR   = "artifacts"
RESULTS_DIR     = os.path.join(ARTIFACTS_DIR, "results")
MODELS_DIR      = os.path.join(ARTIFACTS_DIR, "models-storage")
MODEL_ID = os.getenv("MODEL_ID")
SHARDS_DIR      = os.path.join(ARTIFACTS_DIR, "shards")

KL_PATH         = os.path.join(RESULTS_DIR, "kl_divergence_report.json")
NNI_PATH        = os.path.join(RESULTS_DIR, "nni_sparsity_report.json")
MERGED_PATH     = os.path.join(RESULTS_DIR, "unified_compression_report.json")
METRICS_PATH    = os.path.join(RESULTS_DIR, "metrics.jsonl")

BASE_GGUF       = os.path.join(ARTIFACTS_DIR, "input-f16.gguf")          # original — NEVER touched
FULLQUANT_GGUF  = os.path.join(ARTIFACTS_DIR, "full-q4km.gguf")          # naive Q4_K_M baseline
LAYERWISE_NAME  = "output-layerwise"                                       # no .gguf for ModelPulse

REFERENCE_DOCS  = os.path.join("reference_docs", "agent_reference.md")
EDGE_REF        = os.path.join("reference_docs", "edge_ai_metrics_reference.md")

SERVER_HOST     = "100.81.117.95"
SERVER_PORT     = 8000
SERVER_URL      = f"http://100.81.117.95:{SERVER_PORT}"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NOISE_SCALE      = 0.05
KL_THRESHOLD     = 0.05
BATCH_SIZE       = 20
N_SAMPLES        = 3
ITERATIONS       = 3
METRICS_TIMEOUT  = 120
DEFAULT_SPARSITY = [0.3, 0.4, 0.5]

for d in [ARTIFACTS_DIR, RESULTS_DIR, MODELS_DIR, SHARDS_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_ref(path: str) -> str:
    if not os.path.exists(path):
        print(f"[WARN] Reference doc not found: {path}")
        return ""
    with open(path) as f:
        return f.read().strip()


def wait_for_metrics(path: str, last_seen: float, timeout: int = METRICS_TIMEOUT) -> float:
    """Returns new mtime or -1 on timeout."""
    print(f"\n⏳ Waiting for metrics (timeout={timeout}s)...")
    elapsed = 0
    while elapsed < timeout:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            mtime = os.path.getmtime(path)
            if mtime > last_seen:
                print("✅ Metrics received")
                return mtime
        time.sleep(3)
        elapsed += 3
    print(f"⚠️  No metrics after {timeout}s")
    return -1


def shard_and_upload(model_name: str, gguf_path: str,
                     base_name: str = None, base_shard_dir: str = None):
    """
    Shard a GGUF then upload.
    If base_name + base_shard_dir provided → delta upload (only changed tensors sent).
    """
    shard_dir = os.path.join(SHARDS_DIR, model_name)
    os.makedirs(shard_dir, exist_ok=True)

    # Convert to shards
    manifest = os.path.join(shard_dir, "manifest.json")
    if not os.path.exists(manifest):
        shard_gguf(gguf_path, shard_dir)

    if base_name and base_shard_dir and os.path.exists(base_shard_dir):
        # Delta upload — only sends tensors that changed vs base
        import subprocess
        cmd = [
            "modelpulse", "server", "upload",
            model_name, shard_dir,
            "--base", base_name,
            "--base-dir", base_shard_dir,
            "--server", SERVER_URL,
        ]
        print(f"[Upload] DELTA upload: {model_name} (base={base_name})")
    else:
        import subprocess
        cmd = [
            "modelpulse", "server", "upload",
            model_name, shard_dir,
            "--server", SERVER_URL,
        ]
        print(f"[Upload] FULL upload: {model_name}")

    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Upload failed:\n{result.stderr}")
    print(f"[Upload] Done: {model_name}")
    return shard_dir


def upload_with_retry(model_name: str, gguf_path: str, metrics_path: str,
                      last_seen: float, base_name: str = None,
                      base_shard_dir: str = None) -> tuple[float, str]:
    """Upload + wait for metrics. Re-uploads on timeout. Returns (new_last_seen, shard_dir)."""
    shard_dir = None
    while True:
        shard_dir = shard_and_upload(model_name, gguf_path, base_name, base_shard_dir)
        result = wait_for_metrics(metrics_path, last_seen)
        if result != -1:
            return result, shard_dir
        print("[RETRY] Re-uploading to prompt client...")


def parse_planner_response(response) -> dict:
    raw = ""
    if isinstance(response, dict):
        msgs = response.get("messages", [])
        if msgs:
            raw = getattr(msgs[-1], "content", "") or ""
    elif hasattr(response, "content"):
        raw = response.content or ""
    else:
        raw = str(response)

    clean = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        print("[WARN] Planner response not clean JSON — using defaults")
        parsed = {}

    return {
        "final_quant_assignments": parsed.get("final_quant_assignments", {}),
        "change_required":         bool(parsed.get("change_required", True)),
        "recommended_sparsity":    parsed.get("recommended_sparsity", DEFAULT_SPARSITY),
        "reasoning":               parsed.get("reasoning", "(none)"),
        "raw": raw,
    }


def build_agent_prompt(ctx, metrics_summary, memory_log, ref_docs,
                       edge_ref, iteration, baseline_metrics, fullquant_metrics):
    memory_section    = "\n\n".join(memory_log) if memory_log else "(no memory yet)"
    metrics_section   = json.dumps(metrics_summary, indent=2) if metrics_summary else "(first iteration)"
    baseline_section  = json.dumps(baseline_metrics, indent=2) if baseline_metrics else "(not yet collected)"
    fullquant_section = json.dumps(fullquant_metrics, indent=2) if fullquant_metrics else "(not yet collected)"

    # Cap edge_ref at 2000 chars to stay token-efficient
    edge_ref_capped = edge_ref[:2000] if edge_ref else "(not found)"

    return f"""
You are a quantization planning agent. Iteration {iteration}.

REFERENCE — KL/quant rules (follow strictly):
{ref_docs[:800] if ref_docs else "(not found)"}

EDGE PERFORMANCE REFERENCE (use to judge if metrics are good or bad):
{edge_ref_capped}

ACCUMULATED MEMORY:
{memory_section}

KL SUMMARY:
Top sensitive: {ctx['kl']['top_sensitive_layers'][:8]}
Borderline:    {ctx['kl']['borderline_layers'][:8]}

NNI SUMMARY:
Compress candidates: {len(ctx['nni']['compress_layers'])}
Keep candidates:     {len(ctx['nni']['keep_layers'])}

BASELINE (input-f16.gguf — unquantized):
{baseline_section}

FULL Q4_K_M (naive full quantization — comparison baseline):
{fullquant_section}

CURRENT LAYERWISE METRICS (your previous iteration result):
{metrics_section}

TASK:
1. Compare layerwise metrics vs baseline and full-quant. Is layerwise better?
2. Based on edge_reference thresholds, judge tokens/sec, RAM, latency.
3. Decide new quant assignments (adjust sensitivity-based).
4. Set recommended_sparsity for next NNI run.
5. Set change_required.

Respond ONLY as valid JSON (no markdown fences):
{{
  "final_quant_assignments": {{"layer_name": "Q4_K", ...}},
  "change_required": true,
  "recommended_sparsity": [0.3, 0.4, 0.5],
  "reasoning": "..."
}}
""".strip()


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

# from setup import ensure_imatrix

# LLAMA_IMATRIX_BIN = os.getenv("LLAMA_IMATRIX_BIN")

# # After export_to_gguf_if_needed(), before any quantize call:
# IMATRIX = ensure_imatrix(
#     model_id=MODEL_ID,
#     f16_gguf=Path("artifacts/input-f16.gguf"),
#     llama_imatrix_bin="./llama.cpp/build/bin/llama-imatrix",  # from env or hardcoded path
# )



ctx_obj = build_context()
target_layers = ctx_obj.linear_layers

print("[DEBUG] Model device:", next(ctx_obj.model.parameters()).device)
print("=" * 70)
print("  CHISELED — KL-NNI FEEDBACK LOOP WITH COMPARISON PIPELINE  ")
print("=" * 70)

reference_docs = load_ref(REFERENCE_DOCS)
edge_ref       = load_ref(EDGE_REF)
print(f"[INFO] Reference docs: {len(reference_docs)} chars | Edge ref: {len(edge_ref)} chars")

# Start ModelPulse server
start_modelpulse_server(
    host=SERVER_HOST,
    port=SERVER_PORT,
    log_dir=RESULTS_DIR,
    shard_dir=MODELS_DIR,
)

# ─────────────────────────────────────────────
# PHASE 1 — KL (cached)
# ─────────────────────────────────────────────
print("\n── PHASE 1 — KL Divergence ──")
if not os.path.exists(KL_PATH):
    kl_tool = make_kl_tool(ctx_obj.model, ctx_obj.processor, ctx_obj.images, ctx_obj.IMAGE_DIR)
    kl_tool.invoke({
        "layer_names":    target_layers,
        "noise_scale":    NOISE_SCALE,
        "n_samples":      N_SAMPLES,
        "bits_threshold": KL_THRESHOLD,
        "batch_size":     BATCH_SIZE,
        "output_path":    KL_PATH,
        "resume":         True,
    })
else:
    print(f"[KL] Cached → {KL_PATH}")


# ─────────────────────────────────────────────
# PHASE 2 — BASELINE BENCHMARK (input-f16)
# ─────────────────────────────────────────────
print("\n── PHASE 2 — Baseline Benchmark (input-f16.gguf) ──")
baseline_metrics   = None
fullquant_metrics  = None
last_seen          = 0.0

if not os.path.exists(BASE_GGUF):
    print(f"[ERROR] Base GGUF not found at {BASE_GGUF}. Run setup first.")
    sys.exit(1)

last_seen, base_shard_dir = upload_with_retry(
    model_name="baseline-f16",
    gguf_path=BASE_GGUF,
    metrics_path=METRICS_PATH,
    last_seen=last_seen,
)
raw_baseline = run_metrics_agent(METRICS_PATH, edge_ref)
baseline_metrics = json.loads(raw_baseline) if isinstance(raw_baseline, str) else raw_baseline
os.remove(METRICS_PATH)
print(f"[BASELINE] {json.dumps(baseline_metrics, indent=2)}")


# ─────────────────────────────────────────────
# PHASE 3 — FULL QUANTIZATION BENCHMARK (Q4_K_M all layers)
# ─────────────────────────────────────────────
print("\n── PHASE 3 — Full Q4_K_M Benchmark ──")

if not os.path.exists(FULLQUANT_GGUF):
    print("[FullQuant] Generating naive Q4_K_M GGUF...")
    run_coding_agent(
        merged_report_path=None,           # signals: just do full Q4_K_M
        input_gguf=BASE_GGUF,
        output_gguf=FULLQUANT_GGUF,
        full_quant_mode=True,              # new flag — see coding_agent.py
        dry_run=False,
    )

last_seen, fullquant_shard_dir = upload_with_retry(
    model_name="full-q4km",
    gguf_path=FULLQUANT_GGUF,
    metrics_path=METRICS_PATH,
    last_seen=last_seen,
    base_name="baseline-f16",
    base_shard_dir=base_shard_dir,
)
raw_fullquant = run_metrics_agent(METRICS_PATH, edge_ref)
fullquant_metrics = json.loads(raw_fullquant) if isinstance(raw_fullquant, str) else raw_fullquant
os.remove(METRICS_PATH)
print(f"[FULLQUANT] {json.dumps(fullquant_metrics, indent=2)}")


# ─────────────────────────────────────────────
# MAIN LOOP — LAYERWISE QUANTIZATION
# ─────────────────────────────────────────────
agent_memory_log = []
metrics_summary  = None
current_sparsity = DEFAULT_SPARSITY
prev_layerwise_shard_dir = fullquant_shard_dir   # delta base for first layerwise upload

for it in range(ITERATIONS):
    print("\n" + "=" * 60)
    print(f"  LAYERWISE ITERATION {it + 1} / {ITERATIONS}")
    print("=" * 60)

    # 1. Load cached KL
    with open(KL_PATH) as f:
        kl_report = json.load(f)

    # 2. NNI with planner-controlled sparsity
    print(f"\n[NNI] Sparsity levels: {current_sparsity}")
    disputed_layers = kl_report.get("borderline_layers", target_layers[:20])

    nni_tool = make_sparsity_tool(
        model=ctx_obj.model, processor=ctx_obj.processor,
        images=ctx_obj.images, image_dir=ctx_obj.IMAGE_DIR,
        clean_state=None, device=ctx_obj.device,
    )
    nni_tool.invoke({
        "layer_names":     disputed_layers,
        "sparsity_levels": current_sparsity,
        "js_threshold":    0.1,
        "output_path":     NNI_PATH,
    })

    with open(NNI_PATH) as f:
        nni_report = json.load(f)

    # 3. Build planner context
    compressed_ctx = build_planner_context(kl_report, nni_report, metrics_summary)

    # 4. Build prompt (includes baseline, fullquant, edge ref, memory)
    prompt = build_agent_prompt(
        ctx=compressed_ctx,
        metrics_summary=metrics_summary,
        memory_log=agent_memory_log,
        ref_docs=reference_docs,
        edge_ref=edge_ref,
        iteration=it + 1,
        baseline_metrics=baseline_metrics,
        fullquant_metrics=fullquant_metrics,
    )

    # 5. Planner
    print("\n[PLANNER] Invoking...")
    agent, llm = create_ablation_agent(build_tools(
        model=ctx_obj.model, processor=ctx_obj.processor,
        images=ctx_obj.images, image_dir=ctx_obj.IMAGE_DIR,
    ))
    raw_response = agent.invoke(
        {"messages": [("user", prompt)]},
        config={"configurable": {"thread_id": f"run-{it}"}},
    )
    planner = parse_planner_response(raw_response)

    print(f"[PLANNER] change_required: {planner['change_required']}")
    print(f"[PLANNER] sparsity next:   {planner['recommended_sparsity']}")
    print(f"[PLANNER] reasoning:       {planner['reasoning']}")

    current_sparsity = planner["recommended_sparsity"]

    # 6. Generate new GGUF only if needed
    layerwise_gguf = os.path.join(ARTIFACTS_DIR, f"{LAYERWISE_NAME}-{it}.gguf")
    if planner["change_required"]:
        print(f"\n[GGUF] Generating {layerwise_gguf}...")
        run_coding_agent(
            merged_report_path=MERGED_PATH,
            input_gguf=BASE_GGUF,
            output_gguf=layerwise_gguf,
            dry_run=False,
        )
    else:
        print("\n[GGUF] No change — skipping generation.")
        if not os.path.exists(layerwise_gguf):
            prev = os.path.join(ARTIFACTS_DIR, f"{LAYERWISE_NAME}-{it-1}.gguf")
            layerwise_gguf = prev if os.path.exists(prev) else FULLQUANT_GGUF

    # 7. Upload (delta vs previous layerwise)
    model_name_it = f"{LAYERWISE_NAME}-{it}"
    prev_name     = f"{LAYERWISE_NAME}-{it-1}" if it > 0 else "full-q4km"

    last_seen, prev_layerwise_shard_dir = upload_with_retry(
        model_name=model_name_it,
        gguf_path=layerwise_gguf,
        metrics_path=METRICS_PATH,
        last_seen=last_seen,
        base_name=prev_name,
        base_shard_dir=prev_layerwise_shard_dir,
    )

    # 8. Metrics
    raw_metrics = run_metrics_agent(METRICS_PATH, edge_ref)
    metrics_summary = json.loads(raw_metrics) if isinstance(raw_metrics, str) else raw_metrics
    print(f"\n[METRICS] {json.dumps(metrics_summary, indent=2)}")
    os.remove(METRICS_PATH)
    print("[METRICS] Cleared.")

    # 9. Save + memory
    save_run(
        target_layers=target_layers,
        report_paths={"kl": KL_PATH, "nni": NNI_PATH, "merged": MERGED_PATH},
        agent_response=planner["raw"],
    )

    new_memory = update_agent_memory(
        agent_response=planner["raw"],
        report_paths={"kl": KL_PATH, "nni": NNI_PATH},
        llm=llm,
    )
    if new_memory:
        agent_memory_log.append(f"[Iteration {it + 1}]\n{new_memory}")
        print(f"[MEMORY] Stored ({len(new_memory)} chars)")


# ─────────────────────────────────────────────
# FINAL COMPARISON SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL COMPARISON")
print("=" * 70)
print(f"Baseline (F16):      {json.dumps(baseline_metrics)}")
print(f"Full Q4_K_M:         {json.dumps(fullquant_metrics)}")
print(f"Layerwise (last it): {json.dumps(metrics_summary)}")
print("=" * 70)
print("DONE")

stop_modelpulse_server()