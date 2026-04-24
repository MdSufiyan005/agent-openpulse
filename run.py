# from __future__ import annotations

# import json
# import os
# import time
# import subprocess

# from setup import build_context
# from planner import build_tools, create_ablation_agent
# from memory import save_run, update_agent_memory
# from summarizer_agent import run_metrics_agent
# from tools.kl_divergence_tool import make_kl_tool
# from tools.sparsity_tool import make_sparsity_tool
# from coding_agent import run_coding_agent
# from tools.deployment.modelpulse_tool import make_modelpulse_tool
# from compress_context import build_planner_context

# # ─────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────
# NOISE_SCALE      = 0.05
# KL_THRESHOLD     = 0.05
# BATCH_SIZE       = 20
# N_SAMPLES        = 3
# ITERATIONS       = 3

# SAVE_DIR         = "."
# KL_PATH          = "results/kl_divergence_report.json"
# NNI_PATH         = "results/nni_sparsity_report.json"
# MERGED_PATH      = "results/unified_compression_report.json"
# METRICS_PATH     = "results/metrics.jsonl"
# REFERENCE_DOCS   = "reference_docs.md"

# GGUF_NAME        = "output-layerwise"   # no .gguf — ModelPulse convention
# BASE_GGUF        = "input-f16.gguf"     # original model, always the base

# # Default sparsity levels — planner can override these each iteration
# DEFAULT_SPARSITY = [0.3, 0.4, 0.5]

# os.makedirs("results", exist_ok=True)


# # ─────────────────────────────────────────────
# # CONTEXT
# # ─────────────────────────────────────────────
# ctx = build_context()
# target_layers = ctx.linear_layers

# print("[DEBUG] Model device:", next(ctx.model.parameters()).device)
# print("=" * 70)
# print("  MODEL-PULSE + KL-NNI FEEDBACK LOOP PIPELINE  ")
# print("=" * 70)


# # ─────────────────────────────────────────────
# # LOAD REFERENCE DOCS (once, reused every iter)
# # ─────────────────────────────────────────────
# def load_reference_docs(path: str) -> str:
#     if not os.path.exists(path):
#         print(f"[WARN] reference_docs.md not found at '{path}' — skipping.")
#         return ""
#     with open(path, "r") as f:
#         return f.read().strip()

# reference_docs = load_reference_docs(REFERENCE_DOCS)
# print(f"[INFO] Reference docs loaded: {len(reference_docs)} chars")


# # ─────────────────────────────────────────────
# # WAIT FOR NEW CLIENT METRICS
# # ─────────────────────────────────────────────
# METRICS_TIMEOUT_S = 120   # seconds before re-uploading model to prompt client

# def wait_for_metrics(path: str, last_seen: float, timeout: int = METRICS_TIMEOUT_S) -> float:
#     """
#     Wait for a new metrics.jsonl. Returns the file's mtime when received.
#     Returns -1 if timed out (caller should re-upload the model).
#     """
#     print(f"\n⏳ Waiting for NEW metrics.jsonl (timeout: {timeout}s)...")
#     elapsed = 0
#     while elapsed < timeout:
#         if os.path.exists(path) and os.path.getsize(path) > 0:
#             modified_time = os.path.getmtime(path)
#             if modified_time > last_seen:
#                 print("✅ NEW metrics received")
#                 return modified_time
#         time.sleep(3)
#         elapsed += 3
#     print(f"⚠️  No metrics after {timeout}s — signalling re-upload.")
#     return -1


# def upload_to_modelpulse(model_name: str, model_path: str):
#     modelpulse = make_modelpulse_tool()
#     modelpulse.invoke({"model_name": model_name, "model_shard_dir": model_path})
#     print(f"[ModelPulse] Uploaded: {model_name}")


# def upload_with_retry(model_name: str, model_path: str, metrics_path: str,
#                       last_seen: float) -> float:
#     """
#     Upload model, wait for metrics. If timeout hits, re-upload and wait again.
#     Loops until metrics arrive. Returns the new last_seen mtime.
#     """
#     while True:
#         upload_to_modelpulse(model_name, model_path)
#         print("\n[WAITING] Blocking until client pushes new metrics.jsonl...")
#         result = wait_for_metrics(metrics_path, last_seen)
#         if result != -1:
#             return result
#         print("[RETRY] Re-uploading model to trigger client again...")


# # ─────────────────────────────────────────────
# # PARSE PLANNER RESPONSE
# # ─────────────────────────────────────────────
# def parse_planner_response(response) -> dict:
#     """
#     Safely extract structured fields from the planner agent response.
#     The planner is prompted to return JSON, but we handle raw text gracefully.
#     Expected fields:
#       - final_quant_assignments  (dict)
#       - change_required          (bool)
#       - recommended_sparsity     (list[float])
#       - reasoning                (str)
#     """
#     # LangGraph agents return a dict with a 'messages' key
#     raw = ""
#     if isinstance(response, dict):
#         messages = response.get("messages", [])
#         if messages:
#             last = messages[-1]
#             raw = getattr(last, "content", "") or ""
#     elif hasattr(response, "content"):
#         raw = response.content or ""
#     else:
#         raw = str(response)

#     # Strip markdown fences if the LLM wrapped the JSON
#     clean = raw.strip()
#     for fence in ["```json", "```"]:
#         clean = clean.replace(fence, "")
#     clean = clean.strip()

#     try:
#         parsed = json.loads(clean)
#     except json.JSONDecodeError:
#         print("[WARN] Planner response was not clean JSON — using defaults.")
#         parsed = {}

#     return {
#         "final_quant_assignments":  parsed.get("final_quant_assignments", {}),
#         "change_required":          bool(parsed.get("change_required", True)),
#         "recommended_sparsity":     parsed.get("recommended_sparsity", DEFAULT_SPARSITY),
#         "reasoning":                parsed.get("reasoning", "(no reasoning provided)"),
#         "raw": raw,
#     }


# # ─────────────────────────────────────────────
# # BUILD PLANNER PROMPT
# # ─────────────────────────────────────────────
# def build_agent_prompt(
#     compressed_context: dict,
#     metrics_summary: dict | None,
#     memory_log: list[str],
#     ref_docs: str,
#     iteration: int,
# ) -> str:

#     memory_section = "\n\n".join(memory_log) if memory_log else "(no memory yet)"

#     metrics_section = json.dumps(metrics_summary, indent=2) if metrics_summary else "(no client metrics yet — first iteration)"

#     ref_section = ref_docs if ref_docs else "(no reference docs provided)"

#     return f"""
# You are a quantization planning agent operating in a closed feedback loop.
# This is iteration {iteration}.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REFERENCE DOCS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# {ref_section}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACCUMULATED AGENT MEMORY (all prior iterations)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# {memory_section}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KL DIVERGENCE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Top sensitive layers (protect these):
# {compressed_context['kl']['top_sensitive_layers'][:10]}

# Borderline layers (candidates for compression):
# {compressed_context['kl']['borderline_layers'][:10]}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NNI SPARSITY SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compress candidates: {len(compressed_context['nni']['compress_layers'])}
# Keep candidates:     {len(compressed_context['nni']['keep_layers'])}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLIENT RUNTIME METRICS (from ModelPulse)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# {metrics_section}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOUR TASK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Review the client metrics above. Is the model fast enough? RAM usage acceptable?
#    Is perplexity (if available) within acceptable range?
# 2. Cross-reference with KL + NNI reports and your memory of previous iterations.
# 3. Decide the layer-wise quantization strategy for this iteration.
# 4. Decide the sparsity levels NNI should test next iteration (pick from 0.1–0.7).
# 5. Decide if a new GGUF needs to be generated (change_required).

# IMPORTANT:
# - The base model is always input-f16.gguf — never modify that.
# - If metrics are already good (fast, low RAM, acceptable quality), you may
#   reduce aggression. If quality is degrading, protect more layers.
# - Use your memory of prior iterations to avoid repeating mistakes.

# OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences, no preamble:
# {{
#   "final_quant_assignments": {{"layer_name": "Q4_K", ...}},
#   "change_required": true,
#   "recommended_sparsity": [0.3, 0.4, 0.5],
#   "reasoning": "Brief explanation of your decision based on metrics + memory"
# }}
# """.strip()


# # ─────────────────────────────────────────────
# # PHASE 1 — KL DIVERGENCE (CACHED)
# # ─────────────────────────────────────────────
# print("\n── PHASE 1 — KL Divergence (cached after first run) ──")

# if not os.path.exists(KL_PATH):
#     print("[KL] Running KL divergence analysis...")
#     kl_tool = make_kl_tool(ctx.model, ctx.processor, ctx.images, ctx.IMAGE_DIR)
#     kl_tool.invoke({
#         "layer_names":   target_layers,
#         "noise_scale":   NOISE_SCALE,
#         "n_samples":     N_SAMPLES,
#         "bits_threshold": KL_THRESHOLD,
#         "batch_size":    BATCH_SIZE,
#         "output_path":   KL_PATH,
#         "resume":        True,
#     })
# else:
#     print("[KL] Using cached report →", KL_PATH)


# # ─────────────────────────────────────────────
# # MAIN LOOP
# # ─────────────────────────────────────────────
# last_seen        = 0.0          # timestamp of last metrics file we consumed
# agent_memory_log = []           # grows across iterations — injected into every prompt
# metrics_summary  = None         # None on iter 0, populated from iter 1 onward
# current_sparsity = DEFAULT_SPARSITY  # planner updates this each iteration

# for it in range(ITERATIONS):

#     print("\n" + "=" * 60)
#     print(f"  LOOP ITERATION {it + 1} / {ITERATIONS}")
#     print("=" * 60)

#     # ── 1. LOAD KL (always static / cached) ───────────────────
#     with open(KL_PATH) as f:
#         kl_report = json.load(f)

#     # ── 2. NNI — run with sparsity levels decided by planner ───
#     print(f"\n[NNI] Running with sparsity levels: {current_sparsity}")
#     disputed_layers = kl_report.get("borderline_layers", target_layers[:5])

#     nni_tool = make_sparsity_tool(
#         model=ctx.model,
#         processor=ctx.processor,
#         images=ctx.images,
#         image_dir=ctx.IMAGE_DIR,
#         clean_state=None,
#         device=ctx.device,
#     )
#     nni_tool.invoke({
#         "layer_names":    disputed_layers,
#         "sparsity_levels": current_sparsity,
#         "js_threshold":   0.1,
#         "output_path":    NNI_PATH,
#     })

#     with open(NNI_PATH) as f:
#         nni_report = json.load(f)

#     # ── 3. BUILD PLANNER CONTEXT ───────────────────────────────
#     compressed_context = build_planner_context(
#         kl_report,
#         nni_report,
#         metrics_summary,   # None on first iter — planner is aware of this
#     )

#     # ── 4. BUILD PLANNER PROMPT (metrics + memory + ref docs) ──
#     prompt = build_agent_prompt(
#         compressed_context=compressed_context,
#         metrics_summary=metrics_summary,
#         memory_log=agent_memory_log,
#         ref_docs=reference_docs,
#         iteration=it + 1,
#     )

#     # ── 5. INVOKE PLANNER AGENT ────────────────────────────────
#     print("\n[PLANNER] Invoking agent...")
#     agent, llm = create_ablation_agent(build_tools(
#         model=ctx.model,
#         processor=ctx.processor,
#         images=ctx.images,
#         image_dir=ctx.IMAGE_DIR,
#     ))

#     raw_response = agent.invoke(
#         {"messages": [("user", prompt)]},
#         config={"configurable": {"thread_id": f"run-{it}"}},
#     )

#     planner = parse_planner_response(raw_response)

#     print(f"\n[PLANNER] change_required      : {planner['change_required']}")
#     print(f"[PLANNER] recommended_sparsity : {planner['recommended_sparsity']}")
#     print(f"[PLANNER] reasoning            : {planner['reasoning']}")

#     # Update sparsity for the NEXT iteration based on planner's recommendation
#     current_sparsity = planner["recommended_sparsity"]

#     # ── 6. CODING AGENT — only if planner says change is needed ─
#     if planner["change_required"]:
#         print(f"\n[GGUF] Generating compressed model variant: {GGUF_NAME}-{it}")
#         run_coding_agent(
#             merged_report_path=MERGED_PATH,
#             input_gguf=BASE_GGUF,           # always compress from the original
#             output_gguf=f"{GGUF_NAME}-{it}",
#             dry_run=False,
#         )
#     else:
#         print("\n[GGUF] Planner says no change needed — skipping GGUF generation.")

#     # -- 7. MODELPULSE UPLOAD + WAIT (auto-retries if client missed it) --
#     print(f"\n[ModelPulse] Uploading: {GGUF_NAME}-{it}")
#     last_seen = upload_with_retry(
#         model_name=f"{GGUF_NAME}-{it}",
#         model_path=f"./models-storage/{GGUF_NAME}-{it}",
#         metrics_path=METRICS_PATH,
#         last_seen=last_seen,
#     )

#     # ── 9. SUMMARIZE METRICS ───────────────────────────────────
#     metrics_summary = run_metrics_agent(METRICS_PATH)
#     print(f"\n[METRICS] Summary: {json.dumps(metrics_summary, indent=2)}")

#     # Clear the file so next iteration starts fresh
#     os.remove(METRICS_PATH)
#     print("[METRICS] Consumed and cleared.")

#     # ── 10. SAVE RUN STATE ─────────────────────────────────────
#     save_run(
#         target_layers=target_layers,
#         report_paths={
#             "kl":     KL_PATH,
#             "nni":    NNI_PATH,
#             "merged": MERGED_PATH,
#         },
#         agent_response=planner["raw"],
#     )

#     # ── 11. UPDATE AGENT MEMORY ────────────────────────────────
#     # update_agent_memory() returns new insights based on LLM reflection
#     # on the agent's response. We accumulate these across iterations.
#     new_memory = update_agent_memory(
#         agent_response=planner["raw"],
#         report_paths={"kl": KL_PATH, "nni": NNI_PATH},
#         llm=llm,
#     )

#     if new_memory:
#         iteration_memory = f"[Iteration {it + 1}]\n{new_memory}"
#         agent_memory_log.append(iteration_memory)
#         print(f"\n[MEMORY] New insight stored ({len(new_memory)} chars)")
#     else:
#         print("\n[MEMORY] No new insights returned this iteration.")


# print("\n" + "=" * 70)
# print("  DONE — FULL MODELPULSE FEEDBACK LOOP COMPLETE  ")
# print("=" * 70)
from __future__ import annotations

import atexit
import json
import os
import time

from setup import build_context
from agents.planner import build_tools, create_ablation_agent
from agents.memory import save_run, update_agent_memory
from agents.summarizer_agent import run_metrics_agent
from tools.kl_divergence_tool import make_kl_tool
from tools.sparsity_tool import make_sparsity_tool
from agents.coding_agent import run_coding_agent
from tools.deployment.modelpulse_tool import (
    METRICS_JSONL_PATH,
    upload_model_to_server,
    start_modelpulse_server,
    stop_modelpulse_server,
)
from tools.compress_context import build_planner_context

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NOISE_SCALE      = 0.05
KL_THRESHOLD     = 0.05
BATCH_SIZE       = 20
N_SAMPLES        = 3
ITERATIONS       = 3

SERVER_HOST      = "0.0.0.0"
SERVER_PORT      = 8000
SERVER_URL       = f"http://127.0.0.1:{SERVER_PORT}"
LOG_DIR          = "artifacts/results"          # server writes metrics.jsonl here
SHARD_STORAGE    = "artifacts/models-storage"   # server stores shards here

KL_PATH          = "artifacts/results/kl_divergence_report.json"
NNI_PATH         = "artifacts/results/nni_sparsity_report.json"
MERGED_PATH      = "artifacts/results/unified_compression_report.json"
REFERENCE_DOCS   = "reference_docs/edge_ai_metrics_reference.md"

BASE_GGUF        = "input-f16.gguf"
GGUF_NAME        = "output-layerwise"

# Shard dirs — one per variant so convert is never re-run unnecessarily
SHARD_DIR_BASE   = f"{SHARD_STORAGE}/baseline"
SHARD_DIR_ITER   = f"{SHARD_STORAGE}/{{name}}"

DEFAULT_SPARSITY = [0.3, 0.4, 0.5]
METRICS_TIMEOUT_S = 120

# os.makedirs("artifacts/results", exist_ok=True)
os.makedirs(SHARD_STORAGE, exist_ok=True)


# ─────────────────────────────────────────────
# CONTEXT
# ─────────────────────────────────────────────
ctx = build_context()
target_layers = ctx.linear_layers

print("[DEBUG] Model device:", next(ctx.model.parameters()).device)
print("=" * 70)
print("  MODEL-PULSE + KL-NNI FEEDBACK LOOP PIPELINE  ")
print("=" * 70)


# ─────────────────────────────────────────────
# START SERVER (once)
# ─────────────────────────────────────────────
print("\n── Starting ModelPulse Server ──")
start_modelpulse_server(
    host=SERVER_HOST,
    port=SERVER_PORT,
    log_dir=LOG_DIR,
    shard_dir=SHARD_STORAGE,
    readiness_timeout=20,
)
atexit.register(stop_modelpulse_server)

# METRICS_JSONL_PATH is now set correctly by start_modelpulse_server()
# It equals: results/metrics.jsonl  (server appends every model's results here)
print(f"[INFO] Watching for metrics at: {METRICS_JSONL_PATH}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_reference_docs(path: str) -> str:
    if not os.path.exists(path):
        print(f"[WARN] {path} not found — skipping.")
        return ""
    with open(path) as f:
        return f.read().strip()


def _count_jsonl_lines(path: str) -> int:
    """Count non-empty lines in a JSONL file (0 if missing)."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def wait_for_new_metrics(lines_before: int, timeout: int = METRICS_TIMEOUT_S) -> int:
    """
    The server APPENDS to one metrics.jsonl file (never recreates it).
    We wait until the line count increases beyond lines_before.
    Returns the new line count, or -1 on timeout.

    lines_before: number of lines in the file before we uploaded this model.
    """
    print(f"\n⏳ Waiting for new metrics (have {lines_before} lines, timeout: {timeout}s)...")
    elapsed = 0
    while elapsed < timeout:
        current = _count_jsonl_lines(METRICS_JSONL_PATH)
        if current > lines_before:
            print(f"✅ New metrics received (now {current} lines)")
            return current
        time.sleep(3)
        elapsed += 3
    print(f"⚠️  No new metrics after {timeout}s — will re-upload.")
    return -1


def read_last_n_lines(path: str, n: int) -> list[dict]:
    """Read the last n JSONL lines from path."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return [json.loads(l) for l in lines[-n:]]


def upload_and_wait(model_name: str, input_gguf: str, shard_dir: str) -> list[dict]:
    """
    Upload model, wait for client to post metrics back, return the new entries.
    Retries upload if client doesn't respond within timeout.
    """
    lines_before = _count_jsonl_lines(METRICS_JSONL_PATH)

    while True:
        upload_model_to_server(model_name, input_gguf, shard_dir, SERVER_URL)
        new_count = wait_for_new_metrics(lines_before)
        if new_count != -1:
            new_entries = read_last_n_lines(METRICS_JSONL_PATH, new_count - lines_before)
            return new_entries
        print("[RETRY] Re-uploading to trigger client again...")


def summarize_and_log(entries: list[dict], label: str) -> dict:
    """Write entries to a labelled snapshot file, then summarise via agent."""
    snapshot_path = f"artifacts/results/snapshot_{label}.jsonl"
    with open(snapshot_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    summary = run_metrics_agent(snapshot_path)
    print(f"\n[METRICS:{label}] {json.dumps(summary, indent=2)}")
    return summary


# ─────────────────────────────────────────────
# PARSE PLANNER RESPONSE
# ─────────────────────────────────────────────
def parse_planner_response(response) -> dict:
    raw = ""
    if isinstance(response, dict):
        messages = response.get("messages", [])
        if messages:
            raw = getattr(messages[-1], "content", "") or ""
    elif hasattr(response, "content"):
        raw = response.content or ""
    else:
        raw = str(response)

    clean = raw.strip()
    for fence in ["```json", "```"]:
        clean = clean.replace(fence, "")
    clean = clean.strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        print("[WARN] Planner response not clean JSON — using defaults.")
        parsed = {}

    return {
        "final_quant_assignments": parsed.get("final_quant_assignments", {}),
        "change_required":         bool(parsed.get("change_required", True)),
        "recommended_sparsity":    parsed.get("recommended_sparsity", DEFAULT_SPARSITY),
        "reasoning":               parsed.get("reasoning", "(no reasoning provided)"),
        "raw": raw,
    }


# ─────────────────────────────────────────────
# PLANNER PROMPT
# ─────────────────────────────────────────────
def build_agent_prompt(compressed_context, metrics_summary, memory_log,
                       ref_docs, iteration) -> str:
    memory_section  = "\n\n".join(memory_log) if memory_log else "(no memory yet)"
    metrics_section = json.dumps(metrics_summary, indent=2) if metrics_summary \
                      else "(no client metrics yet — first iteration)"
    ref_section     = ref_docs or "(no reference docs provided)"

    return f"""
You are a quantization planning agent in a closed feedback loop. Iteration {iteration}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCE DOCS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ref_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCUMULATED AGENT MEMORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{memory_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KL DIVERGENCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Top sensitive layers (protect): {compressed_context['kl']['top_sensitive_layers'][:10]}
Borderline layers (candidates): {compressed_context['kl']['borderline_layers'][:10]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NNI SPARSITY SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compress candidates: {len(compressed_context['nni']['compress_layers'])}
Keep candidates:     {len(compressed_context['nni']['keep_layers'])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLIENT RUNTIME METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{metrics_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Decide layer-wise quantization. Base model is always input-f16.gguf — never modify it.
Respond ONLY with valid JSON:
{{
  "final_quant_assignments": {{"layer_name": "Q4_K"}},
  "change_required": true,
  "recommended_sparsity": [0.3, 0.4, 0.5],
  "reasoning": "..."
}}
""".strip()


# ─────────────────────────────────────────────
# PHASE 0 — BASELINE
# ─────────────────────────────────────────────
print("\n── PHASE 0 — Baseline Upload ──")

baseline_entries = upload_and_wait(
    model_name="baseline",
    input_gguf=BASE_GGUF,
    shard_dir=SHARD_DIR_BASE,
)
baseline_metrics = summarize_and_log(baseline_entries, "baseline")

reference_docs = load_reference_docs(REFERENCE_DOCS)
print(f"[INFO] Reference docs: {len(reference_docs)} chars")


# ─────────────────────────────────────────────
# PHASE 1 — KL DIVERGENCE (cached)
# ─────────────────────────────────────────────
print("\n── PHASE 1 — KL Divergence ──")

if not os.path.exists(KL_PATH):
    print("[KL] Running analysis...")
    kl_tool = make_kl_tool(ctx.model, ctx.processor, ctx.images, ctx.IMAGE_DIR)
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
    print("[KL] Using cached →", KL_PATH)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
agent_memory_log = []
metrics_summary  = baseline_metrics   # planner sees baseline on iteration 1
current_sparsity = DEFAULT_SPARSITY

for it in range(ITERATIONS):

    print("\n" + "=" * 60)
    print(f"  ITERATION {it + 1} / {ITERATIONS}")
    print("=" * 60)

    # ── 1. KL (cached) ───────────────────────────────────────
    with open(KL_PATH) as f:
        kl_report = json.load(f)

    # ── 2. NNI ───────────────────────────────────────────────
    print(f"\n[NNI] Sparsity levels: {current_sparsity}")
    disputed_layers = kl_report.get("borderline_layers", target_layers[:5])

    nni_tool = make_sparsity_tool(
        model=ctx.model, processor=ctx.processor,
        images=ctx.images, image_dir=ctx.IMAGE_DIR,
        clean_state=None, device=ctx.device,
    )
    nni_tool.invoke({
        "layer_names":     disputed_layers,
        "sparsity_levels": current_sparsity,
        "js_threshold":    0.1,
        "output_path":     NNI_PATH,
    })

    with open(NNI_PATH) as f:
        nni_report = json.load(f)

    # ── 3. Planner ───────────────────────────────────────────
    compressed_context = build_planner_context(kl_report, nni_report, metrics_summary)
    prompt = build_agent_prompt(
        compressed_context=compressed_context,
        metrics_summary=metrics_summary,
        memory_log=agent_memory_log,
        ref_docs=reference_docs,
        iteration=it + 1,
    )

    print("\n[PLANNER] Invoking...")
    agent, llm = create_ablation_agent(build_tools(
        model=ctx.model, processor=ctx.processor,
        images=ctx.images, image_dir=ctx.IMAGE_DIR,
    ))
    raw_response = agent.invoke(
        {"messages": [("user", prompt)]},
        config={"configurable": {"thread_id": f"run-{it}"}},
    )
    planner = parse_planner_response(raw_response)

    print(f"[PLANNER] change_required      : {planner['change_required']}")
    print(f"[PLANNER] recommended_sparsity : {planner['recommended_sparsity']}")
    print(f"[PLANNER] reasoning            : {planner['reasoning']}")

    current_sparsity = planner["recommended_sparsity"]

    # ── 4. Coding agent ──────────────────────────────────────
    variant_name      = f"{GGUF_NAME}-{it}"
    variant_gguf = f"{variant_name}.gguf"
    variant_shard_dir = SHARD_DIR_ITER.format(name=variant_name)

    if planner["change_required"]:
        print(f"\n[GGUF] Generating: {variant_name}")
        run_coding_agent(
            merged_report_path=MERGED_PATH,
            input_gguf=BASE_GGUF,
            output_gguf=variant_gguf,
            dry_run=False,
        )
    else:
        print("\n[GGUF] No change — reusing baseline shards.")
        variant_gguf      = BASE_GGUF
        variant_shard_dir = SHARD_DIR_BASE

    # ── 5. Upload + wait for metrics ─────────────────────────
    print(f"\n[ModelPulse] Uploading: {variant_name}")
    iter_entries = upload_and_wait(variant_name, variant_gguf, variant_shard_dir)
    metrics_summary = summarize_and_log(iter_entries, f"iter_{it + 1}")

    # ── 6. Save + memory ─────────────────────────────────────
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
        if isinstance(new_memory, dict):
            new_memory = json.dumps(new_memory, indent=2)
        agent_memory_log.append(f"[Iteration {it + 1}]\n{new_memory}")
        print(f"\n[MEMORY] Stored ({len(new_memory)} chars)")
    else:
        print("\n[MEMORY] No new insights.")


print("\n" + "=" * 70)
print("  DONE — MODELPULSE FEEDBACK LOOP COMPLETE  ")
print("=" * 70)