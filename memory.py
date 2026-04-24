# """
# memory.py
# =========
# Persistent agent memory — unchanged architecture from original.

# Two files:
#   memory/run_log.md      — chronological per-run log
#   memory/agent_memory.md — distilled cross-run insights (LLM-generated)

# Removed extractors: _extract_js, _extract_renyi, _extract_conductance, _extract_sparsity
# Kept:               _extract_kl, _extract_merged
# """

# from __future__ import annotations

# import json
# import os
# from datetime import datetime
# from pathlib import Path

# MEMORY_DIR        = Path("memory")
# RUN_LOG_PATH      = MEMORY_DIR / "run_log.md"
# AGENT_MEMORY_PATH = MEMORY_DIR / "agent_memory.md"


# def _ensure_memory_dir():
#     MEMORY_DIR.mkdir(exist_ok=True)


# # ── Load ──────────────────────────────────────────────────────────────────────

# def load_memory() -> str:
#     """
#     Returns all past memory as a string for injection into the system prompt.
#     Includes distilled insights + last 3 run summaries.
#     """
#     _ensure_memory_dir()
#     sections = []

#     if AGENT_MEMORY_PATH.exists():
#         content = AGENT_MEMORY_PATH.read_text(encoding="utf-8").strip()
#         if content:
#             sections.append(f"## What I have learned from past runs\n\n{content}")

#     if RUN_LOG_PATH.exists():
#         content = RUN_LOG_PATH.read_text(encoding="utf-8").strip()
#         if content:
#             runs   = content.split("---")
#             recent = "---".join(runs[-3:]).strip()
#             sections.append(f"## Recent run history (last 3 runs)\n\n{recent}")

#     return "\n\n".join(sections) if sections else "No previous runs recorded yet."


# # ── Save run ──────────────────────────────────────────────────────────────────

# def save_run(
#     target_layers:  list,
#     report_paths:   dict,
#     agent_response: str,
# ):
#     """Appends a run summary to run_log.md."""
#     _ensure_memory_dir()
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
#     findings  = _extract_findings(report_paths)

#     entry = f"""
# ### Run: {timestamp}

# **Layers analysed:** {len(target_layers)}

# **Findings:**
# {findings}

# **Agent conclusion:**
# {agent_response.strip()[:800]}{"..." if len(agent_response) > 800 else ""}

# ---"""

#     with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
#         f.write(entry)

#     print(f"[memory] Run saved → {RUN_LOG_PATH}")


# # ── Update agent memory ───────────────────────────────────────────────────────

# def update_agent_memory(
#     agent_response: str,
#     report_paths:   dict,
#     llm,
# ) -> str:
#     """
#     Distils new insights from this run using the LLM and appends to agent_memory.md.
#     Returns the new insights string.
#     """
#     _ensure_memory_dir()
#     existing = ""
#     if AGENT_MEMORY_PATH.exists():
#         existing = AGENT_MEMORY_PATH.read_text(encoding="utf-8").strip()

#     findings = _extract_findings(report_paths)

#     prompt = f"""You are maintaining a memory file for a neural network quantization agent.

# Previous memory:
# {existing if existing else "None yet."}

# New findings from this run:
# {findings}

# Agent conclusion:
# {agent_response.strip()[:1000]}

# Extract 3-5 key insights for future runs. Focus on:
# - Which layer types are safe vs sensitive for this model
# - KL thresholds that discriminate well
# - Patterns (attention vs MLP, early vs late layers)
# - Quant type assignments that worked

# Write concise bullet points. Do not repeat existing memory. If nothing new, write "No new insights."
# """

#     try:
#         from langchain_core.messages import HumanMessage
#         response     = llm.invoke([HumanMessage(content=prompt)])
#         new_insights = response.content.strip()
#     except Exception as e:
#         new_insights = f"[Could not distil insights: {e}]"

#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
#     entry     = f"\n\n### Insights from {timestamp}\n{new_insights}"

#     with open(AGENT_MEMORY_PATH, "a", encoding="utf-8") as f:
#         if not AGENT_MEMORY_PATH.exists() or AGENT_MEMORY_PATH.stat().st_size == 0:
#             f.write("# Agent Memory — Accumulated Insights\n")
#         f.write(entry)

#     print(f"[memory] Agent memory updated → {AGENT_MEMORY_PATH}")
#     return new_insights


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def _extract_findings(report_paths: dict) -> str:
#     extractors = {
#         "kl":     _extract_kl,
#         "merged": _extract_merged,
#     }
#     lines = []
#     for key, path in report_paths.items():
#         if not path or not os.path.exists(path):
#             continue
#         extractor = extractors.get(key)
#         if not extractor:
#             continue
#         try:
#             with open(path) as f:
#                 data = json.load(f)
#             lines.append(extractor(data))
#         except Exception as e:
#             lines.append(f"- {key}: could not read ({e})")
#     return "\n".join(lines) if lines else "No reports available."


# def _extract_kl(d: dict) -> str:
#     top = d.get("ranked_by_sensitivity", [])[:3]
#     top_str = ", ".join(f"{x['layer']} ({x['kl_bits']:.4f}b)" for x in top)
#     return (
#         f"- KL: {d.get('safe_count', 0)} safe, {d.get('sensitive_count', 0)} sensitive | "
#         f"noise_scale={d.get('noise_scale', '?')} | top sensitive: {top_str}"
#     )


# def _extract_merged(d: dict) -> str:
#     dist = d.get("quant_distribution", {})
#     return (
#         f"- MERGED: COMPRESS={d.get('compress_count', 0)} KEEP={d.get('keep_count', 0)} | "
#         f"quant dist: {dist}"
#     )


"""
memory.py
=========
Persistent agent memory with context-engineering constraints.

Two files:
  memory/run_log.md      — per-run log (chronological)
  memory/agent_memory.md — distilled insights (LLM-generated, across runs)

Context engineering applied:
  - load_memory_compact(): returns at most max_chars of memory text.
    Prioritises agent_memory.md (distilled insights) over raw run log.
    Truncates run log to last 1 run only (not 3) to stay under token budget.
  - _extract_findings(): only reads kl + merged reports (JS/Renyi/conductance removed).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

MEMORY_DIR        = Path("memory")
RUN_LOG_PATH      = MEMORY_DIR / "run_log.md"
AGENT_MEMORY_PATH = MEMORY_DIR / "agent_memory.md"


def _ensure():
    MEMORY_DIR.mkdir(exist_ok=True)


# ── Compact loader (context-budget aware) ─────────────────────────────────────

def load_memory_compact(max_chars: int = 1600) -> str:
    """
    Returns memory string within max_chars budget.

    Priority (Anthropic context engineering — highest signal first):
      1. Distilled agent insights (agent_memory.md) — most token-efficient
      2. Last 1 run summary only from run_log.md — recency without bulk

    Rationale: agent_memory.md is LLM-distilled → high signal/token ratio.
    Raw run logs are verbose; only last run adds marginal value.
    """
    _ensure()
    parts = []
    budget = max_chars

    # 1. Distilled insights — highest priority
    if AGENT_MEMORY_PATH.exists():
        txt = AGENT_MEMORY_PATH.read_text(encoding="utf-8").strip()
        if txt:
            snippet = txt[-budget:]   # take the most recent part if long
            parts.append(f"INSIGHTS:\n{snippet}")
            budget -= len(snippet)

    # 2. Last run summary — only if budget remains
    if budget > 200 and RUN_LOG_PATH.exists():
        txt = RUN_LOG_PATH.read_text(encoding="utf-8").strip()
        if txt:
            runs    = txt.split("---")
            last    = runs[-1].strip() if runs else ""
            snippet = last[:budget]
            if snippet:
                parts.append(f"LAST RUN:\n{snippet}")

    return "\n\n".join(parts) if parts else ""


# Legacy full loader (kept for direct memory inspection scripts)
def load_memory() -> str:
    return load_memory_compact(max_chars=4000)


# ── Save run ──────────────────────────────────────────────────────────────────

def save_run(target_layers: list, report_paths: dict, agent_response: str):
    _ensure()
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M")
    findings = _extract_findings(report_paths)
    entry = (
        f"\n### Run: {ts}\n"
        f"**Layers:** {len(target_layers)}\n"
        f"**Findings:**\n{findings}\n"
        f"**Agent:**\n{agent_response.strip()[:600]}"
        f"{'...' if len(agent_response) > 600 else ''}\n\n---"
    )
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[memory] Run saved → {RUN_LOG_PATH}")


# ── Update distilled memory ───────────────────────────────────────────────────

def update_agent_memory(agent_response: str, report_paths: dict, llm) -> str:
    _ensure()
    existing = ""
    if AGENT_MEMORY_PATH.exists():
        existing = AGENT_MEMORY_PATH.read_text(encoding="utf-8").strip()

    findings = _extract_findings(report_paths)
    prompt = (
        f"Maintain a memory file for a quantization agent.\n\n"
        f"Existing memory:\n{existing[-800:] if existing else 'None'}\n\n"
        f"New findings:\n{findings}\n\n"
        f"Agent conclusion:\n{agent_response.strip()[:600]}\n\n"
        f"Extract 3-5 NEW bullet-point insights about layer sensitivity patterns, "
        f"good KL thresholds, or quant assignments that worked. "
        f"Skip anything already in existing memory. Be concise — max 300 words."
    )
    try:
        from langchain_core.messages import HumanMessage
        resp         = llm.invoke([HumanMessage(content=prompt)])
        new_insights = resp.content.strip()
    except Exception as e:
        new_insights = f"[distillation failed: {e}]"

    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n\n### {ts}\n{new_insights}"
    with open(AGENT_MEMORY_PATH, "a", encoding="utf-8") as f:
        if not AGENT_MEMORY_PATH.exists() or AGENT_MEMORY_PATH.stat().st_size == 0:
            f.write("# Agent Memory\n")
        f.write(entry)

    print(f"[memory] Updated → {AGENT_MEMORY_PATH}")
    return new_insights


# ── Findings extractor ────────────────────────────────────────────────────────

def _extract_findings(report_paths: dict) -> str:
    lines = []
    for key, path in report_paths.items():
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            lines.append(f"- {key}: unreadable ({e})")
            continue

        if key == "kl":
            top3 = [
                f"{x['layer']}={x['kl_bits']:.4f}b"
                for x in d.get("ranked_by_sensitivity", [])[:3]
            ]
            lines.append(
                f"- KL: safe={d.get('safe_count',0)} sensitive={d.get('sensitive_count',0)} "
                f"noise={d.get('noise_scale','?')} top3={top3}"
            )
        elif key == "merged":
            lines.append(
                f"- MERGED: compress={d.get('compress_count',0)} keep={d.get('keep_count',0)} "
                f"dist={d.get('quant_distribution',{})}"
            )
    return "\n".join(lines) if lines else "No reports."