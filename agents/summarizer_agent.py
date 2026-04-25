"""
agents/summarizer_agent.py
==========================
Analyzes runtime metrics from metrics.jsonl using edge_ai_metrics_reference.md
as grounding context so the agent knows what 'good' looks like per hardware class.

Context engineering:
- edge_ref is capped at 1500 chars before injection
- LLM response capped at 512 tokens (summary only, no verbose reasoning)
- Returns a parsed dict (not raw string) so run.py can use fields directly
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "groq").lower()

EDGE_REF_PATH = os.path.join("reference_docs", "edge_ai_metrics_reference.md")

# Max chars of edge reference to inject (keeps token budget tight)
EDGE_REF_MAX_CHARS = 1500


def _load_model():
    if LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct",
            temperature=0, max_tokens=512,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "quant-agent", "X-Title": "QuantAgent"},
        )
    else:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0, max_tokens=512, api_key=GROQ_API_KEY,
        )


def _load_edge_ref(path: str = EDGE_REF_PATH, max_chars: int = EDGE_REF_MAX_CHARS) -> str:
    p = Path(path)
    if not p.exists():
        return "(edge_ai_metrics_reference.md not found)"
    txt = p.read_text(encoding="utf-8").strip()
    # Take the most decision-relevant sections (rules 1-9 are most useful)
    return txt[:max_chars]


def load_latest_metrics(path: str) -> dict:
    with open(path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        raise ValueError(f"metrics.jsonl is empty: {path}")
    return json.loads(lines[-1])


def run_metrics_agent(metrics_path: str, edge_ref: str = None) -> dict:
    """
    Analyze runtime metrics with edge reference context.

    Args:
        metrics_path: path to metrics.jsonl
        edge_ref:     pre-loaded edge reference string (loaded from file if None)

    Returns:
        dict with keys: state, bottlenecks, cpu_pressure, memory_pressure,
                        latency_status, heat_risk, recommendation,
                        tokens_per_sec, ram_used_mb, load_time_s
    """
    metrics = load_latest_metrics(metrics_path)

    if edge_ref is None:
        edge_ref = _load_edge_ref()

    # Cap edge_ref to budget
    edge_ref_snippet = edge_ref[:EDGE_REF_MAX_CHARS]

    prompt = f"""You are a hardware-aware inference diagnostics agent.

Use the edge performance reference below to classify the metrics.
Do NOT hallucinate. If a value is null, note it as unavailable.

EDGE PERFORMANCE REFERENCE (thresholds for CPU/GPU inference):
{edge_ref_snippet}

RUNTIME METRICS:
{json.dumps(metrics, indent=2)}

Return ONLY valid JSON (no markdown, no preamble):
{{
  "state": "fast|stable|slow|critical",
  "bottlenecks": ["list of identified bottlenecks"],
  "cpu_pressure": "low|moderate|high|saturated",
  "memory_pressure": "low|moderate|high|very_high",
  "latency_status": "excellent|good|moderate|high",
  "heat_risk": "low|moderate|high",
  "recommendation": "brief action for planner agent",
  "tokens_per_sec": {metrics.get("tokens_per_sec", "null")},
  "ram_used_mb": {metrics.get("ram_used_mb", "null")},
  "load_time_s": {metrics.get("load_time_s", "null")}
}}"""

    llm = _load_model()
    raw = llm.invoke(prompt).content.strip()

    # Strip fences if LLM added them
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Return a minimal fallback so the pipeline doesn't crash
        print(f"[WARN] summarizer returned non-JSON:\n{raw[:300]}")
        return {
            "state": "unknown",
            "bottlenecks": [],
            "cpu_pressure": "unknown",
            "memory_pressure": "unknown",
            "latency_status": "unknown",
            "heat_risk": "unknown",
            "recommendation": raw[:200],
            "tokens_per_sec": metrics.get("tokens_per_sec"),
            "ram_used_mb":    metrics.get("ram_used_mb"),
            "load_time_s":    metrics.get("load_time_s"),
        }