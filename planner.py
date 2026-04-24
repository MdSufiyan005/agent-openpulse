# # """
# # planner.py
# # ==========
# # Agent factory — KL-only pipeline.

# # Removed: make_js_tool, make_renyi_tool, make_conductance_tool, make_sparsity_tool
# # Kept:    make_kl_tool, merge_analysis_reports, agent memory injection
# # Added:   persistent MemorySaver with configurable thread_id
# # """

# # from __future__ import annotations

# # import os
# # import pathlib

# # from dotenv import load_dotenv
# # from langchain_groq import ChatGroq
# # from langchain_core.messages import SystemMessage
# # from langgraph.prebuilt import create_react_agent
# # from langgraph.checkpoint.memory import MemorySaver

# # from tools.kl_divergence_tool import make_kl_tool
# # from tools.merge_report_tool   import merge_analysis_reports
# # from memory import load_memory

# # load_dotenv()
# # GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# # def build_tools(model, processor, images: list, image_dir: str) -> list:
# #     """Returns the two tools available to the agent: KL analysis + merge."""
# #     return [
# #         make_kl_tool(model, processor, images, image_dir),
# #         merge_analysis_reports,
# #     ]


# # def _load_system_prompt(md_path: str) -> str:
# #     p = pathlib.Path(md_path)
# #     if not p.exists():
# #         # Fallback inline prompt if reference doc missing
# #         return _DEFAULT_SYSTEM_PROMPT
# #     return p.read_text(encoding="utf-8")


# # def create_ablation_agent(
# #     tools: list,
# #     md_path: str = "reference_docs/agent_reference.md",
# #     thread_id: str = "ablation-main",
# #     model_name: str = "llama-3.3-70b-versatile",
# # ):
# #     """
# #     Creates the LangGraph ReAct agent with:
# #     - System prompt loaded from md_path + past memory injected
# #     - MemorySaver for within-session persistent context (handles multi-turn)
# #     - Returns (agent, llm) tuple
# #     """
# #     base_prompt  = _load_system_prompt(md_path)
# #     past_memory  = load_memory()
# #     full_prompt  = (
# #         f"{base_prompt}\n\n"
# #         f"<past_memory>\n{past_memory}\n</past_memory>"
# #     )

# #     llm    = ChatGroq(
# #         model=model_name,
# #         temperature=0,
# #         max_tokens=4096,
# #         api_key=GROQ_API_KEY,
# #     )
# #     memory = MemorySaver()

# #     agent = create_react_agent(
# #         model        = llm,
# #         tools        = tools,
# #         checkpointer = memory,
# #         prompt       = SystemMessage(content=full_prompt),
# #     )
# #     return agent, llm


# # # ── Default system prompt (used if reference_docs/agent_reference.md missing) ─

# # _DEFAULT_SYSTEM_PROMPT = """\
# # You are a neural network compression analysis agent specializing in layerwise quantization.

# # Your goal: analyse each layer's sensitivity using KL divergence, then generate
# # a complete llama-quantize command that assigns optimal quant types per layer.

# # ## Tools available
# # 1. `kl_divergence_analysis` — perturbs each layer, measures output distribution shift (KL bits)
# # 2. `merge_analysis_reports` — converts KL report into quant_assignments + llama-quantize command

# # ## KL interpretation
# # | KL (bits)   | Sensitivity | Quant type |
# # |-------------|-------------|------------|
# # | < 0.01      | very low    | Q4_K_M     |
# # | 0.01 – 0.05 | low         | Q5_K_M     |
# # | 0.05 – 0.20 | moderate    | Q6_K       |
# # | > 0.20      | high        | Q8_0       |

# # ## Workflow
# # 1. Run `kl_divergence_analysis` on ALL layers (use microbatches via batch_size param)
# # 2. Run `merge_analysis_reports` to get the unified plan
# # 3. Review quant_assignments — adjust any suspicious layers
# # 4. Output final JSON with `final_quant_assignments` and `llama_quantize_cmd`

# # ## Output format
# # ```json
# # {
# #   "final_quant_assignments": {"tensor_name": "QUANT_TYPE", ...},
# #   "llama_quantize_cmd": "full command string",
# #   "flags": ["layer: reason"],
# #   "summary": "one paragraph"
# # }
# # ```

# # ## Rules
# # - Vision projector layers are always F16 — never quantize them
# # - First and last transformer layers: prefer Q6_K minimum
# # - When KL=0 for many layers: noise_scale is too low — increase to 0.05+
# # - Always validate that at least 20-30% of layers are COMPRESS before trusting results
# # """


# """
# planner.py  — context-engineered, OpenRouter-capable
# =====================================================
# Context engineering fixes for 413 (TPM exceeded):
#   - System prompt capped at ~300 tokens (compact rules only)
#   - Memory injected as compact summary, capped at 400 tokens
#   - Agent max_tokens reduced to 2048 (synthesis rarely needs more)
#   - OpenRouter added as drop-in fallback via LLM_PROVIDER env var

# Set in .env:
#   LLM_PROVIDER=groq        GROQ_API_KEY=gsk_...
#   LLM_PROVIDER=openrouter  OPENROUTER_API_KEY=sk-or-...
# """

# from __future__ import annotations
# import os, pathlib
# from dotenv import load_dotenv
# from langchain_core.messages import SystemMessage
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
# from tools.kl_divergence_tool import make_kl_tool
# from tools.merge_report_tool   import merge_analysis_reports
# from memory import load_memory_compact

# load_dotenv()

# GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "groq").lower()

# MAX_MEMORY_CHARS = 1600   # ~400 tokens


# def build_tools(model, processor, images: list, image_dir: str) -> list:
#     return [
#         make_kl_tool(model, processor, images, image_dir),
#         merge_analysis_reports,
#     ]


# def _make_llm(provider: str, model_name: str | None = None):
#     if provider == "openrouter":
#         if not OPENROUTER_API_KEY:
#             raise EnvironmentError("OPENROUTER_API_KEY not set")
#         from langchain_openai import ChatOpenAI
#         return ChatOpenAI(
#             model       = model_name or "meta-llama/llama-3.3-70b-instruct",
#             temperature = 0,
#             max_tokens  = 2048,
#             api_key     = OPENROUTER_API_KEY,
#             base_url    = "https://openrouter.ai/api/v1",
#             default_headers={"HTTP-Referer": "quant-agent", "X-Title": "QuantAgent"},
#         )
#     else:
#         if not GROQ_API_KEY:
#             raise EnvironmentError("GROQ_API_KEY not set")
#         from langchain_groq import ChatGroq
#         return ChatGroq(
#             model=model_name or "llama-3.3-70b-versatile",
#             temperature=0, max_tokens=2048, api_key=GROQ_API_KEY,
#         )


# def _build_system_prompt(md_path: str) -> str:
#     """Load compact system prompt + inject capped memory summary."""
#     p = pathlib.Path(md_path)
#     base = p.read_text(encoding="utf-8") if p.exists() else _COMPACT_PROMPT
#     mem  = load_memory_compact(max_chars=MAX_MEMORY_CHARS)
#     if mem:
#         base += f"\n<memory>\n{mem}\n</memory>"
#     total_tokens = len(base) // 4
#     print(f"[planner] System prompt: ~{total_tokens} tokens")
#     if total_tokens > 800:
#         print(f"[planner] WARNING: system prompt {total_tokens} tokens — consider trimming agent_reference.md")
#     return base


# def create_ablation_agent(
#     tools:      list,
#     md_path:    str = "reference_docs/agent_reference.md",
#     thread_id:  str = "ablation-main",
#     provider:   str | None = None,
#     model_name: str | None = None,
# ):
#     chosen = provider or LLM_PROVIDER
#     llm    = _make_llm(chosen, model_name)
#     prompt = _build_system_prompt(md_path)
#     mem    = MemorySaver()
#     print(f"[planner] Provider: {chosen}")
#     agent  = create_react_agent(
#         model=llm, tools=tools, checkpointer=mem,
#         prompt=SystemMessage(content=prompt),
#     )
#     return agent, llm


# _COMPACT_PROMPT = """\
# You are a layerwise quantization agent for transformer models.

# KL bits → quant:
#   <0.01=Q4_K_M  0.01-0.05=Q5_K_M  0.05-0.20=Q6_K  >0.20=Q8_0

# Rules: first/last 2 layers ≥Q6_K. lm_head ≥Q6_K. Vision=F16 always.
# If <20% COMPRESS: noise_scale too low.

# Output exactly one JSON block:
# {"final_quant_assignments":{...},"llama_quantize_cmd":"...","flags":[],"summary":"..."}
# """


"""
planner.py — Multi-signal ablation agent
=========================================
Registers KL + Conductance + Sparsity tools.

Tool usage philosophy (baked into system prompt):
  - KL: always run first on ALL layers
  - Conductance: only for borderline layers (within ±20% of threshold) — short list
  - Sparsity: only for layers still disputed after conductance — very short list (<5)
  - merge_analysis_reports: final step, always

Context budget:
  - System prompt ≤ 800 tokens
  - Memory summary ≤ 400 tokens
  - Agent max_tokens = 2048
"""

from __future__ import annotations
import os, pathlib
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools.kl_divergence_tool  import make_kl_tool
from tools.conductance_tool    import make_conductance_tool
from tools.sparsity_tool       import make_sparsity_tool
from tools.merge_report_tool   import merge_analysis_reports
from memory import load_memory_compact

load_dotenv()

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "openrouter").lower()
MAX_MEMORY_CHARS   = 1600


def build_tools(model, processor, images: list, image_dir: str) -> list:
    """All four analysis tools. Agent decides which optional ones to invoke."""
    return [
        make_kl_tool(model, processor, images, image_dir),
        make_conductance_tool(model, processor, images, image_dir),
        make_sparsity_tool(model, processor, images, image_dir),
        merge_analysis_reports,
    ]


def _make_llm(provider: str, model_name: str | None = None):
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY not set")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "meta-llama/llama-3.3-70b-instruct",
            temperature=0, max_tokens=2048,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "quant-agent", "X-Title": "QuantAgent"},
        )
    else:
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            temperature=0, max_tokens=2048, api_key=GROQ_API_KEY,
        )


def _build_system_prompt(md_path: str) -> str:
    p = pathlib.Path(md_path)
    base = p.read_text(encoding="utf-8") if p.exists() else _COMPACT_PROMPT
    mem = load_memory_compact(max_chars=MAX_MEMORY_CHARS)
    if mem:
        base += f"\n<memory>\n{mem}\n</memory>"
    total_tokens = len(base) // 4
    print(f"[planner] System prompt: ~{total_tokens} tokens")
    if total_tokens > 800:
        print(f"[planner] WARNING: {total_tokens} tokens — trim agent_reference.md")
    return base


def create_ablation_agent(
    tools:      list,
    md_path:    str = "reference_docs/agent_reference.md",
    thread_id:  str = "ablation-main",
    provider:   str | None = None,
    model_name: str | None = None,
):
    chosen = provider or LLM_PROVIDER
    llm    = _make_llm(chosen, model_name)
    prompt = _build_system_prompt(md_path)
    mem    = MemorySaver()
    print(f"[planner] Provider: {chosen}")
    agent  = create_react_agent(
        model=llm, tools=tools, checkpointer=mem,
        prompt=SystemMessage(content=prompt),
    )
    return agent, llm


# ── Compact fallback system prompt ────────────────────────────────────────────
_COMPACT_PROMPT = """\
You are a layerwise quantization agent for transformer models.

## Tool usage order (STRICT)
1. `kl_divergence_analysis` on ALL layers first. This is your primary signal.
2. `conductance_analysis` ONLY for borderline layers (KL within ±20% of threshold).
   Pass a SHORT list (<20 layers). Skip if KL signal is clear.
3. `sparsity_sweep` ONLY for layers still disputed after step 2. Pass <5 layers.
   Skip entirely if conductance resolved all disputes.
4. `merge_analysis_reports` always last. Pass paths to whichever reports you generated.

## KL → quant mapping
  <0.001=Q4_K_M  0.001-0.01=Q5_K_M  0.01-0.05=Q6_K  0.05-0.2=Q8_0  >0.2=F16

## Hard rules
- First/last 2 layers, lm_head, embed: minimum Q6_K always.
- Vision encoder/connector: F16, never quantize, never include in tool calls.
- If <20% COMPRESS after merge: noise_scale too low — re-run KL at 0.1.
- Do NOT call conductance or sparsity on all layers. Cost is too high.

## Identifying the most important weights (DO NOT TOUCH list)
High-importance = ANY of:
  - KL > 0.2 (top tier sensitivity)
  - Conductance rank top 10% (if run)
  - First/last transformer blocks
  - lm_head, embed_tokens
These stay at Q8_0 or F16. Everything else gets compressed by KL signal.

## Final output (one JSON block only)
{"final_quant_assignments":{...},"llama_quantize_cmd":"...","flags":[],"summary":"...","do_not_touch":["layer: reason"]}
"""