import json
from pathlib import Path
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# from quant_agent_v2.quant_agent.planner import OPENROUTER_API_KEY
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_model(provider: str):
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY not set")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct",
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
            model= "openai/gpt-oss-20b",
            temperature=0, max_tokens=2048, api_key=GROQ_API_KEY,
        )

def load_latest_metrics(path="artifacts/results/metrics.jsonl"):
    with open(path, "r") as f:
        return json.loads(f.readlines()[-1])


def build_metrics_agent(llm):
    SYSTEM_PROMPT = """
You are a hardware-aware inference diagnostics agent.

You analyze runtime metrics from an edge AI system and produce a structured reasoning report.

You DO NOT modify the model.
You ONLY interpret system behavior.

You must identify:

1. Performance state (fast / stable / slow / critical)
2. Bottlenecks (CPU, RAM, latency, throughput)
3. Heat pressure (based on CPU usage)
4. Inference inefficiency causes
5. Recommendation for model planner (quantization / sparsity / layer sensitivity direction)

IMPORTANT RULES:
- Do NOT hallucinate missing values
- Do NOT use thresholds blindly — reason from context
- Prefer qualitative reasoning + approximate classification
- Output MUST be structured JSON only
"""

    llm = load_model("groq")
    return llm, SYSTEM_PROMPT


def run_metrics_agent(llm):
    metrics = load_latest_metrics()

    prompt = f"""
        Analyze the following runtime metrics:

        {json.dumps(metrics, indent=2)}

        Return structured JSON:
        {{
        "state": "",
        "bottlenecks": [],
        "cpu_pressure": "",
        "memory_pressure": "",
        "latency_status": "",
        "heat_risk": "",
        "recommendation": ""
        }}
        """
    llm = load_model("groq")
    return llm.invoke(prompt).content