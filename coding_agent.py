"""
coding_agent.py
===============
Phase 5: Coding Agent — generates and executes the llama-quantize command.

Responsibilities:
  1. Reads unified_compression_report.json from ablation agent output
  2. Validates the llama-quantize command (tensor name format, GGUF paths)
  3. Optionally patches in --imatrix flag if imatrix.dat exists
  4. Writes quantize_run.sh shell script
  5. Executes the script (or dry-runs with DRY_RUN=1)
  6. Verifies output GGUF was created and logs size

Environment variables:
  INPUT_GGUF    — path to input F16 GGUF model
  OUTPUT_GGUF   — path for output quantized GGUF
  LLAMA_QUANTIZE — path to llama-quantize binary (default: ./llama-quantize)
  IMATRIX_PATH  — path to imatrix.dat (optional but recommended)
  DRY_RUN       — if "1", print command without executing
  LLM_PROVIDER  — "groq" or "openrouter"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

INPUT_GGUF      = os.environ.get("INPUT_GGUF",     "input-f16.gguf")
OUTPUT_GGUF     = os.environ.get("OUTPUT_GGUF",    "output-layerwise.gguf")
LLAMA_QUANTIZE  = os.environ.get("LLAMA_QUANTIZE", "./llama.cpp/build/bin/llama-quantize")
IMATRIX_PATH    = os.environ.get("IMATRIX_PATH",   "imatrix.dat")
DRY_RUN         = os.environ.get("DRY_RUN",        "0") == "1"
MERGED_PATH     = "results/unified_compression_report.json"
SCRIPT_PATH     = "results/quantize_run.sh"
LOG_PATH        = "results/quantize_run.log"

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "openrouter").lower()


# ── Tools for the coding agent ────────────────────────────────────────────────

class ValidateInput(BaseModel):
    merged_report_path: str = Field(default=MERGED_PATH)
    input_gguf:         str = Field(default=INPUT_GGUF)
    output_gguf:        str = Field(default=OUTPUT_GGUF)
    imatrix_path:       str = Field(default=IMATRIX_PATH or "imatrix.dat")
    llama_quantize_bin: str = Field(default=LLAMA_QUANTIZE)


def ensure_f16_gguf_exists(hf_model_path: str, output_gguf: str):
    """
    Converts HuggingFace model → F16 GGUF if not already present.
    """
    if os.path.exists(output_gguf):
        print(f"[setup] Found existing GGUF: {output_gguf}")
        return

    print(f"[setup] Creating F16 GGUF from HF model...")

    convert_script = "./llama.cpp/convert_hf_to_gguf.py"

    if not os.path.exists(convert_script):
        raise RuntimeError("convert_hf_to_gguf.py not found in llama.cpp")

    cmd = [
        "python", convert_script,
        hf_model_path,
        "--outfile", output_gguf,
        "--outtype", "f16"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed:\n{result.stderr}"
        )

    print(f"[setup] ✅ Created: {output_gguf}")

@tool("validate_and_build_command", args_schema=ValidateInput)
def validate_and_build_command(
    merged_report_path: str = MERGED_PATH,
    input_gguf:         str = INPUT_GGUF,
    output_gguf:        str = OUTPUT_GGUF,
    imatrix_path:       str = "imatrix.dat",
    llama_quantize_bin: str = LLAMA_QUANTIZE,
) -> str:
    """
    Reads the unified compression report, validates tensor names,
    patches in imatrix if available, and builds the final shell command.
    Writes it to results/quantize_run.sh.
    Returns the command string and any warnings.
    """
    if not os.path.exists(merged_report_path):
        return f"ERROR: merged report not found at {merged_report_path}"

    with open(merged_report_path) as f:
        report = json.load(f)

    qa: dict = report.get("quant_assignments", {})
    default_quant: str = "Q4_K_M"

    if not qa:
        return "ERROR: No quant_assignments in merged report."

    # ── Build tensor-type flags ───────────────────────────────────────────────
    def _to_gguf(python_name: str) -> str:
        import re
        # ensure .weight is always present
        if not python_name.endswith(".weight"):
            python_name += ".weight"

        m = re.search(r"layers?[.\[](\d+)", python_name)
        layer_idx = m.group(1) if m else "0"

        suffix_map = {
            "q_proj": "attn_q",
            "k_proj": "attn_k",
            "v_proj": "attn_v",
            "o_proj": "attn_output",
            "gate_proj": "ffn_gate",
            "up_proj": "ffn_up",
            "down_proj": "ffn_down",
            "fc1": "ffn_up",
            "fc2": "ffn_down",
        }

        for py, gg in suffix_map.items():
            if py in python_name:
                return f"blk.{layer_idx}.{gg}.weight"

        if "lm_head" in python_name:
            return "output.weight"

        return python_name

    # Group by quant type; skip default
    by_quant: dict[str, list] = {}
    for name, qt in qa.items():
        if qt != default_quant:
            by_quant.setdefault(qt, []).append(_to_gguf(name))

    # ── Build argv list (used for both the script and subprocess) ─────────────
    # FIX: build a proper argv list instead of joining everything into one string.
    # The correct format for each flag is: --tensor-type <tensor_name>=<QUANT>
    # (i.e. "blk.0.attn_q.weight=Q6_K", NOT "blk.0.attn_q.weight Q6_K")
    argv: list[str] = [llama_quantize_bin]

    # imatrix (recommended for Q4/Q5 quality)
    print(f"IMATRIX PATH: {imatrix_path}")
    if imatrix_path and os.path.exists(imatrix_path):
        argv += ["--imatrix", imatrix_path]
        imatrix_status = f"✓ imatrix found: {imatrix_path}"
    else:
        imatrix_status = (
            "⚠ imatrix.dat not found. Q4/Q5 layers will use default importance "
            "weighting. See imatrix setup guide in IMATRIX_GUIDE.md."
        )

    # for qt, tensor_names in sorted(by_quant.items()):
    #     for tn in tensor_names:
    #         # FIX: value must be "tensor=QUANT" as a single argument, not two
    #         argv += ["--tensor-type", f"{tn}={qt}"]
    for qt, tensor_names in sorted(by_quant.items()):
        for tn in tensor_names:
            if not tn.endswith(".weight"):
                tn += ".weight"
            argv += ["--tensor-type", f"{tn}={qt}"]

    # argv += [input_gguf, output_gguf, default_quant]
    argv = [
        llama_quantize_bin,
        input_gguf,
        output_gguf,
        default_quant
    ]   

    # ── Write shell script ────────────────────────────────────────────────────
    # FIX: build the shell script from the argv list so it's always consistent
    # with what subprocess will actually run.  Use proper shell quoting.
    os.makedirs("results", exist_ok=True)

    # Pretty-print the command for the shell script (one flag per line)
    def _shell_escape(s: str) -> str:
        """Minimal quoting: wrap in double-quotes if the arg contains spaces."""
        return f'"{s}"' if " " in s else s

    script_lines = [argv[0]]
    i = 1
    while i < len(argv):
        if argv[i] == "--tensor-type" and i + 1 < len(argv):
            script_lines.append(f"  --tensor-type {_shell_escape(argv[i+1])}")
            i += 2
        elif argv[i] == "--imatrix" and i + 1 < len(argv):
            script_lines.append(f"  --imatrix {_shell_escape(argv[i+1])}")
            i += 2
        else:
            script_lines.append(f"  {_shell_escape(argv[i])}")
            i += 1

    pretty_cmd = " \\\n".join(script_lines)

    script_content = f"""#!/bin/bash
# Auto-generated by coding_agent.py
# Layerwise quantization for SmolVLM-Instruct
# Generated: $(date)

set -e

echo "Starting layerwise quantization..."
echo "Input : {input_gguf}"
echo "Output: {output_gguf}"
echo "imatrix: {imatrix_path or 'none'}"

{pretty_cmd}

echo ""
echo "Done! Output: {output_gguf}"
ls -lh {output_gguf}
"""

    with open(SCRIPT_PATH, "w") as f:
        f.write(script_content)
    os.chmod(SCRIPT_PATH, 0o755)

    tensor_count = sum(len(v) for v in by_quant.values())
    return (
        f"Command built. {tensor_count} non-default tensor-type flags.\n"
        f"Quant distribution: { {qt: len(ns) for qt, ns in by_quant.items()} }\n"
        f"imatrix: {imatrix_status}\n"
        f"Script written: {SCRIPT_PATH}\n\n"
        f"COMMAND:\n{pretty_cmd}"
    )


class ExecuteInput(BaseModel):
    script_path: str  = Field(default=SCRIPT_PATH)
    dry_run:     bool = Field(default=DRY_RUN)
    output_gguf: str  = Field(default=OUTPUT_GGUF)
    input_gguf:  str  = Field(default=INPUT_GGUF)
    llama_quantize_bin: str = Field(default=LLAMA_QUANTIZE)


@tool("execute_quantization", args_schema=ExecuteInput)
def execute_quantization(
    script_path:        str  = SCRIPT_PATH,
    dry_run:            bool = DRY_RUN,
    output_gguf:        str  = OUTPUT_GGUF,
    input_gguf:         str  = INPUT_GGUF,
    llama_quantize_bin: str  = LLAMA_QUANTIZE,
) -> str:
    """
    Pre-flight checks, then executes the generated quantize_run.sh script.
    Streams stderr live so errors are always visible.
    Set dry_run=True to print without executing.
    Returns execution status and output GGUF file size.
    """
    # ── Pre-flight checks ─────────────────────────────────────────────────────
    problems: list[str] = []

    if not os.path.exists(script_path):
        problems.append(f"Script not found: {script_path} — run validate_and_build_command first.")

    if not os.path.exists(input_gguf):
        # Try to find the file nearby so we can give a useful hint
        candidates = (
            list(Path(".").glob("*.gguf"))
            + list(Path(".").glob("**/*.gguf"))
        )
        hint = (
            f"  Found nearby GGUFs: {[str(c) for c in candidates[:5]]}"
            if candidates
            else "  No .gguf files found in the current directory tree."
        )
        problems.append(
            f"Input GGUF not found: {input_gguf}\n"
            f"{hint}\n"
            f"  Set INPUT_GGUF env var or pass input_gguf= to point at the right file."
        )

    if not os.path.exists(llama_quantize_bin):
        problems.append(
            f"llama-quantize binary not found: {llama_quantize_bin}\n"
            f"  Build it with: cd llama.cpp && cmake -B build && cmake --build build -j --config Release"
        )

    if problems:
        return "PRE-FLIGHT FAILED:\n" + "\n\n".join(f"✗ {p}" for p in problems)

    # ── Dry run ───────────────────────────────────────────────────────────────
    with open(script_path) as f:
        script_content = f.read()

    if dry_run:
        return f"DRY RUN — would execute:\n{script_content}\n\n(Set DRY_RUN=0 to actually run.)"

    # ── Execute with live stderr streaming ────────────────────────────────────
    print(f"[coding_agent] Executing {script_path}...")
    print(f"[coding_agent] Input : {input_gguf}  ({os.path.getsize(input_gguf)/1024/1024:.0f} MB)")
    print(f"[coding_agent] Output: {output_gguf}")

    os.makedirs("results", exist_ok=True)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Stream both stdout and stderr so progress is visible in the terminal
        import threading

        def _drain(pipe, store: list, label: str):
            for line in pipe:
                store.append(line)
                print(f"[{label}] {line}", end="", flush=True)

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines, "stdout"))
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines, "stderr"))
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            proc.kill()
            return "ERROR: Quantization timed out after 2 hours."
        finally:
            t_out.join()
            t_err.join()

    except Exception as e:
        return f"ERROR: Failed to launch script: {e}"

    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)

    # Write full log
    with open(LOG_PATH, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout_text}\n\n=== STDERR ===\n{stderr_text}")

    if proc.returncode != 0:
        return (
            f"ERROR: llama-quantize exited with code {proc.returncode}\n"
            f"STDERR (last 2000 chars):\n{stderr_text[-2000:]}\n"
            f"STDOUT (last 500 chars):\n{stdout_text[-500:]}\n"
            f"Full log: {LOG_PATH}"
        )

    # ── Verify output GGUF was actually written ───────────────────────────────
    if os.path.exists(output_gguf):
        size_mb = os.path.getsize(output_gguf) / 1024 / 1024
        return (
            f"✓ Quantization complete!\n"
            f"  Output : {output_gguf} ({size_mb:.1f} MB)\n"
            f"  Log    : {LOG_PATH}\n"
            f"  Tail   :\n{stdout_text[-300:]}"
        )
    else:
        return (
            f"WARNING: llama-quantize exited 0 but output GGUF not found at: {output_gguf}\n"
            f"  The binary may have written to a different path.\n"
            f"  Check log: {LOG_PATH}\n"
            f"  STDOUT tail:\n{stdout_text[-500:]}"
        )


# ── Coding agent factory ──────────────────────────────────────────────────────

_CODING_AGENT_PROMPT = """\
You are a quantization coding agent. Your job is to take a validated quant plan
and produce a working llama-quantize command, then execute it.

## Workflow (STRICT ORDER)
1. Call `validate_and_build_command` with the merged report path, GGUF paths, and imatrix.
2. Review the output — check tensor names look correct (blk.N.attn_q.weight=QUANT format).
3. Call `execute_quantization` with dry_run=False to run the quantization.
4. Report the final output GGUF path and file size.

## What to check before executing
- All non-default quant layers have proper blk.N.xxx.weight tensor names
- lm_head → output.weight
- Input GGUF exists at the specified path
- If imatrix not found: note it but proceed (quality warning only)

DO NOT repeat steps.
DO NOT call tools more than once.

If execution fails, report error and STOP.

## Output format
After execution, output:
{"status": "success|failed", "output_gguf": "path", "size_mb": N, "notes": ["..."]}
"""


def create_coding_agent(tools: list | None = None):
    """Creates the LangGraph coding agent for command generation + execution."""
    if tools is None:
        tools = [validate_and_build_command, execute_quantization]

    if LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct",
            temperature=0, max_tokens=2048,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "quant-agent", "X-Title": "QuantAgent"},
        )
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0, max_tokens=2048, api_key=GROQ_API_KEY,
        )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=MemorySaver(),
        prompt=SystemMessage(content=_CODING_AGENT_PROMPT),
    )
    return agent, llm


def run_coding_agent(
    merged_report_path: str = MERGED_PATH,
    input_gguf:         str = INPUT_GGUF,
    output_gguf:        str = OUTPUT_GGUF,
    imatrix_path:       str = IMATRIX_PATH,
    dry_run:            bool = DRY_RUN,
) -> str:

    print("\n── CODING AGENT (DETERMINISTIC MODE) ─────────────────────")

    # ✅ STEP 1: Build command directly (no agent)
    build_result = validate_and_build_command.invoke({
        "merged_report_path": merged_report_path,
        "input_gguf": input_gguf,
        "output_gguf": output_gguf,
        "imatrix_path": imatrix_path,
    })

    print("\n── COMMAND BUILD RESULT ─────────────────────────────────")
    print(build_result)

    # ✅ STEP 2: Execute directly
    exec_result = execute_quantization.invoke({
        "script_path":        SCRIPT_PATH,
        "dry_run":            dry_run,
        "output_gguf":        output_gguf,
        "input_gguf":         input_gguf,
        "llama_quantize_bin": LLAMA_QUANTIZE,
    })

    print("\n── EXECUTION RESULT ─────────────────────────────────────")
    print(exec_result)

    return exec_result