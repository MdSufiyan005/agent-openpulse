"""
agents/coding_agent.py
======================
Generates and executes llama-quantize commands.

Changes vs original:
- All output paths under artifacts/ (single source of truth)
- Windows-safe: detects OS, uses .exe path on Windows, warns if not found
- full_quant_mode=True → generates naive Q4_K_M for the whole model (comparison baseline)
- Script written to artifacts/results/quantize_run.sh (Linux) or .bat (Windows)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

IS_WINDOWS = platform.system() == "Windows"

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR   = "artifacts"
RESULTS_DIR     = os.path.join(ARTIFACTS_DIR, "results")

if IS_WINDOWS:
    # Expect prebuilt binary downloaded from llama.cpp releases
    _DEFAULT_QUANTIZE = os.path.join(
        "llama.cpp", "build", "bin", "Release", "llama-quantize.exe"
    )
else:
    _DEFAULT_QUANTIZE = os.path.join("llama.cpp", "build", "bin", "llama-quantize")

INPUT_GGUF      = os.environ.get("INPUT_GGUF",      os.path.join(ARTIFACTS_DIR, "input-f16.gguf"))
OUTPUT_GGUF     = os.environ.get("OUTPUT_GGUF",     os.path.join(ARTIFACTS_DIR, "output-layerwise.gguf"))
LLAMA_QUANTIZE  = os.environ.get("LLAMA_QUANTIZE",  _DEFAULT_QUANTIZE)
IMATRIX_PATH    = os.environ.get("IMATRIX_PATH",    "imatrix.dat")
DRY_RUN         = os.environ.get("DRY_RUN",         "0") == "1"

MERGED_PATH     = os.path.join(RESULTS_DIR, "unified_compression_report.json")
SCRIPT_EXT      = ".bat" if IS_WINDOWS else ".sh"
SCRIPT_PATH     = os.path.join(RESULTS_DIR, f"quantize_run{SCRIPT_EXT}")
LOG_PATH        = os.path.join(RESULTS_DIR, "quantize_run.log")

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "groq").lower()

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Tensor name mapper ────────────────────────────────────────────────────────

def _to_gguf_tensor(python_name: str) -> str:
    import re
    if not python_name.endswith(".weight"):
        python_name += ".weight"
    m = re.search(r"layers?[.\[](\\d+)", python_name)
    layer_idx = m.group(1) if m else "0"
    suffix_map = {
        "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
        "o_proj": "attn_output", "gate_proj": "ffn_gate",
        "up_proj": "ffn_up", "down_proj": "ffn_down",
        "fc1": "ffn_up", "fc2": "ffn_down",
    }
    for py, gg in suffix_map.items():
        if py in python_name:
            return f"blk.{layer_idx}.{gg}.weight"
    if "lm_head" in python_name:
        return "output.weight"
    return python_name


# ── Script writer ─────────────────────────────────────────────────────────────

def _write_script(argv: list[str], input_gguf: str, output_gguf: str,
                  imatrix_path: str) -> str:
    os.makedirs(os.path.dirname(SCRIPT_PATH), exist_ok=True)

    if IS_WINDOWS:
        lines = ["@echo off", "echo Starting quantization..."]
        cmd_parts = [f'"{a}"' if " " in a else a for a in argv]
        lines.append(" ".join(cmd_parts))
        lines.append("echo Done.")
        content = "\r\n".join(lines)
    else:
        def _esc(s): return f'"{s}"' if " " in s else s
        script_lines = [argv[0]]
        i = 1
        while i < len(argv):
            if argv[i] in ("--tensor-type") and i + 1 < len(argv):
                script_lines.append(f"  {argv[i]} {_esc(argv[i+1])}")
                i += 2
            else:
                script_lines.append(f"  {_esc(argv[i])}")
                i += 1
        pretty = " \\\n".join(script_lines)
        content = f"#!/bin/bash\nset -e\necho 'Starting quantization...'\n{pretty}\necho 'Done.'\nls -lh {output_gguf}\n"

    with open(SCRIPT_PATH, "w") as f:
        f.write(content)

    if not IS_WINDOWS:
        os.chmod(SCRIPT_PATH, 0o755)

    return SCRIPT_PATH


# ── Build argv ────────────────────────────────────────────────────────────────

def _build_argv(merged_report_path: str, input_gguf: str, output_gguf: str,
                imatrix_path: str, full_quant_mode: bool = False) -> list[str]:
    default_quant = "Q4_K_M"

    argv = [LLAMA_QUANTIZE]

    # if imatrix_path and os.path.exists(imatrix_path):
    #     argv += ["--imatrix", imatrix_path]

    if not full_quant_mode and merged_report_path and os.path.exists(merged_report_path):
        with open(merged_report_path) as f:
            report = json.load(f)
        qa: dict = report.get("quant_assignments", {})
        by_quant: dict[str, list] = {}
        for name, qt in qa.items():
            if qt != default_quant:
                by_quant.setdefault(qt, []).append(_to_gguf_tensor(name))
        for qt, tensor_names in sorted(by_quant.items()):
            for tn in tensor_names:
                if not tn.endswith(".weight"):
                    tn += ".weight"
                argv += ["--tensor-type", f"{tn}={qt}"]

    argv += [input_gguf, output_gguf, default_quant]
    return argv


# ── Execute script ────────────────────────────────────────────────────────────

def _execute_script(script_path: str, input_gguf: str, output_gguf: str,
                    dry_run: bool) -> str:
    problems = []

    if not os.path.exists(script_path):
        problems.append(f"Script not found: {script_path}")
    if not os.path.exists(input_gguf):
        candidates = list(Path(".").glob("**/*.gguf"))[:5]
        problems.append(
            f"Input GGUF not found: {input_gguf}\n"
            f"  Nearby GGUFs: {[str(c) for c in candidates]}"
        )
    if not os.path.exists(LLAMA_QUANTIZE):
        msg = (
            f"llama-quantize not found: {LLAMA_QUANTIZE}\n"
            "  Windows: download from https://github.com/ggerganov/llama.cpp/releases\n"
            "  Linux: cd llama.cpp && cmake -B build && cmake --build build -j --config Release"
        )
        problems.append(msg)

    if problems:
        return "PRE-FLIGHT FAILED:\n" + "\n\n".join(f"✗ {p}" for p in problems)

    with open(script_path) as f:
        script_content = f.read()

    if dry_run:
        return f"DRY RUN:\n{script_content}"

    shell = ["cmd", "/c", script_path] if IS_WINDOWS else ["bash", script_path]

    import threading
    stdout_lines, stderr_lines = [], []

    try:
        proc = subprocess.Popen(
            shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        def _drain(pipe, store, label):
            for line in pipe:
                store.append(line)
                print(f"[{label}] {line}", end="", flush=True)

        t1 = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines, "stdout"))
        t2 = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines, "stderr"))
        t1.start(); t2.start()
        proc.wait(timeout=7200)
        t1.join(); t2.join()
    except Exception as e:
        return f"ERROR: {e}"

    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)

    with open(LOG_PATH, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout_text}\n\n=== STDERR ===\n{stderr_text}")

    if proc.returncode != 0:
        return (
            f"ERROR (exit {proc.returncode})\n"
            f"STDERR:\n{stderr_text[-2000:]}\n"
            f"Log: {LOG_PATH}"
        )

    if os.path.exists(output_gguf):
        size_mb = os.path.getsize(output_gguf) / 1024 / 1024
        return f"✓ Done. Output: {output_gguf} ({size_mb:.1f} MB)"
    return f"WARNING: exit 0 but output not found at {output_gguf}"


# ── Public entry point ────────────────────────────────────────────────────────

def run_coding_agent(
    merged_report_path: str = MERGED_PATH,
    input_gguf:         str = INPUT_GGUF,
    output_gguf:        str = OUTPUT_GGUF,
    imatrix_path:       str = IMATRIX_PATH,
    dry_run:            bool = DRY_RUN,
    full_quant_mode:    bool = False,
) -> str:
    """
    Build and execute the quantization script.

    full_quant_mode=True → ignore merged_report_path, apply Q4_K_M to all layers.
    This is used for the comparison baseline (Phase 3 in run.py).
    """
    mode_label = "FULL Q4_K_M (comparison baseline)" if full_quant_mode else "LAYERWISE"
    print(f"\n── CODING AGENT [{mode_label}] ─────────────────────────────────")
    print(f"   Platform       : {'Windows' if IS_WINDOWS else 'Linux'}")
    print(f"   llama-quantize : {LLAMA_QUANTIZE}")
    print(f"   input_gguf     : {input_gguf}")
    print(f"   output_gguf    : {output_gguf}")

    argv = _build_argv(
        merged_report_path=merged_report_path,
        input_gguf=input_gguf,
        output_gguf=output_gguf,
        imatrix_path=imatrix_path,
        full_quant_mode=full_quant_mode,
    )

    _write_script(argv, input_gguf, output_gguf, imatrix_path)
    print(f"   Script         : {SCRIPT_PATH}")

    result = _execute_script(SCRIPT_PATH, input_gguf, output_gguf, dry_run)
    print(f"\n── RESULT ──\n{result}")
    return result