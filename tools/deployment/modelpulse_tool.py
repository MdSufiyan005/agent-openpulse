# # import subprocess
# # from langchain_core.tools import tool


# # def make_modelpulse_tool(server_host="0.0.0.0", server_port=8000):

# #     @tool("modelpulse_orchestrator")
# #     def modelpulse_orchestrator(model_name: str, model_shard_dir: str):
# #         """
# #         Starts ModelPulse server, uploads model, and prepares bridge inference.
# #         """

# #         logs = {}

# #         # ─────────────────────────────────────────────
# #         # 1. Start server
# #         # ─────────────────────────────────────────────
# #         server_cmd = [
# #             "modelpulse", "server", "run",
# #             "--host", server_host,
# #             "--port", str(server_port)
# #         ]

# #         logs["server"] = subprocess.Popen(
# #             server_cmd,
# #             stdout=subprocess.PIPE,
# #             stderr=subprocess.PIPE
# #         )

# #         # ─────────────────────────────────────────────
# #         # 2. Upload model
# #         # IMPORTANT: remove .gguf extension as requested
# #         # ─────────────────────────────────────────────
# #         clean_model_name = model_name.replace(".gguf", "")

# #         upload_cmd = [
# #             "modelpulse", "server", "upload",
# #             clean_model_name,
# #             model_shard_dir
# #         ]

# #         upload_result = subprocess.run(
# #             upload_cmd,
# #             capture_output=True,
# #             text=True
# #         )

# #         logs["upload_stdout"] = upload_result.stdout
# #         logs["upload_stderr"] = upload_result.stderr

# #         # ─────────────────────────────────────────────
# #         # 3. Bridge command (returned for client execution)
# #         # ─────────────────────────────────────────────
# #         bridge_cmd = f"modelpulse bridge run http://{server_host}:{server_port}"

# #         logs["bridge_command"] = bridge_cmd

# #         return {
# #             "status": "success",
# #             "model": clean_model_name,
# #             "server": f"http://{server_host}:{server_port}",
# #             "bridge_cmd": bridge_cmd,
# #             "logs": logs
# #         }

# #     return modelpulse_orchestrator

# """
# tools/deployment/modelpulse_tool.py

# Separates server lifecycle from upload/shard operations.
# - start_modelpulse_server()  → call ONCE at program start
# - shard_gguf()               → shards a .gguf before upload
# - make_modelpulse_tool()     → upload-only tool (server already running)
# """

# from __future__ import annotations

# import os
# import time
# import subprocess
# import threading
# from langchain_core.tools import tool


# # ─────────────────────────────────────────────────────────────
# # 1. SERVER LIFECYCLE  (call once at startup, not inside tool)
# # ─────────────────────────────────────────────────────────────

# _server_proc: subprocess.Popen | None = None


# def start_modelpulse_server(host: str = "0.0.0.0", port: int = 8000,
#                              readiness_timeout: int = 15) -> subprocess.Popen:
#     """
#     Launch the ModelPulse server as a background process.
#     Blocks until the server is accepting connections (or timeout).
#     Call this ONCE at the top of run.py before the main loop.

#     Returns the Popen handle so you can terminate it on exit.
#     """
#     global _server_proc

#     if _server_proc is not None and _server_proc.poll() is None:
#         print("[ModelPulse] Server already running — skipping re-launch.")
#         return _server_proc
#     path = "results/"
#     cmd = ["modelpulse", "server", "run", "--host", host, "--port", str(port), "--log-dir", path]
#     print(f"[ModelPulse] Starting server: {' '.join(cmd)}")

#     _server_proc = subprocess.Popen(
#         cmd,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,   # merge so we can read one stream
#         text=True,
#     )

#     # Stream server logs to console in a daemon thread so they don't block
#     def _stream_logs(proc: subprocess.Popen):
#         for line in proc.stdout:
#             print(f"[ModelPulse-server] {line}", end="")

#     t = threading.Thread(target=_stream_logs, args=(_server_proc,), daemon=True)
#     t.start()

#     # Wait until port is accepting connections
#     import socket
#     deadline = time.time() + readiness_timeout
#     while time.time() < deadline:
#         try:
#             with socket.create_connection((host if host != "0.0.0.0" else "127.0.0.1",
#                                            port), timeout=1):
#                 print(f"[ModelPulse] Server ready on {host}:{port}")
#                 return _server_proc
#         except OSError:
#             time.sleep(0.5)

#     raise RuntimeError(
#         f"[ModelPulse] Server did not become ready within {readiness_timeout}s."
#     )


# def stop_modelpulse_server():
#     """Gracefully terminate the server process."""
#     global _server_proc
#     if _server_proc and _server_proc.poll() is None:
#         _server_proc.terminate()
#         _server_proc.wait(timeout=5)
#         print("[ModelPulse] Server stopped.")
#     _server_proc = None


# # ─────────────────────────────────────────────────────────────
# # 2. SHARDING  (must happen before upload)
# # ─────────────────────────────────────────────────────────────

# def shard_gguf(input_gguf: str, output_dir: str, shard_size_mb: int = 512) -> str:
#     """
#     Shard a .gguf file into the output_dir using the modelpulse shard command.
#     Returns the output_dir path (what upload expects).

#     modelpulse shard <input.gguf> <output_dir> --size <MB>
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     cmd = [
#         "modelpulse", "server", "convert",
#         input_gguf,
#         output_dir,
#         # "--size", str(shard_size_mb),
#     ]
#     print(f"[Shard] Running: {' '.join(cmd)}")

#     result = subprocess.run(cmd, capture_output=True, text=True)

#     if result.returncode != 0:
#         raise RuntimeError(
#             f"[Shard] Failed for {input_gguf}:\n"
#             f"  stdout: {result.stdout}\n"
#             f"  stderr: {result.stderr}"
#         )

#     print(f"[Shard] Done → {output_dir}")
#     return output_dir


# # ─────────────────────────────────────────────────────────────
# # 3. UPLOAD TOOL  (server must already be running)
# # ─────────────────────────────────────────────────────────────

# def make_modelpulse_tool(server_host: str = "0.0.0.0", server_port: int = 8000):
#     """
#     Returns a LangChain tool that:
#       1. Shards the .gguf (if not already sharded)
#       2. Uploads the shard dir to the already-running server

#     Assumes start_modelpulse_server() was called at program startup.
#     """

#     @tool("modelpulse_upload")
#     def modelpulse_upload(model_name: str, model_shard_dir: str,
#                           input_gguf: str = "", shard_size_mb: int = 512):
#         """
#         Shard (if needed) and upload a model to the running ModelPulse server.

#         Args:
#             model_name:     Name without .gguf extension.
#             model_shard_dir: Directory where shards live (or will be created).
#             input_gguf:     Path to source .gguf. If provided, sharding runs first.
#             shard_size_mb:  Shard size in MB (default 512).
#         """
#         clean_name = model_name.replace(".gguf", "")

#         # Shard if a source gguf was provided and shard dir is empty/missing
#         if input_gguf and (
#             not os.path.exists(model_shard_dir)
#             or not os.listdir(model_shard_dir)
#         ):
#             shard_gguf(input_gguf, model_shard_dir, shard_size_mb)

#         # Upload
#         upload_cmd = [
#             "modelpulse", "server", "upload",
#             clean_name,
#             model_shard_dir,
#         ]
#         print(f"[ModelPulse] Uploading '{clean_name}' from '{model_shard_dir}'")
#         result = subprocess.run(upload_cmd, capture_output=True, text=True)

#         if result.returncode != 0:
#             raise RuntimeError(
#                 f"[ModelPulse] Upload failed:\n"
#                 f"  stdout: {result.stdout}\n"
#                 f"  stderr: {result.stderr}"
#             )

#         bridge_cmd = f"modelpulse bridge run http://{server_host}:{server_port}"
#         print(f"[ModelPulse] Upload done. Client bridge cmd: {bridge_cmd}")

#         return {
#             "status": "uploaded",
#             "model": clean_name,
#             "server": f"http://{server_host}:{server_port}",
#             "bridge_cmd": bridge_cmd,
#             "upload_stdout": result.stdout,
#         }

#     return modelpulse_upload

"""
tools/deployment/modelpulse_tool.py

Key facts from server.py source:
- Metrics flow: client → WebSocket METRICS msg → server._log_metrics() → metrics.jsonl
  The file is written SERVER-SIDE to --log-dir/metrics.jsonl
- Convert cmd: modelpulse server convert <input.gguf> <output_dir>
- Upload cmd:  modelpulse server upload <model_id> <shard_dir> --server <url>
- Server API:  GET /results/latest  (alternative to file-watching)
"""

from __future__ import annotations

import os
import time
import socket
import subprocess
import threading
from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────
# 1. SERVER LIFECYCLE
# ─────────────────────────────────────────────────────────────

_server_proc: subprocess.Popen | None = None

# Canonical path — set by start_modelpulse_server(), imported by run.py
METRICS_JSONL_PATH: str = "artifacts/results/metrics.jsonl"


def start_modelpulse_server(
    host: str = "100.81.117.95",
    port: int = 8000,
    log_dir: str = "results",           # server writes metrics.jsonl HERE
    shard_dir: str = "models-storage",  # server stores uploaded model shards HERE
    readiness_timeout: int = 20,
) -> subprocess.Popen:
    """
    Launch the ModelPulse server ONCE at program startup.
    --log-dir  → server writes metrics.jsonl to this dir
    --shard-dir → server stores uploaded model shards here
    Blocks until port is accepting connections.
    """
    global _server_proc, METRICS_JSONL_PATH

    if _server_proc is not None and _server_proc.poll() is None:
        print("[ModelPulse] Server already running — skipping.")
        return _server_proc

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)

    METRICS_JSONL_PATH = os.path.join(log_dir, "metrics.jsonl")

    cmd = [
        "modelpulse", "server", "run",
        "--host", host,
        "--port", str(port),
        # "--shard-dir", shard_dir,
        "--ping-interval","120.0",
        "--log-dir", log_dir,
    ]
    print(f"[ModelPulse] Starting server: {' '.join(cmd)}")
    print(f"[ModelPulse] Metrics will be written to: {METRICS_JSONL_PATH}")

    _server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _stream_logs(proc: subprocess.Popen):
        for line in proc.stdout:
            print(f"[ModelPulse-server] {line}", end="")

    threading.Thread(target=_stream_logs, args=(_server_proc,), daemon=True).start()

    check_host = "127.0.0.1" if host == "100.81.117.95" else host
    deadline = time.time() + readiness_timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((check_host, port), timeout=1):
                print(f"[ModelPulse] Server ready → http://{host}:{port}")
                return _server_proc
        except OSError:
            time.sleep(0.5)

    raise RuntimeError(f"[ModelPulse] Server not ready after {readiness_timeout}s")


def stop_modelpulse_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        _server_proc.wait(timeout=5)
        print("[ModelPulse] Server stopped.")
    _server_proc = None


# ─────────────────────────────────────────────────────────────
# 2. SHARDING
# Correct CLI: modelpulse server convert <input.gguf> <output_dir>
# ─────────────────────────────────────────────────────────────

def shard_gguf(input_gguf: str, output_dir: str) -> str:
    """
    Convert a .gguf into tensor shards using `modelpulse server convert`.
    Produces manifest.json + *.shard files in output_dir.
    Returns output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = ["modelpulse", "server", "convert", input_gguf, output_dir]
    print(f"[Shard] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"[Shard] Failed for {input_gguf}:\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )

    manifest_path = os.path.join(output_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            f"[Shard] Convert ran but manifest.json missing in {output_dir}.\n"
            f"stdout: {result.stdout}"
        )

    print(f"[Shard] Done → {output_dir}")
    return output_dir


# ─────────────────────────────────────────────────────────────
# 3. UPLOAD
# ─────────────────────────────────────────────────────────────

def upload_model_to_server(
    model_name: str,
    input_gguf: str,
    shard_dir: str,
    server_url: str = "http://100.81.117.95",
):
    """
    Shard (if needed) then upload to the running server.
    """
    clean_name = model_name.replace(".gguf", "")

    manifest_path = os.path.join(shard_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        shard_gguf(input_gguf, shard_dir)
    else:
        print(f"[Shard] Skipping — manifest already exists in {shard_dir}")

    cmd = [
        "modelpulse", "server", "upload",
        clean_name,
        shard_dir,
        "--server", server_url,
    ]
    print(f"[Upload] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"[Upload] Failed for {clean_name}:\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )

    print(f"[Upload] Done: {clean_name}")
    return result.stdout


def make_modelpulse_tool(server_host: str = "100.81.117.95", server_port: int = 8000):
    @tool("modelpulse_upload")
    def modelpulse_upload(model_name: str, model_shard_dir: str, input_gguf: str = ""):
        """Upload a model to the running ModelPulse server."""
        server_url = f"http://{server_host}:{server_port}"
        upload_model_to_server(model_name, input_gguf, model_shard_dir, server_url)
        return {
            "status": "uploaded",
            "model": model_name.replace(".gguf", ""),
            "bridge_cmd": f"modelpulse bridge run {server_url} --benchmark",
        }

    return modelpulse_upload