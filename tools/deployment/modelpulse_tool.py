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
import sys
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.theme import Theme

# Global console with custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "progress": "magenta",
})
console = Console(theme=custom_theme)


# ─────────────────────────────────────────────────────────────
# 1. SERVER LIFECYCLE
# ─────────────────────────────────────────────────────────────

_server_proc: subprocess.Popen | None = None
_bridge_proc: subprocess.Popen | None = None

# Canonical path — set by start_modelpulse_server(), imported by run.py
# Canonical path — set by start_modelpulse_server(), imported by run.py
METRICS_JSONL_PATH: str = "artifacts/results/metrics.jsonl"

def _modelpulse_base_cmd() -> list[str]:
    """
    Return a runnable ModelPulse command.
    Use module execution to avoid broken shebang issues in copied user-site scripts.
    """
    return [sys.executable, "-m", "modelpulse.main"]


def get_server_ip() -> str:
    """
    Attempt to auto-detect the Tailscale IP (100.64.0.0/10).
    Falls back to '0.0.0.0' if not found.
    """
    import socket
    import subprocess

    # Try getting from 'ip' command on Linux
    try:
        output = subprocess.check_output(["ip", "-4", "addr", "show"], text=True)
        for line in output.splitlines():
            if "inet " in line and "100." in line:
                parts = line.strip().split()
                ip_cidr = parts[1]
                ip = ip_cidr.split("/")[0]
                # Check if in 100.64.0.0/10
                ip_parts = ip.split(".")
                if 64 <= int(ip_parts[1]) <= 127:
                    return ip
    except Exception:
        pass

    # Fallback to socket
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip.startswith("100."):
                ip_parts = ip.split(".")
                if 64 <= int(ip_parts[1]) <= 127:
                    return ip
    except Exception:
        pass

    return "127.0.0.1"


def start_modelpulse_server(
    host: str = "0.0.0.0",
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
    global METRICS_JSONL_PATH
    METRICS_JSONL_PATH = os.path.join(log_dir, "metrics.jsonl")

    # Clear stale metrics
    if os.path.exists(METRICS_JSONL_PATH):
        try:
            os.remove(METRICS_JSONL_PATH)
            console.print(f"[dim]Cleared stale metrics: {METRICS_JSONL_PATH}[/dim]")
        except Exception:
            pass

    # Check if port is already in use
    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    try:
        with socket.create_connection((check_host, port), timeout=0.5):
            console.print(f"[warning]⚠ Port {port} is already occupied. Using existing server instance.[/warning]")
            return None
    except (OSError, ConnectionRefusedError):
        pass

    cmd = [
        *_modelpulse_base_cmd(), "server", "run",
        "--host", host,
        "--port", str(port),
        "--ping-interval","120.0",
        "--log-dir", log_dir,
    ]
    
    with console.status(f"[info]Starting ModelPulse Server on {host}:{port}...[/info]", spinner="dots"):
        _server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def _stream_logs(proc: subprocess.Popen):
            log_path = "artifacts/results/server.log"
            with open(log_path, "a") as f:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()

        threading.Thread(target=_stream_logs, args=(_server_proc,), daemon=True).start()

        deadline = time.time() + readiness_timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((check_host, port), timeout=1):
                    console.print(Panel(
                        f"[success]✔ Server is up and running![/success]\n"
                        f"[info]URL:[/info] http://{check_host}:{port}\n"
                        f"[info]Logs:[/info] {METRICS_JSONL_PATH}",
                        title="ModelPulse Infrastructure",
                        border_style="green"
                    ))
                    return _server_proc
            except OSError:
                time.sleep(0.5)

    raise RuntimeError(f"[ModelPulse] Server not ready after {readiness_timeout}s")


def stop_modelpulse_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        print("[ModelPulse] Server stopped.")
    _server_proc = None


def start_modelpulse_bridge(server_url: str) -> subprocess.Popen:
    """
    Launch the ModelPulse bridge (client) in the background.
    It connects to the server and waits for models to benchmark.
    """
    global _bridge_proc

    if _bridge_proc is not None and _bridge_proc.poll() is None:
        print("[ModelPulse] Bridge already running — skipping.")
        return _bridge_proc

    cmd = [*_modelpulse_base_cmd(), "bridge", "run", server_url, "--benchmark"]
    
    with console.status(f"[info]Launching ModelPulse Bridge...[/info]", spinner="bouncingBar"):
        _bridge_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def _stream_bridge_logs(proc: subprocess.Popen):
            log_path = "artifacts/results/bridge.log"
            with open(log_path, "a") as f:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()

        threading.Thread(target=_stream_bridge_logs, args=(_bridge_proc,), daemon=True).start()
    return _bridge_proc


def wait_for_client(server_url: str, timeout: int = 60):
    """
    Poll the server until at least one client (bridge) is connected.
    """
    import httpx
    deadline = time.time() + timeout
    
    with console.status(f"[progress]⌛ Waiting for bridge connection to {server_url}...[/progress]", spinner="earth") as status:
        while time.time() < deadline:
            try:
                resp = httpx.get(f"{server_url}/ws/clients", timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    count = data.get("count", 0)
                    if count > 0:
                        console.print(f"[success]✔ Bridge connected! (Active IDs: {', '.join(data.get('client_ids', []))})[/success]")
                        return True
            except Exception:
                pass
            time.sleep(2)
    
    console.print(f"[error]✘ No client connected after {timeout}s[/error]")
    return False


def stop_modelpulse_bridge():
    global _bridge_proc
    if _bridge_proc and _bridge_proc.poll() is None:
        _bridge_proc.terminate()
        try:
            _bridge_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _bridge_proc.kill()
        print("[ModelPulse] Bridge stopped.")
    _bridge_proc = None


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

    cmd = [*_modelpulse_base_cmd(), "server", "convert", input_gguf, output_dir]
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
    server_url: str = "http://0.0.0.0",
):
    """
    Shard (if needed) then upload to the running server.
    """
    clean_name = model_name.replace(".gguf", "")

    manifest_path = os.path.join(shard_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        shard_gguf(input_gguf, shard_dir)
    else:
        print(f"[Shard] Using existing manifest/shards in {shard_dir}")

    cmd = [
        *_modelpulse_base_cmd(), "server", "upload",
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


def make_modelpulse_tool(server_host: str = "0.0.0.0", server_port: int = 8000):
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