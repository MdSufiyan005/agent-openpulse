# 🛠️ CHISELED: Iterative Quantization Optimizer

**Chiseled** is a sophisticated feedback-loop optimization pipeline designed to find the perfect balance between model size, inference speed, and output quality for Edge AI deployments.

Using a combination of **KL Divergence Analysis** and **NNI Sparsity Probing**, Chiseled automatically determines the optimal quantization level for every single layer in your model, outperforming naive "global" quantization (like Q4_K_M for everything).

---

## 🚀 Quick Start (Docker) — Recommended

The easiest way to run Chiseled without any dependency issues is using Docker. This ensures `llama.cpp` and all Python libraries are correctly configured.

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd Chiseled
    ```

2.  **Configure Environment**:
    Edit the `.env` file and add your API keys:
    ```bash
    cp .env.example .env  # If .env doesn't exist
    nano .env
    ```

3.  **Launch with Docker Compose**:
    ```bash
    docker compose up --build
    ```
    *Note: If you have an NVIDIA GPU, Docker will automatically utilize it for acceleration.*

---

## 🛠️ Local Installation (Linux/macOS)

If you prefer to run locally, ensure you have `cmake`, `git`, and `python 3.10+` installed.

1.  **Run the Setup Script**:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    This script will:
    - Install Python dependencies.
    - Clone and build `llama.cpp`.
    - Initialize the directory structure.
    - Start the `run.py` pipeline.

---

## ⚙️ Configuration (.env)

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MODEL_ID` | The HuggingFace model ID to optimize. | `Qwen/Qwen2.5-VL-2B-Instruct` |
| `GROQ_API_KEY` | Key for Groq (if using Groq as provider). | - |
| `OPENROUTER_API_KEY` | Key for OpenRouter. | - |
| `LLM_PROVIDER` | `groq`, `openrouter`, or `openai`. | `openrouter` |
| `IMAGE_DIR` | Directory containing calibration images. | `images` |
| `MAX_IMAGES` | Max images used for calibration. | `3` |
| `SERVER_PORT` | Port for the ModelPulse server. | `8000` |

---

## 🧠 How it Works: The Optimization Loop

Chiseled operates in a 3-phase iterative loop:

1.  **Phase 1 — Sensitivity Analysis**: Measures KL Divergence for each layer to identify which parts of the model are most "sensitive" to quantization noise.
2.  **Phase 2 — Sparsity Probing**: Uses NNI to test how layers respond to different compression levels.
3.  **Phase 3 — Planning & Benchmarking**:
    - An **LLM Agent** analyzes the metrics.
    - It creates a custom quantization plan (e.g., Q8_0 for sensitive layers, Q4_K_S for insensitive ones).
    - The model is quantized via `llama.cpp`, uploaded to the edge device via **ModelPulse**, and benchmarked in real-time.
    - The results (TPS, Latency, RAM) are fed back into the agent for the next iteration.

---

## 📊 Directory Structure

- `agents/`: Logic for the Planner and Summarizer agents.
- `tools/`: KL Divergence, Sparsity, and Deployment tools.
- `artifacts/`: Stores generated GGUFs, reports, and shards.
- `reference_docs/`: Documentation used by the Agent to make decisions.
- `images/`: Calibration images for VLMs.

---

## ⚠️ Troubleshooting

- **Llama.cpp build errors**: If running locally, ensure `cmake` and a C++ compiler (`gcc`/`clang`) are installed. On Docker, this is handled automatically.
- **No Client Connected**: Ensure your edge device or bridge is running and can reach the `SERVER_URL`.
- **Out of Memory**: Optimization (especially KL analysis) can be VRAM-intensive. Try reducing `BATCH_SIZE` in `run.py` if you hit OOM.

---

*Built with ❤️ for Edge AI Optimization.*
