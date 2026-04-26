#!/bin/bash
set -e

echo "=============================================="
echo "  CHISELED — Linux Setup & Launch"
echo "=============================================="

# ── Python check ──────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON=$(command -v python3)
echo "[OK] Python: $($PYTHON --version)"

# ── pip dependencies ───────────────────────────────────────────
echo ""
echo "[INSTALL] Installing Python dependencies..."
$PYTHON -m pip install --upgrade pip -q --break-system-packages
$PYTHON -m pip install -r requirements.txt -q --break-system-packages
echo "[OK] Python packages installed."

# ── Build tools check ──────────────────────────────────────────
MISSING_TOOLS=()
if ! command -v cmake &>/dev/null; then MISSING_TOOLS+=("cmake"); fi
if ! command -v gcc &>/dev/null; then MISSING_TOOLS+=("gcc/build-essential"); fi
if ! command -v git &>/dev/null; then MISSING_TOOLS+=("git"); fi

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo ""
    echo "=============================================="
    echo "[ERROR] Missing required build tools: ${MISSING_TOOLS[*]}"
    echo "Please install them using your package manager:"
    echo "  sudo apt update && sudo apt install -y cmake build-essential git"
    echo "=============================================="
    exit 1
fi

# ── llama.cpp build (Linux only) ───────────────────────────────
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo ""
    echo "[BUILD] llama.cpp not found or incomplete — building..."
    if [ ! -d "llama.cpp" ]; then
        echo "[CLONE] Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git --depth=1
    fi
    mkdir -p llama.cpp/build
    cd llama.cpp
    echo "[CMAKE] Configuring..."
    cmake -B build -DLLAMA_CURL=OFF -DLLAMA_CUDA=OFF > /dev/null
    echo "[BUILD] Compiling (this may take a few minutes)..."
    cmake --build build -j$(nproc) --config Release --target llama-quantize > /dev/null
    if [ $? -ne 0 ]; then
        echo "[ERROR] llama.cpp build failed. Check your compiler and cmake version."
        exit 1
    fi
    cd ..
    echo "[OK] llama-quantize built."
else
    echo "[OK] llama-quantize already exists."
fi

# ── Directory structure ─────────────────────────────────────────
mkdir -p artifacts/results artifacts/models-storage artifacts/shards images memory reference_docs
echo "[OK] Directories ready."

# ── .env check ─────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo ""
    echo "[WARN] .env not found. Creating template..."
    cat > .env << 'EOF'
# Fill in your keys before running
GROQ_API_KEY=gsk_uzx1nm6azbJ................................................................
OPENROUTER_API_KEY=sk-or-v1.................................................................
LLM_PROVIDER=openrouter
MODEL_ID=Qwen/Qwen2-VL-2B-Instruct
BASE_GGUF=artifacts/input-f16.gguf
IMAGE_DIR=images
MAX_IMAGES=3
DRY_RUN=0
EOF
    echo "[IMPORTANT] .env created from template. PLEASE EDIT .env AND ADD YOUR API KEYS."
    echo "Then re-run this script."
    exit 1
fi
echo "[OK] .env found."

# ── ModelPulse check ───────────────────────────────────────────
if ! command -v modelpulse &>/dev/null; then
    echo "[INSTALL] Installing modelpulse..."
    $PYTHON -m pip install modelpulse -q --break-system-packages
fi
echo "[OK] modelpulse: $(modelpulse --version 2>/dev/null || echo 'installed')"

# ── NNI check ──────────────────────────────────────────────────
$PYTHON -c "import nni" 2>/dev/null || {
    echo "[INSTALL] Installing nni..."
    $PYTHON -m pip install nni -q --break-system-packages
}
echo "[OK] nni installed."

# ── reference_docs check ───────────────────────────────────────
for f in reference_docs/agent_reference.md reference_docs/edge_ai_metrics_reference.md; do
    if [ ! -f "$f" ]; then
        echo "[WARN] Missing reference doc: $f"
    fi
done

echo ""
echo "=============================================="
echo "  All checks passed. Starting pipeline..."
echo "=============================================="
echo ""

$PYTHON run.py