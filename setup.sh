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
$PYTHON -m pip install --upgrade pip -q
$PYTHON -m pip install -r requirements.txt -q
echo "[OK] Python packages installed."

# ── llama.cpp build (Linux only — not needed on Windows) ───────
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo ""
    echo "[BUILD] llama.cpp not found — building..."
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git --depth=1
    fi
    cd llama.cpp
    cmake -B build -DLLAMA_CURL=OFF 2>&1 | tail -5
    cmake --build build -j$(nproc) --config Release 2>&1 | tail -10
    cd ..
    echo "[OK] llama-quantize built."
else
    echo "[OK] llama-quantize already exists — skipping build."
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
MODEL_ID=unsloth/Qwen3-VL-2B-Thinking-1M-GGUF
IMAGE_DIR=images
MAX_IMAGES=3
DRY_RUN=0
EOF
    echo "[WARN] Please fill in .env and re-run this script."
    exit 1
fi
echo "[OK] .env found."

# ── ModelPulse check ───────────────────────────────────────────
if ! command -v modelpulse &>/dev/null; then
    echo "[INSTALL] Installing modelpulse..."
    $PYTHON -m pip install modelpulse -q
fi
echo "[OK] modelpulse: $(modelpulse --version 2>/dev/null || echo 'installed')"

# ── NNI check ──────────────────────────────────────────────────
$PYTHON -c "import nni" 2>/dev/null || {
    echo "[INSTALL] Installing nni..."
    $PYTHON -m pip install nni -q
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