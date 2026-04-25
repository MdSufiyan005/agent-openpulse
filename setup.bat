@echo off
setlocal enabledelayedexpansion

echo ==============================================
echo   CHISELED -- Windows Setup ^& Launch
echo ==============================================

:: ── Python check ───────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo [OK] %%i

:: ── pip dependencies ────────────────────────────────────────────
echo.
echo [INSTALL] Installing Python dependencies...
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
echo [OK] Python packages installed.

:: ── llama.cpp NOTE (Windows) ────────────────────────────────────
:: llama-quantize must be built manually or downloaded as a prebuilt binary.
:: On Windows: download from https://github.com/ggerganov/llama.cpp/releases
:: Place llama-quantize.exe at: llama.cpp\build\bin\Release\llama-quantize.exe
:: Then set LLAMA_QUANTIZE env var to that path.
echo.
echo [NOTE] Windows: llama-quantize.exe must be downloaded manually.
echo        See: https://github.com/ggerganov/llama.cpp/releases
echo        Expected path: llama.cpp\build\bin\Release\llama-quantize.exe
echo        OR set LLAMA_QUANTIZE env var to point to the binary.

if not exist "llama.cpp\build\bin\Release\llama-quantize.exe" (
    echo [WARN] llama-quantize.exe not found at expected path.
    echo        Pipeline will run in DRY_RUN mode for GGUF generation.
    set DRY_RUN=1
) else (
    echo [OK] llama-quantize.exe found.
    set LLAMA_QUANTIZE=llama.cpp\build\bin\Release\llama-quantize.exe
)

:: ── Directory structure ─────────────────────────────────────────
if not exist "artifacts\results"         mkdir artifacts\results
if not exist "artifacts\models-storage"  mkdir artifacts\models-storage
if not exist "artifacts\shards"          mkdir artifacts\shards
if not exist "images"                    mkdir images
if not exist "memory"                    mkdir memory
if not exist "reference_docs"            mkdir reference_docs
echo [OK] Directories ready.

:: ── .env check ─────────────────────────────────────────────────
if not exist ".env" (
    echo.
    echo [WARN] .env not found. Creating template...
    (
        echo # Fill in your keys before running
        echo GROQ_API_KEY=
        echo OPENROUTER_API_KEY=
        echo LLM_PROVIDER=groq
        echo MODEL_ID=HuggingFaceTB/SmolVLM-Instruct
        echo IMAGE_DIR=images
        echo MAX_IMAGES=4
        echo DRY_RUN=0
    ) > .env
    echo [WARN] Please fill in .env and re-run this script.
    pause
    exit /b 1
)
echo [OK] .env found.

:: ── ModelPulse check ─────────────────────────────────────────────
where modelpulse >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Installing modelpulse...
    python -m pip install modelpulse -q
)
echo [OK] modelpulse installed.

:: ── NNI check ────────────────────────────────────────────────────
python -c "import nni" >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Installing nni...
    python -m pip install nni -q
)
echo [OK] nni installed.

:: ── reference_docs check ─────────────────────────────────────────
if not exist "reference_docs\agent_reference.md" (
    echo [WARN] Missing: reference_docs\agent_reference.md
)
if not exist "reference_docs\edge_ai_metrics_reference.md" (
    echo [WARN] Missing: reference_docs\edge_ai_metrics_reference.md
)

echo.
echo ==============================================
echo   All checks passed. Starting pipeline...
echo ==============================================
echo.

python run.py
pause