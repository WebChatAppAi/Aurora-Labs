@echo off
REM Aurora Backend Setup Script for Windows

echo ================================================
echo Aurora Backend Setup for Windows
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed
    echo Install from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found
python --version
echo.

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed
    echo Install from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK] Git found
echo.

REM Create directories
echo [STEP 1/5] Creating directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "temp" mkdir temp
echo [OK] Directories created
echo.

REM Download models from Hugging Face
echo [STEP 2/5] Downloading models from Hugging Face...
echo This will download ~2.5 GB (may take several minutes)
echo.

if exist "temp\hf-download" rmdir /s /q "temp\hf-download"

git clone https://huggingface.co/alvanalrakib/Aurora-Labs temp\hf-download
if errorlevel 1 (
    echo [ERROR] Failed to download from Hugging Face
    pause
    exit /b 1
)

REM Copy files to models folder
echo Copying model files to models folder...
copy "temp\hf-download\melody_model.safetensors" "models\" >nul
copy "temp\hf-download\melody_model_config.json" "models\" >nul
copy "temp\hf-download\alv_tokenizer-2.0.1-py3-none-any.whl" "models\" >nul

if not exist "models\melody_model.safetensors" (
    echo [ERROR] Model files not found in download
    pause
    exit /b 1
)

echo [OK] Model files downloaded
echo.

REM Install tokenizer first
echo [STEP 3/5] Installing Aurora tokenizer...
python -m pip install --upgrade "models\alv_tokenizer-2.0.1-py3-none-any.whl"
echo [OK] Tokenizer installed
echo.

REM Install other dependencies
echo [STEP 4/5] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo [OK] Dependencies installed
echo.

REM PyTorch installation
echo [STEP 5/5] PyTorch Installation
echo.
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo PyTorch is not installed. Please install it manually:
    echo.
    echo For NVIDIA GPU with CUDA support:
    echo   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo.
    echo For CPU only (slower but works without GPU):
    echo   python -m pip install torch torchvision torchaudio
    echo.
    echo After installing PyTorch, run the server with: python main.py
) else (
    echo [OK] PyTorch already installed
    python -c "import torch; print('  Version:', torch.__version__); print('  CUDA:', 'Available' if torch.cuda.is_available() else 'Not available')"
)
echo.

REM Cleanup
rmdir /s /q "temp\hf-download"

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo Next steps:
echo 1. If PyTorch not installed, install it using commands above
echo 2. Start server: python main.py
echo 3. Open API docs: http://localhost:8000/docs
echo.
echo Optional - Install VST Plugin:
echo   Copy Assets\AruraMelody.vst3 to:
echo   C:\Program Files\Common Files\VST3\
echo.
pause
