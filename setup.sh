#!/bin/bash
# Aurora Backend Setup Script for Linux/Mac

set -e

echo "================================================"
echo "Aurora Backend Setup for Linux/Mac"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "macOS: brew install python3"
    exit 1
fi
echo "[OK] Python found"
python3 --version
echo ""

# Check Git
if ! command -v git &> /dev/null; then
    echo "[ERROR] Git is not installed"
    echo "Ubuntu/Debian: sudo apt-get install git"
    echo "macOS: brew install git"
    exit 1
fi
echo "[OK] Git found"
echo ""

# Create directories
echo "[STEP 1/5] Creating directories..."
mkdir -p models logs temp
echo "[OK] Directories created"
echo ""

# Download models from Hugging Face
echo "[STEP 2/5] Downloading models from Hugging Face..."
echo "This will download ~2.5 GB (may take several minutes)"
echo ""

rm -rf temp/hf-download

git clone https://huggingface.co/alvanalrakib/Aurora-Labs temp/hf-download

# Copy files to models folder
echo "Copying model files to models folder..."
cp temp/hf-download/melody_model.safetensors models/
cp temp/hf-download/melody_model_config.json models/
cp temp/hf-download/alv_tokenizer-2.0.1-py3-none-any.whl models/

if [ ! -f "models/melody_model.safetensors" ]; then
    echo "[ERROR] Model files not found in download"
    exit 1
fi

echo "[OK] Model files downloaded"
echo ""

# Install tokenizer first
echo "[STEP 3/5] Installing Aurora tokenizer..."
python3 -m pip install --upgrade models/alv_tokenizer-2.0.1-py3-none-any.whl
echo "[OK] Tokenizer installed"
echo ""

# Install other dependencies
echo "[STEP 4/5] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "[OK] Dependencies installed"
echo ""

# PyTorch installation
echo "[STEP 5/5] PyTorch Installation"
echo ""
if python3 -c "import torch" &> /dev/null; then
    echo "[OK] PyTorch already installed"
    python3 -c "import torch; print('  Version:', torch.__version__); print('  CUDA:', 'Available' if torch.cuda.is_available() else 'Not available')"
else
    echo "PyTorch is not installed. Please install it manually:"
    echo ""
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "For macOS:"
        echo "  python3 -m pip install torch torchvision torchaudio"
    else
        echo "For NVIDIA GPU with CUDA support:"
        echo "  python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        echo ""
        echo "For CPU only (slower but works without GPU):"
        echo "  python3 -m pip install torch torchvision torchaudio"
    fi
    echo ""
    echo "After installing PyTorch, run the server with: python3 main.py"
fi
echo ""

# Cleanup
rm -rf temp/hf-download

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. If PyTorch not installed, install it using commands above"
echo "2. Start server: python3 main.py"
echo "3. Open API docs: http://localhost:8000/docs"
echo ""

# OS-specific VST path
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Optional - Install VST Plugin:"
    echo "  Copy Assets/AruraMelody.vst3 to:"
    echo "  /Library/Audio/Plug-Ins/VST3/"
else
    echo "Optional - Install VST Plugin:"
    echo "  Copy Assets/AruraMelody.vst3 to:"
    echo "  /usr/lib/vst3/ or ~/.vst3/"
fi
echo ""
