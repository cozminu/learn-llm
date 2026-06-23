#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
    echo "Creating venv with $PYTHON..."
    "$PYTHON" -m venv .venv
fi

if [ -f .venv/bin/activate ]; then
    ACTIVATE=.venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    # Windows venvs (including under Git Bash) put scripts in Scripts/, not bin/
    ACTIVATE=.venv/Scripts/activate
else
    echo "Could not find venv activate script under .venv/bin or .venv/Scripts" >&2
    exit 1
fi

source "$ACTIVATE"

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data checkpoints

LLAMA_CPP_COMMIT=e2e7a9b2d04e0404131e551243d4bd399ccfe606
if [ ! -d llama.cpp ]; then
    echo "Cloning llama.cpp (needed for GGUF export, see scripts/export_gguf.py)..."
    git clone https://github.com/ggml-org/llama.cpp.git
    # Pinned because scripts/export_gguf.py monkeypatches an internal
    # conversion API (conversion.base.TextModel) that isn't guaranteed
    # stable across llama.cpp versions.
    git -C llama.cpp checkout "$LLAMA_CPP_COMMIT"
fi

python3 - <<'EOF'
import torch
print(f"torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"device: cuda ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    print("device: mps")
else:
    print("device: cpu (no GPU backend found)")
EOF

echo ""
echo "Setup done. Activate with: source $ACTIVATE"
echo "Next: download a dataset (scripts/download_data.py) and train a tokenizer"
echo "(scripts/train_tokenizer.py) before running src/train.py -- see LEARNING_PLAN.md."
