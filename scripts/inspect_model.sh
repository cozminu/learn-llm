#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HF_DIR="${1:-checkpoints/stage5-hf}"

source .venv/bin/activate

python3 -c "
from torchinfo import summary
from transformers import LlamaForCausalLM
import torch

model = LlamaForCausalLM.from_pretrained('$HF_DIR')
summary(model, input_size=(1, 8), dtypes=[torch.long], depth=3, verbose=1)
"
