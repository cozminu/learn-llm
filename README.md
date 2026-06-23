# learn-llm

A small language model built from scratch in raw PyTorch -- tokenizer,
transformer, training loop, and sampling -- ending in a GGUF file that
loads in Ollama / LM Studio.

The architecture mirrors Llama (RMSNorm, RoPE, SwiGLU MLP, no biases) so
trained weights can be bridged into `LlamaForCausalLM` and exported with
llama.cpp's existing converter, instead of hand-writing a GGUF writer.

See [`LEARNING_PLAN.md`](LEARNING_PLAN.md) for the full staged build-out
and the reasoning behind each architecture decision, and `notes/` for a
per-stage write-up of what was built and why.

## Requirements

- Python 3.10+
- A GPU backend is optional but recommended: Apple Silicon (MPS) or
  NVIDIA CUDA. Falls back to CPU otherwise.

## Setup

```bash
./setup.sh
```

This creates a `.venv`, installs dependencies from `requirements.txt`,
clones `llama.cpp` (pinned to a known-good commit, needed by
`scripts/export_gguf.py` for the GGUF conversion step), and prints which
compute device (`cuda` / `mps` / `cpu`) PyTorch detects.

`setup.sh` is a bash script, so on Windows run it from Git Bash or WSL.

Activate the venv in new shells with:

```bash
source .venv/bin/activate      # Linux / macOS / WSL
source .venv/Scripts/activate  # Windows (Git Bash)
```

## Usage

1. **Download a dataset** (TinyStories, ~20MB subset by default):
   ```bash
   python3 scripts/download_data.py
   ```
2. **Train a BPE tokenizer** on the downloaded corpus:
   ```bash
   python3 scripts/train_tokenizer.py data/tinystories_train.txt 4096
   ```
3. **Train the model**:
   ```bash
   python3 src/train.py
   ```
4. **Sample from a checkpoint**:
   ```bash
   python3 src/sample.py
   ```
5. **Export to GGUF** and run in Ollama -- see Stage 6 in
   `LEARNING_PLAN.md` and `scripts/export_hf.py` / `scripts/export_gguf.py`.

Instruction-tuning (SFT) and distillation from a larger teacher model are
later stages -- see `src/train_sft.py`, `src/train_teacher.py`,
`src/train_distill.py`, and Stages 8-9 in `LEARNING_PLAN.md`.

## Layout

- `src/` -- model, tokenizer, training loop, sampling.
- `scripts/` -- data download, tokenizer training, HF bridge/export,
  GGUF conversion, verification scripts.
- `data/` -- datasets (gitignored, populated by `download_data.py`).
- `checkpoints/` -- model checkpoints and exported GGUF files (gitignored).
- `notes/` -- short per-stage write-ups of what was built and why.
- `llama.cpp/` -- cloned by `setup.sh` (gitignored); provides
  `convert_hf_to_gguf.py`, used by `scripts/export_gguf.py`.
