# learn-llm

A from-scratch small language model, built in stages, ending in a GGUF
file that loads in Ollama / LM Studio. See `LEARNING_PLAN.md` for the
full staged plan and the reasoning behind it -- read it before resuming
work after a break.

## User context
The user is new to ML/PyTorch and is learning by building this with
Claude writing the code collaboratively -- explain *why*, not just *what*,
especially around tensor shapes, attention, and the export pipeline.

## Environment
- `.venv/` is the project venv (Python 3.14, torch 2.12, MPS backend).
  Activate with `source .venv/bin/activate` before running anything.
- Apple Silicon / MPS: prefer bf16 autocast over fp16, avoid float64,
  set `PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is unsupported rather than
  reworking the model.

## Architecture decisions (don't relitigate without reason)
- Model mirrors Llama (RMSNorm, RoPE, SwiGLU MLP, no biases) specifically
  so trained weights can be bridged into `LlamaForCausalLM` and exported
  with llama.cpp's existing `convert_hf_to_gguf.py`, instead of hand-writing
  a GGUF writer.
- Tokenizer ends as BPE (HF `tokenizers`, `tokenizer.json`) because that's
  what the GGUF converter understands. Char-level tokenizer is Stage-1-only,
  for fast iteration -- it's explicitly replaced in Stage 3, not at the end.
- Export pipeline gets proven on a throwaway model (Stage 2) before any
  real training investment, because it's the least familiar step.

## Layout
- `src/` -- model, tokenizer, training loop, sampling.
- `scripts/` -- data download, HF bridge/export, GGUF conversion helpers.
- `data/` -- datasets (gitignored).
- `checkpoints/` -- model checkpoints and exported GGUF files (gitignored).
- `notes/` -- short per-stage write-ups of what was built and why.

## Conventions
- No comments explaining what code does -- names should do that. Comments
  only for non-obvious *why* (e.g. a shape gotcha, an MPS workaround).
- Explain transformer/training concepts in chat as we build each stage,
  not in code comments or docstrings.
