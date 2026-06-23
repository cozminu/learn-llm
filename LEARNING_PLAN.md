# Learning Plan: Build Your Own Small Language Model

Goal: implement a tiny GPT/Llama-style language model from scratch in raw
PyTorch, train it on Apple Silicon (MPS), and export it to GGUF so it loads
in Ollama / LM Studio.

The two things that make a from-scratch model exportable to GGUF are
decided up front, not bolted on at the end:

1. **Architecture mirrors Llama** (RMSNorm, RoPE, SwiGLU MLP, no biases,
   causal self-attention). This lets us bridge our trained weights into a
   real `LlamaForCausalLM` and reuse llama.cpp's existing, well-tested
   converter instead of hand-writing a GGUF writer.
2. **Tokenizer must end up as BPE** (HF `tokenizers` -> `tokenizer.json`).
   llama.cpp's converter only understands SentencePiece or HF-BPE
   vocabularies. A char-level tokenizer is great for the first fast loop
   but cannot be exported -- it gets swapped out in Stage 3, not at the end.

## Stage 0 -- Setup (done)
- Python 3.14 venv, torch 2.12 (MPS backend confirmed available).
- Project structure: `src/`, `data/`, `scripts/`, `checkpoints/`, `notes/`.

## Stage 1 -- Vertical slice (char-level, tiny-shakespeare)
Build the whole spine end to end on the smallest possible dataset so we
hit every piece of the pipeline in minutes, not hours:
- Char-level tokenizer (trivial, no library needed).
- Llama-style transformer block from scratch: RMSNorm, RoPE, causal
  multi-head self-attention, SwiGLU MLP, residual connections.
- Training loop on MPS (AdamW, lr warmup+decay, grad clipping, checkpointing).
- Sampling script (temperature, top-k).
- Success = model overfits tiny-shakespeare and produces Shakespeare-shaped
  gibberish. Quality doesn't matter yet -- the pipeline working does.

## Stage 2 -- De-risk the export pipeline (on the throwaway Stage 1 model)
Do this *before* investing in a good model, because export is the least
familiar step and the worst place to discover a surprise:
- Write a remap function: our state_dict -> `LlamaForCausalLM` state_dict.
- Run llama.cpp's `convert_hf_to_gguf.py` on the bridged HF model.
- Load the resulting `.gguf` in Ollama and/or LM Studio, confirm it
  produces *some* output (garbage is fine -- "it loads and runs" is the bar).

## Stage 3 -- Real tokenizer (BPE) (done)
- Train a small BPE tokenizer with HF `tokenizers` (vocab ~4k-8k) on the
  target corpus.
- Swap it into the data pipeline and re-verify Stage 1's training loop
  still works with the new vocab.

## Stage 4 -- Real dataset (TinyStories) (done)
- Download a TinyStories subset, tokenize with the BPE tokenizer.
- Retrain the same architecture on real data.

## Stage 5 -- Scale up (done)
- Grow the config (layers/dim/heads) as far as MPS memory comfortably
  allows, train longer, track loss, sample periodically to watch quality
  improve.

## Stage 6 -- Final export (done)
- Re-run the Stage 2 export pipeline on the real trained model.
- Quantize (q4_0/q8_0) with llama.cpp's `quantize` tool.
- Write an Ollama `Modelfile` and confirm `ollama run`.
- Confirm load + generation in LM Studio.

## Stretch goals (Stages 7-9)
The three stretch items are **independent and reorderable**. The default
ordering below puts the lowest-risk, no-training item first (KV cache)
because it also speeds up evaluation/sampling for the two stages after it
-- but instruction-tuning is the highest "it's actually an assistant now"
payoff, so starting there instead is a fine call. Each stage keeps the
house rule from Stages 1-6: a single unambiguous success bar.

Three constraints decided up front (the load-bearing ones, same spirit as
the Stage-0 architecture/tokenizer decisions):
1. **KV cache must not touch the export path.** Export bridges our weights
   into HF's `LlamaForCausalLM`, which has its own cache -- our cache is
   training/sampling-only. The non-cached `forward` stays intact so the
   Stage 2/6 bit-exactness check still passes.
2. **Distillation requires a shared vocabulary.** Logit/KL distillation
   only works if teacher and student share the exact same tokenizer. Our
   BPE is custom-trained, so the teacher must be *our own bigger model on
   our own tokenizer*, not an off-the-shelf LLM (whose vocab wouldn't
   align). This is self-distillation from a larger same-vocab teacher.
3. **Instruction-tuning reuses machinery already in the model.**
   `TinyLM.forward` already passes `ignore_index=-1` to cross-entropy
   (currently unused) -- that's exactly the prompt-token masking SFT needs.

### Stage 7 -- KV-cache-aware generation (done)
The current `generate()` re-runs the full context every step (O(n^2)
recompute). A KV cache stores past keys/values so each new token does O(n)
work instead. Concepts: why attention recomputes without a cache, the
cache tensor shapes `[batch, n_heads, seq, head_dim]`, and the RoPE
position offset for incremental decode.
- Add an *optional* cache path to `Attention`/`LlamaCore`/`TinyLM.forward`
  (a `past_key_values` arg + a position offset for RoPE); leave the
  existing no-cache path untouched.
- The decode-step trap: at a single new token, `q_len=1` but `k_len=N`.
  This step runs with **`is_causal=False`** -- the lone new query
  legitimately attends to *all* past keys; `is_causal=True` here would
  mask the wrong positions. `is_causal=True` is correct only for the
  prefill pass over the prompt.
- Success = cached generation is **token-identical** to the current
  non-cached path under greedy (argmax) decode. Speedup (tokens/sec) is
  the secondary metric and **grows with sequence length** -- a modest win
  at a few hundred tokens is expected, not a failure.

### Stage 8 -- Instruction-tuning (SFT) (done)
Turn the TinyStories base model into something that follows simple
instructions, using TinyStories-Instruct (same vocabulary domain as the
base model, so our byte-level BPE -- which has no OOV -- encodes it
cleanly). Concepts: chat/prompt formatting, loss masking (compute loss
only on response tokens, the prompt is context not a prediction target),
and continuing training from a checkpoint rather than from scratch.
- Up-front verification before committing: TinyStories-Instruct's format
  headers tokenize sanely under our BPE, and examples fit in
  `max_seq_len=512`.
- Set prompt-token targets to `-1` so `ignore_index` drops them from the
  loss; fine-tune from `stage5.pt`.
- New export risk: the chat template must flow into the GGUF
  (`tokenizer.chat_template` metadata) and be picked up by llama.cpp.
  Verify this explicitly -- it's the one part of the export path Stage 6
  didn't already cover.
- Success = the SFT model follows a held-out instruction in the
  TinyStories-Instruct format (visibly conditioned on the instruction,
  not just continuing it as prose), and the re-exported GGUF applies the
  chat template in a real runtime.

### Stage 9 -- Distillation from a larger teacher
Train a larger teacher on TinyStories with *our* tokenizer, then distill
it into a small student and show the student beats training the same-size
model from scratch. Concepts: KL-divergence loss against soft teacher
logits, temperature softening, and combining KD loss with the normal
cross-entropy.
- Use **online distillation** (teacher and student in one loop, teacher
  frozen in eval mode) -- at ~8k vocab and this data size, precomputing/
  caching teacher logits isn't worth the storage/complexity.
- Honest expectation: the deliverable is the *implementation* (KL loss,
  temperature, teacher logits in the loop) plus a controlled
  **distilled-vs-from-scratch-same-size** comparison. On a clean task like
  TinyStories with a student that isn't data-bound, the quality jump may
  be modest -- that's a real result, not a failed stage.
- Success = a clean A/B: student-distilled vs student-from-scratch at
  identical size/step budget, val loss (and a qualitative sample)
  reported for both.

## Why these specific choices
- **MPS**: bf16 autocast preferred over fp16; no float64 ops; if an op is
  unsupported on MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` rather than
  reworking the model.
- **Llama-shaped architecture**: makes Stage 2/6 export reuse existing,
  battle-tested tooling instead of a custom GGUF writer.
- **tiny-shakespeare first, TinyStories second**: cheap fast feedback loop
  before paying the cost of a bigger dataset + real tokenizer.
