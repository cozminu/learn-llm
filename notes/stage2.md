# Stage 2 -- de-risk the export pipeline

Goal: prove our from-scratch weights survive HF bridge -> GGUF -> a real
runtime (llama.cpp / LM Studio), before investing in a good model. Used
the Stage 1 throwaway shakespeare model.

## What worked immediately
`scripts/export_hf.py` builds a `LlamaConfig`/`LlamaForCausalLM` from our
`ModelConfig` and loads our `state_dict` with `strict=True` -- **no key
remapping needed**, because `TinyLM`/`LlamaCore` were named to match HF's
`model.embed_tokens.*`, `model.layers.N.self_attn.*`, `model.layers.N.mlp.*`,
`model.norm.*`, `lm_head.*` exactly from the start. That confirms the
architecture choice (RMSNorm/RoPE/SwiGLU, no biases) really does make us
weight-compatible with stock Llama tooling.

## What didn't work, and why
The char tokenizer needed a tokenizer.model (SentencePiece) for
`convert_hf_to_gguf.py`'s `_set_vocab_sentencepiece` path, since the
alternative (`_set_vocab_gpt2`) requires the BPE pre-tokenizer to match a
known hash llama.cpp recognizes (it never will for a hand-rolled vocab).
Building a raw SentencePiece `ModelProto` by hand surfaced two real
llama.cpp internals worth remembering:

1. **`escape_whitespaces` defaults to `true` for SPM-type vocabs and is not
   exposed as a GGUF override.** llama.cpp always replaces literal spaces
   with `▁` (`▁`) before vocab lookup. Our proto had stored a literal
   `" "` piece for the space token -- crashed (`unordered_map::at: key not
   found`) the moment a prompt contained a space. Fix: store `"▁"` as the
   piece text for the space token id instead of `" "` (`export_hf.py`,
   `build_sentencepiece_tokenizer`).
2. **`add_space_prefix` defaults to `true`** (prepends a dummy leading
   space) and also isn't set by the converter's sentencepiece path. Fixed
   by patching the GGUF metadata after conversion with
   `tokenizer.ggml.add_space_prefix = false`, reusing llama.cpp's own
   `gguf_new_metadata.py:copy_with_new_metadata` helper rather than
   hand-rolling a GGUF rewriter.

Also: `llama-cli` in this llama.cpp build assumes chat-template/conversation
mode and refuses `--no-conversation`; raw completion now lives in a
separate `llama-completion` binary.

Both fixes are folded into `scripts/export_gguf.py`, which runs
HF-bridge -> `convert_hf_to_gguf.py` -> metadata patch as one step.
This script is specific to our char-tokenizer hack -- once Stage 3 swaps
in a real HF BPE tokenizer, the converter's native BPE path should need
neither workaround (a real `tokenizer.json` has actual byte coverage and
a recognizable pre-tokenizer, or we register a custom one properly).

## Final proof
Loaded `stage1.gguf` in LM Studio (`~/.lmstudio/models/learn-llm/`, via
`lms load`), queried the local OpenAI-compatible completions API with a
prompt containing spaces and a newline, got back multi-token, Shakespeare-
shaped continuation. Pipeline fully proven on a throwaway model -- safe to
build real quality on top of it now.

## Faithfulness check (don't trust "it runs" alone)
"Loads and emits plausible text" doesn't prove the from-scratch
attention/RoPE/SwiGLU is *actually* computing the same function as real
Llama -- a wrong RoPE convention could still emit locally-plausible char
soup. Two checks, cheapest first:

1. **`TinyLM` vs HF `LlamaForCausalLM`, same input_ids, f32, eval mode:**
   top-1 argmax agreement = 100%, max abs logit diff = 0.0 (bit-exact).
   This is the keystone -- it proves our from-scratch forward pass *is*
   Llama's forward pass, not an approximation of it.
2. **Teacher-forced top-1 agreement through the real GGUF/ggml runtime**
   (`llama-cpp-python`, 200 positions of real shakespeare text, no
   autoregressive feedback): 100% agreement, zero mismatches. This
   confirms the HF->GGUF conversion (including the q/k RoPE permute that
   `convert_hf_to_gguf.py` applies for "llama" arch) is faithful too.

One red herring along the way: naive greedy *autoregressive* decoding
(PyTorch greedy-decode vs `llama-completion --top-k 1 --ignore-eos`)
diverged after the first generated token. Root cause was **not** a bug --
it's ordinary float non-associativity between PyTorch's SDPA kernel and
ggml's attention kernel (confirmed identical on CPU and Metal, so not GPU
nondeterminism either), invisible per-position but amplified once a
greedy chain feeds a different token back as context. The teacher-forced
check above is what actually isolates "is the math faithful" from "do two
independent floating point implementations of softmax round identically,"
which they never will and don't need to.
