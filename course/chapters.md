# Build an LLM From Scratch -- Course Outline

A staged course that mirrors how `learn-llm` was actually built: a Llama-shaped
transformer written in raw PyTorch, trained on Apple Silicon (MPS), and
exported to a real GGUF file that loads in Ollama/LM Studio. Each chapter
maps to one or more stages in `LEARNING_PLAN.md` and is grounded in the real
decisions, bugs, and results captured in `notes/`.

---

## Chapter 1 -- Why Architecture Choices Made Up Front Determine Everything Later
- The two decisions that make a from-scratch model exportable: a Llama-shaped
  block (RMSNorm, RoPE, SwiGLU MLP, no biases) and a tokenizer that ends as
  BPE -- both decided in Stage 0, not bolted on after training.
- Why mirroring Llama means trained weights can later drop straight into HF's
  `LlamaForCausalLM` and reuse llama.cpp's existing converter, instead of
  hand-writing a GGUF writer.
- Why a char-level tokenizer is fine to start with but cannot survive to
  export -- and why it gets swapped in Stage 3, not deferred to the end.
- The project's "de-risk the unfamiliar step early" philosophy: prove the
  export pipeline on a throwaway model before investing in a good one.
- Environment reality: MPS backend on Apple Silicon, bf16 autocast preferred
  over fp16, avoiding float64, and why a constrained environment shapes
  several later decisions (batch size, model size, scaling pace).

## Chapter 2 -- Tokenization 101: From Characters to a Real Vocabulary
- Building the simplest possible tokenizer (char-level, `vocab=65` on
  tiny-shakespeare) and why "trivial, no library needed" is the right choice
  for a fast first loop.
- What a vocabulary actually is, what an embedding table's shape
  `(vocab_size, dim)` means, and why `vocab_size` is the one hyperparameter
  that touches almost every other tensor shape in the model.
- Moving to byte-level BPE (GPT-2 style): why llama.cpp's converter only
  understands SentencePiece or HF-BPE vocabularies, and why byte-level BPE
  specifically avoids out-of-vocabulary tokens entirely.
- Training a real tokenizer with HF `tokenizers` (`BpeTrainer`,
  `pre_tokenizers.ByteLevel`) and registering special tokens
  (`<|endoftext|>`) so they survive as atomic units instead of being shredded
  into byte fragments by BPE merges.
- Verifying a tokenizer swap didn't break anything: same training loop, new
  vocab, watch param count grow (embedding/lm_head) while the rest of the
  architecture stays untouched.

## Chapter 3 -- Inside a Transformer Block: RMSNorm, RoPE, Attention, SwiGLU
- RMSNorm vs LayerNorm: what normalization is actually for, why Llama drops
  the mean-centering and bias terms, and tracing a tensor's shape through a
  norm layer.
- Causal multi-head self-attention from first principles: Q/K/V projections,
  splitting `dim` into `n_heads * head_dim`, the attention score matrix
  shape, and why causal masking exists (a token must not see the future).
- RoPE (rotary position embeddings): why position information is injected
  into Q/K via rotation instead of an added positional embedding, and the
  shape of the precomputed `cos`/`sin` buffers.
- SwiGLU MLP: what a gated activation is and why Llama's MLP has three
  projections instead of two.
- Multi-head attention as a concept check: predicting what changes (and what
  doesn't) when `n_heads` increases at fixed `dim`.
- Residual connections and why they're what makes deep stacks of these
  blocks trainable at all.

## Chapter 4 -- The First Training Loop and Its First Real Bug
- Anatomy of a training loop: forward pass, loss, backward pass, optimizer
  step, and why AdamW with LR warmup + cosine decay is the default choice
  here.
- Reading a loss curve sanity check: why initial loss should be close to
  `ln(vocab_size)` for a freshly initialized model, and what it means when
  it isn't.
- The real bug: tied embeddings + default `N(0,1)` init blew up logit
  std to ~11, producing a loss of ~94 instead of ~4.2 -- and the fix
  (explicit `N(0, 0.02)` init, standard GPT-2/Llama practice).
- Gradient clipping and checkpointing: what they protect against and why
  they're part of the loop from day one, not added later.
- Reading the Stage 1 result correctly: train loss falling while val loss
  rises is *expected* overfitting on a tiny corpus with a small model --
  not a failure, because pipeline correctness was the actual goal.
- Sampling basics: temperature and top-k, and what "good enough" looks like
  when judging output quality at this stage (Shakespeare-*shaped* output,
  not Shakespeare-*quality* output).

## Chapter 5 -- Proving the Export Pipeline Before Trusting It
- Why export gets de-risked on a throwaway model first: it's the least
  familiar step and the worst place to discover a surprise after investing
  real training time.
- The HF bridge: building a `LlamaConfig`/`LlamaForCausalLM` from a custom
  `ModelConfig` and loading a custom `state_dict` with `strict=True` --
  and why naming modules to match HF's `model.layers.N.self_attn.*` etc.
  from the start meant zero key remapping was ever needed.
- The SentencePiece detour: why a hand-rolled char vocab had to be forced
  through `convert_hf_to_gguf.py`'s SentencePiece path, and the two real
  llama.cpp internals that broke first try (`escape_whitespaces` replacing
  literal spaces with `▁`, and `add_space_prefix` defaulting to true).
- The faithfulness check that actually matters: bit-exact logit comparison
  between the custom model and bridged `LlamaForCausalLM` (not just "it
  produced plausible text") -- this is what proves the from-scratch forward
  pass *is* Llama's forward pass, not an approximation.
- The float non-associativity red herring: why greedy autoregressive
  decoding diverged between PyTorch and ggml even with correct math, and why
  teacher-forced per-position comparison is the right tool to isolate "is
  the math faithful" from "do two floating-point softmax kernels round
  identically."
- Getting a GGUF to actually load somewhere real (LM Studio) as the final
  proof that the pipeline works end to end.

## Chapter 6 -- Real Data Changes Everything
- Why swapping in a real, larger, structurally-cleaner dataset (TinyStories)
  before scaling the model isolates "does data help" from "does model size
  help."
- Practical large-dataset handling: range-GET-ing a byte budget from a
  multi-GB file instead of downloading it all, and trimming to the last
  complete record so training never sees a truncated example.
- Why train/val splits should come from genuinely separate source files
  when available, not a carved-off slice of the same file.
- Re-running the same architecture unchanged on new data as a controlled
  experiment, and reading the result: when train and val loss track closely
  and both keep falling, the model is data-bound, not overfitting -- a
  different diagnosis than Stage 1's, and what it implies for what to fix
  next.
- Special-token handling at scale: why `<|endoftext|>` needs to be
  registered as an explicit added token so it survives as a clean boundary
  signal instead of being split into BPE sub-tokens.

## Chapter 7 -- Scaling Up: Bigger Model, More Data, Real Stopping Criteria
- Growing a config deliberately and together: `dim`, `n_layers`, `n_heads`,
  `intermediate_size`, sequence length, and dataset size scaled as one
  decision, not independently -- and why scaling just one risks
  reintroducing overfitting or data-bound failure modes already diagnosed.
- The head_dim constraint: why `n_heads` must divide `dim` evenly and why
  RoPE specifically requires an even head_dim.
- De-risking a multi-hour run before committing to it: a timing smoke test
  to size step count realistically, and an LR stability smoke test on real
  data to catch instability before it wastes hours.
- Checkpointing on best validation loss instead of "whatever the last step
  happened to be" -- why this matters once a run plateaus or overfits late.
- An MPS-specific operational gotcha: buffered stdout on a redirected
  background process making a healthy training run look stalled, and how to
  actually verify a process is alive instead of guessing from an empty log.
- Reading a real loss curve over thousands of steps and recognizing a
  genuine plateau (LR approaching its floor) versus noisy individual evals.

## Chapter 8 -- Closing the Loop: Final Export, Quantization, Real Runtimes
- Why the tokenizer export path is often the actual blocking work at final
  export time, separate from "does the model weight bridge work."
- Moving from a hand-built SentencePiece workaround to the native format:
  wrapping a trained byte-level BPE tokenizer in `PreTrainedTokenizerFast`
  and letting llama.cpp's GPT-2 vocab path consume it directly.
- The pre-tokenizer hash mismatch: why a custom-trained BPE vocab can never
  match llama.cpp's table of known public tokenizer hashes even when the
  underlying regex algorithm is identical, and why falling back to a known
  pre-tokenizer type is a deliberate choice, not a guess.
- A layered verification strategy for trusting an export: bit-exact logits,
  then tokenizer parity probes (spaces, digits, contractions, punctuation)
  against the real GGUF tokenizer, then a `add_bos` sanity check against
  training-time behavior.
- Quantization in practice: running `q8_0`/`q4_0`, what size/quality
  tradeoff to expect, and why a tiny-dim tensor sometimes falls back to
  higher precision automatically.
- Getting a quantized GGUF running in a second independent runtime (LM
  Studio) as confirmation the file itself is correct, not just the tooling
  that produced it.

## Chapter 9 -- Teaching the Model to Follow Instructions (SFT)
- What instruction-tuning actually changes: same architecture, same
  vocabulary, a different *task* framing (respond to a prompt instead of
  continuing arbitrary prose).
- Loss masking with `ignore_index`: why the prompt is context, not a
  prediction target, and the exact boundary case that has to be handled
  correctly (the prompt's last token still predicts the first response
  token and must stay unmasked).
- A real parsing bug class: why a substring search for a boundary marker
  breaks when that marker can also appear inside free-text content, and why
  anchoring a regex to start-of-line fixes it correctly instead of
  papering over individual bad examples.
- Why truncating overlong examples is worse than dropping them outright when
  truncation risks silently changing what the model is conditioned on.
- Dynamic-padding batching and the MPS shape-compilation tradeoff: why
  padding every batch to its exact max length (instead of a fixed size or a
  rounded bucket) caused a 10-20x slowdown, and why bucketing to multiples
  of 64 fixes it.
- Verifying conditioning, not just fluency: holding one field fixed and
  varying another (e.g. same `Features`, different `Summary`) and checking
  that generations differ in exactly the way the varied field predicts --
  plus how to read a partial-conditioning miss (a detail dropped) versus a
  cosmetic dataset artifact (a name that doesn't match a gendered field).

## Chapter 10 -- Chat Templates and the Last Export Gap
- Why a chat template is part of the export contract too, not just a
  serving-side convenience -- the GGUF needs to carry the exact text shape
  the model was actually trained on.
- Why a generic role-based template (`<|user|>...<|assistant|>`) would
  "work" structurally but silently produce garbage: it tokenizes to a
  different sequence than anything the model has ever seen.
- Designing a template that reproduces training-time formatting exactly, and
  checking it against the real data-parsing code before wiring it into
  export -- not just eyeballing the Jinja.
- Verifying the template actually lands in GGUF metadata
  (`tokenizer.chat_template`) byte-for-byte, and that a real runtime
  (`llama-cli`) auto-detects it and enters conversation mode without extra
  flags.
- What "the model is now an assistant" really means at this scale: visible,
  traceable conditioning on varied instructions, with honest limits (not
  every clause of a complex instruction is guaranteed to be honored).

## Chapter 11 -- Speeding Up Generation: KV Caching
- Why naive generation recomputes the entire context at every step (O(n^2)
  total) and what a KV cache changes about that (O(n) per step).
- Cache tensor shapes `[batch, n_heads, seq, head_dim]` and why the cache is
  training/sampling-only -- it must never touch the non-cached forward path
  that the export bit-exactness check depends on.
- The decode-step trap: why a single new query token against a longer
  cached key/value set needs `is_causal=False`, and why getting this
  backwards silently corrupts generation instead of crashing.
- RoPE position offsets under caching: why the cache must track *absolute*
  position instead of re-basing to zero like the no-cache sliding window
  does.
- A real bug and why weak tests missed it: an off-by-layer cache read bug
  that didn't break greedy-argmax output (RoPE's uniform-shift invariance
  hid it during prefill) and was only caught by comparing raw per-position
  logits, not final tokens -- a lesson in what "the output looks right"
  does and doesn't prove.
- Measuring the actual payoff: confirming token-identical output between
  cached and non-cached greedy decode first, then measuring speedup
  (and why the speedup is expected to grow with sequence length).

## Chapter 12 -- Distillation: Training a Small Model to Punch Above Its Weight
- What knowledge distillation is: training a small "student" model to match
  a larger "teacher" model's output distribution (soft labels), not just
  the ground-truth next token (hard labels).
- Why distillation requires a shared vocabulary between teacher and student,
  and why that forces this project's teacher to be a bigger model trained
  on *our own* custom tokenizer rather than any off-the-shelf LLM.
- KL-divergence loss against softened teacher logits, temperature scaling,
  and combining a distillation loss term with the normal cross-entropy
  term.
- Online distillation versus precomputed teacher logits: why running the
  teacher live (frozen, in eval mode) in the same loop is the right call at
  this vocab/data scale instead of caching logits ahead of time.
- Designing a fair comparison: distilled student vs. same-size student
  trained from scratch, at identical size and step budget, judged on val
  loss and qualitative samples -- and why a modest quality gap is a real,
  reportable result rather than a failed experiment.

---

### Appendix ideas (not chapters, optional reference material)
- A running glossary of every term introduced (mirrors `glossary.md`).
- A "decisions log" cross-referencing each architecture/tokenizer choice in
  `LEARNING_PLAN.md` back to the chapter that explains the reasoning.
