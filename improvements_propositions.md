# Improvement propositions

Discussion notes on possible next steps for the model, given the current
state: Llama-shaped TinyLM (dim=128, 4 layers, 4 heads, vocab from custom
BPE), trained on TinyStories, with KV cache (Stage 7), SFT (Stage 8), and
self-distillation from a larger same-vocab teacher (Stage 9) already done.

1. **GQA (grouped-query attention)** -- fewer KV heads than query heads.
   Cheap win on cache memory/bandwidth, and it's what real small
   Llama-family models (Llama-3, Mistral) actually use, so it'd close the
   gap with "real" architectures.

2. **RoPE scaling / longer context** -- `max_seq_len=256` is tight. Trying
   YaRN or simple linear scaling to push context further is a good
   isolated concept lesson (interpolating rotary frequencies vs.
   extrapolating).

3. **Better tokenizer coverage** -- byte-level BPE is fine, but checking
   vocab efficiency (avg tokens/word) on TinyStories-Instruct vs. a
   slightly larger vocab could meaningfully cut sequence length and
   training cost.

4. **Weight decay / AdamW hyperparameter sweep** -- at this scale a quick
   LR/warmup/weight-decay sweep often buys more loss improvement than
   architecture changes, and it's a cheap, fast experiment given the
   model trains quickly on MPS.

5. **Mixture of data -- replay/curriculum** -- interleave TinyStories base
   text with TinyStories-Instruct during SFT (not just fine-tune-then-stop)
   to fight catastrophic forgetting of the base language modeling
   ability.

6. **Quantization-aware eval** -- already quantize to q4_0/q8_0 for
   Ollama; measuring perplexity or sample quality before vs after
   quantization would teach where the model is fragile to precision
   loss.

7. **Multi-query/no-bias attention variants comparison** -- small
   ablation: MHA vs GQA vs MQA at fixed param budget, comparing val loss
   -- a controlled architecture experiment rather than just "pick one."

8. **Speculative decoding using the small model as drafter for the bigger
   teacher** -- already have a same-vocab teacher/student pair from
   Stage 9, which is exactly the setup speculative decoding needs.
   Natural Stage 10 candidate.

9. **Better sampling -- repetition penalty / top-p / min-p** -- check
   `sample.py`; TinyStories models are notorious for looping. Nucleus or
   repetition penalty sampling is a small, high-payoff addition for
   perceived quality.

10. **Eval harness beyond loss** -- right now "success" is loss + a
    qualitative sample. Adding something like a held-out perplexity table
    across stages, or a tiny multiple-choice/cloze eval (e.g. "does it
    pick the more plausible next word"), gives a number to track
    regressions/improvements by as further tweaks are made.
