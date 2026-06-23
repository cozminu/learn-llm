# Stage 1 -- vertical slice

Char-level tokenizer (vocab=65) over tiny-shakespeare. Model: 4 layers,
dim=128, 4 heads, intermediate=512, tied embeddings -- 1.06M params.

Bug hit: initial loss was ~94 instead of ln(65)~4.2. Cause: `nn.Embedding`
default init is `N(0,1)`, and since lm_head is tied to it, logits came out
with std~11 (dim=128 dot product of two std=1 vectors), blowing up cross
entropy. Fix: explicit `N(0, 0.02)` init on all Linear/Embedding weights
(standard GPT-2/Llama practice) in `TinyLM._init_weights`.

2000 steps, batch 64, block 256, AdamW lr 3e-3 with warmup+cosine decay,
~257s on MPS. train loss 4.2 -> 0.86, val loss plateaus/rises after step
~1000 (expected overfitting on a 1M-char corpus with a 1M-param model --
not a concern for this stage, the point was pipeline correctness).

Sample output is Shakespeare-shaped: correct character names, dialogue
format, period-appropriate vocabulary and punctuation. Good enough to
move to Stage 2 (export de-risking).
