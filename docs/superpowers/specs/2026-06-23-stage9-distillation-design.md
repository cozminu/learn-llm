# Stage 9 — Distillation from a Larger Teacher — Design

## Goal
Train a larger teacher on TinyStories with *our* tokenizer, distill it
into a small student, and show the student beats training the same-size
model from scratch. This is Stage 9 of `LEARNING_PLAN.md`.

## Three runs

1. **Teacher** (`src/train_teacher.py`, new) — new, larger `ModelConfig`:
   `dim=512, n_layers=10, n_heads=8, intermediate_size=2048` (head_dim=64,
   ~2x stage5's ~22M params). Trained on the same `tinystories_train.txt`
   (150MB), same 8000-step budget and warmup+cosine LR schedule shape as
   `src/train.py`. Checkpoint: `checkpoints/stage9_teacher.pt`.
2. **Distilled student** (`src/train_distill.py`, new) — fresh random
   init, **identical config to stage5** (`dim=384, n_layers=8, n_heads=6,
   intermediate_size=1536`), same data, same 8000 steps. Loads the frozen
   teacher checkpoint and trains against a combined hard-label +
   soft-target loss. Checkpoint: `checkpoints/stage9_distilled.pt`.
3. **From-scratch baseline** — `checkpoints/stage5.pt`, already trained
   (best val 1.5687 @ step 7400). Reused as-is, not retrained — it's
   already the same config/data/step-budget as the distilled student, so
   it's a valid control without spending another ~3hrs of compute.

Using stage5.pt as the baseline means the distilled student must **not**
warm-start from stage5's weights — it starts from random init like
stage5 did, so the only variable between the two runs is the loss
function (plain CE vs CE+KD).

## KD loss (`src/train_distill.py`)

Standard Hinton-style combined loss, computed manually per step (not
via `TinyLM.forward`'s built-in CE, since we need both the student's raw
logits and the teacher's):

```python
logits_s, _ = student(x)                      # no targets -> skip internal CE
with torch.no_grad():
    teacher.eval()
    logits_t, _ = teacher(x)

hard_loss = F.cross_entropy(logits_s.view(-1, vocab), y.view(-1))
kd_loss = F.kl_div(
    F.log_softmax(logits_s / T, dim=-1),
    F.softmax(logits_t / T, dim=-1),
    reduction="batchmean",
) * T**2
loss = alpha * hard_loss + (1 - alpha) * kd_loss
```

Defaults: `T=2.0`, `alpha=0.5` (equal weight between matching the true
next token and matching the teacher's softened distribution — standard
starting point, no prior tuning signal to deviate from).

Online distillation: teacher and student share one training loop, teacher
frozen in `eval()` + `no_grad()`. No logit caching/precomputation — at
this vocab size (~8k) and data size, the storage/complexity of caching
isn't worth it (matches the plan's stated reasoning).

## De-risking before the long runs

Same pattern as Stage 5's de-risking, applied to the teacher (the new,
untested config):
1. **Timing smoke test**: a few hundred steps of the teacher config on
   real tokenized data, measure s/step, project total wall-clock for
   8000 steps before committing.
2. **LR stability smoke test**: ~150 steps, confirm loss falls smoothly
   with no spikes/NaN at the chosen LR (start from stage5's 1e-3, adjust
   only if the smoke test shows instability — bigger models sometimes
   need a lower peak LR).

The distilled student reuses stage5's already-validated config/LR/data
sizing, so it does not need its own smoke test — only the KD loss
mechanics need a short correctness check (a few dozen steps, confirm
`hard_loss` and `kd_loss` are both finite and of comparable magnitude,
not one drowning out the other).

## Verification

A controlled A/B, not a new verification script — extend
`src/sample.py` usage (no code changes needed there) to generate from
both `stage9_distilled.pt` and `stage5.pt` with the same prompt and
greedy decoding for the qualitative side.

1. **Quantitative**: val loss for `stage9_distilled.pt` vs `stage5.pt`'s
   recorded best val loss (1.5687), same val set
   (`tinystories_val.txt`), same eval procedure as `train.py`'s
   `estimate_loss`.
2. **Qualitative**: same prompt (`"Once upon a time, there was a"`),
   greedy decode, both checkpoints — read side by side for coherence/
   variety differences.

## Success bar (from LEARNING_PLAN.md, unchanged)
A clean A/B: student-distilled vs student-from-scratch at identical
size/step budget, val loss (and a qualitative sample) reported for both.
Honest framing carried over from the plan: on a clean, non-data-bound
task like TinyStories, the quality jump may be modest — that's a real
result, not a failed stage.

## Files touched
- New: `src/train_teacher.py`, `src/train_distill.py`
- New data: `checkpoints/stage9_teacher.pt`, `checkpoints/stage9_distilled.pt`
- Unchanged: `src/model.py`, `src/config.py`, `src/train.py`,
  `checkpoints/stage5.pt` (reused as the from-scratch baseline, not
  retrained)
