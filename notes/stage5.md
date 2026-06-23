# Stage 5 -- scale up

Goal: grow the architecture and dataset together (carried forward from
Stage 4's "data-bound, not overfitting-bound" finding) and train to a
real loss-based stopping point instead of a fixed step count picked for
a smaller setup.

## What changed
- **Data**: `tinystories_train.txt` regenerated at 150MB (was 20MB) via
  `download_data.py 150000000` -- ~37M BPE tokens. Sized to roughly match
  the step budget below (8000 steps x 32×512 tokens/step ≈ 131M tokens
  sampled, so each token gets visited ~3.5x on average) rather than
  pulling the full 1.9GB file, most of which wouldn't get touched at this
  step count anyway.
- **Model** (`train.py`): dim 128->384, n_layers 4->8, n_heads 4->6
  (head_dim 64, still even -- RoPE requires this), intermediate_size
  512->1536. ~22M params (was ~1.5M), keeping the same SwiGLU ratio (4x
  dim) and tied embeddings as before.
- **Sequence/batch**: BLOCK_SIZE 256->512, BATCH_SIZE 64->32 (longer
  sequences cost more memory per example).
- **Schedule**: LR 3e-3->1e-3 (the old LR was tuned for a 5M-param toy;
  smoke-tested on real data first -- see below), WARMUP_STEPS 150->300,
  MAX_STEPS 3000->8000, EVAL_INTERVAL 300->200.
- **Checkpointing**: `train.py` now tracks `best_val` and only overwrites
  `stage5.pt` when val loss improves at an eval point, instead of saving
  once at the final step. Final print reports the best val loss and which
  step produced it. This matters once a run plateaus or overfits late --
  the saved file reflects peak generalization, not whatever the last step
  happened to be.

## De-risking before the long run
A naive launch at this scale risked wasting hours on a bad guess, so two
checks ran first:
1. **Timing smoke test** (random data, scaled config): ~1.05s/step on
   MPS -> projected ~8000 steps to ~2.5-3hrs, used to size MAX_STEPS
   rather than guessing.
2. **LR stability smoke test** (150 steps on real tokenized data): loss
   fell smoothly 7.8 -> 3.8 with no spikes/NaN at LR=1e-3, confirming the
   new LR is safe before committing to the multi-hour run.

Also confirmed before launch (not blocking, but the thing that would have
hurt most to discover at Stage 6): `export_hf.py` builds `LlamaConfig`
entirely from `ModelConfig` fields (`hidden_size=cfg.dim`,
`num_hidden_layers=cfg.n_layers`, etc.) -- scaling the config doesn't
touch the export path's correctness. (Its *tokenizer* export is still
the Stage 1 char/SentencePiece path -- known stale, explicitly deferred
to Stage 6 per Stage 3's notes, not a Stage 5 concern.)

## One operational snag: buffered stdout
First launch (plain `python src/train.py > log.txt 2>&1 &`) appeared
stalled -- log file stayed empty for 12+ minutes. It wasn't stuck: a
process sample (`sample <pid>`) caught it mid-optimizer-step actively
running MPS graphs, and `checkpoints/stage5.pt` already existed on disk.
Root cause: redirecting stdout to a file (vs a TTY) switches Python to
full buffering, and at one print per 200 steps the buffer doesn't fill
for a long time. Fixed by relaunching with `PYTHONUNBUFFERED=1`. Worth
remembering for any future long background run -- check the buffering,
not just "is the log empty," before assuming a hang.

## Result
```
step  1000 | train 2.0669 | val 2.0920
step  3000 | train 1.6946 | val 1.7679   (Stage 4 final, for comparison: train 1.8607 | val 2.2029)
step  5000 | train 1.5183 | val 1.6469
step  7400 | train 1.4022 | val 1.5687   <- best val, saved checkpoint
step  8000 | train 1.3975 | val 1.5844   (final step, not the saved checkpoint)
```
Train/val gap stayed small (~0.05-0.15) for the whole run -- still not
overfitting, confirming the 150MB bump was sized correctly for a 22M
model. Genuine plateau set in only in the last ~800 steps as LR
approached its floor; a few individual evals ticked up before that but
didn't hold as a trend (the best-val checkpointing logic exists exactly
to not care which of those was noise).

## Qualitative check
Sampled `stage5.pt` with `"Once upon a time, there was a"`: multiple
complete, coherent stories with consistent character names/pronouns
within each story, natural quoted dialogue, and correct generation of
fresh stories at each `<|endoftext|>` boundary. Clear step up from Stage
4 -- still occasional non-sequitur lines ("Don't worry, Lily. Everyone
says so.") but notably fewer and less jarring than Stage 4's output.

## Carried forward to Stage 6
- `stage5.pt` is the model to re-run the Stage 2 export pipeline on.
- `export_hf.py`'s tokenizer export path is stale (char/SentencePiece,
  not BPE) -- this is the actual blocking work for Stage 6, not the model
  side of the export.
- If a future stage wants more headroom, the data/model were sized
  together deliberately (150MB train, ~22M params, 8000 steps) --
  scaling one without the others risks re-introducing either the
  overfitting (Stage 3) or data-bound (Stage 4) failure modes already
  diagnosed.
