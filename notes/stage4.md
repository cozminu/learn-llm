# Stage 4 -- real dataset (TinyStories)

Goal: swap Shakespeare for a real, larger, structurally-cleaner dataset
and retrain the *same* architecture (no scaling yet -- that's Stage 5).

## Getting the data
`roneneldan/TinyStories` on the HF Hub ships as one 1.9GB train file --
too big to fully download for a "subset." `scripts/download_data.py`:
- HTTP range-GETs the first N bytes (default 20MB) of
  `TinyStories-train.txt` directly (via `hf_hub_url` + a `Range` header,
  no need to pull the whole file or use the heavier `datasets` library),
  then trims to the last complete `<|endoftext|>`-delimited story so we
  don't end on a truncated one.
- Downloads the official `TinyStories-valid.txt` (19MB) in full as a
  genuine held-out split, rather than carving our own val set out of the
  train subset -- keeps train/val honest.
- Result: `data/tinystories_train.txt` (~22k stories, 20MB) and
  `data/tinystories_val.txt` (~22k stories, 19MB).

## Tokenizer changes
Retrained `scripts/train_tokenizer.py` on the new corpus, vocab bumped
4096 -> 8000 (richer vocabulary than archaic Shakespeare English). One
real fix needed: `<|endoftext|>` (the story separator) was being split
into 9 byte-level sub-tokens (`<`, `|`, `end`, `o`, `ft`, `e`, `xt`, `|`,
`>`) by plain BPE, which wastes capacity and destroys it as a clean
boundary signal. Fixed with `tokenizer.add_special_tokens(["<|endoftext|>"])`
after training -- this registers it in the tokenizer's `added_vocabulary`,
which intercepts exact-match substrings *before* the byte-level
pre-tokenizer runs, regardless of whether it survived BPE merges. Special
tokens are also auto-excluded from `decode()` by default, which is the
behavior we want for sampling (don't show `<|endoftext|>` mid-story).

## Training loop changes
`train.py` now reads `tinystories_train.txt` / `tinystories_val.txt` as
two separate files instead of carving 90/10 out of one (matches the real
train/val split decision above). `MAX_STEPS` bumped 2000 -> 3000,
`WARMUP_STEPS` 100 -> 150 -- architecture, learning rate, and batch/block
size all unchanged, per the stage goal of isolating "does a real dataset
help" from "does a bigger model help" (Stage 5).

## Result -- the overfitting from Stage 3 is gone
Stage 3 (4MB effective Shakespeare text via BPE) showed train loss
diving to 0.27 while val loss climbed to 9.5 by step 2000 -- classic
overfitting on too little data. Same architecture on TinyStories
(~4.8M train tokens):

```
step  300 | train 2.8730 | val 2.9500
step 1500 | train 2.0805 | val 2.2978
step 3000 | train 1.8607 | val 2.2029
```

Train and val track each other closely the entire run and both are
still trending down at step 3000 -- the dataset is big enough relative
to this tiny model that we're data-bound, not overfitting-bound. That's
the real payoff of this stage: more data exposed that the *previous*
bottleneck was corpus size, not architecture.

## Qualitative check
Sampled from `stage4.pt` with prompt `"Once upon a time, there was a"`:
coherent multi-sentence stories, consistent named characters across
sentences, plausible (if sometimes logically wobbly) TinyStories-style
plots, and generation correctly produces one full story and starts a
fresh one at each learned `<|endoftext|>` boundary. Real qualitative
jump from Stage 1/3's Shakespeare-shaped character soup.

## Carried forward to Stage 5
- Train/val gap still small and both still decreasing at step 3000 --
  the model is currently under-trained *and* likely under-capacity for
  this corpus. Stage 5 should grow the config (dim/layers/heads) and
  also just train longer past 3000 steps, watching for val loss to
  plateau as the actual stopping signal (rather than a fixed step count
  picked for a tiny corpus, as flagged in Stage 3's notes).
- `download_data.py`'s byte-budget is a knob (`sys.argv[1]`) -- if Stage
  5 wants more data for a bigger model, just rerun with a larger number;
  the range-GET approach scales to that without code changes.
