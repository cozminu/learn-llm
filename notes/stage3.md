# Stage 3 -- real tokenizer (BPE)

Goal: swap the char tokenizer for a real HF BPE tokenizer and confirm the
Stage 1 training loop still works unchanged with the new vocab. Still
training on `shakespeare.txt` -- TinyStories is Stage 4, not yet.

## What was built
- `scripts/train_tokenizer.py`: trains a **byte-level BPE** tokenizer
  (GPT-2 style: `pre_tokenizers.ByteLevel` + `decoders.ByteLevel`,
  `BpeTrainer` seeded with the full byte-level alphabet) on
  `data/shakespeare.txt`, vocab_size=4096, saved to
  `checkpoints/tokenizer.json`.
- `src/tokenizer_bpe.py`: thin `BPETokenizer` wrapper around HF
  `tokenizers.Tokenizer` (`encode`/`decode`/`vocab_size`/`load`), same
  shape as `CharTokenizer` so `train.py`/`sample.py` swap with a one-line
  import change.
- `train.py`/`sample.py`: now load the pre-trained `tokenizer.json`
  instead of building a tokenizer from the corpus inline (BPE training
  takes a real pass over the data, unlike char vocab which is just
  `set(text)`). Checkpoint renamed `stage1.pt` -> `stage3.pt` so the
  Stage 1/2 throwaway model (already proven through export, loaded in LM
  Studio) isn't overwritten.

## Why byte-level BPE specifically
llama.cpp's `convert_hf_to_gguf.py` has two vocab paths: SentencePiece
(`_set_vocab_sentencepiece`) and GPT-2-style BPE (`_set_vocab_gpt2`). The
char tokenizer was forced through the SentencePiece path by hand-building
a `ModelProto` (Stage 2's two workarounds: `▁` for space,
`add_space_prefix=false`). A byte-level BPE `tokenizer.json` is the
*native* format the GPT-2 path expects, so Stage 6's re-export should not
need either Stage 2 workaround. (Not re-verified yet -- export happens
again in Stage 6 on the real trained model.)

## Verification
Ran the unchanged training loop (same hyperparameters, `MAX_STEPS=2000`)
with the new tokenizer:
- `vocab=4096` (vs ~70 for char), `params=1,574,016` (embedding/lm_head
  grew, rest of the architecture untouched).
- Train loss fell to 0.27 by step 2000; val loss rose after step ~400
  (overfitting). Expected and not a bug: BPE packs ~4-5x more characters
  per token, so the same step count now does many more epochs over the
  same byte count of text. Not a quality run -- just confirms the pipeline
  (encode -> batch -> forward -> loss -> backward) works with a real
  multi-character vocab.
- Sampled from the trained model (`sample.py "ROMEO:"`): output is
  real, correctly-spelled, correctly-spaced English words in
  Shakespeare-play format (character names, colons, line breaks) --
  confirms BPE encode/decode roundtrips correctly through generation,
  not just on static text.

## Carried forward to Stage 4
- `data/shakespeare.txt` gets replaced by a TinyStories subset.
- Tokenizer vocab size (4096) and training script are reusable as-is --
  just point `train_tokenizer.py` at the new corpus and retrain.
- Overfitting-by-step-2000 behavior is a Stage-1/3 artifact (fixed step
  count tuned for a tiny corpus); Stage 4 needs its own loss-based
  stopping criterion now that there's enough data to actually need one.
