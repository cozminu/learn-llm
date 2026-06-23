# Stage 8 — Instruction-Tuning (SFT) — Design

## Goal
Fine-tune `checkpoints/stage5.pt` (the Stage 5 TinyStories base model) on
TinyStories-Instruct so it follows a simple instruction format instead of
just continuing prose, and confirm the chat template survives re-export
to GGUF in a real runtime. This is Stage 8 of `LEARNING_PLAN.md`.

## Data

Source: `roneneldan/TinyStories-Instruct` (HF dataset). Download a 20MB
train subset via byte-range request (same technique as
`scripts/download_data.py`'s `download_train_subset`) plus the full val
split, saved to `data/tinystories_instruct_train.txt` and
`data/tinystories_instruct_val.txt`.

### Raw format (confirmed by inspection)
Examples are separated by `<|endoftext|>`. Each example is a random
subset of header fields (any of `Words:`, `Features:`, `Random
sentence:`, `Summary:`, in random order — not fixed), followed by a
`Story:` marker, followed by the story text:

```
Features: Dialogue, BadEnding
Summary: Ben and Mia get lost in a subway station...
Random sentence: They wished they had never been eager to ride the subway.
Story: 
Ben and Mia were eager to ride the subway with their mom...
```

### Token length (measured on a 5MB sample of val, our BPE tokenizer)
- mean 251 tokens, p95 385, max seen 1090, ~2% of examples exceed 512
  (the model's `max_seq_len`).

## Parsing (`src/sft_data.py`, new)

For each `<|endoftext|>`-separated block:
1. Locate the `Story:` marker with a line-start-anchored regex (plain
   substring search is wrong — the word "story" can appear inside a
   `Summary:` sentence).
2. Split into `prompt` = everything up to and including `Story:` plus
   its trailing whitespace/newline, and `response` = the story text
   that follows.
3. Build `ids = encode(prompt) + encode(response) + [eos_id]`.
4. Drop the example if `len(ids) > max_seq_len + 1` (the ~2% over 512
   tokens) — simplest handling, negligible data loss.
5. `input_ids = ids[:-1]`, `targets = ids[1:]` with every target
   position covered by `prompt` set to `-1` (uses the already-wired
   `ignore_index=-1` in `TinyLM.forward` — currently dead code, this is
   the first stage to exercise it).

## Batching

New collate function in `src/sft_data.py`: given a list of
`(input_ids, targets)` pairs, pad every sequence in the batch to that
batch's own max length (not a fixed 512) — inputs padded with `eos_id`,
targets padded with `-1`. No model/attention changes needed: causal
masking already means real tokens never attend to trailing padding, and
`-1` targets contribute zero loss.

## Training (`src/train_sft.py`, new)

Mirrors `src/train.py`'s structure but:
- Loads `checkpoints/stage5.pt`'s `state_dict` into a freshly
  constructed `TinyLM` (same `ModelConfig`) instead of random init.
- Fresh `AdamW`, its own warmup+cosine schedule, peak LR ~2e-4 (lower
  than Stage 5's 1e-3 — adapting a converged model, not training from
  scratch), ~1000-1500 steps total.
- Batches drawn by sampling instruct examples (not the sliding-window
  `get_batch` over a concatenated token stream that Stage 5 uses — these
  are discrete examples, not a continuous corpus) and passed through the
  new collate function.
- Val loss computed the same masked way over held-out instruct examples;
  best-val checkpoint saved to `checkpoints/stage8.pt`.

## Chat template (export)

`scripts/export_hf.py` gains an optional `chat_template` string passed
into `PreTrainedTokenizerFast` — only used when exporting `stage8.pt`.
The template reproduces the exact training-time text shape: a `user`
message renders as the raw header block, followed by `Story: `; an
`assistant` message renders as just the story text. This way
`tokenizer.apply_chat_template(...)` in any GGUF-consuming runtime
produces text identical in shape to what the model was trained on.

## Verification (`scripts/verify_sft.py`, new)

1. **Conditioning check**: hold the `Words:`/`Features:` fields fixed
   across a few prompts, vary only `Summary:`, generate with each, and
   confirm the generated stories track the varied summary content (not
   just "looks like a TinyStories story" — that would pass even without
   any instruction-following).
2. **Export check**: re-run the Stage 2/6 export pipeline
   (`scripts/export_gguf.py`) on `stage8.pt`, inspect the resulting
   GGUF's metadata for a populated `tokenizer.chat_template` key, and
   confirm a real runtime (llama.cpp CLI `--chat-template` or LM Studio)
   applies it and produces a story conditioned on the given instruction.

## Success bar (from LEARNING_PLAN.md, unchanged)
The SFT model visibly follows a held-out instruction (not just
continuing it as prose), and the re-exported GGUF applies the chat
template in a real runtime.

## Files touched
- New: `src/sft_data.py`, `src/train_sft.py`, `scripts/verify_sft.py`
- New data: `data/tinystories_instruct_{train,val}.txt`,
  `checkpoints/stage8.pt`, `checkpoints/stage8-hf/`, `checkpoints/stage8.gguf`
- Modified: `scripts/download_data.py` (or a sibling
  `download_instruct_data.py`) for the instruct download, `scripts/export_hf.py`
  (optional `chat_template` param)
- Unchanged: `src/model.py`, `src/config.py`, `src/tokenizer_bpe.py`,
  `scripts/export_gguf.py` (no logic changes, just invoked on `stage8.pt`)
