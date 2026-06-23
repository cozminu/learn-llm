# Stage 6 -- final export

Goal: re-run the Stage 2 export pipeline on the real trained model
(`stage5.pt`), quantize, and confirm it loads and generates in a real
runtime.

## What changed in the export scripts
Stage 5's notes flagged the actual blocking work: `export_hf.py` still
exported the Stage 1 char/SentencePiece tokenizer, not the BPE one
`train.py`/`sample.py` had used since Stage 3.

- **`export_hf.py`**: dropped the char tokenizer + hand-built
  SentencePiece `ModelProto` entirely. Now wraps `checkpoints/tokenizer.json`
  (our trained byte-level BPE) in a `PreTrainedTokenizerFast` and
  `save_pretrained()`s it alongside the bridged `LlamaForCausalLM` --
  produces a real `tokenizer.json` + `tokenizer_config.json`, the native
  format llama.cpp's GPT-2 vocab path expects. `bos_token_id`/`eos_token_id`
  both point at `<|endoftext|>` (id 0), matching how the model was trained
  (no separate BOS).
- **`export_gguf.py`**: dropped both Stage 2 SentencePiece-specific
  workarounds (`▁` space piece, `add_space_prefix` metadata patch) --
  neither applies to the GPT-2/BPE vocab path. Added one new workaround
  instead (see below) and switched from subprocess-invoking
  `convert_hf_to_gguf.py` to importing and calling it in-process, since the
  workaround requires monkeypatching before conversion runs.

## The one new workaround: unrecognized pre-tokenizer hash
`convert_hf_to_gguf.py`'s `get_vocab_base_pre()` identifies a tokenizer's
pre-tokenizer by hashing the *token ids* of a fixed probe string and
matching against a table of hashes from known public tokenizers. Since our
BPE vocab is custom-trained (different merges than any public model), the
hash can never match a known entry, even though our pre-tokenizer
algorithm (`tokenizers.pre_tokenizers.ByteLevel`, `use_regex=True`) *is*
byte-for-byte the GPT-2 regex split -- the hash check conflates
"pre-tokenizer regex" with "exact trained vocab," and only the regex
needs to match.

Fix (`export_gguf.py:patch_unrecognized_pretokenizer`): monkeypatch
`TextModel.get_vocab_base_pre` to catch the `NotImplementedError` it raises
on no match and fall back to `"gpt-2"` -- a deliberate choice based on
knowing our pre-tokenizer config, not a guess to silence an error.

## Verification
1. **Export + convert**: `stage5.pt` -> `stage5-hf/` -> `stage5.gguf`
   (f16, 43.9 MB). Conversion log shows the expected pre-tokenizer warning
   (harmless, explained above), then `Adding 7743 merge(s)` and
   `Setting special token type bos/eos to 0` -- confirms the real BPE
   vocab and merges made it into the GGUF natively, no hand-rolled vocab
   structures this time.
2. **Bit-exactness** (carried over from Stage 2's keystone check, re-run
   since the model weights are new even though the bridge code path is
   unchanged): `TinyLM` vs bridged `LlamaForCausalLM`, same input ids --
   max abs logit diff `0.0`, top-1 agreement `1.0`.
3. **Tokenizer parity** (new for Stage 6 -- the bit-exactness check alone
   doesn't cover this, since it starts from already-identical token ids):
   compared `BPETokenizer.encode()` against `llama_cpp.Llama(...,
   vocab_only=True).tokenize()` on strings probing leading/trailing
   spaces, digit runs, contractions, runs of whitespace, and punctuation.
   All matched exactly -- confirms the `"gpt-2"` pre-tokenizer fallback
   was the *correct* choice, not just one that didn't crash.
4. **`add_bos` sanity check**: the GGUF doesn't carry an explicit
   `tokenizer.ggml.add_bos_token` override (our tokenizer_config.json has
   no `add_bos_token` field for `_set_vocab_gpt2`/`SpecialVocab` to pick
   up). llama.cpp's default for the `gpt-2` pre-type is `add_bos = false`
   (confirmed by reading `llama-vocab.cpp`'s pre-type dispatch table, then
   empirically: `llama_cpp.Llama.tokenize(..., add_bos=True)` does not
   prepend token 0). This matches training, which never prepended a
   leading token -- not an accidental gap.
5. **Generation, f16**: `llama-completion -m stage5.gguf -p "Once upon a
   time, there was a"` -- coherent, multi-sentence story, matching the
   PyTorch model's known quality.
6. **Quantization**: `llama-quantize` to `q8_0` (23.6 MB) and `q4_0`
   (14.2 MB, 1 tensor fell back to a higher-precision quant as usual for
   very small dims). Both re-tested with `llama-completion` -- still
   coherent, no degradation visible at this size.
7. **LM Studio**: `lms import -y -c --user-repo learn-llm/stage5
   stage5-q8_0.gguf` (plain filesystem copy into `~/.lmstudio/models/`
   without `lms import` left the model un-indexed -- `lms load` hung
   waiting on a model key it couldn't resolve). Loaded successfully,
   queried via the local OpenAI-compatible `/v1/completions` API --
   coherent multi-sentence completion, confirming the full GGUF (not just
   the llama.cpp CLI) round-trips correctly outside our own tooling.

## Deferred
Ollama wasn't installed on this machine. Wrote `checkpoints/Modelfile`
(`FROM ./stage5-q8_0.gguf` + sampling params) so `ollama create` / `ollama
run` is a one-command check once Ollama is installed -- not verified yet,
by explicit choice (LM Studio's verification was judged sufficient for
now since it already exercises the same GGUF file through an independent
runtime).

## Result
Stage 6 complete: `stage5.pt` exports faithfully through the full
pipeline (HF bridge -> native BPE GGUF conversion -> quantization -> real
runtime load and generation), with the tokenizer path now genuinely
native rather than the Stage 1/2 SentencePiece workaround. This closes
the original goal: a from-scratch Llama-shaped model, trained start to
finish in raw PyTorch, loads and generates in a GGUF-consuming runtime.
