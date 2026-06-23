# Glossary

A from-scratch reference for every ML/technical term actually used while
building this project, as recorded in `LEARNING_PLAN.md` and `notes/stage1.md`
through `notes/stage8.md`. Alphabetical, one term per entry, self-contained --
cross-references point to other terms by name. Distinct from
`course/glossary.md`, which tracks terms from the standalone course outline
rather than this project's real build history.

---

### `add_bos` / `add_bos_token`

A tokenizer/runtime setting controlling whether a beginning-of-sequence token
is automatically prepended before tokenizing a prompt. (Stage 6) Verified
that this project's GGUF has no explicit `tokenizer.ggml.add_bos_token`
override, so llama.cpp falls back to its `gpt-2` pre-type default of
`add_bos = false` -- which matches training, since this model never
prepended a leading token. A mismatch here would silently feed the model an
out-of-distribution input shape without crashing.

### `add_space_prefix`

A SentencePiece tokenizer option, defaulting to `true`, that prepends a dummy
leading space to input text before tokenization. (Stage 2) Broke the
hand-rolled char-tokenizer SentencePiece export until patched to `false` via
llama.cpp's `gguf_new_metadata.py:copy_with_new_metadata` helper. Stopped
being relevant once Stage 6 switched to native byte-level BPE export, since
that path doesn't go through SentencePiece at all.

### `AdamW`

The optimizer used throughout training (Stage 1 onward). Adam (adaptive
per-parameter learning rates from gradient history) with weight decay
decoupled from the gradient update, which makes decay behave as a cleaner
regularizer than plain L2-penalty-via-gradient. Default choice here because
it tolerates the noisy, sparse-ish updates (e.g. embedding rows) typical of
transformer training.

### `argmax decoding` / `greedy decoding`

Picking the single highest-probability token at every generation step instead
of sampling from the distribution. Used as the verification mode in Stage 2
(autoregressive divergence investigation), Stage 7 (KV cache token-identical
check), and Stage 8 (conditioning check) because it's deterministic --
removes sampling randomness as a variable when comparing two
implementations or two prompts.

### `attention score matrix`

The intermediate `(seq, seq)` matrix per head, per batch element, computed as
`Q @ K^T / sqrt(head_dim)` inside self-attention (Stage 1). Entry `[i, j]` is
how much query position `i` attends to key position `j` before softmax.
**Causal masking** zeroes out entries where `j > i`.

### `autocast`

A PyTorch context manager (`torch.autocast`) that runs ops in lower precision
(bf16 on this project's MPS setup -- never fp16, per `CLAUDE.md`) while
keeping master weights in fp32, trading some precision for speed/memory.
Relevant from Stage 1's training loop onward.

### `backward()` / backpropagation

The PyTorch call (`loss.backward()`) that computes the **gradient** of the
loss with respect to every parameter that has `requires_grad=True`, by
walking the computation graph from the loss back to the inputs (the chain
rule, applied automatically). After this call, every parameter tensor has a
freshly populated `.grad` attribute; nothing has been updated yet -- that's
`optimizer.step()`'s job (Stage 1's training loop, `src/train.py`).

### `batch`

A group of independent training examples processed together in one forward
and backward pass, stacked into a single tensor's leading dimension (e.g.
`x` of shape `(batch=32, seq=512)` in `src/train.py:get_batch`). Batching
exists for GPU/MPS efficiency -- one matrix multiply over 32 sequences at
once is far cheaper than 32 separate calls -- and for gradient stability,
since averaging the loss over more examples gives a less noisy update
direction.

### `bf16` (bfloat16)

The 16-bit floating-point format used for autocast on this project's MPS
backend, preferred over fp16 specifically because Apple Silicon / MPS and
this project's conventions avoid fp16's narrower exponent range. Never
substituted with float64 either -- both extremes are off the table per
project convention.

### `bit-exact` / `bit-exactness check`

The keystone faithfulness test (Stage 2, re-run in Stage 6): running the
custom `TinyLM` and the HF-bridged `LlamaForCausalLM` on identical
`input_ids` in f32, eval mode, and checking that logits match with `0.0` max
absolute difference and 100% top-1 argmax agreement. Proves the from-scratch
forward pass *is* Llama's forward pass, not merely producing similar-looking
output -- the strongest available check before trusting an export pipeline.

### `BOS` / `EOS`

Beginning-of-sequence / end-of-sequence tokens -- special vocabulary entries
marking the start or end of a text span. This project uses `<|endoftext|>`
for both roles (id 0), with no separate BOS token ever prepended during
training (Stage 4, Stage 6).

### `BpeTrainer`

The HF `tokenizers` class that learns BPE merge rules from a corpus (Stage
3), seeded with the full byte-level alphabet so every byte value starts
covered before any merges run. Distinct from `pre_tokenizers.ByteLevel`,
which only governs how raw text is pre-split before merging, not what the
merges themselves are.

### `byte-level BPE`

BPE applied to raw UTF-8 bytes (256 base symbols) rather than Unicode
characters (Stage 3) -- the GPT-2/Llama-style approach used via HF
`tokenizers`' `pre_tokenizers.ByteLevel` + `decoders.ByteLevel`. Guarantees no
out-of-vocabulary token ever occurs, since any string decomposes into bytes
the vocabulary already covers. This is also why TinyStories-Instruct (Stage
8) "encodes cleanly" under this tokenizer -- there's no OOV risk regardless
of domain.

### `catastrophic forgetting`

The phenomenon where fine-tuning a model on a new data distribution (e.g.
**SFT** on TinyStories-Instruct) degrades its competence on the distribution
it was originally trained on (base TinyStories language modeling), because
gradient updates pull shared weights toward the new task without anything
forcing them to stay good at the old one. Mitigated by **replay/curriculum
mixing** -- interleaving old and new data during fine-tuning rather than
training on the new data exclusively -- or by lowering the fine-tuning
learning rate/duration. Not yet measured in this project as of Stage 9; a
candidate next step is checking base-LM val loss before/after Stage 8's SFT
run to see whether it actually happened here.

### `causal masking`

The constraint that token position `i` can only attend to positions `<= i`,
never `> i` (Stage 1), implemented by adding `-inf` to disallowed entries of
the **attention score matrix** before softmax. Necessary because at
generation time the model genuinely cannot see future tokens; see `is_causal`
for the flag controlling this, and the **decode-step trap** for the one case
where it must be turned off.

### `chat template`

A Jinja-style string embedded in tokenizer/model metadata that defines how a
list of role-tagged messages (`user`, `assistant`) gets serialized into the
exact text string fed to the model (Stage 8, `export_hf.CHAT_TEMPLATE`). Part
of the export contract, not a UI convenience: this project's template was
deliberately written to reproduce the *exact* training-time text shape
(`{{ content }}\nStory: \n` for user messages) rather than a generic
`<|user|>...<|assistant|>` template, because a structurally-valid but
differently-tokenized template would silently produce degraded completions.
Verified embedded in the exported GGUF's `tokenizer.chat_template` field,
byte-for-byte equal to the source string, and auto-detected by `llama-cli`
without any extra flags.

### `checkpoint` (model)

A saved snapshot of model weights (and training step context) written to
`checkpoints/` (gitignored). From Stage 5 onward, checkpointing only on
**best validation loss** improvement (tracked via `best_val`) rather than
overwriting at every eval or only at the final step -- matters once a run
plateaus or overfits late, since the final step is not always the best one.

### `device` (CPU / MPS)

The physical location -- which processor's memory -- a tensor or model's
weights actually live in. `model.to(DEVICE)` (Stage 1, `src/train.py`)
copies every parameter tensor from CPU RAM to MPS (Apple Silicon GPU)
memory; afterward, any tensor fed into the model must also be on `DEVICE`
(see `get_batch`'s `.to(device)` calls), since PyTorch ops require both
operands to live on the same device and won't silently move data for you.

### `collate_batch`

The Stage 8 SFT data-loading function (`src/sft_data.py`) that assembles a
batch of variably-sized tokenized examples into a single padded tensor, using
**padding bucket** rounding rather than a single global fixed length.

### `cosine decay`

A learning-rate schedule that decreases LR following a cosine curve from
peak to a floor value over training (Stage 1 onward), paired with **LR
warmup**: ramp up linearly first, then decay smoothly, rather than holding
one constant LR.

### `cross-entropy loss`

The standard next-token-prediction loss: for each position,
`-log(p_correct)` where `p_correct` is the softmax probability assigned to
the actual next token. A freshly initialized, correctly-set-up model should
produce roughly uniform probabilities, so initial loss should sit near
`ln(vocab_size)` (Stage 1) -- e.g. `ln(65) ~ 4.2` for the char tokenizer.
This project's `TinyLM.forward` passes `ignore_index=-1` to cross-entropy
from Stage 1 onward (unused until Stage 8's loss masking).

### `data-bound`

A diagnosis (Stage 4) where train and validation loss track closely and both
keep falling, implying the model has more capacity than the dataset can
currently exercise -- the fix is more/better data, not regularization or a
smaller model. Contrasted with **overfitting** (Stage 1, Stage 3), where
train loss keeps falling while val loss rises or plateaus, implying too
little data for the model's capacity.

### `decode step`

A single generation step producing logits for exactly one new token,
conditioned on an already-cached longer context (Stage 7). Contrasted with
**prefill**. The decode step's query length is 1, but it attends to all
prior cached keys -- this is the case requiring `is_causal=False`. See
**decode-step trap**.

### `decode-step trap`

The specific bug risk flagged in `LEARNING_PLAN.md` and confirmed in Stage 7:
naively reusing the training-time `is_causal=True` masking logic during a
cached **decode step** would mask the wrong positions, since the single new
query token has no "future" among the existing cached keys to mask against.
Fixed in this project with `is_causal = (k.size(2) == t)` -- true only when
key length equals query length (no cache, or prefill), false when a single
new token (`t=1`) queries a longer cached `k`/`v`.

### `dim` (`d_model`)

The model's hidden/embedding dimension -- the width of the vector
representing each token at every layer, e.g. `(batch, seq, dim)`. Scaled
128 -> 384 in Stage 5. The single highest-leverage scaling knob: touches Q/K/V
projections, the SwiGLU `intermediate_size`, RMSNorm's learned scale, and the
**embedding table** all at once.

### `download_data.py` (range-GET download)

The Stage 4 data script that HTTP range-GETs the first N bytes of a large HF
Hub file (`TinyStories-train.txt`, 1.9GB full size) instead of downloading it
in full, then trims to the last complete `<|endoftext|>`-delimited story so
training never sees a truncated one. The byte budget is a CLI arg
(`sys.argv[1]`), reused at 20MB in Stage 4 and bumped to 150MB in Stage 5
without code changes.

### `dynamic-padding batching` / `padding bucket`

Padding each training batch only to the nearest multiple of a fixed chunk
size (`PAD_BUCKET=64` in this project), rather than to the batch's exact max
length or one global fixed size (Stage 8). Exact-length padding produced a
distinct tensor shape per batch; MPS compiles and caches a graph per distinct
shape, so hundreds of unique shapes over a run caused a 10-20x step-time
slowdown and memory pressure toward OOM. Bucketing to multiples of 64 over a
`[64, 512]` range caps shape diversity to ~8 buckets, letting MPS reuse
compiled graphs while still avoiding the waste of always padding to 512.

### `drafter` / `verifier` (speculative decoding)

Roles in **speculative decoding**: the small, fast **drafter** model proposes
several tokens ahead greedily/cheaply, then the larger, slower **verifier**
model checks all of them in a single batched forward pass and accepts the
longest correct prefix, falling back to its own token wherever the drafter
was wrong. This project's Stage 9 teacher/student pair (same vocab, same
tokenizer, different size) is exactly the drafter/verifier shape speculative
decoding needs -- a natural fit for a Stage 10 experiment, since the
infrastructure (frozen teacher in **eval mode**, shared vocab) already
exists.

### `embedding table`

A learned lookup matrix of shape `(vocab_size, dim)` mapping each token ID to
a dense vector. `vocab_size` rows, `dim`-wide rows -- the reason `vocab_size`
changes (Stage 3: 4096, Stage 4: 8000) resize a tensor touching nearly every
other part of the model. See **tied embeddings** for why this table's weight
is shared with **lm_head**.

### `<|endoftext|>`

The special token marking a clean story/document boundary in TinyStories and
TinyStories-Instruct data (Stage 4, Stage 8), and the id used for both BOS and
EOS in this project (id 0). Plain BPE initially split it into 9 byte-level
sub-tokens (Stage 4 bug); fixed with
`tokenizer.add_special_tokens(["<|endoftext|>"])`, which registers it in the
tokenizer's `added_vocabulary` so exact-match substrings are intercepted
*before* the byte-level pre-tokenizer runs, regardless of BPE merge outcome.
Special tokens are also auto-excluded from `decode()` by default.

### `escape_whitespaces`

A SentencePiece tokenizer setting, defaulting to `true` and not exposed as a
GGUF override, that makes llama.cpp always replace literal spaces with the
`▁` glyph before vocab lookup (Stage 2). The char tokenizer's hand-built
SentencePiece `ModelProto` stored a literal `" "` piece for the space token,
crashing (`unordered_map::at: key not found`) on any prompt containing a
space; fixed by storing `"▁"` as the piece text instead. Made moot once
Stage 6 switched to native byte-level BPE export.

### `eval mode`

A PyTorch model state (`model.eval()`) disabling training-only behaviors. Used
throughout this project's verification checks (Stage 2's bit-exactness
comparison, Stage 6's re-verification) to ensure both models being compared
behave deterministically, without dropout or other stochastic training-time
ops in play.

### `export_gguf.py`

The script that runs the full HF-bridge -> `convert_hf_to_gguf.py` ->
(in Stage 2-6) metadata-patch pipeline as one step. Stage 2's version was
specific to the char-tokenizer SentencePiece hack; Stage 6 dropped those
workarounds and added the **pre-tokenizer hash mismatch** monkeypatch
instead, switching from subprocess-invoking the converter to importing and
calling it in-process (required so the monkeypatch applies before conversion
runs). Stage 8 added an optional `chat_template` parameter.

### `export_hf.py`

The script that builds an HF `LlamaConfig`/`LlamaForCausalLM` from this
project's `ModelConfig` and loads the trained `state_dict` into it -- the
**HF bridge**. Stage 2/early stages exported a hand-built SentencePiece
tokenizer; Stage 6 replaced that with wrapping the real
`checkpoints/tokenizer.json` in a `PreTrainedTokenizerFast`. Stage 8 added an
optional `chat_template` parameter threaded through to `tokenizer_config.json`.

### `float non-associativity`

The fact that floating-point addition/multiplication isn't perfectly
associative -- `(a+b)+c` can differ in the last bit from `a+(b+c)` depending
on operation order (Stage 2). Explains the one red herring of the project:
greedy autoregressive decoding diverged between PyTorch's SDPA kernel and
ggml's attention kernel after the first generated token, confirmed identical
on both CPU and Metal (ruling out GPU nondeterminism) -- ordinary kernel
implementation differences, not a bug. Why **teacher forcing** is the right
tool for "is the math faithful," not comparing final generated tokens.

### `GQA` / `MQA` / `MHA` (attention head variants)

Three ways to share **KV cache** keys/values across query heads. **MHA**
(multi-head attention, what this project currently implements) gives every
query head its own K/V head -- `n_heads` of each. **MQA** (multi-query
attention) takes the opposite extreme: all query heads share a single K/V
head, minimizing cache size but costing the most representational capacity.
**GQA** (grouped-query attention) sits between them: query heads are split
into groups, and each group shares one K/V head -- e.g. 4 query heads in 2
groups of 2 share 2 K/V heads instead of 4. The motivation is **KV cache**
memory/bandwidth: cache size scales with the *number of K/V heads*, not query
heads, so GQA cuts cache size roughly in proportion to the group size with
much less quality loss than full MQA. This is what real small Llama-family
models (Llama-3, Mistral) use instead of plain MHA, making it a natural
architecture-parity improvement for this project's `Attention` module.

### `gradient`

A vector telling each parameter "if you increased slightly, the loss would
change by this much, in this direction" -- the calculus derivative of the
loss with respect to that parameter. Computed by **`backward()`** /
backpropagation and consumed by the optimizer's `.step()` to decide how to
adjust each weight. Gradients must be cleared (**`zero_grad`**) before each
new batch, since PyTorch accumulates (adds into) `.grad` by default rather
than overwriting it.

### `GGUF`

The binary file format llama.cpp (and therefore Ollama, LM Studio) loads
models from. Bundles weights, tokenizer vocabulary, and metadata (including
**chat template**) into one file. This project never hand-writes a GGUF --
it bridges into HF `LlamaForCausalLM` and reuses llama.cpp's
`convert_hf_to_gguf.py`, the entire reason the architecture mirrors Llama.

### `gguf_new_metadata.py:copy_with_new_metadata`

llama.cpp's own helper for rewriting GGUF metadata fields post-conversion
(Stage 2), reused (rather than hand-rolling a GGUF rewriter) to patch
`tokenizer.ggml.add_space_prefix = false` onto the char-tokenizer's exported
GGUF.

### `gradient clipping`

Capping the norm of the gradient vector before the optimizer step so no
single batch's update is disproportionately large (Stage 1 onward). A
standard safety net against loss spikes, included in the training loop from
the start.

### `HF` (Hugging Face)

The company/ecosystem behind the `transformers` and `tokenizers` Python
libraries used throughout this project -- not a project-specific term, but
used as shorthand everywhere from Stage 2 onward. Appears in two distinct
roles: training a tokenizer with HF `tokenizers` (`BpeTrainer`,
`pre_tokenizers.ByteLevel`), and bridging this project's trained weights into
HF's `LlamaForCausalLM` class so llama.cpp's existing converter can be reused
instead of hand-writing a GGUF exporter. See **HF bridge**.

### `HF bridge` / `state_dict remap`

The conversion step building an HF `LlamaConfig`/`LlamaForCausalLM` from this
project's `ModelConfig`, then loading the trained `state_dict` into it. In
this project the load is `strict=True` with **zero key remapping needed**
(Stage 2) -- a deliberate Stage-0 decision: `TinyLM`/`LlamaCore` modules were
named to match HF's `model.embed_tokens.*`, `model.layers.N.self_attn.*`,
`model.layers.N.mlp.*`, `model.norm.*`, `lm_head.*` exactly from the start, so
no remap function was ever needed (`LEARNING_PLAN.md` originally anticipated
writing one).

### `ignore_index`

A sentinel label value (`-1` in this project, vs. the more common `-100`
elsewhere) telling `cross_entropy` to skip a position entirely when computing
loss and gradients. Wired into `TinyLM.forward` since Stage 1 but unused
until Stage 8, where it becomes the mechanism for **loss masking** prompt
tokens during SFT.

### `intermediate_size`

The hidden width inside the SwiGLU MLP block, larger than `dim` (512 at
Stage 1, scaled to 1536 in Stage 5, keeping the same 4x-`dim` ratio
throughout). Scaling it gives the MLP more per-layer capacity independent of
attention width.

### `is_causal`

A boolean flag (in `torch.nn.functional.scaled_dot_product_attention`)
controlling whether causal masking is applied automatically. Must be `True`
during training and **prefill**, `False` during a cached **decode step**.
Implemented in this project's `Attention.forward` as
`is_causal = (k.size(2) == t)`. Getting this backwards doesn't crash -- it
silently corrupts generation quality, which is exactly the **decode-step
trap**.

### `KV cache` / `KVCache`

Stored key/value tensors from previous attention computations, reused at each
new **decode step** instead of recomputed (Stage 7). This project's
`KVCache` class is a plain list of per-layer `(k, v)` tensors with shape
`[batch, n_heads, seq, head_dim]`, growing along `seq` by one position per
generated token (`update(layer_idx, k, v)` concatenates; `seq_len(layer_idx)`
reports that layer's cached length). Changes generation cost from O(n^2)
total to O(n) per step. By explicit design constraint, the cache never
touches the export path -- the non-cached `forward` stays byte-for-byte
identical so the bit-exactness check stays valid without rerunning it.

### `KV cache layer-index bug`

A real bug caught in Stage 7: the first `KVCache.seq_len()` implementation
always read layer 0's cache regardless of which layer asked, so every layer
past the first saw the *post-update* length instead of its own pre-update
length, silently shifting RoPE positions for layers 1+. Didn't crash and
didn't even fail the greedy-argmax verification test, because RoPE's
relative-rotation property cancels a *uniform* shift within one forward call
(see `RoPE`) -- only desynchronized once prefill and a later decode step
applied *different* shifts. Caught only by a per-position raw-logit
comparison (`torch.allclose`), not by comparing final generated tokens.
Fixed by parameterizing `seq_len(layer_idx)` to read each layer's own cache
slot.

### `LM Studio` / `lms`

A local model-runtime app used to verify GGUF exports end-to-end (Stage 2,
Stage 6) via its local OpenAI-compatible completions API. `lms import` (not
a plain filesystem copy into `~/.lmstudio/models/`) is required for the model
to be indexed and loadable -- a plain copy left `lms load` hanging on an
unresolved model key (Stage 6).

### `lm_head`

The final linear projection from hidden state `(batch, seq, dim)` to logits
`(batch, seq, vocab_size)`. See **tied embeddings** for why this layer's
weight is shared with the **embedding table** in this project rather than
learned separately.

### `llama-cli` / `llama-completion`

llama.cpp CLI binaries used for real-runtime verification. `llama-completion`
is the raw-completion binary (Stage 2, Stage 6) used with
`--top-k 1 --ignore-eos` for teacher-forced/greedy checks, separate from
`llama-cli`, which in this project's llama.cpp build assumes
chat-template/conversation mode and refuses `--no-conversation` (Stage 2).
Stage 8 confirmed `llama-cli` auto-detects an embedded **chat template** and
enters conversation mode without needing a `--jinja` flag.

### `llama-quantize`

The llama.cpp tool used to produce quantized GGUF variants (Stage 6):
`q8_0` and `q4_0` from an f16 source, re-tested with `llama-completion` for
generation quality after quantizing.

### `loss` (general)

A single number summarizing how wrong the model's predictions were on a
batch -- lower is better. In this project specifically it's always
**cross-entropy loss**: roughly "how surprised was the model by the actual
next token." The number being minimized by the entire training loop;
everything else (gradients, the optimizer, the LR schedule) exists purely
to push this number down.

### `loss masking`

Excluding certain positions from contributing to the training loss, via
**ignore_index** (Stage 8). In this project's SFT, the prompt portion's
targets are set to `-1` so the model isn't penalized for "predicting" text
it was just given as context -- except for the prompt's *last* token's
target, which is the *first response* token and must stay unmasked, since
that's the first real prediction the model is supposed to make. `mask_len =
len(prompt_ids) - 1` is the off-by-one this boundary requires getting right.

### `Modelfile`

An Ollama configuration file (`checkpoints/Modelfile`, Stage 6) specifying
`FROM ./stage5-q8_0.gguf` plus sampling parameters, so `ollama create` /
`ollama run` becomes a one-command check once Ollama is installed. Written
but not verified in Stage 6 (Ollama wasn't installed on the dev machine);
LM Studio's independent-runtime verification was judged sufficient for that
stage.

### `n_heads`

The number of parallel attention heads per layer (4 at Stage 1, scaled to 6
in Stage 5). Must evenly divide `dim` (`head_dim = dim / n_heads`), and the
resulting `head_dim` must stay even for RoPE's pairwise rotation.

### `n_layers`

The number of stacked transformer blocks (4 at Stage 1, scaled to 8 in Stage
5). Scaled together with `dim`/`n_heads` rather than independently, to avoid
reintroducing failure modes (overfitting, data-bound performance) already
diagnosed at smaller scale.

### `no_grad` / `@torch.no_grad()`

A context manager/decorator telling PyTorch not to build a computation
graph or track gradients for the code inside it (Stage 1's `estimate_loss`,
`src/train.py`). Used for any forward pass that's purely for measurement,
not training -- skipping graph-tracking makes it faster and uses less
memory, since there's no intention of ever calling `.backward()` on that
loss.

### `num_params`

A simple count of every learnable number (every entry across every weight
matrix and bias) in the model, printed once at the start of training
(`model.num_params()` in `src/train.py:main`) as a sanity check on model
size and a rough proxy for capacity/memory footprint -- not a metric that
changes during training.

### `online distillation`

Running the teacher model live in the same training loop as the student
(frozen, in `eval mode`), rather than precomputing and caching teacher logits
ahead of time (Stage 9, planned). Chosen as the right tradeoff at this
project's ~8k vocab and data size -- precomputing/caching teacher logits
isn't worth the storage/complexity at this scale.

### `optimizer` / `param_groups`

The object (`torch.optim.AdamW` in this project) that owns the rule for
turning a gradient into a weight update. `optimizer.step()` applies that
rule to every parameter using its current `.grad`. `param_groups` is the
optimizer's internal list of parameter groups, each carrying its own
hyperparameters (notably `lr`); this project has no learning-rate
**scheduler** object, so `src/train.py:main`'s loop manually overwrites
`g["lr"]` for every group, every step, via `lr_at(step)` -- a hand-rolled
schedule rather than a `torch.optim.lr_scheduler` instance.

### `overfitting`

A diagnosis where training loss keeps falling while validation loss rises or
plateaus -- the model is memorizing rather than generalizing. Observed in
Stage 1 (1M-char corpus, 1M-param model -- expected, not a bug) and Stage 3
(BPE on the same small Shakespeare corpus made it pack ~4-5x more text per
token, accelerating the same overfitting). Resolved by Stage 4's larger
TinyStories corpus, which instead produced a **data-bound** diagnosis.
Contrasted explicitly throughout the notes with data-bound.

### `parse_examples`

The Stage 8 SFT data function (`src/sft_data.py`) that splits
TinyStories-Instruct's `<|endoftext|>`-delimited blocks and locates the
prompt/response boundary with a regex anchored to start-of-line,
`(?:^|\n)Story:\s*`, rather than a plain substring search for `"Story:"`.
Necessary because the free-text `Summary:` field can itself contain the word
"story" (e.g. "a story about a curious fox"), which a substring search would
match as a false boundary inside the prompt and corrupt the prompt/response
split. Covered by a regression test,
`scripts/test_sft_data.py:test_parse_examples_splits_headers_from_story`.

### `perplexity`

A standard language-model evaluation metric, `exp(cross-entropy loss)` --
roughly "the model's effective branching factor," i.e. how many tokens it
was, on average, uncertain between at each step. Lower is better, same
direction as **loss**, but more interpretable as a number (perplexity 8 means
"about as confused as choosing uniformly among 8 options"); this project
currently reports raw loss only, not perplexity, in training/eval output. A
natural Stage-10-adjacent addition: track held-out perplexity across stages
as a single comparable number, and re-measure it after **quantization**
(comparing f16 vs. `q8_0`/`q4_0` perplexity) to see where precision loss
actually costs prediction quality rather than just eyeballing sample text.
Related but distinct from a **cloze-style eval** (e.g. "does the model assign
higher probability to the plausible next word than an implausible one"),
which tests a specific behavior rather than averaging over everything.

### `pre_tokenizers.ByteLevel`

The HF `tokenizers` component governing how raw text is split into initial
chunks before BPE merge rules run (Stage 3). Distinct from `BpeTrainer`,
which learns the merges themselves. Used with `use_regex=True`, which Stage
6's notes confirm is byte-for-byte the same regex split GPT-2 uses -- the
basis for the **pre-tokenizer hash mismatch** workaround's correctness.

### `pre-tokenizer hash mismatch`

llama.cpp's `convert_hf_to_gguf.py:get_vocab_base_pre()` identifies a
tokenizer's pre-tokenizer by hashing the *token ids* of a fixed probe string
and matching against a table of hashes from known public tokenizers (Stage
6). A custom-trained BPE vocabulary's hash can never match any table entry,
even when its pre-tokenizer *algorithm* (`ByteLevel`, `use_regex=True`) is
byte-for-byte identical to GPT-2's -- the hash check conflates "exact trained
vocab" with "pre-tokenizer regex," when only the regex needs to match. Fixed
in `export_gguf.py:patch_unrecognized_pretokenizer` by monkeypatching
`TextModel.get_vocab_base_pre` to catch the resulting `NotImplementedError`
and fall back to `"gpt-2"` -- a deliberate choice based on known
pre-tokenizer config, not a guess to silence an error. Verified correct (not
just non-crashing) via the Stage 6 **tokenizer parity** check.

### `.pt` / pickle checkpoint

The file format produced by `torch.save()`, built on Python's `pickle`
serialization -- can store arbitrary Python objects, not just tensors, which
is why this project's training checkpoints (`stage5.pt`, `stage8.pt`) bundle
`state_dict` + optimizer state + step count + config together in one file.
The cost of that flexibility: pickle can execute arbitrary code while
reconstructing objects on load, a real risk for untrusted files (the reason
`torch.load`'s `weights_only` flag exists). Contrasted with
**`safetensors`**, the tensor-only format the HF bridge exports separately
rather than reusing `.pt` for.

### `prefill`

The first forward pass of generation, processing the entire initial prompt in
one shot to populate the **KV cache**, before any **decode steps** run
(Stage 7). `src/sample.py`'s `generate_cached` prefills once, then feeds one
new token at a time.

### `PreTrainedTokenizerFast`

The HF wrapper class letting a trained byte-level BPE tokenizer
(`tokenizer.json`) be consumed directly by llama.cpp's GPT-2 vocab export
path (Stage 6), replacing the earlier hand-built SentencePiece workaround
entirely. `save_pretrained()`s alongside the bridged `LlamaForCausalLM`,
producing a real `tokenizer.json` + `tokenizer_config.json`.

### `quantization`

Reducing stored-weight numerical precision (e.g. f16 down to 8-bit or 4-bit
block-quantized integers) to shrink file size and speed inference, at some
quality cost. This project's Stage 6 export: f16 (43.9MB) -> `q8_0` (23.6MB)
-> `q4_0` (14.2MB), the latter with one tensor falling back to higher
precision (normal for very small dims, since the assumed block size doesn't
divide evenly).

### `RMSNorm`

Root Mean Square Normalization: rescales a vector by its root-mean-square
value, then applies a learned per-dimension scale -- no mean-centering, no
bias term (Stage 1). Llama's normalization choice instead of LayerNorm,
mirrored here specifically so the export bridge into `LlamaForCausalLM`
lines up weight-for-weight.

### `RoPE (Rotary Position Embeddings)`

Injects positional information by rotating pairs of dimensions within each
head's Q and K vectors by an angle depending on token position, rather than
adding a separate positional embedding (Stage 1). Requires `head_dim` to be
even (rotates dimension-pairs together). Crucial relative-rotation property
exploited (and the source of a subtle bug) in Stage 7: a *uniform* shift to
both q and k cancels in the attention dot product, so the **KV cache
layer-index bug**'s wrong-but-uniform shift during prefill didn't change
prefill's output -- only desynchronized once decode applied a *different*
shift relative to cached prefill keys. Under caching, position must be
tracked as the token's *absolute* index in the full sequence
(`pos_offset = kv_cache.seq_len(layer_idx)`), not re-based to 0 per window
like the no-cache sliding-window path does.

### `RoPE scaling` (linear scaling / position interpolation vs. YaRN)

Techniques for extending a RoPE-based model's usable context length beyond
the `max_seq_len` it was trained on, by altering the rotation frequencies fed
into **RoPE** rather than retraining from scratch at the longer length.
**Linear scaling / position interpolation** compresses position indices
(e.g. divide every position by 2 before computing rotation angles) so a
sequence twice as long as training still maps into the same range of angles
the model saw during training -- cheap, but blurs relative-position
resolution since two adjacent real tokens now look closer together to RoPE
than they did in training. Plain **extrapolation** (just feeding RoPE
positions past `max_seq_len` unmodified) degrades sharply because the model
never saw those rotation angles during training and the attention pattern
falls apart. **YaRN** is a more careful version of interpolation: it scales
low and high rotation frequencies differently (low frequencies, which encode
coarse/long-range position, get compressed; high frequencies, which encode
fine local position, are left closer to their trained values), preserving
more of the model's local-position sharpness while still extending range.
This project trains with `max_seq_len=256` (Stage 1 onward) and has not yet
tried any scaling variant -- a good isolated lesson because it changes only
how rotation angles are computed, not the **RoPE** mechanism itself or any
trained weight.

### `sample <pid>` (process sampling)

A macOS diagnostic command used in Stage 5 to confirm a training process
that appeared stalled (empty log file) was actually mid-optimizer-step and
actively running MPS graphs -- the actual cause was **buffered stdout**, not
a hang.

### `safetensors`

A flat, tensor-only file format: a JSON header (tensor names, shapes,
dtypes) followed by raw tensor bytes, with no embedded Python objects and no
code execution on load. Produced by `scripts/export_hf.py`'s **HF bridge**
step as `model.safetensors` (e.g. `checkpoints/stage5-hf/`,
`checkpoints/stage8-hf/`) rather than reusing the project's own `.pt`
checkpoints, since `safetensors` can't hold optimizer state or arbitrary
config -- only weights. Supports zero-copy/mmap loading, and is the format
llama.cpp's `convert_hf_to_gguf.py` expects as input on the way to a
**GGUF**. See **`.pt` / pickle checkpoint** for the contrasting format.

### `SentencePiece`

A subword tokenizer format/library llama.cpp's GGUF converter natively
understands via `_set_vocab_sentencepiece` (Stage 2). This project's char
tokenizer was forced through this path by hand-building a `ModelProto`,
surfacing `escape_whitespaces` and `add_space_prefix` defaults that had to be
patched to match training-time behavior. Entirely replaced by native
byte-level BPE export (`_set_vocab_gpt2` path) starting Stage 6.

### `SFT` (Supervised Fine-Tuning)

Continuing to train an already-pretrained model on labeled (prompt,
response) pairs rather than raw unstructured text (Stage 8). Same
architecture and tokenizer as base training -- only the data shape and
**loss masking** (prompt tokens excluded via **ignore_index**) change. This
project's SFT stage fine-tunes from `stage5.pt` on TinyStories-Instruct.

### `step` vs `epoch`

A **step** is one batch's worth of forward pass + backward pass + weight
update. An **epoch** is one full pass over the entire training dataset.
This project's training loop (`src/train.py:main`) is purely step-counted
(`for step in range(1, MAX_STEPS + 1)`) and never tracks epochs at all --
`get_batch` draws a fresh random slice of `train_data` every step via
`torch.randint`, so "one epoch" isn't a meaningful boundary the code ever
crosses or checks; the dataset is just a pool to sample chunks from.

### `speculative decoding`

A generation-speedup technique using a **drafter**/**verifier** model pair:
the small drafter generates several candidate tokens cheaply and
autoregressively, then the large verifier checks them all in one batched
forward pass (cheaper per-token than generating them one at a time itself)
and accepts the longest prefix that matches what it would have generated,
discarding the rest. Produces output statistically identical to running the
verifier alone -- it's a latency optimization, not a quality/quantity
tradeoff like **quantization** or smaller models are. This project's Stage 9
teacher/student pair is the exact same-vocab **drafter**/**verifier** setup
this technique needs, making it a natural Stage 10 candidate built on
already-existing infrastructure rather than a new architecture.

### `state_dict`

A PyTorch dictionary mapping module names to parameter tensors -- the
standard mechanism for saving/loading/transferring weights between two model
implementations sharing the same parameter shapes and naming. This project's
HF bridge load is `strict=True` with **zero key remapping** because module
names were chosen from the start to match HF's naming convention exactly
(Stage 2).

### `SwiGLU`

A gated MLP variant used in Llama's feed-forward block (Stage 1): instead of
two linear projections around an activation, SwiGLU computes
`silu(x @ W_gate) * (x @ W_up)` then projects that elementwise product down
with a third matrix (`W_down`) -- three projections total. The gating term
lets the network modulate how much of each feature passes through.

### `teacher forcing` / `teacher-forced check`

Feeding the model ground-truth previous tokens as input at every position,
rather than its own predictions, when comparing two implementations. Used as
the Stage 2 keystone faithfulness check (bit-exact, see **bit-exactness**)
and as the Stage 2 "real GGUF runtime" check (200 positions of real
Shakespeare text, no autoregressive feedback, 100% top-1 agreement). The
tool that isolates "is the math faithful" from "do two independent
floating-point softmax implementations round identically" (they never will,
and don't need to -- see **float non-associativity**).

### `temperature`

A scalar dividing logits before softmax during sampling (Stage 1). Lower
temperature sharpens the distribution toward high-probability tokens; higher
flattens it. Used alongside **top-k sampling** in `sample.py`.

### `tied embeddings`

Sharing one weight matrix between the input **embedding table** and the
output **lm_head** projection. Used from Stage 1 onward, but interacted
badly with PyTorch's default `nn.Embedding` init: `N(0,1)` default init,
combined with tying, produced a logit standard deviation of ~11 (dim=128 dot
product of two std=1 vectors) instead of a sane small value, blowing up
initial loss to ~94 instead of the expected `ln(65)~4.2`. Fixed with explicit
small-variance init, `N(0, 0.02)`, on all Linear/Embedding weights in
`TinyLM._init_weights` -- standard GPT-2/Llama practice. The bug was never
tying embeddings itself, it was relying on PyTorch's default init instead of
setting one deliberately.

### `tokens/word` (vocab efficiency)

A simple measure of how much text one token covers on average, computed by
encoding a corpus and dividing token count by word count -- lower means the
tokenizer packs more text per token, which directly cuts sequence length
(and therefore attention's quadratic cost, and training/inference time) for
the same text. This project's `vocab_size` has been bumped before (4096 ->
8000, Stage 3 -> Stage 4) but tokens/word was never measured directly as the
justification; checking it on TinyStories-Instruct against a few candidate
vocab sizes would turn "bigger vocab seems better" into an actual measured
tradeoff (larger vocab shortens sequences but enlarges the **embedding
table** and **lm_head**, so there's a real crossover point, not a free
win).

### `top-p sampling` (nucleus sampling) / `min-p sampling` / `repetition penalty`

Sampling strategies that modify the next-token probability distribution
before drawing a sample, alternatives/complements to this project's existing
**top-k sampling** and **temperature** in `sample.py`. **Top-p (nucleus)
sampling** keeps the smallest set of highest-probability tokens whose
cumulative probability exceeds `p` (e.g. 0.9), then samples only from that
set -- adaptive in size, unlike top-k's fixed count, so it stays narrow when
the model is confident and widens when it's uncertain. **Min-p sampling**
instead keeps any token whose probability is at least `min_p` fraction of the
top token's probability (e.g. keep anything >= 10% as likely as the best
option) -- simpler to reason about and less sensitive to a long flat tail
than top-p. **Repetition penalty** down-weights (or bans) tokens that already
appeared recently in the generated sequence, directly targeting the
degenerate looping TinyStories-scale models are notorious for, which neither
top-k/top-p/min-p fix on their own since looping can stay highly *probable*
token-by-token even though it's a bad sequence overall.

### `token id`

The integer a tokenizer assigns to one vocabulary entry (byte, character,
or BPE subword piece, depending on stage). `BPETokenizer.encode()` turns a
raw string into a list of these integers; `train_data`/`val_data` in
`src/train.py:main` are 1D `torch.long` tensors of token ids -- the entire
training corpus reduced to one long sequence of integers, with `dtype=long`
required because `nn.Embedding`'s lookup (and `cross_entropy`'s target
argument) expect integer indices, not floats.

### `tokenizer parity check`

A Stage 6 verification step (new for that stage, not covered by the
**bit-exactness** check, since that check starts from already-identical
token ids): comparing `BPETokenizer.encode()` against
`llama_cpp.Llama(..., vocab_only=True).tokenize()` on strings probing
leading/trailing spaces, digit runs, contractions, whitespace runs, and
punctuation. All matched exactly, confirming the **pre-tokenizer hash
mismatch** `"gpt-2"` fallback was the *correct* choice, not just one that
didn't crash.

### `train()` / `eval()` mode toggle

`model.train()` and `model.eval()` switch the same module between two
behavior modes (dropout active vs. disabled, batchnorm-style stats updating
vs. frozen -- this project has no dropout/batchnorm currently, but the
convention is followed anyway). `estimate_loss` (Stage 1, `src/train.py`)
calls `model.eval()` before measuring, then `model.train()` again before
returning, so evaluation never leaves the model in the wrong mode for the
next training step.

### `top-k sampling`

A sampling strategy restricting next-token choice to the `k`
highest-probability tokens, renormalizing before sampling (Stage 1). Used
alongside **temperature** in `sample.py` to control diversity vs. coherence.

### `train_tokenizer.py`

The script (Stage 3) that trains a byte-level BPE tokenizer on a target
corpus and saves it to `checkpoints/tokenizer.json`. Re-pointed at a new
corpus and vocab size for Stage 4 (Shakespeare -> TinyStories, 4096 -> 8000)
without code changes -- reusable as-is across datasets.

### `vocab_size`

The total number of distinct tokens a tokenizer can produce / a model can
predict over. 65 (char-level) at Stage 1, 4096 (BPE) at Stage 3, 8000 (BPE)
from Stage 4 onward. Determines the row count of the **embedding table** and
output width of **lm_head**, and sets the random-baseline loss
`ln(vocab_size)` used as an initialization sanity check.

### `warmup steps` / `LR warmup`

Starting training with a small learning rate and increasing it linearly to
its target peak over the first N steps, before **cosine decay** begins.
`WARMUP_STEPS` scaled across stages: 100 (Stage 1) -> 150 (Stage 4) -> 300
(Stage 5), tracking the corresponding increase in `MAX_STEPS` and dataset
size.

### `zero_grad` (`set_to_none=True`)

Clearing every parameter's accumulated `.grad` before computing a new
batch's gradients (`optimizer.zero_grad(set_to_none=True)` in
`src/train.py:main`). Necessary because **`backward()`** adds into existing
`.grad` tensors rather than overwriting them -- without clearing first, each
step's gradient would be contaminated by every previous step's gradient.
`set_to_none=True` frees the old `.grad` tensors outright instead of
zeroing them in place, a minor memory/speed optimization over the
`zero_grad()` default.
