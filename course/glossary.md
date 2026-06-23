# Glossary

A from-scratch reference for every term introduced across the 12 course
chapters (`course/chapters.md`). Alphabetical. Each entry is self-contained --
cross-references point to other terms by name rather than assuming reading
order.

---

### `add_bos`

A tokenizer/runtime setting controlling whether a beginning-of-sequence token
is automatically prepended to input text. (Ch. 8) Must match whatever the
model saw at training time -- if training always prepended a BOS token but
the export's `add_bos` default disagrees, every prompt the runtime tokenizes
is silently shifted out of distribution from what the model learned.

### `add_space_prefix`

A SentencePiece tokenizer option that defaults to `true` and inserts a
leading space before the first token of input text. (Ch. 5) One of two
llama.cpp internals that broke a hand-rolled char-vocab export on first try --
it's not a bug, it's an assumption baked into the SentencePiece path that any
vocab routed through it must satisfy.

### `AdamW`

The optimizer used for training (Ch. 4). A variant of Adam (adaptive
per-parameter learning rates based on gradient history) that decouples weight
decay from the gradient update, which makes the decay behave as a cleaner
regularizer. Default choice for transformer training because it tolerates
noisy gradients and sparse-ish updates (e.g. embedding rows) better than plain
SGD.

### `attention score matrix`

The intermediate `(seq, seq)` matrix (per head, per batch element) produced by
`Q @ K^T / sqrt(head_dim)` inside self-attention (Ch. 3). Entry `[i, j]` is
how much query position `i` "looks at" key position `j` before softmax turns
each row into a probability distribution. Causal masking zeroes out (sets to
`-inf` before softmax) every entry where `j > i`.

### `autocast`

A PyTorch context manager (`torch.autocast`) that runs ops in a lower
precision (here, **bf16**, never fp16 on this project's MPS setup) while
keeping master weights in fp32, trading some numerical precision for speed
and memory. Relevant from Ch. 4 onward wherever training speed matters.

### `BOS / EOS`

Beginning-of-sequence / end-of-sequence tokens -- special vocabulary entries
that mark the start or end of a text span rather than representing real
content. See `<|endoftext|>` and `add_bos`.

### `BPE (Byte Pair Encoding)`

A subword tokenization algorithm that starts from individual characters (or
bytes) and iteratively merges the most frequent adjacent pair into a new
token, building a vocabulary bottom-up. (Ch. 2) Lets a fixed-size vocabulary
represent arbitrary text by falling back to smaller pieces for rare words
instead of needing an `<unk>` token for everything it's never seen.

### `BpeTrainer`

The HF `tokenizers` library class that learns BPE merge rules from a text
corpus (Ch. 2) -- the thing that actually produces a `tokenizer.json`
vocabulary, as opposed to `pre_tokenizers.ByteLevel`, which only governs how
raw text is pre-split before BPE merging runs.

### `byte-level BPE`

BPE applied to raw UTF-8 bytes (256 possible base symbols) instead of
Unicode characters (Ch. 2), the GPT-2/Llama-style approach. Guarantees there
is no out-of-vocabulary token ever, because any string decomposes into bytes
the vocabulary already covers, even byte sequences never seen during
training.

### `causal masking`

The constraint, enforced inside attention, that token at position `i` can
only attend to positions `<= i`, never `> i` (Ch. 3). Implemented by adding
`-inf` to the disallowed entries of the attention score matrix before
softmax, which forces their attention weight to 0. Necessary because at
generation time the model genuinely cannot see future tokens -- training must
match that constraint or the model learns to "cheat."

### `chat template`

A Jinja-style string, embedded in tokenizer/model metadata, that defines how
a list of role-tagged messages (`user`, `assistant`, ...) gets serialized into
the exact text string the model is fed (Ch. 10). Part of the export contract,
not a serving convenience: if the template's output text doesn't byte-match
what training data looked like after formatting, the model receives an
input shape it has never seen and produces degraded output even though
nothing "crashed."

### `checkpoint (model)`

A saved snapshot of model weights (and usually optimizer state) at a point in
training, written to `checkpoints/` (gitignored). From Ch. 4, checkpointing
on **best validation loss** rather than "the latest step" matters once a run
plateaus or starts overfitting late -- the last checkpoint is not always the
best one.

### `cosine decay`

A learning-rate schedule shape that decreases the LR following a cosine
curve from its peak down to a floor value over training (Ch. 4). Paired with
**LR warmup**: warm up linearly first, then decay smoothly, rather than
holding one constant LR for the whole run.

### `cross-entropy loss`

The standard next-token-prediction loss: for each position, `-log(p_correct)`
where `p_correct` is the model's softmax probability assigned to the actual
next token. A freshly initialized model with no learned structure should
produce roughly uniform probabilities over the vocabulary, which is why
initial loss should sit close to `ln(vocab_size)` (Ch. 4) -- a higher value
signals something (often initialization) is actively making the model worse
than random.

### `decode step`

A single generation step where the model produces logits for exactly one new
token, conditioned on a longer context that's mostly already-cached (Ch. 11).
Contrasted with **prefill**, which processes the entire initial prompt in one
forward pass. The decode step's query is length 1, but it must attend to all
prior cached positions -- so it needs `is_causal=False` (the single query
token has nothing *after* it to mask; masking logic written for the
multi-token training-time case doesn't apply).

### `dim` (`d_model`)

The model's hidden/embedding dimension -- the width of the vector representing
each token at every layer (e.g. `(batch, seq, dim)`). The single
highest-leverage hyperparameter for scaling (Ch. 7): increasing it touches Q/K/V
projections, the MLP's `intermediate_size`, RMSNorm's learned scale, and the
embedding table all at once.

### `distillation loss`

See **knowledge distillation**. The combined loss term used to train a
student model against both a teacher's soft labels (via KL-divergence) and
the ground-truth hard labels (via cross-entropy), usually as a weighted sum
of the two (Ch. 12).

### `embedding table`

A learned lookup matrix of shape `(vocab_size, dim)` that maps each token ID
to a dense vector (Ch. 2). `vocab_size` is the row count, `dim` is the row
width -- this is why `vocab_size` is the one hyperparameter that resizes a
tensor touching nearly every other part of the model (also resizes the
**lm_head**, see **tied embeddings**).

### `<|endoftext|>`

The special token marking a clean document/example boundary in training data
(Ch. 2, Ch. 6). Must be explicitly registered as an "added token" with the
tokenizer rather than left to BPE merge rules, otherwise it risks being
shredded into ordinary byte fragments and losing its function as a boundary
signal the model can learn to recognize.

### `eval mode`

A PyTorch model state (`model.eval()`) that disables training-only behaviors
like dropout and switches batch-norm-style layers to use running statistics
instead of batch statistics. Relevant in Ch. 12: the distillation teacher
runs frozen, in eval mode, alongside the student's training-mode forward
pass.

### `float non-associativity`

The fact that floating-point addition/multiplication is not perfectly
associative -- `(a+b)+c` can differ in the last bit from `a+(b+c)` depending
on operation order (Ch. 5). Explains why greedy autoregressive decoding can
diverge between two implementations (PyTorch vs. ggml) even when both
implement *mathematically identical* operations -- different kernel
implementations sum/reduce in different orders. The fix isn't "find the bug,"
it's recognizing this as expected noise and switching to **teacher-forced**
per-position logit comparison instead of comparing final generated tokens.

### `GGUF`

The binary file format llama.cpp (and therefore Ollama, LM Studio) loads
models from (Ch. 1, Ch. 5, Ch. 8). Bundles weights, tokenizer vocabulary, and
metadata (including chat templates) into one self-contained file. This
project never hand-writes a GGUF -- it bridges into HF `LlamaForCausalLM`
and reuses llama.cpp's existing `convert_hf_to_gguf.py`, which is the entire
reason the model architecture mirrors Llama in the first place.

### `gradient clipping`

Capping the norm of the gradient vector before the optimizer step, so no
single batch's update can be disproportionately large (Ch. 4). A standard
safety net against loss spikes and instability, included in the training
loop from the start rather than added reactively after a blowup.

### `hard labels`

The ground-truth next token as a one-hot target (probability 1 on the
correct token, 0 elsewhere) -- what ordinary cross-entropy training uses. (Ch.
12) Contrasted with **soft labels** in distillation, which carry the full
probability distribution a teacher model assigns across the whole vocabulary
rather than a single "correct answer."

### `HF bridge`

The conversion step that builds an HF `LlamaConfig`/`LlamaForCausalLM` from
this project's custom `ModelConfig`, then loads the custom-trained
`state_dict` into it with `strict=True` (Ch. 5). Works with zero key
remapping because the custom model's module names were chosen from the start
to match HF's `model.layers.N.self_attn.*` naming convention -- a Stage-0
decision paying off at export time.

### `head_dim`

The width of each individual attention head: `dim / n_heads` (Ch. 3, Ch. 7).
Q/K/V tensors get reshaped from `(batch, seq, dim)` into
`(batch, n_heads, seq, head_dim)` so each head attends independently over a
smaller slice of the representation. `n_heads` must divide `dim` evenly, and
RoPE specifically requires `head_dim` to be even (it rotates pairs of
dimensions together).

### `ignore_index`

A sentinel label value (commonly `-100`) that tells `cross-entropy loss` to
skip a position entirely when computing loss and gradients (Ch. 9). The
mechanism behind **loss masking**: prompt tokens get their label set to
`ignore_index` so the model is never penalized for "predicting" text it was
just given as context, only for the response tokens it must actually
generate.

### `intermediate_size`

The hidden width inside the SwiGLU MLP block, typically larger than `dim`
(Ch. 3, Ch. 7) -- e.g. Llama commonly uses roughly `8/3 * dim` rounded to a
convenient multiple. Scaling it up gives the MLP more capacity per layer
independent of attention width.

### `is_causal`

A boolean flag (e.g. in `torch.nn.functional.scaled_dot_product_attention`)
controlling whether causal masking is applied automatically. Must be `True`
during training/prefill (a token must not see the future) and `False` during
a cached **decode step**, where there is only one query token attending
backward over a longer cached context and no "future" relative to it to mask.
Getting this backwards doesn't crash -- it silently corrupts generation
quality, which is why Ch. 11 calls it a trap.

### `KL-divergence loss`

Kullback-Leibler divergence: a measure of how different two probability
distributions are. Used in distillation (Ch. 12) to penalize the student
when its output distribution over the vocabulary diverges from the teacher's
(temperature-softened) output distribution -- as opposed to cross-entropy
against a single hard label.

### `knowledge distillation`

Training a smaller "student" model to match a larger "teacher" model's full
output distribution (soft labels), not just the single ground-truth next
token (Ch. 12). The student learns *how confident and how the teacher
spreads probability mass* across the vocabulary, which carries more signal
per training example than a hard label alone. Requires a shared vocabulary
between teacher and student, which is why this project's teacher has to be a
custom model trained on the project's own tokenizer rather than any
off-the-shelf LLM.

### `KV cache`

Stored key/value tensors from previous attention computations, reused at
each new decode step instead of being recomputed (Ch. 11). Cache tensor
shape: `[batch, n_heads, seq, head_dim]`, growing by one position along
`seq` per generated token. Changes generation cost from O(n^2) total
(recomputing the whole context every step) to O(n) per step. Strictly a
training/sampling-time optimization -- it must never alter the non-cached
forward path that the export bit-exactness check depends on, since that path
is the ground truth the cached path is verified against.

### `LayerNorm`

A normalization layer that re-centers a vector (subtract the mean) and
rescales it (divide by standard deviation), then applies a learned
scale-and-shift (Ch. 3). Contrasted with **RMSNorm**, which Llama (and
therefore this project) uses instead.

### `learning rate (LR) warmup`

Starting training with a small LR and increasing it linearly to its target
peak over the first N steps, before any decay schedule begins (Ch. 4).
Prevents large, destabilizing updates while the model's weights are still
close to (noisy) initialization values.

### `lm_head`

The final linear projection from hidden state `(batch, seq, dim)` to logits
`(batch, seq, vocab_size)` -- the layer that turns "what the model is
thinking" back into a probability distribution over the vocabulary. See
**tied embeddings** for why this layer's weight matrix is often shared with
the embedding table rather than learned separately.

### `loss masking`

Excluding certain positions from contributing to the training loss, via
`ignore_index` (Ch. 9). In instruction-tuning, the prompt portion is masked
out because it's context the model was given, not something it should be
penalized for "predicting" -- except for the prompt's *last* token, which
must stay unmasked because it's the one position responsible for predicting
the response's first token.

### `multi-head attention`

Splitting attention into multiple parallel "heads," each operating on a
`head_dim`-wide slice of the representation, then concatenating their
outputs back to `dim` (Ch. 3). Lets the model attend to different kinds of
relationships (e.g. syntax vs. coreference) simultaneously rather than
computing one global attention pattern per layer. Increasing `n_heads` at
fixed `dim` shrinks `head_dim` per head but doesn't change the total
parameter count of the attention block.

### `n_heads`

The number of parallel attention heads per layer (Ch. 3, Ch. 7). Must evenly
divide `dim` (since `head_dim = dim / n_heads`), and the resulting
`head_dim` must be even for RoPE to apply its pairwise rotation.

### `n_layers`

The number of stacked transformer blocks in the model (Ch. 7). Each
additional layer is another full attention + SwiGLU MLP block with its own
weights, scaled together with `dim` and `n_heads` rather than independently,
to avoid reintroducing failure modes (overfitting, data-bound performance)
already diagnosed at a smaller scale.

### `online distillation`

Running the teacher model live, in the same training loop as the student
(frozen, in `eval mode`), rather than precomputing and caching teacher
logits ahead of time (Ch. 12). The right tradeoff at this project's
vocab/data scale -- precomputing only pays off when teacher inference cost
dominates and you expect to reuse the same logits across many epochs.

### `overfitting`

When a model's training loss keeps falling while its validation loss rises
or plateaus -- the model is memorizing training-specific patterns rather than
learning generalizable structure (Ch. 4). On a tiny corpus with a small
model, this is the *expected* outcome of a healthy pipeline, not a sign of a
bug -- contrasted with the **data-bound** diagnosis in Ch. 6, where train and
val loss track closely and both keep falling, implying the fix is more/better
data rather than regularization.

### `padding bucket / dynamic-padding bucketing`

Padding each batch only up to the nearest multiple of a fixed chunk size
(e.g. 64), rather than to the batch's exact max length or a single global
fixed size (Ch. 9). Exact-length padding causes MPS to recompile its
execution graph for nearly every distinct shape it sees, producing a 10-20x
slowdown; bucketing to a small number of recurring shapes lets MPS reuse
compiled graphs while still avoiding the waste of always padding to a single
large fixed size.

### `pre_tokenizers.ByteLevel`

The HF `tokenizers` component that governs how raw text is split into
initial chunks (on whitespace, punctuation, etc.) before BPE merge rules
ever run (Ch. 2). Distinct from `BpeTrainer`, which learns the merges
themselves -- this only decides the starting split points.

### `pre-tokenizer hash mismatch`

llama.cpp's GGUF converter ships a table of known hashes for "famous" public
tokenizer pre-tokenizer configurations (GPT-2, Llama, etc.) and uses hash
matching to decide which pre-tokenizer regex to bake into the GGUF (Ch. 8).
A custom-trained BPE vocabulary can never match any entry in that table, even
when its underlying regex algorithm is functionally identical to a known
one, because the hash is computed over the specific vocabulary contents, not
just the algorithm. The correct response is deliberately falling back to a
known compatible pre-tokenizer type, not treating the mismatch as an error to
chase.

### `prefill`

The first forward pass of generation, where the entire initial prompt is
processed in one shot to populate the **KV cache**, before any decode steps
run (Ch. 11). Contrasted with a decode step, which processes exactly one new
token at a time afterward.

### `PreTrainedTokenizerFast`

The HF `tokenizers`-backed wrapper class that lets a trained byte-level BPE
tokenizer (`tokenizer.json`) be consumed directly by llama.cpp's GPT-2 vocab
export path (Ch. 8), replacing an earlier hand-built SentencePiece workaround
that was only ever a stopgap (see **pre-tokenizer hash mismatch**, **SentencePiece**).

### `quantization`

Reducing the numerical precision of stored weights (e.g. from 16-bit float
down to 8-bit or 4-bit integers with per-block scale factors) to shrink file
size and speed up inference, at some cost to output quality (Ch. 8).
`q8_0` and `q4_0` are llama.cpp quantization schemes (8-bit and 4-bit,
block-wise). A tensor with a very small dimension sometimes automatically
falls back to higher precision because the block size the scheme assumes
doesn't divide evenly into it.

### `residual connection`

Adding a sublayer's input back to its output (`x + sublayer(x)`) rather than
passing only the transformed value forward (Ch. 3). What actually makes deep
stacks of transformer blocks trainable: it gives gradients a direct
"shortcut" path back to earlier layers during backpropagation, avoiding the
vanishing-gradient problem that would otherwise compound across many stacked
layers.

### `RMSNorm`

Root Mean Square Normalization: rescales a vector by dividing by its root-mean-square
value, then applies a learned per-dimension scale -- no mean-centering, no
bias term (Ch. 3). Llama's normalization choice instead of LayerNorm,
because dropping mean-centering is cheaper and works just as well in
practice at this architecture's scale; this project mirrors it specifically
so the export bridge into `LlamaForCausalLM` lines up weight-for-weight.

### `RoPE (Rotary Position Embeddings)`

Injects positional information by rotating pairs of dimensions within each
head's Q and K vectors by an angle that depends on token position, rather
than adding a separate learned or fixed positional embedding vector (Ch. 3).
Precomputed `cos`/`sin` buffers have shape `(seq, head_dim/2)` (one
angle per rotated dimension-pair, per position). Crucial property exploited
in Ch. 11: rotation is relative, so a uniform shift in position partially
masks certain bugs (greedy-argmax output can look fine even when the
rotation is wrong, because relative angles between nearby positions stay
consistent) -- which is why **KV cache** correctness checks need raw
per-position logits, not just final generated tokens. Under caching, position
must be tracked as the token's *absolute* index in the full sequence, not
re-based to zero per window, or the rotation angles a cached token was
written with stop matching the angles used to read it back out.

### `SentencePiece`

A subword tokenizer format/library that llama.cpp's GGUF converter natively
understands (alongside HF-BPE) (Ch. 2, Ch. 5, Ch. 8). This project initially
forced a hand-rolled char-level vocabulary through the SentencePiece export
path as a workaround, which surfaced internals like `escape_whitespaces`
(literal spaces replaced with the `▁` glyph) and `add_space_prefix`
(defaulting to true) that have to match training-time tokenization exactly.
Later replaced by exporting the real byte-level BPE tokenizer directly via
`PreTrainedTokenizerFast` instead.

### `soft labels`

The full probability distribution a teacher model assigns across the entire
vocabulary for a given prediction, rather than a single correct answer (Ch.
12). Carries more information per example than a **hard label** -- e.g. it
reveals that "cat" and "kitten" were both plausible even though only one was
"correct," which a hard label can't express.

### `state_dict`

A PyTorch dictionary mapping module names to their parameter tensors --
the standard way model weights are saved, loaded, and transferred between
two different model implementations that share the same parameter shapes
and naming (Ch. 5). The HF bridge's `strict=True` load works because this
project's module names were written from the start to match HF's
`model.layers.N.self_attn.*` naming exactly.

### `SwiGLU`

A gated MLP variant used in Llama's feed-forward block (Ch. 3): instead of
one linear layer feeding an activation feeding another linear layer (two
projections), SwiGLU computes `silu(x @ W_gate) * (x @ W_up)` then projects
that product back down with a third matrix (`W_down`) -- three projections
total. The elementwise gating term lets the network modulate how much of
each feature passes through, which empirically improves quality over a plain
two-projection MLP at the same parameter budget.

### `teacher forcing`

Feeding the model the *actual* ground-truth previous tokens as input at every
training position, rather than its own (possibly wrong) previous
predictions (Ch. 4 implicitly via the training loop; named explicitly in Ch.
5's verification strategy). Used in Ch. 5 specifically to isolate "is the
math faithful between two implementations" from "do two floating-point
softmax kernels round identically" -- comparing per-position logits under
teacher forcing avoids the compounding divergence that greedy autoregressive
decoding would introduce.

### `temperature`

A scalar that divides logits before softmax during sampling (Ch. 4, also
Ch. 12 for distillation). Lower temperature sharpens the distribution toward
the highest-probability tokens (more deterministic, more repetitive at the
extreme); higher temperature flattens it (more random, more diverse). In
distillation, a *higher* temperature is applied to the teacher's logits
specifically to soften an otherwise overconfident distribution, exposing
more of the relative-probability signal the student is meant to learn from.

### `tied embeddings`

Sharing a single weight matrix between the input embedding table and the
output `lm_head` projection (since both have shape relating `vocab_size` to
`dim`, just transposed in use) (Ch. 4). Reduces parameter count and is
standard practice, but interacts badly with naive default initialization:
combined with default `N(0,1)` init, tied embeddings produced a logit
standard deviation of roughly 11 instead of a sane small value, which blew
up loss to ~94 instead of the expected ~`ln(vocab_size)` (~4.2). Fixed with
explicit small-variance initialization (`N(0, 0.02)`), standard GPT-2/Llama
practice -- the bug wasn't tying embeddings itself, it was relying on
PyTorch's default init instead of setting one deliberately.

### `top-k sampling`

A sampling strategy that restricts the next-token choice to only the `k`
highest-probability tokens (renormalizing their probabilities before
sampling), discarding the long tail of unlikely tokens entirely (Ch. 4).
Used alongside **temperature** to control output diversity vs. coherence.

### `vocab_size`

The total number of distinct tokens a tokenizer can produce / a model can
predict over. Determines the row count of the **embedding table** and the
output width of the **lm_head**, making it the single hyperparameter with
the widest blast radius across the model's tensor shapes (Ch. 2). Also sets
the "random baseline" loss value `ln(vocab_size)` used as a sanity check at
the start of training (Ch. 4).
