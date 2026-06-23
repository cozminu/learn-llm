# Stage 8 -- Instruction-tuning (SFT)

Goal: turn the TinyStories base model (`stage5.pt`) into something that
follows a held-out instruction in the TinyStories-Instruct format --
visibly conditioned on the instruction, not just continuing it as prose
-- and confirm the chat template needed to drive that at inference time
survives the export to GGUF.

## Dataset format discovery
TinyStories-Instruct examples are `<|endoftext|>`-separated blocks, each
with a random subset of header fields (`Random sentence:`, `Features:`,
`Summary:`, etc.) in random order, followed by `Story: \n` and the story
text. `parse_examples` (`src/sft_data.py`) splits on `<|endoftext|>` and
locates the prompt/response boundary with a regex anchored to start-of-line,
`(?:^|\n)Story:\s*` -- not a plain substring search for `"Story:"`. The
`Summary:` field is free-text and can itself contain the word "story"
(e.g. "a story about a curious fox"), so a substring search would
sometimes find a false boundary inside the prompt and corrupt the
prompt/response split. The regression test for exactly this case lives in
`scripts/test_sft_data.py:test_parse_examples_splits_headers_from_story`.

## Masking with `ignore_index=-1`
`TinyLM.forward` has passed `ignore_index=-1` to `cross_entropy` since
Stage 1, unused until now. `build_example` tokenizes `prompt + response +
eos`, builds the usual shifted-by-one `(input_ids, targets)` pair for
next-token prediction, then overwrites every target position whose
predicted token still falls inside the prompt with `-1`:
`mask_len = len(prompt_ids) - 1`. The `-1` is the boundary case to get
right -- the prompt's *last* token's target is the *first response*
token, which must stay unmasked (that's the first real prediction the
model is supposed to make); everything before it predicts another prompt
token and gets dropped from the loss. This is exactly what loss masking
for SFT means: the prompt is context, not a prediction target.

Examples whose full tokenized length exceeds `max_seq_len` are dropped
entirely (`build_example` returns `None`), not truncated -- truncating
would either cut off the story mid-sentence or, worse, cut into the
prompt and silently change what the model is conditioned on.

## Dynamic-padding batching
Per-example token lengths varied far more than Stage 1-6's fixed-length
TinyStories windows (mean 251, p95 385, with ~2% of examples over the
512-token `max_seq_len` and dropped by the rule above). Padding every
batch to a fixed 512 would waste most of the forward pass on positions
that contribute nothing to the loss, since the median example is under
half that length.

`collate_batch` instead pads each batch to its own max length, rounded up
to the nearest multiple of 64 (`PAD_BUCKET`). Padding to the batch's exact
max length (rather than a bucket) was tried first and made every batch a
distinct tensor shape; MPS caches a compiled graph per distinct shape, and
hundreds of distinct shapes over a run visibly degraded step time (10-20x
slower) and pushed memory up toward OOM. Bucketing to multiples of 64 over
the `[64, 512]` range caps shape diversity to ~8 buckets, letting MPS
reuse compiled graphs across most batches at a small, bounded padding
cost. (A separate, unrelated memory-pressure issue from other running
applications was initially suspected to be caused by this and was later
ruled out -- the bucketing was kept anyway as a reasonable hygiene
improvement independent of that investigation.)

## Training
`src/train_sft.py` continues from `stage5.pt`'s weights and config (not
a new model), trains 1200 steps with AdamW (warmup+cosine decay, same
shape as Stage 1/5's schedule), evaluates train/val loss every 100 steps,
and checkpoints only on val-loss improvement. Best val loss: **1.3856**,
saved to `checkpoints/stage8.pt`.

## Chat template design
`export_hf.CHAT_TEMPLATE` (Stage 8 Task 6) is deliberately not a generic
role-based template -- it reproduces the *exact* training-time text shape:

```
{% for message in messages %}
{% if message['role'] == 'user' %}{{ message['content'] }}\nStory: \n
{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}
{% endfor %}
```

A `user` message renders as its content followed by `\nStory: \n`,
matching `parse_examples`'s prompt format precisely (header fields ending
in the `Story: ` marker and its trailing newline); an `assistant` message
renders as plain content with no added markup. This was checked against
`tokenizer.apply_chat_template(...)` output and compared against
`parse_examples`'s actual prompt strings before being wired into
`export_hf.export()` and `export_gguf.main()` as an optional
`chat_template` parameter -- a generic `<|user|>...<|assistant|>`-style
template would tokenize to a different sequence than anything the model
was trained on and silently produce poor/garbage completions despite
"working" structurally.

## Verification (`scripts/verify_sft.py`)

### 1. Conditioning check
Same `Features: Dialogue` field, three different `Summary:` values, greedy
(argmax) decoding, no other variation:

**"A dog finds a lost ball in the park and brings it home."**
> Once upon a time, there was a dog. He was very happy and liked to play.
> One day, he found a ball in the park. He was so excited and ran to show
> his friend. His friend was happy too and said, "Let's play!" The dog
> was so happy and he wagged his tail.
> They played with the ball for a long time. The dog was so happy and he
> kept playing with the ball. He was so glad to have found the ball.
> After a while, the dog's friend said, "Let's go home now." The

**"Two sisters bake a cake for their mother's birthday."**
> Once upon a time, there were two sisters. One was named Lucy and the
> other was named Jack. They were very excited to bake a cake together.
> Lucy said to Jack, "Let's make a cake!"
> Jack said, "Yes, let's do it!"
> So they started to bake the cake. They mixed the ingredients and put
> them in the oven. After a while, the cake was ready.
> Lucy and Jack took it out of the oven. They put it on the table and it
> looked so yummy!
> They both smiled and said, "Let's make a

**"A boy is scared of a thunderstorm until his dad comforts him."**
> Once upon a time, there was a boy. He was very scared. He was scared
> of the thunder.
> One day, his dad said, "Don't be scared, it's just thunder. It won't
> hurt you."
> The boy was still scared, but he was brave. He asked, "Can I see the
> thunder?"
> His dad said, "Yes, but don't worry. I will always be here for you."
> So the boy and his dad went outside. They saw the thunderstorm was very
> loud. The boy was scared, but his dad was there to

Assessment against the Stage 8 success bar ("visibly conditioned on the
instruction, not just continuing it as prose"): clearly met. All three
stories track their own summary's subject matter (dog + ball + park;
two sisters + cake + baking; boy + thunderstorm + dad comforting him) and
none reads as a generic TinyStories continuation that ignores the prompt
-- they differ in characters, setting, and plot in a way that's directly
traceable to the differing `Summary:` field, with the shared `Features:
Dialogue` field also honored (all three stories contain quoted dialogue).
One data-driven quirk worth noting: the model named one of the "two
sisters" Jack -- TinyStories-Instruct's name pool isn't gender-consistent
with header fields, so this is a dataset artifact, not a conditioning
failure (the *sibling relationship and cake-baking plot* are still
correctly conditioned on the summary).

A second, more relevant gap in the same story: the summary's "for their
mother's birthday" detail never surfaces anywhere in the 120-token
generation -- no mention of mother, birthday, or any gift-giving
occasion, just the two sisters baking. This is a real partial-
conditioning miss (the *occasion* named in the instruction was dropped),
distinct from the cosmetic naming quirk above, and is more relevant
evidence about conditioning quality than the name mismatch. It doesn't
change the overall conclusion -- the sibling relationship and
cake-baking activity are still clearly traceable to the summary, and the
other two stories track their summaries closely including specific
details (park, ball, thunder, dad's reassurance) -- but a fuller
instruction with multiple clauses isn't guaranteed to have every clause
honored at this model size/training budget.

### 2. Export + GGUF chat-template metadata
`export_gguf.main("stage8.pt", "stage8", chat_template=CHAT_TEMPLATE)` ran
the full HF-bridge + `convert_hf_to_gguf.py` pipeline (same pre-tokenizer
fallback workaround as Stage 6 -- expected `WARNING: BPE pre-tokenizer not
recognized`, harmless), producing `checkpoints/stage8.gguf` (43.9 MB, f16).
Conversion log shows `Setting chat_template to {% for message in
messages %}...` -- confirming the template is embedded in the GGUF's
`tokenizer.chat_template` metadata field, not just the HF tokenizer config.

Read back with `GGUFReader`: `tokenizer.chat_template` field present,
decoded value byte-for-byte equal to `export_hf.CHAT_TEMPLATE`. Cross-check
against `checkpoints/stage8-hf/tokenizer_config.json`'s `chat_template`
key: also an exact match. Both assertions passed without modification.

### 3. Real-runtime check (`llama-cli`)
```
./llama.cpp/build/bin/llama-cli -m checkpoints/stage8.gguf -st -p "Features: Dialogue
Summary: A dog finds a lost ball in the park and brings it home."
```
`llama-cli` auto-detected the embedded chat template and entered
conversation mode on its own (no `--jinja` flag needed) -- it printed the
chat-style banner and a `>` prompt with `/exit`, `/regen`, `/clear`
commands, confirming the template was picked up automatically rather than
the prompt being treated as raw text completion. Output:

> Once upon a time, there was a dog. The dog was very lonely and didn't
> have any friends. One day, the dog saw a ball. He wanted to take it and
> have fun. He thought and thought about how to find it. Then he had an
> idea. He asked a bird to help him find the ball. The bird said "OK, but
> you must promise to bring it home". The dog was so excited and he said
> "Yes, I promise!" The bird flew away to find the ball. The dog was sad
> that he couldn't catch the ball. But the dog was clever and he found
> the ball. He took the ball home and played with it. He was very happy
> and the dog was

A dog/ball/park story again, distinct in details from the PyTorch-side
generation for the same summary (different opening, a bird helper
subplot) but clearly the same conditioned subject matter -- confirms the
template renders the prompt the same way in a real GGUF-consuming runtime
as during training, end to end.

## Result
Stage 8 complete: the SFT model visibly follows varied held-out
instructions rather than producing generic continuations (Step 2), the
chat template needed to reproduce that behavior at inference time is
embedded correctly in the exported GGUF and verified byte-exact against
its source (Step 3/4), and a real runtime (`llama-cli`) auto-applies that
template and produces a correctly-conditioned story without any extra
flags (Step 5). This closes the loop the Stage 6 export didn't cover --
chat-template metadata -- and is the last piece needed to call this a
from-scratch model that "follows instructions," not just continues
prose.
