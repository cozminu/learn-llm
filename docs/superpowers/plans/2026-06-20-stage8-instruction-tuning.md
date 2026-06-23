# Stage 8 — Instruction-Tuning (SFT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `checkpoints/stage5.pt` on TinyStories-Instruct so the
model follows a simple instruction format (header fields + `Story:`
marker) instead of just continuing prose, and confirm the chat template
survives re-export to GGUF in a real runtime.

**Architecture:** New `src/sft_data.py` parses TinyStories-Instruct text
into masked (input_ids, targets) pairs and dynamically-padded batches;
`src/train_sft.py` continues training from `stage5.pt`'s weights for a
short, low-LR run; `scripts/export_hf.py` gains an optional
`chat_template` string that reproduces the exact training-time prompt
shape; `scripts/verify_sft.py` checks instruction-conditioning
qualitatively and confirms the chat template lands in the exported GGUF.

**Tech Stack:** PyTorch (MPS), HF `tokenizers`/`transformers`, llama.cpp
(`convert_hf_to_gguf.py`, already vendored under `llama.cpp/`, with a
built `llama-cli` at `llama.cpp/build/bin/llama-cli`), `gguf` Python
package (already a `convert_hf_to_gguf.py` dependency).

## Global Constraints

- Model architecture (`src/model.py`) and the non-cached `forward` path
  are NOT modified — this stage only adds a data pipeline and a training
  script; the export bit-exactness check from Stage 2/6 must stay valid.
- `ignore_index=-1` already wired into `TinyLM.forward`'s cross-entropy —
  reuse it, don't add new loss-masking machinery to the model.
- Dynamic per-batch padding (pad to each batch's own max length, not a
  fixed 512) — measured data: mean 251 tokens/example, p95 385, only ~2%
  of examples exceed 512.
- Examples whose prompt+response+eos exceeds `max_seq_len` (512) are
  dropped, not truncated.
- No pytest in this project (`pip show pytest` confirms it's absent) —
  follow the existing convention of standalone assert-based scripts run
  directly with `python` (see `scripts/verify_kv_cache.py`), not a test
  framework.
- New data/checkpoint files are gitignored the same way existing ones
  under `data/` and `checkpoints/` already are — only code and notes get
  committed.

---

### Task 1: Download TinyStories-Instruct data

**Files:**
- Create: `scripts/download_instruct_data.py`

**Interfaces:**
- Produces: `data/tinystories_instruct_train.txt`,
  `data/tinystories_instruct_val.txt` (consumed by Task 3's
  `load_dataset` and Task 4's training script)

- [ ] **Step 1: Write the download script**

Mirrors `scripts/download_data.py`'s pattern exactly, pointed at the
Instruct dataset (confirmed repo/file names: `roneneldan/TinyStories-Instruct`,
`TinyStories-Instruct-train.txt`, `TinyStories-Instruct-valid.txt`, same
`<|endoftext|>` separator).

```python
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

REPO_ID = "roneneldan/TinyStories-Instruct"
TRAIN_FILE = "TinyStories-Instruct-train.txt"
VAL_FILE = "TinyStories-Instruct-valid.txt"
STORY_SEP = "<|endoftext|>"


def download_train_subset(num_bytes: int) -> Path:
    url = hf_hub_url(REPO_ID, TRAIN_FILE, repo_type="dataset")
    resp = requests.get(url, headers={"Range": f"bytes=0-{num_bytes - 1}"}, timeout=60)
    resp.raise_for_status()
    text = resp.content.decode("utf-8", errors="ignore")

    last_sep = text.rfind(STORY_SEP)
    text = text[: last_sep + len(STORY_SEP)]

    out_path = DATA_DIR / "tinystories_instruct_train.txt"
    out_path.write_text(text)
    print(f"train subset: {len(text):,} chars, {text.count(STORY_SEP):,} examples -> {out_path}")
    return out_path


def download_val() -> Path:
    cached = hf_hub_download(REPO_ID, VAL_FILE, repo_type="dataset")
    out_path = DATA_DIR / "tinystories_instruct_val.txt"
    out_path.write_bytes(Path(cached).read_bytes())
    text = out_path.read_text()
    print(f"val: {len(text):,} chars, {text.count(STORY_SEP):,} examples -> {out_path}")
    return out_path


if __name__ == "__main__":
    num_bytes = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
    DATA_DIR.mkdir(exist_ok=True)
    download_train_subset(num_bytes)
    download_val()
```

- [ ] **Step 2: Run it**

Run: `source .venv/bin/activate && python scripts/download_instruct_data.py`
Expected: prints two lines like `train subset: 19,999,xyz chars, N,NNN
examples -> .../data/tinystories_instruct_train.txt` and `val: ...chars,
N,NNN examples -> .../data/tinystories_instruct_val.txt`. Confirm both
files exist and are non-empty:

Run: `ls -la data/tinystories_instruct_train.txt data/tinystories_instruct_val.txt`
Expected: both files listed with non-zero size.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_instruct_data.py
git commit -m "Add TinyStories-Instruct download script"
```

---

### Task 2: Parse TinyStories-Instruct examples (`parse_examples`, `build_example`)

**Files:**
- Create: `src/sft_data.py`
- Create: `scripts/test_sft_data.py`

**Interfaces:**
- Consumes: `src/tokenizer_bpe.py`'s `BPETokenizer` (`.encode(text:
  str) -> list[int]`, already exists)
- Produces: `parse_examples(text: str) -> list[tuple[str, str]]`,
  `build_example(tokenizer: BPETokenizer, prompt: str, response: str,
  eos_id: int, max_seq_len: int) -> tuple[list[int], list[int]] | None`
  (consumed by Task 3's `load_dataset`)

- [ ] **Step 1: Write the failing test**

Create `scripts/test_sft_data.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sft_data import build_example, parse_examples
from tokenizer_bpe import BPETokenizer

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"

SAMPLE_TEXT = """Random sentence: They are very excited and want to fly too.
Features: Dialogue
Summary: Tom and Anna are excited to go on a holiday with their parents, and they fly on a big plane to a place with sun and sand.
Story: 
Tom and Anna are brother and sister. They like to play with their toys.
<|endoftext|>
Summary: A story about a curious fox who finds a story book in the woods.
Story: 
Once there was a fox. The fox found a book.
<|endoftext|>
"""


def test_parse_examples_splits_headers_from_story():
    examples = parse_examples(SAMPLE_TEXT)
    assert len(examples) == 2, f"expected 2 examples, got {len(examples)}"

    prompt0, response0 = examples[0]
    assert prompt0 == (
        "Random sentence: They are very excited and want to fly too.\n"
        "Features: Dialogue\n"
        "Summary: Tom and Anna are excited to go on a holiday with their "
        "parents, and they fly on a big plane to a place with sun and sand.\n"
        "Story: \n"
    ), f"unexpected prompt0: {prompt0!r}"
    assert response0 == "Tom and Anna are brother and sister. They like to play with their toys."

    # Regression check: a Summary containing the word "story" must not
    # cause the Story: marker to be found early.
    prompt1, response1 = examples[1]
    assert "Story: \n" in prompt1
    assert prompt1.endswith("Story: \n")
    assert response1 == "Once there was a fox. The fox found a book."


def test_build_example_masks_prompt_tokens():
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")
    eos_id = tokenizer.encode("<|endoftext|>")[0]
    prompt = "Summary: A cat finds a hat.\nStory: \n"
    response = "Once there was a cat. It found a hat."

    built = build_example(tokenizer, prompt, response, eos_id, max_seq_len=512)
    assert built is not None
    input_ids, targets = built

    prompt_len = len(tokenizer.encode(prompt))
    assert len(input_ids) == len(targets)

    # Targets covering the prompt region (everything predicting a prompt
    # token) must be masked; the first response token must not be.
    assert all(t == -1 for t in targets[: prompt_len - 1])
    assert targets[prompt_len - 1] != -1


def test_build_example_drops_too_long():
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")
    eos_id = tokenizer.encode("<|endoftext|>")[0]
    prompt = "Summary: short.\nStory: \n"
    response = "word " * 1000

    built = build_example(tokenizer, prompt, response, eos_id, max_seq_len=512)
    assert built is None, "expected None for an over-length example"


if __name__ == "__main__":
    test_parse_examples_splits_headers_from_story()
    test_build_example_masks_prompt_tokens()
    test_build_example_drops_too_long()
    print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src python scripts/test_sft_data.py`
Expected: `ModuleNotFoundError: No module named 'sft_data'` (the module
doesn't exist yet).

- [ ] **Step 3: Implement `src/sft_data.py`**

```python
import re

from tokenizer_bpe import BPETokenizer

EOS_TOKEN = "<|endoftext|>"
STORY_MARKER_RE = re.compile(r"(?:^|\n)Story:\s*", re.MULTILINE)


def parse_examples(text: str) -> list[tuple[str, str]]:
    """Split raw TinyStories-Instruct text into (prompt, response) pairs.

    prompt is the header fields (Words/Features/Random sentence/Summary,
    a random subset in random order) plus the 'Story:' marker and its
    trailing whitespace. response is the story text. A plain substring
    search for "Story:" would be wrong here -- the word "story" can
    appear inside a Summary sentence -- so the marker is matched only at
    the start of a line.
    """
    examples = []
    for block in text.split(EOS_TOKEN):
        block = block.strip("\n")
        if not block.strip():
            continue
        match = STORY_MARKER_RE.search(block)
        if match is None:
            continue
        prompt = block[: match.end()]
        response = block[match.end() :].strip("\n")
        if not response:
            continue
        examples.append((prompt, response))
    return examples


def build_example(
    tokenizer: BPETokenizer,
    prompt: str,
    response: str,
    eos_id: int,
    max_seq_len: int,
) -> tuple[list[int], list[int]] | None:
    """Tokenize a (prompt, response) pair into (input_ids, targets).

    targets are shifted by one position from input_ids (next-token
    prediction); every target position whose predicted token still
    belongs to the prompt is set to -1, which TinyLM.forward's
    cross_entropy(ignore_index=-1) already drops from the loss. Returns
    None if the full sequence (prompt + response + eos) exceeds
    max_seq_len, rather than truncating.
    """
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
    ids = prompt_ids + response_ids + [eos_id]
    if len(ids) > max_seq_len + 1:
        return None

    input_ids = ids[:-1]
    targets = ids[1:]
    mask_len = len(prompt_ids) - 1
    targets = [-1] * mask_len + targets[mask_len:]
    return input_ids, targets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && PYTHONPATH=src python scripts/test_sft_data.py`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add src/sft_data.py scripts/test_sft_data.py
git commit -m "Add TinyStories-Instruct example parsing and masking"
```

---

### Task 3: Batch collation and dataset loading (`collate_batch`, `load_dataset`)

**Files:**
- Modify: `src/sft_data.py`
- Modify: `scripts/test_sft_data.py`

**Interfaces:**
- Consumes: `build_example`, `parse_examples` (Task 2)
- Produces: `collate_batch(examples: list[tuple[list[int], list[int]]],
  eos_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]`,
  `load_dataset(path: Path, tokenizer: BPETokenizer, eos_id: int,
  max_seq_len: int) -> list[tuple[list[int], list[int]]]` (both consumed
  by Task 4's `train_sft.py`)

- [ ] **Step 1: Write the failing test**

Append to `scripts/test_sft_data.py` (add the import and new test
functions, then add the calls to the `if __name__ == "__main__":` block):

```python
import torch

from sft_data import collate_batch, load_dataset
```

```python
def test_collate_batch_pads_dynamically():
    eos_id = 0
    examples = [
        ([1, 2, 3], [-1, -1, 30]),
        ([4, 5], [-1, 50]),
    ]
    input_ids, targets = collate_batch(examples, eos_id, device="cpu")

    assert input_ids.shape == (2, 3)
    assert targets.shape == (2, 3)
    assert input_ids[1].tolist() == [4, 5, eos_id]
    assert targets[1].tolist() == [-1, 50, -1]


def test_load_dataset_respects_max_seq_len():
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")
    eos_id = tokenizer.encode("<|endoftext|>")[0]
    text = SAMPLE_TEXT + (
        "Summary: short.\nStory: \n" + "word " * 1000 + "\n<|endoftext|>\n"
    )
    tmp_path = Path("/tmp/sft_data_test_sample.txt")
    tmp_path.write_text(text)

    examples = load_dataset(tmp_path, tokenizer, eos_id, max_seq_len=512)
    assert len(examples) == 2, f"expected the 2 short examples, got {len(examples)}"
    for input_ids, _ in examples:
        assert len(input_ids) <= 512
    tmp_path.unlink()
```

Update the `if __name__ == "__main__":` block to also call the two new
tests:

```python
if __name__ == "__main__":
    test_parse_examples_splits_headers_from_story()
    test_build_example_masks_prompt_tokens()
    test_build_example_drops_too_long()
    test_collate_batch_pads_dynamically()
    test_load_dataset_respects_max_seq_len()
    print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src python scripts/test_sft_data.py`
Expected: `ImportError: cannot import name 'collate_batch' from 'sft_data'`

- [ ] **Step 3: Implement `collate_batch` and `load_dataset`**

Append to `src/sft_data.py`:

```python
from pathlib import Path

import torch


def collate_batch(
    examples: list[tuple[list[int], list[int]]], eos_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of (input_ids, targets) pairs to the batch's own max
    length (not a fixed size) -- most examples are far shorter than
    max_seq_len, so fixed-size padding would waste most of the compute on
    positions contributing nothing to the loss."""
    max_len = max(len(ids) for ids, _ in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), eos_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
    for i, (ids, tgt) in enumerate(examples):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        targets[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return input_ids.to(device), targets.to(device)


def load_dataset(
    path: Path, tokenizer: BPETokenizer, eos_id: int, max_seq_len: int
) -> list[tuple[list[int], list[int]]]:
    text = path.read_text()
    examples = []
    dropped = 0
    for prompt, response in parse_examples(text):
        built = build_example(tokenizer, prompt, response, eos_id, max_seq_len)
        if built is None:
            dropped += 1
            continue
        examples.append(built)
    print(f"{path.name}: {len(examples)} examples, {dropped} dropped (over {max_seq_len} tokens)")
    return examples
```

(Move the `from pathlib import Path` and `import torch` lines to the top
of the file alongside the existing imports rather than leaving them
mid-file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && PYTHONPATH=src python scripts/test_sft_data.py`
Expected: `PASS`

- [ ] **Step 5: Run `load_dataset` against the real downloaded val file**

Run:
```bash
source .venv/bin/activate && PYTHONPATH=src python -c "
from pathlib import Path
from sft_data import load_dataset
from tokenizer_bpe import BPETokenizer
tok = BPETokenizer.load('checkpoints/tokenizer.json')
eos_id = tok.encode('<|endoftext|>')[0]
load_dataset(Path('data/tinystories_instruct_val.txt'), tok, eos_id, max_seq_len=512)
"
```
Expected: a line like `tinystories_instruct_val.txt: NNNN examples, MM
dropped (over 512 tokens)` where `dropped` is roughly ~2% of the total
(matches the measured token-length distribution from the design spec).

- [ ] **Step 6: Commit**

```bash
git add src/sft_data.py scripts/test_sft_data.py
git commit -m "Add dynamic-padding batch collation and dataset loading"
```

---

### Task 4: SFT training script (`src/train_sft.py`)

**Files:**
- Create: `src/train_sft.py`

**Interfaces:**
- Consumes: `load_dataset`, `collate_batch` (Task 3); `TinyLM`,
  `ModelConfig` (`src/model.py`, `src/config.py`, unchanged); `BPETokenizer`
  (`src/tokenizer_bpe.py`, unchanged)
- Produces: `checkpoints/stage8.pt` (same `{"model": state_dict,
  "config": cfg}` shape as `stage5.pt`, consumed by Task 5's training
  run and Task 6's export)

- [ ] **Step 1: Write `src/train_sft.py`**

Mirrors `src/train.py`'s structure (lr schedule, eval loop, best-val
checkpointing) but loads `stage5.pt`'s weights instead of random init,
uses a lower LR for a much shorter run, and draws batches from the
instruct dataset via `load_dataset`/`collate_batch` instead of a sliding
window over a continuous token stream.

```python
import math
import random
import time
from pathlib import Path

import torch

from config import ModelConfig
from model import TinyLM
from sft_data import collate_batch, load_dataset
from tokenizer_bpe import BPETokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
MAX_STEPS = 1200
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
LR = 2e-4
WARMUP_STEPS = 50
GRAD_CLIP = 1.0

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_DATA_PATH = DATA_DIR / "tinystories_instruct_train.txt"
VAL_DATA_PATH = DATA_DIR / "tinystories_instruct_val.txt"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
BASE_CKPT_PATH = CKPT_DIR / "stage5.pt"
TOKENIZER_PATH = CKPT_DIR / "tokenizer.json"


def sample_batch(examples: list, batch_size: int, eos_id: int, device: str):
    batch = random.sample(examples, batch_size)
    return collate_batch(batch, eos_id, device)


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.1 * LR + 0.5 * (LR - 0.1 * LR) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: TinyLM, train_examples, val_examples, eos_id: int):
    model.eval()
    out = {}
    for name, examples in [("train", train_examples), ("val", val_examples)]:
        losses = torch.zeros(EVAL_BATCHES)
        for i in range(EVAL_BATCHES):
            x, y = sample_batch(examples, BATCH_SIZE, eos_id, DEVICE)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    train_examples = load_dataset(TRAIN_DATA_PATH, tokenizer, eos_id, max_seq_len=512)
    val_examples = load_dataset(VAL_DATA_PATH, tokenizer, eos_id, max_seq_len=512)

    base_ckpt = torch.load(BASE_CKPT_PATH, map_location="cpu", weights_only=False)
    cfg: ModelConfig = base_ckpt["config"]
    model = TinyLM(cfg).to(DEVICE)
    model.load_state_dict(base_ckpt["model"])
    print(f"device={DEVICE} params={model.num_params():,} vocab={tokenizer.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    ckpt_path = CKPT_DIR / "stage8.pt"
    best_val = float("inf")
    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        x, y = sample_batch(train_examples, BATCH_SIZE, eos_id, DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            losses = estimate_loss(model, train_examples, val_examples, eos_id)
            elapsed = time.time() - t0
            improved = losses["val"] < best_val
            if improved:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr_at(step):.6f} | {elapsed:.1f}s" + (" | saved" if improved else "")
            )

    print(f"best val {best_val:.4f}, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with a tiny step budget**

Temporarily run with a drastically reduced step count to confirm the
script runs end-to-end without crashing, before committing to the full
run in Task 5.

Run:
```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 PYTHONPATH=src python -c "
import train_sft
train_sft.MAX_STEPS = 10
train_sft.WARMUP_STEPS = 2
train_sft.EVAL_INTERVAL = 5
train_sft.EVAL_BATCHES = 3
train_sft.main()
"
```
Expected: prints the `device=... params=... vocab=...` line, then two
eval lines (steps 5 and 10) with finite, non-NaN train/val losses, then
`best val ..., checkpoint at .../checkpoints/stage8.pt`. Delete this
smoke-test checkpoint afterward so it doesn't get mistaken for the real
run's result:

Run: `rm -f checkpoints/stage8.pt`

- [ ] **Step 3: Commit**

```bash
git add src/train_sft.py
git commit -m "Add SFT training script continuing from stage5.pt"
```

---

### Task 5: Run the real SFT training

**Files:** none (produces `checkpoints/stage8.pt`, gitignored)

**Interfaces:**
- Consumes: `src/train_sft.py` (Task 4)
- Produces: `checkpoints/stage8.pt` (consumed by Task 6/7's export and
  verification)

- [ ] **Step 1: Launch the full run in the background with unbuffered output**

Run (background, following the Stage 5 lesson about buffered stdout on
redirected output):
```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 PYTHONPATH=src python src/train_sft.py > /tmp/train_sft.log 2>&1 &
echo "launched pid $!"
```

- [ ] **Step 2: Monitor until completion**

Run: `tail -f /tmp/train_sft.log` (or poll with `tail -n 20
/tmp/train_sft.log`) until the final `best val ..., checkpoint at
.../checkpoints/stage8.pt` line appears.
Expected: val loss trends downward across eval points and stays finite
(no NaN/inf); given Stage 5's measured ~1.05s/step at this model size and
batch size, 1200 steps should finish in well under 30 minutes.

- [ ] **Step 3: Confirm the checkpoint exists**

Run: `ls -la checkpoints/stage8.pt`
Expected: file present, non-zero size.

(No commit — `checkpoints/` is gitignored.)

---

### Task 6: Chat template support in `export_hf.py` / `export_gguf.py`

**Files:**
- Modify: `scripts/export_hf.py`
- Modify: `scripts/export_gguf.py`

**Interfaces:**
- Produces: `export_hf.CHAT_TEMPLATE` (str constant), `export_hf.export(ckpt_name:
  str, out_dir_name: str, chat_template: str | None = None) -> Path`
  (signature change: new optional trailing param, existing calls
  unaffected), `export_gguf.main(ckpt_name: str, name: str, chat_template:
  str | None = None) -> None` (same backward-compatible signature change)
- Consumed by: Task 7's `verify_sft.py`

- [ ] **Step 1: Write the failing test**

```bash
source .venv/bin/activate && PYTHONPATH=src:scripts python -c "
import json
from export_hf import export, CHAT_TEMPLATE

out_dir = export('stage5.pt', 'stage8-template-test-hf', chat_template=CHAT_TEMPLATE)
cfg = json.loads((out_dir / 'tokenizer_config.json').read_text())
assert cfg.get('chat_template') == CHAT_TEMPLATE, 'chat_template missing from tokenizer_config.json'
print('PASS')
"
```
Expected: `ImportError: cannot import name 'CHAT_TEMPLATE' from 'export_hf'`
(it doesn't exist yet; uses `stage5.pt` here only because it already
exists on disk -- this step is just proving the export-side wiring,
independent of which checkpoint it runs against).

- [ ] **Step 2: Implement the chat template constant and wire it through `export()`**

In `scripts/export_hf.py`, add the constant near the top (after `EOS_TOKEN`)
and thread it through `export()`:

```python
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}\nStory: \n"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}"
    "{% endif %}"
    "{% endfor %}"
)
```

Change the `export` signature and body:

```python
def export(ckpt_name: str, out_dir_name: str, chat_template: str | None = None) -> Path:
    ckpt = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["model"]

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(CKPT_DIR / "tokenizer.json"),
        bos_token=EOS_TOKEN,
        eos_token=EOS_TOKEN,
    )
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
```

(Everything else in `export()` stays unchanged.)

- [ ] **Step 3: Run test to verify it passes**

Run the Step 1 command again.
Expected: `PASS`. Clean up the test directory:

Run: `rm -rf checkpoints/stage8-template-test-hf`

- [ ] **Step 4: Thread `chat_template` through `export_gguf.py`**

In `scripts/export_gguf.py`, change `main`'s signature and its call into
`export`:

```python
def main(ckpt_name: str, name: str, chat_template: str | None = None) -> None:
    hf_dir = export(ckpt_name, f"{name}-hf", chat_template=chat_template)
    gguf_path = CKPT_DIR / f"{name}.gguf"

    patch_unrecognized_pretokenizer()
    import convert_hf_to_gguf

    sys.argv = ["convert_hf_to_gguf.py", str(hf_dir), "--outfile", str(gguf_path), "--outtype", "f16"]
    convert_hf_to_gguf.main()

    print(f"GGUF ready: {gguf_path}")
```

(The `if __name__ == "__main__":` block at the bottom is unchanged --
CLI invocation still only passes `ckpt_name`/`name`, so `chat_template`
defaults to `None` there, which is correct for the base-model export
path.)

- [ ] **Step 5: Regression-check the base model export still works unchanged**

Run: `source .venv/bin/activate && PYTHONPATH=src:llama.cpp:scripts python scripts/export_gguf.py stage5.pt stage5-regression-test`
Expected: completes with `GGUF ready: .../checkpoints/stage5-regression-test.gguf`,
same as the existing Stage 6 export path (proves the new optional
parameter didn't break the no-chat-template case). Clean up:

Run: `rm -rf checkpoints/stage5-regression-test-hf checkpoints/stage5-regression-test.gguf`

- [ ] **Step 6: Commit**

```bash
git add scripts/export_hf.py scripts/export_gguf.py
git commit -m "Add optional chat_template support to the export pipeline"
```

---

### Task 7: Verify instruction-following and chat template export (`scripts/verify_sft.py`)

**Files:**
- Create: `scripts/verify_sft.py`
- Create: `notes/stage8.md`

**Interfaces:**
- Consumes: `checkpoints/stage8.pt` (Task 5), `export_gguf.main`,
  `export_hf.CHAT_TEMPLATE` (Task 6)

- [ ] **Step 1: Write the conditioning check**

```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "."))

from model import TinyLM
from tokenizer_bpe import BPETokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"

FIXED_FIELDS = "Features: Dialogue\n"
SUMMARIES = [
    "Summary: A dog finds a lost ball in the park and brings it home.",
    "Summary: Two sisters bake a cake for their mother's birthday.",
    "Summary: A boy is scared of a thunderstorm until his dad comforts him.",
]


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    kv_cache = model.new_kv_cache()
    logits, _ = model(ids, kv_cache=kv_cache)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out = torch.cat([ids, next_id], dim=1)
    for _ in range(max_new_tokens - 1):
        logits, _ = model(next_id, kv_cache=kv_cache)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
        if next_id.item() == tokenizer.encode("<|endoftext|>")[0]:
            break
    return tokenizer.decode(out[0].tolist())


def check_conditioning(model, tokenizer) -> None:
    print("=== Conditioning check: same Features, varied Summary ===")
    for summary in SUMMARIES:
        prompt = FIXED_FIELDS + summary + "\nStory: \n"
        text = generate(model, tokenizer, prompt)
        print(f"--- {summary}")
        print(text)
        print()


def main():
    ckpt = torch.load(CKPT_DIR / "stage8.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")

    model = TinyLM(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    check_conditioning(model, tokenizer)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and read the output**

Run: `source .venv/bin/activate && PYTHONPATH=src python scripts/verify_sft.py`
Expected: three generated stories printed, one per summary. This is a
qualitative check (matching the project's established pattern for
sampling-quality checks, e.g. Stage 5/7's printed samples) — read all
three and confirm each story's content actually reflects its own
summary (e.g. the dog/ball one should be about a dog and a ball, not
generic prose, and the three should clearly differ from each other in
subject matter). If they don't differ, that's a real finding to report
back, not something to paper over by tweaking the script.

- [ ] **Step 3: Add the export + GGUF metadata check to the same script**

Append to `scripts/verify_sft.py` (add the new imports at the top and
the new function, then call it from `main`):

```python
import json

from gguf import GGUFReader

sys.path.insert(0, str(Path(__file__).parent.parent / "llama.cpp"))
sys.path.insert(0, str(Path(__file__).parent))

from export_gguf import main as export_gguf_main
from export_hf import CHAT_TEMPLATE


def check_export_and_chat_template() -> None:
    print("=== Export check: stage8.pt -> GGUF with chat_template ===")
    export_gguf_main("stage8.pt", "stage8", chat_template=CHAT_TEMPLATE)

    reader = GGUFReader(str(CKPT_DIR / "stage8.gguf"))
    field = reader.fields.get("tokenizer.chat_template")
    assert field is not None, "tokenizer.chat_template missing from stage8.gguf metadata"
    template_bytes = bytes(field.parts[-1])
    template_str = template_bytes.decode("utf-8")
    assert template_str == CHAT_TEMPLATE, "GGUF chat_template does not match export_hf.CHAT_TEMPLATE"
    print("tokenizer.chat_template present in stage8.gguf and matches export_hf.CHAT_TEMPLATE")

    hf_config_path = CKPT_DIR / "stage8-hf" / "tokenizer_config.json"
    hf_config = json.loads(hf_config_path.read_text())
    assert hf_config.get("chat_template") == CHAT_TEMPLATE
    print("tokenizer_config.json chat_template also confirmed")
```

Update `main()`:

```python
def main():
    ckpt = torch.load(CKPT_DIR / "stage8.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")

    model = TinyLM(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    check_conditioning(model, tokenizer)
    check_export_and_chat_template()
```

- [ ] **Step 4: Run it**

Run: `source .venv/bin/activate && PYTHONPATH=src python scripts/verify_sft.py`
Expected: the conditioning output as before, then `GGUF ready:
.../checkpoints/stage8.gguf`, then `tokenizer.chat_template present in
stage8.gguf and matches export_hf.CHAT_TEMPLATE` and
`tokenizer_config.json chat_template also confirmed`. If either assert
fails, stop and investigate before continuing -- don't relax the
assertion to make it pass.

- [ ] **Step 5: Manually confirm a real runtime applies the template**

Run:
```bash
./llama.cpp/build/bin/llama-cli -m checkpoints/stage8.gguf -st -p "Features: Dialogue
Summary: A dog finds a lost ball in the park and brings it home."
```
Expected: `llama-cli` auto-detects the embedded chat template (conversation
mode auto-enables when a chat template is present), renders the prompt
through it (so the model sees the exact `...\nStory: \n`-terminated text
it was trained on), and prints a generated story conditioned on that
summary. Record the actual output in `notes/stage8.md` (Step 6) — if
`llama-cli` doesn't apply the template automatically, pass `--jinja`
explicitly and note that in the write-up.

- [ ] **Step 6: Write `notes/stage8.md`**

Follow the existing per-stage notes convention (see `notes/stage7.md`
for structure: goal, what changed, a verification section, a result
section). Include:
- The dataset format discovery (variable header subset, regex-anchored
  `Story:` marker) and why a plain substring search would have been
  wrong.
- The masking approach (`ignore_index=-1`, first stage to actually
  exercise it).
- The dynamic-padding batching decision and the measured token-length
  data that motivated it (mean 251, p95 385, ~2% over 512 dropped).
- The chat template design — that it's not a generic role-based
  template but one that reproduces the exact training-time text shape,
  verified against `apply_chat_template` output before being wired into
  the plan.
- The conditioning check results (Step 2) — actual generated text
  excerpts, with an honest assessment of whether conditioning was
  "visible" per the Stage 8 success bar.
- The `llama-cli` real-runtime result (Step 5).

- [ ] **Step 7: Commit**

```bash
git add scripts/verify_sft.py notes/stage8.md
git commit -m "Verify SFT conditioning and chat-template export to GGUF"
```

---

## Self-Review Notes

- **Spec coverage:** download (Task 1), parsing/masking (Task 2),
  batching/dataset loading (Task 3), training script + real run (Tasks
  4-5), chat template export (Task 6), conditioning + export + real
  runtime verification + notes (Task 7) — all spec sections covered.
- **Placeholder scan:** no TBD/TODO; every step has runnable code or an
  exact command with a stated expected result.
- **Type consistency:** `build_example`/`collate_batch`/`load_dataset`
  signatures introduced in Task 2/3 are used identically in Task 4's
  `train_sft.py`; `export()`'s new `chat_template` param (Task 6) matches
  its use in Task 7's `verify_sft.py` and `export_gguf.main`.
