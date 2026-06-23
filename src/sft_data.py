import re
from pathlib import Path

import torch

from tokenizer_bpe import BPETokenizer

EOS_TOKEN = "<|endoftext|>"
STORY_MARKER_RE = re.compile(r"(?:^|\n)Story:\s*", re.MULTILINE)
PAD_BUCKET = 64


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


def collate_batch(
    examples: list[tuple[list[int], list[int]]], eos_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of (input_ids, targets) pairs to the batch's own max
    length rounded up to the nearest PAD_BUCKET (not a fixed size, and not
    the exact max length either) -- most examples are far shorter than
    max_seq_len, so fixed-size padding would waste most of the compute on
    positions contributing nothing to the loss. Padding to the exact max
    length makes every batch a distinct tensor shape, which was suspected
    (initially, before the actual cause -- host-level system memory
    pressure from other running applications, not PyTorch's MPS backend --
    was identified) of degrading MPS's per-shape compiled-graph cache.
    That suspicion didn't hold up, but rounding to PAD_BUCKET multiples is
    kept anyway: it caps a run to ~8 distinct shapes (64, 128, ..., 512)
    instead of one per batch, which is a reasonable hygiene improvement
    (less shape churn, near-zero extra padding cost) independent of that
    investigation. See notes/stage8.md for the full root-cause writeup."""
    raw_max_len = max(len(ids) for ids, _ in examples)
    max_len = ((raw_max_len + PAD_BUCKET - 1) // PAD_BUCKET) * PAD_BUCKET
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
