import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from sft_data import build_example, collate_batch, load_dataset, parse_examples
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


def test_collate_batch_pads_to_bucket():
    eos_id = 0
    examples = [
        ([1, 2, 3], [-1, -1, 30]),
        ([4, 5], [-1, 50]),
    ]
    input_ids, targets = collate_batch(examples, eos_id, device="cpu")

    # Raw max length is 3, rounded up to the nearest PAD_BUCKET (64) -> 64.
    # Padding to the exact max length (3) would make every batch a distinct
    # tensor shape; bucketing caps shape diversity as cheap hygiene (see
    # collate_batch's docstring -- this was not the fix for the OOM the
    # project hit, which turned out to be host memory pressure).
    assert input_ids.shape == (2, 64)
    assert targets.shape == (2, 64)
    assert input_ids[1, :3].tolist() == [4, 5, eos_id]
    assert targets[1, :3].tolist() == [-1, 50, -1]
    assert input_ids[1, 3:].tolist() == [eos_id] * 61
    assert targets[1, 3:].tolist() == [-1] * 61


def test_collate_batch_caps_shape_diversity():
    eos_id = 0
    # Three batches with different raw max lengths that all fall in the
    # same PAD_BUCKET range should produce identical tensor shapes.
    batch_a = [([1] * 10, [-1] * 10)]
    batch_b = [([1] * 50, [-1] * 50)]
    batch_c = [([1] * 64, [-1] * 64)]

    shape_a = collate_batch(batch_a, eos_id, device="cpu")[0].shape
    shape_b = collate_batch(batch_b, eos_id, device="cpu")[0].shape
    shape_c = collate_batch(batch_c, eos_id, device="cpu")[0].shape

    assert shape_a == shape_b == (1, 64)
    assert shape_c == (1, 64)


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


if __name__ == "__main__":
    test_parse_examples_splits_headers_from_story()
    test_build_example_masks_prompt_tokens()
    test_build_example_drops_too_long()
    test_collate_batch_pads_to_bucket()
    test_collate_batch_caps_shape_diversity()
    test_load_dataset_respects_max_seq_len()
    print("PASS")
