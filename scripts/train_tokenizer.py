import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).parent.parent
CKPT_DIR = ROOT / "checkpoints"
EOS_TOKEN = "<|endoftext|>"


def train(corpus_path: Path, vocab_size: int) -> Path:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOS_TOKEN],
    )
    tokenizer.train([str(corpus_path)], trainer)
    tokenizer.add_special_tokens([EOS_TOKEN])

    out_path = CKPT_DIR / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"vocab_size={tokenizer.get_vocab_size()} saved {out_path}")
    return out_path


if __name__ == "__main__":
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "shakespeare.txt"
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    train(corpus_path, vocab_size)
