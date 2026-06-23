import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ModelConfig
from model import TinyLM
from tokenizer_bpe import BPETokenizer

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BLOCK_SIZE = 512
BATCH_SIZE = 32
SMOKE_STEPS = 150
LR = 1e-3
FULL_STEPS = 8000

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
DATA_DIR = Path(__file__).parent.parent / "data"


def get_batch(data: torch.Tensor):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix]).to(DEVICE)
    y = torch.stack([data[i + 1 : i + 1 + BLOCK_SIZE] for i in ix]).to(DEVICE)
    return x, y


def main():
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")
    # First 5MB of real training text is enough to exercise real token statistics
    # for an LR-stability check without paying to tokenize the full 150MB file.
    text = (DATA_DIR / "tinystories_train.txt").read_text()[:5_000_000]
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=512,
        n_layers=10,
        n_heads=8,
        intermediate_size=2048,
        max_seq_len=BLOCK_SIZE,
    )
    model = TinyLM(cfg).to(DEVICE)
    print(f"device={DEVICE} params={model.num_params():,} vocab={tokenizer.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    t0 = time.time()
    for step in range(SMOKE_STEPS):
        x, y = get_batch(data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step == SMOKE_STEPS - 1:
            print(f"step {step:4d} loss {loss.item():.4f}")
        if not torch.isfinite(loss):
            print("loss diverged (NaN/Inf)")
            return

    elapsed = time.time() - t0
    print(
        f"elapsed {elapsed:.2f}s for {SMOKE_STEPS} steps -> {elapsed / SMOKE_STEPS:.3f}s/step "
        f"-> projected {FULL_STEPS} steps = {elapsed / SMOKE_STEPS * FULL_STEPS / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
