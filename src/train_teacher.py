import math
import time
from pathlib import Path

import torch

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
MAX_STEPS = 8000
EVAL_INTERVAL = 200
EVAL_STEPS = 50
LR = 1e-3
WARMUP_STEPS = 300
GRAD_CLIP = 1.0

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_DATA_PATH = DATA_DIR / "tinystories_train.txt"
VAL_DATA_PATH = DATA_DIR / "tinystories_val.txt"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
TOKENIZER_PATH = CKPT_DIR / "tokenizer.json"


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.1 * LR + 0.5 * (LR - 0.1 * LR) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: TinyLM, train_data, val_data):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_STEPS)
        for i in range(EVAL_STEPS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    train_data = torch.tensor(tokenizer.encode_large(TRAIN_DATA_PATH.read_text()), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode_large(VAL_DATA_PATH.read_text()), dtype=torch.long)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=512,
        n_layers=10,
        n_heads=8,
        intermediate_size=2048,
        max_seq_len=BLOCK_SIZE,
    )
    model = TinyLM(cfg).to(DEVICE)
    print(f"device={DEVICE} params={model.num_params():,} vocab={tokenizer.vocab_size}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    ckpt_path = CKPT_DIR / "stage9_teacher.pt"
    start_step = 1
    best_val = float("inf")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt["best_val"]
        print(f"resumed from {ckpt_path} at step {ckpt['step']}")

    if start_step > MAX_STEPS:
        print(f"already trained to step {MAX_STEPS}, nothing to do")
        return

    t0 = time.time()
    for step in range(start_step, MAX_STEPS + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            improved = losses["val"] < best_val
            if improved:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val": best_val,
                }, ckpt_path)
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr_at(step):.5f} | {elapsed:.1f}s" + (" | saved" if improved else ""),
                flush=True,
            )

    print(f"best val {best_val:.4f}, checkpoint at {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
