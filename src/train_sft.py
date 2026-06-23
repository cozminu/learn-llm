import math
import random
import time
from pathlib import Path

import torch

from config import ModelConfig
from model import TinyLM
from sft_data import collate_batch, load_dataset
from tokenizer_bpe import BPETokenizer

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
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

    ckpt_path = CKPT_DIR / "stage8.pt"
    start_step = 1
    best_val = float("inf")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg: ModelConfig = ckpt["config"]
        model = TinyLM(cfg).to(DEVICE)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        if "step" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            best_val = ckpt["best_val"]
            print(f"resumed from {ckpt_path} at step {ckpt['step']}")
        else:
            best_val = estimate_loss(model, train_examples, val_examples, eos_id)["val"]
            print(f"resumed weights only from legacy {ckpt_path}, restarting schedule (val={best_val:.4f})")
    else:
        base_ckpt = torch.load(BASE_CKPT_PATH, map_location="cpu", weights_only=False)
        cfg = base_ckpt["config"]
        model = TinyLM(cfg).to(DEVICE)
        model.load_state_dict(base_ckpt["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print(f"device={DEVICE} params={model.num_params():,} vocab={tokenizer.vocab_size}")

    if start_step > MAX_STEPS:
        print(f"already trained to step {MAX_STEPS}, nothing to do")
        return

    t0 = time.time()
    for step in range(start_step, MAX_STEPS + 1):
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
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val": best_val,
                }, ckpt_path)
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr_at(step):.6f} | {elapsed:.1f}s" + (" | saved" if improved else "")
            )

    print(f"best val {best_val:.4f}, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
