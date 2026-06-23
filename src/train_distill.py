import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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

# Distillation temperature and hard/soft loss weight -- standard Hinton et al.
# starting point, no prior tuning signal to deviate from.
TEMPERATURE = 2.0
ALPHA = 0.5

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_DATA_PATH = DATA_DIR / "tinystories_train.txt"
VAL_DATA_PATH = DATA_DIR / "tinystories_val.txt"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
TOKENIZER_PATH = CKPT_DIR / "tokenizer.json"
TEACHER_CKPT_PATH = CKPT_DIR / "stage9_teacher.pt"


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


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor):
    vocab = student_logits.size(-1)
    hard_loss = F.cross_entropy(student_logits.view(-1, vocab), targets.view(-1))
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / TEMPERATURE, dim=-1),
        F.softmax(teacher_logits / TEMPERATURE, dim=-1),
        reduction="batchmean",
    ) * TEMPERATURE**2
    loss = ALPHA * hard_loss + (1 - ALPHA) * kd_loss
    return loss, hard_loss, kd_loss


@torch.no_grad()
def estimate_loss(student: TinyLM, teacher: TinyLM, train_data, val_data):
    student.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_STEPS)
        for i in range(EVAL_STEPS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            student_logits, _ = student(x)
            teacher_logits, _ = teacher(x)
            loss, _, _ = distill_loss(student_logits, teacher_logits, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    student.train()
    return out


def main():
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    train_data = torch.tensor(tokenizer.encode_large(TRAIN_DATA_PATH.read_text()), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode_large(VAL_DATA_PATH.read_text()), dtype=torch.long)

    teacher_ckpt = torch.load(TEACHER_CKPT_PATH, map_location=DEVICE, weights_only=False)
    teacher_cfg: ModelConfig = teacher_ckpt["config"]
    teacher = TinyLM(teacher_cfg).to(DEVICE)
    teacher.load_state_dict(teacher_ckpt["model"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"teacher params={teacher.num_params():,} (best_val={teacher_ckpt['best_val']:.4f})", flush=True)

    student_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=384,
        n_layers=8,
        n_heads=6,
        intermediate_size=1536,
        max_seq_len=BLOCK_SIZE,
    )
    student = TinyLM(student_cfg).to(DEVICE)
    print(f"device={DEVICE} student params={student.num_params():,} vocab={tokenizer.vocab_size}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.01)

    ckpt_path = CKPT_DIR / "stage9_distilled.pt"
    start_step = 1
    best_val = float("inf")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt["model"], strict=True)
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
        student_logits, _ = student(x)
        with torch.no_grad():
            teacher_logits, _ = teacher(x)
        loss, hard_loss, kd_loss = distill_loss(student_logits, teacher_logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            losses = estimate_loss(student, teacher, train_data, val_data)
            elapsed = time.time() - t0
            improved = losses["val"] < best_val
            if improved:
                best_val = losses["val"]
                torch.save({
                    "model": student.state_dict(),
                    "config": student_cfg,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val": best_val,
                }, ckpt_path)
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| hard {hard_loss.item():.4f} | kd {kd_loss.item():.4f} "
                f"| lr {lr_at(step):.5f} | {elapsed:.1f}s" + (" | saved" if improved else ""),
                flush=True,
            )

    print(f"best val {best_val:.4f}, checkpoint at {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
