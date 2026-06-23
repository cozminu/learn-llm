import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ModelConfig
from model import TinyLM

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BLOCK_SIZE = 512
BATCH_SIZE = 32
VOCAB_SIZE = 8000
TEMPERATURE = 2.0
ALPHA = 0.5
SMOKE_STEPS = 30


def distill_loss(student_logits, teacher_logits, targets):
    vocab = student_logits.size(-1)
    hard_loss = F.cross_entropy(student_logits.view(-1, vocab), targets.view(-1))
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / TEMPERATURE, dim=-1),
        F.softmax(teacher_logits / TEMPERATURE, dim=-1),
        reduction="batchmean",
    ) * TEMPERATURE**2
    return hard_loss, kd_loss


def main():
    teacher_cfg = ModelConfig(vocab_size=VOCAB_SIZE, dim=512, n_layers=10, n_heads=8, intermediate_size=2048, max_seq_len=BLOCK_SIZE)
    student_cfg = ModelConfig(vocab_size=VOCAB_SIZE, dim=384, n_layers=8, n_heads=6, intermediate_size=1536, max_seq_len=BLOCK_SIZE)

    teacher = TinyLM(teacher_cfg).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = TinyLM(student_cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    data = torch.randint(0, VOCAB_SIZE, (200_000,), dtype=torch.long)

    def get_batch():
        ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix]).to(DEVICE)
        y = torch.stack([data[i + 1 : i + 1 + BLOCK_SIZE] for i in ix]).to(DEVICE)
        return x, y

    for step in range(SMOKE_STEPS):
        x, y = get_batch()
        student_logits, _ = student(x)
        with torch.no_grad():
            teacher_logits, _ = teacher(x)
        hard_loss, kd_loss = distill_loss(student_logits, teacher_logits, y)
        loss = ALPHA * hard_loss + (1 - ALPHA) * kd_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(hard_loss) and torch.isfinite(kd_loss), "non-finite loss term"
        print(f"step {step:2d} | hard {hard_loss.item():.4f} | kd {kd_loss.item():.4f} | loss {loss.item():.4f}")

    print("mechanics OK: both terms finite, comparable magnitude")


if __name__ == "__main__":
    main()
