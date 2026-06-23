import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model import TinyLM
from tokenizer_bpe import BPETokenizer

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


@torch.no_grad()
def generate(model: TinyLM, ids: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int):
    for _ in range(max_new_tokens):
        block = ids[:, -model.cfg.max_seq_len :]
        logits, _ = model(block)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def generate_cached(model: TinyLM, ids: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int):
    kv_cache = model.new_kv_cache()
    logits, _ = model(ids, kv_cache=kv_cache)
    next_logits = logits[:, -1, :]
    for _ in range(max_new_tokens):
        scaled = next_logits / temperature
        if top_k is not None:
            v, _ = torch.topk(scaled, top_k)
            scaled[scaled < v[:, [-1]]] = float("-inf")
        probs = F.softmax(scaled, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        logits, _ = model(next_id, kv_cache=kv_cache)
        next_logits = logits[:, -1, :]
    return ids


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "\n"
    ckpt = torch.load(CKPT_DIR / "stage5.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")

    model = TinyLM(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    out = generate(model, ids, max_new_tokens=400, temperature=0.8, top_k=40)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
