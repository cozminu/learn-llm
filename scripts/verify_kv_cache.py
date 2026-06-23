import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import TinyLM
from tokenizer_bpe import BPETokenizer

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


def sync_device():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def greedy_no_cache(model, ids, max_new_tokens):
    for _ in range(max_new_tokens):
        block = ids[:, -model.cfg.max_seq_len :]
        logits, _ = model(block)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def greedy_cached(model, ids, max_new_tokens):
    kv_cache = model.new_kv_cache()
    logits, _ = model(ids, kv_cache=kv_cache)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out = torch.cat([ids, next_id], dim=1)
    for _ in range(max_new_tokens - 1):
        logits, _ = model(next_id, kv_cache=kv_cache)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
    return out


@torch.no_grad()
def check_logits_match(model, ids):
    """Greedy argmax-matching can stay identical even if per-token logits are
    wrong (e.g. a RoPE position bug that only corrupts decode, not prefill,
    can still argmax to the same path). Compare raw logits at every position
    between a single no-cache forward pass and an equivalent chunked
    prefill+decode-with-cache run -- this catches what argmax-matching can't."""
    full_logits, _ = model(ids)

    split = ids.size(1) // 2
    kv_cache = model.new_kv_cache()
    chunk_logits, _ = model(ids[:, :split], kv_cache=kv_cache)
    cached_logits = [chunk_logits[:, -1, :]]
    for i in range(split, ids.size(1)):
        step_logits, _ = model(ids[:, i : i + 1], kv_cache=kv_cache)
        cached_logits.append(step_logits[:, -1, :])

    ok = True
    for pos in range(split - 1, ids.size(1)):
        match = torch.allclose(full_logits[:, pos, :], cached_logits[pos - split + 1], atol=1e-5)
        if not match:
            print(f"  logits mismatch at position {pos}")
            ok = False
    return ok


def main():
    ckpt = torch.load(CKPT_DIR / "stage5.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")

    model = TinyLM(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = "Once upon a time"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    max_new_tokens = 200

    check_seq = greedy_no_cache(model, ids, 60)
    logits_match = check_logits_match(model, check_seq)
    print(f"Per-position logits match (chunked-cached vs no-cache, {check_seq.size(1)} tokens): {logits_match}")
    if not logits_match:
        sys.exit(1)

    out_nocache = greedy_no_cache(model, ids, max_new_tokens)
    out_cached = greedy_cached(model, ids, max_new_tokens)

    identical = torch.equal(out_nocache, out_cached)
    print(f"Token-identical (greedy decode): {identical}")
    if not identical:
        print("no-cache:", tokenizer.decode(out_nocache[0].tolist()))
        print("cached:  ", tokenizer.decode(out_cached[0].tolist()))
        sys.exit(1)

    sync_device()
    t0 = time.perf_counter()
    greedy_no_cache(model, ids, max_new_tokens)
    sync_device()
    t_nocache = time.perf_counter() - t0

    sync_device()
    t0 = time.perf_counter()
    greedy_cached(model, ids, max_new_tokens)
    sync_device()
    t_cached = time.perf_counter() - t0

    print(f"No-cache: {t_nocache:.3f}s ({max_new_tokens / t_nocache:.1f} tok/s)")
    print(f"Cached:   {t_cached:.3f}s ({max_new_tokens / t_cached:.1f} tok/s)")
    print(f"Speedup:  {t_nocache / t_cached:.2f}x")


if __name__ == "__main__":
    main()
