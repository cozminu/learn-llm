import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "."))

from model import TinyLM
from tokenizer_bpe import BPETokenizer

from gguf import GGUFReader

sys.path.insert(0, str(Path(__file__).parent.parent / "llama.cpp"))
sys.path.insert(0, str(Path(__file__).parent))

from export_gguf import main as export_gguf_main
from export_hf import CHAT_TEMPLATE

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"

FIXED_FIELDS = "Features: Dialogue\n"
SUMMARIES = [
    "Summary: A dog finds a lost ball in the park and brings it home.",
    "Summary: Two sisters bake a cake for their mother's birthday.",
    "Summary: A boy is scared of a thunderstorm until his dad comforts him.",
]


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    kv_cache = model.new_kv_cache()
    logits, _ = model(ids, kv_cache=kv_cache)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out = torch.cat([ids, next_id], dim=1)
    for _ in range(max_new_tokens - 1):
        logits, _ = model(next_id, kv_cache=kv_cache)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
        if next_id.item() == tokenizer.encode("<|endoftext|>")[0]:
            break
    return tokenizer.decode(out[0].tolist())


def check_conditioning(model, tokenizer) -> None:
    print("=== Conditioning check: same Features, varied Summary ===")
    for summary in SUMMARIES:
        prompt = FIXED_FIELDS + summary + "\nStory: \n"
        text = generate(model, tokenizer, prompt)
        print(f"--- {summary}")
        print(text)
        print()


def check_export_and_chat_template() -> None:
    print("=== Export check: stage8.pt -> GGUF with chat_template ===")
    export_gguf_main("stage8.pt", "stage8", chat_template=CHAT_TEMPLATE)

    reader = GGUFReader(str(CKPT_DIR / "stage8.gguf"))
    field = reader.fields.get("tokenizer.chat_template")
    assert field is not None, "tokenizer.chat_template missing from stage8.gguf metadata"
    template_bytes = bytes(field.parts[-1])
    template_str = template_bytes.decode("utf-8")
    assert template_str == CHAT_TEMPLATE, "GGUF chat_template does not match export_hf.CHAT_TEMPLATE"
    print("tokenizer.chat_template present in stage8.gguf and matches export_hf.CHAT_TEMPLATE")

    hf_config_path = CKPT_DIR / "stage8-hf" / "tokenizer_config.json"
    hf_config = json.loads(hf_config_path.read_text())
    assert hf_config.get("chat_template") == CHAT_TEMPLATE
    print("tokenizer_config.json chat_template also confirmed")


def main():
    ckpt = torch.load(CKPT_DIR / "stage8.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    tokenizer = BPETokenizer.load(CKPT_DIR / "tokenizer.json")

    model = TinyLM(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    check_conditioning(model, tokenizer)
    check_export_and_chat_template()


if __name__ == "__main__":
    main()
