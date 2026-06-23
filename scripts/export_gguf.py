import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CKPT_DIR = ROOT / "checkpoints"

sys.path.insert(0, str(ROOT / "llama.cpp"))
sys.path.insert(0, str(ROOT / "src"))

from export_hf import export  # noqa: E402


def patch_unrecognized_pretokenizer() -> None:
    """Our BPE vocab is custom-trained, so its chkhsh never matches a known
    model's hash -- get_vocab_base_pre() would otherwise raise
    NotImplementedError. We trained with tokenizers' ByteLevel
    pre-tokenizer (use_regex=True), which *is* the GPT-2 regex split, just
    over a different trained vocab, so falling back to the "gpt-2" pre type
    is correct rather than a guess.
    """
    from conversion import base

    original = base.TextModel.get_vocab_base_pre

    def patched(self, tokenizer):
        try:
            return original(self, tokenizer)
        except NotImplementedError:
            return "gpt-2"

    base.TextModel.get_vocab_base_pre = patched


def main(ckpt_name: str, name: str, chat_template: str | None = None) -> None:
    hf_dir = export(ckpt_name, f"{name}-hf", chat_template=chat_template)
    gguf_path = CKPT_DIR / f"{name}.gguf"

    patch_unrecognized_pretokenizer()
    import convert_hf_to_gguf

    sys.argv = ["convert_hf_to_gguf.py", str(hf_dir), "--outfile", str(gguf_path), "--outtype", "f16"]
    convert_hf_to_gguf.main()

    print(f"GGUF ready: {gguf_path}")


if __name__ == "__main__":
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else "stage5.pt"
    name = sys.argv[2] if len(sys.argv) > 2 else "stage5"
    main(ckpt_name, name)
