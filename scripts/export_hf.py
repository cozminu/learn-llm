import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).parent.parent
CKPT_DIR = ROOT / "checkpoints"
EOS_TOKEN = "<|endoftext|>"

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}\nStory: \n"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}"
    "{% endif %}"
    "{% endfor %}"
)


def export(ckpt_name: str, out_dir_name: str, chat_template: str | None = None) -> Path:
    ckpt = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["model"]

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(CKPT_DIR / "tokenizer.json"),
        bos_token=EOS_TOKEN,
        eos_token=EOS_TOKEN,
    )
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    hf_config = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.dim,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_heads,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_word_embeddings,
        hidden_act="silu",
        bos_token_id=eos_id,
        eos_token_id=eos_id,
    )
    hf_config.architectures = ["LlamaForCausalLM"]

    hf_model = LlamaForCausalLM(hf_config)

    our_keys = set(state_dict.keys())
    hf_keys = set(hf_model.state_dict().keys())
    missing = hf_keys - our_keys
    extra = our_keys - hf_keys
    assert not missing and not extra, f"key mismatch missing={missing} extra={extra}"

    hf_model.load_state_dict(state_dict, strict=True)

    out_dir = CKPT_DIR / out_dir_name
    out_dir.mkdir(exist_ok=True)
    hf_model.save_pretrained(out_dir)
    # save_jinja_files defaults to True in transformers>=4.5x, which writes
    # the chat template to a separate chat_template.jinja file instead of
    # inlining it into tokenizer_config.json. We want it inlined, matching
    # what llama.cpp's converter and our own export checks read.
    tokenizer.save_pretrained(out_dir, save_jinja_files=False)

    print(f"exported HF model + tokenizer to {out_dir}")
    return out_dir


if __name__ == "__main__":
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else "stage5.pt"
    out_dir_name = sys.argv[2] if len(sys.argv) > 2 else "stage5-hf"
    export(ckpt_name, out_dir_name)
