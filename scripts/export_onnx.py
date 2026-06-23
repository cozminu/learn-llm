import sys
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

ROOT = Path(__file__).parent.parent
CKPT_DIR = ROOT / "checkpoints"


class NoCacheWrapper(torch.nn.Module):
    """torch.onnx's tracer can't handle the HF DynamicCache object that
    LlamaForCausalLM.forward returns by default -- disable the KV cache and
    unwrap the dict output down to the plain logits tensor it expects."""

    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, use_cache=False, return_dict=False)[0]


def export(hf_dir_name: str, out_name: str) -> Path:
    hf_dir = CKPT_DIR / hf_dir_name
    # sdpa keeps attention as one fused op in the graph instead of unrolling
    # it into a dozen+ matmul/transpose/softmax nodes per layer.
    model = LlamaForCausalLM.from_pretrained(hf_dir, attn_implementation="sdpa")
    model.eval()
    wrapped = NoCacheWrapper(model)

    dummy_input_ids = torch.randint(0, model.config.vocab_size, (1, 8), dtype=torch.long)

    out_path = CKPT_DIR / out_name
    torch.onnx.export(
        wrapped,
        (dummy_input_ids,),
        str(out_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        dynamo=True,
    )
    print(f"exported ONNX graph to {out_path}")
    return out_path


if __name__ == "__main__":
    hf_dir_name = sys.argv[1] if len(sys.argv) > 1 else "stage5-hf"
    out_name = sys.argv[2] if len(sys.argv) > 2 else "stage5.onnx"
    export(hf_dir_name, out_name)
