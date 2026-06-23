from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    intermediate_size: int = 512
    max_seq_len: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
