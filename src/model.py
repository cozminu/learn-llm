import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class KVCache:
    """Per-layer key/value cache for incremental decoding. Training/sampling-only --
    never touches the export path, which always calls forward with cache=None."""

    def __init__(self, n_layers: int):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def seq_len(self, layer_idx: int = 0) -> int:
        return 0 if self.k[layer_idx] is None else self.k[layer_idx].size(2)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=2)
        return self.k[layer_idx], self.v[layer_idx]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        pos_offset = kv_cache.seq_len(layer_idx) if kv_cache is not None else 0
        assert pos_offset + t <= cos.size(0), (
            "kv_cache position exceeds max_seq_len -- cached generation has no "
            "sliding-window support, unlike the no-cache path"
        )
        q, k = apply_rope(q, k, cos[pos_offset : pos_offset + t], sin[pos_offset : pos_offset + t])
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
        # q_len == k_len means there's no cached prefix to attend to beyond the
        # current causal block (prefill, or no cache at all) -> mask the future.
        # q_len < k_len is the cached decode step: the single new query legitimately
        # attends to every past key, so no masking is needed at all.
        is_causal = k.size(2) == t
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaCore(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, kv_cache: KVCache = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, cos, sin, kv_cache, layer_idx)
        return self.norm(x)


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = LlamaCore(cfg)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, kv_cache: KVCache = None):
        hidden = self.model(input_ids, kv_cache)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def new_kv_cache(self) -> KVCache:
        return KVCache(self.cfg.n_layers)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
