# Stage 7 -- KV-cache-aware generation

Goal: add an optional KV cache so incremental decoding does O(n) work per
step instead of recomputing the full context (O(n^2) total), without
touching the non-cached `forward` path the export pipeline depends on.

## What changed
- `src/model.py`: new `KVCache` class -- a plain list of per-layer
  `(k, v)` tensors, `update(layer_idx, k, v)` concatenates along the
  sequence dim, `seq_len(layer_idx)` reports that layer's cached length.
- `Attention.forward`, `DecoderLayer.forward`, `LlamaCore.forward`,
  `TinyLM.forward` all gained an optional `kv_cache` (and `layer_idx`)
  parameter, defaulting to `None`. With `kv_cache=None` every call is
  byte-for-byte the original code path -- this is what keeps Stage 2/6's
  bit-exactness check valid without rerunning it.
- RoPE position offset: `cos[pos_offset : pos_offset + t]` where
  `pos_offset = kv_cache.seq_len(layer_idx)` -- the cache tracks *absolute*
  position, unlike the no-cache path's sliding window which re-bases
  RoPE to 0 every call.
- The decode-step trap from the plan: `is_causal = (k.size(2) == t)`.
  When there's no cache (or it's still empty), `k_len == t` and the block
  needs the standard causal mask. Once a single new token is queried
  against a longer cached `k`/`v` (`k_len > t == 1`), `is_causal=False`
  is correct -- a lone new query has no future among the existing keys to
  mask against.
- `src/sample.py`: added `generate_cached`, prefills once with the full
  prompt building the cache, then feeds one new token at a time.
- Added `Attention.forward` assert: `pos_offset + t <= cos.size(0)`.
  Cached generation has no sliding-window fallback (unlike the no-cache
  path), so running past `max_seq_len` must fail loudly, not silently
  index past the precomputed RoPE buffer.

## A real bug caught along the way
First implementation had `KVCache.seq_len()` always read `self.k[0]`
(layer 0's cache) regardless of which layer asked. Within a single
forward call, layer 0 updates its cache before layer 1 queries
`seq_len()`, so every layer past the first saw the *post-update* length
instead of its own pre-update length -- positions for layers 1+ were
silently shifted.

This didn't show up as a crash, and didn't even show up in
`scripts/verify_kv_cache.py`'s original greedy-argmax check on the real
model: RoPE's defining property is that a *uniform* shift to both q and
k cancels in the attention dot product (same relative offsets), and
during prefill all positions in that one call get the same wrong shift
-- so prefill output is byte-identical despite being "wrong." The bug
only desynchronizes once prefill and a later decode step apply
*different* shifts, corrupting the relative offset between cached
prefill keys and new decode queries. Even then, greedy argmax on a
TinyStories continuation stayed on the same token path anyway, so the
weak test passed both with and without the bug -- a false confidence
signal caught only by checking raw logits, not just final tokens.

Fixed by parameterizing `seq_len(layer_idx)` to read each layer's own
cache slot. Confirmed the fix mattered (not cosmetic) by reverting to
the buggy `seq_len` and rerunning the logits-comparison check: prefill
position matched (as predicted), but every decode position after it
mismatched.

## Verification (`scripts/verify_kv_cache.py`)
1. **Per-position logits match**: single no-cache forward pass over a
   64-token sequence vs. a chunked prefill (half) + one-token-at-a-time
   cached decode (other half), compared with `torch.allclose(atol=1e-5)`
   at every position from the prefill/decode boundary onward. This is
   the test that actually caught the bug above -- argmax-matching alone
   did not.
2. **Token-identical greedy decode**: 200-token greedy continuation,
   cached vs non-cached, on `stage5.pt` -- exact match.
3. **Speedup**: ~2.2-2.3x at 200 generated tokens (8 layers, dim 384,
   max_seq_len 512). Expected to grow with sequence length since the
   non-cached path's recompute cost is O(n^2) while the cached path
   stays O(n) per step; this run's modest multiplier reflects the short
   length, not a ceiling.

## Result
Stage 7 complete: optional KV cache added without touching the
non-cached path (export bit-exactness unaffected), with the
decode-step `is_causal=False` case and the absolute-vs-relative RoPE
position handling verified at the logit level, not just final-token
level.
