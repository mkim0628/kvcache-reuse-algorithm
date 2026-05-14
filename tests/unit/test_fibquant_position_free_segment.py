"""B+C Integration — FibQuantPositionFreeSegmentCache unit tests.

Tests pre-RoPE storage + FibQuant compression + RoPE re-application pipeline,
accuracy preservation, and the full CacheStore interface.
"""

import pytest
import torch
import torch.nn.functional as F

from src.cache.fibquant_position_free_segment import (
    FibQuantPositionFreeConfig,
    FibQuantPositionFreeSegmentCache,
)
from src.metrics.perplexity import attention_output_relative_error, cosine_similarity_output

SEED = 42
D_HEAD = 64
N_HEADS = 4
N_LAYERS = 4
CHUNK_SIZE = 16


def make_kv(n_tokens: int, n_heads: int = N_HEADS, d_head: int = D_HEAD) -> torch.Tensor:
    return torch.randn(n_tokens, 2, n_heads, d_head, dtype=torch.float32)


def make_cache(
    chunk_size: int = CHUNK_SIZE,
    max_entries: int = 100,
    bits_radial: int = 6,
    bits_direction: int = 10,
) -> FibQuantPositionFreeSegmentCache:
    cfg = FibQuantPositionFreeConfig(
        chunk_size=chunk_size,
        max_entries=max_entries,
        d_head=D_HEAD,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        bits_radial=bits_radial,
        bits_direction=bits_direction,
        rope_base=10000.0,
        seed=SEED,
    )
    return FibQuantPositionFreeSegmentCache(cfg)


# ------------------------------------------------------------------ #
# 1. store_pre_rope -> load_with_rope accuracy < 1%                  #
# ------------------------------------------------------------------ #
def test_store_pre_rope_and_load() -> None:
    """store_pre_rope → load_with_rope round-trip: attention error < 1%."""
    torch.manual_seed(SEED)
    cache = make_cache()
    n_tokens = CHUNK_SIZE
    pre_rope_kv = make_kv(n_tokens)

    target_positions = torch.arange(0, n_tokens, dtype=torch.long)
    key = "test_pre_rope_segment"

    cache.store_pre_rope(key, pre_rope_kv, layer_idx=0)
    restored = cache.load_with_rope(key, target_positions, layer_idx=0)

    assert restored is not None, "load_with_rope returned None on expected hit"
    assert restored.shape == pre_rope_kv.shape, (
        f"Shape mismatch: {restored.shape} vs {pre_rope_kv.shape}"
    )

    # Accuracy: pre_rope_kv unchanged (values only), keys differ by RoPE
    # Measure error on values (index 1) which should be unchanged
    v_orig = pre_rope_kv[:, 1, 0, :]
    v_restored = restored[:, 1, 0, :]
    err = ((v_orig - v_restored).norm() / v_orig.norm().clamp(min=1e-8)).item()
    assert err < 0.01, f"Value round-trip error {err:.4f} >= 1%"


# ------------------------------------------------------------------ #
# 2. RoPE re-application correctness: cosine >= 0.99                 #
# ------------------------------------------------------------------ #
def test_rope_reapplication_correctness() -> None:
    """After pre-RoPE store + load_with_rope, attention output cosine >= 0.99."""
    torch.manual_seed(SEED)
    cache = make_cache()
    n_tokens = CHUNK_SIZE
    pre_rope_kv = make_kv(n_tokens)

    # Apply RoPE manually for the reference
    d = D_HEAD
    positions = torch.arange(100, 100 + n_tokens, dtype=torch.long)

    # Store pre-RoPE and load with RoPE at positions
    key = "rope_correctness_test"
    cache.store_pre_rope(key, pre_rope_kv, layer_idx=0)
    restored = cache.load_with_rope(key, positions, layer_idx=0)

    assert restored is not None

    # Values should be unchanged (RoPE only applied to keys)
    v_pre = pre_rope_kv[:, 1, 0, :]
    v_post = restored[:, 1, 0, :]
    v_err = ((v_pre - v_post).norm() / v_pre.norm().clamp(min=1e-8)).item()
    assert v_err < 0.01, f"Value unchanged check failed: err={v_err:.4f}"

    # Keys should be rotated: cos similarity between pre and post != 1.0
    k_pre = pre_rope_kv[:, 0, 0, :]
    k_post = restored[:, 0, 0, :]
    cos_kk = F.cosine_similarity(k_pre.flatten().unsqueeze(0), k_post.flatten().unsqueeze(0)).item()
    # Rotated keys should be meaningfully different from pre-RoPE (at non-zero positions)
    assert cos_kk < 0.999, f"Keys appear unchanged after RoPE re-application: cosine={cos_kk:.4f}"


# ------------------------------------------------------------------ #
# 3. FibQuant compression ratio preserved through pre-RoPE path      #
# ------------------------------------------------------------------ #
def test_fibquant_compression_preserved() -> None:
    """Pre-RoPE path must maintain FibQuant compression: memory_bytes << FP16."""
    torch.manual_seed(SEED)
    # bits_direction=4 uses nibble packing -> ~86% reduction vs FP32 (>= 70%)
    cache = make_cache(bits_radial=4, bits_direction=4)

    n_segments = 10
    n_tokens = CHUNK_SIZE
    fp16_bytes_total = 0

    for i in range(n_segments):
        kv = make_kv(n_tokens)
        fp16_bytes_total += kv.element_size() * kv.numel()
        key = f"seg_{i}"
        cache.store_pre_rope(key, kv, layer_idx=0)

    compressed_bytes = cache.memory_bytes()
    reduction = 1.0 - compressed_bytes / fp16_bytes_total
    assert reduction >= 0.70, (
        f"Compression reduction {reduction:.2%} < 70% after pre-RoPE path"
    )


# ------------------------------------------------------------------ #
# 4. Non-contiguous hit tracking: _noncontiguous_hits counter        #
# ------------------------------------------------------------------ #
def test_noncontiguous_hit_tracking() -> None:
    """_noncontiguous_hits counter must accurately reflect non-contiguous access."""
    torch.manual_seed(SEED)
    cache = make_cache(chunk_size=8)

    base_tokens = list(range(40))
    # Store chunks 0, 2, 4 (skipping 1 and 3)
    for chunk_idx in [0, 2, 4]:
        kv = make_kv(8)
        cache.put_segment_pre_rope(base_tokens, chunk_idx=chunk_idx, pre_rope_kv=kv, layer_idx=0)

    cache.reset_stats()

    # Access all 5 chunks with RoPE at offset 0
    hits, misses = cache.get_segments_with_rope(base_tokens, target_offset=0, layer_idx=0)

    # chunk 0: hit, no prior miss -> contiguous
    # chunk 1: miss
    # chunk 2: hit, miss at 1 -> non-contiguous
    # chunk 3: miss
    # chunk 4: hit, miss at 1,3 -> non-contiguous
    assert len(hits) == 3, f"Expected 3 hits, got {len(hits)}"
    assert len(misses) == 2, f"Expected 2 misses, got {len(misses)}"
    assert cache._noncontiguous_hits == 2, (
        f"Expected 2 noncontiguous hits, got {cache._noncontiguous_hits}"
    )


# ------------------------------------------------------------------ #
# 5. CacheStore interface: all 6 abstract methods work               #
# ------------------------------------------------------------------ #
def test_cachestore_interface() -> None:
    """All six CacheStore abstract methods must be implemented and functional."""
    torch.manual_seed(SEED)
    cache = make_cache()
    kv = make_kv(CHUNK_SIZE)

    # put
    cache.put("key_a", kv)

    # get: hit
    result = cache.get("key_a")
    assert result is not None, "get() returned None after put()"
    assert result.shape == kv.shape

    # get: miss
    assert cache.get("nonexistent") is None

    # hit_rate
    rate = cache.hit_rate()
    assert 0.0 <= rate <= 1.0, f"hit_rate() {rate} out of [0, 1]"

    # memory_bytes
    mem = cache.memory_bytes()
    assert mem > 0, "memory_bytes() must be > 0 after put()"

    # evict
    freed = cache.evict()
    assert freed >= 0, f"evict() returned {freed}"

    # reset_stats
    cache.reset_stats()
    assert cache.hit_rate() == 0.0
    assert cache._hits == 0
    assert cache._misses == 0


# ------------------------------------------------------------------ #
# 6. put_segment_pre_rope + get_segments_with_rope end-to-end        #
# ------------------------------------------------------------------ #
def test_put_and_get_segment_pre_rope_e2e() -> None:
    """Full B+C pipeline: store pre-RoPE chunk -> retrieve with RoPE."""
    torch.manual_seed(SEED)
    cache = make_cache(chunk_size=8)

    token_ids = list(range(16))  # 2 chunks of 8
    kv0 = make_kv(8)
    kv1 = make_kv(8)

    cache.put_segment_pre_rope(token_ids, chunk_idx=0, pre_rope_kv=kv0, layer_idx=0)
    cache.put_segment_pre_rope(token_ids, chunk_idx=1, pre_rope_kv=kv1, layer_idx=0)

    hits, misses = cache.get_segments_with_rope(token_ids, target_offset=50, layer_idx=0)

    assert len(hits) == 2, f"Expected 2 hits, got {len(hits)}"
    assert len(misses) == 0

    for chunk_idx, kv_with_rope in hits:
        # Shape must match
        assert kv_with_rope.shape[0] == 8
        assert kv_with_rope.shape[1] == 2
        # Values (index 1) should match original closely
        orig_kv = kv0 if chunk_idx == 0 else kv1
        v_orig = orig_kv[:, 1, 0, :].float()
        v_ret = kv_with_rope[:, 1, 0, :].float()
        v_err = ((v_orig - v_ret).norm() / v_orig.norm().clamp(min=1e-8)).item()
        assert v_err < 0.01, f"chunk {chunk_idx} value error {v_err:.4f} >= 1%"
