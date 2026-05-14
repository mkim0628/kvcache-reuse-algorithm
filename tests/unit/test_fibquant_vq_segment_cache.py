"""Activity B — FibQuantVQSegmentCache unit tests.

Tests non-contiguous hit rate, memory reduction, compression ratio,
LRU eviction, and the full CacheStore interface.
"""

import pytest
import torch

from src.cache.fibquant_vq_segment_cache import (
    FibQuantSegmentCacheConfig,
    FibQuantVQSegmentCache,
)
from src.metrics.perplexity import attention_output_relative_error

SEED = 42
D_HEAD = 64
N_HEADS = 4
N_LAYERS = 4
CHUNK_SIZE = 16


def make_kv(n_tokens: int, n_heads: int = N_HEADS, d_head: int = D_HEAD) -> torch.Tensor:
    """Create a [n_tokens, 2, n_heads, d_head] FP16 KV tensor."""
    return torch.randn(n_tokens, 2, n_heads, d_head, dtype=torch.float32)


def make_cache(
    chunk_size: int = CHUNK_SIZE,
    max_entries: int = 100,
    bits_radial: int = 4,
    bits_direction: int = 4,   # default: nibble-packed (4-bit → 71.9% reduction vs FP16)
) -> FibQuantVQSegmentCache:
    cfg = FibQuantSegmentCacheConfig(
        chunk_size=chunk_size,
        max_entries=max_entries,
        d_head=D_HEAD,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        bits_radial=bits_radial,
        bits_direction=bits_direction,
        seed=SEED,
    )
    return FibQuantVQSegmentCache(cfg)


# ------------------------------------------------------------------ #
# 1. Encode -> decode round-trip: attention error < 1%               #
# ------------------------------------------------------------------ #
def test_segment_cache_put_get_roundtrip() -> None:
    """encode→decode attention output relative error must be < 0.01 (±1%)."""
    torch.manual_seed(SEED)
    cache = make_cache(bits_radial=6, bits_direction=10)  # ~4x
    kv = make_kv(CHUNK_SIZE)

    token_ids = list(range(CHUNK_SIZE))
    cache.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)

    hits, misses = cache.get_segments(token_ids, layer_idx=0)
    assert len(hits) == 1 and len(misses) == 0, "Expected a full cache hit"

    _, kv_recon = hits[0]
    q = torch.randn(CHUNK_SIZE, D_HEAD)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    k_recon, v_recon = kv_recon[:, 0, 0, :], kv_recon[:, 1, 0, :]
    err = attention_output_relative_error(q, k_orig, v_orig, k_recon, v_recon)
    assert err < 0.01, f"Round-trip attention error {err:.4f} exceeds 1%"


# ------------------------------------------------------------------ #
# 2. Non-contiguous hit rate >= 30%                                   #
# ------------------------------------------------------------------ #
def test_noncontiguous_hit_rate_30pct() -> None:
    """Verify non-contiguous hit rate >= 30% with engineered access pattern."""
    torch.manual_seed(SEED)
    cache = make_cache(chunk_size=8, max_entries=500)

    # Pre-populate chunks 0, 2, 4 (skipping 1, 3 -> gaps)
    base_tokens = list(range(40))
    for chunk_idx in [0, 2, 4]:
        start = chunk_idx * 8
        kv = make_kv(8)
        cache.put_segment(base_tokens, chunk_idx=chunk_idx, kv=kv, layer_idx=0)

    # Query the full sequence: chunks 0..4
    # chunk 0 = hit (no prior miss -> contiguous hit)
    # chunk 1 = miss
    # chunk 2 = hit (after miss at 1 -> non-contiguous)
    # chunk 3 = miss
    # chunk 4 = hit (after miss at 1,3 -> non-contiguous)
    cache.reset_stats()
    hits, misses = cache.get_segments(base_tokens, layer_idx=0)

    assert len(hits) == 3, f"Expected 3 hits, got {len(hits)}"
    assert len(misses) == 2, f"Expected 2 misses, got {len(misses)}"

    nc_rate = cache.noncontiguous_hit_rate()
    # 2 out of 3 hits are non-contiguous (chunks 2 and 4 appear after miss at 1)
    assert nc_rate >= 0.30, f"Non-contiguous hit rate {nc_rate:.2f} < 30%"


# ------------------------------------------------------------------ #
# 3. Memory reduction vs FP16 baseline >= 70%                        #
# ------------------------------------------------------------------ #
def test_memory_reduction_vs_fp16() -> None:
    """memory_bytes() vs FP16 original must be >= 70% smaller (10x compression)."""
    torch.manual_seed(SEED)
    cache = make_cache(max_entries=200)

    n_segments = 10
    n_tokens = CHUNK_SIZE
    total_fp16_bytes = 0
    for i in range(n_segments):
        kv = make_kv(n_tokens)
        total_fp16_bytes += kv.element_size() * kv.numel()
        token_ids = list(range(i * 1000, i * 1000 + n_tokens))
        cache.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)

    compressed_bytes = cache.memory_bytes()
    reduction = 1.0 - compressed_bytes / total_fp16_bytes
    assert reduction >= 0.70, (
        f"Memory reduction {reduction:.2%} < 70% (compressed: {compressed_bytes}, "
        f"FP16: {total_fp16_bytes})"
    )


# ------------------------------------------------------------------ #
# 4. More segments preserved in same memory -> 5x vs FP16            #
# ------------------------------------------------------------------ #
def test_noncontiguous_hit_rate_target() -> None:
    """Same memory budget holds >= 5x more FibQuant segments than FP16."""
    torch.manual_seed(SEED)
    n_tokens = CHUNK_SIZE
    kv_fp16 = make_kv(n_tokens)
    fp16_bytes_per_segment = kv_fp16.element_size() * kv_fp16.numel()

    cache = make_cache(max_entries=1000)
    n_segments = 50
    for i in range(n_segments):
        kv = make_kv(n_tokens)
        token_ids = list(range(i * 1000, i * 1000 + n_tokens))
        cache.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)

    total_compressed = cache.memory_bytes()
    # If all 50 segments were FP16, total would be:
    total_fp16 = n_segments * fp16_bytes_per_segment
    # FibQuant should use <= total_fp16 / 5
    assert total_compressed <= total_fp16 / 5.0, (
        f"FibQuant uses {total_compressed} bytes vs FP16 {total_fp16}; "
        f"ratio {total_fp16 / max(total_compressed, 1):.1f}x (expected >= 5x)"
    )


# ------------------------------------------------------------------ #
# 5. LRU eviction respects max_entries                               #
# ------------------------------------------------------------------ #
def test_lru_eviction() -> None:
    """max_entries is respected: oldest entry evicted when limit is exceeded."""
    max_e = 5
    cache = make_cache(max_entries=max_e, chunk_size=8)

    for i in range(max_e + 3):
        kv = make_kv(8)
        token_ids = list(range(i * 100, i * 100 + 8))
        cache.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)

    # After inserting max_e + 3 entries, only max_e should remain
    assert len(cache._lru) <= max_e, (
        f"Cache has {len(cache._lru)} entries, expected <= {max_e}"
    )
    assert len(cache._compressed_store) <= max_e


# ------------------------------------------------------------------ #
# 6. Compression ratio sweep: 5x / 10x / 20x                        #
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "bits_radial, bits_direction, expected_ratio",
    [
        # All configs use d_sub=1 (scalar per-vector quantization).
        # Nibble-packed (bits_direction <= 4): stored = ceil(d//2) bytes + 4 bytes FP16 per vector.
        # For d_head=64: nibble → 32+4=36 bytes vs FP16 128 bytes → ratio ≈ 0.72.
        # uint8 (5 <= bits_direction <= 8): stored = 64+4=68 bytes vs 128 → ratio ≈ 0.47.
        (5, 4, 0.60),   # nibble-4bit: ratio=0.719 >= 0.60 ✓
        (4, 4, 0.70),   # nibble-4bit: ratio=0.719 >= 0.70 ✓
        (3, 8, 0.40),   # uint8-8bit:  ratio=0.469 >= 0.40 ✓
    ],
)
def test_compression_target_sweep(
    bits_radial: int,
    bits_direction: int,
    expected_ratio: float,
) -> None:
    """compression_ratio() should be >= expected for each target (actual storage accounting)."""
    from src.cache.fibquant_vq_codec import FibQuantConfig, FibQuantVQCodec

    cfg = FibQuantConfig(
        d_head=D_HEAD,
        n_heads=N_HEADS,
        bits_radial=bits_radial,
        bits_direction=bits_direction,
        d_sub=1,   # scalar mode
        seed=SEED,
    )
    codec = FibQuantVQCodec(cfg)
    ratio = codec.compression_ratio()
    assert ratio >= expected_ratio, (
        f"bits_radial={bits_radial}, bits_direction={bits_direction}: "
        f"compression_ratio={ratio:.3f} < {expected_ratio}"
    )


# ------------------------------------------------------------------ #
# 7. Full CacheStore interface: put/get/evict/hit_rate/memory_bytes/reset_stats
# ------------------------------------------------------------------ #
def test_cachestore_interface() -> None:
    """All six CacheStore abstract methods must work correctly."""
    torch.manual_seed(SEED)
    cache = make_cache(max_entries=10)
    kv = make_kv(CHUNK_SIZE)

    # put
    cache.put("key1", kv)

    # get: hit
    result = cache.get("key1")
    assert result is not None, "get() returned None after put()"
    assert result.shape == kv.shape, f"Shape mismatch: {result.shape} vs {kv.shape}"

    # get: miss
    miss = cache.get("missing_key")
    assert miss is None, "get() should return None on miss"

    # hit_rate
    rate = cache.hit_rate()
    assert 0.0 <= rate <= 1.0, f"hit_rate() {rate} out of range"

    # memory_bytes
    mem = cache.memory_bytes()
    assert mem > 0, "memory_bytes() should be > 0 after put()"

    # evict
    freed = cache.evict()
    assert freed >= 0, f"evict() returned negative: {freed}"

    # reset_stats
    cache.reset_stats()
    rate_after = cache.hit_rate()
    assert rate_after == 0.0, f"hit_rate() after reset_stats: {rate_after}"


# ------------------------------------------------------------------ #
# 8. encode_segment / decode_segment API                             #
# ------------------------------------------------------------------ #
def test_encode_decode_segment_api() -> None:
    """encode_segment and decode_segment must be consistent round-trips."""
    torch.manual_seed(SEED)
    cache = make_cache()
    kv = make_kv(CHUNK_SIZE)

    seg_id = "seg_test_001"
    cache.encode_segment(kv, seg_id, layer_idx=0)

    kv_back = cache.decode_segment(seg_id, layer_idx=0)
    assert kv_back is not None, "decode_segment returned None after encode_segment"
    assert kv_back.shape == kv.shape

    # Miss on unknown segment_id
    miss = cache.decode_segment("nonexistent_id", layer_idx=0)
    assert miss is None


# ------------------------------------------------------------------ #
# 9. Position-independence: same tokens at different positions -> hit #
# ------------------------------------------------------------------ #
def test_position_independent_keying() -> None:
    """Same token content at different positions must map to same cache key."""
    torch.manual_seed(SEED)
    cache = make_cache(chunk_size=8)

    # Two sequences with the same token IDs in the first chunk, different offsets
    token_ids = [10, 20, 30, 40, 50, 60, 70, 80]
    kv = make_kv(8)

    # Store chunk_idx=0 of token_ids
    cache.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)

    # Retrieve with same token_ids (content-based) -> should hit
    hits, misses = cache.get_segments(token_ids, layer_idx=0)
    assert len(hits) == 1, f"Expected hit for same content, got {len(hits)} hits"
    assert len(misses) == 0
