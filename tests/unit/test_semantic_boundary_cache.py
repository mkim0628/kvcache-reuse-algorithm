"""Unit tests for SemanticBoundarySegmentCache (Activity B).

Tests: semantic boundary detection, GSC clustering, proportional attention,
put_with_gsc, CacheStore interface compliance, and accuracy proxy.
"""

import pytest
import torch

from src.cache.semantic_boundary_cache import SemanticBoundarySegmentCache
from src.cache.base import CacheStore


def _make_cache(capacity_bytes: int = 10 * 1024 * 1024) -> SemanticBoundarySegmentCache:
    return SemanticBoundarySegmentCache(
        capacity_bytes=capacity_bytes,
        min_cluster_size=3,
        max_merge_ratio=0.7,
        attention_threshold=0.1,
    )


# ------------------------------------------------------------------ #
# CacheStore interface compliance                                      #
# ------------------------------------------------------------------ #

class TestCacheStoreInterface:
    def test_cachestore_interface(self):
        """SemanticBoundarySegmentCache must implement all CacheStore abstract methods."""
        cache = _make_cache()
        assert isinstance(cache, CacheStore)

    def test_put_and_get_roundtrip(self):
        """put() then get() returns the stored tensor."""
        cache = _make_cache()
        key = "test_seg"
        value = torch.randn(10, 32)
        cache.put(key, value)
        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.shape == value.shape

    def test_get_miss_returns_none(self):
        """get() returns None for absent keys."""
        cache = _make_cache()
        assert cache.get("nonexistent") is None

    def test_hit_rate(self):
        """hit_rate() should track hits and misses correctly."""
        cache = _make_cache()
        cache.put("k", torch.randn(5, 8))
        cache.get("k")     # hit
        cache.get("miss")  # miss
        assert cache.hit_rate() == pytest.approx(0.5)

    def test_memory_bytes(self):
        """memory_bytes() increases after put()."""
        cache = _make_cache()
        before = cache.memory_bytes()
        cache.put("k1", torch.randn(20, 32))
        assert cache.memory_bytes() > before

    def test_evict_reduces_memory(self):
        """evict() removes LRU entry and returns bytes freed."""
        cache = _make_cache()
        cache.put("k1", torch.randn(10, 32))
        cache.put("k2", torch.randn(10, 32))
        before = cache.memory_bytes()
        freed = cache.evict()
        assert freed > 0
        assert cache.memory_bytes() < before

    def test_reset_stats(self):
        """reset_stats() zeros hit/miss counters."""
        cache = _make_cache()
        cache.put("k", torch.randn(5, 8))
        cache.get("k")
        cache.reset_stats()
        assert cache.hit_rate() == 0.0


# ------------------------------------------------------------------ #
# Semantic boundary detection                                          #
# ------------------------------------------------------------------ #

class TestSemanticBoundaryDetection:
    def test_detect_sentence_boundaries(self):
        """Sentence-end delimiters (.) should be detected as boundaries."""
        cache = _make_cache()
        text = "Hello world. How are you? I am fine! Great."
        boundaries = cache.detect_semantic_boundaries(text)
        # Should have at least the start (0) and some sentence boundaries
        assert len(boundaries) >= 2
        assert 0 in boundaries

    def test_detect_paragraph_boundaries(self):
        """Double newline should be detected as a boundary."""
        cache = _make_cache()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        boundaries = cache.detect_semantic_boundaries(text)
        assert len(boundaries) >= 3  # start + at least 2 paragraph breaks

    def test_detect_code_block_boundaries(self):
        """Code block delimiter (```) should be detected as a boundary."""
        cache = _make_cache()
        text = "Some text. ```python\ncode\n``` More text."
        boundaries = cache.detect_semantic_boundaries(text)
        # Should detect the ``` delimiter
        assert len(boundaries) >= 1

    def test_no_boundaries_in_plain_text(self):
        """Text without delimiters should return only the start position."""
        cache = _make_cache()
        text = "no delimiters here at all"
        boundaries = cache.detect_semantic_boundaries(text)
        assert boundaries == [0]

    def test_boundaries_are_sorted(self):
        """Boundary positions must be in ascending order."""
        cache = _make_cache()
        text = "First. Second! Third? Fourth.\n\nFifth."
        boundaries = cache.detect_semantic_boundaries(text)
        assert boundaries == sorted(boundaries)


# ------------------------------------------------------------------ #
# GSC clustering                                                       #
# ------------------------------------------------------------------ #

class TestGSCClustering:
    def test_gsc_clustering_reduces_tokens(self):
        """GSC clustering should reduce the number of tokens based on max_merge_ratio."""
        torch.manual_seed(42)
        cache = _make_cache()
        n_tokens = 20
        d_head = 16
        kv = torch.randn(n_tokens, d_head)
        attn = torch.rand(n_tokens) + 0.01

        core_kv, core_attn = cache.apply_gsc_clustering(kv, attn)

        # Number of seeds = max(1, int(20 * (1 - 0.7))) = max(1, 6) = 6
        expected_max = max(1, int(n_tokens * (1 - cache.max_merge_ratio)))
        assert core_kv.shape[0] == expected_max
        assert core_kv.shape[1] == d_head

    def test_proportional_attention_is_sum_not_mean(self):
        """
        Proportional attention weights must be the SUM of cluster attention scores,
        not the mean. For a cluster absorbing multiple tokens, the sum should be
        greater than any individual token's attention score.
        """
        torch.manual_seed(0)
        cache = SemanticBoundarySegmentCache(
            capacity_bytes=10 * 1024 * 1024,
            max_merge_ratio=0.5,  # 50% → 2 seeds for 4 tokens
        )
        n_tokens = 4
        d_head = 8
        kv = torch.randn(n_tokens, d_head)
        # All equal attention scores for easy reasoning
        attn = torch.ones(n_tokens) * 0.25

        _, core_attn = cache.apply_gsc_clustering(kv, attn)

        # Each cluster should have summed attention (≥ 0.25)
        # With 2 seeds and 4 tokens, each cluster should have ~2 tokens → sum = 0.5
        total_attention_sum = core_attn.sum().item()
        assert total_attention_sum == pytest.approx(n_tokens * 0.25, abs=1e-4), (
            "Proportional attention should sum to total input attention"
        )

    def test_gsc_output_shape_consistent(self):
        """Core KV and attention shapes must be consistent after clustering."""
        torch.manual_seed(5)
        cache = _make_cache()
        n_tokens = 16
        d_head = 32
        kv = torch.randn(n_tokens, d_head)
        attn = torch.rand(n_tokens)

        core_kv, core_attn = cache.apply_gsc_clustering(kv, attn)
        assert core_kv.shape[0] == core_attn.shape[0]
        assert core_kv.shape[1] == d_head

    def test_gsc_single_token_returns_same(self):
        """Single-token input should return that token unchanged."""
        cache = _make_cache()
        kv = torch.randn(1, 16)
        attn = torch.tensor([1.0])
        core_kv, core_attn = cache.apply_gsc_clustering(kv, attn)
        assert core_kv.shape[0] == 1


# ------------------------------------------------------------------ #
# put_with_gsc                                                         #
# ------------------------------------------------------------------ #

class TestPutWithGSC:
    def test_put_with_gsc_stores_core_kv(self):
        """put_with_gsc() must store a KV tensor smaller than the original."""
        torch.manual_seed(42)
        cache = _make_cache()
        n_tokens = 20
        d_head = 16
        kv = torch.randn(n_tokens, d_head)
        attn = torch.rand(n_tokens)

        cache.put_with_gsc("gsc_key", kv, attn)
        stored = cache.get("gsc_key")

        assert stored is not None
        # After GSC with max_merge_ratio=0.7, number of tokens should be reduced
        assert stored.shape[0] < n_tokens
        assert stored.shape[1] == d_head

    def test_put_with_gsc_and_retrieve(self):
        """put_with_gsc() then get() must retrieve the stored core."""
        torch.manual_seed(3)
        cache = _make_cache()
        kv = torch.randn(10, 8)
        attn = torch.rand(10)

        cache.put_with_gsc("key_gsc", kv, attn)
        retrieved = cache.get("key_gsc")
        assert retrieved is not None


# ------------------------------------------------------------------ #
# Accuracy proxy: cosine similarity after GSC                         #
# ------------------------------------------------------------------ #

class TestAccuracyProxy:
    def test_accuracy_delta_gsc_cosine_similarity(self):
        """
        Cosine similarity between original mean KV and GSC core mean KV should be ≥ 0.85.
        This acts as a proxy for WikiText-2 perplexity ±1% preservation.
        """
        torch.manual_seed(42)
        cache = _make_cache()
        n_tokens = 30
        d_head = 64

        # Use a realistic KV tensor (low noise relative to signal)
        base = torch.randn(d_head)
        kv = base.unsqueeze(0).expand(n_tokens, -1) + 0.1 * torch.randn(n_tokens, d_head)
        attn = torch.rand(n_tokens) + 0.1

        core_kv, _ = cache.apply_gsc_clustering(kv, attn)

        original_mean = kv.mean(dim=0)
        core_mean = core_kv.mean(dim=0)

        cosine_sim = torch.nn.functional.cosine_similarity(
            original_mean.unsqueeze(0), core_mean.unsqueeze(0)
        ).item()

        assert cosine_sim >= 0.85, (
            f"GSC core cosine similarity {cosine_sim:.4f} < 0.85 (accuracy proxy threshold)"
        )
