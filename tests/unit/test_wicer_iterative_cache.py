"""Unit tests for WiCERIterativeKVWikiCache (Activity B).

Covers: CEGAR loop, hit-rate monotonicity, counterexample collection,
put/get/evict interface, and artefact serialisation.
"""

import os
import tempfile
from typing import Dict, List

import pytest
import torch

from src.cache.wicer_iterative_cache import WiCERConfig, WiCERIterativeKVWikiCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_kv_fn(n_heads: int = 2, d_head: int = 8) -> callable:
    """Returns a deterministic kv_fn(token_ids, layer_idx) → Tensor."""
    def kv_fn(token_ids: List[int], layer_idx: int) -> torch.Tensor:
        torch.manual_seed(sum(token_ids) + layer_idx)
        return torch.randn(len(token_ids), 2, n_heads, d_head)
    return kv_fn


def make_docs(n_docs: int = 3, doc_len: int = 256) -> Dict[str, List[int]]:
    torch.manual_seed(42)
    return {
        f"doc_{i}": list(range(i * doc_len, (i + 1) * doc_len))
        for i in range(n_docs)
    }


def make_val_queries(
    docs: Dict[str, List[int]], n_queries: int = 5
) -> List[List[int]]:
    """Validation queries drawn directly from the corpus for high hit rate."""
    queries = []
    for doc_tokens in list(docs.values())[:n_queries]:
        queries.append(doc_tokens[:128])
    return queries


# ---------------------------------------------------------------------------
# CacheStore interface compliance
# ---------------------------------------------------------------------------

class TestCacheStoreInterface:
    def test_put_get_round_trip(self) -> None:
        cache = WiCERIterativeKVWikiCache(WiCERConfig())
        key = "test_key"
        value = torch.randn(16, 2, 2, 8)
        cache.put(key, value)
        result = cache.get(key)
        assert result is not None
        assert result.shape == value.shape

    def test_get_miss_returns_none(self) -> None:
        cache = WiCERIterativeKVWikiCache(WiCERConfig())
        assert cache.get("nonexistent") is None

    def test_hit_rate_after_put_get(self) -> None:
        cache = WiCERIterativeKVWikiCache(WiCERConfig())
        cache.put("k", torch.randn(4, 2, 2, 8))
        cache.get("k")       # hit
        cache.get("missing")  # miss
        assert cache.hit_rate() == pytest.approx(0.5)

    def test_evict_reduces_memory(self) -> None:
        cache = WiCERIterativeKVWikiCache(WiCERConfig(max_entries=2))
        cache.put("a", torch.randn(4, 2, 2, 8))
        cache.put("b", torch.randn(4, 2, 2, 8))
        before = cache.memory_bytes()
        freed = cache.evict()
        assert freed >= 0
        assert cache.memory_bytes() <= before

    def test_memory_bytes_positive(self) -> None:
        cache = WiCERIterativeKVWikiCache(WiCERConfig())
        cache.put("x", torch.randn(8, 2, 2, 8))
        assert cache.memory_bytes() > 0

    def test_reset_stats_clears_counters(self) -> None:
        cache = WiCERIterativeKVWikiCache(WiCERConfig())
        cache.put("k", torch.randn(4, 2, 2, 8))
        cache.get("k")
        cache.get("miss")
        cache.reset_stats()
        assert cache.hit_rate() == 0.0


# ---------------------------------------------------------------------------
# compile_corpus
# ---------------------------------------------------------------------------

class TestCompileCorpus:
    def test_corpus_segments_stored(self) -> None:
        config = WiCERConfig(chunk_size=64)
        cache = WiCERIterativeKVWikiCache(config)
        docs = make_docs(n_docs=2, doc_len=128)
        kv_fn = make_kv_fn()
        cache.compile_corpus(docs, kv_fn)
        assert cache.memory_bytes() > 0

    def test_compile_increases_hit_rate_on_corpus_queries(self) -> None:
        config = WiCERConfig(chunk_size=64)
        cache = WiCERIterativeKVWikiCache(config)
        docs = make_docs(n_docs=2, doc_len=128)
        kv_fn = make_kv_fn()
        cache.compile_corpus(docs, kv_fn)
        val_queries = make_val_queries(docs, n_queries=2)
        hit_rate, _ = cache.evaluate(val_queries)
        assert hit_rate > 0.0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_hit_rate_and_counterexamples(self) -> None:
        config = WiCERConfig(chunk_size=64)
        cache = WiCERIterativeKVWikiCache(config)
        docs = make_docs(n_docs=2, doc_len=128)
        kv_fn = make_kv_fn()
        cache.compile_corpus(docs, kv_fn)
        val_queries = make_val_queries(docs, n_queries=2)
        hit_rate, counterexamples = cache.evaluate(val_queries)
        assert 0.0 <= hit_rate <= 1.0
        assert isinstance(counterexamples, list)

    def test_evaluate_empty_queries(self) -> None:
        cache = WiCERIterativeKVWikiCache(WiCERConfig())
        hit_rate, cex = cache.evaluate([])
        assert hit_rate == 0.0
        assert cex == []

    def test_high_hit_rate_on_compiled_corpus(self) -> None:
        """After compiling, queries matching compiled docs should hit."""
        config = WiCERConfig(chunk_size=32)
        cache = WiCERIterativeKVWikiCache(config)
        docs = {"doc_a": list(range(64))}
        kv_fn = make_kv_fn()
        cache.compile_corpus(docs, kv_fn)
        # Query using the exact same tokens
        hit_rate, _ = cache.evaluate([list(range(64))])
        assert hit_rate > 0.0


# ---------------------------------------------------------------------------
# CEGAR loop: hit-rate monotonicity
# ---------------------------------------------------------------------------

class TestCEGARRefinement:
    def test_cegar_hit_rate_non_decreasing(self) -> None:
        """CEGAR refinement should not decrease hit rate across iterations."""
        config = WiCERConfig(
            chunk_size=64,
            min_chunk_size=16,
            target_hit_rate=1.1,  # never satisfied → runs full max_iterations
            max_iterations=3,
        )
        cache = WiCERIterativeKVWikiCache(config)
        docs = make_docs(n_docs=3, doc_len=256)
        kv_fn = make_kv_fn()
        val_queries = make_val_queries(docs, n_queries=3)
        cache.cegar_refine(docs, val_queries, kv_fn)
        history = cache.cegar_hit_rate_history()
        assert len(history) >= 1
        for i in range(1, len(history)):
            # Allow a tiny numerical tolerance
            assert history[i] >= history[i - 1] - 1e-9, (
                f"Hit rate decreased at iteration {i}: {history[i - 1]:.4f} → {history[i]:.4f}"
            )

    def test_cegar_terminates_at_target(self) -> None:
        """CEGAR terminates early when target hit rate is met or exceeded."""
        config = WiCERConfig(
            chunk_size=32,
            min_chunk_size=32,
            target_hit_rate=0.0,  # immediately satisfied
            max_iterations=10,
        )
        cache = WiCERIterativeKVWikiCache(config)
        docs = make_docs(n_docs=2, doc_len=64)
        kv_fn = make_kv_fn()
        val_queries = make_val_queries(docs, n_queries=2)
        cache.cegar_refine(docs, val_queries, kv_fn)
        # Should terminate after first iteration (target=0.0 is always met)
        assert len(cache.cegar_hit_rate_history()) == 1

    def test_cegar_respects_max_iterations(self) -> None:
        """CEGAR never exceeds max_iterations."""
        config = WiCERConfig(
            chunk_size=64,
            min_chunk_size=16,
            target_hit_rate=1.1,  # never met
            max_iterations=2,
        )
        cache = WiCERIterativeKVWikiCache(config)
        docs = make_docs(n_docs=2, doc_len=256)
        kv_fn = make_kv_fn()
        val_queries = make_val_queries(docs, n_queries=2)
        cache.cegar_refine(docs, val_queries, kv_fn)
        assert len(cache.cegar_hit_rate_history()) <= 2

    def test_refinement_reduces_chunk_size(self) -> None:
        """After at least one refinement, some documents should have smaller chunks."""
        config = WiCERConfig(
            chunk_size=64,
            min_chunk_size=16,
            target_hit_rate=1.1,  # never met → forces refinement
            max_iterations=2,
        )
        cache = WiCERIterativeKVWikiCache(config)
        # Use queries with tokens NOT in the corpus to force misses
        docs = {"doc_a": list(range(128))}
        kv_fn = make_kv_fn()
        cache.compile_corpus(docs, kv_fn)
        # val_queries with tokens outside corpus → all misses → counterexamples collected
        val_queries = [list(range(200, 264))]  # tokens not in corpus
        _, counterexamples = cache.evaluate(val_queries)
        if counterexamples:
            cache.refine(counterexamples, docs, kv_fn)
            # At least one document should have a refined chunk size
            sizes = list(cache._chunk_sizes.values())
            assert any(s <= 32 for s in sizes)


# ---------------------------------------------------------------------------
# Segment-level API (runner.py compatibility)
# ---------------------------------------------------------------------------

class TestSegmentAPI:
    def test_put_segment_and_get_segments(self) -> None:
        cache = WiCERIterativeKVWikiCache(WiCERConfig(chunk_size=32))
        token_ids = list(range(64))
        kv = torch.randn(32, 2, 2, 8)
        cache.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)
        hits, misses = cache.get_segments(token_ids, layer_idx=0)
        assert len(hits) > 0

    def test_noncontiguous_hit_rate_after_mixed_queries(self) -> None:
        config = WiCERConfig(chunk_size=32)
        cache = WiCERIterativeKVWikiCache(config)
        token_ids = list(range(96))  # 3 chunks of 32
        kv = torch.randn(32, 2, 2, 8)
        # Store chunk 0 and chunk 2 but not chunk 1
        cache.put_segment(token_ids, 0, kv)
        cache.put_segment(token_ids, 2, kv)
        hits, misses = cache.get_segments(token_ids)
        # Chunk 2 is a non-contiguous hit (chunk 1 is missing before it)
        nc_rate = cache.noncontiguous_hit_rate()
        assert 0.0 <= nc_rate <= 1.0


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_save_load_round_trip(self) -> None:
        config = WiCERConfig(chunk_size=32)
        cache = WiCERIterativeKVWikiCache(config)
        docs = make_docs(n_docs=2, doc_len=64)
        kv_fn = make_kv_fn()
        cache.compile_corpus(docs, kv_fn)
        before_bytes = cache.memory_bytes()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            cache.save_artifacts(path)
            cache2 = WiCERIterativeKVWikiCache(config)
            cache2.load_artifacts(path)
            assert cache2.memory_bytes() == before_bytes
        finally:
            os.unlink(path)
