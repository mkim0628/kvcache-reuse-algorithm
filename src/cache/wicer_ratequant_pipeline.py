"""WiCERRateQuantPipeline — Activity B+C integration.

Combines WiCERIterativeKVWikiCache (CEGAR domain artefact cache) with
RateQuantReverseWaterfillingCodec (reverse water-filling bit allocation).

Key properties:
- CEGAR compile/refine steps apply RateQuant compression immediately.
- Per-head bit allocation (r_h) metadata is serialised alongside artefacts
  so that loaded pipelines require no re-quantisation overhead.
- Implements CacheStore fully by delegating to WiCERIterativeKVWikiCache.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.wicer_iterative_cache import WiCERConfig, WiCERIterativeKVWikiCache
from src.cache.ratequant_codec import RateQuantConfig, RateQuantReverseWaterfillingCodec


@dataclass
class WiCERRateQuantConfig:
    wicer: WiCERConfig = None
    ratequant: RateQuantConfig = None

    def __post_init__(self) -> None:
        if self.wicer is None:
            self.wicer = WiCERConfig()
        if self.ratequant is None:
            self.ratequant = RateQuantConfig()


class WiCERRateQuantPipeline(CacheStore):
    """B+C integrated pipeline: CEGAR artefact cache + reverse water-filling quantisation.

    CacheStore interface is fully implemented by delegating to WiCERIterativeKVWikiCache.
    RateQuant compression is applied in compression_hook() and during build_pipeline().
    """

    def __init__(self, pipeline_config: WiCERRateQuantConfig) -> None:
        self.config = pipeline_config
        self.wicer = WiCERIterativeKVWikiCache(pipeline_config.wicer)
        self.codec = RateQuantReverseWaterfillingCodec(pipeline_config.ratequant)
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Compress via RateQuant then store in WiCER."""
        compressed = self.compression_hook(key, value)
        self.wicer.put(key, compressed)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.wicer.get(key)

    def evict(self) -> int:
        return self.wicer.evict()

    def hit_rate(self) -> float:
        return self.wicer.hit_rate()

    def memory_bytes(self) -> int:
        return self.wicer.memory_bytes()

    def reset_stats(self) -> None:
        self.wicer.reset_stats()
        self._hits = 0
        self._misses = 0

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
    ) -> torch.Tensor:
        """RateQuant encode → decode round-trip; falls back to identity if uncalibrated."""
        if not self.codec._calibrated:
            return value
        encoded = self.codec.encode(value, layer_idx=0)
        return self.codec.decode(encoded, layer_idx=0)

    # ------------------------------------------------------------------ #
    # B+C pipeline API                                                     #
    # ------------------------------------------------------------------ #

    def build_pipeline(
        self,
        docs: Dict[str, List[int]],
        val_queries: List[List[int]],
        kv_fn: Callable[[List[int], int], torch.Tensor],
        calibration_kvs: Optional[List[torch.Tensor]] = None,
        layer_idx: int = 0,
    ) -> None:
        """Run the full B+C pipeline:
          1. Calibrate RateQuant codec (if calibration_kvs provided).
          2. Execute CEGAR loop with RateQuant compression applied at every
             compile/refine step.
        """
        if calibration_kvs:
            self.codec.calibrate(calibration_kvs)
        self.wicer.cegar_refine(docs, val_queries, kv_fn, layer_idx, self.codec)

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Delegate to WiCER.get_segments() (runner.py compatibility)."""
        return self.wicer.get_segments(token_ids, layer_idx)

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Delegate to WiCER.put_segment() (runner.py compatibility)."""
        self.wicer.put_segment(token_ids, chunk_idx, kv, layer_idx)

    def noncontiguous_hit_rate(self) -> float:
        return self.wicer.noncontiguous_hit_rate()

    def cegar_hit_rate_history(self) -> List[float]:
        return self.wicer.cegar_hit_rate_history()

    # ------------------------------------------------------------------ #
    # Serialisation (bit-allocation metadata included for zero-overhead load) #
    # ------------------------------------------------------------------ #

    def save_pipeline(self, path: str) -> None:
        """Serialise WiCER artefacts + RateQuant calibration to one file.

        Loading via load_pipeline() requires no re-quantisation because the
        per-head bit allocation (r_h) metadata is stored alongside the artefacts.
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "wicer_store": self.wicer._store._store,
                "wicer_chunk_sizes": self.wicer._chunk_sizes,
                "wicer_hit_rate_history": self.wicer._hit_rate_history,
                "wicer_cegar_iteration": self.wicer._cegar_iteration,
                "codec_bit_allocation": self.codec.bit_allocation,
                "codec_head_variances": self.codec.head_variances,
                "codec_calibrated": self.codec._calibrated,
                "codec_config": self.codec.config,
            },
            path,
        )

    def load_pipeline(self, path: str) -> None:
        """Restore pipeline from a saved file (no re-quantisation required)."""
        from collections import OrderedDict
        data = torch.load(path, weights_only=False)
        self.wicer._store._store = OrderedDict(data["wicer_store"])
        self.wicer._chunk_sizes = data["wicer_chunk_sizes"]
        self.wicer._hit_rate_history = data["wicer_hit_rate_history"]
        self.wicer._cegar_iteration = data.get("wicer_cegar_iteration", 0)
        self.codec.bit_allocation = data["codec_bit_allocation"]
        self.codec.head_variances = data["codec_head_variances"]
        self.codec._calibrated = data.get("codec_calibrated", bool(self.codec.bit_allocation))
        if "codec_config" in data:
            self.codec.config = data["codec_config"]
