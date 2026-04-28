from typing import List, Optional, Tuple
import torch

from src.cache.segmented import SegmentedHashCache
from src.cache.compression import CompressionCodec


class CompressedSegmentCache(SegmentedHashCache):
    """Compressed non-contiguous segment cache (Activity B+C Cross-1).

    Stores KV segments in compressed form (FP16/INT8) to maximise the
    number of reusable segments within a fixed memory budget.
    """

    def __init__(
        self,
        codec: CompressionCodec,
        chunk_size: int = 128,
        max_entries: int = 1000,
    ) -> None:
        super().__init__(chunk_size=chunk_size, max_entries=max_entries)
        self.codec = codec
        # Tracks layer_idx per stored key so decode can use the right precision
        self._key_layer: dict[str, int] = {}

    def put_segment(  # type: ignore[override]
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Compress then store the KV segment (layer-scoped key)."""
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)
        tensor_id = hash(key) % (2**31)
        compressed = self.codec.encode(kv, layer_idx, tensor_id)
        self._key_layer[key] = layer_idx
        self.put(key, compressed)

    def get_segments(  # type: ignore[override]
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Retrieve and decompress cached segments for a given layer."""
        n_chunks = max(1, (len(token_ids) + self.chunk_size - 1) // self.chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        for i in range(n_chunks):
            key = self.chunk_key(token_ids, i, layer_idx)
            compressed = self.get(key)
            if compressed is not None:
                tensor_id = hash(key) % (2**31)
                kv = self.codec.decode(compressed, layer_idx, tensor_id)
                hits.append((i, kv))
            else:
                misses.append(i)

        if hits:
            miss_set = set(misses)
            for idx, _ in hits:
                if any(m < idx for m in miss_set):
                    self._noncontiguous_hits += 1

        return hits, misses

    def memory_bytes(self) -> int:
        """Actual compressed storage size."""
        return sum(v.nbytes for v in self._store.values())

    def compression_ratio(self) -> float:
        """Average compression ratio across all stored segments."""
        return self.codec.average_compression_ratio()
