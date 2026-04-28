from typing import Dict, Tuple
import torch


class CompressionCodec:
    """Mixed-precision KV cache quantization codec (Activity C).

    Early layers (< cutoff) use FP16 to preserve critical information.
    Remaining layers use symmetric per-tensor INT8 quantization (~50% savings).
    """

    def __init__(self, num_layers: int, cutoff_ratio: float = 1 / 3) -> None:
        self.num_layers = num_layers
        self.cutoff = max(1, int(num_layers * cutoff_ratio))
        # Stores INT8 scale factors: key = (layer_idx, tensor_id)
        self._scales: Dict[Tuple[int, int], float] = {}

    def encode(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Compress KV tensor; FP16 for early layers, INT8 for later layers."""
        if layer_idx < self.cutoff:
            return kv.to(torch.float16)

        abs_max = kv.abs().max().item()
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        self._scales[(layer_idx, tensor_id)] = scale
        quantized = (kv.float() / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized

    def decode(
        self,
        compressed: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Decompress back to float32."""
        if layer_idx < self.cutoff:
            return compressed.to(torch.float32)

        scale = self._scales.get((layer_idx, tensor_id), 1.0)
        return compressed.to(torch.float32) * scale

    def compression_ratio(self, layer_idx: int) -> float:
        """Theoretical bytes saved vs FP32 baseline."""
        if layer_idx < self.cutoff:
            return 0.5  # FP16 = 2 bytes vs FP32 = 4 bytes → 50% savings
        return 0.75  # INT8 = 1 byte vs FP32 = 4 bytes → 75% savings

    def average_compression_ratio(self) -> float:
        early = self.cutoff * 0.5
        late = (self.num_layers - self.cutoff) * 0.75
        return (early + late) / self.num_layers
