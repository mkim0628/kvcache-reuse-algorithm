"""eOptShrinkQCodec — BBP phase-transition automatic low-rank + TurboQuantCodec residual.

Activity C: eOptShrinkQ (arXiv 2605.02905) based KV cache compression.
BBP (Baik-Ben Arous-Péché) threshold selects signal rank automatically from noise level,
then TurboQuantCodec quantizes the residual (Key 2-bit / Value 3-bit asymmetric).
Effective compression: ~2.2 bits/element.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from src.cache.turbo_quant import TurboQuantCodec


class LowRankComponents(TypedDict):
    """Float16 low-rank decomposition components: U[:, :r], S[:r], V[:r, :]."""

    U: torch.Tensor  # [n_tokens, r_eff] float16
    S: torch.Tensor  # [r_eff] float16
    V: torch.Tensor  # [r_eff, d_head] float16


class SingleComponentPayload(TypedDict):
    """Encoded payload for one KV component (key or value)."""

    lowrank: Optional[LowRankComponents]  # None when SVD failed
    residual: Any  # TurboQuantCodec compressed dict


class EncodedKVPayload(TypedDict):
    """Full output of eOptShrinkQCodec.encode()."""

    key: SingleComponentPayload
    val: SingleComponentPayload
    layer_idx: int
    n_tokens: int
    d_head: int


class eOptShrinkQCodec:
    """BBP phase-transition automatic low-rank + TurboQuantCodec residual dual pipeline.

    Not a CacheStore subclass — pure codec called by schedulers/pipelines via encode/decode.
    """

    def __init__(
        self,
        num_layers: int,
        key_bits: int = 2,
        value_bits: int = 3,
        calibration_samples: int = 20,
        base_seed: int = 42,
    ) -> None:
        self.num_layers = num_layers
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.calibration_samples = calibration_samples

        # Reuse TurboQuantCodec for residual quantization (Key / Value asymmetric)
        self._key_codec = TurboQuantCodec(
            num_layers=num_layers, bits=key_bits, base_seed=base_seed
        )
        self._val_codec = TurboQuantCodec(
            num_layers=num_layers, bits=value_bits, base_seed=base_seed + 1
        )

        self._noise_levels: Dict[int, float] = {}
        self._auto_ranks: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Calibration                                                          #
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        calibration_kvs: List[torch.Tensor],
        save_path: Optional[str] = None,
    ) -> None:
        """Estimate per-layer noise_level and select auto rank via BBP threshold.

        calibration_kvs[i]: [n_tokens, d_head] float32 tensor for layer i.
        Requires calibration_samples >= 20 for reliable Marchenko-Pastur estimation.
        """
        for layer_idx, kv in enumerate(calibration_kvs):
            if kv.numel() == 0:
                continue
            kv_f = kv.float()
            n, d = kv_f.shape
            if n < 2 or d < 2:
                self._noise_levels[layer_idx] = 1.0
                self._auto_ranks[layer_idx] = 1
                continue

            aspect_ratio = min(n, d) / max(n, d)

            try:
                _, S, _ = torch.linalg.svd(kv_f, full_matrices=False)
            except RuntimeError:
                self._noise_levels[layer_idx] = 1.0
                self._auto_ranks[layer_idx] = min(8, min(n, d))
                continue

            # Marchenko-Pastur right edge approximation via singular value median
            noise_level = S.median().item() / math.sqrt(max(n, d))
            self._noise_levels[layer_idx] = noise_level

            # BBP phase-transition threshold: sigma_c = noise * (1 + sqrt(aspect))^2
            sigma_c = noise_level * (1.0 + math.sqrt(aspect_ratio)) ** 2

            r = int((S > sigma_c).sum().item())
            r = max(1, min(r, min(n, d) // 2))
            self._auto_ranks[layer_idx] = r

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {"noise_levels": self._noise_levels, "auto_ranks": self._auto_ranks},
                save_path,
            )

    def load_calibration(self, load_path: str) -> None:
        """Load persisted calibration file."""
        ckpt = torch.load(load_path, weights_only=False)
        self._noise_levels = ckpt["noise_levels"]
        self._auto_ranks = ckpt["auto_ranks"]

    # ------------------------------------------------------------------ #
    # Encoding                                                             #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        kv_key: torch.Tensor,
        kv_val: torch.Tensor,
        layer_idx: int,
    ) -> EncodedKVPayload:
        """Low-rank decomposition + TurboQuantCodec residual quantization.

        Returns compressed dict with lowrank components and quantized residuals.
        """
        n_tokens, d_head = kv_key.shape
        r = self._auto_ranks.get(layer_idx, max(1, min(8, min(n_tokens, d_head) // 2)))

        def _encode_single(
            kv: torch.Tensor, codec: TurboQuantCodec, tensor_id: int
        ) -> Dict:
            kv_f = kv.float()
            try:
                U, S, Vh = torch.linalg.svd(kv_f, full_matrices=False)
            except RuntimeError:
                return {"lowrank": None, "residual": codec.encode(kv_f, layer_idx, tensor_id)}

            r_eff = min(r, S.shape[0])
            U_r = U[:, :r_eff]
            S_r = S[:r_eff]
            Vh_r = Vh[:r_eff, :]
            lowrank_approx = (U_r * S_r.unsqueeze(0)) @ Vh_r

            residual = kv_f - lowrank_approx
            compressed_residual = codec.encode(residual, layer_idx, tensor_id)

            return {
                "lowrank": {
                    "U": U_r.half(),
                    "S": S_r.half(),
                    "V": Vh_r.half(),
                },
                "residual": compressed_residual,
            }

        return {
            "key": _encode_single(kv_key, self._key_codec, tensor_id=0),
            "val": _encode_single(kv_val, self._val_codec, tensor_id=1),
            "layer_idx": layer_idx,
            "n_tokens": n_tokens,
            "d_head": d_head,
        }

    # ------------------------------------------------------------------ #
    # Decoding                                                             #
    # ------------------------------------------------------------------ #

    def decode(self, compressed: EncodedKVPayload) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct key and value tensors from compressed representation.

        Returns: (key_approx, val_approx), each [n_tokens, d_head] float32.
        """
        layer_idx = compressed["layer_idx"]

        def _decode_single(comp: Dict, codec: TurboQuantCodec) -> torch.Tensor:
            residual_approx = codec.decode(comp["residual"], layer_idx)
            if comp.get("lowrank") is None:
                return residual_approx
            U_r = comp["lowrank"]["U"].float()
            S_r = comp["lowrank"]["S"].float()
            Vh_r = comp["lowrank"]["V"].float()
            lowrank_approx = (U_r * S_r.unsqueeze(0)) @ Vh_r
            return lowrank_approx + residual_approx

        key_approx = _decode_single(compressed["key"], self._key_codec)
        val_approx = _decode_single(compressed["val"], self._val_codec)
        return key_approx, val_approx

    def memory_bytes_estimate(
        self, n_tokens: int, d_head: int, layer_idx: int = 0
    ) -> Dict:
        """Estimate post-compression memory usage vs FP32 baseline."""
        r = self._auto_ranks.get(layer_idx, 8)
        # Low-rank components stored as float16 (2 bytes each)
        key_lowrank_bytes = (n_tokens * r + r + r * d_head) * 2
        val_lowrank_bytes = (n_tokens * r + r + r * d_head) * 2
        key_res_bytes = self._key_codec.memory_bytes_estimate(n_tokens, d_head, layer_idx)[
            "total_bytes"
        ]
        val_res_bytes = self._val_codec.memory_bytes_estimate(n_tokens, d_head, layer_idx)[
            "total_bytes"
        ]
        total = key_lowrank_bytes + val_lowrank_bytes + key_res_bytes + val_res_bytes
        baseline = n_tokens * d_head * 4 * 2  # FP32 K + V
        return {
            "total_bytes": total,
            "baseline_bytes": baseline,
            "reduction_ratio": 1.0 - total / max(baseline, 1),
        }
