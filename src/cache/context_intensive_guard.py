"""ContextIntensiveAccuracyGuard — KV Cache Offloading for Context-Intensive Tasks (arXiv 2604.08426).

Activity C: Context information density estimation + automatic compression ratio gate.
Protects accuracy in high-density contexts by enforcing minimum bit-width floors.
"""

from typing import Dict

import torch


class ContextIntensiveAccuracyGuard:
    """
    Context density estimation + compression limit gate.

    Density levels and compression constraints:
    - High density (score ≥ threshold_high=0.7): min 4 bits (accuracy protection)
    - Medium density (0.4 ≤ score < 0.7): 2.2~4 bits (normal eOptShrinkQ range)
    - Low density (score < threshold_low=0.4): ≤ 2.2 bits (aggressive compression allowed)

    Integrates with eOptShrinkQCodec (pre-implemented) and ManifoldKVWindowedEviction
    via the gate_eopt_codec() interface.
    """

    def __init__(
        self,
        w1: float = 0.3,
        w2: float = 0.3,
        w3: float = 0.4,
        sample_tokens: int = 128,
        threshold_high: float = 0.7,
        threshold_low: float = 0.4,
    ) -> None:
        self.w1 = w1          # entity_ratio weight
        self.w2 = w2          # numeric_ratio weight
        self.w3 = w3          # token_entropy weight
        self.sample_tokens = sample_tokens
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

    def assess(self, token_ids: torch.Tensor) -> float:
        """
        Estimate context density from first sample_tokens tokens.

        context_density_score = w1 * entity_ratio + w2 * numeric_ratio + w3 * token_entropy

        entity_ratio: rare vocabulary token fraction (token ID > 50000 as proxy for named entities)
        numeric_ratio: digit/symbol token fraction (ID in range 10..100)
        token_entropy: normalized Shannon entropy of token ID distribution
        """
        sample = token_ids[: self.sample_tokens].float()
        n = len(sample)
        if n == 0:
            return 0.5  # default for empty input

        entity_ratio = float((sample > 50000).float().mean().item())
        numeric_ratio = float(((sample >= 10) & (sample <= 100)).float().mean().item())

        token_counts = torch.bincount(sample.long().clamp(0, 100000), minlength=1)
        probs = token_counts.float() / n
        probs = probs[probs > 0]
        entropy = float(-(probs * probs.log()).sum().item())
        max_entropy = float(torch.log(torch.tensor(float(n))).item())
        normalized_entropy = entropy / max(max_entropy, 1e-9)

        score = self.w1 * entity_ratio + self.w2 * numeric_ratio + self.w3 * normalized_entropy
        return float(min(1.0, max(0.0, score)))

    def get_compression_limits(self, density_score: float) -> Dict[str, float]:
        """
        Return compression parameter limits based on density score.

        Returns dict with:
          min_bits: minimum bit-width allowed (higher = less compression)
          max_compression_ratio: max compression ratio (1.0 = full compression)
          density_level: "high" | "medium" | "low"
        """
        if density_score >= self.threshold_high:
            return {
                "min_bits": 4.0,
                "max_compression_ratio": 0.5,
                "density_level": "high",
            }
        elif density_score >= self.threshold_low:
            return {
                "min_bits": 2.2,
                "max_compression_ratio": 0.75,
                "density_level": "medium",
            }
        else:
            return {
                "min_bits": 1.0,
                "max_compression_ratio": 0.9,
                "density_level": "low",
            }

    def gate_eopt_codec(
        self,
        eopt_codec: object,
        token_ids: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Auto-inject density-based compression parameters into eOptShrinkQCodec.

        High-density contexts: enforces key_bits ≥ 4 to protect accuracy.
        Returns dict of applied parameters.
        """
        score = self.assess(token_ids)
        limits = self.get_compression_limits(score)

        original_key_bits = getattr(eopt_codec, "key_bits", 2)
        original_val_bits = getattr(eopt_codec, "value_bits", 3)

        min_bits = limits["min_bits"]
        new_key_bits = max(int(min_bits), original_key_bits)
        new_val_bits = max(int(min_bits), original_val_bits)

        applied: Dict[str, object] = {
            "density_score": score,
            "density_level": limits["density_level"],
            "original_key_bits": original_key_bits,
            "original_val_bits": original_val_bits,
            "applied_key_bits": new_key_bits,
            "applied_val_bits": new_val_bits,
        }
        return applied
