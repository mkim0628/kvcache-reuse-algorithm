"""SpecKVCompressionGammaController — SpecKV (arXiv 2605.02888) based gamma selector.

Activity C: Lightweight MLP selects optimal speculative decoding draft length (γ)
based on compression level and draft model signals. Online adaptation via EMA of
verification pass rate maintains accuracy while maximizing throughput.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class _GammaMLP(nn.Module):
    """Lightweight MLP gamma selector: input dim 5 → output 6 (γ ∈ {1,...,6}).

    Input features: [compression_level_onehot(3), min_draft_confidence(1), max_draft_entropy(1)]

    The first linear layer weights are structured so that FP16 activations (dim 0)
    push toward higher γ outputs and NF4 activations (dim 2) push toward lower γ outputs,
    encoding the SpecKV accuracy-preserving prior at initialization.
    """

    def __init__(self) -> None:
        super().__init__()
        # Input: [compression_onehot(3), min_confidence(1), max_entropy(1)] = 5 dims
        # Hidden: 16, Output: 6 (γ logits for {1,...,6})
        linear1 = nn.Linear(5, 16)
        linear2 = nn.Linear(16, 6)
        self.net = nn.Sequential(linear1, nn.ReLU(), linear2)

        # Structure the first-layer weights: FP16 input (index 0) activates "high-γ" neurons
        # (first 8 hidden), NF4 input (index 2) activates "low-γ" neurons (last 8 hidden).
        # Second-layer maps high-γ neurons → γ 4..6 logits, low-γ neurons → γ 1..3 logits.
        with torch.no_grad():
            # FP16 (input dim 0) → strongly activates first 8 hidden units
            linear1.weight.data[:8, 0] += 1.5
            # NF4 (input dim 2) → strongly activates last 8 hidden units
            linear1.weight.data[8:, 2] += 1.5

            # First 8 hidden units → boost γ 4..6 logits (output dims 3,4,5)
            linear2.weight.data[3:, :8] += 0.5
            linear2.weight.data[:3, :8] -= 0.5

            # Last 8 hidden units → boost γ 1..3 logits (output dims 0,1,2)
            linear2.weight.data[:3, 8:] += 0.5
            linear2.weight.data[3:, 8:] -= 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpecKVCompressionGammaController:
    """
    SpecKV (arXiv 2605.02888) based automatic compression-γ joint optimization.

    - Input: compression level (FP16=0/INT8=1/NF4=2), min_draft_confidence, max_draft_entropy
    - Output: γ ∈ {1, 2, 3, 4, 5, 6}
    - eOptShrinkQCodec (pre-implemented) compression level auto-injected into MLP input
    - Online adaptation: EMA-based γ bias correction from verification pass rate feedback

    Accuracy-preserving rationale:
    - Higher compression → lower γ → maintains verification pass rate → prevents accuracy drop
    - Draft signals (min_confidence, max_entropy) estimate optimal γ per batch in real time
    """

    COMPRESSION_FP16 = 0
    COMPRESSION_INT8 = 1
    COMPRESSION_NF4 = 2

    def __init__(
        self,
        base_seed: int = 42,
        ema_alpha: float = 0.05,
        target_verification_rate: float = 0.7,
    ) -> None:
        torch.manual_seed(base_seed)
        self._mlp = _GammaMLP()
        self._mlp.eval()

        self.ema_alpha = ema_alpha
        self.target_verification_rate = target_verification_rate

        self._verification_history: List[bool] = []
        self._gamma_bias: float = 0.0  # positive = upward bias, negative = downward bias

        self._profile_buffer: List[Tuple[int, float, float, int]] = []

    def select_gamma(
        self,
        compression_level: int,
        min_draft_confidence: float,
        max_draft_entropy: float,
    ) -> int:
        """
        Select optimal γ via MLP with online EMA bias correction.

        Returns: γ ∈ {1, 2, 3, 4, 5, 6}
        """
        onehot = torch.zeros(3)
        onehot[compression_level] = 1.0
        x = torch.cat([onehot, torch.tensor([min_draft_confidence, max_draft_entropy])])

        with torch.no_grad():
            logits = self._mlp(x.unsqueeze(0)).squeeze(0)  # [6]
        gamma_idx = int(logits.argmax().item())
        gamma = gamma_idx + 1  # map [0..5] → [1..6]

        # Apply EMA bias correction
        gamma = max(1, min(6, gamma + round(self._gamma_bias)))
        return gamma

    def record_verification(self, was_accepted: bool) -> None:
        """
        Feed speculative decoding verification result for online γ adaptation.

        Pass rate < target → lower γ bias (more conservative)
        Pass rate ≥ target → raise γ bias (more aggressive)
        """
        self._verification_history.append(was_accepted)
        recent = self._verification_history[-50:]  # sliding window of 50
        if len(recent) >= 10:
            actual_rate = sum(recent) / len(recent)
            if actual_rate < self.target_verification_rate:
                self._gamma_bias -= self.ema_alpha
            else:
                self._gamma_bias += self.ema_alpha
            self._gamma_bias = max(-2.0, min(2.0, self._gamma_bias))

    def integrate_with_eopt(
        self,
        eopt_codec: object,
        min_draft_confidence: float,
        max_draft_entropy: float,
    ) -> int:
        """
        Auto-inject eOptShrinkQCodec compression level into MLP input.

        Compression level mapping:
        - key_bits >= 4: FP16 level (COMPRESSION_FP16)
        - key_bits == 3: INT8 level (COMPRESSION_INT8)
        - key_bits <= 2: NF4 level (COMPRESSION_NF4)
        """
        key_bits = getattr(eopt_codec, "key_bits", 3)
        if key_bits >= 4:
            compression_level = self.COMPRESSION_FP16
        elif key_bits == 3:
            compression_level = self.COMPRESSION_INT8
        else:
            compression_level = self.COMPRESSION_NF4
        return self.select_gamma(compression_level, min_draft_confidence, max_draft_entropy)

    def collect_profile_record(
        self,
        compression_level: int,
        min_draft_confidence: float,
        max_draft_entropy: float,
        optimal_gamma: int,
    ) -> None:
        """Collect profiling data for MLP offline retraining (needs ≥ 512 records)."""
        self._profile_buffer.append(
            (compression_level, min_draft_confidence, max_draft_entropy, optimal_gamma)
        )

    def train_mlp_from_profile(self, epochs: int = 50) -> float:
        """
        Retrain MLP from collected profiling data.
        Requires at least 512 records; returns inf otherwise.
        Returns final average training loss.
        """
        if len(self._profile_buffer) < 512:
            return float("inf")

        self._mlp.train()
        optimizer = torch.optim.Adam(self._mlp.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        final_loss = float("inf")

        for _ in range(epochs):
            total_loss = 0.0
            for rec in self._profile_buffer:
                comp_lvl, min_conf, max_ent, opt_gamma = rec
                onehot = torch.zeros(3)
                onehot[comp_lvl] = 1.0
                x = torch.cat([onehot, torch.tensor([min_conf, max_ent])]).unsqueeze(0)
                target = torch.tensor([opt_gamma - 1], dtype=torch.long)
                logits = self._mlp(x)
                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            final_loss = total_loss / max(len(self._profile_buffer), 1)

        self._mlp.eval()
        return final_loss
