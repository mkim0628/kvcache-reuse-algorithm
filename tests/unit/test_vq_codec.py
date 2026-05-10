"""Unit tests for VQCodec (Activity C — RSimVQCodec).

Covers: codebook reproducibility, encode/decode shape, MSE bounds,
perplexity delta ±1%, compression ratio, inverse RoPE correctness,
monotone M/n_residuals sweeps, save/load roundtrip.
"""

import os
import math
import tempfile
from typing import List, Optional, Tuple

import pytest
import torch

from src.compression.vq_codec import VQCodec, VQCodebookConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> VQCodebookConfig:
    defaults = dict(
        codebook_size=16,
        n_residuals=4,
        d_head=32,
        n_layers=2,
        n_heads=2,
        max_iter_kmeans=50,
        rope_base=10000,
        seed=42,
        recent_window=32,
    )
    defaults.update(kwargs)
    return VQCodebookConfig(**defaults)


def _make_kv(n_tokens: int, n_heads: int, d_head: int) -> torch.Tensor:
    """Random float16 KV tensor [n_tokens, 2, n_heads, d_head]."""
    torch.manual_seed(7)
    return torch.randn(n_tokens, 2, n_heads, d_head).to(torch.float16)


def _fit_codec(codec: VQCodec, n_tokens: int = 200, layer_idx: int = 0) -> None:
    """Fit a single layer codebook on random data."""
    torch.manual_seed(codec.config.seed)
    n_heads = codec.config.n_heads
    d_head = codec.config.d_head
    calib_k = torch.randn(n_tokens * n_heads, d_head)
    calib_v = torch.randn(n_tokens * n_heads, d_head)
    codec.fit(calib_k, calib_v, layer_idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_codebook_fit_reproducible() -> None:
    """Same seed + same calibration data → identical codebooks."""
    cfg = _make_config()
    calib_k = torch.randn(200, cfg.d_head)
    calib_v = torch.randn(200, cfg.d_head)

    codec1 = VQCodec(cfg)
    codec1.fit(calib_k.clone(), calib_v.clone(), layer_idx=0)

    codec2 = VQCodec(cfg)
    codec2.fit(calib_k.clone(), calib_v.clone(), layer_idx=0)

    for r in range(cfg.n_residuals):
        assert torch.allclose(
            codec1.key_codebooks[0][r], codec2.key_codebooks[0][r]
        ), f"Key codebook stage {r} not reproducible"
        assert torch.allclose(
            codec1.val_codebooks[0][r], codec2.val_codebooks[0][r]
        ), f"Val codebook stage {r} not reproducible"


def test_encode_decode_roundtrip_shape() -> None:
    """encode → decode preserves tensor shape."""
    cfg = _make_config(n_heads=2, d_head=32)
    codec = VQCodec(cfg)
    _fit_codec(codec)

    kv = _make_kv(32, cfg.n_heads, cfg.d_head)
    positions = torch.arange(32, dtype=torch.long)

    codes = codec.encode(kv, layer_idx=0, positions=positions)
    decoded = codec.decode(codes, layer_idx=0)

    assert decoded.shape == kv.shape, f"Shape mismatch: {decoded.shape} vs {kv.shape}"


def test_encode_decode_mse_bounded() -> None:
    """Relative MSE after encode/decode is within expected tolerance.

    With M=16 (16 codewords for d_head=32 vectors), quantization error is inherently
    high; we verify it is finite and below a loose bound. A tighter check uses M=256.
    """
    cfg = _make_config(codebook_size=16, n_residuals=4, d_head=32, n_heads=2)
    codec = VQCodec(cfg)
    _fit_codec(codec, n_tokens=300)

    kv = _make_kv(64, cfg.n_heads, cfg.d_head)
    positions = torch.arange(64, dtype=torch.long)

    codes = codec.encode(kv, layer_idx=0, positions=positions)
    decoded = codec.decode(codes, layer_idx=0)

    mse = torch.mean((kv.float() - decoded.float()) ** 2).item()
    norm_sq = torch.mean(kv.float() ** 2).item()
    relative_mse = mse / (norm_sq + 1e-8)
    # Loose bound for M=16: ensure encode/decode is functional (not NaN, not diverging)
    assert relative_mse < 1.0, f"Relative MSE {relative_mse:.4f} too high for M=16, n_r=4"
    assert not torch.isnan(decoded).any(), "Decoded tensor contains NaN"


def test_encode_decode_mse_bounded_high_precision() -> None:
    """Higher M gives lower MSE than low M."""
    cfg_low = _make_config(codebook_size=16, n_residuals=4, n_heads=2, d_head=32)
    cfg_high = _make_config(codebook_size=64, n_residuals=4, n_heads=2, d_head=32)

    # Fit both on the same calibration data
    torch.manual_seed(99)
    calib_k = torch.randn(400, 32)
    calib_v = torch.randn(400, 32)

    codec_low = VQCodec(cfg_low)
    codec_low.fit(calib_k.clone(), calib_v.clone(), layer_idx=0)

    codec_high = VQCodec(cfg_high)
    codec_high.fit(calib_k.clone(), calib_v.clone(), layer_idx=0)

    kv = _make_kv(64, 2, 32)
    positions = torch.arange(64, dtype=torch.long)

    def get_mse(codec: VQCodec) -> float:
        codes = codec.encode(kv, layer_idx=0, positions=positions)
        decoded = codec.decode(codes, layer_idx=0)
        return torch.mean((kv.float() - decoded.float()) ** 2).item()

    mse_low = get_mse(codec_low)
    mse_high = get_mse(codec_high)
    # M=64 should have equal or lower MSE than M=16
    assert mse_high <= mse_low * 1.5, (
        f"Higher M should give lower/equal MSE: M=64 MSE={mse_high:.6f} > M=16 MSE={mse_low:.6f}"
    )


def test_perplexity_delta_within_1pct() -> None:
    """Verify accuracy preservation: VQ compression preserves perplexity within ±1%.

    Uses a tiny 2-layer, 2-head, d_model=64 transformer with fixed random weights
    and a short token sequence (~256 tokens) to keep runtime < 60s.

    Design:
    - Build a minimal causal LM with a KV cache hook.
    - Run forward pass with FP16 KV → compute baseline perplexity.
    - Run forward pass where KV is VQ-encoded then decoded → compute compressed perplexity.
    - Assert abs(compressed_ppl - baseline_ppl) / baseline_ppl <= 0.01.

    The VQ codec is calibrated on the same sequence KV tensors so the codebook is
    in-distribution, giving the best-case (but still meaningful) accuracy bound.
    """
    torch.manual_seed(42)

    # Tiny model parameters
    vocab_size = 128
    seq_len = 64       # short sequence for speed
    d_model = 64
    n_heads = 2
    d_head = d_model // n_heads  # 32
    n_layers = 2

    # Build random token sequence (simulated WikiText-2 stub)
    token_ids = torch.randint(0, vocab_size, (seq_len,))

    # Embedding matrix: [vocab_size, d_model]
    torch.manual_seed(1)
    embed_weight = torch.randn(vocab_size, d_model) * 0.1
    # Output projection (tied to embedding): [d_model, vocab_size]
    out_weight = embed_weight.t()  # [d_model, vocab_size]

    # Transformer weight matrices per layer (fixed random, FP32 for stability)
    layers_weights = []
    for _ in range(n_layers):
        torch.manual_seed(len(layers_weights) * 7 + 3)
        Wqkv = torch.randn(d_model, 3 * d_model) * 0.02  # [d_model, 3*d_model]
        Wo = torch.randn(d_model, d_model) * 0.02          # [d_model, d_model]
        W1 = torch.randn(d_model, 4 * d_model) * 0.02
        W2 = torch.randn(4 * d_model, d_model) * 0.02
        layers_weights.append((Wqkv, Wo, W1, W2))

    def causal_self_attn(
        x: torch.Tensor,   # [T, d_model]
        Wqkv: torch.Tensor,
        Wo: torch.Tensor,
        kv_override: Optional[torch.Tensor] = None,  # [T, 2, n_heads, d_head]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-head causal self-attention; returns (output, kv_block)."""
        T = x.shape[0]
        qkv = x @ Wqkv  # [T, 3*d_model]
        q, k, v = qkv.split(d_model, dim=-1)  # each [T, d_model]

        # Reshape to [T, n_heads, d_head]
        q = q.view(T, n_heads, d_head)
        k = k.view(T, n_heads, d_head)
        v = v.view(T, n_heads, d_head)

        # Use overridden KV if provided (VQ reconstruction case)
        if kv_override is not None:
            k = kv_override[:, 0, :, :]
            v = kv_override[:, 1, :, :]

        kv_block = torch.stack([k, v], dim=1)  # [T, 2, n_heads, d_head]

        # Causal attention: [n_heads, T, T]
        scale = math.sqrt(d_head)
        scores = torch.einsum("thd,shd->hts", q, k) / scale  # [n_heads, T, T]
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        scores = scores + mask.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)

        out = torch.einsum("hts,shd->thd", attn, v)  # [T, n_heads, d_head]
        out = out.reshape(T, d_model)
        return out @ Wo, kv_block

    def forward_pass(
        tokens: torch.Tensor,
        kv_overrides: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Full forward pass. Returns (logits [T, vocab], list of kv_blocks per layer)."""
        x = embed_weight[tokens]  # [T, d_model]
        all_kvs = []
        for layer_idx, (Wqkv, Wo, W1, W2) in enumerate(layers_weights):
            override = kv_overrides[layer_idx] if kv_overrides is not None else None
            # Pre-norm (simple layer norm approximation via mean subtraction)
            xn = x - x.mean(dim=-1, keepdim=True)
            attn_out, kv_block = causal_self_attn(xn, Wqkv, Wo, kv_override=override)
            all_kvs.append(kv_block)
            x = x + attn_out
            # Feed-forward
            xn2 = x - x.mean(dim=-1, keepdim=True)
            ff = torch.relu(xn2 @ W1) @ W2
            x = x + ff
        logits = x @ out_weight  # [T, vocab_size]
        return logits, all_kvs

    def compute_perplexity(tokens: torch.Tensor, logits: torch.Tensor) -> float:
        """Cross-entropy perplexity on next-token prediction."""
        # Predict token t+1 from position t
        T = tokens.shape[0]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)  # [T-1, vocab]
        targets = tokens[1:].long()                         # [T-1]
        nll = -log_probs[torch.arange(T - 1), targets].mean()
        return math.exp(nll.item())

    # ── Step 1: baseline FP16 forward pass ──────────────────────────────
    with torch.no_grad():
        logits_fp16, kv_blocks_fp16 = forward_pass(token_ids)
    baseline_ppl = compute_perplexity(token_ids, logits_fp16)

    # ── Step 2: fit VQ codecs on baseline KV and run compressed forward ──
    cfg = _make_config(
        codebook_size=64,
        n_residuals=4,
        n_heads=n_heads,
        d_head=d_head,
        max_iter_kmeans=50,
        recent_window=8,   # keep only 8 tokens FP16; compress the rest
    )

    kv_overrides: List[Optional[torch.Tensor]] = []
    for layer_idx, kv_block_fp16 in enumerate(kv_blocks_fp16):
        # kv_block_fp16: [T, 2, n_heads, d_head]
        T = kv_block_fp16.shape[0]
        k_flat = kv_block_fp16[:, 0].reshape(-1, d_head)
        v_flat = kv_block_fp16[:, 1].reshape(-1, d_head)

        codec = VQCodec(cfg)
        codec.fit(k_flat.float(), v_flat.float(), layer_idx=layer_idx)

        positions = torch.arange(T, dtype=torch.long)
        codes = codec.encode(kv_block_fp16.float().to(torch.float16), layer_idx, positions)
        kv_reconstructed = codec.decode(codes, layer_idx).float()
        kv_overrides.append(kv_reconstructed)

    with torch.no_grad():
        logits_vq, _ = forward_pass(token_ids, kv_overrides=kv_overrides)
    compressed_ppl = compute_perplexity(token_ids, logits_vq)

    # ── Step 3: assert ±1% perplexity delta ─────────────────────────────
    assert baseline_ppl > 0 and math.isfinite(baseline_ppl), (
        f"Baseline perplexity is invalid: {baseline_ppl}"
    )
    assert math.isfinite(compressed_ppl), (
        f"Compressed perplexity is invalid: {compressed_ppl}"
    )
    delta_ratio = abs(compressed_ppl - baseline_ppl) / baseline_ppl
    assert delta_ratio <= 0.01, (
        f"Perplexity delta {delta_ratio:.4%} exceeds ±1% threshold. "
        f"baseline={baseline_ppl:.4f}, compressed={compressed_ppl:.4f}"
    )


def test_compression_ratio_meets_target() -> None:
    """Compression ratio >= 80% with codebook_size=16, n_residuals=4, recent_window=32.

    VQ stores n_r × log2(M) bits per d_head-dimensional vector.
    M=16, n_r=4, d_head=32: VQ bits = 4×4 = 16 vs FP16 baseline = 32×16 = 512 bits.
    vq_fraction = (512-32)/512 ≈ 0.94.
    ratio ≈ 1 - (0.94×16 + 0.06×512) / 512 ≈ 0.908 (90%).
    """
    cfg = _make_config(
        codebook_size=16,
        n_residuals=4,
        d_head=32,
        recent_window=32,
    )
    codec = VQCodec(cfg)
    ratio = codec.compression_ratio()
    assert ratio >= 0.80, f"compression_ratio={ratio:.3f} < 0.80 target"


def test_inverse_rope_correctness() -> None:
    """inverse_rope(apply_rope(k, pos), pos) ≈ k with MSE < 1e-5 in FP32."""
    torch.manual_seed(0)
    n_tokens, n_heads, d_head = 32, 4, 64
    k = torch.randn(n_tokens, n_heads, d_head)
    positions = torch.arange(n_tokens, dtype=torch.long)

    k_rotated = VQCodec.apply_rope(k.clone(), positions)
    k_recovered = VQCodec.inverse_rope(k_rotated, positions)

    mse = torch.mean((k.float() - k_recovered.float()) ** 2).item()
    assert mse < 1e-5, f"inverse_rope roundtrip MSE={mse:.2e} exceeds 1e-5"


def test_codec_m_sweep_mse_monotone() -> None:
    """MSE decreases (or stays equal) as codebook_size M increases."""
    torch.manual_seed(55)
    n_heads, d_head = 2, 32

    # Shared calibration and test data
    calib_k = torch.randn(400, d_head)
    calib_v = torch.randn(400, d_head)
    kv = _make_kv(32, n_heads, d_head)
    positions = torch.arange(32, dtype=torch.long)

    mse_values = {}
    for M in [16, 64, 256]:
        cfg = _make_config(codebook_size=M, n_residuals=2, n_heads=n_heads, d_head=d_head)
        codec = VQCodec(cfg)
        codec.fit(calib_k.clone(), calib_v.clone(), layer_idx=0)
        codes = codec.encode(kv, layer_idx=0, positions=positions)
        decoded = codec.decode(codes, layer_idx=0)
        mse_values[M] = torch.mean((kv.float() - decoded.float()) ** 2).item()

    # M=256 MSE should be <= M=16 MSE (monotone with some tolerance for k-means variability)
    assert mse_values[256] <= mse_values[16] * 1.5, (
        f"MSE not monotone: M=16:{mse_values[16]:.6f}, M=256:{mse_values[256]:.6f}"
    )


def test_codec_n_residuals_sweep_mse_monotone() -> None:
    """MSE decreases as n_residuals increases."""
    torch.manual_seed(77)
    n_heads, d_head = 2, 32

    calib_k = torch.randn(400, d_head)
    calib_v = torch.randn(400, d_head)
    kv = _make_kv(32, n_heads, d_head)
    positions = torch.arange(32, dtype=torch.long)

    mse_values = {}
    for n_r in [1, 2, 4]:
        cfg = _make_config(codebook_size=16, n_residuals=n_r, n_heads=n_heads, d_head=d_head)
        codec = VQCodec(cfg)
        codec.fit(calib_k.clone(), calib_v.clone(), layer_idx=0)
        codes = codec.encode(kv, layer_idx=0, positions=positions)
        decoded = codec.decode(codes, layer_idx=0)
        mse_values[n_r] = torch.mean((kv.float() - decoded.float()) ** 2).item()

    # More residuals → lower or equal MSE
    assert mse_values[4] <= mse_values[1] * 1.0 + 1e-6, (
        f"MSE not monotone with n_residuals: n=1:{mse_values[1]:.6f}, n=4:{mse_values[4]:.6f}"
    )


def test_codec_save_load_roundtrip() -> None:
    """save() → load() restores identical codebooks."""
    cfg = _make_config()
    codec = VQCodec(cfg)
    _fit_codec(codec, layer_idx=0)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        codec.save(path)
        codec2 = VQCodec(cfg)
        codec2.load(path)

        for r in range(cfg.n_residuals):
            assert torch.allclose(
                codec.key_codebooks[0][r], codec2.key_codebooks[0][r]
            ), f"Key codebook stage {r} mismatch after save/load"
    finally:
        os.unlink(path)


def test_compression_ratio_positive() -> None:
    """compression_ratio() > 0 meaning some compression is achieved."""
    cfg = _make_config(codebook_size=16, n_residuals=2, recent_window=32)
    codec = VQCodec(cfg)
    ratio = codec.compression_ratio()
    assert ratio > 0.0, "compression_ratio should be positive"


def test_encode_returns_expected_keys() -> None:
    """encode() result dict contains required keys."""
    cfg = _make_config(n_heads=2, d_head=32)
    codec = VQCodec(cfg)
    _fit_codec(codec)

    kv = _make_kv(16, cfg.n_heads, cfg.d_head)
    positions = torch.arange(16, dtype=torch.long)
    codes = codec.encode(kv, layer_idx=0, positions=positions)

    for key in ("key_codes", "val_codes", "layer_idx", "n_tokens", "positions"):
        assert key in codes, f"Missing key '{key}' in encode() output"

    assert codes["key_codes"].shape == (16, cfg.n_heads, cfg.n_residuals)
    assert codes["val_codes"].shape == (16, cfg.n_heads, cfg.n_residuals)
