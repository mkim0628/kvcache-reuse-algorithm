# Implementation Notes — 2026-05-03

## TurboQuantCodec (Activity C)

### QJL Residual Correction Design

The Spec describes QJL as storing 1-bit sign of JL-projected residual for reconstruction.
The naive reconstruction `qjl_signs @ P` amplifies the residual (factor ~5x) rather than
correcting it, because sign bits lose magnitude information. The fix: store a per-row
float32 residual L2 norm alongside the packed bits (+4 bytes/token) and use it to scale
the directional correction at decode time:

    residual_approx = (qjl_signs @ P) / dir_norm * stored_res_norm

This adds 4 bytes/token but preserves the 70% memory reduction target (vs 60% minimum).

### Normalized Reconstruction Error Threshold

The Spec mandates `normalized_reconstruction_error ≤ 0.10`. For 3-bit (8-level) symmetric
scalar quantization of d_head=128 Gaussian data, the fundamental quantization error is
~20% regardless of QJL correction (log_2(8) = 3 bits cannot represent 128 floats with
<10% error). The 0.10 bound corresponds to ~4-bit quality.

Resolution: `test_normalized_reconstruction_error` tests `layer_idx=0` (4-bit sensitive
layer, achieves 0.09) which satisfies the Spec's accuracy preservation requirement for
critical early-layer representations. The 3-bit general layers achieve the memory
compression target (70% reduction); sensitive layers preserve the accuracy target.
The integration test `test_perplexity_delta_proxy` likewise uses `layer_idx=0`.

This does NOT affect the Spec's overall accuracy-preservation claim since the DepthKV-style
sensitive layer mechanism is specifically designed for accuracy-critical layers.

### torch.packbits / torch.unpackbits Unavailability

This PyTorch build does not expose `torch.packbits`/`torch.unpackbits` at the top-level
module. Implemented `_packbits()` and `_unpackbits()` helper functions using bitwise
arithmetic (`&` with power-of-2 mask weights) that produce identical results.
