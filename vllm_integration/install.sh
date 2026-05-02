#!/bin/bash
# install.sh — Install the latest vLLM and verify the A+B+C+SignVQ integration.
#
# Usage:
#   bash vllm_integration/install.sh
#
# This script:
#   1. Upgrades vLLM to the latest available version (no version pinning).
#   2. Prints the installed version for record-keeping.
#   3. Runs smoke tests for all activities (A, B, C, and B+C SignVQ cycle).

set -euo pipefail

echo "=== Installing latest vLLM ==="
pip install --upgrade vllm

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
echo "vLLM version: ${VLLM_VERSION}"

echo ""
echo "=== Smoke-testing vllm_integration imports (A+B+C) ==="
python - <<'PYEOF'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch

# --- Activity C: HadamardInt4Codec ---
from vllm_integration.compression_codec import HadamardInt4Codec, CompressionCodec

codec_int4 = HadamardInt4Codec(num_layers=32, cutoff_ratio=0.2)
codec_int8 = CompressionCodec(num_layers=32)

kv = torch.randn(8, 64)
for layer_idx in [0, 10, 20, 31]:
    enc = codec_int4.encode(kv, layer_idx, tensor_id=0)
    dec = codec_int4.decode(enc, layer_idx, tensor_id=0)
    assert dec.shape == kv.shape, f"HadamardInt4Codec shape mismatch at layer {layer_idx}"
print("Activity C (HadamardInt4Codec): OK")

# --- Activity B: NonContiguousKVCacheManager + CompressedSegmentIndex ---
from vllm_integration.block_manager_patch import (
    SegmentHashMixin,
    CompressedSegmentIndex,
    NonContiguousKVCacheManager,
)

index = CompressedSegmentIndex(codec=codec_int4, max_entries=100)
key = SegmentHashMixin.get_segment_key([1, 2, 3, 4, 5, 6, 7, 8], chunk_idx=0, layer_idx=0, chunk_size=8)
assert len(key) == 64, "Expected 64-char hex key"
kv8 = torch.randn(8, 64)
index.put(key, kv8, layer_idx=5, tensor_id=0)
retrieved = index.get(key, tensor_id=0)
assert retrieved is not None and retrieved.shape == kv8.shape
print("Activity B (SegmentHashMixin + CompressedSegmentIndex): OK")

# --- Activity A: CacheHitAwareRequestQueue ---
from vllm_integration.scheduler_patch import (
    CacheHitAwareRequestQueue,
    create_cache_hit_aware_queue,
)

queue = create_cache_hit_aware_queue(segment_index=index, chunk_size=8)
assert len(queue) == 0
print("Activity A (CacheHitAwareRequestQueue): OK")

# --- Attention backend hook ---
from vllm_integration.attention_backend_patch import (
    CompressedKVHook,
    NonContiguousAttentionWrapper,
)
hook = CompressedKVHook(codec_int4)
kv_3d = torch.randn(8, 4, 64)
enc = hook.encode_kv(kv_3d, layer_idx=5, is_key=True)
dec = hook.decode_kv(enc, layer_idx=5, is_key=True)
assert dec.shape == kv_3d.shape
print("Activity B+C attention hook: OK")

print(f"\nAll A+B+C smoke tests passed.  vLLM version: {__import__('vllm').__version__}")
PYEOF

echo ""
echo "=== 2026-05-02 B+C SignVQ cycle smoke tests ==="
python - <<'PYEOF2'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch

# LeverageCompressor
from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor
comp = VllmLeverageCompressor(rank=4, tier1_ratio=0.20, tier3_ratio=0.20)
k = torch.randn(16, 32)
v = torch.randn(16, 32)
storage = comp.encode_block(k, v, layer_idx=0, block_id=0)
decoded = comp.decode_block(storage)
assert decoded.shape == (2, 16, 32), f"Unexpected shape: {decoded.shape}"
print("VllmLeverageCompressor encode/decode: OK")

# SignVQSegmentIndex
from vllm_integration.sign_vq_block_manager_patch import SignVQSegmentIndex
idx = SignVQSegmentIndex(compressor=comp, chunk_size=16, max_entries=100)
tokens = list(range(32))
idx.put(tokens, 0, k, v, layer_idx=0)
kv_out, hit_type = idx.get(tokens, 0, layer_idx=0, query_keys=k)
assert hit_type in ("exact_fp16", "approx_sign"), f"Expected hit, got {hit_type}"
print(f"SignVQSegmentIndex lookup hit_type={hit_type}: OK")

# CacheConfig extension
from vllm_integration.cache_config_extension import (
    SignVQCacheParams, build_sign_vq_compressor, build_sign_vq_index,
)
params = SignVQCacheParams(enable_sign_vq=True, tier1_ratio=0.20,
                           tier2_ratio=0.60, hamming_threshold=0.15)
c2 = build_sign_vq_compressor(params)
i2 = build_sign_vq_index(params)
assert abs(params.tier3_ratio - 0.20) < 1e-9
print("SignVQCacheParams + factories: OK")

print("\nAll 2026-05-02 B+C SignVQ smoke tests passed.")
PYEOF2

echo ""
echo "=== Installation complete ==="
