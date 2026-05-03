#!/bin/bash
# install.sh — Install the latest vLLM and verify the A+B+C integration.
#
# Usage:
#   bash vllm_integration/install.sh
#
# This script:
#   1. Upgrades vLLM to the latest available version (no version pinning).
#   2. Prints the installed version for record-keeping.
#   3. Runs smoke tests for Activity A (DualMapSchedulerMixin),
#      Activity B (SemanticNonContiguousKVCacheManager),
#      Activity C (VllmTurboQuantCodec + TurboQuantKVHook).

set -euo pipefail

echo "=== Installing latest vLLM ==="
pip install --upgrade vllm

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
echo "vLLM version: ${VLLM_VERSION}"

echo ""
echo "=== 2026-05-03 A+B+C smoke tests (DualMap + DHD + TurboQuant) ==="
python - <<'PYEOF'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch

# -----------------------------------------------------------------------
# Activity C: VllmTurboQuantCodec + CacheCompressionConfig
# -----------------------------------------------------------------------
from vllm_integration.compression_codec import (
    VllmTurboQuantCodec,
    TurboQuantCodec,
    CacheCompressionConfig,
)

codec = VllmTurboQuantCodec(num_layers=12, bits=3, base_seed=42)
kv_2d = torch.randn(16, 64)
kv_3d = torch.randn(8, 4, 64)

# 2D encode/decode
enc2 = codec.encode_tokens(kv_2d, layer_idx=0)
dec2 = codec.decode_tokens(enc2, layer_idx=0)
assert dec2.shape == kv_2d.shape, f"2D shape mismatch: {dec2.shape}"

# 3D encode/decode (multi-head)
enc3 = codec.encode_tokens(kv_3d, layer_idx=3)
dec3 = codec.decode_tokens(enc3, layer_idx=3)
assert dec3.shape == kv_3d.shape, f"3D shape mismatch: {dec3.shape}"

# Compression ratio
ratio = codec.compression_ratio(layer_idx=6)
assert ratio >= 0.60, f"Compression ratio too low: {ratio:.3f}"

# CacheCompressionConfig
cfg = CacheCompressionConfig(compression_method="turbo3", num_layers=12)
built_codec = cfg.build_codec()
assert built_codec is not None
print(f"Activity C (VllmTurboQuantCodec + CacheCompressionConfig): OK  ratio={ratio:.3f}")

# -----------------------------------------------------------------------
# Activity C: TurboQuantKVHook
# -----------------------------------------------------------------------
from vllm_integration.attention_backend_patch import TurboQuantKVHook

hook = TurboQuantKVHook(codec=codec, enabled=True)
kv_tensor = torch.randn(32, 64)
compressed = hook.write_to_cache(kv_tensor, layer_idx=5, tensor_id=0)
decoded = hook.read_from_cache(compressed, layer_idx=5, tensor_id=0)
assert decoded.shape == kv_tensor.shape, f"Hook decode shape mismatch: {decoded.shape}"
print(f"Activity C (TurboQuantKVHook write/read): OK")

# -----------------------------------------------------------------------
# Activity B: SemanticSegmentIndex
# -----------------------------------------------------------------------
from vllm_integration.block_manager_patch import SemanticSegmentIndex

index = SemanticSegmentIndex(
    codec=codec,
    chunk_size=16,
    max_entries=200,
    top_k=3,
    similarity_threshold=0.70,
    deviation_threshold=0.50,
)

token_ids = list(range(32))
keys_chunk = torch.randn(16, 64)
vals_chunk = torch.randn(16, 64)

stored_key = index.store_segment(token_ids, chunk_idx=0, keys=keys_chunk, values=vals_chunk, layer_idx=0)
assert len(stored_key) == 64, "Expected 64-char SHA-256 hex key"

# Exact hit
k_ret, v_ret, hit_type = index.lookup_segment(token_ids, chunk_idx=0, query_keys=keys_chunk, layer_idx=0)
assert hit_type == "exact", f"Expected exact hit, got {hit_type}"
assert k_ret is not None and k_ret.shape == keys_chunk.shape

# Semantic hit with similar keys
noisy_keys = keys_chunk + 0.01 * torch.randn_like(keys_chunk)
other_ids = [x + 1000 for x in token_ids]  # different token hash → forces semantic path
k_sem, v_sem, hit_sem = index.lookup_segment(other_ids, chunk_idx=0, query_keys=noisy_keys, layer_idx=0)
print(f"Activity B (SemanticSegmentIndex exact hit, semantic path={hit_sem}): OK")

stats = index.semantic_hit_rates()
assert "noncontiguous_ratio" in stats
print(f"Activity B (hit_rate={index.hit_rate():.2f}): OK")

# -----------------------------------------------------------------------
# Activity A: DualMapRoutingMixin + DualMapSchedulerMixin
# -----------------------------------------------------------------------
from vllm_integration.scheduler_patch import (
    DualMapNodeState,
    DualMapRoutingMixin,
    DualMapSchedulerMixin,
)

nodes = [
    DualMapNodeState(node_id="n0", semantic_index=index._semantic_index, current_load=0.3),
    DualMapNodeState(node_id="n1", current_load=0.1),
]
router = DualMapRoutingMixin(nodes=nodes, top_k_semantic=3)

# Single request routing
target = router.route_request("req_001", list(range(1, 33)))
assert target in ("n0", "n1"), f"Unknown node: {target}"
print(f"Activity A (DualMapRoutingMixin route → {target}): OK")

# Batch sort
class MockReq:
    def __init__(self, rid, tids):
        self.request_id = rid
        self.prompt_token_ids = tids

reqs = [MockReq(f"r{i}", list(range(i * 10, i * 10 + 20))) for i in range(5)]
sorted_reqs = router.sort_by_cache_affinity(
    reqs,
    get_request_id=lambda r: r.request_id,
    get_token_ids=lambda r: r.prompt_token_ids,
)
assert len(sorted_reqs) == 5
print(f"Activity A (sort_by_cache_affinity 5 reqs): OK")

# DualMapSchedulerMixin overhead check
mixin = DualMapSchedulerMixin(nodes=nodes, top_k_semantic=3)
import time
t0 = time.monotonic()
for _ in range(100):
    mixin.sort_by_cache_affinity(
        reqs,
        get_request_id=lambda r: r.request_id,
        get_token_ids=lambda r: r.prompt_token_ids,
    )
elapsed_ms = (time.monotonic() - t0) * 1000
overhead_per_req = elapsed_ms / (100 * 5)
assert overhead_per_req < 5.0, f"Scheduling overhead too high: {overhead_per_req:.3f}ms/req"
print(f"Activity A (scheduling overhead={overhead_per_req:.4f}ms/req < 5ms): OK")

print(f"\nAll 2026-05-03 A+B+C smoke tests passed.  vLLM version: {__import__('vllm').__version__}")
PYEOF

echo ""
echo "=== Prior cycle backward-compat checks ==="
python - <<'PYEOF2'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
import torch

# Prior-cycle compression codecs
from vllm_integration.compression_codec import HadamardInt4Codec, CompressionCodec
codec_int4 = HadamardInt4Codec(num_layers=32, cutoff_ratio=0.2)
kv = torch.randn(8, 64)
enc = codec_int4.encode(kv, layer_idx=10, tensor_id=0)
dec = codec_int4.decode(enc, layer_idx=10, tensor_id=0)
assert dec.shape == kv.shape
print("Prior-cycle HadamardInt4Codec: OK")

# Prior-cycle attention hooks
from vllm_integration.attention_backend_patch import CompressedKVHook, TriStateKVHook
hook = CompressedKVHook(codec_int4)
enc2 = hook.encode(kv, layer_idx=10)
dec2 = hook.decode(enc2, layer_idx=10)
assert dec2.shape == kv.shape
print("Prior-cycle CompressedKVHook: OK")

# Prior-cycle scheduler
from vllm_integration.scheduler_patch import create_cache_hit_aware_queue
queue = create_cache_hit_aware_queue(chunk_size=8)
assert len(queue) == 0
print("Prior-cycle CacheHitAwareRequestQueue: OK")

print("\nAll backward-compat checks passed.")
PYEOF2

echo ""
echo "=== Installation complete ==="
echo "vLLM version: ${VLLM_VERSION}"
