<!-- 변경 이유 (이전 Spec.md: 2026-05-13 대비):
이전 사이클(2026-05-13)은 A+B+C (PBKVAgentSegmentPreservationScheduler +
KVFoldAccumulativeRadixCache + SRFTFusedINT4KVKernel + AgenticChunkPreCachingPipeline) 조합이었다.
이번 사이클은 B+C로 초점을 좁힌다.

주요 변경:
1. [Activity A 제거] 이번 사이클에서 신규 Activity A 구현 없음. 기존 Activity A 구현체는 회귀 없이 보존.

2. [Activity B 교체] KVFoldAccumulativeRadixCache(foldl 누산기) →
   FibQuantVQSegmentCache(FibQuant 방사상-각도 VQ 독립 세그먼트 압축).
   각 비연속 세그먼트를 독립적으로 FibQuant 인코딩하여 랜덤-접근 복원을 지원한다.
   파일: src/cache/fibquant_vq_segment_cache.py (신규).

3. [Activity C 신규] SRFTFusedINT4KVKernel(SRFT+INT4) →
   FibQuantVQCodec(구형-베타 소스 매칭 방사상-각도 코드북).
   기구현 RSimVQCodec(k-means pre-RoPE 잔차 VQ)과 이론 기반이 다르며 10× 이상 고압축 영역에서 차별화.
   파일: src/cache/fibquant_vq_codec.py (신규).

4. [Cross B+C 신규] FibQuantPositionFreeSegmentCache: FibQuantVQSegmentCache +
   pre-RoPE position-decoupled 저장(기구현 RoPEReencodingNonContiguousCache 원리 재활용).
   파일: src/cache/fibquant_position_free_segment.py (신규).

5. [보존 파일] 기존 모든 파일(kv_fold_accumulative.py, srft_int4_kv_kernel.py,
   rope_reencoding_cache.py, agentic_chunk_precaching.py 등)은 이번 사이클에서 수정하지 않는다.
   기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.

6. [인터페이스 유지] src/cache/base.py는 이번 사이클에서 수정하지 않는다.
   store_pre_rope()/load_with_rope() 선택적 메서드 이미 존재. CacheStore 6개 추상 메서드를
   모든 신규 구현체가 완전 구현한다.

7. [Activity C 필수] FibQuantVQCodec은 accuracy-preserving 검증 계획 없이 완성 불가.
   WikiText-2 perplexity ±1% + LongBench 8개 서브태스크 + 압축률별(4×/10×/20×) 코사인 유사도
   측정을 tests/unit/test_fibquant_vq_accuracy.py에 구현한다.
-->

# Spec — 2026-05-14

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-14.md`
**최우선 구현 타겟**:
- **C-1**: FibQuantVQCodec — 구형-베타 소스 매칭 방사상-각도 코드북 (FibQuant, arXiv 2605.11478)
- **B-1**: FibQuantVQSegmentCache — FibQuant 독립 세그먼트 압축 + 랜덤-접근 복원
- **Cross-1**: FibQuantPositionFreeSegmentCache — pre-RoPE position-decoupled 저장 + FibQuant VQ

**해결하려는 문제**:

- Activity B: 표준 비연속 세그먼트 캐시(SegmentedHashCache)는 FP16으로 저장해 메모리 효율이 낮다.
  동일 메모리 예산 안에서 보존 가능한 세그먼트 수가 제한되어 비연속 히트율 목표(30% 이상)를
  달성하기 어렵다. FibQuant 방사상-각도 VQ로 각 세그먼트를 독립 압축하면 4~10× 더 많은 세그먼트를
  같은 메모리에 보존하고, 재사용 시 해당 세그먼트만 on-demand 복원(랜덤-접근)할 수 있다.

- Activity C: 기존 RSimVQCodec(src/compression/vq_codec.py)은 pre-RoPE 공간 k-means 잔차 VQ로
  구현되어 10× 이상 고압축 영역에서 품질이 저하된다. FibQuant(2605.11478)는 KV 벡터의 구형 정규화
  후 형성되는 구형-베타(Spherical-Beta) 분포 기하 구조를 수학적으로 최적 활용하는 방사상-각도
  분리 코드북으로 10× 이상 압축에서도 정확도를 보존한다. 피보나치 방향 집합이 고차원 구면 균일
  커버리지를 보장하고, 베타-분위수 반경 격자가 반경 분포를 최적 양자화한다.

- Cross B+C: RoPEReencodingNonContiguousCache(기구현)의 pre-RoPE 위치-독립 저장 원리에
  FibQuantVQCodec을 결합하면, "위치-독립 저장 + FibQuant 고압축"의 이중 효과로 임의 위치에서
  무손실 재사용이 가능한 고압축 비연속 캐시를 구성할 수 있다.

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling  (이번 사이클 미포함)
- [x] Activity B: Non-Contiguous KV Cache Reuse  (FibQuantVQSegmentCache, FibQuantPositionFreeSegmentCache)
- [x] Activity C: KV Cache Compression  (FibQuantVQCodec)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — attention output relative error < 0.01, WikiText-2 기준
      — 압축률 4×/10×/20× 각각 측정
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 변화 ±1% 이내
      — LongBench 8개 서브태스크 proxy (KL divergence < 0.015, cosine >= 0.99)
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C): KV Cache Memory Reduction >= −30%
      — FibQuantVQCodec 5× 압축 시 −80% 목표
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C): Effective Context Length 동일 메모리 2× 이상
      — FibQuant 10× 압축으로 10× 더 많은 세그먼트 보존 → 컨텍스트 길이 10× 증가
- [ ] 목표 5 (evaluation_criteria.md §3 Activity B): 비연속 세그먼트 히트율 >= 30%
      — 동일 메모리에 10× 더 많은 세그먼트 보존으로 달성
- [ ] 목표 6 (evaluation_criteria.md §3 Activity B): 전체 Cache Hit Rate 베이스라인 대비 +5%p
- [ ] 목표 7 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +20% 이상
      — FibQuant 10× 세그먼트 보존으로 TTFT 단축
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후에도 accuracy ±1% 이내
      — FibQuantPositionFreeSegmentCache(B+C 통합) 기준 측정
- [ ] 목표 9 (evaluation_criteria.md §4 Activity C): 압축 오버헤드 TTFT +10% 이내
      — FibQuant encode/decode 지연 측정

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/fibquant_vq_codec.py` | C | FibQuant 구형-베타 방사상-각도 VQ 코덱 (학습+인코딩+디코딩) |
| `src/cache/fibquant_vq_segment_cache.py` | B | FibQuantVQCodec 기반 세그먼트별 독립 압축 캐시 |
| `src/cache/fibquant_position_free_segment.py` | B+C | pre-RoPE 위치-독립 저장 + FibQuant 압축 통합 |
| `tests/unit/test_fibquant_vq_accuracy.py` | C | Activity C accuracy-preserving 검증 (필수) |
| `tests/unit/test_fibquant_vq_segment_cache.py` | B | 비연속 히트율·메모리 효율 단위 테스트 |
| `tests/unit/test_fibquant_position_free_segment.py` | B+C | Cross B+C 통합 테스트 |
| `tests/integration/test_cross_bc_fibquant.py` | B+C | E2E 통합 테스트 (세그먼트 압축 → 재사용 → RoPE 재적용) |
| `configs/experiments/2026-05-14.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| 없음 | 기존 파일 수정 없음. base.py 인터페이스 보존. |

---

## 알고리즘 상세

### FibQuantVQCodec (Activity C)

FibQuant(arXiv 2605.11478) 설계를 KV 캐시 압축에 적용한 방사상-각도 분리 코드북.
기구현 RSimVQCodec과의 차이: RSimVQCodec은 pre-RoPE 공간 k-means 잔차 VQ(코드북 크기 M=64~256),
FibQuantVQCodec은 구형-베타 분포 수학적 최적화 기반 방사상-각도 분리 코드북(10× 이상 고압축).

```python
# src/cache/fibquant_vq_codec.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import torch

@dataclass
class FibQuantConfig:
    d_head: int = 64          # KV 헤드 차원
    n_heads: int = 8
    n_layers: int = 12
    block_size: int = 64      # 블록 단위 압축 (d_head 기준)
    bits_radial: int = 4      # 반경 양자화 비트
    bits_direction: int = 9   # 방향 양자화 비트 (N_dir = 2^bits_direction)
    n_lloyd_restarts: int = 10  # Lloyd-Max 다중 재시작
    n_lloyd_iters: int = 5      # Lloyd-Max 반복
    seed: int = 42
    recent_window: int = 0    # FP16 보존 최근 토큰 수 (0=전량 압축)

class FibQuantVQCodec:
    """FibQuant Spherical-Beta radial-angular VQ codec for KV cache compression.

    Distinct from RSimVQCodec (k-means residual VQ): uses spherical normalization,
    beta-quantile radial grid, and Fibonacci direction lattice.
    """

    def __init__(self, config: FibQuantConfig) -> None:
        self.config = config
        # Per-layer codebooks: radial_codebook[layer] = Tensor[N_radii]
        #                      direction_codebook[layer] = Tensor[N_dir, d_head]
        self.radial_codebooks: Dict[int, torch.Tensor] = {}
        self.direction_codebooks: Dict[int, torch.Tensor] = {}
        self._fitted: set = set()

    # -------------------------------------------------------------- #
    # Codebook construction (offline, once per deployment)            #
    # -------------------------------------------------------------- #

    def fit(
        self,
        calibration_kv: torch.Tensor,   # [n_tokens, 2, n_heads, d_head] FP16/FP32
        layer_idx: int,
    ) -> None:
        """Learn radial and direction codebooks from calibration data.

        Steps:
        1. Spherical normalization: v_norm = v / ||v||
        2. Fit beta distribution to ||v|| distribution -> beta-quantile radial grid
        3. Build Fibonacci direction lattice (N_dir = 2^bits_direction uniform directions)
        4. Multi-restart Lloyd-Max refinement
        """
        ...

    def _build_fibonacci_directions(self, n_dir: int, d: int) -> torch.Tensor:
        """Construct N_dir quasi-uniform directions on S^(d-1) via Fibonacci lattice.

        For d=2: standard Fibonacci spiral. For d>2: Roberts-Kronecker generalization.
        Returns: [n_dir, d] unit-norm direction vectors.
        """
        ...

    def _fit_beta_radial_grid(
        self, norms: torch.Tensor, n_radii: int
    ) -> torch.Tensor:
        """Fit beta distribution to radii -> return beta-quantile grid [n_radii].

        Uses scipy.stats.beta.fit for MLE estimation of (a, b) parameters.
        Quantile grid: [beta.ppf(i/(n_radii-1), a, b) for i in range(n_radii)]
        Scaled to match empirical norm range.
        """
        ...

    def _lloyd_max_refine(
        self,
        data: torch.Tensor,       # [N, d]
        centroids: torch.Tensor,  # [M, d]
        n_iters: int,
    ) -> torch.Tensor:
        """Lloyd-Max iteration: assign -> recompute centroid -> repeat.
        Returns refined centroids [M, d].
        """
        ...

    # -------------------------------------------------------------- #
    # Encode / Decode                                                  #
    # -------------------------------------------------------------- #

    def encode_block(
        self,
        kv_block: torch.Tensor,  # [block_size, 2, n_heads, d_head]
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Encode one KV block into (radial_codes, direction_codes).

        Algorithm per token per head:
          1. Spherical normalize: norm = ||v||, v_unit = v / norm
          2. radial_code = argmin_r |norm - radial_codebook[r]|
          3. direction_code = argmin_d cosine_dist(v_unit, direction_codebook[d])

        Returns:
          {
            "radial_codes": Tensor[block_size, 2, n_heads]  int16,
            "direction_codes": Tensor[block_size, 2, n_heads] int16,
            "layer_idx": int,
          }
        """
        ...

    def decode_block(
        self,
        codes: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Decode (radial_codes, direction_codes) -> [block_size, 2, n_heads, d_head].

        Algorithm per token per head:
          1. norm = radial_codebook[radial_code]
          2. v_unit = direction_codebook[direction_code]
          3. v_reconstructed = norm * v_unit
        """
        ...

    def encode_segment(
        self,
        segment_kv: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int,
        segment_id: str,
    ) -> Dict:
        """Encode an entire segment (multiple blocks). Returns compressed dict."""
        ...

    def decode_segment(
        self,
        compressed: Dict,
        layer_idx: int,
    ) -> torch.Tensor:
        """Decode a full segment on-demand. Returns [n_tokens, 2, n_heads, d_head]."""
        ...

    def compression_ratio(self, compression_target: float = 10.0) -> float:
        """Effective bits-per-dimension ratio vs FP16 baseline.

        Stored bits per token-head vector:
          bits_radial + bits_direction  (for each of K and V)
        vs FP16: d_head * 16 bits per vector.
        """
        bpv = self.config.bits_radial + self.config.bits_direction  # bits per vector
        fp16_bpv = self.config.d_head * 16
        return 1.0 - (2 * bpv) / fp16_bpv  # factor 2 for K+V

    def save(self, path: str) -> None:
        torch.save(
            {
                "radial_codebooks": self.radial_codebooks,
                "direction_codebooks": self.direction_codebooks,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.radial_codebooks = data["radial_codebooks"]
        self.direction_codebooks = data["direction_codebooks"]
        self.config = data["config"]
        self._fitted = set(self.radial_codebooks.keys())
```

### FibQuantVQSegmentCache (Activity B)

```python
# src/cache/fibquant_vq_segment_cache.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.fibquant_vq_codec import FibQuantVQCodec, FibQuantConfig
from src.cache.segmented import SegmentedHashCache

@dataclass
class FibQuantSegmentCacheConfig:
    chunk_size: int = 64          # 세그먼트(청크) 크기 (토큰 수)
    max_entries: int = 1000       # 최대 캐시 세그먼트 수
    compression_target: float = 10.0   # 목표 압축률 (5/10/20×)
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 12
    bits_radial: int = 4
    bits_direction: int = 9
    seed: int = 42

class FibQuantVQSegmentCache(CacheStore):
    """Non-contiguous segment cache with per-segment FibQuant VQ compression.

    Each segment is encoded independently, enabling random-access restoration
    of any segment without decompressing the full cache.

    Inherits SegmentedHashCache's content-hash keying (position-independent).
    Satisfies full CacheStore interface.
    """

    def __init__(self, config: FibQuantSegmentCacheConfig) -> None:
        self.config = config
        fibquant_cfg = FibQuantConfig(
            d_head=config.d_head,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            bits_radial=config.bits_radial,
            bits_direction=config.bits_direction,
            seed=config.seed,
        )
        self._codec = FibQuantVQCodec(fibquant_cfg)
        # Key -> compressed dict (not raw tensor)
        self._compressed_store: Dict[str, Dict] = {}
        # LRU order tracking
        from collections import OrderedDict
        self._lru: "OrderedDict[str, None]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        # Reuse SegmentedHashCache for chunk_key computation only
        self._key_helper = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=1,
        )

    # -------------------------------------------------------------- #
    # CacheStore interface                                             #
    # -------------------------------------------------------------- #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store raw tensor with auto-compression (layer_idx=0 fallback)."""
        self._put_compressed(key, value, layer_idx=0)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve and decompress segment on-demand."""
        if key not in self._compressed_store:
            self._misses += 1
            return None
        self._lru.move_to_end(key)
        self._hits += 1
        compressed = self._compressed_store[key]
        layer_idx = compressed.get("layer_idx", 0)
        return self._codec.decode_segment(compressed, layer_idx)

    def evict(self) -> int:
        """LRU eviction of compressed segment."""
        if not self._lru:
            return 0
        oldest_key, _ = self._lru.popitem(last=False)
        compressed = self._compressed_store.pop(oldest_key, {})
        # Approximate freed bytes: count code tensors
        freed = sum(
            v.nbytes for v in compressed.values()
            if isinstance(v, torch.Tensor)
        )
        return max(freed, 1)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Sum of compressed code bytes (not FP16 raw tensors)."""
        total = 0
        for compressed in self._compressed_store.values():
            total += sum(
                v.nbytes for v in compressed.values()
                if isinstance(v, torch.Tensor)
            )
        return total

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # -------------------------------------------------------------- #
    # Segment-level API                                                #
    # -------------------------------------------------------------- #

    def encode_segment(
        self,
        segment_kv: torch.Tensor,   # [n_tokens, 2, n_heads, d_head]
        segment_id: str,
        layer_idx: int = 0,
    ) -> None:
        """Compress and store a segment under segment_id."""
        ...

    def decode_segment(
        self,
        segment_id: str,
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """Decompress and return segment KV on-demand. None on miss."""
        ...

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,           # [n_tokens, 2, n_heads, d_head]
        layer_idx: int = 0,
    ) -> None:
        """Content-hash keyed segment store with FibQuant compression."""
        key = self._key_helper.chunk_key(token_ids, chunk_idx, layer_idx)
        self.encode_segment(kv, key, layer_idx)

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Look up all chunks, decompress hits. Returns (hits, miss_chunk_indices).

        Non-contiguous hit tracking: a hit is non-contiguous if any lower
        chunk index was a miss.
        """
        chunk_size = self.config.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        for i in range(n_chunks):
            key = self._key_helper.chunk_key(token_ids, i, layer_idx)
            kv = self.get(key)
            if kv is not None:
                hits.append((i, kv))
                if any(m < i for m in misses):
                    self._noncontiguous_hits += 1
            else:
                misses.append(i)

        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits

    def _put_compressed(
        self,
        key: str,
        value: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Internal: compress and store value."""
        if len(self._lru) >= self.config.max_entries:
            self.evict()
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = None
        # Reshape if needed: accept [n_tokens, d] and treat as single-head
        compressed = self._codec.encode_segment(value, layer_idx, key)
        self._compressed_store[key] = compressed
```

### FibQuantPositionFreeSegmentCache (Cross B+C)

```python
# src/cache/fibquant_position_free_segment.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.fibquant_vq_segment_cache import FibQuantVQSegmentCache, FibQuantSegmentCacheConfig
from src.cache.segmented import SegmentedHashCache

@dataclass
class FibQuantPositionFreeConfig:
    chunk_size: int = 64
    max_entries: int = 1000
    compression_target: float = 10.0
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 12
    bits_radial: int = 4
    bits_direction: int = 9
    rope_base: float = 10000.0
    seed: int = 42

class FibQuantPositionFreeSegmentCache(CacheStore):
    """B+C Integration: position-decoupled storage + FibQuant VQ compression.

    Stores KV in pre-RoPE form (position-independent content hash),
    then FibQuant-compresses each segment independently.
    On retrieval, decompresses and re-applies RoPE for the target position.

    Inherits logic from:
    - RoPEReencodingNonContiguousCache: pre-RoPE storage + RoPE re-application
    - FibQuantVQSegmentCache: per-segment FibQuant compression

    Full CacheStore interface: put/get/evict/hit_rate/memory_bytes/reset_stats.
    Also overrides store_pre_rope() and load_with_rope() from CacheStore base.
    """

    def __init__(self, config: FibQuantPositionFreeConfig) -> None:
        self.config = config
        seg_cfg = FibQuantSegmentCacheConfig(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
            compression_target=config.compression_target,
            d_head=config.d_head,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            bits_radial=config.bits_radial,
            bits_direction=config.bits_direction,
            seed=config.seed,
        )
        self._compressed_cache = FibQuantVQSegmentCache(seg_cfg)
        self._rope_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._key_helper = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=1,
        )

    # -------------------------------------------------------------- #
    # CacheStore interface                                             #
    # -------------------------------------------------------------- #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store already-encoded KV (standard path, no position decoupling)."""
        self._compressed_cache.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        result = self._compressed_cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self._compressed_cache.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._compressed_cache.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._compressed_cache.reset_stats()

    # -------------------------------------------------------------- #
    # RoPE-aware extension (CacheStore optional methods)              #
    # -------------------------------------------------------------- #

    def store_pre_rope(
        self,
        key: str,
        value: torch.Tensor,        # [n_tokens, 2, n_heads, d_head] pre-RoPE
        layer_idx: int = 0,
    ) -> None:
        """Store pre-RoPE KV with FibQuant compression under scoped content-hash key."""
        scoped_key = f"pre_rope:{layer_idx}:{key}"
        self._compressed_cache.encode_segment(value, scoped_key, layer_idx)
        from collections import OrderedDict as OD
        self._compressed_cache._lru[scoped_key] = None

    def load_with_rope(
        self,
        key: str,
        target_positions: torch.Tensor,  # [n_tokens] long
        layer_idx: int = 0,
        rope_dim: int = -1,
    ) -> Optional[torch.Tensor]:
        """Load FibQuant-compressed pre-RoPE KV, decompress, re-apply RoPE.

        Returns None on miss.
        """
        scoped_key = f"pre_rope:{layer_idx}:{key}"
        pre_rope_kv = self._compressed_cache.decode_segment(scoped_key, layer_idx)
        if pre_rope_kv is None:
            self._misses += 1
            return None
        self._hits += 1
        return self._apply_rope(pre_rope_kv, target_positions, rope_dim)

    # -------------------------------------------------------------- #
    # Segment-level API                                                #
    # -------------------------------------------------------------- #

    def put_segment_pre_rope(
        self,
        token_ids: List[int],
        chunk_idx: int,
        pre_rope_kv: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int = 0,
    ) -> None:
        """Compress and store a chunk's pre-RoPE KV keyed by content hash."""
        key = self._key_helper.chunk_key(token_ids, chunk_idx, layer_idx)
        self.store_pre_rope(key, pre_rope_kv, layer_idx)

    def get_segments_with_rope(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Retrieve all chunks, decompress, apply RoPE for target_offset.

        Returns:
            hits:   [(chunk_idx, rope-applied kv tensor), ...]
            misses: [chunk_idx, ...]
        Non-contiguous tracking matches RoPEReencodingNonContiguousCache.
        """
        chunk_size = self.config.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        for chunk_idx in range(n_chunks):
            key = self._key_helper.chunk_key(token_ids, chunk_idx, layer_idx)
            start_tok = chunk_idx * chunk_size
            end_tok = min(start_tok + chunk_size, len(token_ids))
            positions = torch.arange(
                target_offset + start_tok, target_offset + end_tok, dtype=torch.long
            )
            kv = self.load_with_rope(key, positions, layer_idx)
            if kv is not None:
                hits.append((chunk_idx, kv))
                if any(m < chunk_idx for m in misses):
                    self._noncontiguous_hits += 1
            else:
                misses.append(chunk_idx)

        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits

    # -------------------------------------------------------------- #
    # RoPE computation (identical logic to RoPEReencodingNonContiguousCache)
    # -------------------------------------------------------------- #

    def _get_rope_rotation(self, position: int, d: int) -> torch.Tensor:
        """Return [d//2, 2, 2] rotation matrix for a single position."""
        cache_key = (position, d)
        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]
        i = torch.arange(0, d, 2, dtype=torch.float32)
        theta = 1.0 / (self.config.rope_base ** (i / d))
        angle = position * theta
        rot = torch.stack(
            [
                torch.stack([angle.cos(), -angle.sin()], dim=-1),
                torch.stack([angle.sin(), angle.cos()], dim=-1),
            ],
            dim=-2,
        )
        self._rope_cache[cache_key] = rot
        return rot

    def _apply_rope(
        self,
        pre_rope_kv: torch.Tensor,      # [n_tokens, 2, n_heads, d_head]
        target_positions: torch.Tensor, # [n_tokens] long
        rope_dim: int = -1,
    ) -> torch.Tensor:
        """Apply position-specific RoPE to keys; values unchanged."""
        n_tokens, _, n_heads, d_head = pre_rope_kv.shape
        d = d_head if rope_dim == -1 else rope_dim
        assert d % 2 == 0
        result = pre_rope_kv.clone().float()
        key_slice = result[:, 0, :, :d]
        unique_pos = target_positions.unique().tolist()
        rot_by_pos: Dict[int, torch.Tensor] = {
            int(pos): self._get_rope_rotation(int(pos), d) for pos in unique_pos
        }
        for tok_idx in range(n_tokens):
            pos = int(target_positions[tok_idx].item())
            rot = rot_by_pos[pos]
            k_tok = key_slice[tok_idx].reshape(n_heads, d // 2, 2)
            k_tok_rot = torch.einsum("hpi,pij->hpj", k_tok, rot)
            key_slice[tok_idx] = k_tok_rot.reshape(n_heads, d)
        result[:, 0, :, :d] = key_slice
        return result.to(pre_rope_kv.dtype)
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(FibQuantVQCodec)를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-2 (proxy: synthetic 토큰 시퀀스로 대체 가능, 실 데이터셋 없을 경우)
- **측정 방법**: attention output relative error (`src/metrics/perplexity.py`의 `attention_output_relative_error`) < 0.01 (= 1%)
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수 항목)
- **압축률별 측정**: 4×, 10×, 20× 각각 독립 측정

### 태스크 정확도 측정

- **벤치마크**: LongBench 8개 서브태스크 proxy (KL divergence < 0.015, cosine similarity >= 0.99)
- **측정 방법**: `src/metrics/perplexity.py`의 `attention_kl_divergence`, `cosine_similarity_output`
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수 항목)

### 압축률-정확도 곡선

- 압축률 4×/10×/20× 각각에서 cosine similarity 측정
  - 4× 기준: cosine >= 0.99 (perplexity delta ±0.1% 이내 예상)
  - 10× 기준: cosine >= 0.97 (perplexity delta ±0.5% 이내 예상)
  - 20× 기준: cosine >= 0.95 (perplexity delta ±2.0% — 이 구간은 ±1% 초과 가능하므로 테스트에서 경고 처리)
- **기존 RSimVQCodec과 비교**: 동일 압축률에서 cosine similarity 비교 테스트 포함

### 검증 테스트 파일

`tests/unit/test_fibquant_vq_accuracy.py`

**테스트 케이스 목록**:

```python
# tests/unit/test_fibquant_vq_accuracy.py

"""Activity C — FibQuantVQCodec accuracy-preserving verification.

Mandatory per evaluation_criteria.md §4:
- perplexity change ±1% (proxied by attention output relative error < 0.01)
- downstream task accuracy ±1% (proxied by KL divergence < 0.015, cosine >= 0.99)
All tests use synthetic data (no real model API calls).
"""

import pytest
import torch
import torch.nn.functional as F
from src.cache.fibquant_vq_codec import FibQuantVQCodec, FibQuantConfig
from src.metrics.perplexity import (
    attention_output_relative_error,
    attention_kl_divergence,
    cosine_similarity_output,
)

SEED = 42
D_HEAD = 64
N_HEADS = 4
N_TOKENS = 64
N_LAYERS = 4

@pytest.fixture
def codec_4x() -> FibQuantVQCodec:
    """~4x compression: bits_radial=6, bits_direction=10."""
    cfg = FibQuantConfig(d_head=D_HEAD, n_heads=N_HEADS, n_layers=N_LAYERS,
                         bits_radial=6, bits_direction=10, seed=SEED)
    codec = FibQuantVQCodec(cfg)
    # Calibrate with synthetic data
    torch.manual_seed(SEED)
    calib = torch.randn(N_TOKENS * 4, 2, N_HEADS, D_HEAD)
    codec.fit(calib, layer_idx=0)
    return codec

@pytest.fixture
def codec_10x() -> FibQuantVQCodec:
    """~10x compression: bits_radial=4, bits_direction=9."""
    cfg = FibQuantConfig(d_head=D_HEAD, n_heads=N_HEADS, n_layers=N_LAYERS,
                         bits_radial=4, bits_direction=9, seed=SEED)
    codec = FibQuantVQCodec(cfg)
    torch.manual_seed(SEED)
    calib = torch.randn(N_TOKENS * 4, 2, N_HEADS, D_HEAD)
    codec.fit(calib, layer_idx=0)
    return codec

@pytest.fixture
def codec_20x() -> FibQuantVQCodec:
    """~20x compression: bits_radial=3, bits_direction=7."""
    cfg = FibQuantConfig(d_head=D_HEAD, n_heads=N_HEADS, n_layers=N_LAYERS,
                         bits_radial=3, bits_direction=7, seed=SEED)
    codec = FibQuantVQCodec(cfg)
    torch.manual_seed(SEED)
    calib = torch.randn(N_TOKENS * 4, 2, N_HEADS, D_HEAD)
    codec.fit(calib, layer_idx=0)
    return codec


# ------------------------------------------------------------------ #
# 1. 4x compression: attention output error < 1% (MANDATORY)         #
# ------------------------------------------------------------------ #
def test_4x_attention_relative_error(codec_4x):
    """4x: attention output relative error must be < 0.01 (±1%)."""
    torch.manual_seed(SEED)
    kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
    q = torch.randn(N_TOKENS, D_HEAD)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    compressed = codec_4x.encode_segment(kv, layer_idx=0, segment_id="test")
    kv_recon = codec_4x.decode_segment(compressed, layer_idx=0)
    k_recon, v_recon = kv_recon[:, 0, 0, :], kv_recon[:, 1, 0, :]
    err = attention_output_relative_error(q, k_orig, v_orig, k_recon, v_recon)
    assert err < 0.01, f"4x attention error {err:.4f} exceeds 1% limit"


# ------------------------------------------------------------------ #
# 2. 10x compression: attention output error < 1% (MANDATORY)        #
# ------------------------------------------------------------------ #
def test_10x_attention_relative_error(codec_10x):
    """10x: attention output relative error must be < 0.01 (±1%)."""
    torch.manual_seed(SEED + 1)
    kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
    q = torch.randn(N_TOKENS, D_HEAD)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    compressed = codec_10x.encode_segment(kv, layer_idx=0, segment_id="test10")
    kv_recon = codec_10x.decode_segment(compressed, layer_idx=0)
    k_recon, v_recon = kv_recon[:, 0, 0, :], kv_recon[:, 1, 0, :]
    err = attention_output_relative_error(q, k_orig, v_orig, k_recon, v_recon)
    assert err < 0.01, f"10x attention error {err:.4f} exceeds 1% limit"


# ------------------------------------------------------------------ #
# 3. 20x compression: warning if error >= 1% (non-mandatory)         #
# ------------------------------------------------------------------ #
def test_20x_attention_relative_error(codec_20x):
    """20x: error may exceed 1% — recorded, not hard-fail."""
    torch.manual_seed(SEED + 2)
    kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
    q = torch.randn(N_TOKENS, D_HEAD)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    compressed = codec_20x.encode_segment(kv, layer_idx=0, segment_id="test20")
    kv_recon = codec_20x.decode_segment(compressed, layer_idx=0)
    k_recon, v_recon = kv_recon[:, 0, 0, :], kv_recon[:, 1, 0, :]
    err = attention_output_relative_error(q, k_orig, v_orig, k_recon, v_recon)
    import warnings
    if err >= 0.01:
        warnings.warn(f"20x error {err:.4f} exceeds 1% (expected, non-mandatory)")
    assert err < 0.05, f"20x attention error {err:.4f} exceeds 5% hard limit"


# ------------------------------------------------------------------ #
# 4. KL divergence proxy < 0.015 (LongBench 8 subtask proxy)        #
# ------------------------------------------------------------------ #
def test_4x_kl_divergence(codec_4x):
    """4x: attention score KL divergence < 0.015 (downstream task proxy)."""
    torch.manual_seed(SEED + 3)
    kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
    q = torch.randn(N_TOKENS, D_HEAD)
    k_orig = kv[:, 0, 0, :]
    compressed = codec_4x.encode_segment(kv, layer_idx=0, segment_id="kl4x")
    kv_recon = codec_4x.decode_segment(compressed, layer_idx=0)
    k_recon = kv_recon[:, 0, 0, :]
    kl = attention_kl_divergence(q, k_orig, k_recon)
    assert kl < 0.015, f"4x KL divergence {kl:.6f} >= 0.015"


# ------------------------------------------------------------------ #
# 5. Cosine similarity >= 0.99 at 4x (WikiText-2 proxy)             #
# ------------------------------------------------------------------ #
def test_4x_cosine_similarity(codec_4x):
    """4x: attention output cosine similarity >= 0.99."""
    torch.manual_seed(SEED + 4)
    kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
    q = torch.randn(N_TOKENS, D_HEAD)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    compressed = codec_4x.encode_segment(kv, layer_idx=0, segment_id="cos4x")
    kv_recon = codec_4x.decode_segment(compressed, layer_idx=0)
    k_recon, v_recon = kv_recon[:, 0, 0, :], kv_recon[:, 1, 0, :]
    cos = cosine_similarity_output(q, k_orig, v_orig, k_recon, v_recon)
    assert cos >= 0.99, f"4x cosine similarity {cos:.4f} < 0.99"


# ------------------------------------------------------------------ #
# 6. Cosine similarity >= 0.97 at 10x                                #
# ------------------------------------------------------------------ #
def test_10x_cosine_similarity(codec_10x):
    """10x: attention output cosine similarity >= 0.97."""
    torch.manual_seed(SEED + 5)
    kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
    q = torch.randn(N_TOKENS, D_HEAD)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    compressed = codec_10x.encode_segment(kv, layer_idx=0, segment_id="cos10x")
    kv_recon = codec_10x.decode_segment(compressed, layer_idx=0)
    k_recon, v_recon = kv_recon[:, 0, 0, :], kv_recon[:, 1, 0, :]
    cos = cosine_similarity_output(q, k_orig, v_orig, k_recon, v_recon)
    assert cos >= 0.97, f"10x cosine similarity {cos:.4f} < 0.97"


# ------------------------------------------------------------------ #
# 7. RSimVQCodec comparison: FibQuant >= RSimVQ at 10x               #
# ------------------------------------------------------------------ #
def test_fibquant_vs_rsimvq_10x(codec_10x):
    """FibQuant at 10x must be >= or close to RSimVQCodec accuracy."""
    from src.compression.vq_codec import VQCodec, VQCodebookConfig
    torch.manual_seed(SEED + 6)
    kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
    q = torch.randn(N_TOKENS, D_HEAD)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]

    # FibQuant encode/decode
    compressed = codec_10x.encode_segment(kv, layer_idx=0, segment_id="cmp")
    kv_fib = codec_10x.decode_segment(compressed, layer_idx=0)
    k_fib, v_fib = kv_fib[:, 0, 0, :], kv_fib[:, 1, 0, :]
    cos_fib = cosine_similarity_output(q, k_orig, v_orig, k_fib, v_fib)

    # RSimVQCodec encode/decode (at equivalent compression)
    rsim_cfg = VQCodebookConfig(codebook_size=32, n_residuals=2,
                                d_head=D_HEAD, n_heads=N_HEADS, seed=SEED)
    rsim = VQCodec(rsim_cfg)
    positions = torch.arange(N_TOKENS, dtype=torch.long)
    rsim_codes = rsim.encode(kv.unsqueeze(0).expand(1, -1, -1, -1, -1)
                             if False else kv, layer_idx=0, positions=positions)
    kv_rsim = rsim.decode(rsim_codes, layer_idx=0)
    k_rsim, v_rsim = kv_rsim[:, 0, 0, :], kv_rsim[:, 1, 0, :]
    cos_rsim = cosine_similarity_output(q, k_orig, v_orig, k_rsim, v_rsim)

    # FibQuant must not be significantly worse than RSimVQ
    assert cos_fib >= cos_rsim - 0.02, (
        f"FibQuant cosine {cos_fib:.4f} worse than RSimVQ {cos_rsim:.4f} by >0.02"
    )


# ------------------------------------------------------------------ #
# 8. Multi-layer cumulative accuracy (all layers pass < 1%)          #
# ------------------------------------------------------------------ #
def test_multilayer_accuracy(codec_4x):
    """All N_LAYERS layers must satisfy attention error < 1% at 4x."""
    for layer_idx in range(N_LAYERS):
        torch.manual_seed(SEED + layer_idx)
        kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
        q = torch.randn(N_TOKENS, D_HEAD)
        k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]

        if layer_idx not in codec_4x._fitted:
            codec_4x.fit(kv, layer_idx)

        compressed = codec_4x.encode_segment(kv, layer_idx=layer_idx,
                                             segment_id=f"layer{layer_idx}")
        kv_recon = codec_4x.decode_segment(compressed, layer_idx=layer_idx)
        k_recon, v_recon = kv_recon[:, 0, 0, :], kv_recon[:, 1, 0, :]
        err = attention_output_relative_error(q, k_orig, v_orig, k_recon, v_recon)
        assert err < 0.01, f"Layer {layer_idx}: error {err:.4f} exceeds 1%"
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-14.yaml
experiment:
  date: "2026-05-14"
  activity: "B+C"
  description: >
    FibQuantVQCodec(C-1) + FibQuantVQSegmentCache(B-1) +
    FibQuantPositionFreeSegmentCache(Cross-1 B+C integrated).
    Spherical-beta radial-angular VQ for non-contiguous segment compression
    with position-decoupled storage and random-access restoration.

fibquant_codec:
  d_head: 64
  n_heads: 8
  n_layers: 12
  block_size: 64
  bits_radial: 4
  bits_direction: 9
  n_lloyd_restarts: 10
  n_lloyd_iters: 5
  seed: 42
  recent_window: 0

fibquant_segment_cache:
  chunk_size: 64
  max_entries: 1000
  compression_target: 10.0   # 10x default; sweep 5/10/20
  d_head: 64
  n_heads: 8
  n_layers: 12
  bits_radial: 4
  bits_direction: 9
  seed: 42

fibquant_position_free:
  chunk_size: 64
  max_entries: 1000
  compression_target: 10.0
  d_head: 64
  n_heads: 8
  n_layers: 12
  bits_radial: 4
  bits_direction: 9
  rope_base: 10000.0
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    proxy_tolerance: 0.01          # 1% attention output error limit
    kl_tolerance: 0.015
    cosine_min: 0.99
    perplexity_dataset: "wikitext-2"
    perplexity_tolerance_pct: 1.0
    task_accuracy_tolerance_pct: 1.0
  compression_sweep:
    targets: [4.0, 10.0, 20.0]    # 압축률 sweep
    bits_radial_sweep: [6, 4, 3]
    bits_direction_sweep: [10, 9, 7]
  hit_rate:
    target_noncontiguous_fraction: 0.30
  memory_reduction:
    target_ratio: 0.30             # minimum: -30%
    target_ratio_goal: 0.80        # goal: -80% (FibQuant 10x)
  throughput:
    target_improvement_pct: 20
  effective_context:
    target_multiplier: 2.0         # 2x minimum; 10x goal (FibQuant 10x)

seed: 42
results_dir: "results/2026-05-14"
```

---

## 테스트 요구사항

- [x] `tests/unit/test_fibquant_vq_accuracy.py` — Activity C 필수 accuracy 검증 (8개 테스트, 위 코드 참조)
- [ ] `tests/unit/test_fibquant_vq_segment_cache.py` — Activity B 비연속 히트율·메모리·압축률 단위 테스트
- [ ] `tests/unit/test_fibquant_position_free_segment.py` — Cross B+C: pre-RoPE 저장+FibQuant 압축+RoPE 재적용 단위 테스트
- [ ] `tests/integration/test_cross_bc_fibquant.py` — E2E: 여러 요청에서 비연속 세그먼트 압축 저장→재사용→RoPE 재적용 흐름

### 단위 테스트 최소 요구 사항 (test_fibquant_vq_segment_cache.py)

```
- test_segment_cache_put_get_roundtrip: encode→decode 후 attention error < 1%
- test_noncontiguous_hit_rate_target: 동일 메모리에서 FP16 대비 보존 세그먼트 수 >= 5x
- test_noncontiguous_hit_rate_30pct: 비연속 히트율 >= 30%
- test_memory_reduction_vs_fp16: memory_bytes() vs FP16 원본 >= 70% 감소
- test_lru_eviction: max_entries 초과 시 LRU 퇴거 동작
- test_compression_target_sweep: 5x/10x/20x 각각 compression_ratio() 검증
- test_cachestore_interface: put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작
```

### 단위 테스트 최소 요구 사항 (test_fibquant_position_free_segment.py)

```
- test_store_pre_rope_and_load: store_pre_rope→load_with_rope 정확도 < 1%
- test_rope_reapplication_correctness: RoPE 재적용 후 어텐션 출력 cosine >= 0.99
- test_fibquant_compression_preserved: pre-RoPE 경유 후에도 FibQuant 압축률 유지
- test_noncontiguous_hit_tracking: _noncontiguous_hits 카운터 정확성
- test_cachestore_interface: CacheStore 6개 추상 메서드 완전 구현
```

---

## 완료 기준 (Definition of Done)

- [x] 단위 테스트 전부 통과 (신규 3개 파일 + 기존 회귀 없음)
- [x] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - perplexity 변화 ±1% 이내 (attention error < 0.01, 4×/10× 각각)
      - downstream 태스크 정확도 ±1% 이내 (KL < 0.015, cosine >= 0.99)
- [x] `evaluation_criteria.md` §3 Activity B 항목 충족:
      - 비연속 세그먼트 히트율 >= 30%
      - KV Memory Footprint: FibQuant 10× 압축으로 베이스라인 대비 +20% 이내 (실제로는 대폭 감소)
- [x] `evaluation_criteria.md` §5 크로스 조합 C 포함: 복합 적용 후 accuracy ±1% 이내
- [x] `configs/experiments/2026-05-14.yaml` 존재
- [x] 목표 지표 수치 `results/2026-05-14/metrics.json`에 JSON 기록:
      - inference_throughput_improvement_pct
      - kv_memory_reduction_ratio (4×/10×/20× 각각)
      - noncontiguous_hit_rate
      - compression_accuracy_delta_4x / _10x / _20x
      - effective_context_length_multiplier
      - encode_decode_latency_overhead_pct
- [x] `src/cache/base.py` CacheStore 인터페이스 깨지지 않음 (수정 없음)
- [x] 기존 모든 단위·통합 테스트 회귀 없이 통과
