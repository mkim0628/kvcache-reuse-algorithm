<!-- 변경 이유 (이전 Spec.md: 2026-05-09 대비):
이전 사이클(2026-05-09)은 A+B (PPDAppendPrefillRouter + TriangleInequalitySegmentIndex +
HitAwarePPDRouter + SemanticBoundarySegmentCache + SpecKVCompressionGammaController)
조합이었다. 이번 사이클은 B+C (KVPacketSoftAdapterCache + RSimVQCodec +
ContextFreeCompressedKVPacket) 조합으로 전환한다.

주요 변경:
1. [초점 전환] A+B → B+C. 이전 두 사이클(05-08 A+C, 05-09 A+B)에서 A 조합에
   집중했으므로 이번 사이클은 B+C 통합에 초점.

2. [Activity B 교체] TriangleInequalitySegmentIndex(O(log N) 계층 인덱스) →
   KVPacketSoftAdapterCache(소프트-토큰 어댑터 기반 재계산-프리 맥락 독립 패킷).
   이전 모든 B 기법의 "선택적 재계산 의존" 한계를 "재계산 FLOPs 0%"로 돌파.
   KV Packet(arXiv 2604.13226) 기반.

3. [Activity C 신규] RSimVQCodec (pre-RoPE 공간 잔차 벡터 양자화 + 훈련 불필요).
   스칼라 양자화(eOptShrinkQ, TurboQuant)와 직교하는 VQ 압축 패러다임.
   VQKV(arXiv 2603.16435) 기반. 82.8% 압축 + LongBench 98.6% 성능.

4. [Cross-1 신규] ContextFreeCompressedKVPacket: B(어댑터 패킷) + C(VQ 압축) 통합.
   TriangleInequalitySegmentIndex(05-09 기구현)를 세그먼트 인덱스 백엔드로 재사용.

5. [compression 모듈 신설] src/compression/ 패키지 생성.
   VQCodec을 cache 모듈과 분리해 독립 재사용 가능하게 설계.

6. [보존 파일] 이전 사이클 파일(ppd_append_prefill_router.py,
   hit_aware_ppd_router.py, semantic_boundary_cache.py, triangle_index.py,
   speckv_gamma_controller.py, context_intensive_guard.py, eopt_shrinkq_codec.py,
   static_dynamic_segment.py, 기타 모든 기존 파일)은 이번 사이클에서 수정하지 않는다.
   기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-10

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-10.md`
**최우선 구현 타겟**: Cross-1 (B+C) — KVPacketSoftAdapterCache(B-1) + RSimVQCodec(C-1) +
ContextFreeCompressedKVPacket 통합

**해결하려는 문제**:
- 비연속 KV 세그먼트 재사용의 근본 한계: 어텐션 분포가 맥락에 따라 달라지는 "맥락 의존성"으로
  인해 이전 모든 B 기법이 선택적 재계산에 의존했다. KV Packet(arXiv 2604.13226)의 소프트-토큰
  어댑터로 맥락 의존성을 제거해 재계산 FLOPs 0%로 비연속 재사용을 실현한다.
- KV 캐시 압축에서 기존 스칼라 양자화(eOptShrinkQ, TurboQuant)와 저랭크 근사가 동시에
  높은 압축률과 재구성 충실도를 달성하지 못하는 한계를, VQKV(arXiv 2603.16435)의
  Residual Simple VQ(RSimVQ) + pre-RoPE 공간 코드북으로 해소한다. 82.8% 압축(5.8x)
  + LongBench 98.6% 성능 유지.
- B+C 통합: 맥락 독립 어댑터 패킷에 VQ 압축을 적용해 "재계산 없는 + 고압축" 비연속 KV 세그먼트
  저장·재사용 파이프라인을 구성한다. 처리량 +40~55%, 메모리 -82~88%, 비연속 히트율 +30~40%p 목표.

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling
- [x] Activity B: Non-Contiguous KV Cache Reuse
- [x] Activity C: KV Cache Compression

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §3 Activity B): 비연속 세그먼트 히트율 >= 30% (전체 히트 중 비연속 구간 발생 비율)
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C): KV 메모리 감소 >= -30% (베이스라인 대비); 목표치 -82%
- [ ] 목표 3 (evaluation_criteria.md §4 필수): perplexity 변화 +-1% 이내 (WikiText-2 기준)
- [ ] 목표 4 (evaluation_criteria.md §4 필수): downstream 태스크 정확도 변화 +-1% 이내 (LongBench 8개 서브태스크)
- [ ] 목표 5 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +10% 이상; 목표치 +20%
- [ ] 목표 6 (evaluation_criteria.md §5 크로스 조합): 단일 Activity 대비 복합 처리량 추가 +5% 이상
- [ ] 목표 7 (evaluation_criteria.md §5 크로스 조합): 단일 Activity 대비 복합 메모리 감소 추가 -10% 이상

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/compression/__init__.py` | C | compression 패키지 초기화. VQCodec 공개 |
| `src/compression/vq_codec.py` | C | RSimVQCodec — pre-RoPE 공간 잔차 VQ 인코더/디코더 + 코드북 학습 |
| `src/cache/kv_packet_adapter.py` | B | KVPacketSoftAdapterCache — 소프트-토큰 어댑터 래핑 비연속 KV 패킷 캐시 |
| `src/cache/context_free_compressed_packet.py` | B+C | ContextFreeCompressedKVPacket — B 어댑터 패킷 + C VQ 압축 통합 캐시 |
| `tests/unit/test_vq_codec.py` | C | VQCodec 단위 테스트 (코드북 학습, encode/decode 왕복, perplexity delta) |
| `tests/unit/test_kv_packet_adapter.py` | B | KVPacketSoftAdapterCache 단위 테스트 (어댑터 훈련, 재계산-프리 F1 동등성) |
| `tests/unit/test_context_free_compressed.py` | B+C | ContextFreeCompressedKVPacket 단위 테스트 (B+C 통합 히트율 + 압축률) |
| `tests/integration/test_cross_bc_packet_vq.py` | B+C | 통합 E2E 테스트 (처리량 측정, 압축 정확도 유지 검증) |
| `configs/experiments/2026-05-10.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/base.py` | `compression_hook` 선택적 메서드 추가 (기본 구현 제공, 기존 서브클래스 깨지 않음) |
| `src/cache/segmented.py` | `put_segment` / `get_segments` 시그니처에 `codec: Optional[VQCodec] = None` 파라미터 추가 |
| `tests/unit/test_segmented_cache.py` | 기존 테스트 보존 + VQCodec 연동 시 hit rate 회귀 없음 검증 케이스 추가 |

---

## 알고리즘 상세

### [VQCodec / RSimVQCodec] (Activity C) — `src/compression/vq_codec.py`

pre-RoPE 공간에서 잔차 벡터 양자화(Residual Simple VQ)를 수행한다.
Key 캐시는 RoPE 적용 이후 저장되므로, 역 RoPE 변환으로 pre-RoPE 표현을 복원한 뒤
k-means 코드북으로 인코딩한다. Value 캐시는 RoPE가 없으므로 직접 인코딩한다.

**데이터 흐름 (encode)**:
```
k_post_rope [n_tokens, n_heads, d_head]
  → inverse_rope(positions) → k_pre_rope
  → RSimVQ encode (n_residuals stages):
      code_0 = argmin_j ||k_pre_rope - codebook[j]||
      residual_1 = k_pre_rope - codebook_0[code_0]
      code_1 = argmin_j ||residual_1 - codebook_1[j]||
      ...
  → key_codes [n_tokens, n_heads, n_residuals] int16

v [n_tokens, n_heads, d_head]
  → RSimVQ encode (no RoPE step needed)
  → val_codes [n_tokens, n_heads, n_residuals] int16
```

**데이터 흐름 (decode)**:
```
key_codes [n_tokens, n_heads, n_residuals]
  → k_pre_rope = sum(codebook_r[code_r] for r in range(n_residuals))
  → apply_rope(positions) → k_post_rope [n_tokens, n_heads, d_head]

val_codes → v = sum(val_codebook_r[code_r] for r)
```

```python
# 의사코드 — src/compression/vq_codec.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class VQCodebookConfig:
    codebook_size: int = 256      # M: 코드워드 수 (8비트 인덱스)
    n_residuals: int = 4          # 잔차 VQ 단계 수
    d_head: int = 128             # head 차원
    n_layers: int = 32            # 모델 레이어 수
    n_heads: int = 8              # KV head 수
    max_iter_kmeans: int = 100    # k-means 최대 반복 횟수
    rope_base: int = 10000        # RoPE 주파수 base
    seed: int = 42


class VQCodec:
    """Training-free Residual Simple VQ codec (VQKV arXiv 2603.16435).

    코드북은 fit()으로 한 번만 학습하고 이후 모든 캐시에 재사용한다.
    key_codebooks[layer_idx][residual_idx]: Tensor [M, d_head]
    val_codebooks[layer_idx][residual_idx]: Tensor [M, d_head]
    """

    def __init__(self, config: VQCodebookConfig) -> None:
        self.config = config
        # layer x n_residuals codebooks (초기 빈 dict, fit() 후 채워짐)
        self.key_codebooks: Dict[int, List[torch.Tensor]] = {}
        self.val_codebooks: Dict[int, List[torch.Tensor]] = {}

    def fit(
        self,
        calibration_keys: torch.Tensor,  # [n_tokens, d_head] pre-RoPE key (단일 head)
        calibration_vals: torch.Tensor,  # [n_tokens, d_head] val
        layer_idx: int,
        head_idx: int = 0,
    ) -> None:
        """k-means로 코드북 학습. 레이어·head별로 호출.
        실제 구현에서는 모든 head를 병렬 처리하되, 인터페이스는 단순하게 유지."""
        # torch.manual_seed(config.seed) 로 재현성 보장
        # 잔차 코드북 n_residuals 단계:
        #   residual = calibration_keys.clone()
        #   for r in range(n_residuals):
        #     codebook_r = kmeans(residual, M, max_iter)
        #     codes_r = assign(residual, codebook_r)
        #     residual -= codebook_r[codes_r]
        #     key_codebooks[layer_idx][r] = codebook_r
        ...

    def encode(
        self,
        kv: torch.Tensor,         # [n_tokens, 2, n_heads, d_head] float16
        layer_idx: int,
        positions: torch.Tensor,  # [n_tokens] int64
    ) -> dict:
        """반환:
        {
          'key_codes': Tensor [n_tokens, n_heads, n_residuals] int16,
          'val_codes': Tensor [n_tokens, n_heads, n_residuals] int16,
          'layer_idx': int,
          'n_tokens': int,
          'positions': Tensor [n_tokens] int64,
        }"""
        # key: kv[:,0] → inverse_rope(positions) → pre-RoPE
        #      → n_residuals 단계 잔차 VQ 인코딩
        # val: kv[:,1] → n_residuals 단계 잔차 VQ 인코딩 (RoPE 없음)
        ...

    def decode(
        self,
        codes: dict,
        layer_idx: int,
    ) -> torch.Tensor:
        """반환: [n_tokens, 2, n_heads, d_head] float16
        key: 잔차 합산 → k_pre_rope → apply_rope(positions) → k_post_rope
        val: 잔차 합산 → val"""
        ...

    @staticmethod
    def inverse_rope(
        k_post: torch.Tensor,    # [n_tokens, n_heads, d_head]
        positions: torch.Tensor, # [n_tokens] int64
        base: int = 10000,
    ) -> torch.Tensor:
        """RoPE 역변환: k_post_rope → k_pre_rope.
        짝수/홀수 차원 쌍 (d0, d1)에 대해:
          k_pre_d0 = k_post_d0 * cos(theta) + k_post_d1 * sin(theta)
          k_pre_d1 = -k_post_d0 * sin(theta) + k_post_d1 * cos(theta)
        여기서 theta = pos / (base^(2i/d_head)), i = 차원 인덱스.
        검증: inverse_rope(apply_rope(k, pos), pos) == k (MSE < 1e-5)."""
        ...

    @staticmethod
    def apply_rope(
        k_pre: torch.Tensor,     # [n_tokens, n_heads, d_head]
        positions: torch.Tensor, # [n_tokens] int64
        base: int = 10000,
    ) -> torch.Tensor:
        """표준 RoPE 적용:
          k_out_d0 = k_pre_d0 * cos - k_pre_d1 * sin
          k_out_d1 = k_pre_d0 * sin + k_pre_d1 * cos"""
        ...

    def compression_ratio(self) -> float:
        """실효 압축률 계산.
        n_residuals * ceil(log2(M)) bits/원소 / 16 bits(FP16).
        M=256, n=4: 4*8/16 = 2.0 (2배 증가 = 압축률 아님, 슬라이딩 윈도우 결합 필요).
        M=256, n=2: 2*8/16 = 1.0 (동등).
        실제 82.8% 압축은 recent_window FP16 보존 + 오래된 토큰 VQ 조합으로 달성."""
        ...

    def save(self, path: str) -> None:
        """torch.save({'key_codebooks': self.key_codebooks,
                       'val_codebooks': self.val_codebooks,
                       'config': self.config}, path)"""
        ...

    def load(self, path: str) -> None:
        """torch.load로 코드북 복원."""
        ...
```

**압축률 달성 방법**: 단순 VQ 코드 인덱스(n_residuals × 8비트)만으로는 FP16 대비 압축이 없다.
실효 압축은 두 가지 조합으로 달성한다:
1. `recent_window` 슬라이딩 윈도우: 최근 N 토큰은 FP16 유지, 오래된 토큰만 VQ. 전체의 80%+ 토큰이 VQ 압축됨.
2. M=256, n_residuals=2 (16비트 등가) 설정 시 codebook decode 오류만큼 압축 이득 없음.
   M=256, n_residuals=4 설정 + int16→int8 패킹 시 50% 감소.
   실제 VQKV 구현은 4비트 인덱스(M=16) + n_residuals=4로 1비트/원소 달성.
   구현에서는 `codebook_size=16` (4비트 인덱스)를 기본으로 사용하고
   `codebook_size=256` (8비트)를 고정밀 옵션으로 제공한다.

---

### [KVPacketSoftAdapterCache] (Activity B) — `src/cache/kv_packet_adapter.py`

각 KV 블록(문서 단위)을 경량 소프트-토큰 어댑터로 래핑해 맥락 독립 패킷을 구성한다.
어댑터는 새 맥락에서 발생하는 어텐션 분포 이동을 보정한다. 재계산 FLOPs 0%.

**핵심 아이디어**: 문서 KV 블록을 원래 맥락과 무관하게 다양한 새 맥락에서 사용할 수 있도록
`SoftTokenAdapter`(소프트-토큰 파라미터)를 prepend해 어텐션 시 맥락 갭을 보완한다.

**데이터 흐름**:
```
문서 KV 블록 [n_tokens, 2, n_heads, d_head]
  → SoftTokenAdapter.adapt()
  → [adapter_rank + n_tokens, 2, n_heads, d_head]  (어댑터 소프트 토큰 prepend)
  → 다양한 맥락에서 직접 어텐션 계산 (재계산 없음)
```

```python
# 의사코드 — src/cache/kv_packet_adapter.py

import torch
import torch.nn as nn
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from src.cache.base import CacheStore


class SoftTokenAdapter(nn.Module):
    """경량 소프트-토큰 어댑터.

    크기: 2 * n_heads * rank * d_head * 2bytes (FP16).
    rank=8, n_heads=8, d_head=128: 2*8*8*128*2 = 32,768 bytes ≈ 32 KB.
    전체 KV 블록(512 토큰, 8 heads, 128 d_head): 512*2*8*128*2 = 2,097,152 bytes = 2 MB.
    어댑터 크기 비율: 32 KB / 2 MB = 1.6% (<<1% 목표에 근접).
    """

    def __init__(self, n_heads: int, d_head: int, rank: int = 8) -> None:
        super().__init__()
        self.rank = rank
        # Key 소프트 토큰 파라미터: [rank, n_heads, d_head]
        self.soft_key = nn.Parameter(torch.zeros(rank, n_heads, d_head))
        # Value 소프트 토큰 파라미터: [rank, n_heads, d_head]
        self.soft_val = nn.Parameter(torch.zeros(rank, n_heads, d_head))
        nn.init.normal_(self.soft_key, std=0.02)
        nn.init.normal_(self.soft_val, std=0.02)

    def adapt(
        self,
        kv_block: torch.Tensor,   # [n_tokens, 2, n_heads, d_head]
    ) -> torch.Tensor:
        """어댑터 소프트 토큰을 KV 블록 앞에 연결.
        반환: [rank + n_tokens, 2, n_heads, d_head]"""
        # soft_tokens: [rank, 2, n_heads, d_head]
        # torch.stack([self.soft_key, self.soft_val], dim=1)
        # torch.cat([soft_tokens, kv_block], dim=0)
        ...


@dataclass
class KVPacket:
    """맥락 독립 KV 패킷."""
    doc_id: str
    kv_block: torch.Tensor          # [n_tokens, 2, n_heads, d_head] float16
    adapter: SoftTokenAdapter       # 경량 어댑터 (훈련 전: 항등에 가까움)
    embedding: Optional[torch.Tensor] = None  # [d_embed] float32, 인덱스 검색용


class KVPacketSoftAdapterCache(CacheStore):
    """KV Packet(arXiv 2604.13226) 기반 재계산-프리 맥락 독립 비연속 KV 패킷 캐시.

    CacheStore 인터페이스를 완전 구현한다.
    put(key, value): key=doc_id, value=kv_block [n_tokens, 2, n_heads, d_head]
    get(key): 어댑터 적용 후 [rank+n_tokens, 2, n_heads, d_head] 반환
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        adapter_rank: int = 8,
        max_packets: int = 512,
        embedding_dim: int = 64,
    ) -> None:
        self.n_heads = n_heads
        self.d_head = d_head
        self.adapter_rank = adapter_rank
        self.max_packets = max_packets
        self.embedding_dim = embedding_dim
        self._store: OrderedDict[str, KVPacket] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        # 최근 접근 순서 추적 (비연속 히트 측정용)
        self._access_order: List[str] = []

    # CacheStore 인터페이스
    def put(self, key: str, value: torch.Tensor) -> None:
        """kv_block을 어댑터로 래핑해 패킷으로 저장.
        기존 key 존재 시 LRU 업데이트. 초과 시 evict() 호출."""
        ...

    def get(self, key: str) -> Optional[torch.Tensor]:
        """miss: None. hit: adapter.adapt(packet.kv_block) 반환."""
        ...

    def evict(self) -> int:
        """LRU 기반 퇴거. 반환: 해제된 bytes."""
        ...

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """패킷 kv_block bytes + 어댑터 파라미터 bytes 합산."""
        ...

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._access_order.clear()

    # Packet-level API
    def create_packet(
        self,
        doc_id: str,
        kv_block: torch.Tensor,   # [n_tokens, 2, n_heads, d_head]
        embedding: Optional[torch.Tensor] = None,
    ) -> KVPacket:
        """SoftTokenAdapter를 초기화해 KVPacket 반환. 저장은 하지 않음."""
        ...

    def train_adapter(
        self,
        packet: KVPacket,
        context_kvs: List[torch.Tensor],  # 재계산된 KV [n_tokens, 2, n_heads, d_head] x 8~16개
        n_steps: int = 1000,
        lr: float = 1e-3,
    ) -> None:
        """자기지도 증류 훈련. packet.adapter 파라미터를 in-place 갱신.
        손실: MSE(adapter.adapt(packet.kv_block)[:n_tokens], target_kv)
        각 context_kvs 원소가 하나의 훈련 샘플."""
        # optimizer = Adam(packet.adapter.parameters(), lr=lr)
        # for step in range(n_steps):
        #   target = random.choice(context_kvs)
        #   pred = adapter.adapt(packet.kv_block)[adapter_rank:]  # soft 토큰 제외
        #   loss = F.mse_loss(pred, target)
        #   optimizer.zero_grad(); loss.backward(); optimizer.step()
        ...

    def pack(
        self,
        doc_ids: List[str],
    ) -> Optional[torch.Tensor]:
        """여러 패킷을 단순 연결(concatenation). 재계산 없음.
        각 패킷: adapter.adapt(kv_block) → [rank+n_tokens, 2, n_heads, d_head]
        전체: cat along dim=0 → [sum(rank+n_tokens_i), 2, n_heads, d_head]
        miss 패킷 존재 시 None 반환."""
        ...

    def noncontiguous_hit_rate(self) -> float:
        """비연속 패킷 히트 비율 (전체 히트 중 비연속 구간에서 발생한 비율).
        비연속: 접근 순서에서 연속된 doc_id pair가 아닌 경우."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits
```

---

### [ContextFreeCompressedKVPacket] (Activity B+C) — `src/cache/context_free_compressed_packet.py`

`KVPacketSoftAdapterCache`와 `VQCodec`을 통합한 B+C 통합 캐시.
어댑터(FP16 유지) + KV 블록(VQ 압축) 형태로 저장한다.
`TriangleInequalitySegmentIndex`(05-09 기구현)를 검색 백엔드로 선택적 사용한다.

**저장 구조**:
```
CompressedPacket:
  doc_id: str
  adapter_state_dict: dict          # SoftTokenAdapter.state_dict() FP16
  kv_vq_codes: dict                 # VQCodec.encode() 결과 (오래된 토큰)
  kv_recent_fp16: Tensor            # [recent_window, 2, n_heads, d_head] FP16 (최근 토큰)
  codebook_layer_idx: int
  embedding: Tensor                 # [d_embed] 검색용
```

**재사용 흐름**:
```
CompressedPacket 조회
  → VQCodec.decode(kv_vq_codes) → kv_old [n_old_tokens, 2, n_heads, d_head]
  → cat([kv_old, kv_recent_fp16], dim=0) → kv_full [n_tokens, 2, n_heads, d_head]
  → adapter.adapt(kv_full) → [rank+n_tokens, 2, n_heads, d_head]
  → 맥락 독립 어텐션 계산 (재계산 없음)
```

```python
# 의사코드 — src/cache/context_free_compressed_packet.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from src.cache.base import CacheStore
from src.cache.kv_packet_adapter import KVPacketSoftAdapterCache, SoftTokenAdapter
from src.compression.vq_codec import VQCodec


@dataclass
class CompressedPacket:
    doc_id: str
    adapter_state_dict: dict         # SoftTokenAdapter.state_dict()
    kv_vq_codes: Optional[dict]      # VQCodec.encode() 결과. None이면 fully FP16
    kv_recent_fp16: torch.Tensor     # [min(recent_window, n_tokens), 2, n_heads, d_head]
    layer_idx: int
    positions: torch.Tensor          # [n_tokens] int64 전체 포지션
    embedding: Optional[torch.Tensor] = None


class ContextFreeCompressedKVPacket(CacheStore):
    """B+C 통합: 어댑터 래핑(B) + VQ 압축(C) 비연속 KV 패킷 캐시.

    저장: (adapter_state_dict_fp16, kv_vq_codes, kv_recent_fp16) 3-tuple.
    검색: TriangleInequalitySegmentIndex O(log N) (segment_index 제공 시).
    재사용: VQ 디코딩 → concat recent_fp16 → 어댑터 적용 → 맥락 독립 어텐션.
    """

    def __init__(
        self,
        vq_codec: VQCodec,
        n_heads: int,
        d_head: int,
        adapter_rank: int = 8,
        max_packets: int = 512,
        recent_window: int = 64,
        segment_index: Optional[object] = None,  # TriangleInequalitySegmentIndex
    ) -> None:
        self.vq_codec = vq_codec
        self.n_heads = n_heads
        self.d_head = d_head
        self.adapter_rank = adapter_rank
        self.max_packets = max_packets
        self.recent_window = recent_window
        self.segment_index = segment_index
        self._store: Dict[str, CompressedPacket] = {}
        self._lru: List[str] = []  # LRU 순서
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # CacheStore 인터페이스
    def put(self, key: str, value: torch.Tensor) -> None:
        """key=doc_id, value=kv_block [n_tokens, 2, n_heads, d_head].
        positions는 [0..n_tokens-1] 기본값 사용.
        VQ 코드북이 fit() 되어 있으면 VQ 압축, 아니면 FP16 저장."""
        positions = torch.arange(value.shape[0], dtype=torch.long)
        self.put_compressed(key, value, positions, layer_idx=0)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """VQ 디코딩 → concat → 어댑터 적용 후 반환."""
        ...

    def evict(self) -> int:
        """LRU 기반 퇴거."""
        ...

    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...

    # B+C 통합 API
    def put_compressed(
        self,
        doc_id: str,
        kv_block: torch.Tensor,      # [n_tokens, 2, n_heads, d_head]
        positions: torch.Tensor,     # [n_tokens] int64
        layer_idx: int,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        """1. kv_block[-recent_window:] → FP16 보존 (kv_recent_fp16)
           2. kv_block[:-recent_window] → VQCodec.encode() (코드북 fit 후)
           3. SoftTokenAdapter 초기화 (어댑터 파라미터 초기값)
           4. CompressedPacket 생성·저장
           5. segment_index 있으면 embedding 등록"""
        ...

    def get_decompressed(
        self,
        doc_id: str,
    ) -> Optional[torch.Tensor]:
        """VQ 디코딩 → concat recent_fp16 → adapt() → [rank+n_tokens, 2, n_heads, d_head]"""
        ...

    def search_similar(
        self,
        query_embedding: torch.Tensor,  # [d_embed]
        top_k: int = 5,
    ) -> List[str]:
        """segment_index.search(query_embedding, top_k) 위임.
        segment_index 없으면 [] 반환."""
        ...

    def pack_packets(
        self,
        doc_ids: List[str],
    ) -> Optional[torch.Tensor]:
        """여러 패킷을 단순 연결. 재계산 없음.
        반환: [sum(rank+n_tokens_i), 2, n_heads, d_head] 또는 None (miss 존재 시)"""
        ...

    def noncontiguous_hit_rate(self) -> float: ...

    def compression_ratio(self) -> float:
        """전체 저장된 패킷의 실효 압축률.
        bytes_saved = sum(p.kv_block_orig_bytes - p.vq_codes_bytes - p.recent_fp16_bytes)
        ratio = bytes_saved / total_orig_bytes"""
        ...
```

---

### [CacheStore 확장] (공통) — `src/cache/base.py`

기존 인터페이스를 깨지 않도록 `compression_hook`을 선택적 메서드로 추가한다.
기존 서브클래스는 변경이 필요 없다.

```python
# 추가할 메서드 (기본 구현 제공 — 기존 동작 유지)

def compression_hook(
    self,
    key: str,
    value: torch.Tensor,
) -> torch.Tensor:
    """Optional compression hook called before storing in put().
    Default implementation is identity (no compression).
    Subclasses with compression (Activity C) may override to compress value."""
    return value
```

---

### [SegmentedHashCache 확장] (Activity B) — `src/cache/segmented.py`

기존 API를 완전 보존하면서 VQCodec 선택적 연동 파라미터를 추가한다.
`codec=None`이 기본값이므로 기존 동작은 전혀 변경되지 않는다.

```python
# 변경할 시그니처

def put_segment(
    self,
    token_ids: List[int],
    chunk_idx: int,
    kv: torch.Tensor,
    layer_idx: int = 0,
    codec: Optional["VQCodec"] = None,          # 신규 (기본 None → 기존 동작)
    positions: Optional[torch.Tensor] = None,   # 신규 (codec 사용 시 필요)
) -> None:
    """codec 제공 시 VQ 인코딩 후 저장, None 시 기존 동작 완전 유지."""
    key = self.chunk_key(token_ids, chunk_idx, layer_idx)
    if codec is not None and positions is not None:
        kv = codec.compression_hook(key, kv)  # base.py compression_hook 사용
    self.put(key, kv)

def get_segments(
    self,
    token_ids: List[int],
    layer_idx: int = 0,
    codec: Optional["VQCodec"] = None,   # 신규 (기본 None → 기존 동작)
) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
    """codec 제공 시 VQ 디코딩 후 반환, None 시 기존 동작 완전 유지."""
    ...
```

---

## Activity C — Accuracy Preservation 검증 계획

**Activity C(RSimVQCodec)를 포함하므로 이 섹션은 필수다. 이 계획 없이 Spec.md는 불완전하다.**

### perplexity 측정

| 항목 | 내용 |
|------|------|
| 데이터셋 | WikiText-2 (test split, stride=512, max_length=2048) |
| 모델 | GPT-2 small (단위 테스트 환경에서 빠른 검증 < 60초), 선택적으로 LLaMA-3.1-8B |
| 측정 방법 | 베이스라인(FP16 KV) perplexity vs VQ 압축 KV perplexity 비교 |
| 허용 오차 | +-1% 이내 (evaluation_criteria.md §4 필수 항목) |
| 코드북 크기 스윕 | M=64, M=128, M=256 각각 측정 |
| 잔차 수 스윕 | n_residuals=1, 2, 4 각각 측정 |
| 결과 저장 | `results/2026-05-10/perplexity_sweep.json` |

### 태스크 정확도 측정

| 항목 | 내용 |
|------|------|
| 벤치마크 | LongBench 8개 서브태스크 (qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, narrativeqa, qmsum, gov_report) |
| 측정 지표 | F1 score / ROUGE-L: 베이스라인 vs VQ 압축 KV |
| 허용 오차 | +-1% 이내 (evaluation_criteria.md §4 필수 항목) |
| 어댑터 검증 | 어댑터 ON vs OFF 비교로 SoftTokenAdapter 기여도 측정 |
| 결과 저장 | `results/2026-05-10/longbench_results.json` |

### 압축률-정확도 트레이드오프 곡선

| codebook_size M | n_residuals | bits/원소 | 예상 압축률 (recent_window 50% 가정) | perplexity delta 목표 |
|----------------|-------------|---------|----------------------------------|-------------------|
| 16 (4비트) | 2 | 8비트 | ~50% | +-1% |
| 16 (4비트) | 4 | 16비트 | ~0% (동등) | +-0.5% |
| 256 (8비트) | 2 | 16비트 | ~0% (동등) | +-0.5% |
| 16 (4비트) | 2 + recent_window=128 | — | ~60~70% | +-1% |

**실효 82.8% 압축 달성 조건**: `codebook_size=16`, `n_residuals=4`, `recent_window` = 전체 길이의 10% 이하.
단위 테스트에서는 `codebook_size=16`, `n_residuals=2`, `recent_window=32`로 축소 검증.

### 검증 테스트 파일 — `tests/unit/test_vq_codec.py`

```python
# 필수 테스트 케이스

def test_codebook_fit_reproducible():
    """동일 seed, 동일 calibration 데이터 → 동일 코드북. 재현성 보장.
    torch.manual_seed(42) → fit() → codebook_1
    torch.manual_seed(42) → fit() → codebook_2
    assert torch.allclose(codebook_1, codebook_2)"""

def test_encode_decode_roundtrip_shape():
    """encode → decode 후 텐서 shape이 원본과 동일.
    kv [32, 2, 4, 64] → encode → decode → shape == [32, 2, 4, 64]"""

def test_encode_decode_mse_bounded():
    """MSE(original, decoded) / ||original||^2 < 0.1 (codebook_size=16, n_residuals=4).
    더 엄격한 threshold: codebook_size=256, n_residuals=4 시 < 0.02."""

def test_perplexity_delta_within_1pct():
    """WikiText-2 stub 데이터(1000 토큰)로 perplexity delta +-1% 검증.
    전체 LongBench는 통합 테스트에서 수행. 소형 GPT-2 사용."""
    baseline_ppl = compute_ppl_fp16(model, tokens)
    compressed_ppl = compute_ppl_vq(model, tokens, vq_codec)
    assert abs(compressed_ppl - baseline_ppl) / baseline_ppl <= 0.01

def test_compression_ratio_meets_target():
    """codebook_size=16, n_residuals=4, recent_window=32 설정에서
    실효 compression_ratio >= 0.50 (50% 이상 메모리 감소)."""

def test_inverse_rope_correctness():
    """inverse_rope(apply_rope(k, pos), pos) ≈ k.
    MSE < 1e-5 (FP32 계산 기준)."""

def test_codec_m_sweep_mse_monotone():
    """M=16, 64, 256 코드북 크기별 decode MSE 단조 감소.
    M=256이 M=16보다 MSE가 낮거나 같아야 함."""

def test_codec_n_residuals_sweep_mse_monotone():
    """n_residuals=1,2,4 별 decode MSE 단조 감소."""

def test_codec_save_load_roundtrip():
    """save() → load() 후 동일 코드북 복원."""
```

### 검증 테스트 파일 — `tests/integration/test_cross_bc_packet_vq.py`

```python
# 필수 통합 테스트 케이스

def test_bc_pipeline_noncontiguous_hit_rate():
    """ContextFreeCompressedKVPacket B+C 통합 비연속 히트율 >= 30% 검증.
    N개 문서 캐시 후 비연속 순서로 접근 → noncontiguous_hit_rate() >= 0.30"""

def test_bc_compression_accuracy_preserved():
    """B+C 통합 후 decode된 KV로 계산한 perplexity delta +-1% 이내.
    stub WikiText-2 + 소형 GPT-2 사용."""

def test_bc_memory_reduction():
    """B+C 통합 후 memory_bytes() <= 베이스라인의 70% (>= 30% 감소)."""

def test_bc_throughput_improvement():
    """pack_packets() 기반 처리량이 재계산 기준선 대비 >= +10% 향상.
    재계산 기준선: 동일 토큰 수를 forward pass로 재계산하는 시간.
    pack_packets 기준선: 캐시에서 가져와 어댑터 적용만 하는 시간."""

def test_bc_cachestore_interface_compliance():
    """ContextFreeCompressedKVPacket이 CacheStore 추상 메서드를 전부 구현함."""

def test_bc_adapter_identity_accuracy():
    """어댑터 훈련 없는 경우(항등 어댑터)에서도 KV 재사용 정확도 유지.
    adapt(kv)[:n_tokens] ≈ kv (rank 차원 제외)"""
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-10.yaml
experiment:
  date: "2026-05-10"
  activity: "B+C"
  description: >
    ContextFreeCompressedKVPacket (Cross-1: B+C):
    KVPacketSoftAdapterCache(B-1) + RSimVQCodec(C-1) 통합.
    재계산-프리 맥락 독립 비연속 KV 패킷 + VQ 압축 저장.

vq_codec:
  codebook_size: 16               # M: 4비트 인덱스 (82.8% 압축 달성용)
  n_residuals: 4                  # 잔차 VQ 단계 수
  d_head: 64                      # head 차원 (단위 테스트용 소형; 실모델 128/256으로 조정)
  n_layers: 12                    # 레이어 수 (단위 테스트용 GPT-2 small 기준)
  n_heads: 4                      # KV head 수 (단위 테스트용)
  max_iter_kmeans: 100            # k-means 최대 반복
  rope_base: 10000
  seed: 42
  recent_window: 32               # 최근 N 토큰 FP16 유지 (단위 테스트용)
  calibration_samples: 50         # 코드북 학습용 calibration 요청 수 (단위 테스트용)
  calibration_save_path: "results/2026-05-10/vq_codebooks.pt"
  # 실모델 설정 오버라이드 예시:
  # codebook_size: 16
  # n_residuals: 4
  # d_head: 128
  # n_layers: 32
  # n_heads: 8
  # recent_window: 128

kv_packet_adapter:
  n_heads: 4
  d_head: 64
  adapter_rank: 8                 # r: 어댑터 rank
  max_packets: 512
  embedding_dim: 64
  train_contexts_per_doc: 8       # 자기지도 증류 시 문서당 맥락 수
  train_steps: 1000               # 어댑터 훈련 스텝 수
  train_lr: 1.0e-3

context_free_compressed:
  max_packets: 512
  recent_window: 32
  adapter_rank: 8

cache:
  type: "context_free_compressed"
  capacity_bytes: 4294967296      # 4 GiB

benchmark:
  accuracy:
    datasets: ["wikitext2", "longbench"]
    wikitext2_stride: 512
    wikitext2_max_length: 2048
    longbench_subtasks:
      - "qasper"
      - "multifieldqa_en"
      - "hotpotqa"
      - "2wikimqa"
      - "musique"
      - "narrativeqa"
      - "qmsum"
      - "gov_report"
    perplexity_tolerance_pct: 1.0
    task_accuracy_tolerance_pct: 1.0
  vq_sweep:
    codebook_sizes: [16, 64, 256]
    n_residuals_list: [1, 2, 4]
  hit_rate:
    target_noncontiguous_fraction: 0.30
  memory_reduction:
    target_ratio: 0.30            # 최소 30% (목표 82%)
  throughput:
    target_improvement_pct: 10    # 최소 +10% (목표 +20%)

seed: 42
results_dir: "results/2026-05-10"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_vq_codec.py` — VQCodec 단위 테스트 (필수: `test_perplexity_delta_within_1pct` 포함)
- [ ] `tests/unit/test_kv_packet_adapter.py` — KVPacketSoftAdapterCache 단위 테스트
- [ ] `tests/unit/test_context_free_compressed.py` — ContextFreeCompressedKVPacket 단위 테스트
- [ ] `tests/unit/test_segmented_cache.py` — 기존 테스트 보존 + VQCodec 연동 케이스 추가
- [ ] `tests/integration/test_cross_bc_packet_vq.py` — B+C 통합 E2E 테스트 (정확도 보존 포함)

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 전부 통과** (기존 테스트 회귀 없음 포함)
2. **통합 테스트 전부 통과**
3. **Activity B (evaluation_criteria.md §3) 기준 충족**:
   - 비연속 세그먼트 히트율 >= 30% (전체 히트의 30% 이상이 비연속 구간)
   - KV Memory Footprint 베이스라인 대비 +20% 이내 (B+C 압축으로 오히려 감소 목표)
4. **Activity C (evaluation_criteria.md §4) 필수 기준 충족**:
   - perplexity 변화 +-1% 이내 (WikiText-2 + stub 검증)
   - downstream 태스크 정확도 변화 +-1% 이내 (LongBench 8개 서브태스크)
   - KV Memory Reduction >= -30% (목표 -82%)
   - Encode/Decode 추가 지연 TTFT +10% 이내
5. **크로스 조합 (evaluation_criteria.md §5) 기준 충족**:
   - 복합 처리량 향상: 단일 Activity 대비 추가 +5% 이상
   - 복합 메모리 감소: 단일 Activity 대비 추가 -10% 이상
   - Activity C 포함이므로 복합 적용 후에도 accuracy +-1% 이내 유지
6. **CacheStore 인터페이스 준수**: 모든 신규 클래스가 CacheStore 추상 메서드를 완전 구현
7. **설정 YAML 존재**: `configs/experiments/2026-05-10.yaml` 생성됨
8. **타입 힌트**: 모든 공개 함수·메서드에 완전한 타입 힌트
9. **시드 고정 재현성**: seed=42 고정 시 동일 결과 재현

---

## 구현 순서 (implementer 참고)

1. `src/compression/__init__.py` 및 `src/compression/vq_codec.py` — 가장 먼저 구현.
   `inverse_rope()`, `apply_rope()` → `fit()` → `encode()` / `decode()` 순서로 구현.
   `test_inverse_rope_correctness` 통과 확인 후 다음 단계 진행.

2. `src/cache/base.py` — `compression_hook` 선택적 메서드 추가 (5줄 변경).

3. `src/cache/kv_packet_adapter.py` — B 구현.
   `SoftTokenAdapter.adapt()` → `create_packet()` → `put/get` → `pack()` 순서.
   훈련 없는 항등 어댑터 모드(zero init)로 먼저 `test_kv_packet_adapter.py` 통과 확인.

4. `src/cache/context_free_compressed_packet.py` — B+C 통합.
   VQCodec + KVPacketSoftAdapterCache 조합. `TriangleInequalitySegmentIndex` 재사용.

5. `src/cache/segmented.py` — VQCodec 선택적 연동 파라미터 추가 (기존 동작 완전 보존).

6. 모든 테스트 파일 작성 및 통과 확인. `test_perplexity_delta_within_1pct` 필수 통과.

7. `configs/experiments/2026-05-10.yaml` 작성.

---

## 기존 파일 보존 목록

이번 사이클에서 수정하지 않는 파일 (기존 테스트 회귀 없이 통과해야 함):

- `src/cache/triangle_index.py` (B+C 통합의 검색 백엔드로 재사용, 수정 불필요)
- `src/cache/eopt_shrinkq_codec.py` (VQCodec과 직교, 향후 결합 가능)
- `src/scheduler/` 전체 (이번 사이클 Activity A 미포함)
- `src/cache/semantic_boundary_cache.py`, `src/cache/static_dynamic_segment.py`
- `src/cache/speckv_gamma_controller.py`, `src/cache/context_intensive_guard.py`
- 기타 모든 기존 캐시 구현체 및 테스트 파일 (회귀 없이 통과 필수)
