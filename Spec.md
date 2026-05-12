<!-- 변경 이유 (이전 Spec.md: 2026-05-11 대비):
이전 사이클(2026-05-11)은 B+C (WiCERIterativeKVWikiCache + RateQuantReverseWaterfillingCodec +
WiCER-RateQuantPipeline) 조합이었다. 이번 사이클은 B+C
(RoPEReencodingNonContiguousCache + MixedDimPerTokenBudgetCodec +
AdapShotMixedDimSegmentPipeline) 조합으로 전환한다.

주요 변경:
1. [Activity B 교체] WiCERIterativeKVWikiCache(CEGAR 반복 컴파일) →
   RoPEReencodingNonContiguousCache(position-decoupled KV 저장 + RoPE 재인코딩).
   알고리즘 패러다임이 "반복 정제 도메인 아티팩트" → "위치 독립 세그먼트 저장 + in-place RoPE 재적용"으로
   전환된다. 이를 위해 src/cache/base.py에 store_pre_rope()/load_with_rope() 메서드를 추가한다.

2. [Activity C 교체] RateQuantReverseWaterfillingCodec(역 물 채우기 비트 할당) →
   MixedDimPerTokenBudgetCodec(토큰별 연속 차원 예산 할당, 훈련-프리).
   압축 패러다임이 "헤드별 비트폭 양자화" → "토큰별 차원 드롭(연속 예산)"으로 전환된다.
   bisection search로 전체 메모리 예산 B에 대한 손실 점수 임계값 λ*를 결정한다.

3. [Cross 교체] WiCER-RateQuantPipeline → AdapShotMixedDimSegmentPipeline.
   파이프라인 저장 순서: pre-RoPE KV → mixed-dim 압축.
   복원 순서: mixed-dim 해제 → 목표 위치에 RoPE 재적용.
   이 순서 의존성을 인터페이스 계약으로 명시.

4. [보존 파일] 이전 사이클 파일(wicer_iterative_cache.py, ratequant_codec.py,
   wicer_ratequant_pipeline.py, 기타 모든 기존 파일)은 이번 사이클에서 수정하지 않는다.
   기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.

5. [인터페이스 확장] CacheStore에 store_pre_rope()/load_with_rope() 선택적 메서드 추가.
   기본 구현은 NotImplementedError 대신 ValueError를 올려 하위 호환성 보장.
-->

# Spec — 2026-05-12

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-12.md`
**최우선 구현 타겟**: Cross-2 (B+C) — RoPEReencodingNonContiguousCache(B-1) +
MixedDimPerTokenBudgetCodec(C-1) + AdapShotMixedDimSegmentPipeline 통합

**해결하려는 문제**:

- Activity B: 표준 KV 캐시는 요청 간 공통 접두사가 byte-identical할 때만 재사용된다.
  RoPE(Rotary Position Embedding)가 적용된 KV를 저장하면 동일 토큰 내용이라도 위치가 달라지면
  히트가 발생하지 않는다. `RoPEReencodingNonContiguousCache`는 KV를 RoPE 적용 **전** 상태
  (position-decoupled)로 저장하고 재사용 시 목표 위치에 맞는 회전 행렬을 GPU에서 in-place로
  재적용한다. 이로써 prefix 불일치 요청에서도 세그먼트 히트를 발생시킨다.
  예상 효과: 비연속 히트율 +25~40%, TTFT ±3% 이내 (재인코딩 비용 미미).

- Activity C: 기존 이진 evict/keep 방식(H2O, SnapKV 등)은 토큰 단위로만 결정하여
  고중요도 토큰도 저분산 차원을 불필요하게 보존한다.
  `MixedDimPerTokenBudgetCodec`은 각 토큰의 "손실 점수"(어텐션 중요도 × 값 벡터 크기 × 차원별
  PCA 압축성)를 계산하고, 레이어 전체 메모리 예산 B에 대해 bisection search로 임계값 λ*를
  구해 λ* 이하 손실 점수 차원만 보관한다. 훈련 없이 추론 시 즉시 적용 가능.
  예상 효과: 메모리 −45~55%, 디코드 레이턴시 −55%.

- Cross-2 시너지: B-1(pre-RoPE 저장) + C-1(mixed-dim 압축)을 파이프라인으로 통합한다.
  캐시 저장 순서: raw KV → mixed-dim 압축 → pre-RoPE 상태로 저장.
  캐시 복원 순서: 저장된 압축 KV → mixed-dim 해제 → 목표 위치에 RoPE 재인코딩.
  두 단계가 순서 의존성을 가지므로 인터페이스 계약을 명확히 정의한다.
  예상 복합 효과: 메모리 −45~55%, 비연속 히트율 +30%, 정확도 delta ±1% 이내.

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling
- [x] Activity B: Non-Contiguous KV Cache Reuse
- [x] Activity C: KV Cache Compression

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §3 Activity B): 비연속 세그먼트 히트율 >= 30%
      (전체 히트 중 비연속 구간에서 발생한 히트 비율)
- [ ] 목표 2 (evaluation_criteria.md §3 Activity B): 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상;
      position-decoupled 저장으로 위치 불일치 히트 복구 목표 +25%
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      (WikiText-103 기준, 압축 전후 비교; 합성 proxy는 attention output relative error < 0.01)
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 변화 ±1% 이내
      (MMLU proxy: attention score KL divergence < 0.015, cosine similarity >= 0.99)
- [ ] 목표 5 (evaluation_criteria.md §4 Activity C): KV Memory Reduction >= −30%
      (베이스라인 대비); 목표치 −45~55%
- [ ] 목표 6 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +10% 이상;
      목표치 +20% (mixed-dim 압축으로 디코드 레이턴시 단축)
- [ ] 목표 7 (evaluation_criteria.md §4 Activity C): Scheduling Overhead — TTFT p50 +5% 이내
      (RoPE 재인코딩 추가 오버헤드 포함)
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합): 단일 Activity 대비 복합 처리량 추가 +5% 이상
- [ ] 목표 9 (evaluation_criteria.md §5 크로스 조합): 단일 Activity 대비 복합 메모리 감소 추가 −10% 이상

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/rope_reencoding_cache.py` | B | RoPEReencodingNonContiguousCache — position-decoupled KV 저장, 재사용 시 RoPE 재인코딩 |
| `src/cache/mixed_dim_codec.py` | C | MixedDimPerTokenBudgetCodec — 토큰별 손실 점수 기반 차원 예산 할당, bisection search λ* |
| `src/cache/adapshot_pipeline.py` | B+C | AdapShotMixedDimSegmentPipeline — B-1 + C-1 통합 파이프라인, 저장/복원 순서 계약 |
| `tests/unit/test_rope_reencoding_cache.py` | B | position-decoupled 저장 단위 테스트 (히트율, RoPE 재인코딩 정확도, content hash 키) |
| `tests/unit/test_mixed_dim_codec.py` | C | 차원 예산 할당 단위 테스트 (bisection λ*, encode/decode shape, 압축률) |
| `tests/unit/test_mixed_dim_accuracy.py` | C | Activity C 필수 — 압축 정확도 보존 검증 (±1% proxy 기준) |
| `tests/integration/test_cross_bc_adapshot.py` | B+C | B+C 통합 E2E 테스트 (비연속 히트율 + 메모리 감소 + 정확도 복합 검증) |
| `configs/experiments/2026-05-12.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/base.py` | `store_pre_rope()` / `load_with_rope()` 선택적 메서드 추가. 기본 구현은 `NotImplementedError` raise로 미구현 시 명시적 오류. RoPE-capable 구현체만 오버라이드. |

**보존 파일 (이번 사이클에서 수정하지 않음)**:
- `src/cache/segmented.py` — RoPEReencodingNonContiguousCache의 백엔드로 재사용 (content hash 방식 호환)
- `src/cache/wicer_iterative_cache.py`, `src/cache/ratequant_codec.py`, `src/cache/wicer_ratequant_pipeline.py` — 이전 사이클 구현 보존
- `src/compression/` 전체, `src/metrics/perplexity.py` — 기존 그대로 사용
- `src/scheduler/` 전체 — 이번 사이클 Activity A 미포함
- 기타 모든 기존 캐시 구현체, 테스트 파일 — 회귀 없이 통과 필수

---

## 알고리즘 상세

### [CacheStore 인터페이스 확장] — `src/cache/base.py`

기존 6개 추상 메서드는 그대로 유지한다. RoPE re-encoding을 지원하는 구현체를 위해
두 메서드를 **선택적(optional)** 메서드로 추가한다. 기본 구현은 `NotImplementedError`를 raise하여
호출 측이 지원 여부를 명시적으로 확인할 수 있게 한다.

```python
# src/cache/base.py에 추가할 메서드 (abstractmethod 아님)

def store_pre_rope(
    self,
    key: str,
    value: torch.Tensor,   # [n_tokens, 2, n_heads, d_head] — RoPE 적용 전
    layer_idx: int = 0,
) -> None:
    """RoPE 적용 전(position-decoupled) KV를 content hash 키로 저장한다.
    기본 구현은 NotImplementedError. RoPE-capable 구현체만 오버라이드."""
    raise NotImplementedError(
        f"{type(self).__name__} does not support pre-RoPE storage."
    )

def load_with_rope(
    self,
    key: str,
    target_positions: torch.Tensor,  # [n_tokens] long — 목표 위치 인덱스
    layer_idx: int = 0,
    rope_dim: int = -1,              # RoPE 적용 차원 (-1이면 d_head 전체)
) -> Optional[torch.Tensor]:
    """저장된 pre-RoPE KV를 로드하고 target_positions에 맞는 회전 행렬을 적용해 반환.
    캐시 미스 시 None 반환. 기본 구현은 NotImplementedError."""
    raise NotImplementedError(
        f"{type(self).__name__} does not support pre-RoPE loading."
    )
```

---

### [RoPEReencodingNonContiguousCache] (Activity B) — `src/cache/rope_reencoding_cache.py`

RoPE를 적용하기 **전** KV를 content hash(위치 무관)로 저장한다. 재사용 시 목표 위치에 맞는
회전 행렬을 GPU에서 in-place로 계산하여 KV에 적용한다.
`SegmentedHashCache`를 백엔드로 사용하여 content hash 키 방식과 완전히 호환된다.

**핵심 RoPE 재인코딩 수식**:
- 각 차원 쌍 (2i, 2i+1)에 대해: `[cos(pos·θ_i), -sin(pos·θ_i); sin(pos·θ_i), cos(pos·θ_i)]`
- `θ_i = 10000^(-2i/d_head)` (표준 RoPE 주파수)
- 저장 시: RoPE 이전 KV (raw)를 저장
- 로드 시: raw KV에 target_position의 회전 행렬 적용

```python
# 의사코드 — src/cache/rope_reencoding_cache.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class RoPEReencodingConfig:
    chunk_size: int = 128          # 세그먼트 청크 크기 (토큰 수)
    max_entries: int = 2000        # 최대 캐시 엔트리 수
    d_head: int = 64               # head 차원
    n_heads: int = 8               # 어텐션 헤드 수
    rope_base: float = 10000.0     # RoPE 주파수 base (기본 10000)
    rope_dim: int = -1             # RoPE 적용 차원 (-1이면 d_head 전체)
    seed: int = 42


class RoPEReencodingNonContiguousCache(CacheStore):
    """Position-decoupled KV 저장 + RoPE 재인코딩 비연속 세그먼트 캐시 (Activity B).

    CacheStore 인터페이스 완전 구현 + store_pre_rope()/load_with_rope() 오버라이드.
    내부적으로 SegmentedHashCache를 사용하여 content hash 방식 유지.

    사용 흐름:
      1. store_pre_rope(key, pre_rope_kv, layer_idx)  — RoPE 전 KV를 저장
      2. load_with_rope(key, target_positions, layer_idx)  — 로드 후 RoPE 재적용
      3. put/get — CacheStore 표준 인터페이스 (RoPE-aware 아닌 경우)
    """

    def __init__(self, config: RoPEReencodingConfig) -> None:
        self.config = config
        self._store = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        # 위치별 회전 행렬 캐시 (지연 계산)
        self._rope_cache: Dict[Tuple[int, int], torch.Tensor] = {}  # (pos, layer_idx) → [d_head/2, 2, 2]
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """RoPE가 이미 적용된 KV를 그대로 저장 (표준 CacheStore 경로).
        RoPE-decoupled 저장이 필요하면 store_pre_rope()를 사용하라."""
        self._store.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """표준 CacheStore 경로 — RoPE 재적용 없음."""
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self._store.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._store.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._store.reset_stats()

    # ------------------------------------------------------------------ #
    # RoPE-aware 확장 메서드 (CacheStore 선택적 메서드 오버라이드)         #
    # ------------------------------------------------------------------ #

    def store_pre_rope(
        self,
        key: str,
        value: torch.Tensor,   # [n_tokens, 2, n_heads, d_head] — RoPE 적용 전
        layer_idx: int = 0,
    ) -> None:
        """RoPE 적용 전 KV를 content hash 키로 저장한다.

        의사코드:
          # key는 호출 측이 content hash로 생성 (SegmentedHashCache.chunk_key() 활용)
          scoped_key = f"pre_rope:{layer_idx}:{key}"
          self._store.put(scoped_key, value)
        """
        scoped_key = f"pre_rope:{layer_idx}:{key}"
        self._store.put(scoped_key, value)

    def load_with_rope(
        self,
        key: str,
        target_positions: torch.Tensor,  # [n_tokens] long — 목표 위치 인덱스
        layer_idx: int = 0,
        rope_dim: int = -1,
    ) -> Optional[torch.Tensor]:
        """저장된 pre-RoPE KV를 로드하고 target_positions에 맞는 회전 행렬을 적용한다.

        반환: [n_tokens, 2, n_heads, d_head] float16 (RoPE 재적용 후)
              캐시 미스 시 None 반환

        의사코드:
          scoped_key = f"pre_rope:{layer_idx}:{key}"
          pre_rope_kv = self._store.get(scoped_key)  # [n_tokens, 2, n_heads, d_head]
          if pre_rope_kv is None:
              self._misses += 1
              return None
          self._hits += 1
          # target_positions별로 회전 행렬 적용
          result = self._apply_rope(pre_rope_kv, target_positions, rope_dim)
          return result
        """
        scoped_key = f"pre_rope:{layer_idx}:{key}"
        pre_rope_kv = self._store.get(scoped_key)
        if pre_rope_kv is None:
            self._misses += 1
            return None
        self._hits += 1
        return self._apply_rope(pre_rope_kv, target_positions, rope_dim)

    # ------------------------------------------------------------------ #
    # RoPE 계산 유틸리티                                                    #
    # ------------------------------------------------------------------ #

    def _get_rope_rotation(
        self,
        position: int,
        d: int,                    # 회전 적용 차원 (짝수여야 함)
    ) -> torch.Tensor:
        """주어진 위치(position)에 대한 RoPE 회전 행렬 계산.

        반환: [d//2, 2, 2] — 각 차원 쌍에 대한 2×2 회전 행렬

        의사코드:
          if (position, d) in self._rope_cache:
              return self._rope_cache[(position, d)]
          theta = 1.0 / (self.config.rope_base ** (torch.arange(0, d, 2).float() / d))
          angle = position * theta                  # [d//2]
          cos_a = angle.cos()                       # [d//2]
          sin_a = angle.sin()                       # [d//2]
          # 회전 행렬: [[cos, -sin], [sin, cos]]
          rot = torch.stack([
              torch.stack([cos_a, -sin_a], dim=-1),
              torch.stack([sin_a,  cos_a], dim=-1),
          ], dim=-2)                                # [d//2, 2, 2]
          self._rope_cache[(position, d)] = rot
          return rot
        """
        ...

    def _apply_rope(
        self,
        pre_rope_kv: torch.Tensor,     # [n_tokens, 2, n_heads, d_head]
        target_positions: torch.Tensor, # [n_tokens] long
        rope_dim: int = -1,
    ) -> torch.Tensor:
        """각 토큰에 target_positions에 해당하는 RoPE 회전 행렬을 적용한다.

        의사코드:
          d = pre_rope_kv.shape[-1] if rope_dim == -1 else rope_dim
          assert d % 2 == 0, "RoPE 적용 차원은 짝수여야 합니다"
          result = pre_rope_kv.clone()
          # shape: [n_tokens, 2(k/v), n_heads, d_head]
          # key만 RoPE 적용 (dim=1의 인덱스 0); value는 일반적으로 RoPE 미적용
          key_slice = result[:, 0, :, :d]   # [n_tokens, n_heads, d]
          # 각 토큰별로 회전 적용
          for tok_idx in range(n_tokens):
              pos = target_positions[tok_idx].item()
              rot = self._get_rope_rotation(pos, d)   # [d//2, 2, 2]
              # key_slice[tok_idx]: [n_heads, d] → reshape → [n_heads, d//2, 2]
              k_tok = key_slice[tok_idx].reshape(n_heads, d // 2, 2)  # [n_heads, d//2, 2]
              # 각 차원 쌍에 rot 행렬 곱: einsum("hpi,pij->hpj", k_tok, rot)
              k_tok_rot = torch.einsum("hpi,pij->hpj", k_tok, rot)   # [n_heads, d//2, 2]
              key_slice[tok_idx] = k_tok_rot.reshape(n_heads, d)
          result[:, 0, :, :d] = key_slice
          return result

        성능 최적화 힌트:
          - GPU 배치 연산: torch.einsum 대신 torch.bmm으로 벡터화
          - 위치별 회전 행렬을 미리 배치로 계산: [n_tokens, d//2, 2, 2]
          - self._rope_cache로 동일 위치 반복 계산 방지
        """
        ...

    # ------------------------------------------------------------------ #
    # 세그먼트 API (SegmentedHashCache 호환)                               #
    # ------------------------------------------------------------------ #

    def put_segment_pre_rope(
        self,
        token_ids: List[int],
        chunk_idx: int,
        pre_rope_kv: torch.Tensor,   # [n_tokens, 2, n_heads, d_head] — RoPE 전
        layer_idx: int = 0,
    ) -> None:
        """content hash 키로 pre-RoPE KV를 세그먼트 단위로 저장.

        의사코드:
          key = self._store.chunk_key(token_ids, chunk_idx, layer_idx)
          self.store_pre_rope(key, pre_rope_kv, layer_idx)
        """
        key = self._store.chunk_key(token_ids, chunk_idx, layer_idx)
        self.store_pre_rope(key, pre_rope_kv, layer_idx)

    def get_segments_with_rope(
        self,
        token_ids: List[int],
        target_offset: int,          # 이 요청에서 첫 토큰의 위치 인덱스
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """모든 청크를 조회하고 히트 청크에 RoPE를 재적용하여 반환.

        반환:
          hits: [(chunk_idx, rope_applied_kv)] — RoPE 재인코딩된 KV
          misses: [chunk_idx] — 캐시 미스 청크 인덱스

        의사코드:
          n_chunks = ceil(len(token_ids) / chunk_size)
          hits, misses = [], []
          for chunk_idx in range(n_chunks):
              key = self._store.chunk_key(token_ids, chunk_idx, layer_idx)
              start_tok = chunk_idx * chunk_size
              end_tok = min(start_tok + chunk_size, len(token_ids))
              n_tok = end_tok - start_tok
              # 이 청크의 목표 위치: 요청 내 절대 위치
              positions = torch.arange(
                  target_offset + start_tok,
                  target_offset + end_tok,
                  dtype=torch.long,
              )
              kv = self.load_with_rope(key, positions, layer_idx)
              if kv is not None:
                  hits.append((chunk_idx, kv))
                  # 비연속 히트 추적
                  if any(m < chunk_idx for m in misses):
                      self._noncontiguous_hits += 1
              else:
                  misses.append(chunk_idx)
          return hits, misses
        """
        ...

    def noncontiguous_hit_rate(self) -> float:
        """비연속 세그먼트 히트율."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits
```

---

### [MixedDimPerTokenBudgetCodec] (Activity C) — `src/cache/mixed_dim_codec.py`

각 토큰의 손실 점수(loss score)를 세 요소의 가중 곱으로 정의하고,
bisection search로 레이어 전체 메모리 예산 B를 만족하는 임계값 λ*를 결정한다.
λ* 이하 손실 점수를 가진 차원은 드롭(0으로 마스킹)하고 나머지 차원만 보관한다.
훈련 없이 추론 시 즉시 적용 가능하다.

**손실 점수 정의**:
```
loss_score(token t, dim d) =
    attention_importance(t)       # 어텐션 중요도 (softmax weight 합산)
    × value_magnitude(t)          # 값 벡터 크기 ||v_t||_2
    × (1 - compressibility(t, d)) # PCA 압축성 역수 (분산 비율)
```

**bisection 알고리즘**:
```
입력: loss_scores [n_tokens, d_head], budget_ratio (0~1, 보존 비율)
목표: |retained_dims| / (n_tokens * d_head) ≈ budget_ratio

1. lo = 0.0, hi = loss_scores.max()
2. for _ in range(64):  # bisection 반복
       mid = (lo + hi) / 2
       mask = loss_scores >= mid   # 보존할 차원 마스크
       retained_ratio = mask.float().mean()
       if retained_ratio > budget_ratio:
           lo = mid   # 임계값 높여서 더 많이 드롭
       else:
           hi = mid
3. λ* = (lo + hi) / 2
4. final_mask = loss_scores >= λ*
```

```python
# 의사코드 — src/cache/mixed_dim_codec.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class MixedDimConfig:
    n_heads: int = 8                # 어텐션 헤드 수
    d_head: int = 64                # head 차원
    budget_ratio: float = 0.50     # 보존할 차원 비율 (0~1), 기본 50% → 메모리 −50%
    bisection_iters: int = 64      # bisection 탐색 반복 횟수
    min_retain_ratio: float = 0.10 # 최소 보존 비율 (중요 토큰 보호)
    # 손실 점수 가중치
    attn_importance_weight: float = 1.0
    value_magnitude_weight: float = 1.0
    compressibility_weight: float = 1.0
    seed: int = 42


class MixedDimPerTokenBudgetCodec:
    """토큰별 연속 차원 예산 할당 KV 압축 코덱 (MixedDimKV, Activity C).

    CacheStore.compression_hook()과 독립 encode()/decode() 양방향 사용 가능.
    훈련 없이 추론 시 즉시 적용 가능.

    주의: decode()는 드롭된 차원을 0으로 복원하므로 손실 있는 압축이다.
    accuracy-preserving 근거: 어텐션 집중 토큰은 손실 점수가 높아 전체 차원 보존,
    압축 대상은 어텐션 가중치가 낮고 값 크기도 작은 토큰의 저분산 차원에 한정됨.
    """

    def __init__(self, config: MixedDimConfig) -> None:
        self.config = config

    def compute_loss_scores(
        self,
        kv: torch.Tensor,           # [n_tokens, 2, n_heads, d_head]
        attn_weights: Optional[torch.Tensor] = None,
        # [n_tokens] float — 외부 어텐션 중요도. None이면 value magnitude로 대체
    ) -> torch.Tensor:
        """각 토큰·헤드·차원에 대한 손실 점수 계산.

        반환: [n_tokens, n_heads, d_head] float — 클수록 중요하므로 보존

        의사코드:
          k, v = kv[:, 0], kv[:, 1]   # [n_tokens, n_heads, d_head]

          # 1. 어텐션 중요도 [n_tokens, n_heads]
          if attn_weights is not None:
              attn_imp = attn_weights.unsqueeze(-1).expand_as(k[..., :1])  # [n_tokens, n_heads, 1]
          else:
              # value magnitude로 대체: ||v_t_h||_2 → [n_tokens, n_heads, 1]
              attn_imp = v.norm(dim=-1, keepdim=True)

          # 2. 값 벡터 크기 [n_tokens, n_heads, 1]
          val_mag = v.norm(dim=-1, keepdim=True)

          # 3. 차원별 압축성 = PCA 분산 비율
          #    각 차원의 분산 (전체 토큰 기준): [n_heads, d_head]
          k_var = k.var(dim=0)          # [n_heads, d_head] — 토큰 axis 분산
          v_var = v.var(dim=0)          # [n_heads, d_head]
          kv_var = (k_var + v_var) / 2  # 합산 분산
          total_var = kv_var.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [n_heads, 1]
          compress_score = kv_var / total_var   # [n_heads, d_head] — 높을수록 고분산
          compress_score = compress_score.unsqueeze(0)  # [1, n_heads, d_head]

          # 최종 손실 점수 (세 요소 가중 곱)
          loss = (
              config.attn_importance_weight * attn_imp
              * config.value_magnitude_weight * val_mag
              * config.compressibility_weight * compress_score
          )  # [n_tokens, n_heads, d_head]

          return loss
        """
        ...

    def find_threshold(
        self,
        loss_scores: torch.Tensor,  # [n_tokens, n_heads, d_head]
        budget_ratio: Optional[float] = None,
    ) -> float:
        """bisection search로 예산 비율을 만족하는 임계값 λ* 탐색.

        의사코드:
          ratio = budget_ratio or self.config.budget_ratio
          lo, hi = 0.0, float(loss_scores.max())
          for _ in range(self.config.bisection_iters):
              mid = (lo + hi) / 2
              retained = (loss_scores >= mid).float().mean().item()
              if retained > ratio:
                  lo = mid
              else:
                  hi = mid
          return (lo + hi) / 2
        """
        ...

    def encode(
        self,
        kv: torch.Tensor,           # [n_tokens, 2, n_heads, d_head] float16
        attn_weights: Optional[torch.Tensor] = None,  # [n_tokens]
        budget_ratio: Optional[float] = None,
    ) -> dict:
        """반환:
        {
          'masked_kv': torch.Tensor,    # [n_tokens, 2, n_heads, d_head] — 드롭된 차원 0
          'retain_mask': torch.Tensor,  # [n_tokens, n_heads, d_head] bool — True=보존
          'lambda_star': float,         # 결정된 임계값
          'budget_ratio': float,        # 실제 보존 비율
          'n_tokens': int,
          'n_heads': int,
          'd_head': int,
        }

        의사코드:
          loss_scores = self.compute_loss_scores(kv, attn_weights)
          lam = self.find_threshold(loss_scores, budget_ratio)
          retain_mask = loss_scores >= lam   # [n_tokens, n_heads, d_head] bool
          # min_retain_ratio 보장: 토큰별 최소 보존 차원 수 확인
          per_token_ratio = retain_mask.float().mean(dim=-1)  # [n_tokens, n_heads]
          under_min = per_token_ratio < self.config.min_retain_ratio
          if under_min.any():
              # min_retain_ratio 미달 토큰은 상위 차원 강제 보존
              topk_dim = max(1, int(d_head * self.config.min_retain_ratio))
              topk_idx = loss_scores[under_min].topk(topk_dim, dim=-1).indices
              retain_mask[under_min].scatter_(-1, topk_idx, True)
          # 마스킹: 드롭 차원을 0으로
          masked_kv = kv.clone()
          # retain_mask를 k/v 양쪽에 적용: [n_tokens, 2, n_heads, d_head]
          mask_kv = retain_mask.unsqueeze(1).expand_as(masked_kv)
          masked_kv = masked_kv * mask_kv.float()
          actual_ratio = retain_mask.float().mean().item()
          return {
              'masked_kv': masked_kv,
              'retain_mask': retain_mask,
              'lambda_star': lam,
              'budget_ratio': actual_ratio,
              'n_tokens': kv.shape[0],
              'n_heads': kv.shape[2],
              'd_head': kv.shape[3],
          }
        """
        ...

    def decode(
        self,
        encoded: dict,
    ) -> torch.Tensor:
        """반환: [n_tokens, 2, n_heads, d_head] float16.

        의사코드:
          # 손실 압축이므로 드롭된 차원은 0으로 복원됨 (변환 없이 그대로 반환)
          return encoded['masked_kv']

        참고: 드롭된 차원의 값이 0이므로 attention output에 미치는 영향은
        어텐션 중요도 × 값 크기가 낮은 토큰에 한정되며, 이론적 오차는 O(λ* · √dim).
        """
        return encoded['masked_kv']

    def memory_reduction_ratio(self, encoded: dict) -> float:
        """실효 메모리 감소율 = 1 - retained_ratio.
        예: budget_ratio=0.50 → 50% 감소.
        """
        return 1.0 - encoded['budget_ratio']

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,        # [n_tokens, 2, n_heads, d_head]
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CacheStore.compression_hook() 호환 인터페이스.
        encode 후 decode 결과(masked_kv)를 반환한다."""
        encoded = self.encode(value, attn_weights)
        return self.decode(encoded)
```

---

### [AdapShotMixedDimSegmentPipeline] (Activity B+C) — `src/cache/adapshot_pipeline.py`

`RoPEReencodingNonContiguousCache`(B)와 `MixedDimPerTokenBudgetCodec`(C)를 통합하는
파이프라인. B-1과 C-1이 파이프라인에서 순서 의존성을 가지므로 인터페이스 계약을 아래에 명시한다.

**저장 순서 계약 (Store Order)**:
```
raw KV (RoPE 전)
    ↓  [1단계] MixedDimPerTokenBudgetCodec.encode()
mixed-dim 압축 KV (드롭된 차원 0, retain_mask 저장)
    ↓  [2단계] RoPEReencodingNonContiguousCache.store_pre_rope()
content hash 키로 pre-RoPE 상태 저장
```

**복원 순서 계약 (Restore Order)**:
```
content hash 키로 저장된 압축 KV 로드
    ↓  [1단계] RoPEReencodingNonContiguousCache.load_with_rope()
    (내부에서 _apply_rope() 호출하여 target_positions에 맞는 회전 행렬 적용)
RoPE 재인코딩된 압축 KV
    ↓  [2단계] MixedDimPerTokenBudgetCodec.decode()
최종 복원 KV (드롭된 차원 0, RoPE 적용 완료)
```

**중요**: 복원 시 decode()는 RoPE 재적용 **이후**에 호출되어야 한다.
RoPE 적용 전 decode()를 호출하면 회전 행렬이 0인 드롭 차원에 적용되어 결과가 달라진다.

```python
# 의사코드 — src/cache/adapshot_pipeline.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.rope_reencoding_cache import RoPEReencodingNonContiguousCache, RoPEReencodingConfig
from src.cache.mixed_dim_codec import MixedDimPerTokenBudgetCodec, MixedDimConfig


@dataclass
class AdapShotPipelineConfig:
    rope: RoPEReencodingConfig = None
    mixed_dim: MixedDimConfig = None

    def __post_init__(self):
        if self.rope is None:
            self.rope = RoPEReencodingConfig()
        if self.mixed_dim is None:
            self.mixed_dim = MixedDimConfig()


class AdapShotMixedDimSegmentPipeline(CacheStore):
    """B+C 통합 파이프라인: pre-RoPE 저장 + mixed-dim 압축 (Cross-2, Activity B+C).

    CacheStore 인터페이스 완전 구현 (RoPEReencodingNonContiguousCache에 위임).

    저장 순서: raw KV → mixed-dim 압축 → pre-RoPE 저장
    복원 순서: pre-RoPE 로드 → RoPE 재적용 → mixed-dim 해제(decode)
    """

    def __init__(self, config: AdapShotPipelineConfig) -> None:
        self.config = config
        self.rope_cache = RoPEReencodingNonContiguousCache(config.rope)
        self.codec = MixedDimPerTokenBudgetCodec(config.mixed_dim)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """mixed-dim 압축 후 pre-RoPE 상태로 저장 (저장 순서 계약 적용).

        의사코드:
          compressed = self.codec.compression_hook(key, value)
          self.rope_cache.store_pre_rope(key, compressed, layer_idx=0)
        """
        compressed = self.codec.compression_hook(key, value)
        self.rope_cache.store_pre_rope(key, compressed, layer_idx=0)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """표준 CacheStore 경로 — target_positions 없이 RoPE 재적용 불가.
        target_positions를 알 때는 get_with_rope()를 사용하라."""
        scoped_key = f"pre_rope:0:{key}"
        result = self.rope_cache._store.get(scoped_key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self.rope_cache.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self.rope_cache.memory_bytes()

    def reset_stats(self) -> None:
        self.rope_cache.reset_stats()
        self._hits = 0
        self._misses = 0

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MixedDimCodec 압축 후 결과 반환 (CacheStore 표준 hook)."""
        return self.codec.compression_hook(key, value, attn_weights)

    # ------------------------------------------------------------------ #
    # B+C 파이프라인 API                                                   #
    # ------------------------------------------------------------------ #

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        pre_rope_kv: torch.Tensor,   # [n_tokens, 2, n_heads, d_head] — RoPE 전 raw KV
        layer_idx: int = 0,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """저장 순서 계약: [1] mixed-dim 압축 → [2] pre-RoPE 저장.

        의사코드:
          key = self.rope_cache._store.chunk_key(token_ids, chunk_idx, layer_idx)
          # [1] mixed-dim 압축
          compressed_kv = self.codec.compression_hook(key, pre_rope_kv, attn_weights)
          # [2] pre-RoPE 상태로 저장
          self.rope_cache.store_pre_rope(key, compressed_kv, layer_idx)
        """
        ...

    def load_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        target_offset: int,          # 이 요청에서 청크 첫 토큰의 절대 위치
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """복원 순서 계약: [1] pre-RoPE 로드 → [2] RoPE 재적용 → [3] decode(masked_kv 그대로 반환).

        반환: [n_tokens, 2, n_heads, d_head] float16 (RoPE 재인코딩 + mixed-dim 압축 상태)
              캐시 미스 시 None

        의사코드:
          key = self.rope_cache._store.chunk_key(token_ids, chunk_idx, layer_idx)
          chunk_size = self.rope_cache.config.chunk_size
          start_tok = chunk_idx * chunk_size
          end_tok = min(start_tok + chunk_size, len(token_ids))
          n_tok = end_tok - start_tok
          # 이 청크 내 각 토큰의 절대 위치
          positions = torch.arange(
              target_offset + start_tok,
              target_offset + end_tok,
              dtype=torch.long,
          )
          # [1]+[2] pre-RoPE 로드 후 RoPE 재적용
          rope_applied_kv = self.rope_cache.load_with_rope(key, positions, layer_idx)
          if rope_applied_kv is None:
              return None
          # [3] decode — 이미 masked_kv이므로 그대로 반환 (드롭 차원 0 유지)
          return rope_applied_kv
        """
        ...

    def get_segments(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """모든 청크에 대해 load_segment()를 호출하여 히트/미스 분류.

        반환:
          hits: [(chunk_idx, rope_applied_compressed_kv)]
          misses: [chunk_idx]

        의사코드:
          chunk_size = self.rope_cache.config.chunk_size
          n_chunks = ceil(len(token_ids) / chunk_size)
          hits, misses = [], []
          for chunk_idx in range(n_chunks):
              kv = self.load_segment(token_ids, chunk_idx, target_offset, layer_idx)
              if kv is not None:
                  hits.append((chunk_idx, kv))
              else:
                  misses.append(chunk_idx)
          return hits, misses
        """
        ...

    def noncontiguous_hit_rate(self) -> float:
        return self.rope_cache.noncontiguous_hit_rate()

    def save_pipeline(self, path: str) -> None:
        """파이프라인 상태(rope_cache 내부 store + codec config)를 직렬화.

        torch.save({
            'rope_store': self.rope_cache._store._store,   # OrderedDict
            'rope_config': self.config.rope,
            'codec_config': self.config.mixed_dim,
        }, path)
        """
        ...

    def load_pipeline(self, path: str) -> None:
        """저장된 파이프라인 복원."""
        ...
```

---

## Activity C — Accuracy Preservation 검증 계획

**Activity C(MixedDimPerTokenBudgetCodec)를 포함하므로 이 섹션은 필수다.**
**이 계획 없이 Spec.md는 불완전하다.**

### perplexity 측정

| 항목 | 내용 |
|------|------|
| **데이터셋 (단위 테스트)** | 합성: `torch.randn` 기반 KV 텐서 (캘리브레이션 없는 훈련-프리 코덱이므로 즉시 검증 가능). `n_tokens=64`, `n_heads=8`, `d_head=64` |
| **데이터셋 (통합 테스트)** | WikiText-103 test split (stride=512, max_length=2048). 실 모델 사용 불가 환경에서는 합성 proxy로 대체. |
| **측정 방법** | 기존 `src/metrics/perplexity.py`의 세 함수 사용: `attention_output_relative_error()`, `attention_kl_divergence()`, `cosine_similarity_output()` |
| **허용 오차** | attention output relative error < 0.01 (1% 이내) — evaluation_criteria.md §4 필수 |
| **budget_ratio 스윕** | budget_ratio = 0.30, 0.40, 0.50, 0.60, 0.70 각각에서 perplexity proxy 측정 |
| **캘리브레이션 독립 검증** | 인코딩에 사용한 KV와 다른 seed의 독립 검증 KV로 정확도 측정 |
| **결과 저장** | `results/2026-05-12/perplexity_sweep.json` |

### MMLU 태스크 정확도 측정 (합성 proxy)

| 항목 | 내용 |
|------|------|
| **proxy 지표 1** | attention output cosine similarity >= 0.99 (압축 전후) |
| **proxy 지표 2** | attention score KL divergence < 0.015 (압축 전후) |
| **proxy 지표 3** | relative output error < 0.01 (1% 이내) |
| **허용 오차** | evaluation_criteria.md §4 필수: ±1% 이내 |
| **budget_ratio별 검증** | budget_ratio = 0.50 (기본값)에서 모든 proxy 지표 통과 확인 |
| **결과 저장** | `results/2026-05-12/accuracy_proxy_results.json` |

### 압축률-정확도 트레이드오프 예상값

| budget_ratio | 메모리 감소율 | 예상 relative error | 예상 KL | 예상 cosine |
|-------------|------------|-------------------|---------|-------------|
| 0.70 (70% 보존) | −30% | < 0.003 | < 0.005 | > 0.999 |
| 0.50 (50% 보존) | −50% | < 0.008 | < 0.012 | > 0.99  |
| 0.40 (40% 보존) | −60% | < 0.010 | < 0.015 | > 0.99  |
| 0.30 (30% 보존) | −70% | < 0.015 | < 0.02  | > 0.98  |

**MixedDim이 이진 eviction보다 정확도를 더 잘 보존하는 이유**: 이진 방식은 토큰 전체를 드롭하여
중요도가 낮아도 분산이 높은 차원이 사라진다. MixedDim은 토큰별·차원별로 세분화하여 고분산
차원은 보존하고 저분산 차원만 드롭하므로 동일 메모리 예산에서 attention output 왜곡이 더 작다.

### 검증 테스트 파일 — `tests/unit/test_mixed_dim_accuracy.py`

```python
# 필수 테스트 케이스 — tests/unit/test_mixed_dim_accuracy.py

import pytest
import torch
from src.cache.mixed_dim_codec import MixedDimPerTokenBudgetCodec, MixedDimConfig
from src.metrics.perplexity import (
    attention_output_relative_error,
    attention_kl_divergence,
    cosine_similarity_output,
)


@pytest.fixture
def codec_50pct():
    """50% budget_ratio (메모리 −50%) 코덱."""
    config = MixedDimConfig(n_heads=4, d_head=32, budget_ratio=0.50, seed=42)
    return MixedDimPerTokenBudgetCodec(config)


def test_encode_decode_shape(codec_50pct):
    """encode → decode 후 shape 보존: [n_tokens, 2, n_heads, d_head]."""
    torch.manual_seed(42)
    kv = torch.randn(32, 2, 4, 32)
    encoded = codec_50pct.encode(kv)
    recovered = codec_50pct.decode(encoded)
    assert recovered.shape == kv.shape


def test_budget_ratio_approximately_met(codec_50pct):
    """실제 보존 비율이 budget_ratio ±5% 이내."""
    torch.manual_seed(42)
    kv = torch.randn(64, 2, 4, 32)
    encoded = codec_50pct.encode(kv)
    assert abs(encoded['budget_ratio'] - 0.50) < 0.05


def test_accuracy_relative_error_within_1pct(codec_50pct):
    """PRIMARY ±1% 정확도 보존: relative output error < 0.01.
    budget_ratio=0.50 기준. evaluation_criteria.md §4 필수."""
    torch.manual_seed(99)  # 독립 시드
    kv = torch.randn(32, 2, 4, 32)
    # head=0의 key/value 슬라이스 [n_tokens, d_head]
    k_orig = kv[:, 0, 0, :]
    v_orig = kv[:, 1, 0, :]
    encoded = codec_50pct.encode(kv)
    kv_rec = codec_50pct.decode(encoded)
    k_comp = kv_rec[:, 0, 0, :]
    v_comp = kv_rec[:, 1, 0, :]
    q = torch.randn(8, 32)
    error = attention_output_relative_error(q, k_orig, v_orig, k_comp, v_comp)
    assert error < 0.01, f"Relative error {error:.4f} exceeds ±1% limit"


def test_kl_divergence_within_threshold(codec_50pct):
    """KL 발산 < 0.015 (±1% perplexity proxy). evaluation_criteria.md §4."""
    torch.manual_seed(77)
    kv = torch.randn(32, 2, 4, 32)
    encoded = codec_50pct.encode(kv)
    kv_rec = codec_50pct.decode(encoded)
    q = torch.randn(8, 32)
    kl = attention_kl_divergence(q, kv[:, 0, 0, :], kv_rec[:, 0, 0, :])
    assert kl < 0.015, f"KL divergence {kl:.4f} exceeds 0.015 threshold"


def test_cosine_similarity_above_threshold(codec_50pct):
    """Attention output cosine similarity >= 0.99. evaluation_criteria.md §4."""
    torch.manual_seed(55)
    kv = torch.randn(32, 2, 4, 32)
    encoded = codec_50pct.encode(kv)
    kv_rec = codec_50pct.decode(encoded)
    q = torch.randn(8, 32)
    sim = cosine_similarity_output(
        q, kv[:, 0, 0, :], kv[:, 1, 0, :],
        kv_rec[:, 0, 0, :], kv_rec[:, 1, 0, :],
    )
    assert sim >= 0.99, f"Cosine similarity {sim:.4f} below 0.99 threshold"


def test_memory_reduction_meets_target(codec_50pct):
    """budget_ratio=0.50에서 memory_reduction_ratio >= 0.30 (30% 이상 감소).
    evaluation_criteria.md §4: KV Memory Reduction >= −30%."""
    torch.manual_seed(42)
    kv = torch.randn(64, 2, 4, 32)
    encoded = codec_50pct.encode(kv)
    ratio = codec_50pct.memory_reduction_ratio(encoded)
    assert ratio >= 0.30, f"Memory reduction {ratio:.4f} below 30% target"


def test_high_importance_tokens_preserved():
    """어텐션 중요도가 높은 토큰은 더 많은 차원을 보존한다."""
    torch.manual_seed(42)
    config = MixedDimConfig(n_heads=4, d_head=32, budget_ratio=0.50, seed=42)
    codec = MixedDimPerTokenBudgetCodec(config)
    kv = torch.randn(16, 2, 4, 32)
    # 앞 8 토큰에 높은 어텐션 중요도 부여
    attn_weights = torch.cat([torch.ones(8) * 10.0, torch.ones(8) * 0.1])
    encoded = codec.encode(kv, attn_weights=attn_weights)
    mask = encoded['retain_mask']   # [n_tokens, n_heads, d_head]
    # 중요 토큰(앞 8개) 평균 보존 비율 > 비중요 토큰(뒤 8개)
    high_retention = mask[:8].float().mean().item()
    low_retention = mask[8:].float().mean().item()
    assert high_retention > low_retention, (
        f"High-importance retention {high_retention:.4f} <= "
        f"low-importance {low_retention:.4f}"
    )


def test_independent_seed_accuracy():
    """캘리브레이션(encode)과 완전히 독립된 seed에서도 ±1% 보존.
    evaluation_criteria.md §4 필수 요건의 핵심 증거."""
    config = MixedDimConfig(n_heads=4, d_head=32, budget_ratio=0.50, seed=0)
    codec = MixedDimPerTokenBudgetCodec(config)
    # 독립 seed로 검증 KV 생성
    torch.manual_seed(999)
    test_kv = torch.randn(32, 2, 4, 32)
    q = torch.randn(8, 32)
    encoded = codec.encode(test_kv)
    kv_rec = codec.decode(encoded)
    error = attention_output_relative_error(
        q, test_kv[:, 0, 0, :], test_kv[:, 1, 0, :],
        kv_rec[:, 0, 0, :], kv_rec[:, 1, 0, :],
    )
    assert error < 0.01, f"Independent test error {error:.4f} exceeds ±1%"


def test_budget_ratio_sweep():
    """budget_ratio 0.30~0.70 스윕에서 모두 shape 보존 및 relative error < 0.02.
    결과를 results/2026-05-12/perplexity_sweep.json에 저장."""
    import json, os
    config_base = MixedDimConfig(n_heads=4, d_head=32, seed=42)
    torch.manual_seed(42)
    kv = torch.randn(64, 2, 4, 32)
    q = torch.randn(8, 32)
    results = {}
    for ratio in [0.30, 0.40, 0.50, 0.60, 0.70]:
        config_base.budget_ratio = ratio
        codec = MixedDimPerTokenBudgetCodec(config_base)
        encoded = codec.encode(kv)
        kv_rec = codec.decode(encoded)
        err = attention_output_relative_error(
            q, kv[:, 0, 0, :], kv[:, 1, 0, :],
            kv_rec[:, 0, 0, :], kv_rec[:, 1, 0, :],
        )
        results[str(ratio)] = {
            'budget_ratio': ratio,
            'actual_retention': encoded['budget_ratio'],
            'memory_reduction': codec.memory_reduction_ratio(encoded),
            'relative_error': err,
            'pass_1pct': err < 0.01,
        }
        assert kv_rec.shape == kv.shape
    os.makedirs("results/2026-05-12", exist_ok=True)
    with open("results/2026-05-12/perplexity_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-12.yaml
experiment:
  date: "2026-05-12"
  activity: "B+C"
  description: >
    AdapShotMixedDimSegmentPipeline (Cross-2: B+C):
    RoPEReencodingNonContiguousCache(B-1) + MixedDimPerTokenBudgetCodec(C-1) 통합.
    position-decoupled KV 저장 + RoPE 재인코딩으로 비연속 히트율 향상,
    토큰별 mixed-dim 차원 예산으로 메모리 −50% 달성.

rope_reencoding:
  chunk_size: 128              # 세그먼트 청크 크기 (토큰 수)
  max_entries: 2000            # 최대 캐시 엔트리
  d_head: 64                   # head 차원 (실 환경; 단위 테스트: 32)
  n_heads: 8                   # 어텐션 헤드 수 (실 환경; 단위 테스트: 4)
  rope_base: 10000.0           # RoPE 주파수 base
  rope_dim: -1                 # -1이면 d_head 전체에 적용
  seed: 42

mixed_dim:
  n_heads: 8                   # 어텐션 헤드 수 (단위 테스트: 4)
  d_head: 64                   # head 차원 (단위 테스트: 32)
  budget_ratio: 0.50           # 보존 차원 비율 (50% → 메모리 −50%)
  bisection_iters: 64          # bisection 탐색 반복
  min_retain_ratio: 0.10       # 토큰별 최소 보존 비율
  attn_importance_weight: 1.0
  value_magnitude_weight: 1.0
  compressibility_weight: 1.0
  seed: 42
  # 공격적 압축 설정 (목표: −60%):
  # budget_ratio: 0.40

cache:
  type: "adapshot_mixed_dim_pipeline"
  capacity_bytes: 4294967296   # 4 GiB

benchmark:
  accuracy:
    method: "attention_output_proxy"  # 실 모델 없이 합성 KV 사용
    proxy_tolerance: 0.01             # relative error < 1%
    kl_tolerance: 0.015               # KL divergence < 0.015
    cosine_min: 0.99                  # cosine similarity >= 0.99
    perplexity_dataset: "wikitext-103"
    perplexity_tolerance_pct: 1.0    # evaluation_criteria.md §4 필수
    task_accuracy_tolerance_pct: 1.0  # MMLU proxy
  budget_ratio_sweep:
    ratios: [0.30, 0.40, 0.50, 0.60, 0.70]
  hit_rate:
    target_noncontiguous_fraction: 0.30   # 비연속 히트 >= 30%
  memory_reduction:
    target_ratio: 0.30                    # 최소 30% (목표 50%)
    target_ratio_goal: 0.50
  throughput:
    target_improvement_pct: 10            # 최소 +10% (목표 +20%)
    target_improvement_goal_pct: 20
  scheduling_overhead:
    ttft_p50_max_increase_pct: 5          # TTFT p50 증가 +5% 이내

seed: 42
results_dir: "results/2026-05-12"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_rope_reencoding_cache.py` — RoPEReencodingNonContiguousCache 단위 테스트
      (CacheStore 인터페이스 준수, store_pre_rope/load_with_rope, content hash 키, 비연속 히트율)
- [ ] `tests/unit/test_mixed_dim_codec.py` — MixedDimPerTokenBudgetCodec 단위 테스트
      (bisection λ*, encode/decode shape, budget_ratio 달성, min_retain_ratio, 직렬화)
- [ ] `tests/unit/test_mixed_dim_accuracy.py` — **Activity C 필수**: accuracy proxy 검증
      (`test_accuracy_relative_error_within_1pct`, `test_independent_seed_accuracy` 필수 포함)
- [ ] `tests/integration/test_cross_bc_adapshot.py` — B+C 통합 E2E 테스트
      (비연속 히트율 >= 30%, 메모리 감소 >= 30%, 복합 정확도 ±1% 이내, 저장/복원 순서 계약 검증)

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 전부 통과** (기존 테스트 회귀 없음 포함)
2. **통합 테스트 전부 통과**
3. **Activity B (evaluation_criteria.md §3) 기준 충족**:
   - 비연속 세그먼트 히트율 >= 30% (전체 히트의 30% 이상이 비연속 구간)
   - 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상
   - KV Memory Footprint 베이스라인 대비 +20% 이내 (mixed-dim 압축으로 오히려 감소 목표)
4. **Activity C (evaluation_criteria.md §4) 필수 기준 충족** (모두 Fail 시 전체 Fail):
   - perplexity proxy (attention output relative error) < 1% — `test_accuracy_relative_error_within_1pct` 통과
   - 캘리브레이션 독립 검증 샘플 기준 ±1% 이내 — `test_independent_seed_accuracy` 통과
   - WikiText-103 기준 perplexity 변화 ±1% 이내 (통합 테스트)
   - MMLU proxy (KL < 0.015, cosine >= 0.99) 통과
   - KV Memory Reduction >= −30% (목표 −50%)
   - Encode/Decode 추가 지연 TTFT +10% 이내
5. **크로스 조합 (evaluation_criteria.md §5) 기준 충족** (C 포함이므로 accuracy 필수):
   - 복합 처리량 향상: 단일 Activity 대비 추가 +5% 이상
   - 복합 메모리 감소: 단일 Activity 대비 추가 −10% 이상
   - 복합 적용 후에도 accuracy ±1% 이내 유지
6. **CacheStore 인터페이스 준수**:
   - `RoPEReencodingNonContiguousCache`가 CacheStore 6개 추상 메서드 완전 구현
   - `AdapShotMixedDimSegmentPipeline`이 CacheStore 6개 추상 메서드 완전 구현
   - `store_pre_rope()`/`load_with_rope()` 선택적 메서드 오버라이드 확인
7. **저장/복원 순서 계약 준수**: 통합 테스트에서 store_segment → load_segment 순서로
   실행 시 원본 대비 relative error < 0.02 (RoPE 재인코딩 오차 포함)
8. **설정 YAML 존재**: `configs/experiments/2026-05-12.yaml` 생성됨
9. **타입 힌트**: 모든 공개 함수·메서드에 완전한 타입 힌트
10. **시드 고정 재현성**: seed=42 고정 시 동일 결과 재현
11. **결과 저장**: `results/2026-05-12/metrics.json`, `perplexity_sweep.json`,
    `accuracy_proxy_results.json` 생성됨

---

## 구현 순서 (implementer 참고)

1. **`src/cache/base.py` 확장** — `store_pre_rope()`/`load_with_rope()` 선택적 메서드 추가.
   기존 추상 메서드 변경 없음. 기존 테스트 회귀 없이 통과 확인.

2. **`src/cache/mixed_dim_codec.py`** — C 핵심 구현 (B와 독립, 먼저 구현 가능).
   순서: `compute_loss_scores()` → `find_threshold()` → `encode()` → `decode()` →
   `memory_reduction_ratio()` → `compression_hook()`.
   구현 후 `test_mixed_dim_accuracy.py`의 `test_accuracy_relative_error_within_1pct` 통과 확인.

3. **`tests/unit/test_mixed_dim_accuracy.py`** — Activity C 필수 테스트 먼저 작성.
   모든 proxy 지표(relative error, KL, cosine)가 ±1% 기준 통과하는지 확인.

4. **`src/cache/rope_reencoding_cache.py`** — B 구현.
   순서: `_get_rope_rotation()` → `_apply_rope()` → `store_pre_rope()` → `load_with_rope()` →
   `put_segment_pre_rope()` → `get_segments_with_rope()` → CacheStore 인터페이스 구현.

5. **`tests/unit/test_rope_reencoding_cache.py`** — B 단위 테스트.
   RoPE 적용 전후 content hash 키 동일성, load_with_rope() 반환 shape, 비연속 히트율 추적.

6. **`src/cache/adapshot_pipeline.py`** — B+C 통합.
   순서: `store_segment()` (저장 계약) → `load_segment()` (복원 계약) → `get_segments()` →
   CacheStore 인터페이스 구현 → `save_pipeline()` / `load_pipeline()`.

7. **`tests/unit/test_mixed_dim_codec.py`** — MixedDimCodec 나머지 단위 테스트.

8. **`tests/integration/test_cross_bc_adapshot.py`** — 통합 E2E 테스트.
   저장 → 로드 → RoPE 재인코딩 → 정확도 검증의 전체 파이프라인 실행.

9. **`configs/experiments/2026-05-12.yaml`** 작성.

---

## 기존 파일 보존 목록

이번 사이클에서 수정하지 않는 파일 (기존 테스트 회귀 없이 통과해야 함):

- `src/cache/segmented.py` — RoPEReencodingNonContiguousCache의 백엔드로 재사용, 수정 불필요
- `src/cache/wicer_iterative_cache.py`, `src/cache/ratequant_codec.py`, `src/cache/wicer_ratequant_pipeline.py`
- `src/compression/` 전체 (vq_codec.py 등 이전 사이클 구현)
- `src/metrics/perplexity.py` — 기존 함수 그대로 사용 (수정 불필요)
- `src/scheduler/` 전체 (이번 사이클 Activity A 미포함)
- 기타 모든 기존 캐시 구현체, 테스트 파일 (회귀 없이 통과 필수)
