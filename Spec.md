<!-- 변경 이유 (이전 Spec.md: 2026-05-10 대비):
이전 사이클(2026-05-10)은 B+C (KVPacketSoftAdapterCache + RSimVQCodec +
ContextFreeCompressedKVPacket) 조합이었다. 이번 사이클은 B+C
(WiCERIterativeKVWikiCache + RateQuantReverseWaterfillingCodec + WiCER-RateQuantPipeline)
조합으로 전환한다.

주요 변경:
1. [Activity C 교체] RSimVQCodec(잔차 벡터 양자화) → RateQuantReverseWaterfillingCodec
   (정보 이론의 역 물 채우기 알고리즘 기반 헤드별 최적 비트 할당).
   알고리즘 패러다임이 VQ(이산 코드북) → 율-왜곡 이론(연속 비트 예산 최적화)으로
   본질적으로 다르다.

2. [Activity B 교체] KVPacketSoftAdapterCache(소프트-토큰 어댑터 재계산-프리) →
   WiCERIterativeKVWikiCache(CEGAR 반복 컴파일 도메인 KV 아티팩트).
   B 핵심 알고리즘이 "맥락 독립 어댑터" → "반복 컴파일 보장 커버리지"로 전환.

3. [Cross-1 신규] WiCER-RateQuantPipeline: B(CEGAR KV 아티팩트) + C(역 물 채우기 양자화)
   직접 통합. CEGAR 정제 단계에서 재컴파일된 청크에도 즉시 최적 양자화 적용.
   헤드별 r_h 메타데이터를 아티팩트와 함께 직렬화하여 로드 시 재양자화 오버헤드 제거.

4. [보존 파일] 이전 사이클 파일(kv_packet_adapter.py, context_free_compressed_packet.py,
   src/compression/ 전체, 기타 모든 기존 파일)은 이번 사이클에서 수정하지 않는다.
   기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-11

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-11.md`
**최우선 구현 타겟**: Cross-1 (B+C) — WiCERIterativeKVWikiCache(B-1) +
RateQuantReverseWaterfillingCodec(C-3) + WiCER-RateQuantPipeline 통합

**해결하려는 문제**:
- Activity B: 표준 KV 캐시는 prefix가 byte-identical할 때만 재사용된다.
  WiCER(CEGAR 반복 컴파일)은 도메인 문서를 KV 아티팩트로 사전 구축하고,
  검증 쿼리의 미스 청크(반례)를 기반으로 청크 분할을 반복 정제하여
  커버리지가 보장된 영구 비연속 KV 캐시를 구축한다. TTFT −40% (도메인 반복 재계산 제거).

- Activity C: 기존 균일 양자화(INT8 등)는 모든 어텐션 헤드에 동일 비트를 할당하여
  분산이 낮은 헤드를 과도 압축하거나 분산이 높은 헤드를 과소 압축한다.
  RateQuant의 역 물 채우기 알고리즘은 캘리브레이션 데이터로 측정한 헤드별 분산에
  따라 총 비트 예산을 이론적으로 최적 분배한다. 2.5비트 평균에서 perplexity ±0.3%
  유지(균일 4비트 대비 perplexity −70% 개선) 목표.

- Cross-1 시너지: WiCER 아티팩트는 크기가 크고 장기 보존이 필요하므로 RateQuant 압축이
  직접 적용된다. CEGAR 정제 단계에서 재컴파일된 청크도 즉시 최적 비트 할당으로 양자화되어
  아티팩트 증가를 억제한다. 예상 복합 효과: 처리량 +35%, 메모리 −55%, 정확도 ±0.5%.

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling
- [x] Activity B: Non-Contiguous KV Cache Reuse
- [x] Activity C: KV Cache Compression

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §3 Activity B): 비연속 세그먼트 히트율 >= 30%
      (전체 히트 중 비연속 구간 발생 비율)
- [ ] 목표 2 (evaluation_criteria.md §3 Activity B): 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상;
      도메인 쿼리 기준 목표치 +20%p
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      (WikiText-2 기준, 압축 전후 비교)
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 변화 ±1% 이내
      (합성 attention-output 프록시 기준; 실 LongBench는 통합 테스트에서 선택적 수행)
- [ ] 목표 5 (evaluation_criteria.md §4 Activity C): KV Memory Reduction >= −30%
      (베이스라인 대비); 목표치 −55%
- [ ] 목표 6 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +10% 이상;
      목표치 +35% (CEGAR 도메인 TTFT 단축 + RateQuant 압축)
- [ ] 목표 7 (evaluation_criteria.md §5 크로스 조합): 단일 Activity 대비 복합 처리량 추가 +5% 이상
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합): 단일 Activity 대비 복합 메모리 감소 추가 −10% 이상

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/wicер_iterative_cache.py` | B | WiCERIterativeKVWikiCache — CEGAR 반복 컴파일 도메인 KV 아티팩트 캐시 |
| `src/cache/ratequant_codec.py` | C | RateQuantReverseWaterfillingCodec — 역 물 채우기 헤드별 최적 비트 할당 양자화 |
| `src/cache/wicер_ratequant_pipeline.py` | B+C | WiCERRateQuantPipeline — B 아티팩트 + C 양자화 통합 파이프라인 |
| `tests/unit/test_wicер_iterative_cache.py` | B | CEGAR 루프 단위 테스트 (히트율 단조 증가, 반례 수집, 직렬화) |
| `tests/unit/test_ratequant_codec.py` | C | 역 물 채우기 단위 테스트 (비트 할당, 양자화, 압축률) |
| `tests/unit/test_ratequant_accuracy.py` | C | **Activity C 필수** — 압축 정확도 보존 검증 (perplexity proxy ±1%) |
| `tests/integration/test_cross_bc_wicер_ratequant.py` | B+C | B+C 통합 E2E 테스트 (히트율 + 메모리 감소 + 정확도 보존) |
| `configs/experiments/2026-05-11.yaml` | 공통 | 이번 사이클 실험 설정 |

**참고**: 파일명에서 `wicер`는 `wicer`로 표기한다 (ASCII 한정).
실제 파일 경로:
- `src/cache/wicer_iterative_cache.py`
- `src/cache/ratequant_codec.py`
- `src/cache/wicer_ratequant_pipeline.py`
- `tests/unit/test_wicer_iterative_cache.py`
- `tests/unit/test_ratequant_codec.py`
- `tests/unit/test_ratequant_accuracy.py`
- `tests/integration/test_cross_bc_wicer_ratequant.py`

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/metrics/perplexity.py` (신규 생성) | perplexity 측정 유틸리티 — compute_attention_ppl_proxy() 추가. 실 모델 없이 합성 KV로 attention-output MSE 기반 perplexity proxy 계산. |

**보존 파일 (이번 사이클에서 수정하지 않음)**:
- `src/cache/base.py` — CacheStore 인터페이스 변경 없음 (compression_hook 이미 추가됨)
- `src/cache/segmented.py` — 기존 그대로 사용
- `src/compression/` 전체 — VQCodec 이전 사이클 구현 보존
- `src/cache/kv_packet_adapter.py`, `src/cache/context_free_compressed_packet.py` — 보존
- 기타 모든 기존 캐시 구현체, 스케줄러, 테스트 파일 — 회귀 없이 통과 필수

---

## 알고리즘 상세

### [RateQuantReverseWaterfillingCodec] (Activity C) — `src/cache/ratequant_codec.py`

정보 이론의 역 물 채우기(Reverse Water-Filling) 알고리즘으로 각 어텐션 헤드의 KV 채널별
분산에 비례한 최적 비트 수를 할당한다. 총 비트 예산 R이 고정일 때 총 재구성 왜곡의
이론적 하한을 달성하는 해다.

**알고리즘 (역 물 채우기)**:
```
주어진 입력:
  head_variances: [n_heads] float — 각 헤드의 KV 채널 분산
  total_bit_budget: float — 총 비트 예산 (예: n_heads * 4비트)

역 물 채우기 공식:
  r_h = max(0, (1/2) * log2(sigma²_h / lambda))

여기서 lambda는 총 예산을 맞추는 라그랑지 승수:
  sum_h max(0, (1/2) * log2(sigma²_h / lambda)) = total_bit_budget

이진 탐색으로 lambda를 찾은 후 각 헤드의 비트 수 r_h를 계산한다.
r_h를 정수로 반올림하고 최소 2비트, 최대 8비트로 클램핑한다.
```

**캘리브레이션 단계 (512 샘플)**:
```python
# 캘리브레이션: 각 헤드의 KV 채널 분산 측정
for each calibration_sample:
    for each layer l:
        kv_l = sample[l]  # [n_tokens, 2, n_heads, d_head]
        for each head h:
            channel_var[l][h] += kv_l[:, :, h, :].var(dim=-1).mean()
channel_var /= n_calibration_samples
```

**양자화 적용**:
- r_h 비트폭에 따라 INT2 ~ INT8 균일 양자화 적용
- r_h=2 → INT2 (INT4 packed로 저장), r_h=4 → INT4, r_h=8 → INT8
- 역양자화: 저장된 scale + bias로 FP16 복원

```python
# 의사코드 — src/cache/ratequant_codec.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import math


@dataclass
class RateQuantConfig:
    n_heads: int = 8               # 어텐션 헤드 수
    n_layers: int = 12             # 모델 레이어 수
    d_head: int = 64               # head 차원
    total_bit_budget: float = 4.0  # 평균 비트 예산 (헤드당, 기본 4비트)
    min_bits: int = 2              # 헤드당 최소 비트
    max_bits: int = 8              # 헤드당 최대 비트
    calibration_samples: int = 512 # 캘리브레이션 샘플 수
    seed: int = 42


class RateQuantReverseWaterfillingCodec:
    """역 물 채우기 최적 비트 할당 KV 양자화 코덱 (RateQuant arXiv 2025).

    CacheStore.compression_hook() 과 독립 encode/decode() 양방향 사용 가능.
    캘리브레이션(calibrate()) 후 각 레이어·헤드에 최적 비트폭을 할당한다.
    """

    def __init__(self, config: RateQuantConfig) -> None:
        self.config = config
        # [layer_idx][head_idx] → bit_width (int)
        self.bit_allocation: Dict[int, List[int]] = {}
        # [layer_idx][head_idx] → (scale, zero_point) Tensor
        self.scales: Dict[Tuple[int, int], torch.Tensor] = {}
        self.zero_points: Dict[Tuple[int, int], torch.Tensor] = {}
        # 캘리브레이션으로 측정된 헤드별 분산 [n_layers, n_heads]
        self.head_variances: Optional[torch.Tensor] = None
        self._calibrated: bool = False

    def calibrate(
        self,
        calibration_kvs: List[torch.Tensor],
        # List of [n_tokens, 2, n_heads, d_head] float tensors, length = calibration_samples
    ) -> None:
        """512 캘리브레이션 샘플로 헤드별 분산 측정 후 역 물 채우기 비트 할당 계산.

        의사코드:
          1. 각 샘플의 [n_tokens, 2, n_heads, d_head] KV에서 head h의 분산:
               var_h = KV[:, :, h, :].var() (key + value 합산)
          2. 모든 샘플 분산의 평균 → head_variances [n_heads]
          3. _reverse_waterfilling(head_variances, total_bit_budget) 호출
          4. 결과를 self.bit_allocation[layer_idx=0][h] 에 저장
             (layer-independent 단순화; layer별 캘리브레이션은 멀티 레이어 호출로 확장)
        """
        ...

    def calibrate_layer(
        self,
        calibration_kvs: List[torch.Tensor],  # [n_tokens, 2, n_heads, d_head] × N
        layer_idx: int,
    ) -> None:
        """단일 레이어에 대해 캘리브레이션 수행. layer별 독립 비트 할당 지원."""
        ...

    @staticmethod
    def _reverse_waterfilling(
        variances: torch.Tensor,   # [n_heads] float
        total_budget: float,       # 총 비트 예산 (n_heads * avg_bits)
        min_bits: int = 2,
        max_bits: int = 8,
    ) -> List[int]:
        """역 물 채우기 알고리즘.

        이진 탐색으로 라그랑지 승수 lambda를 찾는다:
          r_h = max(0, (1/2) * log2(sigma²_h / lambda))
          sum_h r_h = total_budget

        의사코드:
          lo, hi = 1e-10, variances.max().item()
          for _ in range(100):  # 이진 탐색
              mid = (lo + hi) / 2
              bits = [max(0, 0.5 * log2(v / mid)) for v in variances]
              if sum(bits) > total_budget: hi = mid
              else: lo = mid
          bits_int = [clamp(round(b), min_bits, max_bits) for b in bits]
          return bits_int
        """
        ...

    def encode(
        self,
        kv: torch.Tensor,    # [n_tokens, 2, n_heads, d_head] float16
        layer_idx: int,
    ) -> dict:
        """반환:
        {
          'quantized': List[torch.Tensor],  # 헤드별 int8/int4 텐서 [n_tokens, 2, d_head]
          'scales':    List[torch.Tensor],  # 헤드별 scale [2, d_head] 또는 scalar
          'zero_pts':  List[torch.Tensor],  # 헤드별 zero_point
          'bit_widths': List[int],          # 헤드별 비트폭
          'layer_idx': int,
          'n_tokens': int,
          'n_heads': int,
        }

        의사코드:
          if not calibrated: raise RuntimeError
          result = {'quantized': [], 'scales': [], 'zero_pts': [], 'bit_widths': []}
          for h in range(n_heads):
              bits = bit_allocation[layer_idx][h]
              kv_h = kv[:, :, h, :]       # [n_tokens, 2, d_head]
              scale, zero_pt = _compute_scale(kv_h, bits)
              q = _quantize(kv_h, scale, zero_pt, bits)  # int8 저장 (int2/int4도 int8에 패킹)
              result['quantized'].append(q)
              result['scales'].append(scale)
              result['zero_pts'].append(zero_pt)
              result['bit_widths'].append(bits)
          return result
        """
        ...

    def decode(
        self,
        encoded: dict,
        layer_idx: int,
    ) -> torch.Tensor:
        """반환: [n_tokens, 2, n_heads, d_head] float16

        의사코드:
          tensors = []
          for h, (q, scale, zero_pt, bits) in enumerate(zip(...)):
              kv_h = _dequantize(q, scale, zero_pt, bits)  # float32
              tensors.append(kv_h)
          return torch.stack(tensors, dim=2).half()
        """
        ...

    @staticmethod
    def _compute_scale(
        kv_h: torch.Tensor,  # [n_tokens, 2, d_head]
        bits: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """per-channel min-max scale 계산.
        scale = (max - min) / (2^bits - 1)
        zero_point = round(-min / scale)
        """
        ...

    @staticmethod
    def _quantize(
        kv_h: torch.Tensor,   # [n_tokens, 2, d_head] float
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bits: int,
    ) -> torch.Tensor:
        """균일 양자화. 결과는 int8 텐서로 저장 (bits<=4이면 int8에 2값 패킹 없이 저장)."""
        ...

    @staticmethod
    def _dequantize(
        q: torch.Tensor,      # int8
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bits: int,
    ) -> torch.Tensor:
        """역양자화 → float32."""
        ...

    def compression_ratio(self, layer_idx: int) -> float:
        """실효 압축률.
        FP16(16비트) 기준 평균 비트폭 avg_bits 으로:
          ratio = 1.0 - avg_bits / 16.0
        예: avg 4비트 → ratio = 0.75 (75% 감소)
        """
        ...

    def memory_bytes(
        self,
        encoded: dict,
    ) -> int:
        """인코딩된 dict의 실제 메모리 바이트 계산.
        quantized 텐서 bytes + scale/zero_pt float32 bytes 합산."""
        ...

    def save_calibration(self, path: str) -> None:
        """bit_allocation + head_variances를 torch.save로 직렬화."""
        ...

    def load_calibration(self, path: str) -> None:
        """저장된 캘리브레이션 결과 복원."""
        ...
```

---

### [WiCERIterativeKVWikiCache] (Activity B) — `src/cache/wicer_iterative_cache.py`

CEGAR(Counterexample-Guided Abstraction Refinement) 패턴을 KV 캐시 구축에 적용한다.
도메인 문서 코퍼스를 KV 아티팩트로 컴파일하고, 검증 쿼리의 미스 청크를 반례로 수집하여
청크 분할을 반복 정제한다. `SegmentedHashCache`를 내부 스토어로 재사용한다.

**CEGAR 루프 (3단계)**:
```
1. 컴파일: 도메인 문서 → 청크 분할 → KV 사전 계산 → SegmentedHashCache 저장
2. 평가: 검증 쿼리 실행 → 미스 청크 수집 (반례)
3. 정제: 반례 청크를 더 세밀한 서브청크로 분할하거나 인접 청크와 병합 → 재컴파일
4. 종료: hit_rate >= target_hit_rate OR iterations >= max_iterations
```

```python
# 의사코드 — src/cache/wicer_iterative_cache.py

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class WiCERConfig:
    chunk_size: int = 128           # 초기 청크 크기 (토큰 수)
    min_chunk_size: int = 16        # 정제 시 최소 청크 크기
    max_chunk_size: int = 512       # 병합 시 최대 청크 크기
    target_hit_rate: float = 0.80   # CEGAR 종료 조건: 히트율 목표
    max_iterations: int = 5         # 최대 CEGAR 반복 횟수
    max_entries: int = 2000         # 캐시 최대 엔트리 수
    seed: int = 42


class WiCERIterativeKVWikiCache(CacheStore):
    """CEGAR 반복 컴파일 도메인 KV 아티팩트 캐시 (Activity B).

    CacheStore 인터페이스 완전 구현.
    내부적으로 SegmentedHashCache를 스토어로 사용한다.

    사용 흐름:
      1. compile_corpus(docs, kv_fn) — 초기 KV 아티팩트 구축
      2. cegar_refine(val_queries, kv_fn) — 반복 정제 (내부에서 컴파일/평가/정제)
      3. put/get — CacheStore 표준 인터페이스로 접근
    """

    def __init__(self, config: WiCERConfig) -> None:
        self.config = config
        # 현재 청크 크기 스케줄: doc_id → chunk_size (정제 후 문서별로 다를 수 있음)
        self._chunk_sizes: Dict[str, int] = {}
        # 내부 세그먼트 캐시 (SegmentedHashCache를 백엔드로 사용)
        self._store = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._cegar_iteration = 0
        self._counterexamples: List[Tuple[str, int]] = []  # (doc_id, chunk_idx)
        # CEGAR 히트율 이력 (단조 증가 검증용)
        self._hit_rate_history: List[float] = []

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """key=doc_id, value=kv_block [n_tokens, 2, n_heads, d_head].
        내부적으로 SegmentedHashCache.put()으로 위임."""
        self._store.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """SegmentedHashCache.get() 위임. hit/miss 카운터 갱신."""
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
    # CEGAR API                                                            #
    # ------------------------------------------------------------------ #

    def compile_corpus(
        self,
        docs: Dict[str, List[int]],    # doc_id → token_ids
        kv_fn: callable,               # kv_fn(token_ids, layer_idx) → Tensor [n_tokens, hidden]
        layer_idx: int = 0,
        codec: Optional[object] = None,  # RateQuantReverseWaterfillingCodec (선택적)
    ) -> None:
        """초기 컴파일 단계: 도메인 문서를 현재 청크 크기로 분할하고 KV 사전 계산.

        의사코드:
          for doc_id, token_ids in docs.items():
              chunk_size = self._chunk_sizes.get(doc_id, self.config.chunk_size)
              n_chunks = ceil(len(token_ids) / chunk_size)
              for chunk_idx in range(n_chunks):
                  chunk_tokens = token_ids[chunk_idx*cs : (chunk_idx+1)*cs]
                  kv = kv_fn(chunk_tokens, layer_idx)  # [len(chunk_tokens), hidden]
                  if codec is not None:
                      kv = codec.compression_hook(f"{doc_id}_{chunk_idx}", kv)
                  self._store.put_segment(token_ids, chunk_idx, kv, layer_idx)
        """
        ...

    def evaluate(
        self,
        val_queries: List[List[int]],  # 검증 쿼리 token_ids 리스트
        layer_idx: int = 0,
    ) -> Tuple[float, List[Tuple[str, int]]]:
        """평가 단계: 검증 쿼리 실행, 미스 청크를 반례로 수집.

        반환: (hit_rate, counterexamples)
          counterexamples: List[(query_prefix_hash, miss_chunk_idx)]

        의사코드:
          total_hits, total_misses = 0, 0
          counterexamples = []
          for token_ids in val_queries:
              hits, misses = self._store.get_segments(token_ids, layer_idx)
              total_hits += len(hits)
              total_misses += len(misses)
              for miss_chunk_idx in misses:
                  counterexamples.append((hash_prefix(token_ids), miss_chunk_idx))
          hit_rate = total_hits / (total_hits + total_misses) if total > 0 else 0
          return hit_rate, counterexamples
        """
        ...

    def refine(
        self,
        counterexamples: List[Tuple[str, int]],
        docs: Dict[str, List[int]],
        kv_fn: callable,
        layer_idx: int = 0,
        codec: Optional[object] = None,
    ) -> None:
        """정제 단계: 반례 청크를 더 세밀한 서브청크로 분할하거나 인접 청크와 병합.

        의사코드:
          refined_chunks = set()
          for (query_hash, miss_chunk_idx) in counterexamples:
              # 해당 청크의 doc_id를 역추적하거나 all-docs 스캔으로 처리
              for doc_id, token_ids in docs.items():
                  cur_chunk_size = self._chunk_sizes.get(doc_id, self.config.chunk_size)
                  # 반례 청크가 이 문서에 속하는지 확인
                  if miss_chunk_idx * cur_chunk_size < len(token_ids):
                      new_chunk_size = max(self.config.min_chunk_size, cur_chunk_size // 2)
                      self._chunk_sizes[doc_id] = new_chunk_size
                      refined_chunks.add(doc_id)
          # 정제된 문서를 재컴파일
          refined_docs = {k: v for k, v in docs.items() if k in refined_chunks}
          self.compile_corpus(refined_docs, kv_fn, layer_idx, codec)
        """
        ...

    def cegar_refine(
        self,
        docs: Dict[str, List[int]],
        val_queries: List[List[int]],
        kv_fn: callable,
        layer_idx: int = 0,
        codec: Optional[object] = None,
    ) -> None:
        """CEGAR 메인 루프: 컴파일 → 평가 → 정제 반복.

        의사코드:
          self.compile_corpus(docs, kv_fn, layer_idx, codec)
          for iteration in range(self.config.max_iterations):
              self._cegar_iteration = iteration
              hit_rate, counterexamples = self.evaluate(val_queries, layer_idx)
              self._hit_rate_history.append(hit_rate)
              if hit_rate >= self.config.target_hit_rate:
                  break  # 목표 달성 → 현재 상태를 "검증된 추상화"로 고정
              if not counterexamples:
                  break  # 반례 없음 → 더 이상 정제 불가
              self.refine(counterexamples, docs, kv_fn, layer_idx, codec)
        """
        ...

    def noncontiguous_hit_rate(self) -> float:
        """비연속 세그먼트 히트율 (SegmentedHashCache 위임)."""
        return self._store.noncontiguous_hit_rate()

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """SegmentedHashCache.get_segments() 위임 (runner.py 호환성)."""
        return self._store.get_segments(token_ids, layer_idx)

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """SegmentedHashCache.put_segment() 위임 (runner.py 호환성)."""
        self._store.put_segment(token_ids, chunk_idx, kv, layer_idx)

    def cegar_hit_rate_history(self) -> List[float]:
        """CEGAR 반복별 히트율 이력 (단조 증가 검증용)."""
        return list(self._hit_rate_history)

    def save_artifacts(self, path: str) -> None:
        """KV 아티팩트 + 청크 크기 스케줄 + CEGAR 이력을 직렬화.

        torch.save({
            'store_state': self._store._store,  # OrderedDict
            'chunk_sizes': self._chunk_sizes,
            'hit_rate_history': self._hit_rate_history,
            'cegar_iteration': self._cegar_iteration,
        }, path)
        """
        ...

    def load_artifacts(self, path: str) -> None:
        """저장된 아티팩트 복원."""
        ...
```

---

### [WiCERRateQuantPipeline] (Activity B+C) — `src/cache/wicer_ratequant_pipeline.py`

`WiCERIterativeKVWikiCache`(B)와 `RateQuantReverseWaterfillingCodec`(C)을 통합하는
파이프라인. CEGAR 루프의 각 컴파일/정제 단계에서 RateQuant 압축을 즉시 적용한다.
헤드별 비트 할당 메타데이터(r_h)를 아티팩트와 함께 직렬화하여 로드 시 재양자화 없이
즉시 사용 가능하다.

```python
# 의사코드 — src/cache/wicer_ratequant_pipeline.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.wicer_iterative_cache import WiCERIterativeKVWikiCache, WiCERConfig
from src.cache.ratequant_codec import RateQuantReverseWaterfillingCodec, RateQuantConfig


@dataclass
class WiCERRateQuantConfig:
    wicer: WiCERConfig = None
    ratequant: RateQuantConfig = None

    def __post_init__(self):
        if self.wicer is None:
            self.wicer = WiCERConfig()
        if self.ratequant is None:
            self.ratequant = RateQuantConfig()


class WiCERRateQuantPipeline(CacheStore):
    """B+C 통합 파이프라인: CEGAR KV 아티팩트 + 역 물 채우기 양자화.

    CacheStore 인터페이스 완전 구현 (WiCERIterativeKVWikiCache에 위임).

    주요 특성:
    - CEGAR 루프 내 컴파일/정제 시 RateQuant 양자화 자동 적용
    - 헤드별 비트 할당(r_h) 메타데이터를 아티팩트와 함께 저장/로드
    - compression_hook()으로 CacheStore 표준 인터페이스와 통합
    """

    def __init__(self, pipeline_config: WiCERRateQuantConfig) -> None:
        self.config = pipeline_config
        self.wicer = WiCERIterativeKVWikiCache(pipeline_config.wicer)
        self.codec = RateQuantReverseWaterfillingCodec(pipeline_config.ratequant)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """compression_hook으로 RateQuant 압축 후 WiCER 저장."""
        compressed = self.compression_hook(key, value)
        self.wicer.put(key, compressed)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.wicer.get(key)

    def evict(self) -> int:
        return self.wicer.evict()

    def hit_rate(self) -> float:
        return self.wicer.hit_rate()

    def memory_bytes(self) -> int:
        return self.wicer.memory_bytes()

    def reset_stats(self) -> None:
        self.wicer.reset_stats()
        self._hits = 0
        self._misses = 0

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,   # [n_tokens, 2, n_heads, d_head]
    ) -> torch.Tensor:
        """RateQuant 인코딩 후 디코딩 결과 반환 (정밀도 손실 포함).
        codec이 캘리브레이션되지 않은 경우 원본 그대로 반환 (graceful fallback)."""
        if not self.codec._calibrated:
            return value
        encoded = self.codec.encode(value, layer_idx=0)
        return self.codec.decode(encoded, layer_idx=0)

    # ------------------------------------------------------------------ #
    # B+C 파이프라인 API                                                   #
    # ------------------------------------------------------------------ #

    def build_pipeline(
        self,
        docs: Dict[str, List[int]],
        val_queries: List[List[int]],
        kv_fn: callable,
        calibration_kvs: Optional[List[torch.Tensor]] = None,
        layer_idx: int = 0,
    ) -> None:
        """전체 파이프라인 실행:
          1. calibration_kvs로 RateQuant 캘리브레이션 (제공 시)
          2. CEGAR 루프 실행 (내부에서 RateQuant 압축 적용)

        의사코드:
          if calibration_kvs:
              self.codec.calibrate(calibration_kvs)
          self.wicer.cegar_refine(docs, val_queries, kv_fn, layer_idx, self.codec)
        """
        ...

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """WiCER.get_segments() 위임 (runner.py 호환성)."""
        return self.wicer.get_segments(token_ids, layer_idx)

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """WiCER.put_segment() 위임 (runner.py 호환성)."""
        self.wicer.put_segment(token_ids, chunk_idx, kv, layer_idx)

    def noncontiguous_hit_rate(self) -> float:
        return self.wicer.noncontiguous_hit_rate()

    def cegar_hit_rate_history(self) -> List[float]:
        return self.wicer.cegar_hit_rate_history()

    def save_pipeline(self, path: str) -> None:
        """WiCER 아티팩트 + RateQuant 캘리브레이션 데이터를 하나의 파일에 직렬화.

        torch.save({
            'wicer_store': wicer._store._store,
            'wicer_chunk_sizes': wicer._chunk_sizes,
            'wicer_hit_rate_history': wicer._hit_rate_history,
            'codec_bit_allocation': codec.bit_allocation,
            'codec_head_variances': codec.head_variances,
        }, path)
        비트 메타데이터(r_h)가 함께 저장되므로 로드 시 재양자화 오버헤드 없음.
        """
        ...

    def load_pipeline(self, path: str) -> None:
        """저장된 파이프라인 복원."""
        ...
```

---

### [PerplexityProxy 유틸리티] (공통) — `src/metrics/perplexity.py`

실 모델 없이 합성 KV 텐서 기반 attention-output MSE로 perplexity 변화를 근사한다.
Activity C의 accuracy-preserving 검증에 사용된다.

```python
# 의사코드 — src/metrics/perplexity.py

from typing import Optional
import torch
import torch.nn.functional as F


def compute_attention_output(
    query: torch.Tensor,   # [n_q, d_head]
    key: torch.Tensor,     # [n_kv, d_head]
    value: torch.Tensor,   # [n_kv, d_head]
) -> torch.Tensor:
    """Scaled dot-product attention. 반환: [n_q, d_head]."""
    scale = query.size(-1) ** -0.5
    scores = (query @ key.T) * scale
    attn = F.softmax(scores, dim=-1)
    return attn @ value


def attention_output_relative_error(
    q: torch.Tensor,
    k_orig: torch.Tensor,
    v_orig: torch.Tensor,
    k_comp: torch.Tensor,
    v_comp: torch.Tensor,
) -> float:
    """압축 전후 attention output의 상대 오차 (0.0–1.0).
    이 값이 0.01 미만이면 ±1% perplexity 보존 기준을 통과한다고 간주."""
    out_orig = compute_attention_output(q, k_orig, v_orig)
    out_comp = compute_attention_output(q, k_comp, v_comp)
    return ((out_orig - out_comp).norm() / out_orig.norm().clamp(min=1e-8)).item()


def attention_kl_divergence(
    q: torch.Tensor,
    k_orig: torch.Tensor,
    k_comp: torch.Tensor,
) -> float:
    """압축 전후 attention score distribution의 KL 발산.
    0.015 미만이면 ±1% perplexity 근사 보존."""
    scale = q.size(-1) ** -0.5
    attn_orig = F.softmax((q @ k_orig.T) * scale, dim=-1)
    attn_comp = F.softmax((q @ k_comp.T) * scale, dim=-1)
    kl = F.kl_div(
        attn_comp.log().clamp(min=-100),
        attn_orig,
        reduction="batchmean",
    ).item()
    return kl


def cosine_similarity_output(
    q: torch.Tensor,
    k_orig: torch.Tensor,
    v_orig: torch.Tensor,
    k_comp: torch.Tensor,
    v_comp: torch.Tensor,
) -> float:
    """압축 전후 attention output의 코사인 유사도 (0.99 이상이면 ±1% 보존)."""
    out_orig = compute_attention_output(q, k_orig, v_orig).flatten()
    out_comp = compute_attention_output(q, k_comp, v_comp).flatten()
    return F.cosine_similarity(
        out_orig.unsqueeze(0), out_comp.unsqueeze(0)
    ).item()
```

---

## Activity C — Accuracy Preservation 검증 계획

**Activity C(RateQuantReverseWaterfillingCodec)를 포함하므로 이 섹션은 필수다.**
**이 계획 없이 Spec.md는 불완전하다.**

### perplexity 측정

| 항목 | 내용 |
|------|------|
| 측정 방법 | 실 모델 없이 합성 attention output MSE 기반 perplexity proxy 사용 (단위 테스트용). 실 모델 사용 시 WikiText-2 test split으로 GPT-2 small 측정 (선택적). |
| 데이터셋 | 합성: torch.randn 기반 KV 텐서 512 샘플 (캘리브레이션용) + 128 샘플 (검증용). 실 모델: WikiText-2 test split (stride=512, max_length=2048). |
| 허용 오차 | attention output relative error < 0.01 (1% 이내) — evaluation_criteria.md §4 필수 항목 |
| 비트폭 스윕 | avg_bits = 2.5, 3.0, 4.0, 6.0, 8.0 각각 측정 |
| 캘리브레이션 독립 검증 | 캘리브레이션에 사용하지 않은 독립 128 샘플로 정확도 측정 |
| 결과 저장 | `results/2026-05-11/perplexity_sweep.json` |

### 태스크 정확도 측정 (합성 proxy)

| 항목 | 내용 |
|------|------|
| proxy 지표 1 | attention output cosine similarity >= 0.99 (압축 전후) |
| proxy 지표 2 | attention score KL divergence < 0.015 (압축 전후) |
| proxy 지표 3 | relative output error < 0.01 (1% 이내) |
| 허용 오차 | evaluation_criteria.md §4 필수: ±1% 이내 |
| 비트폭별 검증 | avg_bits = 2.5, 4.0 각각에서 모든 proxy 지표 통과 확인 |
| 결과 저장 | `results/2026-05-11/accuracy_proxy_results.json` |

### 압축률-정확도 트레이드오프 예상값

| avg_bits | 메모리 감소율 | 예상 relative error | 예상 KL |
|---------|------------|-------------------|---------|
| 8.0 (INT8 균일) | 50% | < 0.001 | < 0.001 |
| 4.0 (역 물 채우기) | 75% | < 0.005 | < 0.01 |
| 2.5 (역 물 채우기) | 84% | < 0.01 | < 0.015 |
| 2.0 (INT2 하한) | 87.5% | < 0.015 | < 0.02 |

**역 물 채우기가 균일 할당보다 우월한 이유**: 분산이 낮은 헤드에는 더 적은 비트를 할당하고
분산이 높은 헤드에는 더 많은 비트를 할당하므로 동일 총 비트 예산에서 총 재구성 왜곡이 최소화된다.
RateQuant 원논문에서 Qwen3-8B 2.5비트 시 perplexity 49.3→14.9(균일 대비 크게 낮음)를 실증.

### 검증 테스트 파일 — `tests/unit/test_ratequant_accuracy.py`

```python
# 필수 테스트 케이스 — tests/unit/test_ratequant_accuracy.py

import pytest
import torch
from src.cache.ratequant_codec import RateQuantReverseWaterfillingCodec, RateQuantConfig
from src.metrics.perplexity import (
    attention_output_relative_error,
    attention_kl_divergence,
    cosine_similarity_output,
)


@pytest.fixture
def calibrated_codec():
    """512 합성 캘리브레이션 샘플로 사전 캘리브레이션된 코덱."""
    config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=42)
    codec = RateQuantReverseWaterfillingCodec(config)
    torch.manual_seed(42)
    cal_kvs = [torch.randn(64, 2, 4, 32) for _ in range(20)]  # 단위 테스트용 축소
    codec.calibrate(cal_kvs)
    return codec


def test_reverse_waterfilling_total_budget(calibrated_codec):
    """역 물 채우기 결과 비트 합이 총 예산과 근사 일치.
    |sum(r_h) - total_budget * n_heads| < 1 비트."""
    ...


def test_encode_decode_shape(calibrated_codec):
    """encode → decode 후 shape 보존: [n_tokens, 2, n_heads, d_head]."""
    ...


def test_accuracy_relative_error_within_1pct(calibrated_codec):
    """PRIMARY ±1% 정확도 보존: relative output error < 0.01.
    avg_bits=4.0 기준. 캘리브레이션 독립 검증 샘플 128개 사용."""
    torch.manual_seed(99)  # 캘리브레이션(seed=42)과 독립
    q = torch.randn(16, 32)
    k = torch.randn(64, 2, 4, 32)
    v = k.clone()
    encoded = calibrated_codec.encode(k, layer_idx=0)
    k_dec = calibrated_codec.decode(encoded, layer_idx=0)
    error = attention_output_relative_error(
        q, k[:, 0, 0, :], k[:, 1, 0, :],   # head 0
        k_dec[:, 0, 0, :], k_dec[:, 1, 0, :],
    )
    assert error < 0.01, f"Relative error {error:.4f} exceeds 1% limit"


def test_kl_divergence_within_threshold(calibrated_codec):
    """KL 발산 < 0.015 (±1% perplexity proxy)."""
    ...


def test_cosine_similarity_above_threshold(calibrated_codec):
    """Attention output cosine similarity >= 0.99."""
    ...


def test_compression_ratio_meets_target(calibrated_codec):
    """avg_bits=4.0 설정에서 compression_ratio >= 0.70 (70% 이상 메모리 감소).
    FP16(16비트) 대비 4비트 = 75% 이론 감소."""
    ...


def test_low_variance_head_gets_fewer_bits(calibrated_codec):
    """역 물 채우기 핵심 성질: 분산이 낮은 헤드는 분산이 높은 헤드보다 비트 수가 적거나 같다."""
    ...


def test_calibration_independent_accuracy():
    """캘리브레이션 샘플과 완전히 독립된 테스트 데이터에서 ±1% 보존 확인.
    이것이 evaluation_criteria.md §4 필수 요건의 핵심 증거다."""
    torch.manual_seed(0)
    config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=0)
    codec = RateQuantReverseWaterfillingCodec(config)
    # 캘리브레이션: seed 0 기반 데이터
    cal_kvs = [torch.randn(32, 2, 4, 32) for _ in range(10)]
    codec.calibrate(cal_kvs)
    # 검증: seed 999 기반 완전 독립 데이터
    torch.manual_seed(999)
    test_kv = torch.randn(32, 2, 4, 32)
    q = torch.randn(8, 32)
    encoded = codec.encode(test_kv, layer_idx=0)
    kv_dec = codec.decode(encoded, layer_idx=0)
    error = attention_output_relative_error(
        q, test_kv[:, 0, 0, :], test_kv[:, 1, 0, :],
        kv_dec[:, 0, 0, :], kv_dec[:, 1, 0, :],
    )
    assert error < 0.01, f"Independent test relative error {error:.4f} exceeds ±1%"
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-11.yaml
experiment:
  date: "2026-05-11"
  activity: "B+C"
  description: >
    WiCER-RateQuantPipeline (Cross-1: B+C):
    WiCERIterativeKVWikiCache(B-1) + RateQuantReverseWaterfillingCodec(C-3) 통합.
    CEGAR 반복 컴파일 도메인 KV 아티팩트 + 역 물 채우기 최적 비트 할당 양자화.

wicer:
  chunk_size: 128              # 초기 청크 크기 (토큰 수)
  min_chunk_size: 16           # 정제 시 최소 청크 크기
  max_chunk_size: 512          # 병합 시 최대 청크 크기
  target_hit_rate: 0.80        # CEGAR 종료 조건 히트율
  max_iterations: 5            # 최대 CEGAR 반복 횟수
  max_entries: 2000            # 캐시 최대 엔트리 수
  seed: 42
  artifact_save_path: "results/2026-05-11/wicer_artifacts.pt"

ratequant:
  n_heads: 4                   # 어텐션 헤드 수 (단위 테스트용; 실모델 8/32)
  n_layers: 12                 # 레이어 수 (단위 테스트용)
  d_head: 32                   # head 차원 (단위 테스트용; 실모델 64/128)
  total_bit_budget: 4.0        # 평균 비트 예산 (헤드당)
  min_bits: 2                  # 헤드당 최소 비트
  max_bits: 8                  # 헤드당 최대 비트
  calibration_samples: 512     # 캘리브레이션 샘플 수 (단위 테스트에서는 20으로 축소)
  seed: 42
  calibration_save_path: "results/2026-05-11/ratequant_calibration.pt"
  # 실모델 설정 오버라이드:
  # n_heads: 32
  # d_head: 128
  # total_bit_budget: 2.5      # 목표 2.5비트 평균 (84% 메모리 감소)

pipeline:
  codec_calibration_before_cegar: true   # CEGAR 전에 RateQuant 캘리브레이션 수행
  pipeline_save_path: "results/2026-05-11/wicer_ratequant_pipeline.pt"

cache:
  type: "wicer_ratequant_pipeline"
  capacity_bytes: 4294967296    # 4 GiB

benchmark:
  accuracy:
    method: "attention_output_proxy"  # 실 모델 없이 합성 KV 사용
    proxy_tolerance: 0.01            # relative error < 1%
    kl_tolerance: 0.015              # KL divergence < 0.015
    cosine_min: 0.99                 # cosine similarity >= 0.99
    perplexity_tolerance_pct: 1.0   # evaluation_criteria.md §4 필수
    task_accuracy_tolerance_pct: 1.0
    calibration_samples: 20          # 단위 테스트용 축소 (실 환경 512)
    validation_samples: 10           # 독립 검증용 (실 환경 128)
  bit_width_sweep:
    avg_bits_list: [2.5, 3.0, 4.0, 6.0, 8.0]
  hit_rate:
    target_noncontiguous_fraction: 0.30
    cegar_target_hit_rate: 0.80
  memory_reduction:
    target_ratio: 0.30               # 최소 30% (목표 55%)
    target_ratio_goal: 0.55
  throughput:
    target_improvement_pct: 10       # 최소 +10% (목표 +35%)
    target_improvement_goal_pct: 35

seed: 42
results_dir: "results/2026-05-11"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_ratequant_codec.py` — RateQuantReverseWaterfillingCodec 단위 테스트
      (역 물 채우기 비트 할당, encode/decode shape, 압축률, 직렬화)
- [ ] `tests/unit/test_ratequant_accuracy.py` — **Activity C 필수**: accuracy proxy 검증
      (test_accuracy_relative_error_within_1pct, test_calibration_independent_accuracy 필수 포함)
- [ ] `tests/unit/test_wicer_iterative_cache.py` — WiCERIterativeKVWikiCache 단위 테스트
      (CEGAR 루프 히트율 단조 증가, 반례 수집, put/get/evict, 직렬화)
- [ ] `tests/unit/test_ratequant_accuracy.py` — perplexity proxy ±1% 이내 검증
- [ ] `tests/integration/test_cross_bc_wicer_ratequant.py` — B+C 통합 E2E 테스트
      (비연속 히트율 >= 30%, 메모리 감소 >= 30%, 압축 정확도 ±1% 복합 유지)

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 전부 통과** (기존 테스트 회귀 없음 포함)
2. **통합 테스트 전부 통과**
3. **Activity B (evaluation_criteria.md §3) 기준 충족**:
   - 비연속 세그먼트 히트율 >= 30% (전체 히트의 30% 이상이 비연속 구간)
   - 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상
   - KV Memory Footprint 베이스라인 대비 +20% 이내 (RateQuant 압축으로 오히려 감소 목표)
4. **Activity C (evaluation_criteria.md §4) 필수 기준 충족** (모두 Fail 시 전체 Fail):
   - perplexity proxy (attention output relative error) < 1% — `test_accuracy_relative_error_within_1pct` 통과
   - 캘리브레이션 독립 검증 샘플 기준 ±1% 이내 — `test_calibration_independent_accuracy` 통과
   - KV Memory Reduction >= −30% (목표 −55%)
   - Encode/Decode 추가 지연 TTFT +10% 이내
5. **크로스 조합 (evaluation_criteria.md §5) 기준 충족** (C 포함이므로 accuracy 필수):
   - 복합 처리량 향상: 단일 Activity 대비 추가 +5% 이상
   - 복합 메모리 감소: 단일 Activity 대비 추가 −10% 이상
   - 복합 적용 후에도 accuracy ±1% 이내 유지
6. **CacheStore 인터페이스 준수**: 모든 신규 클래스(WiCERIterativeKVWikiCache,
   RateQuantReverseWaterfillingCodec, WiCERRateQuantPipeline)가 CacheStore 추상 메서드 완전 구현
7. **설정 YAML 존재**: `configs/experiments/2026-05-11.yaml` 생성됨
8. **타입 힌트**: 모든 공개 함수·메서드에 완전한 타입 힌트
9. **시드 고정 재현성**: seed=42 고정 시 동일 결과 재현
10. **비트 할당 메타데이터 직렬화**: save_pipeline()에서 r_h 메타데이터가 함께 저장되어
    로드 시 재양자화 없이 즉시 사용 가능

---

## 구현 순서 (implementer 참고)

1. **`src/metrics/perplexity.py`** 먼저 구현 (5~6개 함수, 의존성 없음).
   `compute_attention_output()`, `attention_output_relative_error()`,
   `attention_kl_divergence()`, `cosine_similarity_output()`.

2. **`src/cache/ratequant_codec.py`** — C 핵심 구현.
   순서: `_reverse_waterfilling()` → `calibrate()` → `_compute_scale()` →
   `_quantize()` / `_dequantize()` → `encode()` / `decode()` → `compression_ratio()`.
   `test_ratequant_accuracy.py`의 `test_accuracy_relative_error_within_1pct` 통과 확인 후 다음 단계.

3. **`tests/unit/test_ratequant_accuracy.py`** — Activity C 필수 테스트 먼저 작성.
   모든 proxy 지표(relative error, KL, cosine)가 ±1% 기준 통과하는지 확인.

4. **`src/cache/wicer_iterative_cache.py`** — B 구현.
   순서: `compile_corpus()` → `evaluate()` → `refine()` → `cegar_refine()` →
   `save_artifacts()` / `load_artifacts()`.
   `test_wicer_iterative_cache.py`의 히트율 단조 증가 테스트 통과 확인.

5. **`src/cache/wicer_ratequant_pipeline.py`** — B+C 통합.
   `compression_hook()` → `build_pipeline()` → `save_pipeline()` / `load_pipeline()`.

6. **`tests/unit/test_ratequant_codec.py`**, **`tests/unit/test_wicer_iterative_cache.py`**
   나머지 단위 테스트 완성.

7. **`tests/integration/test_cross_bc_wicer_ratequant.py`** — 통합 E2E 테스트.

8. **`configs/experiments/2026-05-11.yaml`** 작성.

---

## 기존 파일 보존 목록

이번 사이클에서 수정하지 않는 파일 (기존 테스트 회귀 없이 통과해야 함):

- `src/cache/base.py` — compression_hook 이미 추가됨, 변경 불필요
- `src/cache/segmented.py` — WiCERIterativeKVWikiCache의 백엔드로 재사용, 수정 불필요
- `src/compression/` 전체 (vq_codec.py 등 이전 사이클 구현)
- `src/cache/kv_packet_adapter.py`, `src/cache/context_free_compressed_packet.py`
- `src/scheduler/` 전체 (이번 사이클 Activity A 미포함)
- 기타 모든 기존 캐시 구현체, 테스트 파일 (회귀 없이 통과 필수)
