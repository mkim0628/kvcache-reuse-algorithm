<!-- 변경 이유 (이전 Spec.md: 2026-05-12 대비):
이전 사이클(2026-05-12)은 B+C (RoPEReencodingNonContiguousCache + MixedDimPerTokenBudgetCodec +
AdapShotMixedDimSegmentPipeline) 조합이었다. 이번 사이클은 A+B+C로 전환한다.

주요 변경:
1. [Activity A 신규] PBKVAgentSegmentPreservationScheduler 추가.
   PBKV(2605.06472) 스타일 예측기로 에이전틱 워크로드에서 미래 세그먼트 재사용 확률을 추정해
   GPU Radix 트리 보존 정책을 동적으로 결정한다.
   파일: src/scheduler/pbkv_agent_segment_scheduler.py (신규).

2. [Activity B 교체] RoPEReencodingNonContiguousCache(prefix 위치 독립 재인코딩) →
   KVFoldAccumulativeRadixCache(foldl 누산기 기반 청크 재귀, 훈련-프리 128K NIAH 100%).
   KV-Fold(2605.12471) foldl 프로토콜을 src/cache/radix.py 확장으로 구현한다.
   파일: src/cache/kv_fold_accumulative.py (신규).

3. [Activity C 교체] MixedDimPerTokenBudgetCodec(토큰별 차원 드롭) →
   SRFTFusedINT4KVKernel(SRFT Gaussianization + INT4 nibble 패킹, 3× 메모리 압축).
   When Quantization Is Free(2605.05699) 설계를 torch.fft 기반 CUDA 백엔드로 일반화.
   기존 RateQuantReverseWaterfillingCodec의 헤드별 비트폭 출력을 SRFT 커널 입력으로 연결.
   파일: src/cache/srft_int4_kv_kernel.py (신규).

4. [Cross A+B 신규] AgenticChunkPreCachingPipeline: PBKV 예측기가 미래 청크 세트를 결정하고
   KVFoldAccumulativeRadixCache가 그 청크들을 foldl 순서로 사전 누산.
   파일: src/cache/agentic_chunk_precaching.py (신규).

5. [보존 파일] 이전 사이클 파일 전체(rope_reencoding_cache.py, mixed_dim_codec.py,
   adapshot_pipeline.py, 기타 모든 기존 파일)는 이번 사이클에서 수정하지 않는다.
   기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.

6. [인터페이스 유지] src/cache/base.py는 이번 사이클에서 수정하지 않는다.
   store_pre_rope()/load_with_rope() 선택적 메서드는 이미 이전 사이클에 추가됨.
   CacheStore 6개 추상 메서드를 모든 신규 구현체가 완전 구현한다.
-->

# Spec — 2026-05-13

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-13.md`
**최우선 구현 타겟**:
- **A-1**: PBKVAgentSegmentPreservationScheduler — PBKV 다단계 예측 기반 에이전트 KV 세그먼트 보존
- **B-1**: KVFoldAccumulativeRadixCache — foldl 누산기 청크 재귀 (훈련-프리, 128K NIAH 100%)
- **C-2**: SRFTFusedINT4KVKernel — SRFT Gaussianization + INT4 nibble 패킹 (3× 메모리 압축)
- **Cross-1**: AgenticChunkPreCachingPipeline — A-1 + B-1 통합 (예측 기반 사전 누산)

**해결하려는 문제**:

- Activity A: 기존 스케줄러는 정적 워크플로우 가정 또는 단순 히트율 예측에 의존해
  에이전틱 동적 워크플로우에서 캐시 퇴거율이 높다. PBKV(2605.06472) 방식의 예측기로
  미래 N 스텝 에이전트 호출에서 재사용될 KV 세그먼트를 사전 추정하고, 재사용 잠재력에
  따라 GPU Radix 트리 보존 / 호스트 메모리 퇴거를 결정한다.
  예상 효과: 에이전틱 TTFT 최대 1.39× 단축, 비연속 히트율 +15~25%.

- Activity B: 표준 RadixAttention은 공통 접두사 byte-identical 매칭에만 재사용을 허용한다.
  KV-Fold(2605.12471)의 foldl 프로토콜을 src/cache/ 레이어에 통합하여, 청크 단위로
  순서 의존적 누산 재사용을 지원한다. 이는 StreamingLLM의 메모리 효율과 완전 정보 보존을
  결합하여 단일 GPU에서 128K 컨텍스트 NIAH 100%를 훈련-프리로 달성한다.
  예상 효과: 비연속 히트율 +20~35%, 유효 컨텍스트 길이 2× 이상.

- Activity C: 기존 FP16 KV 캐시는 메모리 효율이 낮다. SRFT(Subsampled Randomized Fourier
  Transform)으로 아웃라이어 채널을 Gaussianize하고 INT4 nibble 패킹으로 4× 이론 압축(3× 실효)을
  달성한다. 기존 RateQuantReverseWaterfillingCodec의 헤드별 비트폭 출력을 SRFT 커널 입력으로
  연결하는 어댑터를 구현해 플러그인 방식으로 통합한다.
  예상 효과: 메모리 −60~67%, 추론 속도 FP16 대비 0~+10%, accuracy delta ±0.5% 이내.

- Cross-1: PBKV 예측기가 미래 에이전트 호출에서 필요할 청크 세트를 결정하고,
  KVFoldAccumulativeRadixCache가 그 청크들을 foldl 순서로 사전 누산(pre-folded prefix)하여
  RadixCache 리프 노드에 저장한다. 에이전트 실제 요청 시 이미 누산된 KV에서
  incremental foldl 처리만 수행하여 TTFT를 대폭 단축한다.
  예상 복합 효과: 처리량 +35~50%, TTFT −50~65%, 비연속 히트율 +25~40%.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling (PBKVAgentSegmentPreservationScheduler)
- [x] Activity B: Non-Contiguous KV Cache Reuse (KVFoldAccumulativeRadixCache)
- [x] Activity C: KV Cache Compression (SRFTFusedINT4KVKernel)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §2 Activity A): 스케줄링 오버헤드 TTFT p50 +5% 이내;
      에이전틱 워크로드(5~10 스텝 동적 워크플로우)에서 캐시 히트율 +10%p 이상
- [ ] 목표 2 (evaluation_criteria.md §3 Activity B): 비연속 세그먼트 히트율 >= 30%
      (전체 히트 중 비연속 구간 발생 비율); NIAH 16K/64K/128K 컨텍스트 검색 정확도 100%
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      (attention output relative error < 0.01, WikiText-2 기준)
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 변화 ±1% 이내
      (LongBench 8개 서브태스크 proxy 기준, KL divergence < 0.015, cosine >= 0.99)
- [ ] 목표 5 (evaluation_criteria.md §4 Activity C): KV Memory Reduction >= −30%
      (목표치 −60% 이상, SRFT+INT4 3× 압축 기준)
- [ ] 목표 6 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +20% 이상
      (Cross-1 A+B+C 복합 기준)
- [ ] 목표 7 (evaluation_criteria.md §4 Activity C): Effective Context Length 동일 메모리 예산 2× 이상
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합): 단일 Activity 대비 복합 처리량 추가 +5% 이상
- [ ] 목표 9 (evaluation_criteria.md §5 크로스 조합): 단일 Activity 대비 복합 메모리 감소 추가 −10% 이상
- [ ] 목표 10 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후에도 accuracy ±1% 이내

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/kv_fold_accumulative.py` | B | KVFoldAccumulativeRadixCache — foldl 누산기 + 드리프트 플래토 감지, radix.py 확장 |
| `src/cache/srft_int4_kv_kernel.py` | C | SRFTFusedINT4KVKernel — SRFT Gaussianization + 채널별 스케일 + 그룹 abs-max INT4 + nibble 패킹 |
| `src/cache/agentic_chunk_precaching.py` | A+B | AgenticChunkPreCachingPipeline — PBKV 예측 + KVFold 사전 누산 통합 |
| `src/scheduler/pbkv_agent_segment_scheduler.py` | A | PBKVAgentSegmentPreservationScheduler — MLP 예측기 기반 세그먼트 GPU/호스트 계층 보존 |
| `tests/unit/test_kv_fold_accumulative.py` | B | foldl 누산기 단위 테스트 (히트율, 드리프트 플래토, 누산 상태 shape) |
| `tests/unit/test_srft_int4_kv_kernel.py` | C | SRFT+INT4 단위 테스트 (encode/decode shape, 압축률, nibble 패킹 정확도) |
| `tests/unit/test_srft_int4_accuracy.py` | C | Activity C 필수 — SRFT+INT4 accuracy preservation 검증 (±1% proxy 기준) |
| `tests/unit/test_pbkv_scheduler.py` | A | PBKV 스케줄러 단위 테스트 (예측기, 보존 정책, Lipschitz 강건성) |
| `tests/unit/test_agentic_precaching.py` | A+B | 사전 누산 파이프라인 단위 테스트 (예측 미스 fallback, pre-folded prefix 등록) |
| `tests/integration/test_cross_abc_kvfold_srft.py` | A+B+C | A+B+C 통합 E2E 테스트 (처리량, 히트율, 메모리, 정확도 복합 검증) |
| `configs/experiments/2026-05-13.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/base.py` | 변경 없음. store_pre_rope()/load_with_rope() 이미 존재. |

**보존 파일 (이번 사이클에서 수정하지 않음)**:
- `src/cache/rope_reencoding_cache.py`, `src/cache/mixed_dim_codec.py`, `src/cache/adapshot_pipeline.py` — 이전 사이클 B+C 구현 보존
- `src/cache/ratequant_codec.py`, `src/cache/wicer_iterative_cache.py`, `src/cache/wicer_ratequant_pipeline.py`
- `src/cache/segmented.py` — KVFoldAccumulativeRadixCache의 백엔드로 재사용
- `src/scheduler/cache_aware_scheduler.py` — PBKV 스케줄러의 부모가 아닌 독립 구현; 기존 동작 유지
- `src/metrics/perplexity.py` — 기존 함수 그대로 사용 (attention_output_relative_error, attention_kl_divergence, cosine_similarity_output)
- 기타 모든 기존 파일 — 회귀 없이 통과 필수

---

## 알고리즘 상세

### [KVFoldAccumulativeRadixCache] (Activity B) — `src/cache/kv_fold_accumulative.py`

KV 캐시를 함수형 프로그래밍의 `foldl` 누산기로 취급한다. 각 청크 처리 시
기존 누산된 KV를 접두사(prefix)로 참조하며 새 청크를 처리하고,
생성된 K·V를 누산 캐시에 추가한다. 이 단일 업데이트 규칙을 반복해
임의 깊이의 청크 체인을 구성한다.

**핵심 안정성 관찰 (KV-Fold 2605.12471)**:
- 스텝당 드리프트가 초기 상승 후 플래토에 수렴 (`drift_threshold=1e-3`)
- 플래토는 수치 정밀도 10,000배 변화, 청크 크기, 모델 패밀리에 무감
- 플래토 수렴 후 추가 청크 처리를 조기 종료해 효율 확보

**RadixCache와의 통합 경로**:
- RadixAttention prefix 매칭 성공 → 기존 경로 사용
- prefix 매칭 실패 → KVFold 누산 경로로 fallback
- "pre-folded prefix" 노드를 RadixCache 리프 노드에 등록하여 향후 히트 증가

```python
# 의사코드 — src/cache/kv_fold_accumulative.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import hashlib
import struct
import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class KVFoldConfig:
    chunk_size: int = 128           # 청크당 토큰 수
    max_entries: int = 2000         # 최대 캐시 엔트리 수
    drift_threshold: float = 1e-3   # 플래토 감지 임계값 (스텝당 KV drift)
    max_fold_depth: int = 511       # 최대 foldl 체인 깊이 (KV-Fold 논문 기준 511)
    enable_streaming_fallback: bool = True  # 메모리 압박 시 StreamingLLM 방식 fallback
    window_size: int = 32           # StreamingLLM 슬라이딩 윈도우 크기 (청크 수)
    d_head: int = 64                # head 차원
    n_heads: int = 8                # 어텐션 헤드 수
    n_layers: int = 12              # 레이어 수
    seed: int = 42


@dataclass
class _FoldState:
    """누산된 KV 상태 — foldl 체인의 누산기."""
    accumulated_kv: torch.Tensor   # [n_accumulated_tokens, 2, n_heads, d_head]
    chunk_ids: List[int]           # 처리된 청크 인덱스 목록
    fold_depth: int                # 현재 foldl 체인 깊이
    last_drift: float              # 마지막 스텝 드리프트
    plateau_reached: bool          # 플래토 수렴 여부


class KVFoldAccumulativeRadixCache(CacheStore):
    """foldl 누산기 기반 비연속 KV 재사용 캐시 (Activity B).

    KV-Fold(2605.12471) foldl 프로토콜을 src/cache/ 레이어에 훈련-프리로 통합.
    기존 SegmentedHashCache를 세그먼트 백엔드로 사용하고,
    foldl 누산 상태를 별도 OrderedDict에 관리한다.

    CacheStore 인터페이스 완전 구현 + fold_chunk() / get_folded() 추가 API.

    사용 흐름:
      1. fold_chunk(chunk_tokens, layer_idx) — 청크를 누산 상태에 foldl 처리
      2. get_folded_prefix(key) — 누산된 KV 상태를 접두사로 반환
      3. put/get — CacheStore 표준 인터페이스 (RadixAttention prefix 매칭 경로)
    """

    def __init__(self, config: KVFoldConfig) -> None:
        self.config = config
        self._store = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        # foldl 누산 상태: fold_key → _FoldState
        self._fold_states: OrderedDict[str, _FoldState] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._fold_hits = 0         # foldl 경로에서 발생한 히트 수

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """표준 prefix KV 저장 (RadixAttention 호환 경로)."""
        self._store.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """표준 prefix KV 조회. 미스 시 None 반환."""
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        """LRU 퇴거. 세그먼트 스토어와 fold 상태 모두 대상."""
        freed = self._store.evict()
        if self._fold_states:
            oldest_key = next(iter(self._fold_states))
            state = self._fold_states.pop(oldest_key)
            freed += state.accumulated_kv.nbytes
        return freed

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        fold_bytes = sum(
            s.accumulated_kv.nbytes for s in self._fold_states.values()
        )
        return self._store.memory_bytes() + fold_bytes

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._fold_hits = 0
        self._store.reset_stats()

    # ------------------------------------------------------------------ #
    # foldl 누산 API                                                       #
    # ------------------------------------------------------------------ #

    def fold_chunk(
        self,
        chunk_tokens: List[int],    # 새 청크의 토큰 ID
        layer_idx: int = 0,
        existing_fold_key: Optional[str] = None,
        # 이전 누산 상태 키. None이면 새 foldl 체인 시작
    ) -> Tuple[str, torch.Tensor]:
        """청크를 foldl 누산 상태에 통합한다.

        반환:
          fold_key: str — 갱신된 누산 상태의 키 (다음 fold_chunk 호출에 전달)
          accumulated_kv: Tensor [n_acc_tokens, 2, n_heads, d_head] — 갱신된 누산 KV

        의사코드:
          # 1. 기존 누산 상태 로드 또는 초기화
          if existing_fold_key and existing_fold_key in self._fold_states:
              state = self._fold_states[existing_fold_key]
          else:
              state = _FoldState(
                  accumulated_kv=torch.empty(0, 2, n_heads, d_head),
                  chunk_ids=[], fold_depth=0, last_drift=float('inf'), plateau_reached=False
              )

          # 2. 플래토 수렴 확인 — 수렴 후 조기 종료
          if state.plateau_reached:
              # foldl 체인 안정적 — 현재 누산 상태 그대로 반환 (연산 절감)
              return existing_fold_key, state.accumulated_kv

          # 3. 새 청크 KV 생성 (합성 또는 실 모델 계산)
          chunk_key = self._store.chunk_key(chunk_tokens, 0, layer_idx)
          cached_kv = self._store.get(chunk_key)
          if cached_kv is None:
              # 캐시 미스 — 새 청크 KV를 합성 생성 (실제 엔진에서는 모델 포워드 패스)
              new_chunk_kv = self._compute_chunk_kv(chunk_tokens, layer_idx,
                                                     state.accumulated_kv)
              self._store.put(chunk_key, new_chunk_kv)
          else:
              new_chunk_kv = cached_kv

          # 4. foldl 누산: accumulated_kv에 new_chunk_kv를 cat
          if state.accumulated_kv.shape[0] == 0:
              updated_kv = new_chunk_kv
          else:
              updated_kv = torch.cat([state.accumulated_kv, new_chunk_kv], dim=0)

          # 5. 드리프트 계산: ||new_chunk_kv - prev_chunk_kv||_F / ||prev_chunk_kv||_F
          if state.accumulated_kv.shape[0] > 0:
              prev_tail = state.accumulated_kv[-new_chunk_kv.shape[0]:]
              drift = (new_chunk_kv - prev_tail).norm().item() / \
                      prev_tail.norm().clamp(min=1e-8).item()
          else:
              drift = float('inf')

          # 6. 플래토 감지
          plateau = (drift < self.config.drift_threshold and
                     state.fold_depth >= 2)  # 최소 2 스텝 필요

          # 7. StreamingLLM 폴백 (enable_streaming_fallback + 메모리 압박 시)
          if (self.config.enable_streaming_fallback and
              updated_kv.shape[0] > self.config.window_size * self.config.chunk_size):
              # 슬라이딩 윈도우: 초기 어텐션 싱크 토큰 보존 + 최신 window_size 청크
              sink_tokens = min(4, updated_kv.shape[0])
              window_tokens = self.config.window_size * self.config.chunk_size
              updated_kv = torch.cat([
                  updated_kv[:sink_tokens],
                  updated_kv[-(window_tokens - sink_tokens):]
              ], dim=0)

          # 8. 새 fold_key 생성 및 상태 저장
          chunk_hash = self._store.chunk_key(chunk_tokens, 0, layer_idx)
          new_fold_key = f"fold:{layer_idx}:{chunk_hash}:{state.fold_depth}"
          new_state = _FoldState(
              accumulated_kv=updated_kv,
              chunk_ids=state.chunk_ids + [hash(tuple(chunk_tokens))],
              fold_depth=state.fold_depth + 1,
              last_drift=drift,
              plateau_reached=plateau,
          )
          if len(self._fold_states) >= self.config.max_entries:
              self.evict()
          self._fold_states[new_fold_key] = new_state

          return new_fold_key, updated_kv
        """
        ...

    def get_folded_prefix(
        self,
        fold_key: str,
    ) -> Optional[torch.Tensor]:
        """누산된 KV 상태를 접두사로 반환 (캐시 미스 시 None).

        반환: [n_acc_tokens, 2, n_heads, d_head] float16
        """
        state = self._fold_states.get(fold_key)
        if state is None:
            return None
        self._fold_hits += 1
        self._hits += 1
        return state.accumulated_kv

    def register_prefolded_prefix(
        self,
        fold_key: str,
        accumulated_kv: torch.Tensor,
        chunk_ids: List[int],
    ) -> None:
        """AgenticChunkPreCachingPipeline에서 사전 누산된 KV를 등록.

        A-1 예측기가 결정한 청크들을 사전에 fold_chunk로 처리한 뒤,
        이 메서드로 RadixCache 리프 노드에 "pre-folded prefix"로 저장한다.

        의사코드:
          state = _FoldState(
              accumulated_kv=accumulated_kv,
              chunk_ids=chunk_ids,
              fold_depth=len(chunk_ids),
              last_drift=0.0,
              plateau_reached=True,   # 사전 누산이므로 안정 상태로 표시
          )
          self._fold_states[fold_key] = state
          # RadixCache 호환: 청크 해시 키로도 접근 가능하도록 put() 등록
          self._store.put(fold_key, accumulated_kv)
        """
        ...

    def noncontiguous_hit_rate(self) -> float:
        """foldl 누산 경로 + 비연속 구간에서 발생한 히트 비율."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return (self._noncontiguous_hits + self._fold_hits) / total_hits

    def get_fold_depth_stats(self) -> Dict[str, float]:
        """foldl 체인 통계: 평균 깊이, 플래토 수렴 비율."""
        ...

    # ------------------------------------------------------------------ #
    # 세그먼트 API (SegmentedHashCache 호환)                               #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """표준 세그먼트 저장 (RadixAttention 호환)."""
        self._store.put_segment(token_ids, chunk_idx, kv, layer_idx)

    def get_segments_with_fold(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
        fold_key: Optional[str] = None,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int], Optional[torch.Tensor]]:
        """RadixAttention prefix 매칭 + foldl fallback 통합 조회.

        반환:
          hits: [(chunk_idx, kv)]
          misses: [chunk_idx]
          fold_prefix: Optional[Tensor] — foldl 누산 접두사 (있으면 활용)

        의사코드:
          hits, misses = self._store.get_segments(token_ids, layer_idx)
          fold_prefix = self.get_folded_prefix(fold_key) if fold_key else None
          # 비연속 히트 추적
          if hits:
              miss_set = set(misses)
              for idx, _ in hits:
                  if any(m < idx for m in miss_set):
                      self._noncontiguous_hits += 1
          return hits, misses, fold_prefix
        """
        ...

    # ------------------------------------------------------------------ #
    # 내부 유틸리티                                                          #
    # ------------------------------------------------------------------ #

    def _compute_chunk_kv(
        self,
        chunk_tokens: List[int],
        layer_idx: int,
        prefix_kv: torch.Tensor,   # [n_acc, 2, n_heads, d_head] — 누산된 접두사 KV
    ) -> torch.Tensor:
        """청크의 KV를 합성 생성 (실 모델 없이 시뮬레이션용).

        실제 엔진에서는 src/engine/runner.py를 통해 모델 포워드 패스 결과를 받는다.
        합성: torch.randn 기반 deterministic KV (chunk_tokens seed 사용).

        의사코드:
          n_tok = len(chunk_tokens)
          seed = sum(chunk_tokens) + layer_idx
          torch.manual_seed(seed % (2**31))
          kv = torch.randn(n_tok, 2, self.config.n_heads, self.config.d_head)
          return kv
        """
        ...
```

---

### [SRFTFusedINT4KVKernel] (Activity C) — `src/cache/srft_int4_kv_kernel.py`

SRFT(Subsampled Randomized Fourier Transform) Gaussianization과
INT4 nibble 패킹을 결합한 KV 캐시 압축 코덱.
When Quantization Is Free(2605.05699)의 단일 퓨즈드 Metal 커널 설계를
`torch.fft.fft` 기반 CUDA/CPU 백엔드로 일반화한다.

**압축 파이프라인 (순서 중요)**:
```
raw KV [n_tokens, 2, n_heads, d_head] float16
  ↓  [1] 부호 무작위화(sign randomization): d[i] *= r[i], r[i] ∈ {+1, -1}
  ↓  [2] FFT: torch.fft.fft(d, dim=-1).real → Gaussianize 아웃라이어 채널
  ↓  [3] 채널별 스케일링 λ: scale = abs(d).max(dim=0) → FP16 사이드카 저장
  ↓  [4] 그룹별 abs-max 양자화: group_size=128, INT4 범위 [-7, 7]
  ↓  [5] INT4 nibble 패킹: 2개 INT4 값 → 1 uint8 바이트
결과: packed_kv [n_tokens, 2, n_heads, d_head//2] uint8
      scales [n_heads, d_head//group_size] float16 (사이드카)
      sign_seeds int (재현 가능 부호 무작위화 시드)
```

**복원 파이프라인**:
```
packed_kv + scales + sign_seeds
  ↓  [1] nibble 언패킹: uint8 → INT4 쌍
  ↓  [2] 역 스케일링: dequant = int4_val * scale_per_group
  ↓  [3] 역 FFT: torch.fft.ifft(복원 float).real
  ↓  [4] 부호 역적용: d[i] /= r[i]
결과: 복원 KV [n_tokens, 2, n_heads, d_head] float16
```

**RateQuant 어댑터**:
- `RateQuantReverseWaterfillingCodec`의 헤드별 비트폭 출력(`bit_allocation[layer_idx]`)을
  SRFT 커널 입력으로 받아 동적 비트폭 양자화 적용.
- 비트폭 8 헤드: INT8 양자화 (nibble 미적용)
- 비트폭 4 헤드: INT4 nibble 패킹 (기본)
- 비트폭 2 헤드: INT2 쌍 패킹 (실험적)

```python
# 의사코드 — src/cache/srft_int4_kv_kernel.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class SRFTInt4Config:
    n_heads: int = 8                  # 어텐션 헤드 수
    d_head: int = 64                  # head 차원 (FFT 적용 차원)
    group_size: int = 128             # 그룹별 abs-max 양자화 그룹 크기
    n_bits: int = 4                   # 기본 양자화 비트폭 (INT4)
    use_srft: bool = True             # SRFT Gaussianization 활성화
    ratequant_adapter: bool = False   # RateQuant 헤드별 비트폭 어댑터 활성화
    seed: int = 42                    # 부호 무작위화 시드


class SRFTFusedINT4KVKernel:
    """SRFT Gaussianization + INT4 nibble 패킹 KV 압축 코덱 (Activity C).

    HuggingFace Cache 서브클래스 인터페이스와 CacheStore.compression_hook() 호환.
    훈련 없이 추론 시 즉시 적용 가능.

    accuracy-preserving 근거:
      SRFT가 아웃라이어 채널을 Gaussianize하여 INT4 양자화 오류를 최소화.
      원논문(2605.05699)에서 3× 메모리 압축 + 품질 유지가 실증됨.
      그룹 크기 G=128에서 스케일 팩터가 정밀하게 각 그룹을 커버.
    """

    def __init__(self, config: SRFTInt4Config) -> None:
        self.config = config
        # 재현 가능 부호 무작위화 벡터: [d_head] ∈ {+1, -1}
        torch.manual_seed(config.seed)
        self._sign_vector = torch.randint(0, 2, (config.d_head,)) * 2 - 1
        # float16으로 변환 (연산 속도 최적화)
        self._sign_vector = self._sign_vector.float()

    # ------------------------------------------------------------------ #
    # 메인 encode / decode API                                             #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        kv: torch.Tensor,           # [n_tokens, 2, n_heads, d_head] float16
        head_bits: Optional[List[int]] = None,
        # RateQuant 어댑터: 헤드별 비트폭 리스트. None이면 config.n_bits 균일 적용
    ) -> dict:
        """SRFT + INT4 압축 인코딩.

        반환:
        {
          'packed_kv': torch.Tensor,    # [n_tokens, 2, n_heads, d_head//2] uint8
          'scales': torch.Tensor,       # [n_tokens, 2, n_heads, d_head//group_size] float16
          'sign_seed': int,             # 부호 무작위화 시드 (복원용)
          'n_bits': int or List[int],   # 적용된 비트폭
          'n_tokens': int,
          'n_heads': int,
          'd_head': int,
          'group_size': int,
        }

        의사코드:
          n_tokens, _, n_heads, d_head = kv.shape
          bits_per_head = head_bits or [self.config.n_bits] * n_heads

          # [1] 부호 무작위화
          sign = self._sign_vector.to(kv.device)  # [d_head]
          kv_signed = kv * sign.view(1, 1, 1, -1)

          # [2] FFT Gaussianization
          if self.config.use_srft:
              kv_fft = torch.fft.fft(kv_signed.float(), dim=-1).real.to(kv.dtype)
          else:
              kv_fft = kv_signed

          # [3] 그룹별 abs-max 스케일 계산
          G = self.config.group_size
          n_groups = (d_head + G - 1) // G
          kv_grouped = kv_fft.reshape(n_tokens, 2, n_heads, n_groups, -1)
          scales = kv_grouped.abs().amax(dim=-1).clamp(min=1e-8)  # [n_t, 2, n_h, n_g]

          # [4] 그룹별 INT4 양자화: [-7, 7]
          kv_norm = kv_grouped / scales.unsqueeze(-1)
          kv_int4 = kv_norm.mul(7.0).round().clamp(-7, 7).to(torch.int8)
          kv_int4_flat = kv_int4.reshape(n_tokens, 2, n_heads, d_head)

          # [5] nibble 패킹: 2 INT4 → 1 uint8
          even = kv_int4_flat[..., 0::2] & 0x0F   # [n_t, 2, n_h, d_head//2]
          odd  = (kv_int4_flat[..., 1::2] & 0x0F) << 4
          packed = (even | odd).to(torch.uint8)    # [n_t, 2, n_h, d_head//2]

          return {
              'packed_kv': packed,
              'scales': scales.to(torch.float16),
              'sign_seed': self.config.seed,
              'n_bits': bits_per_head if head_bits else self.config.n_bits,
              'n_tokens': n_tokens,
              'n_heads': n_heads,
              'd_head': d_head,
              'group_size': G,
          }
        """
        ...

    def decode(
        self,
        encoded: dict,
    ) -> torch.Tensor:
        """INT4 nibble 언패킹 + 역 SRFT로 KV 복원.

        반환: [n_tokens, 2, n_heads, d_head] float16

        의사코드:
          packed = encoded['packed_kv']           # [n_t, 2, n_h, d//2] uint8
          scales = encoded['scales']              # [n_t, 2, n_h, n_groups] float16
          d_head = encoded['d_head']
          G = encoded['group_size']
          n_tokens, n_heads = encoded['n_tokens'], encoded['n_heads']

          # [1] nibble 언패킹
          low  = (packed & 0x0F).to(torch.int8)              # [n_t, 2, n_h, d//2]
          high = ((packed >> 4) & 0x0F).to(torch.int8)
          # sign extension: INT4 → INT8 ([-8,7] 범위)
          low  = torch.where(low  >= 8, low  - 16, low)
          high = torch.where(high >= 8, high - 16, high)
          interleaved = torch.stack([low, high], dim=-1).reshape(n_tokens, 2, n_heads, d_head)

          # [2] 역 스케일링 (그룹별)
          n_groups = (d_head + G - 1) // G
          int4_grouped = interleaved.reshape(n_tokens, 2, n_heads, n_groups, -1).float()
          kv_dequant = (int4_grouped * scales.float().unsqueeze(-1)).reshape(
              n_tokens, 2, n_heads, d_head)

          # [3] 역 FFT
          if 'use_srft' implied by sign_seed:
              kv_ifft = torch.fft.ifft(
                  torch.complex(kv_dequant, torch.zeros_like(kv_dequant)), dim=-1
              ).real

          # [4] 부호 역적용
          sign = self._sign_vector.to(kv_ifft.device)
          kv_restored = kv_ifft / sign.view(1, 1, 1, -1).clamp(min=1e-8)

          return kv_restored.to(torch.float16)
        """
        ...

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,        # [n_tokens, 2, n_heads, d_head] float16
        head_bits: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """CacheStore.compression_hook() 호환 인터페이스.
        encode 후 즉시 decode하여 압축-복원 KV를 반환 (손실 압축).

        실제 스토리지에는 packed_kv + scales만 저장하므로, 이 메서드는
        in-place 정확도 검증용이다. 실 스토리지 절감은 store_compressed()를 사용.
        """
        encoded = self.encode(value, head_bits)
        return self.decode(encoded)

    # ------------------------------------------------------------------ #
    # 메모리 절감 스토리지 API                                               #
    # ------------------------------------------------------------------ #

    def memory_reduction_ratio(self, n_tokens: int, d_head: int, n_heads: int) -> float:
        """INT4 nibble 패킹으로 인한 이론적 메모리 감소율.

        FP16 대비: 16비트 → 4비트 = 4× 이론, 스케일 사이드카 포함 ≈3× 실효.

        의사코드:
          fp16_bytes = n_tokens * 2 * n_heads * d_head * 2        # float16
          packed_bytes = n_tokens * 2 * n_heads * (d_head // 2)   # uint8 nibble
          G = self.config.group_size
          scale_bytes = n_tokens * 2 * n_heads * (d_head // G) * 2  # float16 scales
          total_compressed = packed_bytes + scale_bytes
          return 1.0 - total_compressed / fp16_bytes
        """
        ...

    # ------------------------------------------------------------------ #
    # RateQuant 어댑터                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_ratequant(
        codec,                          # RateQuantReverseWaterfillingCodec 인스턴스
        layer_idx: int,
        base_config: Optional['SRFTInt4Config'] = None,
    ) -> 'SRFTFusedINT4KVKernel':
        """RateQuantReverseWaterfillingCodec의 헤드별 비트폭을 SRFT 커널에 연결.

        의사코드:
          head_bits = codec.bit_allocation.get(layer_idx, None)
          cfg = base_config or SRFTInt4Config()
          cfg.ratequant_adapter = True
          kernel = SRFTFusedINT4KVKernel(cfg)
          kernel._ratequant_head_bits = head_bits
          return kernel
        """
        ...
```

---

### [PBKVAgentSegmentPreservationScheduler] (Activity A) — `src/scheduler/pbkv_agent_segment_scheduler.py`

PBKV(2605.06472) 방식의 예측기를 세그먼트 레벨로 재구성한 에이전틱 KV 스케줄러.
과거 에이전트 호출 이력과 현재 컨텍스트를 입력으로 받아 미래 N 스텝에서
어떤 비연속 KV 세그먼트가 재사용될 확률이 높은지 추정한다.

**2계층 보존 정책**:
- 재사용 확률 > `gpu_preserve_threshold` (기본 0.6): GPU Radix 트리 보존
- 재사용 확률 <= `gpu_preserve_threshold`: 호스트 메모리 퇴거 (HiCache 스타일)

**Lipschitz-연속 강건성 보장**:
- `preemption_margin=0.3`: 예측 확률이 임계값보다 0.3 낮아야 퇴거 결정
- 예측 오차가 크더라도 최악 성능이 예측 오차에 연속적으로 의존

**스케줄링 결정 단위**: 요청(request) 수준 + 세그먼트(segment) 수준 2계층
**캐시 상태 접근 방법**: `cache._store` 키 목록 조회 (히트/미스 통계 미오염)

```python
# 의사코드 — src/scheduler/pbkv_agent_segment_scheduler.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn

from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


@dataclass
class PBKVConfig:
    segment_emb_dim: int = 256       # 세그먼트 임베딩 차원
    history_steps: int = 10          # 과거 에이전트 호출 이력 스텝 수
    prediction_horizon: int = 5      # 미래 N 스텝 예측 범위
    gpu_preserve_threshold: float = 0.6   # GPU 보존 임계값
    host_evict_threshold: float = 0.3    # 호스트 퇴거 임계값
    preemption_margin: float = 0.3       # Lipschitz 강건성 마진
    fairness_max_wait: int = 10          # 공정성 최대 대기 스텝
    chunk_size: int = 128                # 세그먼트 청크 크기
    seed: int = 42


class _SegmentMLP(nn.Module):
    """세그먼트 재사용 확률 예측기 (경량 MLP).

    입력: 세그먼트 임베딩(d=256) + 최근 호출 이력(10 스텝 합산)
    출력: 세그먼트별 재사용 확률 스칼라 ∈ [0, 1]
    """

    def __init__(self, segment_emb_dim: int, history_steps: int) -> None:
        super().__init__()
        input_dim = segment_emb_dim + history_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력: [batch, input_dim] → 출력: [batch, 1]"""
        return self.net(x)


class PBKVAgentSegmentPreservationScheduler:
    """PBKV 예측 기반 에이전트 세그먼트 보존 스케줄러 (Activity A).

    스케줄링 결정 단위: 요청(request) 수준으로 배치 순서 결정 +
                      세그먼트(segment) 수준으로 GPU/호스트 보존 결정.

    캐시 상태 접근: cache._store 키 직접 조회 (get() 미호출 — 통계 미오염).

    두 가지 주요 동작:
      1. schedule(requests) — 재사용 확률 기반 요청 배치 순서 결정
      2. update_preservation_policy(cache) — 보존/퇴거 결정 세그먼트 갱신

    기존 CacheAwareScheduler와의 차이:
      - CacheAwareScheduler: 현재 캐시 히트율 기반 정적 우선순위
      - PBKVAgentSegmentPreservationScheduler: 미래 N 스텝 예측 기반 동적 보존 정책
    """

    def __init__(
        self,
        cache: CacheStore,
        config: Optional[PBKVConfig] = None,
    ) -> None:
        self.cache = cache
        self.config = config or PBKVConfig()
        self._predictor = _SegmentMLP(
            self.config.segment_emb_dim,
            self.config.history_steps,
        )
        # 에이전트 호출 이력: agent_id → 최근 chunk_key 목록 (최대 history_steps)
        self._agent_history: Dict[str, List[str]] = {}
        # 세그먼트 보존 상태: chunk_key → {'gpu': bool, 'prob': float}
        self._preservation_map: Dict[str, Dict] = {}
        # 요청 대기 스텝: request_id → wait_steps
        self._wait_steps: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # 스케줄링 API                                                          #
    # ------------------------------------------------------------------ #

    def schedule(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        """재사용 확률 기반 요청 배치 순서 결정.

        우선순위 공식:
          priority = predicted_reuse_prob × (1 − wait_penalty)
          wait_penalty = min(wait_steps / fairness_max_wait, 1.0)

        의사코드:
          scored = []
          for req in requests:
              prob = self._predict_segment_reuse(req)
              wait = self._wait_steps.get(req.request_id, 0)
              wait_penalty = min(wait / max(config.fairness_max_wait, 1), 1.0)
              priority = prob * (1.0 - wait_penalty)
              scored.append((-priority, -wait, req.request_id, req))
          scored.sort(key=lambda t: (t[0], t[1]))
          return [item[3] for item in scored]
        """
        ...

    def update_preservation_policy(
        self,
        processed_request_ids: List[str],
        all_request_ids: List[str],
    ) -> Tuple[Set[str], Set[str]]:
        """세그먼트별 GPU 보존 / 호스트 퇴거 결정 갱신.

        반환:
          preserve_keys: Set[str] — GPU에 보존할 chunk_key 집합
          evict_keys: Set[str] — 호스트로 퇴거할 chunk_key 집합

        의사코드:
          # 캐시 키 목록 조회 (get() 미호출)
          store = getattr(self.cache, '_store', None)
          if store is None:
              return set(), set()
          all_keys = list(store.keys())

          preserve_keys, evict_keys = set(), set()
          for key in all_keys:
              emb = self._get_segment_embedding(key)
              history_vec = self._get_history_vector()
              input_vec = torch.cat([emb, history_vec]).unsqueeze(0)
              with torch.no_grad():
                  prob = self._predictor(input_vec).item()

              # Lipschitz 강건성: preemption_margin 마진 적용
              effective_threshold = config.gpu_preserve_threshold - config.preemption_margin
              if prob >= effective_threshold:
                  preserve_keys.add(key)
              elif prob < config.host_evict_threshold:
                  evict_keys.add(key)

          self._preservation_map = {
              k: {'gpu': k in preserve_keys, 'prob': 0.0} for k in all_keys
          }
          return preserve_keys, evict_keys
        """
        ...

    def update_agent_history(
        self,
        agent_id: str,
        accessed_chunk_keys: List[str],
    ) -> None:
        """에이전트 호출 이력 갱신 (최근 history_steps 유지)."""
        history = self._agent_history.get(agent_id, [])
        history.extend(accessed_chunk_keys)
        self._agent_history[agent_id] = history[-self.config.history_steps:]

    def update_wait(
        self,
        processed_ids: List[str],
        all_ids: List[str],
    ) -> None:
        """처리되지 않은 요청의 대기 스텝 증가."""
        processed_set = set(processed_ids)
        for rid in all_ids:
            if rid not in processed_set:
                self._wait_steps[rid] = self._wait_steps.get(rid, 0) + 1

    # ------------------------------------------------------------------ #
    # 내부 유틸리티                                                          #
    # ------------------------------------------------------------------ #

    def _predict_segment_reuse(
        self,
        request: InferenceRequest,
    ) -> float:
        """요청의 세그먼트들에 대한 평균 재사용 확률 예측.

        의사코드:
          chunk_size = self.config.chunk_size
          n_chunks = max(1, (len(request.token_ids) + chunk_size - 1) // chunk_size)
          probs = []
          for chunk_idx in range(n_chunks):
              key = self._chunk_key(request.token_ids, chunk_idx)
              emb = self._get_segment_embedding(key)
              hist = self._get_history_vector(request.request_id)
              inp = torch.cat([emb, hist]).unsqueeze(0)
              with torch.no_grad():
                  prob = self._predictor(inp).item()
              probs.append(prob)
          return sum(probs) / len(probs) if probs else 0.0
        """
        ...

    def _get_segment_embedding(self, chunk_key: str) -> torch.Tensor:
        """chunk_key에서 d=256 임베딩 생성 (해시 기반 deterministic)."""
        ...

    def _get_history_vector(
        self,
        agent_or_request_id: str = "",
    ) -> torch.Tensor:
        """에이전트 호출 이력에서 history_steps 길이 벡터 생성."""
        ...

    def _chunk_key(self, token_ids: List[int], chunk_idx: int) -> str:
        """SegmentedHashCache와 동일한 청크 키 생성 방식."""
        ...
```

---

### [AgenticChunkPreCachingPipeline] (Activity A+B) — `src/cache/agentic_chunk_precaching.py`

PBKV 예측기(A)와 KVFoldAccumulativeRadixCache(B)를 통합하는
에이전틱 청크 사전 누산 파이프라인.

**인터페이스 계약**:
1. PBKV 예측기가 미래 N 스텝 내 재사용 확률 높은 청크 세트 S를 결정 (chunk_key 세트 반환)
2. KVFoldAccumulativeRadixCache가 S의 청크들을 foldl 순서로 사전 누산 처리
3. 누산된 KV 상태를 RadixCache 리프 노드에 "pre-folded prefix"로 등록
4. 에이전트 요청 시 pre-folded prefix에서 추가 청크만 incremental foldl 처리
5. 예측 미스 시 fallback: `fallback_to_radix_attention()` — 기존 RadixAttention 경로

```python
# 의사코드 — src/cache/agentic_chunk_precaching.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.kv_fold_accumulative import KVFoldAccumulativeRadixCache, KVFoldConfig
from src.scheduler.pbkv_agent_segment_scheduler import (
    PBKVAgentSegmentPreservationScheduler, PBKVConfig
)


@dataclass
class AgenticPreCachingConfig:
    kvfold: KVFoldConfig = None
    pbkv: PBKVConfig = None
    precache_top_k: int = 10    # 사전 누산할 최대 청크 수
    precache_min_prob: float = 0.5  # 사전 누산 최소 재사용 확률 임계값

    def __post_init__(self):
        if self.kvfold is None:
            self.kvfold = KVFoldConfig()
        if self.pbkv is None:
            self.pbkv = PBKVConfig()


class AgenticChunkPreCachingPipeline(CacheStore):
    """PBKV 예측 + KVFold 사전 누산 통합 파이프라인 (Activity A+B).

    CacheStore 인터페이스 완전 구현 (KVFoldAccumulativeRadixCache에 위임).

    핵심 흐름:
      1. precache_predicted_chunks() — 배치 전 예측 + 사전 누산 실행
      2. get_with_precache() — 요청 처리 시 pre-folded prefix 우선 활용
      3. 예측 미스 → fallback_to_radix_attention()
    """

    def __init__(self, config: AgenticPreCachingConfig) -> None:
        self.config = config
        self.fold_cache = KVFoldAccumulativeRadixCache(config.kvfold)
        self.scheduler = PBKVAgentSegmentPreservationScheduler(
            self.fold_cache, config.pbkv
        )
        # pre-folded prefix 레지스트리: fold_key → {'chunk_ids': [...], 'prob': float}
        self._prefolded_registry: Dict[str, Dict] = {}
        self._hits = 0
        self._misses = 0
        self._precache_hits = 0   # pre-folded prefix에서 발생한 히트 수
        self._fallback_count = 0  # 예측 미스로 fallback 발생 수

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 (KVFoldAccumulativeRadixCache 위임)             #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        self.fold_cache.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        result = self.fold_cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self.fold_cache.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self.fold_cache.memory_bytes()

    def reset_stats(self) -> None:
        self.fold_cache.reset_stats()
        self._hits = 0
        self._misses = 0
        self._precache_hits = 0
        self._fallback_count = 0

    # ------------------------------------------------------------------ #
    # A+B 파이프라인 API                                                    #
    # ------------------------------------------------------------------ #

    def precache_predicted_chunks(
        self,
        agent_id: str,
        candidate_chunk_tokens_list: List[List[int]],
        # 후보 청크들의 토큰 ID 리스트 (순서: foldl 처리 순서)
        layer_idx: int = 0,
    ) -> Optional[str]:
        """PBKV 예측 + KVFold 사전 누산 실행.

        반환: fold_key (사전 누산된 pre-folded prefix 키) 또는 None (예측 확률 임계값 미달)

        의사코드:
          # 1. PBKV 예측기로 각 청크의 재사용 확률 추정
          scored_chunks = []
          for chunk_tokens in candidate_chunk_tokens_list:
              req_proxy = InferenceRequest(
                  request_id=f"{agent_id}_{hash(tuple(chunk_tokens))}",
                  token_ids=chunk_tokens,
              )
              prob = self.scheduler._predict_segment_reuse(req_proxy)
              scored_chunks.append((prob, chunk_tokens))

          # 2. 확률 높은 상위 precache_top_k 청크 선택
          scored_chunks.sort(key=lambda x: -x[0])
          selected = [
              (p, t) for p, t in scored_chunks[:config.precache_top_k]
              if p >= config.precache_min_prob
          ]
          if not selected:
              return None  # 임계값 미달 — 사전 누산 불필요

          # 3. KVFold foldl 사전 누산 (원래 순서 복원)
          original_order = [t for _, t in scored_chunks if (_, t) in [(p,t) for p,t in selected]]
          fold_key = None
          for chunk_tokens in original_order:
              fold_key, _ = self.fold_cache.fold_chunk(chunk_tokens, layer_idx, fold_key)

          # 4. RadixCache 리프 노드에 pre-folded prefix 등록
          if fold_key:
              accumulated = self.fold_cache.get_folded_prefix(fold_key)
              chunk_ids = [hash(tuple(t)) for _, t in selected]
              self.fold_cache.register_prefolded_prefix(fold_key, accumulated, chunk_ids)
              self._prefolded_registry[fold_key] = {
                  'chunk_ids': chunk_ids,
                  'prob': sum(p for p, _ in selected) / len(selected),
              }
          return fold_key
        """
        ...

    def get_with_precache(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
        precache_fold_key: Optional[str] = None,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int], Optional[torch.Tensor]]:
        """pre-folded prefix 우선 활용 조회.

        반환:
          hits: [(chunk_idx, kv)]
          misses: [chunk_idx]
          fold_prefix: Optional[Tensor] — 사용 가능한 pre-folded prefix

        의사코드:
          # 1. pre-folded prefix 조회
          if precache_fold_key and precache_fold_key in self._prefolded_registry:
              fold_prefix = self.fold_cache.get_folded_prefix(precache_fold_key)
              self._precache_hits += 1
          else:
              fold_prefix = None

          # 2. 세그먼트 조회 (RadixAttention + foldl fallback)
          hits, misses, _ = self.fold_cache.get_segments_with_fold(
              token_ids, layer_idx, precache_fold_key
          )
          return hits, misses, fold_prefix
        """
        ...

    def fallback_to_radix_attention(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """예측 미스 시 기존 RadixAttention(SegmentedHashCache) 경로로 fallback.

        의사코드:
          self._fallback_count += 1
          return self.fold_cache._store.get_segments(token_ids, layer_idx)
        """
        ...

    def noncontiguous_hit_rate(self) -> float:
        return self.fold_cache.noncontiguous_hit_rate()

    def precache_efficiency(self) -> float:
        """사전 누산 히트율: precache_hits / total_hits."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._precache_hits / total
```

---

## Activity C — Accuracy Preservation 검증 계획

**Activity C(SRFTFusedINT4KVKernel)를 포함하므로 이 섹션은 필수다.**
**이 계획 없이 Spec.md는 불완전하다.**

### perplexity 측정 계획

| 항목 | 내용 |
|------|------|
| **데이터셋 (단위 테스트)** | 합성: `torch.randn` 기반 KV 텐서 (훈련-프리 즉시 검증). `n_tokens=64`, `n_heads=8`, `d_head=64`, 독립 seed(42 ≠ 검증 seed 99) |
| **데이터셋 (통합 테스트)** | WikiText-2 test split (stride=512, max_length=2048). 실 모델 없는 환경에서는 합성 proxy 사용. |
| **측정 방법** | `src/metrics/perplexity.py`의 3개 함수 그대로 사용: `attention_output_relative_error()`, `attention_kl_divergence()`, `cosine_similarity_output()` |
| **허용 오차** | attention output relative error < 0.01 (1% 이내) — evaluation_criteria.md §4 필수 |
| **group_size 스윕** | group_size = 64, 128, 256 각각에서 perplexity proxy 측정 → 압축률 vs 정확도 곡선 |
| **캘리브레이션 독립 검증** | encode에 사용한 KV와 다른 seed의 독립 KV로 정확도 측정 (seed 99) |
| **결과 저장** | `results/2026-05-13/perplexity_sweep.json` |
| **SRFT ON/OFF 비교** | `use_srft=True` vs `use_srft=False` (단순 INT4) 정확도 차이 측정 — SRFT Gaussianization 효과 실증 |

### 태스크 정확도 측정 계획 (LongBench proxy)

| 항목 | 내용 |
|------|------|
| **proxy 지표 1** | attention output cosine similarity >= 0.99 (압축 전후) |
| **proxy 지표 2** | attention score KL divergence < 0.015 (압축 전후) |
| **proxy 지표 3** | relative output error < 0.01 (1% 이내) |
| **허용 오차** | evaluation_criteria.md §4 필수: ±1% 이내 |
| **group_size별 검증** | group_size=128 (기본값)에서 모든 proxy 지표 통과 확인 |
| **RateQuant 어댑터 검증** | 헤드별 비트폭 4/8 혼합 시나리오에서도 ±1% 이내 확인 |
| **결과 저장** | `results/2026-05-13/accuracy_proxy_results.json` |

### 압축률-정확도 트레이드오프 예상값

| group_size | 메모리 감소율 | 예상 relative error | 예상 KL | 예상 cosine | SRFT 효과 |
|-----------|------------|-------------------|---------|-------------|-----------|
| 256 | −62% | < 0.004 | < 0.006 | > 0.999 | 아웃라이어 완전 억제 |
| 128 | −63% | < 0.007 | < 0.010 | > 0.995 | 아웃라이어 억제 |
| 64  | −65% | < 0.010 | < 0.014 | > 0.990 | 부분 억제 |

**SRFT가 단순 INT4 대비 정확도를 더 잘 보존하는 이유**:
단순 INT4는 아웃라이어 채널 값이 범위를 초과해 클리핑 오류가 크다.
SRFT FFT가 채널값 분포를 Gaussianize하여 아웃라이어를 제거하고,
모든 채널이 비슷한 스케일을 가져 INT4 양자화 오류가 최소화된다.
그룹별 스케일(λ)이 각 group_size 범위 내 정밀한 스케일 팩터를 제공한다.

### 검증 테스트 파일 — `tests/unit/test_srft_int4_accuracy.py`

```python
# 필수 테스트 케이스 — tests/unit/test_srft_int4_accuracy.py

import json
import os
import pytest
import torch

from src.cache.srft_int4_kv_kernel import SRFTFusedINT4KVKernel, SRFTInt4Config
from src.metrics.perplexity import (
    attention_output_relative_error,
    attention_kl_divergence,
    cosine_similarity_output,
)


@pytest.fixture
def kernel_default():
    """group_size=128, INT4, SRFT ON 기본 커널."""
    config = SRFTInt4Config(n_heads=8, d_head=64, group_size=128, n_bits=4, seed=42)
    return SRFTFusedINT4KVKernel(config)


def test_encode_decode_shape(kernel_default):
    """encode → decode 후 shape 보존: [n_tokens, 2, n_heads, d_head]."""
    torch.manual_seed(42)
    kv = torch.randn(64, 2, 8, 64)
    encoded = kernel_default.encode(kv)
    recovered = kernel_default.decode(encoded)
    assert recovered.shape == kv.shape


def test_memory_reduction_ratio(kernel_default):
    """INT4 nibble 패킹으로 메모리 >= 60% 감소. evaluation_criteria.md §4."""
    ratio = kernel_default.memory_reduction_ratio(n_tokens=64, d_head=64, n_heads=8)
    assert ratio >= 0.60, f"Memory reduction {ratio:.4f} below 60% target"


def test_accuracy_relative_error_within_1pct(kernel_default):
    """PRIMARY ±1% 정확도 보존: relative output error < 0.01.
    group_size=128, SRFT ON. evaluation_criteria.md §4 필수."""
    torch.manual_seed(99)  # 독립 seed
    kv = torch.randn(32, 2, 8, 64)
    k_orig = kv[:, 0, 0, :]   # head=0 key
    v_orig = kv[:, 1, 0, :]   # head=0 value
    encoded = kernel_default.encode(kv)
    kv_rec = kernel_default.decode(encoded)
    k_comp = kv_rec[:, 0, 0, :]
    v_comp = kv_rec[:, 1, 0, :]
    q = torch.randn(8, 64)
    error = attention_output_relative_error(q, k_orig, v_orig, k_comp, v_comp)
    assert error < 0.01, f"Relative error {error:.4f} exceeds ±1% limit"


def test_kl_divergence_within_threshold(kernel_default):
    """KL 발산 < 0.015. evaluation_criteria.md §4."""
    torch.manual_seed(77)
    kv = torch.randn(32, 2, 8, 64)
    encoded = kernel_default.encode(kv)
    kv_rec = kernel_default.decode(encoded)
    q = torch.randn(8, 64)
    kl = attention_kl_divergence(q, kv[:, 0, 0, :], kv_rec[:, 0, 0, :])
    assert kl < 0.015, f"KL divergence {kl:.4f} exceeds 0.015 threshold"


def test_cosine_similarity_above_threshold(kernel_default):
    """Cosine similarity >= 0.99. evaluation_criteria.md §4."""
    torch.manual_seed(55)
    kv = torch.randn(32, 2, 8, 64)
    encoded = kernel_default.encode(kv)
    kv_rec = kernel_default.decode(encoded)
    q = torch.randn(8, 64)
    sim = cosine_similarity_output(
        q, kv[:, 0, 0, :], kv[:, 1, 0, :],
        kv_rec[:, 0, 0, :], kv_rec[:, 1, 0, :],
    )
    assert sim >= 0.99, f"Cosine similarity {sim:.4f} below 0.99 threshold"


def test_srft_vs_plain_int4_accuracy():
    """SRFT ON이 OFF(단순 INT4)보다 relative error가 낮음을 검증.
    SRFT Gaussianization 효과 실증."""
    torch.manual_seed(42)
    # 아웃라이어 채널이 있는 KV (실제 모델 KV 모방)
    kv = torch.randn(32, 2, 8, 64)
    kv[:, :, :, ::8] *= 10.0  # 1/8 채널에 아웃라이어 주입

    config_srft = SRFTInt4Config(n_heads=8, d_head=64, group_size=128, use_srft=True, seed=42)
    config_plain = SRFTInt4Config(n_heads=8, d_head=64, group_size=128, use_srft=False, seed=42)
    kernel_srft = SRFTFusedINT4KVKernel(config_srft)
    kernel_plain = SRFTFusedINT4KVKernel(config_plain)

    q = torch.randn(8, 64)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]

    enc_srft = kernel_srft.encode(kv)
    rec_srft = kernel_srft.decode(enc_srft)
    err_srft = attention_output_relative_error(q, k_orig, v_orig, rec_srft[:, 0, 0, :], rec_srft[:, 1, 0, :])

    enc_plain = kernel_plain.encode(kv)
    rec_plain = kernel_plain.decode(enc_plain)
    err_plain = attention_output_relative_error(q, k_orig, v_orig, rec_plain[:, 0, 0, :], rec_plain[:, 1, 0, :])

    assert err_srft <= err_plain, (
        f"SRFT error {err_srft:.4f} should be <= plain INT4 error {err_plain:.4f}"
    )


def test_independent_seed_accuracy(kernel_default):
    """캘리브레이션과 완전히 독립된 seed에서도 ±1% 보존.
    evaluation_criteria.md §4 필수 요건의 핵심 증거."""
    torch.manual_seed(999)
    test_kv = torch.randn(32, 2, 8, 64)
    q = torch.randn(8, 64)
    encoded = kernel_default.encode(test_kv)
    kv_rec = kernel_default.decode(encoded)
    error = attention_output_relative_error(
        q, test_kv[:, 0, 0, :], test_kv[:, 1, 0, :],
        kv_rec[:, 0, 0, :], kv_rec[:, 1, 0, :],
    )
    assert error < 0.01, f"Independent test error {error:.4f} exceeds ±1%"


def test_group_size_sweep():
    """group_size 64/128/256 스윕 — 압축률 vs 정확도 곡선.
    결과를 results/2026-05-13/perplexity_sweep.json에 저장."""
    torch.manual_seed(42)
    kv = torch.randn(64, 2, 8, 64)
    q = torch.randn(8, 64)
    results = {}
    for gs in [64, 128, 256]:
        config = SRFTInt4Config(n_heads=8, d_head=64, group_size=gs, seed=42)
        kernel = SRFTFusedINT4KVKernel(config)
        encoded = kernel.encode(kv)
        kv_rec = kernel.decode(encoded)
        err = attention_output_relative_error(
            q, kv[:, 0, 0, :], kv[:, 1, 0, :],
            kv_rec[:, 0, 0, :], kv_rec[:, 1, 0, :],
        )
        mem_red = kernel.memory_reduction_ratio(n_tokens=64, d_head=64, n_heads=8)
        results[str(gs)] = {
            'group_size': gs,
            'memory_reduction': mem_red,
            'relative_error': err,
            'pass_1pct': err < 0.01,
        }
        assert kv_rec.shape == kv.shape
    os.makedirs("results/2026-05-13", exist_ok=True)
    with open("results/2026-05-13/perplexity_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
```

---

## 크로스 Activity 통합 — A+B+C 상호작용

### 통합 흐름

```
에이전트 요청 배치 도착
        │
        ▼
[1. Activity A] PBKVAgentSegmentPreservationScheduler
  - 미래 N 스텝 재사용 확률 예측 (MLP predictor)
  - GPU 보존 / 호스트 퇴거 결정 갱신
  - 요청 배치 순서 결정 (재사용 확률 × 공정성 가중치)
        │
        ▼
[2. Activity A+B] AgenticChunkPreCachingPipeline
  - 예측 확률 >= precache_min_prob 청크 사전 누산
  - KVFoldAccumulativeRadixCache.fold_chunk() 순차 실행
  - pre-folded prefix 노드 RadixCache에 등록
        │
        ├──[예측 히트]──────────────────────────────────────┐
        │                                                    │
        ▼                                                    │
[3. Activity B] KVFoldAccumulativeRadixCache                │
  - RadixAttention prefix 매칭 시도                          │
  - 미스 시 foldl fallback                                   │
  - 히트 청크 KV 반환                                         │
        │                                                    │
        ▼                                                    │
[4. Activity C] SRFTFusedINT4KVKernel                      │
  - 누산/신규 KV를 SRFT+INT4 압축                            │
  - RateQuant 어댑터로 헤드별 비트폭 적용                     │
  - CacheStore.compression_hook() 통해 투명하게 적용          │
        │                                                    │
        ▼                                                    │
서빙 결과 ←──────────────────────────────────────────────────┘
  [예측 미스 → fallback_to_radix_attention()]
```

### 인터페이스 결합 지점

| 연결 지점 | 인터페이스 | 설명 |
|-----------|-----------|------|
| A → B | `AgenticChunkPreCachingPipeline.precache_predicted_chunks()` | 예측기가 청크 세트 S 결정 → KVFold가 foldl 누산 |
| B → A (결과 피드백) | `PBKVAgentSegmentPreservationScheduler.update_agent_history()` | 실제 히트된 청크 → 이력 갱신 |
| B → C | `KVFoldAccumulativeRadixCache.put()` 내부에서 `compression_hook()` 호출 | 누산 KV 저장 시 SRFT+INT4 자동 압축 |
| C → B (선택적) | `SRFTFusedINT4KVKernel.from_ratequant()` | RateQuant 비트폭 → SRFT 커널 동적 비트폭 |
| A → C | `PBKVAgentSegmentPreservationScheduler`가 낮은 확률 세그먼트를 INT4로 강등 | 예측 기반 적응 비트폭 (A-2 아이디어 부분 통합) |

### CompressedFoldAccumulator (B+C 핵심 통합 모듈)

누산 KV 상태를 시간 기준으로 분리하여 관리:
- 최신 `window_size` 청크: FP16 보존 (정밀도 우선)
- 오래된 청크: SRFT+INT4 압축 강등 (메모리 우선)

이 로직은 `KVFoldAccumulativeRadixCache.fold_chunk()` 내부에서
`SRFTFusedINT4KVKernel` 인스턴스를 주입받아 처리한다.
`KVFoldConfig`에 `compressor: Optional[SRFTFusedINT4KVKernel] = None` 필드 추가.

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-13.yaml
experiment:
  date: "2026-05-13"
  activity: "A+B+C"
  description: >
    PBKVAgentSegmentPreservationScheduler(A-1) +
    KVFoldAccumulativeRadixCache(B-1) + SRFTFusedINT4KVKernel(C-2) +
    AgenticChunkPreCachingPipeline(Cross-1) A+B+C 통합.
    PBKV 예측 기반 에이전트 KV 보존 + foldl 누산 비연속 재사용 +
    SRFT+INT4 3× 메모리 압축.

pbkv_scheduler:
  segment_emb_dim: 256           # 세그먼트 임베딩 차원
  history_steps: 10              # 과거 호출 이력 스텝
  prediction_horizon: 5          # 미래 예측 범위
  gpu_preserve_threshold: 0.6    # GPU 보존 임계값
  host_evict_threshold: 0.3      # 호스트 퇴거 임계값
  preemption_margin: 0.3         # Lipschitz 강건성 마진
  fairness_max_wait: 10          # 공정성 최대 대기
  chunk_size: 128
  seed: 42

kv_fold:
  chunk_size: 128
  max_entries: 2000
  drift_threshold: 1e-3          # 플래토 감지 임계값
  max_fold_depth: 511            # 최대 foldl 체인 깊이
  enable_streaming_fallback: true
  window_size: 32                # StreamingLLM 슬라이딩 윈도우 (청크 수)
  d_head: 64
  n_heads: 8
  n_layers: 12
  seed: 42

srft_int4:
  n_heads: 8
  d_head: 64                     # 단위 테스트: 64
  group_size: 128                # 그룹별 abs-max 양자화 그룹 크기 (64/128/256 스윕)
  n_bits: 4                      # INT4
  use_srft: true                 # SRFT Gaussianization 활성화
  ratequant_adapter: false       # RateQuant 헤드별 비트폭 어댑터 (선택적)
  seed: 42

agentic_precaching:
  precache_top_k: 10             # 사전 누산 최대 청크 수
  precache_min_prob: 0.5         # 사전 누산 최소 확률 임계값

benchmark:
  accuracy:
    method: "attention_output_proxy"
    proxy_tolerance: 0.01             # relative error < 1%
    kl_tolerance: 0.015               # KL divergence < 0.015
    cosine_min: 0.99                  # cosine similarity >= 0.99
    perplexity_dataset: "wikitext-2"
    perplexity_tolerance_pct: 1.0    # evaluation_criteria.md §4 필수
    task_accuracy_tolerance_pct: 1.0  # LongBench proxy ±1%
  group_size_sweep:
    sizes: [64, 128, 256]
  niah:
    context_lengths: [16384, 65536, 131072]   # 16K / 64K / 128K
    target_accuracy_pct: 100.0               # 100% 정확 검색
  hit_rate:
    target_noncontiguous_fraction: 0.30      # 비연속 히트 >= 30%
  memory_reduction:
    target_ratio: 0.30                       # 최소 30% (목표 60%)
    target_ratio_goal: 0.60
  throughput:
    target_improvement_pct: 20              # 베이스라인 대비 +20%
  scheduling_overhead:
    ttft_p50_max_increase_pct: 5            # TTFT p50 +5% 이내
  agentic_workload:
    workflow_steps: [5, 10]                 # 동적 워크플로우 스텝 수
    ttft_reduction_target_x: 1.39          # 레이턴시 1.39× 단축 목표

seed: 42
results_dir: "results/2026-05-13"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_kv_fold_accumulative.py` — KVFoldAccumulativeRadixCache 단위 테스트
      (CacheStore 인터페이스 준수, fold_chunk 누산, drift 플래토 감지, noncontiguous_hit_rate,
      register_prefolded_prefix, foldl 체인 깊이 통계)
- [ ] `tests/unit/test_srft_int4_kv_kernel.py` — SRFTFusedINT4KVKernel 단위 테스트
      (encode/decode shape, nibble 패킹 정확도, 메모리 감소율, RateQuant 어댑터)
- [ ] `tests/unit/test_srft_int4_accuracy.py` — **Activity C 필수**: accuracy preservation 검증
      (`test_accuracy_relative_error_within_1pct`, `test_independent_seed_accuracy`,
      `test_srft_vs_plain_int4_accuracy`, `test_group_size_sweep` 필수 포함)
- [ ] `tests/unit/test_pbkv_scheduler.py` — PBKVAgentSegmentPreservationScheduler 단위 테스트
      (MLP predictor, 보존/퇴거 결정, Lipschitz 강건성, 공정성, 에이전트 이력 갱신)
- [ ] `tests/unit/test_agentic_precaching.py` — AgenticChunkPreCachingPipeline 단위 테스트
      (사전 누산 실행, pre-folded prefix 등록, 예측 미스 fallback,
      CacheStore 인터페이스 준수, precache_efficiency 측정)
- [ ] `tests/integration/test_cross_abc_kvfold_srft.py` — A+B+C 통합 E2E 테스트
      (비연속 히트율 >= 30%, 메모리 감소 >= 30%, 복합 정확도 ±1% 이내,
      에이전틱 워크로드 TTFT 측정, 처리량 측정, NIAH proxy 검증)

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 전부 통과** (기존 테스트 회귀 없음 포함)
2. **통합 테스트 전부 통과**
3. **Activity A (evaluation_criteria.md §2) 기준 충족**:
   - 스케줄링 오버헤드 TTFT p50 +5% 이내
   - 에이전틱 워크로드(5~10 스텝 동적 워크플로우)에서 캐시 히트율 +10%p 이상
   - 요청 처리 공정성: 최대 대기 시간 2× 초과하지 않음
4. **Activity B (evaluation_criteria.md §3) 기준 충족**:
   - 비연속 세그먼트 히트율 >= 30% (전체 히트의 30% 이상이 비연속 구간)
   - 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상
   - NIAH 16K/64K/128K 컨텍스트 검색 정확도 100% (foldl 누산 경로)
   - KV Memory Footprint 베이스라인 대비 +20% 이내 (SRFT 압축으로 오히려 감소 목표)
5. **Activity C (evaluation_criteria.md §4) 필수 기준 충족** (모두 Fail 시 전체 Fail):
   - perplexity proxy (attention output relative error) < 1% — `test_accuracy_relative_error_within_1pct` 통과
   - 캘리브레이션 독립 검증 샘플 기준 ±1% 이내 — `test_independent_seed_accuracy` 통과
   - KL divergence < 0.015, cosine similarity >= 0.99 통과
   - KV Memory Reduction >= −30% (목표 −60%)
   - Encode/Decode 추가 지연 TTFT +10% 이내
   - SRFT ON이 SRFT OFF(단순 INT4) 대비 정확도 향상 실증 — `test_srft_vs_plain_int4_accuracy` 통과
   - group_size 64/128/256 스윕 결과 `results/2026-05-13/perplexity_sweep.json` 저장됨
6. **크로스 조합 (evaluation_criteria.md §5) 기준 충족** (C 포함이므로 accuracy 필수):
   - 복합 처리량 향상: 단일 Activity 대비 추가 +5% 이상
   - 복합 메모리 감소: 단일 Activity 대비 추가 −10% 이상
   - 복합 적용 후에도 accuracy ±1% 이내 유지
7. **CacheStore 인터페이스 준수**:
   - `KVFoldAccumulativeRadixCache`가 CacheStore 6개 추상 메서드 완전 구현
   - `AgenticChunkPreCachingPipeline`이 CacheStore 6개 추상 메서드 완전 구현
   - `SRFTFusedINT4KVKernel`이 `compression_hook()` 호환 인터페이스 구현
8. **설정 YAML 존재**: `configs/experiments/2026-05-13.yaml` 생성됨
9. **타입 힌트**: 모든 공개 함수·메서드에 완전한 타입 힌트
10. **시드 고정 재현성**: seed=42 고정 시 동일 결과 재현
11. **결과 저장**: `results/2026-05-13/metrics.json`, `perplexity_sweep.json`,
    `accuracy_proxy_results.json` 생성됨

---

## 구현 순서 (implementer 참고)

1. **`src/cache/srft_int4_kv_kernel.py`** — Activity C 핵심 구현 (A/B와 독립).
   순서: `encode()` [nibble 패킹 핵심] → `decode()` → `memory_reduction_ratio()` →
   `compression_hook()` → `from_ratequant()` 어댑터.
   구현 직후 `test_srft_int4_accuracy.py`의 `test_accuracy_relative_error_within_1pct` 통과 확인.
   **Activity C 필수 조건을 먼저 만족시켜야 이후 통합 가능.**

2. **`tests/unit/test_srft_int4_accuracy.py`** — Activity C 필수 테스트 먼저 작성.
   모든 proxy 지표 통과 + group_size 스윕 + SRFT vs plain INT4 비교 확인.

3. **`src/cache/kv_fold_accumulative.py`** — Activity B 구현.
   순서: `_compute_chunk_kv()` → `fold_chunk()` [드리프트 플래토 포함] →
   `get_folded_prefix()` → `register_prefolded_prefix()` →
   `get_segments_with_fold()` → `noncontiguous_hit_rate()` → CacheStore 인터페이스 구현.

4. **`tests/unit/test_kv_fold_accumulative.py`** — B 단위 테스트.
   fold_chunk 누산 shape, drift 플래토 감지 수렴, noncontiguous_hit_rate 추적.

5. **`src/scheduler/pbkv_agent_segment_scheduler.py`** — Activity A 구현.
   순서: `_SegmentMLP` → `_get_segment_embedding()` → `_get_history_vector()` →
   `_predict_segment_reuse()` → `schedule()` → `update_preservation_policy()` →
   `update_agent_history()` → `update_wait()`.

6. **`tests/unit/test_pbkv_scheduler.py`** — A 단위 테스트.
   MLP 예측기 입출력 shape, 보존/퇴거 결정 Lipschitz 마진, 공정성 wait 증가.

7. **`src/cache/agentic_chunk_precaching.py`** — A+B 통합.
   순서: `precache_predicted_chunks()` → `get_with_precache()` →
   `fallback_to_radix_attention()` → CacheStore 인터페이스 구현.

8. **`tests/unit/test_agentic_precaching.py`** — A+B 파이프라인 단위 테스트.

9. **`tests/unit/test_srft_int4_kv_kernel.py`** — SRFT+INT4 나머지 단위 테스트.

10. **`tests/integration/test_cross_abc_kvfold_srft.py`** — A+B+C 통합 E2E 테스트.
    AgenticChunkPreCachingPipeline + SRFTFusedINT4KVKernel 결합 후
    에이전틱 워크로드 시뮬레이션(5~10 스텝 동적 워크플로우) 실행.
    비연속 히트율, TTFT p50/p99, 처리량, 메모리 감소율, accuracy proxy 복합 측정.

11. **`configs/experiments/2026-05-13.yaml`** 작성.

---

## 기존 파일 보존 목록

이번 사이클에서 수정하지 않는 파일 (기존 테스트 회귀 없이 통과해야 함):

- `src/cache/base.py` — 수정 불필요 (store_pre_rope/load_with_rope 이미 존재)
- `src/cache/segmented.py` — KVFoldAccumulativeRadixCache의 세그먼트 백엔드로 재사용
- `src/cache/rope_reencoding_cache.py`, `src/cache/mixed_dim_codec.py`, `src/cache/adapshot_pipeline.py` — 이전 사이클 보존
- `src/cache/ratequant_codec.py` — SRFT 어댑터 입력으로 재사용 (수정 불필요)
- `src/scheduler/cache_aware_scheduler.py` — 기존 스케줄러 독립 유지
- `src/metrics/perplexity.py` — accuracy 검증 함수 그대로 사용
- `src/engine/runner.py` — 모든 모델 API 호출은 이 파일을 통해
- 기타 모든 기존 캐시 구현체, 이전 사이클 테스트 파일
