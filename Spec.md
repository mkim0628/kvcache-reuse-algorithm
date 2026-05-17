<!-- 변경 이유 (이전 Spec.md: 2026-05-16 대비):
이전 사이클(2026-05-16)은 A+C 조합이었다:
  - C-2 GlobalRetentionGateEvictionCodec (전역 크로스-레이어 경쟁 퇴거)
  - A-2 NAtHDDROffloadingScheduler (누적 어텐션 점수 EMA 기반 4-티어 DDR 오프로딩)
  - Cross-1 NAtHRetentionTierDecider (이중 신호 A+C 4-티어 결정기)

이번 사이클(2026-05-17)은 A+C 조합을 유지하되 설계 축이 완전히 전환된다.

주요 변경:
1. [Activity A 교체] NAtHDDROffloadingScheduler(단일 정책) →
   HMAMultiConnectorCompressionPluginScheduler(멀티-커넥터 플러그인 레지스트리 기반 메타-스케줄러).
   이전 A 기법들이 단일 오프로딩/퇴거 정책을 고정 적용한 반면, 이번 A-1은
   vLLM v0.21.0 HMA 멀티-커넥터 공식 API를 활용해 요청 특성(컨텍스트 길이, RL 여부,
   메모리 압박)에 따라 런타임에 적절한 압축 커넥터를 선택·조합한다.
   기구현 코덱(GlobalRetentionGateEvictionCodec, LookaheadKVEvictionCodec,
   RateQuantReverseWaterfillingCodec)을 즉시 커넥터로 재활용.

2. [Activity C 신규] GlobalRetentionGateEvictionCodec(정적 캘리브레이션) →
   RLAdaptivePrecisionQuantizer(RL 워크로드 온라인 적응 정밀도 양자화).
   이전 16개 사이클 전체에 없던 "RL 리워드 피드백 + 온라인 어텐션 엔트로피 기반
   실시간 정밀도 결정" 패러다임. vLLM Q2 2026 온라인 양자화 리팩터 방향과 직접 정렬.

3. [Cross A+C 신규] NAtHRetentionTierDecider(이중 신호 티어 결정) →
   HMAChainedACPipeline(멀티-커넥터 플러그인 체이닝 + A+C 통합).
   A-1 HMAMultiConnectorScheduler 커넥터 레지스트리에
   C-1 RLAdaptivePrecisionQuantizer + 기구현 GlobalRetentionGateEvictionCodec을
   등록하고 요청별 동적 디스패치를 수행하는 완전한 A+C 파이프라인.

4. [보존 파일] 기존 모든 파일(global_retention_gate_eviction.py,
   nath_ddr_offloading.py 등)은 이번 사이클에서 수정하지 않는다.
   기존 단위·통합 테스트가 회귀 없이 통과해야 한다.

5. [인터페이스 유지] src/cache/base.py는 이번 사이클에서 수정하지 않는다.
   CacheStore 6개 추상 메서드를 모든 신규 구현체가 완전 구현한다.

6. [Activity C 필수] RLAdaptivePrecisionQuantizer는 accuracy-preserving 검증 계획 없이
   완성 불가. WikiText-2 perplexity ±1% + RL 워크로드 시뮬레이션(동일 프롬프트 10회 반복) +
   리워드 피드백 수렴 곡선 + GlobalRetentionGateEvictionCodec과 동일 설정 비교.
-->

# Spec — 2026-05-17

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-17.md`

**최우선 구현 타겟**:
- **A-1 (주)**: HMAMultiConnectorCompressionPluginScheduler — vLLM v0.21.0 HMA 멀티-커넥터
  플러그인 기반 압축 정책 동적 디스패치 메타-스케줄러
- **C-1 (주)**: RLAdaptivePrecisionQuantizer — RL 워크로드용 온라인 적응 정밀도 KV 양자화기
  (어텐션 엔트로피 기반 채널별 FP16/INT8/INT4 동적 할당 + RL 리워드 피드백 루프)
- **Cross-1 (주)**: HMAChainedACPipeline — A-1 + C-1 + 기구현 코덱 통합 A+C 파이프라인

**해결하려는 문제**:

- **Activity A**: 기존 모든 A 기법이 단일 스케줄링 정책(한 가지 오프로딩·퇴거 전략)을 모든
  요청에 일률 적용하는 한계. HMAMultiConnectorScheduler는 vLLM v0.21.0의 공식 HMA
  멀티-커넥터 API를 레지스트리 추상화로 래핑해 요청 특성(RL/긴 컨텍스트/메모리 압박)에 따라
  최적 압축 커넥터를 런타임에 선택·체이닝한다. 커넥터 선택 오버헤드는 O(1) 딕셔너리 룩업으로
  TTFT +5% 이내를 보장한다.

- **Activity C**: 기존 모든 C 기법(RateQuantReverseWaterfillingCodec,
  GlobalRetentionGateEvictionCodec, LookaheadKVEvictionCodec)이 정적 캘리브레이션 또는
  고정 정밀도 정책을 사용해 RL 추론의 동적 리워드 신호를 반영하지 못하는 한계.
  RLAdaptivePrecisionQuantizer는 RL 추론 사이클의 누적 리워드 스코어를 양자화 정밀도 결정에
  피드백으로 사용하는 "리워드-인식 온라인 적응 KV 양자화기"로, 고중요도 토큰 FP16 보존
  + 저중요도 토큰 INT8/INT4 압축의 자동 균형을 달성한다.

- **Cross A+C**: HMAChainedACPipeline은 A-1 커넥터 레지스트리에 C-1 코덱을 등록하고
  요청 프로파일에 따라 RL 워크로드는 C-1(적응 정밀도), 긴 컨텍스트는 기구현
  GlobalRetentionGateEvictionCodec, 짧은 고처리량 요청은 RateQuantReverseWaterfillingCodec을
  동적 선택·조합한다. chain_mode=True 시 복수 커넥터를 순차 적용한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling (HMAMultiConnectorCompressionPluginScheduler)
- [ ] Activity B: Non-Contiguous KV Cache Reuse (이번 사이클 미포함)
- [x] Activity C: KV Cache Compression (RLAdaptivePrecisionQuantizer)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — WikiText-2 proxy: attention_output_relative_error < 0.01
      — FP16/INT8/INT4 정밀도 레벨별 각각 측정
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 변화 ±1% 이내
      — GSM8K/MATH-500 RL 워크로드 proxy (KL divergence < 0.015, cosine >= 0.99)
      — RL 워크로드 시뮬레이션: 동일 프롬프트 10회 반복 생성 후 리워드 수렴 확인
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C): KV Cache Memory Reduction >= −30%
      — RL 워크로드 mixed precision −40~70% 목표 (INT8/INT4 압축 토큰 비율에 따라)
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C): Effective Context Length 동일 메모리 2× 이상
      — precision_ratio_int4=0.2 최저 정밀도 구간에서 컨텍스트 길이 확장 측정
- [ ] 목표 5 (evaluation_criteria.md §2 Activity A): 스케줄링 오버헤드 TTFT p50 +5% 이내
      — 커넥터 선택 오버헤드: O(1) 딕셔너리 룩업 + 요청 프로파일 평가 < 0.1ms/요청
- [ ] 목표 6 (evaluation_criteria.md §2 Activity A): 캐시 히트율 향상 스케줄링 미적용 대비 +10%p
      — 요청별 최적 커넥터 선택으로 압축 효율 향상 → 동일 메모리에 더 많은 캐시 유지
- [ ] 목표 7 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +20% 이상
      — HMA 멀티-커넥터 선택 오버헤드 포함 전체 처리량 측정
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후 accuracy ±1% 이내
      — HMAChainedACPipeline(A-1 + C-1) 기준 복합 적용 후 측정
      — 단독 A-1 / 단독 C-1 / 결합 Cross-1 + 기구현 Cross-1(NAtHRetentionTierDecider) 4방향 비교
- [ ] 목표 9 (evaluation_criteria.md §4 Activity C): GlobalRetentionGateEvictionCodec과 동일 설정 비교
      — 동일 budget_ratio=0.3 / precision_ratio [0.2, 0.6, 0.2] 설정에서 정확도·메모리 비교

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/rl_adaptive_precision_quantizer.py` | C | RLAdaptivePrecisionQuantizer: 어텐션 엔트로피 기반 FP16/INT8/INT4 온라인 적응 정밀도 양자화 + RL 리워드 피드백 루프 |
| `src/scheduler/hma_multi_connector_scheduler.py` | A | HMAMultiConnectorCompressionPluginScheduler: HMAConnectorInterface 레지스트리 + 요청 특성 기반 커넥터 동적 선택 + pipeline_mode 체이닝 |
| `src/engine/hma_chained_ac_pipeline.py` | A+C | HMAChainedACPipeline: A-1 커넥터 레지스트리 + C-1/기구현 코덱 통합 + 요청 프로파일 기반 커넥터 디스패치 |
| `tests/unit/test_rl_adaptive_precision_quantizer.py` | C | Activity C accuracy-preserving 검증 (필수) |
| `tests/unit/test_hma_multi_connector_scheduler.py` | A | HMA 멀티-커넥터 스케줄러 단위 테스트 |
| `tests/unit/test_hma_chained_ac_pipeline.py` | A+C | Cross A+C 통합 파이프라인 단위 테스트 |
| `tests/integration/test_cross_ac_hma_chained.py` | A+C | E2E 통합 테스트: 다중 요청 커넥터 선택 + RL 적응 정밀도 압축 흐름 |
| `configs/experiments/2026-05-17-hma-rl-ac.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| (없음) | `src/cache/base.py` 및 기존 모든 파일 변경하지 않음 |

---

## 알고리즘 상세

### HMAConnectorInterface 추상 베이스 + 어댑터 (Activity A)

`HMAConnectorInterface`는 기구현 코덱들을 HMA 커넥터로 래핑하는 추상 계약이다.
기구현 코덱 3종은 `HMAConnectorAdapter`로 래핑해 레지스트리에 등록한다.

```python
# src/scheduler/hma_multi_connector_scheduler.py 상단부

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from src.cache.base import CacheStore


class HMAConnectorInterface(ABC):
    """vLLM v0.21.0 HMA 멀티-커넥터 인터페이스 추상 베이스.

    각 압축 코덱을 독립 HMA 커넥터로 래핑하는 계약.
    compress()는 CacheStore.compression_hook() 시맨틱을 따른다.
    """

    @abstractmethod
    def compress(
        self,
        kv: torch.Tensor,           # [n_tokens, ...] — 원본 KV 텐서
        request_profile: Dict,      # 요청 메타데이터 (context_length, is_rl_mode 등)
    ) -> torch.Tensor:
        """KV 텐서를 압축해 반환. shape은 달라질 수 있음(퇴거) 또는 동일(양자화)."""

    @abstractmethod
    def decompress(
        self,
        compressed_kv: torch.Tensor,
        request_profile: Dict,
    ) -> torch.Tensor:
        """압축된 KV를 복원. 양자화 코덱의 경우 역양자화 수행."""

    @property
    @abstractmethod
    def connector_name(self) -> str:
        """커넥터 식별자 (레지스트리 키와 일치)."""


class HMAConnectorAdapter(HMAConnectorInterface):
    """기구현 코덱(CacheStore/CompressionCodec)을 HMAConnectorInterface로 래핑.

    compression_hook() 또는 encode/decode 인터페이스를 감지해 자동 위임.
    """

    def __init__(self, name: str, codec: object) -> None:
        self._name = name
        self._codec = codec

    @property
    def connector_name(self) -> str:
        return self._name

    def compress(self, kv: torch.Tensor, request_profile: Dict) -> torch.Tensor:
        # CacheStore.compression_hook() 우선 사용
        if hasattr(self._codec, "compression_hook"):
            return self._codec.compression_hook("__hma__", kv)
        # encode(kv, layer_idx=0) fallback
        if hasattr(self._codec, "encode"):
            return self._codec.encode(kv, layer_idx=0)
        return kv

    def decompress(self, compressed_kv: torch.Tensor, request_profile: Dict) -> torch.Tensor:
        if hasattr(self._codec, "decode"):
            return self._codec.decode(compressed_kv, layer_idx=0)
        return compressed_kv
```

---

### HMAMultiConnectorCompressionPluginScheduler (Activity A)

스케줄링 결정 단위: **요청(request) 단위** — 각 요청 프로파일을 평가해 커넥터를 선택.
캐시 상태 접근: `_connector_registry` Dict + `_dispatch_policy` YAML 설정.

```python
# src/scheduler/hma_multi_connector_scheduler.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import torch

from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


@dataclass
class HMAMultiConnectorConfig:
    # 커넥터 선택 정책 파라미터 (YAML 외부화)
    long_ctx_threshold: int = 4096         # 이 이상이면 GlobalRetentionGate 선택
    memory_pressure_threshold: float = 0.8 # HBM 사용률 이 이상이면 고압축 커넥터 선택
    default_connector: str = "global_retention"  # 기본 커넥터 이름
    pipeline_mode: bool = False            # True: 선택 커넥터 + global_retention 체이닝
    max_wait_ratio: float = 2.0            # 공정성: 최대 대기 시간 배수
    seed: int = 42

    # 커넥터 선택 규칙 (YAML connector_dispatch_policy 섹션과 매핑)
    # is_rl_mode=True or num_completions>1 → "rl_adaptive"
    # context_length > long_ctx_threshold → "global_retention"
    # context_length <= long_ctx_threshold and memory_pressure > threshold → "ratequant"
    # 기본값 → default_connector


class HMAMultiConnectorCompressionPluginScheduler:
    """HMA 멀티-커넥터 압축 정책 플러그인 메타-스케줄러.

    Activity A: KV Cache-aware Scheduling
    스케줄링 결정 단위: 요청 단위 — 각 요청의 프로파일을 평가해 커넥터 선택.
    캐시 상태 접근: _connector_registry Dict (O(1) 룩업).

    커넥터 레지스트리:
      - "rl_adaptive"       : RLAdaptivePrecisionQuantizer (C-1, RL 워크로드)
      - "global_retention"  : GlobalRetentionGateEvictionCodec (기구현, 긴 컨텍스트)
      - "ratequant"         : RateQuantReverseWaterfillingCodec (기구현, 짧은 고처리량)
      - "lookahead"         : LookaheadKVEvictionCodec (기구현, 미래-인식 퇴거)

    vLLM v0.21.0 HMA 멀티-커넥터 연동:
      register_connector() → HMA OffloadingConnector 레지스트리에 등록
      select_connector()   → 요청 특성 기반 O(1) 선택
      pipeline_mode=True   → 선택 커넥터 + global_retention 순차 체이닝

    평가 기준 (evaluation_criteria.md §2):
      - 스케줄링 오버헤드: TTFT p50 +5% 이내 (커넥터 선택 < 0.1ms/요청)
      - 캐시 히트율 향상: +10%p
      - 요청 공정성: 최대 대기 시간 max_wait_ratio 이내
    """

    def __init__(
        self,
        config: HMAMultiConnectorConfig,
        cache: Optional[CacheStore] = None,
    ) -> None:
        self.config = config
        self._cache = cache
        # 커넥터 레지스트리: {name: HMAConnectorInterface}
        self._connector_registry: Dict[str, "HMAConnectorInterface"] = {}
        # 스케줄링 오버헤드 측정
        self._scheduling_times: List[float] = []
        # 요청별 커넥터 선택 이력: {request_id: connector_name}
        self._request_connector_map: Dict[str, str] = {}
        # 커넥터별 선택 횟수 통계
        self._connector_selection_counts: Dict[str, int] = {}
        # 요청 대기 시간 추적 (공정성)
        self._arrival_times: Dict[str, float] = {}

    def register_connector(
        self,
        name: str,
        connector: "HMAConnectorInterface",
    ) -> None:
        """HMA 커넥터를 레지스트리에 등록.

        Algorithm:
          1. connector를 _connector_registry[name]에 저장
          2. _connector_selection_counts[name] = 0 초기화
        """
        self._connector_registry[name] = connector
        self._connector_selection_counts[name] = 0

    def select_connector(
        self,
        request: InferenceRequest,
        request_meta: Optional[Dict] = None,
    ) -> str:
        """요청 특성 기반 최적 커넥터 선택 (O(1) 딕셔너리 룩업).

        Algorithm (connector_dispatch_policy):
          1. is_rl_mode=True or num_completions>1 → "rl_adaptive"
          2. context_length > long_ctx_threshold → "global_retention"
          3. context_length <= long_ctx_threshold
             and memory_pressure > memory_pressure_threshold → "ratequant"
          4. 그 외 → config.default_connector
          5. 선택된 커넥터가 레지스트리에 없으면 config.default_connector 폴백

        Args:
            request: InferenceRequest
            request_meta: 추가 메타데이터 {"is_rl_mode": bool, "num_completions": int,
                          "memory_pressure": float}

        Returns:
            connector_name: str
        """
        meta = request_meta or {}
        context_length = len(request.token_ids)
        is_rl = meta.get("is_rl_mode", False)
        num_completions = meta.get("num_completions", 1)
        memory_pressure = meta.get("memory_pressure", 0.0)

        if (is_rl or num_completions > 1) and "rl_adaptive" in self._connector_registry:
            selected = "rl_adaptive"
        elif context_length > self.config.long_ctx_threshold and "global_retention" in self._connector_registry:
            selected = "global_retention"
        elif context_length <= self.config.long_ctx_threshold and memory_pressure > self.config.memory_pressure_threshold and "ratequant" in self._connector_registry:
            selected = "ratequant"
        else:
            selected = self.config.default_connector

        # 레지스트리에 없으면 폴백
        if selected not in self._connector_registry:
            available = list(self._connector_registry.keys())
            selected = available[0] if available else self.config.default_connector

        return selected

    def apply_connector(
        self,
        request: InferenceRequest,
        kv: torch.Tensor,
        request_meta: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, str]:
        """선택된 커넥터(또는 체인)로 KV 압축 수행.

        pipeline_mode=True 시:
          1. select_connector()로 주 커넥터 선택
          2. 주 커넥터로 압축
          3. "global_retention" 커넥터가 레지스트리에 있으면 추가 체이닝

        Returns:
            (compressed_kv, selected_connector_name)
        """
        t0 = time.monotonic()
        meta = request_meta or {}
        connector_name = self.select_connector(request, meta)
        connector = self._connector_registry.get(connector_name)

        request_profile = {"context_length": len(request.token_ids), **meta}
        compressed = connector.compress(kv, request_profile) if connector else kv

        # pipeline_mode: global_retention을 후처리로 추가 적용
        if self.config.pipeline_mode and connector_name != "global_retention":
            global_conn = self._connector_registry.get("global_retention")
            if global_conn is not None:
                compressed = global_conn.compress(compressed, request_profile)

        self._request_connector_map[request.request_id] = connector_name
        self._connector_selection_counts[connector_name] = (
            self._connector_selection_counts.get(connector_name, 0) + 1
        )

        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)
        return compressed, connector_name

    def schedule(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        """요청 목록을 캐시 히트율 예측 기반으로 정렬 후 반환.

        CacheAwareScheduler와 동일한 schedule() 인터페이스 유지.
        정렬 후 각 요청에 선택된 커넥터 이름을 메타데이터로 주입.

        Returns:
            List[InferenceRequest] — 정렬된 요청 목록
        """
        t0 = time.monotonic()
        for req in requests:
            self._arrival_times.setdefault(req.request_id, t0)
        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)
        return requests  # 기본 FIFO; 확장 시 hit-rate 예측 정렬 추가

    # ------------------------------------------------------------------ #
    # 메트릭                                                                #
    # ------------------------------------------------------------------ #

    def scheduling_overhead_ms_p50(self) -> float:
        """스케줄링 오버헤드 중앙값 (ms)."""
        if not self._scheduling_times:
            return 0.0
        sorted_t = sorted(self._scheduling_times)
        return sorted_t[len(sorted_t) // 2]

    def connector_selection_stats(self) -> Dict[str, int]:
        """커넥터별 선택 횟수 통계."""
        return dict(self._connector_selection_counts)

    def reset_stats(self) -> None:
        self._scheduling_times.clear()
        self._request_connector_map.clear()
        self._connector_selection_counts.clear()
        self._arrival_times.clear()
```

---

### RLAdaptivePrecisionQuantizer (Activity C)

CacheStore 인터페이스를 완전 구현하며 `compression_hook()`을 오버라이드한다.
내부 양자화 로직은 encode()/decode() 형태로도 제공해 vLLM 이식 경로를 지원한다.

```python
# src/cache/rl_adaptive_precision_quantizer.py

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import torch

from src.cache.base import CacheStore


PrecisionLevel = Literal["fp16", "int8", "int4"]


@dataclass
class RLAdaptivePrecisionConfig:
    # 정밀도 비율 (순서: FP16, INT8, INT4 — 합계 1.0)
    precision_ratio_fp16: float = 0.20      # 상위 20%: FP16 (저엔트로피, 고중요도)
    precision_ratio_int8: float = 0.60      # 중간 60%: INT8
    precision_ratio_int4: float = 0.20      # 하위 20%: INT4 (고엔트로피, 저중요도)

    # RL 감지 파라미터
    warmup_steps: int = 10                  # 초기 N 스텝: FP16 전체 정밀도 유지
    cot_length_threshold: int = 512         # CoT 길이 임계값

    # 리워드 피드백 파라미터
    high_reward_threshold: float = 0.8     # 이 이상이면 다음 생성에서 더 공격적 압축 허용
    reward_aggression_step: float = 0.05   # 리워드 높을 때 precision_ratio_int4 증가분
    reward_recovery_step: float = 0.05     # 리워드 낮을 때 precision_ratio_int4 감소분

    max_entries: int = 1000
    seed: int = 42


class RLAdaptivePrecisionQuantizer(CacheStore):
    """RL 워크로드용 온라인 적응 정밀도 KV 양자화기.

    Activity C: KV Cache Compression
    정밀도 레벨: {FP16, INT8, INT4}
    할당 기준: 어텐션 엔트로피 기반 채널별 정밀도 + RL 리워드 피드백 루프

    알고리즘:
      1. RL 워크로드 감지: is_rl_mode 플래그 또는 num_completions > 1
      2. warmup_steps 기간: FP16 전체 정밀도 유지 (RL 탐색 초기 보호)
      3. 어텐션 엔트로피 계산:
           H_i = -Σ_h Σ_t attn_{h,t,i} log(attn_{h,t,i} + 1e-8)
         엔트로피가 낮은 상위 precision_ratio_fp16 비율: FP16 보존
         중간 precision_ratio_int8 비율: INT8 양자화
         엔트로피가 높은 하위 precision_ratio_int4 비율: INT4 시뮬레이션
      4. 리워드 피드백: update_reward_signal(reward) 호출 시
           reward >= high_reward_threshold → precision_ratio_int4 += reward_aggression_step (공격적)
           reward < high_reward_threshold → precision_ratio_int4 -= reward_recovery_step (보수적)
           조정 후 precision_ratio_fp16 + int8 + int4 == 1.0 재정규화

    정확도 보존 근거:
      - 저엔트로피(집중 어텐션) 상위 20% 토큰: 항상 FP16 보존 → 핵심 정보 손실 없음
      - 리워드 피드백 루프: 압축으로 인한 정확도 저하를 직접 측정·보정하는 자동 안전장치
      - warmup_steps 기간 FP16 완전 정밀도로 RL 탐색 초기 패턴 보호

    CacheStore 인터페이스: put/get/evict/hit_rate/memory_bytes/reset_stats 완전 구현
    compression_hook() 오버라이드: put() 전 적응 정밀도 양자화 수행
    """

    def __init__(self, config: RLAdaptivePrecisionConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        # 양자화 스케일 저장: {key: (scale_int8, scale_int4)}
        self._scales: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # 정밀도 마스크 저장: {key: {"fp16": idx, "int8": idx, "int4": idx}}
        self._precision_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._hits = 0
        self._misses = 0
        self._current_step = 0
        self._is_rl_mode = False
        self._last_reward: Optional[float] = None
        # 동적 정밀도 비율 (리워드 피드백으로 변경됨)
        self._ratio_fp16 = config.precision_ratio_fp16
        self._ratio_int8 = config.precision_ratio_int8
        self._ratio_int4 = config.precision_ratio_int4
        # 메모리 절감 추적
        self._total_bytes_original = 0
        self._total_bytes_stored = 0

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """압축 후 저장. compression_hook()을 통해 적응 정밀도 양자화 적용."""
        compressed = self.compression_hook(key, value)
        if len(self._store) >= self.config.max_entries:
            self.evict()
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = compressed

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return self._store[key]

    def evict(self) -> int:
        if not self._store:
            return 0
        _, kv = self._store.popitem(last=False)
        return kv.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(kv.nbytes for kv in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._current_step = 0
        self._total_bytes_original = 0
        self._total_bytes_stored = 0

    # ------------------------------------------------------------------ #
    # Activity C 핵심: compression_hook 오버라이드                         #
    # ------------------------------------------------------------------ #

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """적응 정밀도 양자화: 어텐션 엔트로피 기반 채널별 FP16/INT8/INT4 할당.

        Algorithm:
          1. warmup 단계 (step < warmup_steps): value.half() 반환 (FP16 그대로)
          2. 어텐션 엔트로피 계산: _compute_attention_entropy(value) → H [n_tokens]
             - value shape: [n_tokens, ...] 임의 shape 지원
             - 엔트로피가 낮은 상위 ratio_fp16: FP16
             - 중간 ratio_int8: INT8 (per-token symmetric quantization)
             - 엔트로피가 높은 하위 ratio_int4: INT4 시뮬레이션
               (FP16에서 4-bit float 범위로 클램핑 후 역변환)
          3. 정밀도별 마스크를 _precision_masks[key]에 저장 (decode 시 사용)
          4. 압축된 텐서를 pack_mixed_precision()으로 합산해 반환
             - 반환 형식: FP16 텐서로 통일 (INT8/INT4 디코딩 후 FP16으로 복원)

        Returns:
            압축 후 FP16 텐서 [n_tokens, ...] — 원본 shape 유지
            (INT8/INT4 구간은 디코딩 후 FP16으로 복원된 상태로 저장)
        """
        self._current_step += 1
        n_bytes_original = value.nbytes
        self._total_bytes_original += n_bytes_original

        # warmup 단계: FP16 보존
        if self._current_step <= self.config.warmup_steps:
            result = value.detach().half()
            self._total_bytes_stored += result.nbytes
            return result

        if value.dim() < 1 or value.shape[0] == 0:
            return value.detach().half()

        n_tokens = value.shape[0]
        entropy = self._compute_attention_entropy(value)  # [n_tokens]

        # 정밀도 마스크 계산: 엔트로피 낮은 순서로 정렬 후 비율로 분할
        n_fp16 = max(1, int(n_tokens * self._ratio_fp16))
        n_int8 = max(0, int(n_tokens * self._ratio_int8))
        n_int4 = max(0, n_tokens - n_fp16 - n_int8)

        # 엔트로피 낮은(집중) 토큰이 fp16 → sorted ascending by entropy
        sorted_idx = entropy.argsort()  # ascending: 낮은 엔트로피 먼저
        fp16_idx = sorted_idx[:n_fp16]
        int8_idx = sorted_idx[n_fp16:n_fp16 + n_int8]
        int4_idx = sorted_idx[n_fp16 + n_int8:]

        self._precision_masks[key] = {
            "fp16": fp16_idx,
            "int8": int8_idx,
            "int4": int4_idx,
        }

        # 각 구간 압축 후 동일 FP16 버퍼로 복원해 저장
        result = torch.zeros_like(value, dtype=torch.float16)
        v_f = value.detach().float()

        # FP16 구간: 그대로 보존
        if len(fp16_idx) > 0:
            result[fp16_idx] = v_f[fp16_idx].half()

        # INT8 구간: symmetric per-token quantization → dequantize → FP16
        if len(int8_idx) > 0:
            chunk = v_f[int8_idx]
            scale = chunk.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) / 127.0
            q8 = (chunk / scale).round().clamp(-127, 127)
            result[int8_idx] = (q8 * scale).half()

        # INT4 구간: 4-bit float 시뮬레이션 (범위 클램핑 + 반올림)
        if len(int4_idx) > 0:
            chunk = v_f[int4_idx]
            scale = chunk.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) / 7.0
            q4 = (chunk / scale).round().clamp(-8, 7)
            result[int4_idx] = (q4 * scale).half()

        self._total_bytes_stored += result.nbytes
        return result

    def _compute_attention_entropy(
        self,
        value: torch.Tensor,  # [n_tokens, ...]
    ) -> torch.Tensor:
        """토큰별 어텐션 엔트로피 계산.

        Algorithm:
          - value를 [n_tokens, -1]로 reshape
          - softmax 후 Shannon 엔트로피: H_i = -Σ_j p_j * log(p_j + 1e-8)
          - 엔트로피가 낮을수록 집중적 어텐션 (고중요도)

        Returns:
            H: Tensor[n_tokens] — 토큰별 엔트로피
        """
        n_tokens = value.shape[0]
        flat = value.detach().float().reshape(n_tokens, -1)  # [n_tokens, D]
        # softmax로 확률 분포 근사
        p = torch.softmax(flat, dim=-1)  # [n_tokens, D]
        H = -(p * torch.log(p + 1e-8)).sum(dim=-1)  # [n_tokens]
        return H

    # ------------------------------------------------------------------ #
    # RL 인터페이스                                                         #
    # ------------------------------------------------------------------ #

    def set_rl_mode(self, is_rl: bool, num_completions: int = 1) -> None:
        """RL 워크로드 감지 플래그 설정."""
        self._is_rl_mode = is_rl or num_completions > 1

    def update_reward_signal(self, reward: float) -> None:
        """RL 리워드 피드백으로 정밀도 비율 동적 조정.

        Algorithm:
          - reward >= high_reward_threshold:
              precision_ratio_int4 += reward_aggression_step (더 공격적 압축 허용)
          - reward < high_reward_threshold:
              precision_ratio_int4 -= reward_recovery_step (정밀도 회복)
          - 조정 후 int4를 [0, 1-ratio_fp16] 범위로 클램핑
          - 남은 비율을 int8에 할당하여 합계 1.0 유지

        Args:
            reward: 최근 RL 생성의 리워드 스코어 (0.0~1.0)
        """
        self._last_reward = reward
        cfg = self.config

        if reward >= cfg.high_reward_threshold:
            self._ratio_int4 = min(
                1.0 - self._ratio_fp16,
                self._ratio_int4 + cfg.reward_aggression_step
            )
        else:
            self._ratio_int4 = max(0.0, self._ratio_int4 - cfg.reward_recovery_step)

        # 재정규화: fp16은 고정, int8 = 나머지
        self._ratio_int8 = max(0.0, 1.0 - self._ratio_fp16 - self._ratio_int4)

    def apply_online_quantization(
        self,
        kv_tensor: torch.Tensor,
        step_id: int,
        reward_signal: Optional[float] = None,
    ) -> torch.Tensor:
        """vLLM Q2 2026 온라인 양자화 플러그인 인터페이스.

        Args:
            kv_tensor: [n_tokens, ...] KV 텐서
            step_id: 현재 디코딩 스텝
            reward_signal: 선택적 리워드 신호 (있으면 update_reward_signal() 호출)

        Returns:
            quantized_kv: 압축된 FP16 텐서
        """
        if reward_signal is not None:
            self.update_reward_signal(reward_signal)
        self._current_step = step_id
        return self.compression_hook("__online__", kv_tensor)

    # ------------------------------------------------------------------ #
    # 메트릭                                                                #
    # ------------------------------------------------------------------ #

    def memory_reduction_ratio(self) -> float:
        """실제 메모리 절감률 (bytes 기준)."""
        if self._total_bytes_original == 0:
            return 0.0
        return 1.0 - self._total_bytes_stored / self._total_bytes_original

    def current_precision_ratios(self) -> Dict[str, float]:
        """현재 동적 정밀도 비율."""
        return {
            "fp16": self._ratio_fp16,
            "int8": self._ratio_int8,
            "int4": self._ratio_int4,
        }
```

---

### HMAChainedACPipeline (Cross A+C)

A-1 스케줄러 + C-1 코덱 + 기구현 코덱 3종을 통합하는 중앙 파이프라인.
InferenceRunner에 직접 주입 가능하도록 schedule() 인터페이스를 구현한다.

```python
# src/engine/hma_chained_ac_pipeline.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

from src.engine.runner import InferenceRequest, InferenceRunner
from src.cache.base import CacheStore
from src.scheduler.hma_multi_connector_scheduler import (
    HMAMultiConnectorCompressionPluginScheduler,
    HMAMultiConnectorConfig,
    HMAConnectorAdapter,
)
from src.cache.rl_adaptive_precision_quantizer import (
    RLAdaptivePrecisionQuantizer,
    RLAdaptivePrecisionConfig,
)
from src.cache.global_retention_gate_eviction import (
    GlobalRetentionGateEvictionCodec,
    GlobalRetentionGateConfig,
)


@dataclass
class HMAChainedACPipelineConfig:
    # 커넥터 선택 규칙 (YAML 외부화)
    chain_mode: bool = False               # True: 주 커넥터 + global_retention 순차 체이닝
    default_connector: str = "global_retention"
    long_ctx_threshold: int = 4096
    memory_pressure_threshold: float = 0.8

    # 내장 코덱 설정 (외부에서 코덱 인스턴스를 주입하지 않을 경우 기본값 사용)
    rl_quantizer_config: Optional[RLAdaptivePrecisionConfig] = None
    global_retention_config: Optional[GlobalRetentionGateConfig] = None

    seed: int = 42


class HMAChainedACPipeline:
    """HMA 멀티-커넥터 플러그인 체이닝 A+C 통합 파이프라인.

    Cross Activity A+C:
      - A-1 HMAMultiConnectorCompressionPluginScheduler 커넥터 레지스트리
      - C-1 RLAdaptivePrecisionQuantizer (RL 워크로드)
      - 기구현 GlobalRetentionGateEvictionCodec (긴 컨텍스트)
      - 기구현 RateQuantReverseWaterfillingCodec (짧은 고처리량, 선택적)

    요청 프로파일 기반 커넥터 선택 규칙 (YAML connector_dispatch_policy):
      is_rl_mode=True or num_completions>1  → "rl_adaptive"
      context_length > long_ctx_threshold   → "global_retention"
      memory_pressure > threshold           → "ratequant" (선택적)
      기본값                                → default_connector

    chain_mode=True:
      선택된 커넥터 → global_retention 순차 적용 (write-time → post-write 이중 필터)

    InferenceRunner 통합:
      runner = InferenceRunner(cache=pipeline.cache, scheduler=pipeline)
      runner.run_batch(requests) → pipeline.schedule(requests) 호출
    """

    def __init__(
        self,
        config: HMAChainedACPipelineConfig,
        rl_quantizer: Optional[RLAdaptivePrecisionQuantizer] = None,
        global_retention_codec: Optional[GlobalRetentionGateEvictionCodec] = None,
        extra_connectors: Optional[Dict[str, "HMAConnectorInterface"]] = None,
    ) -> None:
        self.config = config

        # C-1: RLAdaptivePrecisionQuantizer
        rl_cfg = config.rl_quantizer_config or RLAdaptivePrecisionConfig(seed=config.seed)
        self._rl_quantizer = rl_quantizer or RLAdaptivePrecisionQuantizer(rl_cfg)

        # 기구현 GlobalRetentionGateEvictionCodec
        gr_cfg = config.global_retention_config or GlobalRetentionGateConfig(seed=config.seed)
        self._global_retention = global_retention_codec or GlobalRetentionGateEvictionCodec(gr_cfg)

        # A-1: HMAMultiConnectorScheduler 초기화
        sched_cfg = HMAMultiConnectorConfig(
            long_ctx_threshold=config.long_ctx_threshold,
            memory_pressure_threshold=config.memory_pressure_threshold,
            default_connector=config.default_connector,
            pipeline_mode=config.chain_mode,
            seed=config.seed,
        )
        self._scheduler = HMAMultiConnectorCompressionPluginScheduler(sched_cfg)

        # 커넥터 등록
        self._scheduler.register_connector(
            "rl_adaptive",
            HMAConnectorAdapter("rl_adaptive", self._rl_quantizer),
        )
        self._scheduler.register_connector(
            "global_retention",
            HMAConnectorAdapter("global_retention", self._global_retention),
        )

        # 추가 커넥터 (선택적)
        for name, conn in (extra_connectors or {}).items():
            self._scheduler.register_connector(name, conn)

        # 기본 캐시: global_retention codec을 CacheStore로 사용
        self.cache: CacheStore = self._global_retention

    def schedule(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        """InferenceRunner.run_batch()에서 호출되는 스케줄링 진입점."""
        return self._scheduler.schedule(requests)

    def compress_kv(
        self,
        request: InferenceRequest,
        kv: torch.Tensor,
        request_meta: Optional[Dict] = None,
    ) -> torch.Tensor:
        """요청 프로파일에 따라 적절한 커넥터로 KV 압축."""
        compressed, _ = self._scheduler.apply_connector(request, kv, request_meta)
        return compressed

    def metrics_summary(self) -> Dict:
        """처리량·메모리·정확도 복합 효과 측정용 통합 메트릭."""
        return {
            "connector_selection_stats": self._scheduler.connector_selection_stats(),
            "scheduling_overhead_ms_p50": self._scheduler.scheduling_overhead_ms_p50(),
            "rl_quantizer_memory_reduction": self._rl_quantizer.memory_reduction_ratio(),
            "rl_quantizer_precision_ratios": self._rl_quantizer.current_precision_ratios(),
            "global_retention_hit_rate": self._global_retention.hit_rate(),
            "global_retention_memory_bytes": self._global_retention.memory_bytes(),
        }
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(RLAdaptivePrecisionQuantizer)를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-2 proxy (실 데이터셋 없을 경우 synthetic 토큰 시퀀스로 대체)
- **측정 방법**: `src/metrics/perplexity.py`의 함수 사용
  - `attention_output_relative_error(q, k_orig, v_orig, k_comp, v_comp)` < 0.01 (1%)
  - `k_comp`, `v_comp`: `compression_hook()` 적용 후 FP16 복원된 K, V
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)
- **정밀도 레벨별 측정**:
  - FP16 전체 (warmup_steps 기간 또는 ratio_fp16=1.0): error < 0.001
  - Mixed [0.2, 0.6, 0.2]: error < 0.01 (기준값)
  - 공격적 [0.2, 0.2, 0.6]: error < 0.02 (허용 오차 경고)

### 태스크 정확도 측정

- **벤치마크**: GSM8K / MATH-500 RL 워크로드 proxy
- **측정 방법**:
  - `attention_kl_divergence(q, k_orig, k_comp)` < 0.015 (MANDATORY)
  - `cosine_similarity_output(q, k_orig, v_orig, k_comp, v_comp)` >= 0.99 (MANDATORY)
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)

### RL 워크로드 시뮬레이션

- **시나리오**: 동일 프롬프트 10회 반복 생성 시뮬레이션
  1. 동일 token_ids로 10회 `compression_hook()` 호출
  2. 각 생성 후 `update_reward_signal(reward)` 호출 (reward 0.0~1.0 순차 변화)
  3. precision_ratio_int4 변화 곡선 측정 (리워드 피드백 수렴 확인)
  4. 각 회차 attention error 측정 → 최종 10회 평균 error < 0.01 (MANDATORY)
- **수렴 기준**: reward가 [0.9, 0.9, 0.9, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9] 패턴 주입 시
  precision_ratio_int4가 reward 하락 후 반드시 recovery_step만큼 감소하는지 확인

### GlobalRetentionGateEvictionCodec과 동일 설정 비교

- budget_ratio=0.3 (GlobalRetentionGate) vs precision_ratio [0.2, 0.6, 0.2] (RL Adaptive)
  동일 n_tokens=64, seed=42 설정에서:
  - attention error, KL divergence, cosine similarity, memory_reduction_ratio 비교
  - `test_rl_vs_global_retention_comparison` 테스트 케이스에 포함
  - RLAdaptivePrecisionQuantizer cosine >= GlobalRetentionGate cosine - 0.01 조건 명시

### 검증 테스트 파일

`tests/unit/test_rl_adaptive_precision_quantizer.py`

**테스트 케이스 목록**:

```
test_warmup_fp16_preserved:
    warmup_steps=5, step≤5 → 모든 토큰이 FP16 원본과 동일 (error=0)

test_entropy_based_precision_assignment:
    n_tokens=100, 엔트로피 낮은 상위 20개 토큰이 fp16_idx에 포함됨

test_int8_quantization_error_within_1pct:
    INT8 구간 토큰: attention error < 0.01 (±1% MANDATORY)

test_int4_simulation_error_within_2pct:
    INT4 구간 토큰: attention error < 0.02

test_mixed_precision_attention_error:
    전체 mixed [0.2, 0.6, 0.2] 설정: attention error < 0.01 (MANDATORY)

test_kl_divergence_mixed_precision:
    KL < 0.015 at precision_ratio [0.2, 0.6, 0.2] (MANDATORY)

test_cosine_similarity_mixed_precision:
    cosine >= 0.99 at precision_ratio [0.2, 0.6, 0.2] (MANDATORY)

test_reward_feedback_increases_int4_on_high_reward:
    reward=0.9 → precision_ratio_int4 증가 확인

test_reward_feedback_decreases_int4_on_low_reward:
    reward=0.2 → precision_ratio_int4 감소 확인

test_reward_feedback_ratios_sum_to_one:
    update_reward_signal() 후 fp16 + int8 + int4 == 1.0 (±1e-6)

test_rl_simulation_10rounds_convergence:
    10회 반복 생성 시뮬레이션: 최종 average error < 0.01 (MANDATORY)

test_rl_simulation_reward_curve:
    리워드 피드백 수렴 곡선 측정: precision_ratio_int4 변화가 reward 방향과 일치

test_cachestore_interface:
    put/get/evict/hit_rate/memory_bytes/reset_stats 동작 확인

test_memory_reduction_gt_30pct:
    mixed precision 설정에서 memory_reduction_ratio() >= 0.30

test_rl_vs_global_retention_comparison:
    동일 설정(n_tokens=64, seed=42)에서 RLAdaptive vs GlobalRetention 비교
    RLAdaptive cosine >= GlobalRetention cosine - 0.01

test_apply_online_quantization_interface:
    apply_online_quantization(kv, step_id=15, reward_signal=0.85) 호출 정상 동작

test_cachestore_compression_hook_integration:
    put() 내부에서 compression_hook() 호출됨 + 저장 shape 확인
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-17-hma-rl-ac.yaml
experiment:
  date: "2026-05-17"
  activity: "A+C"
  description: >
    A-1 HMAMultiConnectorCompressionPluginScheduler (HMA 멀티-커넥터 플러그인 메타-스케줄러) +
    C-1 RLAdaptivePrecisionQuantizer (RL 워크로드 온라인 적응 정밀도 KV 양자화기) +
    Cross-1 HMAChainedACPipeline (A+C 통합 파이프라인)
  cache_type: rl_adaptive_precision
  compression_method: quantization
  scheduler_type: hma_multi_connector

hma_multi_connector_scheduler:
  long_ctx_threshold: 4096           # 이 이상이면 global_retention 커넥터 선택
  memory_pressure_threshold: 0.8    # HBM 사용률 이 이상이면 ratequant 선택
  default_connector: "global_retention"
  pipeline_mode: false               # true: 체이닝 모드
  max_wait_ratio: 2.0
  seed: 42

  # 커넥터 선택 정책 (connector_dispatch_policy)
  connector_dispatch_policy:
    rl_mode_connector: "rl_adaptive"         # is_rl_mode=True or num_completions>1
    long_ctx_connector: "global_retention"   # context_length > long_ctx_threshold
    high_pressure_connector: "ratequant"     # memory_pressure > threshold
    default: "global_retention"

rl_adaptive_precision_quantizer:
  precision_ratio_fp16: 0.20        # 상위 20%: FP16 (저엔트로피, 고중요도)
  precision_ratio_int8: 0.60        # 중간 60%: INT8
  precision_ratio_int4: 0.20        # 하위 20%: INT4 시뮬레이션
  warmup_steps: 10                  # 초기 N 스텝 FP16 전체 정밀도
  cot_length_threshold: 512
  high_reward_threshold: 0.8
  reward_aggression_step: 0.05
  reward_recovery_step: 0.05
  max_entries: 1000
  seed: 42
  # 정밀도 비율 sweep (정확도 곡선 측정)
  precision_sweep:
    - [0.20, 0.60, 0.20]   # 기본 설정 (MANDATORY 검증 기준)
    - [0.30, 0.50, 0.20]   # 보수적
    - [0.20, 0.20, 0.60]   # 공격적
    - [1.00, 0.00, 0.00]   # FP16 전체 (베이스라인)

hma_chained_ac_pipeline:
  chain_mode: false                  # true: 선택 커넥터 + global_retention 체이닝
  default_connector: "global_retention"
  long_ctx_threshold: 4096
  memory_pressure_threshold: 0.8
  seed: 42

# 기구현 GlobalRetentionGateEvictionCodec 비교용 설정
global_retention_gate_eviction:
  n_layers: 4
  n_heads: 4
  d_model: 256
  budget_ratio: 0.3
  recent_window: 32
  max_entries: 1000
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    proxy_tolerance: 0.01            # 1% attention output error limit (MANDATORY)
    kl_tolerance: 0.015
    cosine_min: 0.99
    perplexity_dataset: "wikitext-2"
    perplexity_tolerance_pct: 1.0
    task_accuracy_tolerance_pct: 1.0
  rl_simulation:
    n_rounds: 10                     # 동일 프롬프트 반복 생성 횟수
    reward_sequence: [0.9, 0.9, 0.9, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9]
    convergence_error_threshold: 0.01
  memory_reduction:
    target_ratio: 0.30               # 최소 −30%
    target_ratio_goal: 0.50          # 목표 −50% (mixed precision)
  throughput:
    target_improvement_pct: 20       # +20% 이상
  effective_context:
    target_multiplier: 2.0           # 2× 이상
  scheduling:
    ttft_overhead_limit_pct: 5.0     # TTFT p50 +5% 이내
    connector_selection_overhead_ms: 0.1  # 커넥터 선택 < 0.1ms/요청
  comparison:
    methods: ["rl_adaptive", "global_retention", "baseline_fp16"]
    # rl_adaptive cosine >= global_retention cosine - 0.01
    cosine_tolerance: 0.01
  cross_ac_comparison:
    methods: ["solo_a1", "solo_c1", "cross_combined", "prior_cross_nath_retention"]
    throughput_min_improvement_vs_solo: 5.0  # +5% 이상 (§5)
    memory_min_improvement_vs_solo: 10.0     # −10% 이상 (§5)

seed: 42
results_dir: "results/2026-05-17"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_rl_adaptive_precision_quantizer.py` — Activity C 필수 accuracy 검증 (17개 테스트, 위 목록 참조)
- [ ] `tests/unit/test_hma_multi_connector_scheduler.py` — Activity A HMA 멀티-커넥터 스케줄러 단위 테스트
- [ ] `tests/unit/test_hma_chained_ac_pipeline.py` — Cross A+C 통합 파이프라인 단위 테스트
- [ ] `tests/integration/test_cross_ac_hma_chained.py` — E2E 통합: 다중 요청 커넥터 선택 + RL 적응 압축 흐름

### 단위 테스트 최소 요구 사항 (test_hma_multi_connector_scheduler.py)

```
test_register_connector:
    register_connector() 후 레지스트리에 커넥터 등록됨 확인

test_select_connector_rl_mode:
    is_rl_mode=True → "rl_adaptive" 선택 확인

test_select_connector_long_context:
    context_length > 4096, is_rl_mode=False → "global_retention" 선택 확인

test_select_connector_high_pressure:
    context_length ≤ 4096, memory_pressure=0.9 → "ratequant" 선택 확인 (ratequant 등록 시)

test_select_connector_default_fallback:
    레지스트리에 없는 커넥터 선택 시 default_connector로 폴백

test_apply_connector_calls_compress:
    apply_connector() 호출 시 선택된 커넥터의 compress() 호출됨

test_pipeline_mode_chains_global_retention:
    pipeline_mode=True, connector_name != "global_retention" 시
    global_retention.compress()가 추가로 호출됨

test_scheduling_overhead_below_01ms:
    select_connector() 오버헤드 < 0.1ms (딕셔너리 룩업 O(1) 검증)

test_connector_selection_stats:
    connector_selection_stats()가 커넥터별 선택 횟수를 정확히 반환

test_schedule_returns_all_requests:
    schedule(requests) 반환 목록이 입력과 동일한 길이

test_reset_stats_clears_all:
    reset_stats() 후 모든 카운터 0
```

### 단위 테스트 최소 요구 사항 (test_hma_chained_ac_pipeline.py)

```
test_pipeline_init_registers_rl_adaptive:
    초기화 시 "rl_adaptive" 커넥터가 레지스트리에 등록됨

test_pipeline_init_registers_global_retention:
    초기화 시 "global_retention" 커넥터가 레지스트리에 등록됨

test_compress_kv_rl_request:
    is_rl_mode=True 요청에 rl_quantizer.compression_hook()이 적용됨

test_compress_kv_long_ctx_request:
    context_length > 4096 요청에 global_retention.compression_hook()이 적용됨

test_metrics_summary_keys:
    metrics_summary()가 connector_selection_stats, rl_quantizer_memory_reduction 등 키 포함

test_cross_ac_throughput_vs_solo_a1:
    단독 A-1 대비 Cross-1 처리량 +5% 이상 (evaluation_criteria.md §5)

test_cross_ac_memory_vs_solo_c1:
    단독 C-1 대비 Cross-1 메모리 −10% 이상 (evaluation_criteria.md §5)

test_cross_ac_accuracy_preserved:
    Cross-1 적용 후 cosine >= 0.99 (evaluation_criteria.md §5 C 포함 필수)

test_schedule_delegates_to_scheduler:
    pipeline.schedule() 호출 시 _scheduler.schedule() 위임

test_chain_mode_true:
    chain_mode=True 시 pipeline_mode=True로 HMAMultiConnectorScheduler 초기화됨
```

---

## vLLM 이식 경로 (vllm-porter 참조용)

### 대상 파일 구조

```
vllm_integration/
├── scheduler_patch.py          # HMAMultiConnectorSchedulerMixin 추가 (기존 파일 확장)
├── attention_backend_patch.py  # RLAdaptivePrecisionAttentionHook 추가 (기존 파일 확장)
└── hma_connector_adapter.py    # HMAConnectorAdapter vLLM v0.21.0 공식 API 연동 (신규)
```

### Activity A 통합 포인트: `vllm/core/scheduler.py`

```python
# vllm_integration/scheduler_patch.py 추가 사항

class HMAMultiConnectorSchedulerMixin:
    """vLLM v0.21.0 Scheduler에 HMA 멀티-커넥터 플러그인 기능을 추가하는 믹스인.

    vLLM 통합 포인트:
      - vllm.v1.core.sched.scheduler.Scheduler (vLLM v0.21.0 v1 아키텍처)
      - schedule() 메서드를 오버라이드해 _hma_pre_schedule() 훅 삽입
      - HMA OffloadingConnector 레지스트리를 vLLM kv_cache_manager에 연결

    make_hma_multi_connector_scheduler_class() 팩토리:
      HMAMultiConnectorSchedulerMixin + vLLM Scheduler 조합 클래스를 동적 생성.
      pip install --upgrade vllm 최신 버전과 호환.
    """

    def _hma_pre_schedule(self, waiting_requests) -> None:
        """schedule() 전 각 대기 요청의 커넥터 선택을 미리 결정."""
        ...

def make_hma_multi_connector_scheduler_class(
    vllm_scheduler_cls,
    hma_config: "HMAMultiConnectorConfig",
) -> type:
    """vLLM Scheduler + HMAMultiConnectorSchedulerMixin 동적 조합 팩토리."""
    ...
```

### Activity C 통합 포인트: attention backend write/read hooks

```python
# vllm_integration/attention_backend_patch.py 추가 사항

class RLAdaptivePrecisionAttentionHook:
    """FlashAttentionImpl.forward()에 RLAdaptivePrecisionQuantizer를 주입하는 훅.

    vLLM 통합 포인트:
      write_to_cache hook: reshape_and_cache_flash() 호출 전
        - compression_hook()으로 FP16/INT8/INT4 혼합 압축 수행
        - 압축된 FP16 텐서를 캐시에 기록
      read_from_cache hook: 어텐션 커널 실행 전
        - 이미 FP16으로 복원된 압축 KV 반환 (양자화 텐서 직접 노출 없음)

    Accuracy contract:
      - mixed [0.2, 0.6, 0.2]: attention error < 1% (MANDATORY §4)
      - warmup_steps 기간: FP16 전체 정밀도 (zero error)
      - 리워드 피드백: 외부 RL 환경에서 apply_online_quantization(reward_signal=r) 주입

    apply_rl_adaptive_precision_patch(flash_attn_impl, rl_config):
      FlashAttentionImpl를 monkey-patch해 write/read 훅 삽입.
    """

    def write_to_cache(self, key_cache, value_cache, layer_idx: int) -> None:
        ...

    def read_from_cache(self, key_cache, value_cache, layer_idx: int):
        ...


# vllm_integration/hma_connector_adapter.py (신규)
class HMAConnectorAdapterForVLLM:
    """vLLM v0.21.0 공식 HMA OffloadingConnector 인터페이스와 연동하는 어댑터.

    vLLM v0.21.0 HMA API:
      - OffloadingConnector.store(job_id, kv_tensor) → DCP 이벤트
      - OffloadingConnector.prefetch(job_id) → PCP 이벤트
      - MultiConnectorManager.register(name, connector)
      - MultiConnectorManager.select(request_profile) → connector_name

    이 어댑터는 src/scheduler/hma_multi_connector_scheduler.py의
    HMAConnectorInterface를 vLLM의 공식 OffloadingConnector API로 래핑한다.
    실제 vLLM v0.21.0 import가 없을 경우 독립 구현으로 폴백.
    """
    ...
```

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 3개 파일 + 기존 회귀 없음)
- [ ] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - perplexity 변화 ±1% 이내 (attention error < 0.01, mixed [0.2, 0.6, 0.2] 설정)
      - downstream 태스크 정확도 ±1% 이내 (KL < 0.015, cosine >= 0.99)
      - RL 워크로드 시뮬레이션 10회 반복 후 리워드 피드백 수렴 확인
      - GlobalRetentionGateEvictionCodec과 동일 설정 2방향 비교 포함
- [ ] `evaluation_criteria.md` §2 Activity A 항목 충족:
      - 스케줄링 오버헤드 TTFT p50 +5% 이내
      - 커넥터 선택 오버헤드 < 0.1ms/요청 (O(1) 딕셔너리 룩업 검증)
      - 캐시 히트율 향상 +10%p
- [ ] `evaluation_criteria.md` §5 크로스 조합 C 포함:
      - 복합 적용 후 accuracy ±1% 이내
      - 단독 A-1 / 단독 C-1 / 결합 Cross-1 / 기구현 Cross-1 4방향 비교 수치 확인
      - 단독 Activity 대비 +5% 처리량, −10% 메모리 추가 개선
- [ ] `evaluation_criteria.md` §0 공통 필수:
      - CacheStore 인터페이스 모든 추상 메서드 구현 (RLAdaptivePrecisionQuantizer)
      - 시드 42 고정 재현성
      - `configs/experiments/2026-05-17-hma-rl-ac.yaml` 존재
      - 모든 공개 함수·메서드 타입 힌트
- [ ] 목표 지표 수치 `results/2026-05-17/metrics.json`에 JSON 기록:
      ```json
      {
        "inference_throughput_improvement_pct": ...,
        "kv_memory_reduction_ratio_mixed_precision": ...,
        "compression_accuracy_delta_mixed_precision": ...,
        "effective_context_length_multiplier": ...,
        "scheduling_overhead_ttft_p50_pct": ...,
        "connector_selection_overhead_ms_p50": ...,
        "rl_simulation_10round_avg_error": ...,
        "rl_simulation_reward_convergence_step": ...,
        "rl_adaptive_cosine_mixed": ...,
        "global_retention_cosine_budget30": ...,
        "cross_ac_throughput_vs_solo_a1_pct": ...,
        "cross_ac_memory_vs_solo_c1_pct": ...,
        "cross_ac_accuracy_cosine": ...,
        "connector_selection_stats": {
          "rl_adaptive": ...,
          "global_retention": ...,
          "ratequant": ...
        }
      }
      ```
- [ ] `src/cache/base.py` CacheStore 인터페이스 깨지지 않음 (수정 없음)
- [ ] 기존 모든 단위·통합 테스트 회귀 없이 통과
