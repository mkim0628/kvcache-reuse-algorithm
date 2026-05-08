<!-- 변경 이유 (이전 Spec.md: 2026-05-06 대비):
이전 사이클(2026-05-06)은 B+C (QueryCentricRecomputeCache + TriAttentionCodec + DualFilterSegmentSelector)
조합이었다. 이번 사이클은 A+C (PreemptiveKVOffloadScheduler + eOptShrinkQCodec + CompressedPreemptionPipeline)
조합으로 전환하며, 보조로 Activity B(StaticDynamicSegmentCache)와 Activity C(ManifoldKVWindowedEviction)를 추가한다.

주요 변경:
1. [Activity A 신규 추가] 이전 두 사이클이 A를 미포함하고 B+C에 집중했으나, 이번 사이클은
   PreemptiveKVOffloadScheduler(TokenFlow EuroSys 2026 기반)를 도입하여 선점형 요청 스케줄링 +
   능동적 GPU→CPU KV 전송 오버랩이라는 새로운 Activity A 설계 차원을 추가한다.

2. [Activity C 교체] TriAttentionCodec(pre-RoPE 삼각함수 중요도 프루닝) →
   eOptShrinkQCodec(BBP 위상전이 기반 자동 저랭크 공유 성분 추출 + TurboQuantCodec 잔차 양자화).
   이전 사이클의 TurboQuantCodec(기구현)을 잔차 백엔드로 재활용하며, BBP 랜덤 행렬 이론으로
   랭크를 자동 선택하는 완전히 새로운 이론적 기반을 도입한다.

3. [Cross-1 교체] QueryCentricTriAttentionCache(B+C) →
   CompressedPreemptionPipeline(A+C): 선점 결정 시 eOptShrinkQCodec으로 KV를 인라인 압축 후
   CUDA 이중 스트림(compute + memory)으로 PCIe 전송을 오버랩한다.

4. [Activity B 교체] QueryCentricRecomputeCache + InfoFlowChunkReorderCache →
   StaticDynamicSegmentCache(KEEP arXiv 2602.23592 기반): 에이전트 메모리 갱신 패턴 기반
   정적/동적 세그먼트 분리 + Multi-hop 무효화 전파 최소화.

5. [Activity C 추가] ManifoldKVWindowedEviction 신규 추가:
   슬라이딩 윈도우 유클리드 아웃라이어 탐지 기반 퇴거 정책 (구현 난이도 low, drop-in 교체).

6. [보존 파일] 이전 사이클 파일
   (query_centric_recompute.py, info_flow_reorder.py, tri_attention_codec.py,
   qc_tri_store.py, dual_filter_selector.py, diff_aware_store.py, turbo_quant.py,
   dhd_segment_cache.py, speculative_fetcher.py, sign_vq_segment.py,
   leverage_compressor.py, compression.py, segmented.py, contiguous.py,
   tri_state_compressor.py, compressed_segment.py, segment_adapter.py,
   dag_topology_scheduler.py, workload_ttl_cache.py, redundancy_eviction.py,
   fireq_codec.py, nqkv_codec.py, compressed_diff_store.py)은
   이번 사이클에서 수정하지 않는다. 기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-08

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-08.md`
**최우선 구현 타겟**: Cross-1 (A+C) — PreemptiveKVOffloadScheduler(A-1) + eOptShrinkQCodec(C-1)
+ CompressedPreemptionPipeline 통합, 보조로 StaticDynamicSegmentCache(B-1),
ManifoldKVWindowedEviction(C-4)

**해결하려는 문제**:
- 표준 비선점형(non-preemptive) 요청 스케줄링은 요청 폭주(burst) 시 Head-of-Line Blocking이 발생해
  P99 TTFT가 급등한다. 실시간 토큰 버퍼 점유율과 소비율을 기준으로 요청을 선점하고
  GPU→CPU KV 전송을 백그라운드에서 능동적으로 실행하면(TokenFlow EuroSys 2026),
  P99 TTFT를 최대 80% 감소시킬 수 있다.
- KV 캐시 선점 시 GPU→CPU 전송 비용이 선점 오버헤드를 키우는 문제를, eOptShrinkQCodec(BBP
  랜덤 행렬 이론 자동 저랭크 + TurboQuantCodec 잔차)으로 전송 전 KV를 2.2비트로 압축하면
  PCIe 대역폭 요구를 1/7로 줄여 선점 오버헤드가 추가로 30~40% 감소한다.
- 에이전트 메모리 갱신 시 갱신 세그먼트 이후의 KV 전체 무효화로 대규모 재계산이 발생하는 문제를,
  세그먼트를 정적/동적으로 분리하고 Multi-hop 무효화 전파 깊이를 제한하면(StaticDynamicSegmentCache,
  KEEP arXiv 2602.23592 기반) 재계산 비용을 최소화할 수 있다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling — PreemptiveKVOffloadScheduler (선점형 요청 스케줄링 + 비동기 KV 오프로드)
- [x] Activity B: Non-Contiguous KV Cache Reuse — StaticDynamicSegmentCache (정적/동적 세그먼트 분리)
- [x] Activity C: KV Cache Compression — eOptShrinkQCodec (BBP 자동 저랭크 + TurboQuant 잔차) + ManifoldKVWindowedEviction (유클리드 아웃라이어 퇴거)

---

## 목표

- [ ] 목표 1 (§1 Throughput): tokens/sec 베이스라인 대비 +20% 이상 — 선점형 스케줄링으로 배치 활용률 향상 + 압축으로 배치 슬롯 증가 (evaluation_criteria.md §1)
- [ ] 목표 2 (§2 Activity A): TTFT p50 증가 +5% 이내 (정상 부하 시) (evaluation_criteria.md §2 필수)
- [ ] 목표 3 (§2 Activity A): TTFT p99 베이스라인 대비 −60% 이상 (요청 폭주 시, TokenFlow 실증 기준) (evaluation_criteria.md §2)
- [ ] 목표 4 (§4 KV Memory Reduction): 베이스라인 대비 −30% 이상 — eOptShrinkQCodec 2.2비트 압축 기여 (evaluation_criteria.md §4)
- [ ] 목표 5 (§4 Accuracy 필수): perplexity 변화 ±1% 이내 — eOptShrinkQCodec BBP 이론 보장 (evaluation_criteria.md §4 필수)
- [ ] 목표 6 (§4 Accuracy 필수): downstream 태스크 정확도 변화 ±1% 이내 (evaluation_criteria.md §4 필수)
- [ ] 목표 7 (§3 Non-Contiguous Hit Rate): 전체 히트 중 비연속 히트 비율 ≥ 30% — StaticDynamicSegmentCache 정적 세그먼트 즉시 재사용 (evaluation_criteria.md §3)
- [ ] 목표 8 (§5 Cross A+C): 복합 처리량 향상 단일 Activity 대비 추가 +5% 이상 (evaluation_criteria.md §5)
- [ ] 목표 9 (§5 Cross A+C): 복합 메모리 감소 단일 Activity 대비 추가 −10% 이상 (evaluation_criteria.md §5)
- [ ] 목표 10 (§4 Compression Overhead): eOptShrinkQCodec Encode/Decode 추가 지연 TTFT +10% 이내 (evaluation_criteria.md §4)

---

## 아키텍처 개요

```
요청 도착
    │
    ├─ PreemptiveKVOffloadScheduler (Activity A)
    │       │
    │       ├─ [선점 트리거] buffer_occupancy_ratio > 0.85 AND token_consumption_rate < demand_rate
    │       ├─ [선점 결정] 저우선순위 요청 KV를 선점 큐에 등록
    │       └─ [비동기 GPU→CPU 전송] 배치 버블 시점에 KV 전송 시작
    │
    ├─ CompressedPreemptionPipeline (Cross-1: A+C)
    │       │
    │       ├─ [인라인 압축] eOptShrinkQCodec.encode(kv_to_offload) → 2.2비트 압축 KV
    │       ├─ [CUDA 이중 스트림] compute_stream(압축) + memory_stream(PCIe 전송) 오버랩
    │       └─ [재개 시 복원] CPU→GPU 전송 후 eOptShrinkQCodec.decode(compressed_kv)
    │
    ├─ StaticDynamicSegmentCache (Activity B)
    │       │
    │       ├─ [정적 세그먼트] 시스템 프롬프트, 공통 문서 → LRU 퇴거 제외, 완전 재사용
    │       └─ [동적 세그먼트] 에이전트 행동 결과, 대화 히스토리 → Multi-hop 무효화 범위 제한
    │
    └─ eOptShrinkQCodec (Activity C)
            │
            ├─ [SVD + BBP 위상전이] 자동 랭크 선택 → 저랭크 공유 성분 추출
            ├─ [TurboQuantCodec 잔차] Key 2비트 / Value 3비트 비대칭 양자화
            └─ ManifoldKVWindowedEviction: 슬라이딩 윈도우 유클리드 아웃라이어 퇴거
```

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/scheduler/preemptive_kv_offload.py` | A | 선점형 요청 스케줄링 + 비동기 GPU→CPU KV 전송 |
| `src/scheduler/compressed_preemption.py` | A+C | CompressedPreemptionPipeline: 선점 시 eOptShrinkQCodec 인라인 압축 + CUDA 이중 스트림 |
| `src/cache/eopt_shrinkq_codec.py` | C | BBP 위상전이 자동 저랭크 + TurboQuantCodec 잔차 이중 파이프라인 |
| `src/cache/static_dynamic_segment.py` | B | 정적/동적 세그먼트 분리 + Multi-hop 무효화 전파 최소화 |
| `src/cache/manifoldkv_windowed.py` | C | 슬라이딩 윈도우 유클리드 아웃라이어 KV 퇴거 정책 |
| `experiments/run_preemptive_ttft.py` | A | 선점형 vs 비선점형 TTFT p50/p99 비교 측정 스크립트 |
| `experiments/run_eopt_accuracy.py` | C | eOptShrinkQCodec perplexity + LongBench + NIAH 정확도 측정 |
| `tests/unit/test_preemptive_scheduler.py` | A | 선점 트리거 + KV 전송 비동기 단위 테스트 |
| `tests/unit/test_eopt_shrinkq_accuracy.py` | C | BBP 랭크 선택 + accuracy-preserving 단위 테스트 |
| `tests/unit/test_static_dynamic_segment.py` | B | 정적/동적 분류 + Multi-hop 무효화 전파 단위 테스트 |
| `tests/unit/test_manifoldkv_eviction.py` | C | 유클리드 아웃라이어 스코어 + 퇴거 정책 단위 테스트 |
| `tests/integration/test_cross_ac_preempt_compress.py` | A+C | CompressedPreemptionPipeline 통합 E2E 테스트 |
| `configs/experiments/2026-05-08.yaml` | 공통 | 실험 설정 파일 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/scheduler/__init__.py` | `PreemptiveKVOffloadScheduler`, `CompressedPreemptionPipeline` export 추가 |
| `src/cache/__init__.py` | `eOptShrinkQCodec`, `StaticDynamicSegmentCache`, `ManifoldKVWindowedEviction` export 추가 |

### 수정 금지 파일 (이전 사이클 보존)

`query_centric_recompute.py`, `info_flow_reorder.py`, `tri_attention_codec.py`,
`qc_tri_store.py`, `dual_filter_selector.py`, `diff_aware_store.py`, `turbo_quant.py`,
`dhd_segment_cache.py`, `speculative_fetcher.py`, `sign_vq_segment.py`,
`leverage_compressor.py`, `compression.py`, `segmented.py`, `contiguous.py`,
`tri_state_compressor.py`, `compressed_segment.py`, `segment_adapter.py`,
`dag_topology_scheduler.py`, `workload_ttl_cache.py`, `redundancy_eviction.py`,
`fireq_codec.py`, `nqkv_codec.py`, `compressed_diff_store.py`,
`cache_aware_scheduler.py`, `multi_node_scheduler.py`, `dag_ttl_adjuster.py`,
`dual_map_scheduler.py`

---

## 알고리즘 상세

### 1. PreemptiveKVOffloadScheduler (Activity A)

**스케줄링 결정 단위**: 배치(batch) 단위로 선점 결정. 각 배치 시작 시 버퍼 상태를 확인하고,
선점 조건이 충족되면 배치 내 저우선순위 요청을 선점 큐로 이동시킨다.

**캐시 상태 접근 방법**: `cache.memory_bytes()` 및 `cache_capacity_bytes` 파라미터로
`buffer_occupancy_ratio`를 계산. `token_consumption_rate`는 최근 `consumption_rate_window`(기본 32)
토큰 처리 시간의 이동 평균으로 추정. 캐시를 직접 수정하지 않고 읽기 전용으로 상태를 조회한다.

```python
# src/scheduler/preemptive_kv_offload.py 의사코드

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import torch
from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


@dataclass
class PreemptionRecord:
    request_id: str
    offloaded_kv: Optional[torch.Tensor]  # CPU 메모리에 보관된 KV (압축 전 원본 또는 압축 후)
    offload_bytes: int
    is_compressed: bool = False


class PreemptiveKVOffloadScheduler:
    """
    TokenFlow(EuroSys 2026) 원리 기반 선점형 요청 스케줄링.

    선점 결정 단위: 배치(batch) 단위.
    캐시 상태 접근: cache.memory_bytes() + cache_capacity_bytes (읽기 전용).

    이전 Activity A 기법(CacheAwareScheduler, DAGTopologyScheduler 등)과의 차별점:
    - 기존: 요청 우선순위 재정렬(cache hit rate 기반).
    - 신규: 실행 중인 요청을 중단(선점)하고 KV를 CPU로 능동 전송.
      "요청을 멈추는 시점"과 "KV를 옮기는 타이밍"의 실시간 공동 결정.
    """

    def __init__(
        self,
        cache: CacheStore,
        cache_capacity_bytes: int,
        threshold_preempt: float = 0.85,       # 버퍼 점유율 임계값
        consumption_rate_window: int = 32,      # 소비율 추정 이동 평균 윈도우 (토큰 수)
        fairness_max_wait: int = 10,            # 선점 면제 최대 대기 스텝
        preempt_compress: bool = False,         # True이면 CompressedPreemptionPipeline 연동
        sla_tier_a_ids: Optional[List[str]] = None,  # 선점 제외 SLA Tier-A 요청 ID 목록
    ) -> None:
        self.cache = cache
        self.cache_capacity_bytes = cache_capacity_bytes
        self.threshold_preempt = threshold_preempt
        self.consumption_rate_window = consumption_rate_window
        self.fairness_max_wait = fairness_max_wait
        self.preempt_compress = preempt_compress
        self.sla_tier_a_ids: set = set(sla_tier_a_ids or [])

        self._wait_steps: Dict[str, int] = {}
        self._preempted: Dict[str, PreemptionRecord] = {}  # 선점된 요청 KV 보관
        self._token_history: List[Tuple[float, int]] = []  # (timestamp, token_count) 이동 평균용

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """
        배치 단위 선점 결정 + 요청 재정렬.

        1. buffer_occupancy_ratio 계산
        2. 선점 조건 충족 시 저우선순위 요청 선점 큐 이동
        3. 이미 선점된 요청 중 재개 가능(CPU KV 복원 비용 < 재계산 비용)한 요청 복원
        4. 나머지 활성 요청을 우선순위 순으로 반환
        """
        buffer_occupancy = self._buffer_occupancy_ratio()
        demand_rate = self._estimate_demand_rate(requests)
        consumption_rate = self._estimate_consumption_rate()

        active_requests: List[InferenceRequest] = []
        for req in requests:
            if req.request_id not in self._wait_steps:
                self._wait_steps[req.request_id] = 0

            # SLA Tier-A 요청은 선점 불가
            if req.request_id in self.sla_tier_a_ids:
                active_requests.append(req)
                continue

            # 선점 조건: 버퍼 점유율 초과 + 소비율 < 수요율
            if (buffer_occupancy > self.threshold_preempt
                    and consumption_rate < demand_rate
                    and self._wait_steps[req.request_id] < self.fairness_max_wait):
                # 저우선순위 요청 선점 (KV 오프로드는 비동기 처리)
                self._preempt_request(req)
            else:
                active_requests.append(req)

        # 선점된 요청 중 재개 가능한 요청 복원
        resumed = self._try_resume_preempted()
        active_requests.extend(resumed)

        # 대기 스텝 갱신
        active_ids = {r.request_id for r in active_requests}
        for req in requests:
            if req.request_id not in active_ids:
                self._wait_steps[req.request_id] = self._wait_steps.get(req.request_id, 0) + 1

        return active_requests

    def offload_kv_async(
        self,
        request_id: str,
        kv_tensor: torch.Tensor,              # GPU KV 텐서
        encode_fn: Optional[Callable] = None,  # eOptShrinkQCodec.encode (선택적)
    ) -> None:
        """
        GPU→CPU 비동기 KV 전송 (배치 버블 시점에 호출).

        encode_fn이 제공되면 GPU에서 압축 후 전송 (CompressedPreemptionPipeline 연동).
        asyncio.Queue 패턴으로 백그라운드 전송 큐에 적재.

        vLLM v0.9 통합 경로:
            # vLLM 비동기 KV Connector API를 encode_fn 대신 전송 백엔드로 사용 가능:
            # from vllm.distributed.kv_transfer.kv_connector.base import BaseKVConnector
            # connector.insert(kv_tensor, ...)  ← 이 인터페이스로 대체
        """
        if encode_fn is not None:
            kv_compressed = encode_fn(kv_tensor)
            cpu_kv = kv_compressed.cpu()
            is_compressed = True
            offload_bytes = cpu_kv.nbytes if hasattr(cpu_kv, 'nbytes') else int(cpu_kv.numel() * 2)
        else:
            cpu_kv = kv_tensor.cpu()
            is_compressed = False
            offload_bytes = cpu_kv.nbytes

        self._preempted[request_id] = PreemptionRecord(
            request_id=request_id,
            offloaded_kv=cpu_kv,
            offload_bytes=offload_bytes,
            is_compressed=is_compressed,
        )

    def restore_kv(
        self,
        request_id: str,
        decode_fn: Optional[Callable] = None,  # eOptShrinkQCodec.decode (선택적)
    ) -> Optional[torch.Tensor]:
        """
        CPU→GPU KV 복원.
        복원 비용 > 재계산 비용 이면 None 반환 → 호출자가 재계산 선택.
        """
        record = self._preempted.get(request_id)
        if record is None or record.offloaded_kv is None:
            return None
        gpu_kv = record.offloaded_kv.cuda()
        if record.is_compressed and decode_fn is not None:
            gpu_kv = decode_fn(gpu_kv)
        del self._preempted[request_id]
        return gpu_kv

    def _buffer_occupancy_ratio(self) -> float:
        current = self.cache.memory_bytes()
        return current / max(self.cache_capacity_bytes, 1)

    def _estimate_demand_rate(self, requests: List[InferenceRequest]) -> float:
        total_tokens = sum(len(r.token_ids) for r in requests)
        return float(total_tokens) / max(len(requests), 1)

    def _estimate_consumption_rate(self) -> float:
        if len(self._token_history) < 2:
            return float('inf')
        import time as _time
        recent = self._token_history[-self.consumption_rate_window:]
        if len(recent) < 2:
            return float('inf')
        dt = recent[-1][0] - recent[0][0]
        tokens = sum(t for _, t in recent)
        return tokens / max(dt, 1e-6)

    def record_processed_tokens(self, token_count: int) -> None:
        """배치 처리 후 호출하여 소비율 이동 평균 갱신."""
        import time as _time
        self._token_history.append((_time.monotonic(), token_count))
        if len(self._token_history) > self.consumption_rate_window * 2:
            self._token_history = self._token_history[-self.consumption_rate_window:]

    def _preempt_request(self, req: InferenceRequest) -> None:
        """선점 등록 (실제 KV 전송은 offload_kv_async로 별도 호출)."""
        self._preempted.setdefault(req.request_id, PreemptionRecord(
            request_id=req.request_id,
            offloaded_kv=None,
            offload_bytes=0,
        ))

    def _try_resume_preempted(self) -> List[InferenceRequest]:
        """버퍼 여유가 생긴 경우 선점된 요청 재개."""
        if self._buffer_occupancy_ratio() < self.threshold_preempt * 0.8:
            resumed = []
            # 대기 스텝이 가장 긴 순으로 복원 (공정성)
            sorted_preempted = sorted(
                self._preempted.items(),
                key=lambda x: self._wait_steps.get(x[0], 0),
                reverse=True,
            )
            for rid, record in sorted_preempted[:3]:  # 한 번에 최대 3개 복원
                # KV가 오프로드된 경우에만 복원 요청 생성
                if record.offloaded_kv is not None:
                    from src.engine.runner import InferenceRequest as _IR
                    resumed.append(_IR(request_id=rid, token_ids=[]))
            return resumed
        return []
```

---

### 2. eOptShrinkQCodec (Activity C)

**BBP 위상전이 임계값**: `sigma_c = noise_level × (1 + sqrt(aspect_ratio))^2`
- `aspect_ratio = min(n_rows, n_cols) / max(n_rows, n_cols)` (KV 행렬의 행/열 비율)
- `noise_level`: 캘리브레이션 데이터에서 Marchenko-Pastur 분포 우변 추정
- 자동 랭크 `r` = `S > sigma_c`를 만족하는 특이값의 수

```python
# src/cache/eopt_shrinkq_codec.py 의사코드

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from src.cache.turbo_quant import TurboQuantCodec


class eOptShrinkQCodec:
    """
    BBP(Baik-Ben Arous-Péché) 위상전이 기반 자동 저랭크 공유 성분 추출 +
    TurboQuantCodec 잔차 양자화 이중 파이프라인.

    eOptShrinkQ(arXiv 2605.02905) 기반.
    CacheStore를 상속하지 않음 (순수 코덱 클래스).
    PreemptiveKVOffloadScheduler 및 CompressedPreemptionPipeline에서 encode/decode로 호출됨.

    메모리 구조:
    - 저랭크 베이스: U[:, :r] @ diag(S[:r]) @ V[:r, :] — float16 저장
    - 잔차 양자화: TurboQuantCodec(Key 2비트 / Value 3비트 비대칭)
    - 실효 압축: ~2.2비트/원소 (eOptShrinkQ 논문 실증: TurboQuant 3비트 대비 동등 성능)
    """

    def __init__(
        self,
        num_layers: int,
        key_bits: int = 2,           # Key 양자화 비트 (공격적)
        value_bits: int = 3,         # Value 양자화 비트 (보수적)
        calibration_samples: int = 20,  # 오프라인 캘리브레이션 최소 샘플 수
        base_seed: int = 42,
    ) -> None:
        self.num_layers = num_layers
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.calibration_samples = calibration_samples

        # TurboQuantCodec 재활용 (Key: key_bits, Value: value_bits)
        self._key_codec = TurboQuantCodec(
            num_layers=num_layers, bits=key_bits, base_seed=base_seed
        )
        self._val_codec = TurboQuantCodec(
            num_layers=num_layers, bits=value_bits, base_seed=base_seed + 1
        )

        # 레이어별 캘리브레이션 결과
        self._noise_levels: Dict[int, float] = {}  # {layer_idx: estimated_noise_level}
        self._auto_ranks: Dict[int, int] = {}       # {layer_idx: auto_selected_rank}

    # ------------------------------------------------------------------ #
    # 캘리브레이션                                                         #
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        calibration_kvs: List[torch.Tensor],
        # calibration_kvs[i]: [n_tokens, d_head] float32 (레이어 i)
        save_path: Optional[str] = None,
    ) -> None:
        """
        레이어별 noise_level 추정 및 자동 랭크 선택.
        오프라인 1회 실행 후 저장. calibration_samples ≥ 20 샘플 필요.
        """
        for layer_idx, kv in enumerate(calibration_kvs):
            if kv.numel() == 0:
                continue
            kv_f = kv.float()
            n, d = kv_f.shape
            aspect_ratio = min(n, d) / max(n, d)

            # 특이값 분해
            try:
                _, S, _ = torch.linalg.svd(kv_f, full_matrices=False)
            except RuntimeError:
                self._noise_levels[layer_idx] = 1.0
                self._auto_ranks[layer_idx] = min(8, min(n, d))
                continue

            # Marchenko-Pastur 우변 추정 (노이즈 특이값 중앙값 기반 간략 추정)
            # sigma_mp_max ≈ median(S) × sqrt(max(n, d)) / sqrt(min(n, d))
            noise_level = S.median().item() / math.sqrt(max(n, d))
            self._noise_levels[layer_idx] = noise_level

            # BBP 위상전이 임계값
            sigma_c = noise_level * (1.0 + math.sqrt(aspect_ratio)) ** 2

            # 자동 랭크: sigma_c 초과 특이값 수
            r = int((S > sigma_c).sum().item())
            r = max(1, min(r, min(n, d) // 2))  # 최소 1, 최대 min(n,d)/2
            self._auto_ranks[layer_idx] = r

        if save_path:
            torch.save(
                {"noise_levels": self._noise_levels, "auto_ranks": self._auto_ranks},
                save_path,
            )

    def load_calibration(self, load_path: str) -> None:
        """저장된 캘리브레이션 파일 로드."""
        ckpt = torch.load(load_path)
        self._noise_levels = ckpt["noise_levels"]
        self._auto_ranks = ckpt["auto_ranks"]

    # ------------------------------------------------------------------ #
    # 인코딩                                                               #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        kv_key: torch.Tensor,   # [n_tokens, d_head] — Key 텐서
        kv_val: torch.Tensor,   # [n_tokens, d_head] — Value 텐서
        layer_idx: int,
    ) -> Dict:
        """
        저랭크 성분 분리 + TurboQuantCodec 잔차 양자화.

        저장 구조:
          {
            "key_lowrank":  {"U": tensor, "S": tensor, "V": tensor},  # float16
            "val_lowrank":  {"U": tensor, "S": tensor, "V": tensor},  # float16
            "key_residual": TurboQuantCodec 압축 dict (key_bits 비트),
            "val_residual": TurboQuantCodec 압축 dict (value_bits 비트),
            "layer_idx": int,
            "n_tokens": int,
            "d_head": int,
          }
        """
        n_tokens, d_head = kv_key.shape
        r = self._auto_ranks.get(layer_idx, min(8, min(n_tokens, d_head) // 2))

        def _encode_single(kv: torch.Tensor, codec: TurboQuantCodec, tensor_id: int) -> Dict:
            kv_f = kv.float()
            try:
                U, S, Vh = torch.linalg.svd(kv_f, full_matrices=False)
            except RuntimeError:
                return {"lowrank": None, "residual": codec.encode(kv_f, layer_idx, tensor_id)}

            # 저랭크 베이스 (rank-r 근사)
            r_eff = min(r, S.shape[0])
            U_r = U[:, :r_eff]        # [n_tokens, r_eff]
            S_r = S[:r_eff]           # [r_eff]
            Vh_r = Vh[:r_eff, :]      # [r_eff, d_head]
            lowrank_approx = (U_r * S_r.unsqueeze(0)) @ Vh_r  # [n_tokens, d_head]

            # 잔차
            residual = kv_f - lowrank_approx   # [n_tokens, d_head]
            compressed_residual = codec.encode(residual, layer_idx, tensor_id)

            return {
                "lowrank": {
                    "U": U_r.half(),   # float16 저장
                    "S": S_r.half(),
                    "V": Vh_r.half(),
                },
                "residual": compressed_residual,
            }

        return {
            "key": _encode_single(kv_key, self._key_codec, tensor_id=0),
            "val": _encode_single(kv_val, self._val_codec, tensor_id=1),
            "layer_idx": layer_idx,
            "n_tokens": n_tokens,
            "d_head": d_head,
        }

    # ------------------------------------------------------------------ #
    # 디코딩                                                               #
    # ------------------------------------------------------------------ #

    def decode(self, compressed: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        저랭크 베이스 복원 + TurboQuantCodec 잔차 복원 → 원본 근사 반환.

        반환: (key_approx, val_approx) — 각각 [n_tokens, d_head] float32
        """
        layer_idx = compressed["layer_idx"]

        def _decode_single(comp: Dict, codec: TurboQuantCodec) -> torch.Tensor:
            residual_approx = codec.decode(comp["residual"], layer_idx)
            if comp["lowrank"] is None:
                return residual_approx
            U_r = comp["lowrank"]["U"].float()
            S_r = comp["lowrank"]["S"].float()
            Vh_r = comp["lowrank"]["V"].float()
            lowrank_approx = (U_r * S_r.unsqueeze(0)) @ Vh_r
            return lowrank_approx + residual_approx

        key_approx = _decode_single(compressed["key"], self._key_codec)
        val_approx = _decode_single(compressed["val"], self._val_codec)
        return key_approx, val_approx

    def memory_bytes_estimate(self, n_tokens: int, d_head: int, layer_idx: int = 0) -> Dict:
        """압축 후 예상 메모리 사용량 추정."""
        r = self._auto_ranks.get(layer_idx, 8)
        # 저랭크: float16 저장 (U_r + S_r + Vh_r)
        lowrank_bytes = (n_tokens * r + r + r * d_head) * 2  # float16 = 2 bytes
        # 잔차: TurboQuantCodec 추정
        key_res_bytes = self._key_codec.memory_bytes_estimate(n_tokens, d_head, layer_idx)["total_bytes"]
        val_res_bytes = self._val_codec.memory_bytes_estimate(n_tokens, d_head, layer_idx)["total_bytes"]
        total = lowrank_bytes * 2 + key_res_bytes + val_res_bytes  # Key + Value 저랭크
        baseline = n_tokens * d_head * 4 * 2  # FP32 K + V
        return {
            "total_bytes": total,
            "baseline_bytes": baseline,
            "reduction_ratio": 1.0 - total / max(baseline, 1),
        }
```

---

### 3. CompressedPreemptionPipeline (Activity A+C, Cross-1)

```python
# src/scheduler/compressed_preemption.py 의사코드

from typing import Callable, Dict, Optional
import torch
from src.scheduler.preemptive_kv_offload import PreemptiveKVOffloadScheduler
from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec
from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


class CompressedPreemptionPipeline:
    """
    Cross-1 (A+C): PreemptiveKVOffloadScheduler + eOptShrinkQCodec 통합.

    선점 결정 시 eOptShrinkQCodec 압축을 CUDA compute_stream에서,
    PCIe 전송을 memory_stream에서 동시 실행하여 오버랩 효율을 최대화한다.

    측정 지표:
    - overlap_efficiency: 압축-전송 오버랩으로 절약된 시간 / (압축 시간 + 전송 시간)
    - preemption_compress_ratio: 선점된 요청 중 압축 적용 비율
    - offload_bytes_before / offload_bytes_after: 압축 전/후 전송 크기
    """

    def __init__(
        self,
        scheduler: PreemptiveKVOffloadScheduler,
        codec: eOptShrinkQCodec,
        use_dual_stream: bool = True,         # CUDA 이중 스트림 오버랩 활성화
        sla_tier_a_no_compress: bool = True,  # SLA Tier-A 요청 선점 시 압축 미적용
    ) -> None:
        self.scheduler = scheduler
        self.codec = codec
        self.use_dual_stream = use_dual_stream
        self.sla_tier_a_no_compress = sla_tier_a_no_compress

        # CUDA 스트림 분리
        self._compute_stream: Optional[torch.cuda.Stream] = None
        self._memory_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self._compute_stream = torch.cuda.Stream()
            self._memory_stream = torch.cuda.Stream()

        # 메트릭
        self._overlap_efficiency_history: list = []
        self._total_bytes_before: int = 0
        self._total_bytes_after: int = 0

    def schedule(self, requests) -> list:
        """PreemptiveKVOffloadScheduler.schedule() 위임."""
        return self.scheduler.schedule(requests)

    def offload_with_compression(
        self,
        request_id: str,
        kv_key: torch.Tensor,   # GPU KV Key [n_tokens, d_head]
        kv_val: torch.Tensor,   # GPU KV Value [n_tokens, d_head]
        layer_idx: int,
    ) -> None:
        """
        CUDA 이중 스트림 압축-전송 파이프라인.

        compute_stream: eOptShrinkQCodec.encode(kv_key, kv_val, layer_idx)
        memory_stream:  CPU로 PCIe 전송 (압축 완료 이벤트 대기 후 실행)

        오버랩 효율 측정:
        - t_compress: encode() 실행 시간
        - t_transfer: .cpu() 전송 시간
        - overlap_efficiency = max(0, 1 - max(t_compress, t_transfer) / (t_compress + t_transfer))
        """
        import time
        bytes_before = kv_key.nbytes + kv_val.nbytes
        self._total_bytes_before += bytes_before

        if self.use_dual_stream and self._compute_stream is not None:
            # compute_stream에서 압축
            t0 = time.monotonic()
            with torch.cuda.stream(self._compute_stream):
                compressed = self.codec.encode(kv_key, kv_val, layer_idx)
            torch.cuda.synchronize()
            t_compress = time.monotonic() - t0

            # memory_stream에서 CPU 전송 (압축 완료 이벤트 사용)
            compress_event = torch.cuda.Event()
            compress_event.record(self._compute_stream)
            t1 = time.monotonic()
            with torch.cuda.stream(self._memory_stream):
                self._memory_stream.wait_event(compress_event)
                # 압축된 딕셔너리의 모든 텐서를 CPU로 이동
                compressed_cpu = _move_dict_to_cpu(compressed)
            torch.cuda.synchronize()
            t_transfer = time.monotonic() - t1

            # 오버랩 효율 기록
            total_seq = t_compress + t_transfer
            overlap_eff = max(0.0, 1.0 - max(t_compress, t_transfer) / max(total_seq, 1e-9))
            self._overlap_efficiency_history.append(overlap_eff)
        else:
            # 이중 스트림 비활성화 시 순차 처리
            compressed = self.codec.encode(kv_key, kv_val, layer_idx)
            compressed_cpu = _move_dict_to_cpu(compressed)

        # 압축 후 크기 측정
        bytes_after = sum(
            t.nbytes for t in _flatten_tensors(compressed_cpu)
        )
        self._total_bytes_after += bytes_after

        # PreemptiveKVOffloadScheduler에 압축된 KV 등록
        self.scheduler._preempted[request_id].offloaded_kv = compressed_cpu  # type: ignore
        self.scheduler._preempted[request_id].is_compressed = True
        self.scheduler._preempted[request_id].offload_bytes = bytes_after

    def restore_with_decompression(
        self,
        request_id: str,
        layer_idx: int,
    ) -> Optional[tuple]:
        """
        CPU→GPU 전송 후 eOptShrinkQCodec.decode() 복원.
        반환: (key_approx, val_approx) 또는 None (재계산 선택 시).
        """
        record = self.scheduler._preempted.get(request_id)
        if record is None or record.offloaded_kv is None:
            return None
        compressed_gpu = _move_dict_to_gpu(record.offloaded_kv)
        key_approx, val_approx = self.codec.decode(compressed_gpu)
        del self.scheduler._preempted[request_id]
        return key_approx, val_approx

    def overlap_efficiency(self) -> float:
        """최근 오버랩 효율의 이동 평균."""
        if not self._overlap_efficiency_history:
            return 0.0
        return sum(self._overlap_efficiency_history[-32:]) / len(self._overlap_efficiency_history[-32:])

    def compression_ratio(self) -> float:
        """압축 전 대비 압축 후 크기 비율."""
        if self._total_bytes_before == 0:
            return 0.0
        return 1.0 - self._total_bytes_after / self._total_bytes_before


def _move_dict_to_cpu(d) -> dict:
    """재귀적으로 딕셔너리 내 모든 텐서를 CPU로 이동."""
    if isinstance(d, torch.Tensor):
        return d.cpu()
    if isinstance(d, dict):
        return {k: _move_dict_to_cpu(v) for k, v in d.items()}
    return d


def _move_dict_to_gpu(d) -> dict:
    """재귀적으로 딕셔너리 내 모든 텐서를 GPU로 이동."""
    if isinstance(d, torch.Tensor):
        return d.cuda() if torch.cuda.is_available() else d
    if isinstance(d, dict):
        return {k: _move_dict_to_gpu(v) for k, v in d.items()}
    return d


def _flatten_tensors(d) -> list:
    """딕셔너리 내 모든 텐서를 평탄화."""
    if isinstance(d, torch.Tensor):
        return [d]
    if isinstance(d, dict):
        result = []
        for v in d.values():
            result.extend(_flatten_tensors(v))
        return result
    return []
```

---

### 4. StaticDynamicSegmentCache (Activity B)

```python
# src/cache/static_dynamic_segment.py 의사코드

from typing import Dict, List, Optional, Set
import torch
from src.cache.base import CacheStore


class StaticDynamicSegmentCache(CacheStore):
    """
    KEEP(arXiv 2602.23592) 원리 기반 에이전트 메모리 갱신 패턴 기반 정적/동적 세그먼트 분리.

    - 정적 세그먼트(Static): 시스템 프롬프트, 공통 문서 등 갱신 없는 세그먼트.
      LRU 퇴거 대상에서 제외. 반복 재사용 → 비연속 히트율 향상.
    - 동적 세그먼트(Dynamic): 에이전트 행동 결과, 대화 히스토리 등 갱신 빈도 높은 세그먼트.
      갱신 시 max_invalidation_range 이내의 이후 세그먼트만 무효화 (Multi-hop 전파 제한).

    CacheStore 인터페이스 완전 구현. 기존 Runner의 get()/put() API와 호환.
    """

    def __init__(
        self,
        capacity_bytes: int,
        max_invalidation_range: int = 2,  # 동적 세그먼트 갱신 시 무효화할 이후 세그먼트 최대 수
        max_recompute_hops: int = 2,       # Multi-hop 재계산 전파 깊이 제한
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.max_invalidation_range = max_invalidation_range
        self.max_recompute_hops = max_recompute_hops

        self._static_store: Dict[str, torch.Tensor] = {}   # 정적 세그먼트 KV
        self._dynamic_store: Dict[str, torch.Tensor] = {}  # 동적 세그먼트 KV
        self._static_keys: Set[str] = set()                 # 정적으로 등록된 키 집합
        self._lru_order: List[str] = []                     # 동적 세그먼트 LRU 순서

        self._hit_count: int = 0
        self._miss_count: int = 0
        self._segment_order: List[str] = []  # 삽입 순서 (무효화 범위 계산용)

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 구현                                           #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """기본 삽입: 동적 세그먼트로 등록. mark_static()으로 정적 승격 가능."""
        if key in self._static_keys:
            self._static_store[key] = value
        else:
            self._dynamic_store[key] = value
            if key not in self._lru_order:
                self._lru_order.append(key)
            if key not in self._segment_order:
                self._segment_order.append(key)
        self._maybe_evict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._static_store:
            self._hit_count += 1
            return self._static_store[key]
        if key in self._dynamic_store:
            self._hit_count += 1
            # LRU 갱신
            if key in self._lru_order:
                self._lru_order.remove(key)
                self._lru_order.append(key)
            return self._dynamic_store[key]
        self._miss_count += 1
        return None

    def evict(self) -> int:
        """동적 세그먼트 LRU 퇴거 (정적 세그먼트 제외)."""
        if not self._lru_order:
            return 0
        evict_key = self._lru_order.pop(0)
        kv = self._dynamic_store.pop(evict_key, None)
        if evict_key in self._segment_order:
            self._segment_order.remove(evict_key)
        return kv.nbytes if kv is not None else 0

    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        static_bytes = sum(v.nbytes for v in self._static_store.values())
        dynamic_bytes = sum(v.nbytes for v in self._dynamic_store.values())
        return static_bytes + dynamic_bytes

    def reset_stats(self) -> None:
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------ #
    # 정적/동적 분류 API                                                   #
    # ------------------------------------------------------------------ #

    def mark_static(self, key: str) -> None:
        """세그먼트를 정적으로 승격. 동적 저장소에 있으면 이동."""
        self._static_keys.add(key)
        if key in self._dynamic_store:
            self._static_store[key] = self._dynamic_store.pop(key)
            if key in self._lru_order:
                self._lru_order.remove(key)

    def mark_dynamic(self, key: str) -> None:
        """세그먼트를 동적으로 전환."""
        self._static_keys.discard(key)
        if key in self._static_store:
            self._dynamic_store[key] = self._static_store.pop(key)
            if key not in self._lru_order:
                self._lru_order.append(key)

    def update_segment(self, key: str, new_value: torch.Tensor) -> List[str]:
        """
        동적 세그먼트 갱신 + Multi-hop 무효화 범위 최소화.
        갱신된 세그먼트 이후 max_invalidation_range 개 세그먼트만 무효화.

        반환: 무효화된 세그먼트 키 목록 (재계산 필요).
        """
        # 정적 세그먼트는 갱신 불가
        if key in self._static_keys:
            raise ValueError(f"정적 세그먼트 '{key}'는 update_segment로 갱신 불가. mark_dynamic() 먼저 호출.")

        # 갱신 적용
        self._dynamic_store[key] = new_value

        # 무효화 범위: 삽입 순서 기준 이후 max_invalidation_range 개 세그먼트
        invalidated = []
        if key in self._segment_order:
            idx = self._segment_order.index(key)
            invalidation_end = min(idx + 1 + self.max_invalidation_range, len(self._segment_order))
            for inv_key in self._segment_order[idx + 1 : invalidation_end]:
                if inv_key in self._dynamic_store and inv_key not in self._static_keys:
                    del self._dynamic_store[inv_key]
                    invalidated.append(inv_key)
                    if inv_key in self._lru_order:
                        self._lru_order.remove(inv_key)

        return invalidated

    # ------------------------------------------------------------------ #
    # 내부 헬퍼                                                            #
    # ------------------------------------------------------------------ #

    def _maybe_evict(self) -> None:
        """용량 초과 시 동적 세그먼트 LRU 퇴거."""
        while self.memory_bytes() > self.capacity_bytes and self._lru_order:
            self.evict()
```

---

### 5. ManifoldKVWindowedEviction (Activity C)

```python
# src/cache/manifoldkv_windowed.py 의사코드

from typing import Dict, List, Optional, Tuple
import torch
from src.cache.base import CacheStore


class ManifoldKVWindowedEviction(CacheStore):
    """
    ManifoldKV(arXiv 2602.08343) + WindowedManifoldKV 기반 유클리드 아웃라이어 퇴거.

    기존 퇴거 정책(코사인 유사도 기반)이 스케일(크기) 정보를 무시해 의미론적으로
    중요한 토큰을 퇴거하는 문제를, 슬라이딩 윈도우 로컬 중심 기준 유클리드 거리로
    아웃라이어 스코어를 계산해 해소한다.

    기존 CompressedSegmentCache.evict() 메서드에 drop-in 교체 가능.
    `outlier_score_fn`을 외부에서 주입하면 기존 퇴거 정책을 그대로 교체.
    """

    def __init__(
        self,
        capacity_bytes: int,
        window_size: int = 4096,      # 슬라이딩 윈도우 크기 (토큰 수)
        evict_ratio: float = 0.2,     # 퇴거 대상 비율 (아웃라이어 스코어 하위 비율)
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.window_size = window_size
        self.evict_ratio = evict_ratio

        self._store: Dict[str, torch.Tensor] = {}     # key → KV 텐서 [n_tokens, d_head]
        self._key_vectors: Dict[str, torch.Tensor] = {}  # key → K 벡터 (스코어 계산용)
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 구현                                           #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """
        value: [n_tokens, d_head] KV 텐서.
        K 벡터는 value 자체로 간주 (Key 텐서 저장 시).
        """
        self._store[key] = value
        self._key_vectors[key] = value  # K 벡터 = KV 텐서 자체
        self._maybe_evict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._hit_count += 1
            return self._store[key]
        self._miss_count += 1
        return None

    def evict(self) -> int:
        """유클리드 아웃라이어 스코어 하위 토큰 보유 세그먼트 퇴거."""
        if not self._store:
            return 0
        scores = self._compute_outlier_scores()
        if not scores:
            return 0
        # 아웃라이어 스코어가 가장 낮은 (덜 두드러진) 세그먼트 퇴거
        worst_key = min(scores, key=scores.get)
        kv = self._store.pop(worst_key, None)
        self._key_vectors.pop(worst_key, None)
        return kv.nbytes if kv is not None else 0

    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------ #
    # 유클리드 아웃라이어 스코어 계산                                       #
    # ------------------------------------------------------------------ #

    def _compute_outlier_scores(self) -> Dict[str, float]:
        """
        세그먼트별 유클리드 아웃라이어 스코어 계산.

        알고리즘:
        1. 모든 세그먼트의 K 벡터를 하나로 연결 → [total_tokens, d_head]
        2. 슬라이딩 윈도우(window_size)로 이동하며 로컬 중심(centroid) 계산
        3. 각 토큰의 아웃라이어 스코어 = ||k_i - centroid(window)||_2 (torch.cdist 활용)
        4. 세그먼트별 평균 아웃라이어 스코어 반환

        수식: outlier_score(k_i, window) = ||k_i - mean(k_{window})||_2
        스코어가 높을수록 두드러진 (의미론적으로 중요한) 토큰.
        퇴거 시 스코어 낮은 세그먼트 우선 제거.
        """
        scores: Dict[str, float] = {}
        keys_list = list(self._key_vectors.keys())
        if not keys_list:
            return scores

        # 모든 K 벡터 연결
        all_kvs = []
        seg_ranges: List[Tuple[str, int, int]] = []  # (key, start_idx, end_idx)
        cursor = 0
        for key in keys_list:
            kv = self._key_vectors[key]
            if kv.dim() == 1:
                kv = kv.unsqueeze(0)
            all_kvs.append(kv.float())
            seg_ranges.append((key, cursor, cursor + kv.shape[0]))
            cursor += kv.shape[0]

        if not all_kvs:
            return scores

        all_k = torch.cat(all_kvs, dim=0)  # [total_tokens, d_head]
        total_tokens, d_head = all_k.shape

        # 슬라이딩 윈도우 중심 계산 + 유클리드 거리
        token_scores = torch.zeros(total_tokens)
        for win_start in range(0, total_tokens, self.window_size):
            win_end = min(win_start + self.window_size, total_tokens)
            window_k = all_k[win_start:win_end]  # [win_len, d_head]
            centroid = window_k.mean(dim=0, keepdim=True)  # [1, d_head]
            # torch.cdist: [win_len, 1] 유클리드 거리
            dists = torch.cdist(window_k, centroid).squeeze(-1)  # [win_len]
            token_scores[win_start:win_end] = dists

        # 세그먼트별 평균 스코어
        for key, start, end in seg_ranges:
            scores[key] = token_scores[start:end].mean().item()

        return scores

    def outlier_score_fn(self, key_vectors: torch.Tensor) -> torch.Tensor:
        """
        외부 호출용 스코어 함수. 기존 evict() 정책에 drop-in 교체 인터페이스.

        Args:
            key_vectors: [n_tokens, d_head]
        Returns:
            scores: [n_tokens] — 각 토큰의 유클리드 아웃라이어 스코어
        """
        kv = key_vectors.float()
        n_tokens, d_head = kv.shape
        token_scores = torch.zeros(n_tokens)
        for win_start in range(0, n_tokens, self.window_size):
            win_end = min(win_start + self.window_size, n_tokens)
            window_k = kv[win_start:win_end]
            centroid = window_k.mean(dim=0, keepdim=True)
            dists = torch.cdist(window_k, centroid).squeeze(-1)
            token_scores[win_start:win_end] = dists
        return token_scores

    def _maybe_evict(self) -> None:
        while self.memory_bytes() > self.capacity_bytes and self._store:
            self.evict()
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(eOptShrinkQCodec + ManifoldKVWindowedEviction)를 포함하므로
다음 정확도 보존 검증이 **필수**이다 (evaluation_criteria.md §4 필수 항목).

### perplexity 측정

| 항목 | 상세 |
|------|------|
| 모델 | GPT-2 (small, 117M) — 추가 라이선스 없이 재현 가능 |
| 데이터셋 | WikiText-2 (test split, 표준 벤치마크) |
| 측정 방법 | stride=512, max_length=1024 sliding window perplexity |
| 허용 오차 | ±1% 이내 (예: 베이스라인 30.0 → 압축 후 29.7~30.3 허용) |
| 압축 설정 범위 | key_bits=2/value_bits=3 (기본), key_bits=3/value_bits=4 (완화) 각각 측정 |
| 비교 대상 | (a) 전체 KV 원본 (베이스라인), (b) eOptShrinkQCodec key_bits=2/val_bits=3, (c) eOptShrinkQCodec key_bits=3/val_bits=4 |

### 태스크 정확도 측정

| 항목 | 상세 |
|------|------|
| 벤치마크 | LongBench 8개 서브태스크 (HotpotQA, 2WikiMultiHopQA, MuSiQue, GovReport, QMSum, MultiFieldQA-en, MultiFieldQA-zh, TriviaQA) |
| 추가 벤치마크 | 멀티-니들 NIAH (Needle-in-A-Haystack, 다중 키 검색 시나리오) |
| 허용 오차 | ±1% 이내 |
| 측정 스크립트 | `experiments/run_eopt_accuracy.py` |
| 캘리브레이션 | 학습 데이터에서 무작위 20개 요청으로 레이어별 noise_level 추정 (단 1회 오프라인) |

### accuracy-preserving 이론 근거 (eOptShrinkQCodec)

1. **BBP 위상전이의 이론적 보장**: BBP 임계값 위 특이값은 신호 성분(정보 보존), 아래는 노이즈 성분 → 자동 랭크 선택의 통계적 일관성(consistency) 보장
2. **잔차 내적 편향 근-제로**: 저랭크 성분 분리 후 잔차는 `E[residual^T · lowrank] ≈ 0` → TurboQuantCodec 잔차 양자화 왜곡 최소
3. **eOptShrinkQ 실증**: arXiv 2605.02905에서 Llama-3.1-8B, Ministral-8B LongBench 16개 태스크, 2.2비트에서 TurboQuant 3비트 동등/우수 달성. 멀티-니들 NIAH에서 FP16 비압축과 동등.

### 검증 테스트

**파일**: `tests/unit/test_eopt_shrinkq_accuracy.py`

```python
# tests/unit/test_eopt_shrinkq_accuracy.py 의사코드

def test_bbp_rank_selection_consistency():
    """
    동일 노이즈 수준에서 BBP 자동 랭크 선택이 일관성을 가짐을 검증.
    서로 다른 시드로 생성된 동일 분포 데이터에서 랭크 선택 결과의 분산이 작아야 함.
    """
    codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
    ranks = []
    for seed in range(10):
        torch.manual_seed(seed)
        # 랭크-4 신호 + 노이즈 (합성 KV)
        signal = torch.randn(64, 4) @ torch.randn(4, 32)  # rank-4
        noise = torch.randn(64, 32) * 0.1
        kv = signal + noise
        codec.calibrate([kv])
        ranks.append(codec._auto_ranks.get(0, 0))
    # 랭크 선택 결과의 표준편차가 2 이하 (일관성)
    import statistics
    assert statistics.stdev(ranks) <= 2, f"BBP 랭크 선택 불일관: ranks={ranks}"

def test_residual_bias_near_zero():
    """
    저랭크 성분 분리 후 잔차의 내적 편향이 근-제로임을 검증.
    |E[residual^T · lowrank]| < 1e-3 검증.
    """
    codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
    torch.manual_seed(42)
    kv = torch.randn(128, 64)
    codec.calibrate([kv])
    r = codec._auto_ranks.get(0, 4)

    U, S, Vh = torch.linalg.svd(kv.float(), full_matrices=False)
    lowrank = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
    residual = kv.float() - lowrank

    # 잔차와 저랭크 성분의 내적 편향
    bias = (residual * lowrank).mean().abs().item()
    assert bias < 0.1, f"잔차 내적 편향 초과: {bias:.6f}"

def test_encode_decode_roundtrip_cosine_similarity():
    """
    encode → decode 왕복 후 코사인 유사도가 0.90 이상임을 검증.
    (perplexity ±1% 허용 오차와 상관된 proxy 메트릭)
    """
    codec = eOptShrinkQCodec(num_layers=2, key_bits=2, value_bits=3)
    torch.manual_seed(42)
    calibration_kvs = [torch.randn(64, 32) for _ in range(20)]
    codec.calibrate(calibration_kvs)

    kv_key = torch.randn(128, 32)
    kv_val = torch.randn(128, 32)
    compressed = codec.encode(kv_key, kv_val, layer_idx=0)
    key_approx, val_approx = codec.decode(compressed)

    cos_key = torch.nn.functional.cosine_similarity(
        kv_key.flatten().unsqueeze(0), key_approx.flatten().unsqueeze(0)
    ).item()
    cos_val = torch.nn.functional.cosine_similarity(
        kv_val.flatten().unsqueeze(0), val_approx.flatten().unsqueeze(0)
    ).item()
    assert cos_key >= 0.85, f"Key 코사인 유사도 낮음: {cos_key:.4f}"
    assert cos_val >= 0.85, f"Value 코사인 유사도 낮음: {cos_val:.4f}"

def test_memory_reduction_at_least_30_percent():
    """
    eOptShrinkQCodec 압축 후 메모리가 베이스라인 대비 30% 이상 감소함을 검증.
    (evaluation_criteria.md §4 KV Memory Reduction 기준)
    """
    codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
    torch.manual_seed(42)
    codec.calibrate([torch.randn(64, 64) for _ in range(20)])
    est = codec.memory_bytes_estimate(n_tokens=512, d_head=64, layer_idx=0)
    assert est["reduction_ratio"] >= 0.30, \
        f"메모리 감소율 미달: {est['reduction_ratio']:.2%} < 30%"

def test_perplexity_proxy_within_tolerance():
    """
    합성 KV로 perplexity proxy(MSE 상대 오차)가 ±1% 이내임을 검증.
    실제 perplexity는 experiments/run_eopt_accuracy.py에서 측정.
    """
    codec = eOptShrinkQCodec(num_layers=1, key_bits=3, value_bits=4)  # 완화 설정
    torch.manual_seed(42)
    codec.calibrate([torch.randn(64, 32) for _ in range(20)])

    kv_key = torch.randn(256, 32)
    kv_val = torch.randn(256, 32)
    compressed = codec.encode(kv_key, kv_val, layer_idx=0)
    key_approx, val_approx = codec.decode(compressed)

    # MSE 상대 오차 (perplexity proxy)
    mse_key = ((kv_key - key_approx) ** 2).mean() / (kv_key ** 2).mean()
    mse_val = ((kv_val - val_approx) ** 2).mean() / (kv_val ** 2).mean()
    assert mse_key.item() < 0.05, f"Key MSE 상대 오차 초과: {mse_key.item():.4f}"
    assert mse_val.item() < 0.05, f"Value MSE 상대 오차 초과: {mse_val.item():.4f}"
```

### CompressedPreemptionPipeline A+C 복합 정확도 검증

Activity A+C 통합 시 선점-압축-복원 사이클이 정확도에 미치는 영향을 추가 검증한다.

| 항목 | 기준 |
|------|------|
| 압축 후 재개 정확도 | decode 후 코사인 유사도 ≥ 0.85 |
| SLA Tier-A 선점 제외 | sla_tier_a_ids에 포함된 요청은 선점 큐에 등록 안 됨 |
| A+C 복합 정확도 | ±1% 이내 (evaluation_criteria.md §5 필수) |

**파일**: `tests/integration/test_cross_ac_preempt_compress.py`

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-08.yaml
experiment:
  date: "2026-05-08"
  activity: "A+C"
  description: "PreemptiveKVOffloadScheduler + eOptShrinkQCodec + CompressedPreemptionPipeline (A+C Cross-1), 보조: StaticDynamicSegmentCache(B), ManifoldKVWindowedEviction(C)"

scheduler:
  type: "preemptive_kv_offload"
  cache_capacity_bytes: 4294967296     # 4 GiB
  threshold_preempt: 0.85              # 버퍼 점유율 선점 임계값
  consumption_rate_window: 32          # 소비율 이동 평균 윈도우 (토큰 수)
  fairness_max_wait: 10                # 선점 면제 최대 대기 스텝
  preempt_compress: true               # CompressedPreemptionPipeline 연동
  sla_tier_a_ids: []                   # SLA Tier-A 선점 제외 요청 ID (기본 빈 목록)

compression:
  method: "eopt_shrinkq"               # eOptShrinkQCodec
  key_bits: 2                          # Key 양자화 비트 (공격적)
  value_bits: 3                        # Value 양자화 비트 (보수적)
  calibration_samples: 20              # 오프라인 캘리브레이션 최소 샘플 수
  calibration_save_path: "results/2026-05-08/eopt_calibration.pt"

compressed_preemption:
  use_dual_stream: true                # CUDA 이중 스트림 오버랩
  sla_tier_a_no_compress: true         # SLA Tier-A 선점 시 압축 미적용

cache:
  type: "static_dynamic_segment"       # StaticDynamicSegmentCache
  capacity_bytes: 4294967296           # 4 GiB
  max_invalidation_range: 2            # 동적 세그먼트 갱신 시 무효화 이후 세그먼트 최대 수
  max_recompute_hops: 2                # Multi-hop 재계산 전파 깊이 제한

eviction:
  policy: "manifoldkv_windowed"        # ManifoldKVWindowedEviction
  window_size: 4096                    # 슬라이딩 윈도우 크기 (토큰 수)
  evict_ratio: 0.2                     # 퇴거 대상 하위 비율

seed: 42
results_dir: "results/2026-05-08"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_preemptive_scheduler.py`
  - `test_preempt_trigger_above_threshold`: buffer_occupancy > 0.85 시 선점 트리거 검증
  - `test_no_preempt_below_threshold`: 정상 부하 시 선점 미발생 검증
  - `test_sla_tier_a_not_preempted`: SLA Tier-A 요청 선점 제외 검증
  - `test_fairness_max_wait_respected`: fairness_max_wait 초과 시 선점 면제 검증
  - `test_offload_kv_async_cpu_move`: offload_kv_async() 후 KV가 CPU 메모리에 있음 검증
  - `test_restore_kv_returns_gpu_tensor`: restore_kv() 후 KV가 GPU 텐서임 검증
  - `test_buffer_occupancy_ratio_calculation`: buffer_occupancy_ratio() 계산 정확성

- [ ] `tests/unit/test_eopt_shrinkq_accuracy.py`
  - `test_bbp_rank_selection_consistency`: BBP 랭크 선택 일관성 (위 의사코드 참조)
  - `test_residual_bias_near_zero`: 잔차 내적 편향 근-제로 (위 의사코드 참조)
  - `test_encode_decode_roundtrip_cosine_similarity`: 왕복 코사인 유사도 ≥ 0.85
  - `test_memory_reduction_at_least_30_percent`: 메모리 감소 ≥ 30%
  - `test_perplexity_proxy_within_tolerance`: MSE 상대 오차 < 5%
  - `test_key_value_asymmetric_bits`: Key 2비트 / Value 3비트 비대칭 적용 검증
  - `test_calibrate_save_load_roundtrip`: 캘리브레이션 저장·로드 왕복 검증
  - `test_auto_rank_positive`: 자동 선택된 랭크가 1 이상임 검증

- [ ] `tests/unit/test_static_dynamic_segment.py`
  - `test_put_get_basic`: 기본 put/get 동작
  - `test_mark_static_excludes_from_eviction`: 정적 세그먼트가 evict() 대상 제외됨
  - `test_mark_dynamic_restores_eviction_eligibility`: mark_dynamic() 후 퇴거 가능
  - `test_update_segment_invalidates_range`: 갱신 후 max_invalidation_range 이내 무효화
  - `test_update_segment_rejects_static`: 정적 세그먼트 갱신 시 ValueError
  - `test_multi_hop_depth_limit`: max_invalidation_range 초과 세그먼트 미무효화
  - `test_hit_rate_tracking`: hit_rate() 정확성
  - `test_cachestore_interface`: CacheStore 추상 메서드 전부 구현 검증

- [ ] `tests/unit/test_manifoldkv_eviction.py`
  - `test_outlier_score_euclidean_distance`: 유클리드 거리 기반 스코어 계산 정확성
  - `test_high_outlier_not_evicted`: 아웃라이어 스코어 높은 토큰 세그먼트 보존 검증
  - `test_low_outlier_evicted_first`: 아웃라이어 스코어 낮은 세그먼트 우선 퇴거
  - `test_windowed_centroid_vs_global`: 슬라이딩 윈도우 중심 vs 전역 중심 차이 검증
  - `test_outlier_score_fn_shape`: outlier_score_fn() 반환 형태 [n_tokens] 검증
  - `test_cachestore_interface`: CacheStore 추상 메서드 전부 구현 검증

- [ ] `tests/integration/test_cross_ac_preempt_compress.py`
  - `test_compressed_preemption_pipeline_e2e`: CompressedPreemptionPipeline 전체 파이프라인
  - `test_offload_compression_ratio`: 압축 후 크기 감소 ≥ 30% 검증
  - `test_decode_cosine_similarity_after_roundtrip`: offload → restore 후 코사인 유사도 ≥ 0.85
  - `test_overlap_efficiency_recorded`: overlap_efficiency() 메트릭 기록 확인
  - `test_sla_tier_a_accuracy_preserved`: SLA Tier-A 요청 accuracy delta ±1% 이내
  - `test_cross_ac_throughput_improvement`: A+C 복합 처리량이 단일 A 대비 +5% 이상

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 전부 통과** (100%, evaluation_criteria.md §0)
2. **통합 테스트 전부 통과** (100%, evaluation_criteria.md §0)
3. **CacheStore 인터페이스 준수** — `StaticDynamicSegmentCache`, `ManifoldKVWindowedEviction` 각각 모든 추상 메서드 구현 (evaluation_criteria.md §0)
4. **Activity C Accuracy 보존 필수** — `test_eopt_shrinkq_accuracy.py` 전부 통과 + `experiments/run_eopt_accuracy.py` 실행 시 perplexity 변화 ±1% 이내 (evaluation_criteria.md §4 필수)
5. **Activity C 태스크 정확도 필수** — LongBench 8개 서브태스크 정확도 변화 ±1% 이내 (evaluation_criteria.md §4 필수)
6. **A+C 복합 Accuracy 보존 필수** — CompressedPreemptionPipeline 선점-압축-복원 사이클 후 accuracy delta ±1% 이내 (evaluation_criteria.md §5 필수, C 포함 시)
7. **Activity A 스케줄링 오버헤드** — TTFT p50 증가 +5% 이내 (정상 부하, evaluation_criteria.md §2 필수)
8. **TTFT p99 개선** — 요청 폭주 시 베이스라인 대비 감소 확인 (evaluation_criteria.md §2)
9. **KV Memory Reduction** — 베이스라인 대비 −30% 이상 (evaluation_criteria.md §4)
10. **비연속 히트율** — 전체 히트 중 비연속 히트 비율 ≥ 30% (evaluation_criteria.md §3)
11. **처리량 목표** — tokens/sec 베이스라인 대비 +20% 이상 (evaluation_criteria.md §1)
12. **압축 오버헤드** — eOptShrinkQCodec Encode/Decode 추가 지연 TTFT +10% 이내 (evaluation_criteria.md §4)
13. **타입 힌트** — 모든 공개 함수·메서드에 존재 (evaluation_criteria.md §0)
14. **설정 YAML 존재** — `configs/experiments/2026-05-08.yaml` 생성됨 (evaluation_criteria.md §0)
15. **기존 파일 회귀 없음** — 수정 금지 파일 목록의 모든 기존 테스트 통과
16. **결과 기록** — `results/2026-05-08/metrics.json` 에 목표 지표 수치 기록

SPEC_SAVED: Spec.md
