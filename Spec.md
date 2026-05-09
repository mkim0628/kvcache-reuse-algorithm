<!-- 변경 이유 (이전 Spec.md: 2026-05-08 대비):
이전 사이클(2026-05-08)은 A+C (PreemptiveKVOffloadScheduler + eOptShrinkQCodec + CompressedPreemptionPipeline)
조합이었다. 이번 사이클은 A+B (PPDAppendPrefillRouter + TriangleInequalitySegmentIndex + HitAwarePPDRouter)
조합으로 전환하며, 보조로 Activity B(SemanticBoundarySegmentCache) 및
Activity C(SpecKVCompressionGammaController + ContextIntensiveAccuracyGuard)를 추가한다.

주요 변경:
1. [Activity A 교체] PreemptiveKVOffloadScheduler(TokenFlow 기반 선점형 스케줄링) →
   PPDAppendPrefillRouter(PPD arXiv 2603.13358 기반 D 노드 append-prefill 동적 선택).
   P/D 역할을 요청 유형과 세그먼트 히트율에 따라 동적으로 결정하는 완전히 새로운
   라우팅 패러다임. Turn 2+ TTFT 68% 단축 실증 데이터 기반.

2. [Activity B 교체] StaticDynamicSegmentCache(KEEP 기반 정적/동적 분리) →
   TriangleInequalitySegmentIndex(LycheeCluster arXiv 2603.08453 기반 O(log N) 계층 인덱스)
   + SemanticBoundarySegmentCache(SemantiCache arXiv 2603.14303 기반 의미 경계 청킹).
   O(N) 검색 병목을 O(log N)으로 해결하는 근본적 자료구조 개선.

3. [Cross-1 신규] HitAwarePPDRouter: A-1(PPDAppendPrefillRouter) + B-1(TriangleInequalitySegmentIndex)
   통합. D 노드에서 비연속 세그먼트를 로그 시간에 검색하여 append-prefill 처리.

4. [Activity C 신규 추가] SpecKVCompressionGammaController(SpecKV arXiv 2605.02888 기반
   압축-γ 결합 최적화) + ContextIntensiveAccuracyGuard(컨텍스트 밀도 기반 압축 한도 보호).
   eOptShrinkQCodec(05-08 기구현)과 통합.

5. [보존 파일] 이전 사이클 파일
   (preemptive_kv_offload.py, compressed_preemption.py, eopt_shrinkq_codec.py,
   static_dynamic_segment.py, manifoldkv_windowed.py,
   query_centric_recompute.py, info_flow_reorder.py, tri_attention_codec.py,
   qc_tri_store.py, dual_filter_selector.py, diff_aware_store.py, turbo_quant.py,
   dhd_segment_cache.py, speculative_fetcher.py, sign_vq_segment.py,
   leverage_compressor.py, compression.py, segmented.py, contiguous.py,
   tri_state_compressor.py, compressed_segment.py, segment_adapter.py,
   dag_topology_scheduler.py, workload_ttl_cache.py, redundancy_eviction.py,
   fireq_codec.py, nqkv_codec.py, compressed_diff_store.py,
   cache_aware_scheduler.py, multi_node_scheduler.py, dag_ttl_adjuster.py,
   dual_map_scheduler.py)은 이번 사이클에서 수정하지 않는다.
   기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-09

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-09.md`
**최우선 구현 타겟**: Cross-1 (A+B) — PPDAppendPrefillRouter(A-1) + TriangleInequalitySegmentIndex(B-1)
+ HitAwarePPDRouter 통합, 보조로 SemanticBoundarySegmentCache(B-2),
SpecKVCompressionGammaController(C-2), ContextIntensiveAccuracyGuard(C-3)

**해결하려는 문제**:
- 멀티턴 LLM 서빙에서 매 턴마다 이전 응답 KV를 P 노드로 재전송 후 재프리필하는 기존 P/D 분리 구조의
  두 가지 비효율(Turn 2+ KV 재전송 병목, P→D KV 전송 대역폭 포화)을, D 노드에서 직접
  append-prefill(새 입력 토큰만 처리 + 캐시된 KV 재사용)을 처리하는 동적 라우터로 해소한다.
  PPD(arXiv 2603.13358) 기반으로 Turn 2+ TTFT 68% 단축이 가능하다.
- 비연속 KV 세그먼트 검색이 O(N) 선형 스캔에 의존해 세그먼트 수 증가 시 병목이 되는 문제를,
  삼각부등식 기반 재귀 계층 인덱스로 O(log N) 검색을 실현한다. LycheeCluster(arXiv 2603.08453)
  기반으로 검색 속도 3.6× 향상이 가능하다.
- PPDAppendPrefillRouter의 핵심 라우팅 신호인 "세그먼트 히트 예상 여부"를 TriangleInequalitySegmentIndex
  O(log N) 검색으로 빠르게 추정하여, "히트 예상 시 D 노드 append, 미스 예상 시 P 노드 full prefill"
  하는 캐시-인식 PPD 라우터(HitAwarePPDRouter)를 실현한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling — PPDAppendPrefillRouter (D 노드 append-prefill 동적 선택)
- [x] Activity B: Non-Contiguous KV Cache Reuse — TriangleInequalitySegmentIndex (O(log N) 계층 인덱스)
  + SemanticBoundarySegmentCache (의미 경계 청킹 + GSC 클러스터링)
- [x] Activity C: KV Cache Compression — SpecKVCompressionGammaController (압축-γ 결합 최적화)
  + ContextIntensiveAccuracyGuard (컨텍스트 밀도 기반 압축 한도 보호)

---

## 목표

- [ ] 목표 1 (§1 Throughput): tokens/sec 베이스라인 대비 +20% 이상 — PPD append-prefill로 Turn 2+ 재전송 제거 + 인덱스 검색 오버헤드 제거 (evaluation_criteria.md §1)
- [ ] 목표 2 (§2 Activity A): TTFT p50 증가 +5% 이내 (정상 부하 시) (evaluation_criteria.md §2 필수)
- [ ] 목표 3 (§2 Activity A): Turn 2+ TTFT p50 −68% (D 노드 append-prefill, PPD arXiv 2603.13358 실증) (evaluation_criteria.md §2)
- [ ] 목표 4 (§2 Activity A): 히트 확률 기반 라우팅으로 캐시 히트율 +10%p 이상 향상 (evaluation_criteria.md §2)
- [ ] 목표 5 (§3 Non-Contiguous Hit Rate): 전체 히트 중 비연속 히트 비율 ≥ 30% — TriangleInequalitySegmentIndex + SemanticBoundarySegmentCache 통합 (evaluation_criteria.md §3 높음)
- [ ] 목표 6 (§3 Activity B): TriangleInequalitySegmentIndex 검색 속도 O(N) 대비 3.6× 이상 향상 (N=10K 세그먼트 기준) (evaluation_criteria.md §3)
- [ ] 목표 7 (§3 Activity B): KV Memory Footprint 베이스라인 대비 +20% 이내 (evaluation_criteria.md §3 높음)
- [ ] 목표 8 (§4 Accuracy 필수): perplexity 변화 ±1% 이내 — SpecKVCompressionGammaController + ContextIntensiveAccuracyGuard 압축 경로 (evaluation_criteria.md §4 필수)
- [ ] 목표 9 (§4 Accuracy 필수): downstream 태스크 정확도 변화 ±1% 이내 (evaluation_criteria.md §4 필수)
- [ ] 목표 10 (§5 Cross A+B): 복합 처리량 향상 단일 Activity 대비 추가 +5% 이상 (evaluation_criteria.md §5)
- [ ] 목표 11 (§5 Cross A+B): 복합 메모리 감소 단일 Activity 대비 추가 −10% 이상 (evaluation_criteria.md §5)

---

## 아키텍처 개요

```
요청 도착
    │
    ├─ HitAwarePPDRouter (Cross-1: A+B)
    │       │
    │       ├─ [Turn 분류] session_id + turn_count 메타데이터로 Turn 1 / Turn 2+ 결정
    │       │
    │       ├─ Turn 1 (신규 세션) → PPDAppendPrefillRouter → P 노드 full prefill
    │       │
    │       └─ Turn 2+ (멀티턴 계속)
    │               │
    │               ├─ TriangleInequalitySegmentIndex.estimate_hit_probability()
    │               │   → O(log N) 히트 확률 추정 (삼각부등식 서브트리 가지치기)
    │               │
    │               ├─ hit_prob > threshold_append (기본 0.7)
    │               │   → D 노드 append-prefill (캐시된 KV + 새 토큰만 처리)
    │               │
    │               └─ hit_prob ≤ 0.7
    │                   → P 노드 full prefill (KV 재전송, 정확도 보장)
    │
    ├─ TriangleInequalitySegmentIndex (Activity B)
    │       │
    │       ├─ [인덱스 구조] 재귀 계층 트리: pivot 선택(가장 먼 두 세그먼트) + max_radius 저장
    │       ├─ [검색] best-first 우선순위 큐 + 삼각부등식 서브트리 가지치기
    │       └─ [통합] StaticDynamicSegmentCache(05-08 기구현) 위에 인덱스 레이어로 추가
    │
    ├─ SemanticBoundarySegmentCache (Activity B)
    │       │
    │       ├─ [청킹] 의미 경계 탐지(구분자: .!?\n\n + 코드 블록 ```)
    │       ├─ [GSC 클러스터링] 어텐션 스코어 상위 시드 토큰 중심 탐욕적 병합
    │       ├─ [비례 어텐션] attention_weight_core = Σ(cluster attention weights)
    │       └─ [통합] TriangleInequalitySegmentIndex에 의미 코어 임베딩 등록
    │
    ├─ ContextIntensiveAccuracyGuard (Activity C)
    │       │
    │       ├─ [밀도 추정] 첫 128 토큰 샘플링: entity_ratio + numeric_ratio + token_entropy 가중합
    │       └─ [압축 한도 게이트] 고밀도(≥0.7) → ≥4비트; 중간(0.4~0.7) → 2.2~4비트; 저밀도(≤0.4) → ≤2.2비트
    │
    └─ SpecKVCompressionGammaController (Activity C)
            │
            ├─ [입력] eOptShrinkQCodec(기구현) 압축 수준 + min_draft_confidence + max_draft_entropy
            ├─ [경량 MLP] 입력 차원 3 → 출력 6 (γ ∈ {1,2,3,4,5,6})
            ├─ [eOptShrinkQCodec 통합] 압축 수준 출력을 자동으로 MLP 입력으로 주입
            └─ [온라인 적응] 검증 통과율 EMA 기반 γ 보정
```

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/triangle_index.py` | B | TriangleInequalitySegmentIndex: 삼각부등식 기반 재귀 계층 KV 세그먼트 인덱스 (O(log N) 검색) |
| `src/cache/semantic_boundary_cache.py` | B | SemanticBoundarySegmentCache: 의미 경계 청킹 + GSC 클러스터링 + 비례 어텐션 |
| `src/scheduler/ppd_append_prefill_router.py` | A | PPDAppendPrefillRouter: Turn 1/2+ 분류 + D/P 노드 동적 선택 |
| `src/scheduler/hit_aware_ppd_router.py` | A+B | HitAwarePPDRouter: A-1 + B-1 통합, 히트 확률 임계값 온라인 적응 |
| `src/cache/speckv_gamma_controller.py` | C | SpecKVCompressionGammaController: 경량 MLP γ 컨트롤러 + eOptShrinkQCodec 통합 |
| `src/cache/context_intensive_guard.py` | C | ContextIntensiveAccuracyGuard: 컨텍스트 밀도 기반 압축 한도 보호 게이트 |
| `experiments/run_index_speed_benchmark.py` | B | TriangleInequalitySegmentIndex O(N) vs O(log N) 검색 속도 벤치마크 (N=100/1K/10K) |
| `experiments/run_ppd_ttft.py` | A | PPD 라우팅 TTFT p50/p99 측정 (Turn 1 vs Turn 2+, 5턴 대화 시뮬레이션) |
| `experiments/run_text2json_accuracy.py` | C | Text2JSON 벤치마크 + LongBench + WikiText-2 perplexity 정확도 측정 |
| `tests/unit/test_triangle_index_search.py` | B | O(log N) 가지치기 검증 + 검색 속도 비교 단위 테스트 |
| `tests/unit/test_ppd_router.py` | A | Turn 분류 + 히트 확률 기반 P/D 선택 단위 테스트 |
| `tests/unit/test_semantic_boundary_cache.py` | B | 의미 경계 탐지 + GSC 병합 + 비례 어텐션 단위 테스트 |
| `tests/unit/test_speckv_gamma.py` | C | 압축 수준별 γ 선택 + perplexity proxy ±1% 단위 테스트 |
| `tests/unit/test_context_intensive_guard.py` | C | 밀도 추정 + 압축 한도 게이팅 단위 테스트 |
| `tests/integration/test_cross_ab_ppd_index.py` | A+B | HitAwarePPDRouter + TriangleInequalitySegmentIndex E2E 통합 테스트 |
| `configs/experiments/2026-05-09.yaml` | 공통 | 실험 설정 파일 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/scheduler/__init__.py` | `PPDAppendPrefillRouter`, `HitAwarePPDRouter` export 추가 |
| `src/cache/__init__.py` | `TriangleInequalitySegmentIndex`, `SemanticBoundarySegmentCache`, `SpecKVCompressionGammaController`, `ContextIntensiveAccuracyGuard` export 추가 |

### 수정 금지 파일 (이전 사이클 보존)

`preemptive_kv_offload.py`, `compressed_preemption.py`, `eopt_shrinkq_codec.py`,
`static_dynamic_segment.py`, `manifoldkv_windowed.py`,
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

### 1. TriangleInequalitySegmentIndex (Activity B)

**스케줄링 결정 단위**: 검색은 요청(request) 단위로 실행. 각 Turn 2+ 요청의 세그먼트 목록을
쿼리로 사용해 캐시에서 O(log N) 검색 실행.

**캐시 상태 접근 방법**: `CacheStore` 인터페이스(base.py)를 통해 세그먼트 임베딩에 접근.
내부적으로 `StaticDynamicSegmentCache`(기구현)을 백엔드 저장소로 사용하고,
그 위에 인덱스 레이어를 추가. `put()`/`get()` 모두 인덱스를 자동으로 갱신.

```python
# src/cache/triangle_index.py 의사코드

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq
import torch
from src.cache.base import CacheStore


@dataclass
class _IndexNode:
    """계층 인덱스의 단일 노드."""
    center_key: str              # 이 노드의 중심 세그먼트 키
    center_embedding: torch.Tensor  # 중심 임베딩 벡터 [d_embed]
    max_radius: float            # 서브트리 내 중심과 가장 먼 세그먼트까지의 거리
    left: Optional["_IndexNode"] = None
    right: Optional["_IndexNode"] = None
    segment_keys: List[str] = field(default_factory=list)  # 리프 노드의 세그먼트 키 목록


class TriangleInequalitySegmentIndex(CacheStore):
    """
    LycheeCluster(arXiv 2603.08453) 원리 기반 삼각부등식 재귀 계층 KV 세그먼트 인덱스.

    - 삼각부등식 가지치기: d(q,c) - max_radius(node) > best_dist → 서브트리 전체 가지치기
    - 이론적 검색 복잡도: O(N) → O(log N)
    - 백엔드: StaticDynamicSegmentCache(기구현) 위에 인덱스 레이어로 추가
    - CacheStore 인터페이스 완전 구현 (put/get/evict/hit_rate/memory_bytes/reset_stats)
    """

    def __init__(
        self,
        backend_cache: CacheStore,     # StaticDynamicSegmentCache 인스턴스
        embedding_dim: int = 64,       # 세그먼트 임베딩 차원 (Key 벡터 평균)
        leaf_size: int = 8,            # 리프 노드 최대 세그먼트 수 (분기 기준)
        distance_fn: str = "cosine",   # "cosine" 또는 "euclidean"
    ) -> None:
        self._backend = backend_cache
        self._embedding_dim = embedding_dim
        self._leaf_size = leaf_size
        self._distance_fn = distance_fn

        self._embeddings: Dict[str, torch.Tensor] = {}  # key → 세그먼트 임베딩 [d_embed]
        self._root: Optional[_IndexNode] = None
        self._dirty: bool = False  # 인덱스 재구성 필요 여부

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 구현                                           #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """세그먼트 저장 + 임베딩 추출 + 인덱스 업데이트."""
        self._backend.put(key, value)
        # 임베딩: Key 텐서의 평균 벡터 [d_embed]
        embedding = self._extract_embedding(value)
        self._embeddings[key] = embedding
        self._dirty = True  # 다음 검색 시 인덱스 재구성

    def get(self, key: str) -> Optional[torch.Tensor]:
        """직접 키 조회 (인덱스 우회, 정확한 키 알 때 사용)."""
        return self._backend.get(key)

    def evict(self) -> int:
        """백엔드 캐시 퇴거 + 인덱스에서 해당 키 제거."""
        freed = self._backend.evict()
        # 퇴거된 키를 임베딩 저장소에서도 제거
        current_keys = self._get_backend_keys()
        evicted_keys = set(self._embeddings.keys()) - set(current_keys)
        for k in evicted_keys:
            self._embeddings.pop(k, None)
        if evicted_keys:
            self._dirty = True
        return freed

    def hit_rate(self) -> float:
        return self._backend.hit_rate()

    def memory_bytes(self) -> int:
        embedding_bytes = sum(e.nbytes for e in self._embeddings.values())
        return self._backend.memory_bytes() + embedding_bytes

    def reset_stats(self) -> None:
        self._backend.reset_stats()

    # ------------------------------------------------------------------ #
    # 핵심 검색 API                                                        #
    # ------------------------------------------------------------------ #

    def search_nearest(
        self,
        query_embedding: torch.Tensor,  # [d_embed]
        top_k: int = 5,
        max_distance: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """
        삼각부등식 기반 best-first 탐색으로 top_k 최근접 세그먼트 반환.

        반환: [(key, distance), ...] 거리 오름차순 정렬.

        알고리즘:
        1. 인덱스가 dirty이면 _rebuild_index() 호출
        2. 우선순위 큐에 루트 노드 추가 (초기 거리 0)
        3. 큐에서 최소 거리 노드를 꺼내며:
           a. 리프 노드이면 세그먼트 키를 후보에 추가
           b. 내부 노드이면 자식 노드로 재귀 탐색
           c. 삼각부등식 조건 검사: d(q, node.center) - node.max_radius > current_best → 가지치기
        4. top_k 개 수집 후 반환
        """
        if self._dirty or self._root is None:
            self._rebuild_index()
        if self._root is None:
            return []

        results: List[Tuple[float, str]] = []  # (distance, key) min-heap
        # 우선순위 큐: (lower_bound_distance, node)
        pq: List[Tuple[float, _IndexNode]] = []
        heapq.heappush(pq, (0.0, self._root))

        while pq and len(results) < top_k * 2:
            lb_dist, node = heapq.heappop(pq)

            # 삼각부등식 가지치기:
            # 현재 best 거리보다 이 서브트리의 하한이 크면 탐색 불필요
            best_so_far = results[top_k - 1][0] if len(results) >= top_k else float("inf")
            if lb_dist > best_so_far:
                break

            if node.left is None and node.right is None:
                # 리프 노드: 세그먼트들의 실제 거리 계산
                for seg_key in node.segment_keys:
                    if seg_key in self._embeddings:
                        dist = self._distance(query_embedding, self._embeddings[seg_key])
                        if dist <= max_distance:
                            heapq.heappush(results, (-dist, seg_key))
                            if len(results) > top_k:
                                heapq.heappop(results)
            else:
                # 내부 노드: 자식 노드 탐색 후보 추가
                for child in [node.left, node.right]:
                    if child is not None:
                        child_dist = self._distance(query_embedding, child.center_embedding)
                        # 삼각부등식 하한: d(q, child_center) - child.max_radius
                        lower_bound = max(0.0, child_dist - child.max_radius)
                        if lower_bound <= best_so_far:
                            heapq.heappush(pq, (lower_bound, child))

        # 결과 정렬 (거리 오름차순)
        final = [(-d, k) for d, k in results]
        final.sort(key=lambda x: x[0])
        return [(k, d) for d, k in final[:top_k]]

    def estimate_hit_probability(
        self,
        query_segments: List[torch.Tensor],  # 쿼리 세그먼트 KV 텐서 목록
        threshold_distance: float = 0.3,
    ) -> float:
        """
        Turn 2+ 요청의 D 노드 캐시 히트 확률 O(log N) 추정.
        PPDAppendPrefillRouter의 라우팅 결정에 사용.

        각 쿼리 세그먼트에 대해 최근접 세그먼트 검색 후
        threshold_distance 이내인 세그먼트가 있으면 히트로 카운트.
        hit_probability = 히트 세그먼트 수 / 전체 쿼리 세그먼트 수.
        """
        if not query_segments or not self._embeddings:
            return 0.0
        hits = 0
        for seg_tensor in query_segments:
            query_emb = self._extract_embedding(seg_tensor)
            nearest = self.search_nearest(query_emb, top_k=1, max_distance=threshold_distance)
            if nearest and nearest[0][1] <= threshold_distance:
                hits += 1
        return hits / len(query_segments)

    # ------------------------------------------------------------------ #
    # 인덱스 재구성                                                        #
    # ------------------------------------------------------------------ #

    def _rebuild_index(self) -> None:
        """
        전체 세그먼트 임베딩으로 재귀 계층 인덱스 재구성.

        pivot 선택: 각 분기에서 가장 먼 두 세그먼트를 pivot으로 선택.
        max_radius 저장: subtree 내 중심과 가장 먼 세그먼트까지의 거리.
        """
        keys = list(self._embeddings.keys())
        if not keys:
            self._root = None
            self._dirty = False
            return
        self._root = self._build_node(keys)
        self._dirty = False

    def _build_node(self, keys: List[str]) -> _IndexNode:
        """재귀적 인덱스 노드 구성."""
        if len(keys) <= self._leaf_size:
            # 리프 노드
            embeddings = torch.stack([self._embeddings[k] for k in keys])
            center_emb = embeddings.mean(dim=0)
            # 임시 키를 center로 사용 (가장 가까운 실제 키 선택)
            dists = [self._distance(center_emb, self._embeddings[k]) for k in keys]
            center_key = keys[int(torch.tensor(dists).argmin().item())]
            max_radius = max(dists) if dists else 0.0
            return _IndexNode(
                center_key=center_key,
                center_embedding=center_emb,
                max_radius=max_radius,
                segment_keys=keys,
            )

        # 가장 먼 두 세그먼트를 pivot으로 선택
        pivot_a_key, pivot_b_key = self._select_pivots(keys)
        emb_a = self._embeddings[pivot_a_key]
        emb_b = self._embeddings[pivot_b_key]

        # 각 세그먼트를 더 가까운 pivot 쪽으로 분할
        left_keys: List[str] = []
        right_keys: List[str] = []
        for k in keys:
            d_a = self._distance(self._embeddings[k], emb_a)
            d_b = self._distance(self._embeddings[k], emb_b)
            if d_a <= d_b:
                left_keys.append(k)
            else:
                right_keys.append(k)

        # 균형 보장: 한쪽이 빈 경우 절반으로 분할
        if not left_keys or not right_keys:
            half = len(keys) // 2
            left_keys, right_keys = keys[:half], keys[half:]

        # 현재 노드의 중심 = 전체 평균
        all_embs = torch.stack([self._embeddings[k] for k in keys])
        center_emb = all_embs.mean(dim=0)
        dists_all = [self._distance(center_emb, self._embeddings[k]) for k in keys]
        center_key = keys[int(torch.tensor(dists_all).argmin().item())]
        max_radius = max(dists_all) if dists_all else 0.0

        node = _IndexNode(
            center_key=center_key,
            center_embedding=center_emb,
            max_radius=max_radius,
        )
        node.left = self._build_node(left_keys)
        node.right = self._build_node(right_keys)
        return node

    def _select_pivots(self, keys: List[str]) -> Tuple[str, str]:
        """가장 먼 두 세그먼트 키 반환 (O(N²) 샘플링, 최대 64개 샘플)."""
        sample = keys[:64]  # 대규모 집합에서 샘플링
        max_dist = -1.0
        pivot_a, pivot_b = sample[0], sample[-1]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                d = self._distance(self._embeddings[sample[i]], self._embeddings[sample[j]])
                if d > max_dist:
                    max_dist = d
                    pivot_a, pivot_b = sample[i], sample[j]
        return pivot_a, pivot_b

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """두 임베딩 사이의 거리 (코사인 또는 유클리드)."""
        a_f = a.float()
        b_f = b.float()
        if self._distance_fn == "cosine":
            cos_sim = torch.nn.functional.cosine_similarity(
                a_f.unsqueeze(0), b_f.unsqueeze(0)
            ).item()
            return 1.0 - cos_sim  # 거리 = 1 - 유사도
        else:  # euclidean
            return torch.norm(a_f - b_f).item()

    def _extract_embedding(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        """KV 텐서에서 임베딩 추출: 평균 Key 벡터 [d_embed]."""
        if kv_tensor.dim() == 1:
            return kv_tensor.float()
        return kv_tensor.float().mean(dim=0)

    def _get_backend_keys(self) -> List[str]:
        """백엔드 캐시의 현재 키 목록 (StaticDynamicSegmentCache 호환)."""
        backend = self._backend
        keys = []
        if hasattr(backend, "_static_store"):
            keys.extend(backend._static_store.keys())
        if hasattr(backend, "_dynamic_store"):
            keys.extend(backend._dynamic_store.keys())
        return keys
```

---

### 2. PPDAppendPrefillRouter (Activity A)

**스케줄링 결정 단위**: 요청(request) 단위로 Turn 1 / Turn 2+ 분류 후 D/P 노드 선택.

**캐시 상태 접근 방법**: `TriangleInequalitySegmentIndex.estimate_hit_probability()`로
D 노드 캐시 히트 가능성을 O(log N) 추정. 캐시를 직접 수정하지 않고 읽기 전용으로 조회.

```python
# src/scheduler/ppd_append_prefill_router.py 의사코드

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from src.cache.triangle_index import TriangleInequalitySegmentIndex


@dataclass
class PPDRoutingDecision:
    request_id: str
    turn: int                    # 1 = Turn 1, 2+ = Turn 2 이상
    node_type: str               # "P" (full prefill) 또는 "D" (append-prefill)
    hit_probability: float       # D 노드 캐시 히트 예상 확률
    threshold_used: float        # 라우팅 결정 시 사용된 임계값


class PPDAppendPrefillRouter:
    """
    PPD(arXiv 2603.13358) 기반 D 노드 append-prefill 동적 선택 라우터.

    스케줄링 결정 단위: 요청(request).
    캐시 상태 접근: TriangleInequalitySegmentIndex.estimate_hit_probability() (읽기 전용).

    이전 Activity A 기법과의 차별점:
    - PreemptiveKVOffloadScheduler: 실행 중 요청 선점 + GPU→CPU 전송 타이밍 결정.
    - PPDAppendPrefillRouter: P/D 역할 자체를 요청 유형과 세그먼트 히트율에 따라 동적 결정.
    """

    def __init__(
        self,
        segment_index: TriangleInequalitySegmentIndex,
        threshold_append: float = 0.7,       # D 노드 선택 히트 확률 임계값
        threshold_distance: float = 0.3,     # 히트 판정 최대 거리
        slo_ttft_budget_ms: float = 200.0,   # SLO TTFT 예산 (ms). 임박 시 임계값 하향.
        slo_aggressive_factor: float = 0.9,  # SLO 임박 시 threshold_append 승수
    ) -> None:
        self.segment_index = segment_index
        self.threshold_append = threshold_append
        self.threshold_distance = threshold_distance
        self.slo_ttft_budget_ms = slo_ttft_budget_ms
        self.slo_aggressive_factor = slo_aggressive_factor

        self._session_turns: Dict[str, int] = {}  # session_id → 현재 턴 수

    def route(
        self,
        request_id: str,
        session_id: str,
        input_segments: List[torch.Tensor],  # 이번 요청의 KV 세그먼트 목록
        remaining_ttft_ms: Optional[float] = None,  # 현재 TTFT 예산 잔여 (ms)
    ) -> PPDRoutingDecision:
        """
        단일 요청에 대한 P/D 노드 라우팅 결정.

        1. turn_count = session_turns.get(session_id, 0) + 1
        2. Turn 1 → P 노드 (full prefill, KV 초기 생성)
        3. Turn 2+:
           a. segment_index.estimate_hit_probability(input_segments) → hit_prob
           b. SLO 임박 여부 확인 → threshold 조정
           c. hit_prob > threshold_append → D 노드 append-prefill
           d. hit_prob ≤ threshold_append → P 노드 full prefill
        4. session_turns 갱신
        """
        turn = self._session_turns.get(session_id, 0) + 1
        self._session_turns[session_id] = turn

        # Turn 1: 항상 P 노드
        if turn == 1:
            return PPDRoutingDecision(
                request_id=request_id,
                turn=turn,
                node_type="P",
                hit_probability=0.0,
                threshold_used=self.threshold_append,
            )

        # Turn 2+: 히트 확률 추정
        hit_prob = self.segment_index.estimate_hit_probability(
            input_segments, threshold_distance=self.threshold_distance
        )

        # SLO 기반 임계값 조정: TTFT SLO 임박 시 더 공격적으로 D 노드 선택
        effective_threshold = self.threshold_append
        if remaining_ttft_ms is not None and remaining_ttft_ms < self.slo_ttft_budget_ms * 0.3:
            effective_threshold *= self.slo_aggressive_factor

        node_type = "D" if hit_prob > effective_threshold else "P"
        return PPDRoutingDecision(
            request_id=request_id,
            turn=turn,
            node_type=node_type,
            hit_probability=hit_prob,
            threshold_used=effective_threshold,
        )

    def reset_session(self, session_id: str) -> None:
        """세션 종료 시 턴 카운터 초기화."""
        self._session_turns.pop(session_id, None)
```

---

### 3. HitAwarePPDRouter (Activity A+B, Cross-1)

```python
# src/scheduler/hit_aware_ppd_router.py 의사코드

from typing import Dict, List, Optional, Tuple
import torch
from src.scheduler.ppd_append_prefill_router import PPDAppendPrefillRouter, PPDRoutingDecision
from src.cache.triangle_index import TriangleInequalitySegmentIndex


class HitAwarePPDRouter:
    """
    Cross-1 (A+B): PPDAppendPrefillRouter + TriangleInequalitySegmentIndex 통합.

    - 히트 확률 임계값 온라인 적응: 실제 D 노드 히트율 피드백으로 threshold_append EMA 갱신
    - Turn 1 / Turn 2+ 히트율 분리 측정
    - SemanticBoundarySegmentCache(B-2)와 통합: 의미 경계 청크 임베딩으로 히트 예측 정밀도 향상

    측정 지표:
    - d_node_ratio: Turn 2+ 중 D 노드 선택 비율
    - actual_hit_rate_d: D 노드에서 실제 세그먼트 히트 비율
    - threshold_history: threshold_append 온라인 적응 이력
    """

    def __init__(
        self,
        ppd_router: PPDAppendPrefillRouter,
        segment_index: TriangleInequalitySegmentIndex,
        ema_alpha: float = 0.1,          # 임계값 EMA 갱신 계수
        min_threshold: float = 0.3,      # threshold_append 하한
        max_threshold: float = 0.95,     # threshold_append 상한
        target_hit_rate: float = 0.7,    # 목표 D 노드 실제 히트율
    ) -> None:
        self.ppd_router = ppd_router
        self.segment_index = segment_index
        self.ema_alpha = ema_alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_hit_rate = target_hit_rate

        # 측정 통계
        self._turn1_count: int = 0
        self._turn2plus_count: int = 0
        self._d_node_count: int = 0
        self._d_node_actual_hits: int = 0
        self._threshold_history: List[float] = []

    def route(
        self,
        request_id: str,
        session_id: str,
        input_segments: List[torch.Tensor],
        remaining_ttft_ms: Optional[float] = None,
    ) -> PPDRoutingDecision:
        """라우팅 결정 후 통계 갱신."""
        decision = self.ppd_router.route(
            request_id, session_id, input_segments, remaining_ttft_ms
        )
        if decision.turn == 1:
            self._turn1_count += 1
        else:
            self._turn2plus_count += 1
            if decision.node_type == "D":
                self._d_node_count += 1
        return decision

    def record_actual_hit(self, request_id: str, was_hit: bool) -> None:
        """
        D 노드 처리 후 실제 히트 여부 피드백으로 threshold_append 온라인 적응.

        실제 히트율 < target_hit_rate → threshold_append 상향 (더 보수적)
        실제 히트율 ≥ target_hit_rate → threshold_append 하향 (더 공격적)
        EMA: threshold = (1-α)*threshold + α*(adjustment)
        """
        if was_hit:
            self._d_node_actual_hits += 1

        # 최소 10회 D 노드 처리 후 적응 시작
        if self._d_node_count >= 10:
            actual_hit_rate = self._d_node_actual_hits / self._d_node_count
            current = self.ppd_router.threshold_append
            if actual_hit_rate < self.target_hit_rate:
                # 히트율 낮음 → 임계값 상향 (D 노드 선택 줄임)
                new_threshold = current + self.ema_alpha * (current * 0.1)
            else:
                # 히트율 충분 → 임계값 하향 (D 노드 선택 늘림)
                new_threshold = current - self.ema_alpha * (current * 0.05)

            new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
            self.ppd_router.threshold_append = new_threshold
            self._threshold_history.append(new_threshold)

    def d_node_ratio(self) -> float:
        """Turn 2+ 중 D 노드 선택 비율."""
        if self._turn2plus_count == 0:
            return 0.0
        return self._d_node_count / self._turn2plus_count

    def actual_hit_rate_d(self) -> float:
        """D 노드에서 실제 히트 비율."""
        if self._d_node_count == 0:
            return 0.0
        return self._d_node_actual_hits / self._d_node_count
```

---

### 4. SemanticBoundarySegmentCache (Activity B)

```python
# src/cache/semantic_boundary_cache.py 의사코드

import re
from typing import Dict, List, Optional, Tuple
import torch
from src.cache.base import CacheStore


class SemanticBoundarySegmentCache(CacheStore):
    """
    SemantiCache(arXiv 2603.14303) 원리 기반 의미 경계 청킹 + GSC 클러스터링 + 비례 어텐션.

    - 의미 경계 탐지: 구분자(.!?\n\n + 코드 블록 ```) 기반 규칙
    - GSC(Greedy Seed-based Clustering): 어텐션 스코어 상위 시드 토큰 중심 탐욕 병합
    - 비례 어텐션: attention_weight_core = Σ(cluster attention weights) (합산, 평균 아님)

    CacheStore 인터페이스 완전 구현.
    TriangleInequalitySegmentIndex와 통합: 의미 코어 임베딩으로 인덱스 구성 시 정밀도 향상.
    """

    # 의미 경계 패턴 (컴파일 상수)
    _BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=\n\n)|(?<=```)")

    def __init__(
        self,
        capacity_bytes: int,
        min_cluster_size: int = 3,      # GSC 최소 클러스터 크기
        max_merge_ratio: float = 0.7,   # 최대 병합 비율 (세그먼트 내 토큰의 최대 70% 병합)
        attention_threshold: float = 0.1,  # 시드 토큰 어텐션 스코어 임계값
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.min_cluster_size = min_cluster_size
        self.max_merge_ratio = max_merge_ratio
        self.attention_threshold = attention_threshold

        self._store: Dict[str, torch.Tensor] = {}   # key → 의미 코어 KV 텐서
        self._lru_order: List[str] = []
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 구현                                           #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """의미 코어 KV만 저장 (GSC 병합 후 크기 감소)."""
        self._store[key] = value
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._lru_order.append(key)
        self._maybe_evict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._hit_count += 1
            self._lru_order.remove(key)
            self._lru_order.append(key)
            return self._store[key]
        self._miss_count += 1
        return None

    def evict(self) -> int:
        if not self._lru_order:
            return 0
        evict_key = self._lru_order.pop(0)
        kv = self._store.pop(evict_key, None)
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
    # 의미 경계 청킹 + GSC API                                            #
    # ------------------------------------------------------------------ #

    def detect_semantic_boundaries(self, text: str) -> List[int]:
        """
        텍스트에서 의미 경계 위치(토큰 인덱스 추정) 반환.
        구분자: 문장 끝(.!?), 단락(\n\n), 코드 블록(```)
        """
        positions = [0]
        for match in self._BOUNDARY_PATTERN.finditer(text):
            positions.append(match.start())
        return sorted(set(positions))

    def apply_gsc_clustering(
        self,
        kv_tensor: torch.Tensor,          # [n_tokens, d_head] KV 텐서
        attention_scores: torch.Tensor,   # [n_tokens] 어텐션 스코어
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GSC 클러스터링으로 의미 코어 KV 생성.

        1. 시드 토큰 선택: attention_scores > attention_threshold 이고 상위 (1 - max_merge_ratio) 비율
        2. 각 비-시드 토큰을 가장 가까운 시드 클러스터에 할당 (코사인 유사도 기준)
        3. 클러스터 병합: core_kv = Σ(attention_weight_t * kv_t) / Σ(attention_weight_t) for t in cluster
        4. 비례 어텐션: attention_weight_core = Σ(attention_weights in cluster) (합산)

        반환: (core_kv [n_cores, d_head], core_attention_weights [n_cores])
        """
        n_tokens, d_head = kv_tensor.shape
        max_seeds = max(1, int(n_tokens * (1 - self.max_merge_ratio)))

        # 시드 선택: 어텐션 스코어 상위 토큰
        _, seed_indices = torch.topk(attention_scores, k=max_seeds)
        seed_indices = seed_indices.sort().values

        seed_kv = kv_tensor[seed_indices]  # [n_seeds, d_head]
        seed_attn = attention_scores[seed_indices]

        # 각 토큰을 가장 가까운 시드에 할당
        cluster_sums = seed_kv.clone() * seed_attn.unsqueeze(-1)  # [n_seeds, d_head]
        cluster_attn_sums = seed_attn.clone()  # [n_seeds]
        cluster_counts = torch.ones(len(seed_indices), dtype=torch.int32)

        seed_set = set(seed_indices.tolist())
        for t in range(n_tokens):
            if t in seed_set:
                continue
            # 가장 가까운 시드 클러스터 찾기 (코사인 유사도)
            sims = torch.nn.functional.cosine_similarity(
                kv_tensor[t].unsqueeze(0), seed_kv
            )
            nearest_seed_idx = sims.argmax().item()

            # 클러스터 합산 (가중 평균용)
            cluster_sums[nearest_seed_idx] += kv_tensor[t] * attention_scores[t]
            # 비례 어텐션: 합산 (평균 아님)
            cluster_attn_sums[nearest_seed_idx] += attention_scores[t]
            cluster_counts[nearest_seed_idx] += 1

        # 의미 코어: 가중 평균
        core_kv = cluster_sums / cluster_attn_sums.unsqueeze(-1).clamp(min=1e-9)
        core_attn_weights = cluster_attn_sums  # 비례 어텐션 합산값

        return core_kv, core_attn_weights

    def put_with_gsc(
        self,
        key: str,
        kv_tensor: torch.Tensor,          # [n_tokens, d_head]
        attention_scores: torch.Tensor,   # [n_tokens]
    ) -> None:
        """GSC 클러스터링 후 의미 코어만 저장."""
        core_kv, _ = self.apply_gsc_clustering(kv_tensor, attention_scores)
        self.put(key, core_kv)

    def _maybe_evict(self) -> None:
        while self.memory_bytes() > self.capacity_bytes and self._lru_order:
            self.evict()
```

---

### 5. SpecKVCompressionGammaController (Activity C)

```python
# src/cache/speckv_gamma_controller.py 의사코드

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class _GammaMLP(nn.Module):
    """경량 MLP γ 선택기: 입력 차원 4 → 출력 6 (γ ∈ {1,2,3,4,5,6})."""

    def __init__(self) -> None:
        super().__init__()
        # 입력: [압축수준_onehot(3), min_confidence(1), max_entropy(1)] = 5차원
        # → 은닉층 16 → 출력 6 (γ logits)
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpecKVCompressionGammaController:
    """
    SpecKV(arXiv 2605.02888) 기반 압축 수준별 최적 추측 길이(γ) 자동 선택.

    - 입력: 압축 수준(FP16=0/INT8=1/NF4=2), min_draft_confidence, max_draft_entropy
    - 출력: γ ∈ {1, 2, 3, 4, 5, 6}
    - eOptShrinkQCodec(기구현) 압축 수준 출력을 자동으로 MLP 입력으로 주입
    - 온라인 적응: 검증 통과율 EMA 기반 γ 보정

    accuracy-preserving 근거:
    - 압축 수준 높을수록 γ 낮춤 → 검증 통과율 유지 → 정확도 보호
    - draft 신호(min_confidence, max_entropy)로 현재 배치의 최적 γ 실시간 추정
    """

    # 압축 수준 매핑
    COMPRESSION_FP16 = 0
    COMPRESSION_INT8 = 1
    COMPRESSION_NF4 = 2

    def __init__(
        self,
        base_seed: int = 42,
        ema_alpha: float = 0.05,           # 온라인 EMA 보정 계수
        target_verification_rate: float = 0.7,  # 목표 검증 통과율
    ) -> None:
        torch.manual_seed(base_seed)
        self._mlp = _GammaMLP()
        self._mlp.eval()

        self.ema_alpha = ema_alpha
        self.target_verification_rate = target_verification_rate

        # 온라인 적응 상태
        self._verification_history: List[bool] = []
        self._gamma_bias: float = 0.0  # γ 보정값 (양수 = 상향)

        # 프로파일링 데이터 버퍼 (MLP 학습용)
        self._profile_buffer: List[Tuple] = []  # [(compression_level, min_conf, max_ent, optimal_gamma), ...]

    def select_gamma(
        self,
        compression_level: int,          # COMPRESSION_FP16/INT8/NF4
        min_draft_confidence: float,     # draft 모델의 최소 토큰 신뢰도
        max_draft_entropy: float,        # draft 모델의 최대 토큰 엔트로피
    ) -> int:
        """
        MLP로 최적 γ 선택 (온라인 EMA 보정 포함).

        반환: γ ∈ {1, 2, 3, 4, 5, 6}
        """
        # 입력 특성 구성
        onehot = torch.zeros(3)
        onehot[compression_level] = 1.0
        x = torch.cat([onehot, torch.tensor([min_draft_confidence, max_draft_entropy])])

        with torch.no_grad():
            logits = self._mlp(x.unsqueeze(0)).squeeze(0)  # [6]
        gamma_idx = logits.argmax().item()
        gamma = int(gamma_idx) + 1  # {1,...,6}

        # EMA 보정 적용 (실제 검증 통과율 기반)
        gamma = max(1, min(6, gamma + round(self._gamma_bias)))
        return gamma

    def record_verification(self, was_accepted: bool) -> None:
        """
        추측 디코딩 검증 결과 피드백으로 온라인 γ 보정.

        통과율 < target → γ 하향 보정 (더 보수적)
        통과율 ≥ target → γ 상향 보정 (더 공격적)
        """
        self._verification_history.append(was_accepted)
        recent = self._verification_history[-50:]  # 최근 50회
        if len(recent) >= 10:
            actual_rate = sum(recent) / len(recent)
            if actual_rate < self.target_verification_rate:
                self._gamma_bias -= self.ema_alpha
            else:
                self._gamma_bias += self.ema_alpha
            self._gamma_bias = max(-2.0, min(2.0, self._gamma_bias))

    def integrate_with_eopt(
        self,
        eopt_codec,  # eOptShrinkQCodec 인스턴스
        min_draft_confidence: float,
        max_draft_entropy: float,
    ) -> int:
        """
        eOptShrinkQCodec(기구현) 압축 수준을 자동으로 MLP 입력으로 주입.

        eOptShrinkQCodec의 key_bits를 기준으로 압축 수준 결정:
        - key_bits ≥ 4: FP16 수준 (COMPRESSION_FP16)
        - key_bits == 3: INT8 수준 (COMPRESSION_INT8)
        - key_bits <= 2: NF4 수준 (COMPRESSION_NF4)
        """
        key_bits = getattr(eopt_codec, "key_bits", 3)
        if key_bits >= 4:
            compression_level = self.COMPRESSION_FP16
        elif key_bits == 3:
            compression_level = self.COMPRESSION_INT8
        else:
            compression_level = self.COMPRESSION_NF4
        return self.select_gamma(compression_level, min_draft_confidence, max_draft_entropy)

    def collect_profile_record(
        self,
        compression_level: int,
        min_draft_confidence: float,
        max_draft_entropy: float,
        optimal_gamma: int,
    ) -> None:
        """프로파일링 데이터 수집 (최소 512개 레코드 후 MLP 재학습 가능)."""
        self._profile_buffer.append(
            (compression_level, min_draft_confidence, max_draft_entropy, optimal_gamma)
        )

    def train_mlp_from_profile(self, epochs: int = 50) -> float:
        """
        수집된 프로파일링 데이터로 MLP 학습.
        최소 512개 레코드 필요. 반환: 최종 훈련 손실.
        """
        if len(self._profile_buffer) < 512:
            return float("inf")
        self._mlp.train()
        optimizer = torch.optim.Adam(self._mlp.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        final_loss = float("inf")
        for _ in range(epochs):
            total_loss = 0.0
            for rec in self._profile_buffer:
                comp_lvl, min_conf, max_ent, opt_gamma = rec
                onehot = torch.zeros(3)
                onehot[comp_lvl] = 1.0
                x = torch.cat([onehot, torch.tensor([min_conf, max_ent])]).unsqueeze(0)
                target = torch.tensor([opt_gamma - 1], dtype=torch.long)
                logits = self._mlp(x)
                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            final_loss = total_loss / max(len(self._profile_buffer), 1)
        self._mlp.eval()
        return final_loss
```

---

### 6. ContextIntensiveAccuracyGuard (Activity C)

```python
# src/cache/context_intensive_guard.py 의사코드

from typing import Dict, Optional, Tuple
import torch


class ContextIntensiveAccuracyGuard:
    """
    KV Cache Offloading for Context-Intensive Tasks(arXiv 2604.08426) 기반
    컨텍스트 정보 밀도 추정 + 압축 비율 자동 조정 게이트.

    - 고밀도 컨텍스트(≥0.7): 최대 4비트 압축 (정확도 보호)
    - 중간 밀도(0.4~0.7): 2.2~4비트 (정상 eOptShrinkQ 동작)
    - 저밀도(≤0.4): ≤2.2비트 공격적 압축 허용

    eOptShrinkQCodec(기구현) 및 ManifoldKVWindowedEviction(기구현)과의 게이트 인터페이스 구현.
    """

    def __init__(
        self,
        w1: float = 0.3,            # entity_ratio 가중치
        w2: float = 0.3,            # numeric_ratio 가중치
        w3: float = 0.4,            # token_entropy 가중치
        sample_tokens: int = 128,   # 밀도 추정 샘플 토큰 수 (첫 N 토큰)
        threshold_high: float = 0.7,    # 고밀도 임계값
        threshold_low: float = 0.4,     # 저밀도 임계값
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.sample_tokens = sample_tokens
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

    def assess(self, token_ids: torch.Tensor) -> float:
        """
        첫 sample_tokens 개 토큰 샘플링으로 컨텍스트 밀도 스코어 반환 [0.0, 1.0].

        context_density_score = w1 * entity_ratio + w2 * numeric_ratio + w3 * token_entropy

        entity_ratio: 어휘 ID 기준 희귀 토큰 비율 (ID > 50000 비율로 근사)
        numeric_ratio: 숫자·기호 토큰 비율 (ID 10~100 범위 비율로 근사)
        token_entropy: 토큰 ID 분포의 엔트로피 (정규화)
        """
        sample = token_ids[:self.sample_tokens].float()
        n = len(sample)
        if n == 0:
            return 0.5  # 기본값

        # entity_ratio: 희귀 어휘 토큰 비율 (고유 명사 근사)
        entity_ratio = (sample > 50000).float().mean().item()

        # numeric_ratio: 숫자·기호 범위 토큰 비율
        numeric_ratio = ((sample >= 10) & (sample <= 100)).float().mean().item()

        # token_entropy: 샘플 내 토큰 ID 분포 엔트로피 (정규화)
        token_counts = torch.bincount(sample.long().clamp(0, 100000), minlength=1)
        probs = token_counts.float() / n
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = torch.log(torch.tensor(float(n))).item()
        normalized_entropy = entropy / max(max_entropy, 1e-9)

        score = self.w1 * entity_ratio + self.w2 * numeric_ratio + self.w3 * normalized_entropy
        return float(min(1.0, max(0.0, score)))

    def get_compression_limits(
        self,
        density_score: float,
    ) -> Dict[str, float]:
        """
        밀도 스코어 기반 압축 파라미터 한도 반환.

        반환:
          {
            "min_bits": 하한 비트 수 (이 이상 비트로만 압축),
            "max_compression_ratio": 최대 압축 비율 (0~1, 높을수록 더 공격적),
            "density_level": "high" | "medium" | "low",
          }
        """
        if density_score >= self.threshold_high:
            return {
                "min_bits": 4.0,
                "max_compression_ratio": 0.5,
                "density_level": "high",
            }
        elif density_score >= self.threshold_low:
            return {
                "min_bits": 2.2,
                "max_compression_ratio": 0.75,
                "density_level": "medium",
            }
        else:
            return {
                "min_bits": 1.0,
                "max_compression_ratio": 0.9,
                "density_level": "low",
            }

    def gate_eopt_codec(
        self,
        eopt_codec,              # eOptShrinkQCodec 인스턴스
        token_ids: torch.Tensor,
    ) -> Dict:
        """
        eOptShrinkQCodec(기구현)에 밀도 기반 압축 파라미터 자동 주입.
        고밀도 컨텍스트에서 압축 비트 수를 4비트 이상으로 제한.
        반환: 적용된 파라미터 딕셔너리.
        """
        score = self.assess(token_ids)
        limits = self.get_compression_limits(score)
        original_key_bits = getattr(eopt_codec, "key_bits", 2)
        original_val_bits = getattr(eopt_codec, "value_bits", 3)

        min_bits = limits["min_bits"]
        new_key_bits = max(int(min_bits), original_key_bits)
        new_val_bits = max(int(min_bits), original_val_bits)

        # 파라미터 임시 조정 (컨텍스트별)
        applied = {
            "density_score": score,
            "density_level": limits["density_level"],
            "original_key_bits": original_key_bits,
            "original_val_bits": original_val_bits,
            "applied_key_bits": new_key_bits,
            "applied_val_bits": new_val_bits,
        }
        return applied
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(SpecKVCompressionGammaController + ContextIntensiveAccuracyGuard)를 포함하므로
다음 정확도 보존 검증이 **필수**이다 (evaluation_criteria.md §4 필수 항목).

### perplexity 측정

| 항목 | 상세 |
|------|------|
| 모델 | GPT-2 (small, 117M) — 추가 라이선스 없이 재현 가능 |
| 데이터셋 | WikiText-2 (test split, 표준 벤치마크) |
| 측정 방법 | stride=512, max_length=1024 sliding window perplexity |
| 허용 오차 | ±1% 이내 (베이스라인 대비 perplexity 변화) |
| 압축 설정 범위 | (a) 베이스라인(비압축), (b) SpecKVGamma + eOptShrinkQ 2.2비트, (c) ContextIntensiveGuard 고밀도(4비트) |
| 비교 대상 | 압축 수준별 γ=4 고정 vs SpecKVGammaController 자동 선택 γ |

### 태스크 정확도 측정

| 항목 | 상세 |
|------|------|
| 벤치마크 | LongBench 8개 서브태스크 (HotpotQA, 2WikiMultiHopQA, MuSiQue, GovReport, QMSum, MultiFieldQA-en, MultiFieldQA-zh, TriviaQA) |
| 추가 벤치마크 | **Text2JSON** (arXiv 2604.08426 컨텍스트-집약 태스크, ContextIntensiveAccuracyGuard 검증 필수) |
| 허용 오차 | ±1% 이내 |
| 측정 스크립트 | `experiments/run_text2json_accuracy.py` |

### accuracy-preserving 이론 근거

1. **SpecKV 압축-γ 결합 최적화**: 압축 수준 높아질수록 γ 낮춤 → 검증 통과율 유지 → 정확도 저하 방지 (SpecKV arXiv 2605.02888 실증: FP16/INT8/NF4 수준별 최적 γ 도출)
2. **ContextIntensiveAccuracyGuard 밀도 적응**: 고밀도 컨텍스트에서 4비트 하한 제한 → "정보가 많은 컨텍스트는 덜 압축"이라는 원칙으로 정확도 보호
3. **Text2JSON 검증**: arXiv 2604.08426이 기존 방법들의 컨텍스트-집약 태스크 정확도 저하를 체계적으로 문서화. ContextIntensiveAccuracyGuard가 이 저하를 방지하는지 실질적으로 검증.

### 검증 테스트 파일

**파일**: `tests/unit/test_speckv_gamma.py`

```python
# tests/unit/test_speckv_gamma.py 핵심 테스트 케이스

def test_high_compression_selects_lower_gamma():
    """NF4 압축(높은 압축)에서 γ가 FP16 대비 낮거나 같음을 검증."""
    ctrl = SpecKVCompressionGammaController(base_seed=42)
    gamma_fp16 = ctrl.select_gamma(SpecKVCompressionGammaController.COMPRESSION_FP16, 0.9, 0.1)
    gamma_nf4 = ctrl.select_gamma(SpecKVCompressionGammaController.COMPRESSION_NF4, 0.9, 0.1)
    assert gamma_nf4 <= gamma_fp16, f"NF4 γ({gamma_nf4}) > FP16 γ({gamma_fp16})"

def test_gamma_in_valid_range():
    """γ가 항상 {1,...,6} 범위임을 검증."""
    ctrl = SpecKVCompressionGammaController(base_seed=42)
    for comp_lvl in [0, 1, 2]:
        for min_conf in [0.1, 0.5, 0.9]:
            gamma = ctrl.select_gamma(comp_lvl, min_conf, 1.0 - min_conf)
            assert 1 <= gamma <= 6, f"γ={gamma} 범위 초과"

def test_online_adaptation_lowers_gamma_on_low_pass_rate():
    """검증 통과율 낮으면 γ 보정값이 하향 조정됨을 검증."""
    ctrl = SpecKVCompressionGammaController(base_seed=42, ema_alpha=0.2)
    for _ in range(20):
        ctrl.record_verification(was_accepted=False)  # 0% 통과율
    assert ctrl._gamma_bias < 0.0, f"γ_bias={ctrl._gamma_bias:.4f} (기대: 음수)"

def test_perplexity_proxy_within_tolerance():
    """압축 후 재구성 MSE 상대 오차 < 5% (perplexity ±1% proxy)."""
    from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec
    codec = eOptShrinkQCodec(num_layers=1, key_bits=3, value_bits=4)
    torch.manual_seed(42)
    codec.calibrate([torch.randn(64, 32) for _ in range(20)])
    kv_key = torch.randn(256, 32)
    kv_val = torch.randn(256, 32)
    compressed = codec.encode(kv_key, kv_val, layer_idx=0)
    key_approx, val_approx = codec.decode(compressed)
    mse_key = ((kv_key - key_approx) ** 2).mean() / (kv_key ** 2).mean()
    mse_val = ((kv_val - val_approx) ** 2).mean() / (kv_val ** 2).mean()
    assert mse_key.item() < 0.05
    assert mse_val.item() < 0.05
```

**파일**: `tests/unit/test_context_intensive_guard.py`

```python
# tests/unit/test_context_intensive_guard.py 핵심 테스트 케이스

def test_high_density_limits_compression_to_4bit():
    """고밀도 컨텍스트(score≥0.7)에서 min_bits ≥ 4.0임을 검증."""
    guard = ContextIntensiveAccuracyGuard()
    limits = guard.get_compression_limits(0.8)
    assert limits["min_bits"] >= 4.0
    assert limits["density_level"] == "high"

def test_low_density_allows_aggressive_compression():
    """저밀도 컨텍스트(score≤0.4)에서 min_bits ≤ 2.2임을 검증."""
    guard = ContextIntensiveAccuracyGuard()
    limits = guard.get_compression_limits(0.2)
    assert limits["min_bits"] <= 2.2
    assert limits["density_level"] == "low"

def test_assess_returns_valid_range():
    """assess() 결과가 [0.0, 1.0] 범위임을 검증."""
    guard = ContextIntensiveAccuracyGuard()
    token_ids = torch.randint(0, 100000, (200,))
    score = guard.assess(token_ids)
    assert 0.0 <= score <= 1.0

def test_gate_eopt_codec_raises_bits_in_high_density():
    """고밀도 컨텍스트에서 eOptShrinkQ의 key_bits가 4 이상으로 조정됨을 검증."""
    from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec
    guard = ContextIntensiveAccuracyGuard(threshold_high=0.0)  # 항상 고밀도
    codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
    token_ids = torch.randint(50001, 100000, (128,))  # 고밀도 토큰
    applied = guard.gate_eopt_codec(codec, token_ids)
    assert applied["applied_key_bits"] >= 4
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-09.yaml
experiment:
  date: "2026-05-09"
  activity: "A+B+C"
  description: >
    HitAwarePPDRouter (Cross-1: A+B): PPDAppendPrefillRouter + TriangleInequalitySegmentIndex +
    SemanticBoundarySegmentCache. 보조: SpecKVCompressionGammaController(C) +
    ContextIntensiveAccuracyGuard(C).

scheduler:
  type: "hit_aware_ppd_router"
  threshold_append: 0.7              # D 노드 선택 히트 확률 임계값
  threshold_distance: 0.3           # 히트 판정 최대 세그먼트 거리
  slo_ttft_budget_ms: 200.0         # TTFT SLO 예산 (ms)
  slo_aggressive_factor: 0.9        # SLO 임박 시 threshold_append 승수
  ema_alpha: 0.1                    # 임계값 EMA 갱신 계수
  min_threshold: 0.3                # threshold_append 하한
  max_threshold: 0.95               # threshold_append 상한
  target_hit_rate: 0.7              # 목표 D 노드 실제 히트율

index:
  type: "triangle_inequality"
  embedding_dim: 64                 # 세그먼트 임베딩 차원
  leaf_size: 8                      # 리프 노드 최대 세그먼트 수
  distance_fn: "cosine"             # "cosine" 또는 "euclidean"

cache:
  type: "semantic_boundary"
  capacity_bytes: 4294967296        # 4 GiB
  min_cluster_size: 3               # GSC 최소 클러스터 크기
  max_merge_ratio: 0.7              # 최대 병합 비율
  attention_threshold: 0.1          # 시드 토큰 어텐션 스코어 임계값

compression:
  gamma_controller:
    enabled: true
    base_seed: 42
    ema_alpha: 0.05
    target_verification_rate: 0.7
  context_guard:
    enabled: true
    w1: 0.3                         # entity_ratio 가중치
    w2: 0.3                         # numeric_ratio 가중치
    w3: 0.4                         # token_entropy 가중치
    sample_tokens: 128
    threshold_high: 0.7
    threshold_low: 0.4
  eopt_backend:
    key_bits: 2
    value_bits: 3
    calibration_samples: 20
    calibration_save_path: "results/2026-05-09/eopt_calibration.pt"

benchmark:
  index_speed:
    segment_counts: [100, 1000, 10000]
    trials: 10
  ppd_ttft:
    num_turns: 5                    # 멀티턴 시뮬레이션 턴 수
    num_sessions: 100
  accuracy:
    datasets: ["wikitext2", "longbench", "text2json"]
    wikitext2_stride: 512
    wikitext2_max_length: 1024

seed: 42
results_dir: "results/2026-05-09"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_triangle_index_search.py`
  - `test_search_returns_nearest_segment`: 알려진 최근접 세그먼트를 정확히 반환
  - `test_triangle_inequality_pruning_reduces_nodes`: 가지치기로 탐색 노드 수가 전체의 50% 미만
  - `test_search_speed_olog_n_vs_linear`: N=100/1K/10K에서 삼각인덱스 검색이 선형 검색보다 빠름
  - `test_rebuild_on_dirty`: dirty 상태에서 자동 인덱스 재구성 검증
  - `test_estimate_hit_probability_range`: estimate_hit_probability() 결과 [0.0, 1.0] 검증
  - `test_cachestore_interface`: CacheStore 추상 메서드 전부 구현 검증
  - `test_put_get_evict_roundtrip`: put/get/evict 기본 동작 검증
  - `test_evict_removes_from_embeddings`: evict() 후 임베딩 저장소에서도 제거됨 검증

- [ ] `tests/unit/test_ppd_router.py`
  - `test_turn1_always_routes_to_p_node`: Turn 1 요청은 항상 P 노드
  - `test_turn2_high_hit_prob_routes_to_d_node`: hit_prob > threshold → D 노드
  - `test_turn2_low_hit_prob_routes_to_p_node`: hit_prob ≤ threshold → P 노드
  - `test_slo_aggressive_lowers_threshold`: SLO 임박 시 threshold 하향 조정
  - `test_session_turn_counter_increments`: 동일 session_id에서 turn 카운터 증가
  - `test_reset_session_clears_counter`: reset_session() 후 카운터 초기화

- [ ] `tests/unit/test_semantic_boundary_cache.py`
  - `test_detect_sentence_boundaries`: 문장 끝(.) 경계 탐지 검증
  - `test_detect_paragraph_boundaries`: 단락(\n\n) 경계 탐지 검증
  - `test_gsc_clustering_reduces_tokens`: GSC 후 토큰 수 감소 (max_merge_ratio 적용)
  - `test_proportional_attention_is_sum_not_mean`: 비례 어텐션이 합산임을 검증
  - `test_put_with_gsc_stores_core_kv`: put_with_gsc() 후 저장된 KV가 원본보다 작음
  - `test_cachestore_interface`: CacheStore 추상 메서드 전부 구현 검증
  - `test_accuracy_delta_gsc_cosine_similarity`: GSC 병합 후 코사인 유사도 ≥ 0.85
    (WikiText-2 perplexity ±1% proxy)

- [ ] `tests/unit/test_speckv_gamma.py`
  - `test_high_compression_selects_lower_gamma`: NF4 γ ≤ FP16 γ (위 의사코드 참조)
  - `test_gamma_in_valid_range`: γ ∈ {1,...,6} (위 의사코드 참조)
  - `test_online_adaptation_lowers_gamma_on_low_pass_rate`: 검증 0% 시 γ_bias 하향 (위 의사코드 참조)
  - `test_perplexity_proxy_within_tolerance`: MSE 상대 오차 < 5% (위 의사코드 참조)
  - `test_integrate_with_eopt_selects_nf4_for_2bit`: key_bits=2이면 NF4 수준 선택
  - `test_train_mlp_requires_512_records`: 512개 미만 레코드에서 inf 반환

- [ ] `tests/unit/test_context_intensive_guard.py`
  - `test_high_density_limits_compression_to_4bit`: (위 의사코드 참조)
  - `test_low_density_allows_aggressive_compression`: (위 의사코드 참조)
  - `test_assess_returns_valid_range`: (위 의사코드 참조)
  - `test_gate_eopt_codec_raises_bits_in_high_density`: (위 의사코드 참조)
  - `test_medium_density_allows_2_2_bits`: 중간 밀도에서 min_bits ≤ 2.2 허용
  - `test_density_score_components_weighted`: 가중치 w1+w2+w3=1.0 검증

- [ ] `tests/integration/test_cross_ab_ppd_index.py`
  - `test_hit_aware_router_e2e_multiturn`: 5턴 대화 시뮬레이션 E2E 파이프라인
  - `test_d_node_ratio_increases_with_turns`: 턴 증가 시 D 노드 선택 비율 증가
  - `test_online_threshold_adaptation`: record_actual_hit() 후 threshold 변화 확인
  - `test_semantic_boundary_cache_feeds_index`: SemanticBoundarySegmentCache 의미 코어가 인덱스에 등록됨
  - `test_cross_ab_hit_rate_above_30_percent`: 비연속 히트율 ≥ 30% (evaluation_criteria.md §3)
  - `test_cross_ab_throughput_improvement`: 복합 처리량 단일 Activity 대비 +5% 이상 (evaluation_criteria.md §5)

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 전부 통과** (100%, evaluation_criteria.md §0)
2. **통합 테스트 전부 통과** (100%, evaluation_criteria.md §0)
3. **CacheStore 인터페이스 준수** — `TriangleInequalitySegmentIndex`, `SemanticBoundarySegmentCache` 각각 모든 추상 메서드 구현 (evaluation_criteria.md §0)
4. **시드 고정 재현성** — `seed: 42` 설정으로 동일 결과 재현 (evaluation_criteria.md §0)
5. **Activity C Accuracy 보존 필수** — `test_speckv_gamma.py` 전부 통과 + `experiments/run_text2json_accuracy.py` 실행 시 perplexity 변화 ±1% 이내 (evaluation_criteria.md §4 필수)
6. **Activity C 태스크 정확도 필수** — LongBench 8개 서브태스크 + **Text2JSON** 정확도 변화 ±1% 이내 (evaluation_criteria.md §4 필수)
7. **Activity A 스케줄링 오버헤드** — TTFT p50 증가 +5% 이내 (정상 부하, evaluation_criteria.md §2 필수)
8. **Turn 2+ TTFT 개선** — Turn 2+ TTFT p50 −68% (D 노드 append-prefill, evaluation_criteria.md §2)
9. **캐시 히트율 향상** — 스케줄링 미적용 대비 히트율 +10%p 이상 (evaluation_criteria.md §2)
10. **비연속 히트율** — 전체 히트 중 비연속 히트 비율 ≥ 30% (evaluation_criteria.md §3 높음)
11. **인덱스 검색 속도** — N=10K에서 O(N) 대비 3.6× 이상 향상 (`experiments/run_index_speed_benchmark.py`)
12. **처리량 목표** — tokens/sec 베이스라인 대비 +20% 이상 (evaluation_criteria.md §1)
13. **복합 처리량 향상** — 단일 Activity 대비 추가 +5% 이상 (evaluation_criteria.md §5)
14. **복합 메모리 감소** — 단일 Activity 대비 추가 −10% 이상 (evaluation_criteria.md §5)
15. **타입 힌트** — 모든 공개 함수·메서드에 존재 (evaluation_criteria.md §0)
16. **설정 YAML 존재** — `configs/experiments/2026-05-09.yaml` 생성됨 (evaluation_criteria.md §0)
17. **기존 파일 회귀 없음** — 수정 금지 파일 목록의 모든 기존 테스트 통과
18. **결과 기록** — `results/2026-05-09/metrics.json` 에 목표 지표 수치 기록
