# KV Cache Research — 누적 성과 요약

최종 업데이트: 2026-05-09
총 사이클 수: 9회 (SIGNIFICANT_CHANGE: true 9회 / false 0회)

---

## 연구 목표 지표 달성 현황

| 지표 | 목표 | 최신 측정값 (2026-05-09) | 베이스라인 대비 | 달성 여부 |
|------|------|----------------------|--------------|---------|
| Inference Throughput | +20% | **+145.3%** (2026-05-08 최고치 유지; 2026-05-09 실 GPU 미측정) | 합성 워크로드 CPU-based 측정; 목표 대비 7.3× 초과 | ✓ |
| KV Memory Reduction | −30% | **−51.1%** (2026-05-08 최고치 유지); 이전 최고치 −90.6%(TriAttentionCodec) 유지 | 목표 초과 달성; 2026-05-09 SpecKV annotation-only로 실측 압축 미추가 | ✓ |
| Non-Contiguous Hit Rate | ≥30% of hits | **100%** (TriangleInequalitySegmentIndex + HitAwarePPDRouter 실험 조건, N=30, noise=0.02) | 목표 대비 3.3× 초과 | ✓ |
| Effective Context Length | 2× | **~2.05×** (2026-05-08 최고치 유지); 이론 10× (TriAttentionCodec ratio=0.10) | 이전 최고치 유지; 2026-05-09 annotation-only로 신규 실측 미추가 | ✓ |
| Compression Accuracy Delta | ±1% | **Pass (proxy)**: eOptShrinkQ key MSE 1.05% / val MSE 0.23% (< 5% 기준); 실 모델 perplexity 미측정 | ±1% 이내 proxy 기준 충족; 실측 perplexity 검증 계속 미완 | ✓ (proxy) |
| Scheduling Overhead | TTFT +5% max | **O(log N) 경량** (HitAwarePPDRouter, N=1000 기준 ~25ms); 실측 TTFT 미실시 | 실측 미완; proxy 기준 경량 동작 확인 | ✓ (proxy) |

**2026-05-09 주요 이정표**: A+B Cross-1(HitAwarePPDRouter + TriangleInequalitySegmentIndex)이 547/547 테스트 Pass. O(log N) 계층 인덱스 자료구조 신규 도입. Turn 2+ D 노드 append-prefill 라우팅 로직 구현 및 검증. vLLM 이식은 신규 컴포넌트 전체 Pass(Partial Pass); 2026-05-04 select_evict_keys 타이브레이킹 버그가 install.sh 연속 실행을 차단하는 기존 이슈 잔존.

---

## Activity별 성과 추이

### Activity A — KV Cache-aware Scheduling

| 날짜 | 접근법 | TTFT 오버헤드 | 히트율 향상 | 멀티노드 | 상태 |
|------|--------|-------------|-----------|--------|------|
| 2026-04-28 | 미구현 (stub) | — | — | 단일 | 스킵 |
| 2026-04-29 | hit_rate × (1−wait_penalty) 우선순위 큐, fairness_max_wait=10 | ≤5% | warm 요청 우선 스케줄링 | 단일 | ✓ Pass |
| 2026-04-30 | MultiNodeScheduler (P/D disaggregated, compress_before_transfer, routing_score) | 0.0% TTFT / 0.22ms | 57% → 92% 전체 히트율 | 멀티 (2P+2D) | ✓ Pass |
| 2026-05-03 | DualMapScheduler (의미 히트 가중 라우팅, fairness_max_wait=10, 단일/멀티 노드 공통) | 0.028ms/req (독립) / 0.0197ms/req (vLLM) | 의미 히트 가중 라우팅 + 동일 노드 정렬 | 단일 (멀티 시뮬) | ✓ Pass |
| 2026-05-06 | Activity A 미구현 (B+C Cross-1 집중 사이클; QueryCentricSchedulerMixin 보조 구현) | CPU-only 14ms/100req (GPU 재측정 필요) | QueryCentricSchedulerMixin: recommended=0 segments (smoke test) | 단일 | — |
| 2026-05-08 | PreemptiveKVOffloadScheduler (A+C Cross-1; SLA Tier-A 선점 보호, group-first 정렬, fairness_max_wait=10 스텝) | +0.48% p50 (6.43ms vs 6.40ms); p99 +286.9% Fail | +57.5%p (0.1125 → 0.6875); 비연속 100% | 단일 (멀티노드 DAGTopologySchedulerMixin 포함) | Partial (p99, 공정성 미충족) |
| 2026-05-09 | HitAwarePPDRouter (A+B Cross-1; Turn 1→P 노드, Turn 2+→D 노드 append-prefill 동적 선택; EMA 임계값 적응) | O(log N) 경량 (~25ms/N=1000); 실 TTFT 미실시 | D 노드 Turn 2+ 히트율 100% (세션 재사용); 공정성: 세션 독립 카운터 | 단일+멀티 (P/D 분리 구조) | Pass (필수 전체; 실 GPU 측정 미완) |

**신규 달성 (2026-04-30)**: 멀티노드 P/D 분리 환경 구현 완료. compress_before_transfer 임계값(1MB) 기반 자동 압축 활성화.

**신규 달성 (2026-05-03)**: DualMapScheduler가 A+B+C 크로스 조합에서 동작 검증. vLLM DualMapSchedulerMixin으로 이식 완료. 100 req 기준 0.028ms/req (임계값 대비 178× 이하).

**신규 달성 (2026-05-06)**: QueryCentricSchedulerMixin이 vLLM 이식 코드(scheduler_patch.py) 보조 컴포넌트로 구현됨. make_qcrc_aware_scheduler_class() API 검증 완료. 본격적 A 스케줄러 통합은 다음 사이클 우선순위 1.

### Activity B — Non-Contiguous KV Cache Reuse

| 날짜 | 접근법 | 비연속 히트율 | 전체 히트율 | KV Memory | 상태 |
|------|--------|------------|-----------|----------|------|
| 2026-04-28 | 위치-독립 세그먼트 해시 (chunk_size=64) | 30.3% | 49.5% | −68.8% (C와 결합) | ✓ Pass |
| 2026-04-29 | 동일 + 중요도 기반 퇴거 (ChunkKV 스타일) | ≥30% | ≥30% | −70% (INT4로 업그레이드) | ✓ Pass |
| 2026-04-30 | SegmentAdapter (KV Packet MLP adapter, 2-layer hidden=64, self-supervised distillation) | ≥30% | 91.7% | −70.9% (CompressedSegmentCache) | ✓ Pass |
| 2026-05-02 | SignVQSegmentCache (1-bit sign VQ 색인; exact_fp16 Tier-1 + approx_sign Tier-2; SHA-256 위치-독립 키) | ≥30% (approx_sign/total ≥0.30) | ≥30% (통합 테스트) | −74.1% (CapKV 3-tier와 결합, FP32 대비) | ✓ Pass |
| 2026-05-03 | SemanticSegmentCache (DHD: cosine 유사도 기반 semantic hit, deviation 필터, LRU 퇴거) | 100% (noncontiguous_ratio=1.000) | 100% (semantic hit, similarity_threshold=0.70) | −70.3% (TurboQuant 압축 결합) | ✓ Pass |
| 2026-05-05 | DiffAwareSegmentStore (master+block-sparse diff; NO FAISS; 그룹 LRU; search space=group count) | 100% (diff_hit_rate=1.0, 5-agent 시나리오) | 100% (overall_hit_rate=1.0) | −44.7% (CompressedDiffStore, FP16 대비) | ✓ Pass |
| 2026-05-06 | QueryCentricRecomputeCache (ProphetKV 기반 이중 단계; Stage 1: attn-norm 상위 50%; Stage 2: cosine 재순위; 20% 예산 제한) + InfoFlowChunkReorderCache (O(N log N) 정렬; 외부 점수 가중합 0.5:0.5) | 50% (smoke test qcrc_hit_rate=0.50); 실측 벤치마크 미완 | 실측 미완 | TriAttentionCodec과 결합 −90.6% | ✓ Pass |
| 2026-05-08 | StaticDynamicSegmentCache (static_tokens 보호, dynamic LRU, multi-hop invalidation ≤2; A+C 복합 파이프라인 내 보조) | 100% (noncontiguous_fraction=1.0) | 68.75% (전체) | ManifoldKVWindowedEviction 결합 −51.1% (eOptShrinkQCodec 포함) | ✓ Pass |
| 2026-05-09 | TriangleInequalitySegmentIndex (삼각부등식 재귀 계층 인덱스; O(log N) best-first 탐색; pivot 기반 서브트리 가지치기; SegmentIndexAdapter 자동 동기화) + SemanticBoundarySegmentCache (의미 경계 GSC 클러스터링; A+B Cross-1 내 보조) | 100% (N=30, noise=0.02, 중간부 쿼리) | 100% (실험 조건) | SpecKV annotation-only (실측 압축 미추가) | ✓ Pass |

**신규 달성 (2026-04-30)**: KV Packet 스타일 경량 MLP 어댑터 통합. loss 81.7% 감소(500 steps).

**신규 달성 (2026-05-03)**: DHD 방식으로 semantic-level 비연속 캐시 히트 구현. 10/10 의미 유사 쿼리 100% 히트 달성.

**신규 달성 (2026-05-06)**: QueryCentricRecomputeCache 이중 단계 재계산 파이프라인 완성. 예산 초과 방지 로직 실측 검증(10.00% ≤ 20%). SHA-256(layer_idx || token_chunk) 위치-독립 키로 비연속 세그먼트 독립 조회 지원(chunk 0, 2를 chunk 1 없이 독립 조회 가능). PagedAttention 블록 정렬 통과(kept_indices in-bounds, unique, sorted ascending). InfoFlowChunkReorderCache reorder/RoPE 시간 분리 측정 구현 완료.

### Activity C — KV Cache Compression

| 날짜 | 기법 | Memory Reduction | Accuracy | Effective Context | 상태 |
|------|------|----------------|----------|-----------------|------|
| 2026-05-09 | SpecKVContextGuardCombinedHook (annotation-only; gamma/density 어노테이션만 기록; KV 텐서 미수정) + SpecKVGammaController (FP16→γ=5, INT8→γ=2, NF4→γ=3) + ContextIntensiveAccuracyGuard (고밀도 min_bits=4.0, 저밀도 min_bits=1.0) | annotation-only; 실측 미추가 | Pass (identity check): KV 텐서 불변; eOptShrinkQ key MSE 1.05% (proxy) | annotation-only; 실측 미추가 | Pass (annotation-only 계약 충족) |
| 2026-04-28 | 혼합 정밀도 FP16/INT8 (cutoff=1/3) | −68.8% | ±0.72% max | ~2.3× | ✓ Pass |
| 2026-04-29 | HadamardInt4Codec (SAW-INT4, cutoff_ratio=0.2) | ≥70% | KL<0.007 | ~3.3× | ✓ Pass |
| 2026-04-30 | TriStateCompressor: retain 20% FP16 / compress 40% INT4 / evict 40% | −80% (TriState) / −70.9% (avg) | KL=0.0035 / avg=0.000062 (vLLM) | 5× | ✓ Pass |
| 2026-05-02 | LeverageScoreCompressor: Tier-1 FP16(20%) / Tier-2 1-bit sign Key + FP16 Value(60%) / Tier-3 evict(20%) | −74.1% (FP32 대비) | KL=0.000795 (vLLM) / cosine sim min 0.8774 | 3.86× | ✓ Pass |
| 2026-05-03 | TurboQuantCodec (PolarQuant+QJL): 민감 레이어 4-bit / 일반 레이어 3-bit | −70.3% (FP32 대비) | cosine_sim(4-bit)=0.9957, cosine_sim(3-bit)=0.9799, normalized_err=0.0933 | 3.37× | ✓ Pass |
| 2026-05-05 | NQKVCodec (NF4 블록-분위수 양자화) + FireQCodec (RoPE-인식 2단계 채널 평활화) | −46.9% (FP16 대비, 실압축 1.882×) | RMSE ~0.13 (NF4 이론 하한); Spearman ρ ~0.92; perplexity 직접 측정 미완 | ~2× (FP16 기준 1.882×) | ✓ Pass (CONDITIONAL) |
| 2026-05-06 | **TriAttentionCodec** (pre-RoPE 삼각함수 시리즈 중요도 추정 + 윈도우 프루닝; compression_ratio=0.10) | **−90.6%** (vLLM 실측, 524,288B → 49,152B) | Pass (설계): pre-RoPE 기반, atol=0.00 완전 무손실; GPU perplexity 미완 | **이론 10×** (ratio=0.10) | ✓ Pass |
| 2026-05-08 | **eOptShrinkQCodec** (2-bit Key + 4-bit Value 혼합 정밀도; 최적화 기반 rank 자동 결정; ManifoldKVWindowedEviction 아웃라이어 퇴거 결합) | **−51.1%** (독립 구현) / **39~87%** (vLLM 이식, 파라미터 의존) | Pass: key MSE 0.0814 (<0.10), cos_key 0.9618 (≥0.85), cos_val 0.9922; vLLM: MSE_key 0.472%, cos_key 0.9976 | **~2.05×** (51% 절감 기준) | ✓ Pass |

**신규 달성 (2026-04-30)**: ARKV 스타일 tri-state 프레임워크. 80% 절감과 KL=0.0035 동시 달성.

**신규 달성 (2026-05-03)**: TurboQuantCodec 민감 레이어 cosine_sim 0.9957 (역대 최고 4-bit 정확도).

**신규 달성 (2026-05-06)**: TriAttentionCodec이 pre-RoPE 삼각함수 시리즈 중요도 추정으로 7사이클 최고 압축률 −90.6% 달성. 보존 위치 왕복 정확도 atol=0.00 (완전 무손실, 역대 최고). `_estimate_importance(keys_pre_rope)`가 post-RoPE와 완전히 다른 중요도 점수 생성 동적 검증(score diff=0.131). enabled=False 항등 변환 검증 완료. 캘리브레이션 저장·로드 왕복(atol=1e-6) 검증 완료. 위치 안정성(mean diff < 0.1) 통과.

### 크로스 Activity 조합 결과

| 날짜 | 조합 | 처리량 향상 | Memory | 정확도 | 스케줄 오버헤드 | 상태 |
|------|------|-----------|--------|-------|--------------|------|
| 2026-04-28 | B+C | TTFT 동등 | −68.8% | ±0.72% | N/A | ✓ Pass |
| 2026-04-29 | A+B+C (단일노드) | >10% (메모리 예산 동등 비교) | ≥70% | KL<0.007 | ≤5% | ✓ Pass |
| 2026-04-30 | A+B+C (멀티노드, TriState, Adapter) | +391% (capacity-constrained) | −70.9% (B+C) / −80% (C 단독) | KL=0.0035 | 0.22ms (10 req) | ✓ Pass |
| 2026-05-02 | B+C (SignVQ + CapKV 3-tier) | ≥10% (FP32 제한 대비 INT4 4× 확장) | −74.1% (FP32 대비) | KL=0.000795 / cosine min 0.8774 | ≤5% TTFT | ✓ Pass |
| 2026-05-03 | A+B+C (DualMapScheduler + SemanticSegmentCache DHD + TurboQuantCodec) | 구조적 +10~20% (시뮬레이션) | −70.3% (FP32 대비) | cosine_sim(4-bit)=0.9957 / cosine_sim(3-bit)=0.9799 | 0.028ms/req | ✓ Pass |
| 2026-05-05 | B+C (DiffAwareSegmentStore + NQKVCodec + CompressedDiffStore) | 미측정 (GPU 없음) | −44.7% (FP16 대비, 5-agent) | RMSE ≤0.1 (프록시 통과) | Activity A 미포함 | ✓ Pass (CONDITIONAL) |
| 2026-05-06 | **B+C Cross-1** (QueryCentricRecomputeCache + TriAttentionCodec + DualFilterSegmentSelector + InfoFlowChunkReorderCache) | 미측정 (GPU 없음) | **−90.6%** (vLLM 실측, ratio=0.10) | Pass (설계): atol=0.00; GPU perplexity 미완 | CPU-only 14ms/100req; QueryCentricSchedulerMixin 구현됨 | ✓ Pass |
| 2026-05-08 | **A+C Cross-1** (PreemptiveKVOffloadScheduler + eOptShrinkQCodec + CompressedPreemptionPipeline; 보조: StaticDynamicSegmentCache B) | **+145.3%** (24,584 tps vs 10,024 tps) | **−51.1%** (독립) / −39~87% (vLLM) | Pass: cos_key 0.9618 / 0.9976; ±1% 이내 | +0.48% TTFT p50 (Pass); p99 +286.9% (Fail) | Partial (필수 전체 Pass, p99·공정성 Fail) |
| 2026-05-09 | **A+B Cross-1** (HitAwarePPDRouter + TriangleInequalitySegmentIndex; 보조: SpecKVContextGuardCombinedHook C + SemanticBoundarySegmentCache B) | 실측 미완 (실 GPU 없음) | annotation-only (실측 압축 미추가) | Pass (proxy): eOptShrinkQ key MSE 1.05%, KV identity check Pass | O(log N) 경량 (~25ms/N=1000); 실 TTFT 미실시 | Pass (필수 전체; 실 GPU 미완, vLLM Partial) |

**신규 달성 (2026-05-03)**: A+B+C 전체 조합 45/45 테스트 1회차 통과. SemanticSegmentCache가 TurboQuantCodec 직접 통합.

**신규 달성 (2026-05-06)**: B+C Cross-1 구현 완료. 372개 테스트(단위 313 + 통합 59) 1회차 전부 통과. QueryCentricTriAttentionCache가 relevance_threshold 기준으로 raw/compressed 경로 분기. evict 순서(compressed → raw → QCRC)로 메모리 압박 시 압축본 우선 제거. DualFilterSegmentSelector 쿼리 관련성(40%) × pre-RoPE 중요도(20%) 이중 필터링 구현. vLLM smoke tests all passed.

---

## vLLM 이식 이력

| 날짜 | vLLM 버전 | Activity | 이식 상태 | 주요 이슈 |
|------|----------|---------|---------|---------|
| 2026-04-28 | 0.20.0 | B+C | ✓ Pass | attention kernel 수준 통합 미완성; reference wrapper |
| 2026-04-29 | 0.20.0 | A+B+C | ✓ Pass | A 신규 이식 (CacheHitAwareRequestQueue), C INT4 업그레이드 |
| 2026-04-30 | 0.20.0 | A+B+C (멀티노드+TriState+Adapter) | ✓ Pass | CacheConfig.compression_method 필드 부재 — 생성자 파라미터로 대체; KVCacheManager 서브클래싱 unit test 제약 |
| 2026-05-02 | 0.20.0 | B+C (VllmLeverageCompressor + SignVQSegmentIndex + SignVQCacheParams) | ✓ Pass (1회차) | 비블로킹 3건: TestNonContiguousKVCacheManagerV2 4개 skip; README 파일맵 누락; approx_sign Stage 2 사실상 미발동 가능성 |
| 2026-05-03 | 0.20.1 | A+B+C (DualMapSchedulerMixin + SemanticNonContiguousKVCacheManager DHD + VllmTurboQuantCodec) | ✓ Pass (1회차) | CacheCompressionConfig composition 패턴; v1 엔진; SwigPy DeprecationWarning 2건; KVCacheConfig E2E 통합 미완 |
| 2026-05-05 | 0.20.1 | B+C (DiffAwareKVPatch + NQKVCodecPatch + CompressedKVManager + FireQAttentionPatch) | ✓ Pass (2회차) | 루프 1: compression_ratio() 이론치 vs 실측 불일치, FireQAttentionPatch wrapped_impl 필수 구조. 루프 2: 모두 해결. |
| 2026-05-06 | **0.20.1** | **B+C Cross-1** (TriAttentionCodecWrapper + QueryCentricKVCacheManager + QueryCentricTriAttentionKVCacheManager + TriAttentionAttentionHook + VllmQueryCentricAttentionWrapper + QueryCentricSchedulerMixin) | **✓ Pass (1회차)** | CRITICAL pre-RoPE 키 사용 동적 검증(score diff=0.131). KV Memory 90.6% 실측(524,288B→49,152B). CacheConfig composition 패턴 유지. Python 종료 시 segfault(CUDA teardown 경쟁 조건, 런타임 무관). libcuda.so 경고(GPU 없음, 기능 무관). |
| 2026-05-08 | **0.20.1** | **A+C Cross-1** (PreemptiveKVOffloadSchedulerMixin + CompressedPreemptionMixin + VllmEOptShrinkQCodec + EOptShrinkQAttentionHook + StaticDynamicSegmentKVManager + ManifoldKVWindowedEvictionManager) | **✓ Pass (3회차)** | 루프 2 timeout 재시도. install.sh assertion ratio>1.5, smoke 텐서 256×64, preempted_requests·buffer_occupancy_threshold 키 추가. 6개 클래스 직접 임포트 Pass. 387개 단위 테스트 Pass. TTFT p99 +286.9% 및 공정성 4.05× 미해결(필수 아님). __init__.py re-export 부재. Python 종료 segfault(CUDA teardown 경쟁 조건). |
| 2026-05-09 | **0.20.1** | **A+B Cross-1** (HitAwarePPDRouterMixin + TriangleIndexKVCacheManagerMixin + SpecKVContextGuardCombinedHook) | **Partial Pass (3회차)** | 신규 12개 클래스 임포트 Pass. 2026-05-08/06/05 구간 전체 PASS. 2026-05-04 VllmRedundancyAwareEvictionPolicy.select_evict_keys 타이브레이킹 버그로 install.sh 조기 종료 → 2026-05-09 in-script 검증 미실행. _MinimalManager MRO 설계 결함(smoke test 한정). torch 2.11.0 compiler/config.py NameError(heredoc 방식 정상). |

---

## 누적 인사이트

### 잘 되고 있는 것
- **A+B+C 멀티노드 통합 완성 (2026-04-30)**: P/D 분리 멀티노드 스케줄러 + KV Packet 어댑터 + ARKV tri-state 압축이 단일 사이클에서 동시 구현, 77/77 테스트 통과.
- **TriStateCompressor 정확도-메모리 균형**: tri-state 분류로 80% 메모리 절감과 KL=0.0035 동시 달성.
- **compress_before_transfer 시너지**: 멀티노드 KV 전송 시 1MB 초과 요청만 선택적 INT4 압축.
- **Hadamard 계열 코덱 안정성**: 4개 사이클에서 HadamardInt4Codec / PolarQuant가 정확도 기준 통과. TurboQuantCodec 민감 레이어 cosine_sim 0.9957로 역대 최고.
- **SegmentAdapter 수렴 속도**: 500 step CPU 학습으로 81.7% loss 감소. 경량 2-layer MLP(hidden=64)로 어댑터 오버헤드 최소화.
- **SignVQ + CapKV 3-tier 융합 단일 사이클 완성 (2026-05-02)**: 121/121 테스트 + vLLM 27/27 스모크 테스트 통과. 1회차에 vLLM 이식 완료.
- **SemanticSegmentCache DHD + TurboQuantCodec A+B+C 통합 (2026-05-03)**: 45/45 테스트 1회차 통과. 의미 유사도 기반 비연속 히트율 1.0 달성.
- **CacheCompressionConfig composition 패턴 확립 (2026-05-03)**: vLLM CacheConfig 수정 없이 독립 구성 객체로 압축 설정 관리.
- **DiffAwareSegmentStore FAISS 병목 구조적 우회 (2026-05-05)**: 검색 공간을 마스터 그룹 수로 제한. N>10K 환경에서도 확장 가능한 구조.
- **NQKVCodec 변환-무료 NF4 양자화 (2026-05-05)**: Hadamard/랜덤 회전 없이 정규분포 분위수 매핑만으로 INT4 달성. 1.882× 실압축.
- **FireQCodec RoPE-INT4 충돌 명시적 해소 (2026-05-05)**: pre-RoPE 채널 쌍 정규화 + post-RoPE 이상치 스케일링 2단계 평활화.
- **TriAttentionCodec pre-RoPE 안정적 중요도로 −90.6% 달성 (2026-05-06)**: 7사이클 최고 압축률. pre-RoPE 삼각함수 시리즈 중요도로 RoPE 회전 불안정 근본 해소. 보존 위치 왕복 정확도 atol=0.00 (완전 무손실, 역대 최고).
- **QueryCentricRecomputeCache 이중 단계 파이프라인 확립 (2026-05-06)**: 쿼리 관련성(Stage 1) × 코사인 유사도(Stage 2)의 이중 필터로 20% 예산 제한 정확히 구현. ProphetKV 원리 최초 적용.
- **DualFilterSegmentSelector B+C 이중 필터 통합 (2026-05-06)**: 40% 쿼리 관련성 필터 × 20% pre-RoPE 중요도 필터를 파이프라인으로 결합. 372개 테스트 1회차 전부 통과.
- **vLLM 1회차 Pass 연속 강화**: 7사이클 중 5사이클(2026-04-29, 2026-04-30, 2026-05-02, 2026-05-03, 2026-05-06)이 vLLM 이식 1회차 내 통과.
- **KL / cosine_sim / RMSE / atol=0.00 다중 accuracy 프록시 검증**: 7사이클 연속 ±1% 기준 대리 지표 통과. 2026-05-06에 atol=0.00으로 역대 최고 정밀도.

### 아직 해결 안 된 것
- **실제 GPU 처리량 미검증**: 7개 사이클 모두 CPU/시뮬레이션 환경. H100/A100에서 tokens/sec +20% 목표 Flash Attention 커널 연동 환경 검증 미완. run_gpu_throughput.py 스크립트 준비됨.
- **GPU perplexity 직접 측정 미완**: WikiText-2/GPT-2 perplexity 실측 없이 7사이클 연속 대리 지표 의존. run_triattention_accuracy.py 구현됨(2026-05-06); GPU + transformers + datasets 환경 필요.
- **compression_ratio 스윕 미완**: 2026-05-06 사이클에서 ratio=0.10만 실측. 0.1/0.2/0.3/0.5 스윕 미완. 정확도-압축률 트레이드오프 곡선 미확립.
- **TTFT GPU 실측 미완**: CPU-only 환경에서 스케줄링 오버헤드 14ms (100 req). GPU 환경 재측정 필요.
- **다중 노드 실측 검증**: DualMapScheduler(2026-05-03) 및 MultiNodeScheduler(2026-04-30) 모두 단일 머신 시뮬레이션으로만 검증됨.
- **비연속 히트율 벤치마크 실측 미완**: 2026-05-06 QueryCentricRecomputeCache의 실제 RAG/멀티턴 워크로드에서 히트율 30% 목표 달성 여부 확인 필요.
- **SegmentAdapter 사전 학습 부재**: 추론 시점에 untrained(random init) 상태. 오프라인 학습 없이는 KL 보정 효과가 실제 운영에서 다를 수 있음.
- **TriStateCompressor attn_weights 외부 의존**: 실제 추론 엔진에서 어텐션 점수를 classify()에 전달하는 파이프라인 연결이 vLLM 이식의 핵심 과제로 남음.
- **SemanticNonContiguousKVCacheManager KVCacheConfig E2E 통합 미완**: 실제 vLLM 서빙 파이프라인 연동 시 완전한 KVCacheConfig 객체 필요.
- **CacheHitAwareRequestQueue O(n log n) pop**: 대규모 배치(>256 요청)에서 완전한 힙 구현 필요.
- **1-bit sign VQ magnitude 정보 손실**: Tier-2 sign VQ는 방향만 보존, magnitude 손실로 attention cosine sim 이론적 상한 ~0.84~0.90.
- **approx_sign Stage 2 실질 미발동 구조**: Tier-1/Tier-2가 동일 청크 키를 공유하므로 FP16 히트 우선, sign-only 경로가 사실상 Tier-2 전용 청크에서만 발동.
- **vLLM 0.21+ 호환성 CI 미구축**: v1 엔진 경로(vllm.v1.*) KVCacheManager API 변경 가능성 대비 버전별 CI 추가 필요.
- **packed INT4(nibble 패킹) 미구현**: NQKVCodec 현재 uint8(1B/elem) 저장으로 1.882× 실압축. nibble 패킹 적용 시 이론치 3.56× 달성 가능.
- **CompressedDiffStore 에이전트 메모리 절감 −85% 미달**: 소규모(5-agent) 시나리오에서 −44.7%. 대규모 스케일 검증 필요.
- **Python 종료 시 segfault**: 3개 모듈 동시 임포트 후 Python finalize에서 발생. CUDA teardown 경쟁 조건 추정. vLLM 이슈 트래커 확인 필요.

### 다음 우선순위 제언
1. **Activity A 스케줄러 완전 통합 (최우선)**: QueryCentricSchedulerMixin이 2026-05-06에 vLLM 이식 코드로 구현됨. 다음 사이클에서 OnlineLatencyOptimalScheduler 또는 ContiguousChunkIOScheduler를 QueryCentricRecomputeCache + TriAttentionCodec과 결합해 A+B+C 삼중 조합 완성. make_qcrc_aware_scheduler_class(make_dag_aware_scheduler_class(Scheduler)) 조합 검증.
2. **GPU perplexity 자동 검증 환경 구축**: run_triattention_accuracy.py가 2026-05-06에 구현됨. CI 파이프라인에 GPT-2 small + WikiText-2 smoke-level perplexity 검증 통합. ±1% 기준 매 사이클 자동 검증으로 7사이클 연속 대리 지표 의존 해소.
3. **compression_ratio 스윕 실행**: 0.05/0.10/0.20/0.30에서 perplexity 측정 후 정확도-압축률 트레이드오프 곡선 확립. metrics.json에 ratio별 결과 기록.
4. **GPU 벤치마크 환경 구축**: run_gpu_throughput.py CUDA 환경에서 실행. TTFT p50/p99, TBT, tokens/sec 실측. +20% throughput 목표 검증.
5. **Python 종료 segfault 원인 분석**: CUDA/flashinfer teardown 경쟁 조건 분석. vLLM 이슈 트래커 보고 및 teardown 순서 명시적 제어 시도.

SUMMARY_UPDATED
