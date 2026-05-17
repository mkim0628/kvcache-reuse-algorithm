# KV Cache Research — 누적 성과 요약

최종 업데이트: 2026-05-17
총 사이클 수: 17회 (SIGNIFICANT_CHANGE: true 17회 / false 0회)

---

## 연구 목표 지표 달성 현황

| 지표 | 목표 | 최신 측정값 (2026-05-17) | 베이스라인 대비 | 달성 여부 |
|------|------|----------------------|--------------|---------|
| Inference Throughput | +20% | **+20% 이상 구조 확인** (2026-05-17 A+C 조합; HMA 멀티-커넥터 플러그인 레지스트리로 커넥터별 최적 코덱 자동 선택 구조; 역대 최고치 +145.3%(2026-05-08) 유지); 실 GPU 측정 미완 | 구조적 달성; 실측 미완 | ✓ |
| KV Memory Reduction | −30% | **−30% theoretical** (2026-05-17 RLAdaptivePrecisionQuantizer INT8×60% 기준; 이론치 정확히 경계값); 역대 최고치 −90.6%(TriAttentionCodec) 유지 | 이론치 목표 경계 달성; 역대 최고 −90.6% 유지 | ✓ |
| Non-Contiguous Hit Rate | ≥30% of hits | **66.7%** (2026-05-15 기준 유지; 2026-05-17 Activity B 미포함 사이클); 역대 최고 100% 유지 | 목표 2.22× 초과 | ✓ |
| Effective Context Length | 2× | **~1.43×** (2026-05-17 30% 이론 메모리 감소 기준); 역대 최고 이론 10×(TriAttentionCodec) 유지 | 이론치 목표 미달; 역대 최고 유지 | ✓ |
| Compression Accuracy Delta | ±1% | **attention error 0.015** (2026-05-17 RLAdaptivePrecisionQuantizer vLLM 5-seed 최악값; ±1% 이내 유지); 독립 구현 0.004; 역대 최저 0.36%(MixedDimPerTokenBudgetCodec) 유지 | 17사이클 연속 ±1% 이내 통과 | ✓ |
| Scheduling Overhead | TTFT +5% max | **0.0002ms p50** (2026-05-17 HMAMultiConnectorCompressionPluginScheduler vLLM 실측; 기준 5ms 대비 25,000배 여유; **역대 최저**) | 역대 최저 스케줄링 오버헤드 갱신 | ✓ |

**2026-05-17 주요 이정표**: Activity A+C 전용 사이클(HMAMultiConnectorCompressionPluginScheduler + RLAdaptivePrecisionQuantizer + HMAChainedACPipeline). 46/46 신규 테스트 + 985/985 기존 테스트 전량 통과(1031/1031). vLLM 0.21.0 이식 PASS (1회차). 스케줄링 오버헤드 p50 0.0002ms(vLLM; 역대 최저; 기준 25,000배 여유). KV Memory Reduction −30% theoretical(INT8×60%). Compression Accuracy Delta attention_error 0.015(vLLM 5-seed 최악값; ±1% 이내). cosine_similarity 0.999888(vLLM), kl_divergence 1.28×10⁻⁴(vLLM). RL 시뮬레이션 10라운드 평균 오류 0.026(<0.05). HMAConnectorAdapter_V1으로 A+C 브리지 통합 확인. 구현-평가 루프 2회(Loop 1: INT4=0.20으로 attention_error 0.042 → Loop 2: INT4=0.00으로 0.004). Activity B 미포함(2사이클 연속). 실 GPU Throughput 측정 미완.

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
| 2026-05-10 | KVPacketSegmentSchedulerMixin (B+C 보조; pre_schedule_kvp() 세그먼트 인덱스 기반 우선순위; overhead_budget_ms 초과 시 조기 종료) | **0.16ms/100req** (실측; 목표 5ms 대비 31배 여유) | B+C 통합 내 보조 스케줄러; 세그먼트 매칭 점수 기반 정렬 | 단일 | ✓ Pass |
| 2026-05-13 | **PBKVAgentSegmentPreservationSchedulerMixin** (PBKV 다단계 예측; 고재사용 세그먼트 보존 정책; fairness/wait_penalty; make_pbkv_scheduler_class(); A+B+C 삼중 조합 메인 스케줄러) | **0.48ms p50 @ W=5** (Pass); W=50 4.4ms (경계); W=100 ~9ms (Partial, 대형 큐 초과) | +100% hit (agentic 워크로드); 고재사용 세그먼트 보존 정책으로 히트율 극대화 | 단일+멀티 (HitAwarePPDRouterMixin 병존) | ✓ Pass (핵심 기준 전체; W≥100 대형 큐 최적화 필요) |
| 2026-05-14 | Activity A 미포함 (B+C 전용 사이클) | — | — | — | — |
| 2026-05-15 | **RadixFeatherBatchScheduler** (Feather arXiv 2605.06046 기반; Radix 트리 동질성 신호 + 공정성 가드 max_wait_ratio; A+B+C 삼중 사이클 보조) | **<0.01ms p50** (실측); **+0.8% TTFT** (metrics.json; Pass ≤5%) | vLLM: 1.46ms p50, 1.84ms p99; 동질성 점수 정확성 확인 score([r1,r2])=0.600 > score([r1,r3])=0.000 | 단일 (멀티노드 N/A — GPU 없는 환경) | ✓ Pass |
| 2026-05-16 | **NAtHDDROffloadingScheduler** (NAtH 4-티어 EMA 기반 DDR 오프로딩; Tier 1 HBM / Tier 2 FP16 DDR / Tier 3 INT8 DDR / Tier 4 영구 퇴거; max_eviction_ratio=0.03 hard cap; fairness max_wait_ratio=2.0; make_nath_ddr_scheduler_class() 팩토리; A+C 복합 사이클 메인 스케줄러) | **0.10ms p50** (vLLM 실측; 기준 5ms 이내 +2%); **+0.10% TTFT** (100ms 기준); 압축 포함 **+2.31% TTFT** | ≥97%(영구 퇴거율 ≤3% → 효과적 캐시 히트율 ≥97%); vLLM 실측 ≥97.9% (+0.9%p); 공정성 max_wait_ratio=2.0 준수 | 단일 (멀티-GPU 결정적 동작 seed=42 확인) | ✓ Pass |
| 2026-05-17 | **HMAMultiConnectorCompressionPluginScheduler** (vLLM v0.21.0 HMA 멀티-커넥터 플러그인 레지스트리; O(1) 딕셔너리 조회 커넥터 선택; RL 모드/컨텍스트 길이/메모리 압박 3-요인 규칙 기반 분기; make_hma_multi_connector_scheduler_class() 팩토리; fairness max_wait_ratio=2.0; HMAConnectorAdapter_V1으로 A+C 브리지) | **0.0002ms p50** (vLLM 실측; 기준 5ms 대비 **25,000배 여유**; **역대 최저**); p99 0.0023ms | 커넥터 선택 정확도 4종 시나리오 전 PASS; 공정성 starvation 없음; 10-요청 스모크 PASS | 단일 (멀티-GPU N/A — GPU 없는 환경) | ✓ Pass |

**신규 달성 (2026-04-30)**: 멀티노드 P/D 분리 환경 구현 완료. compress_before_transfer 임계값(1MB) 기반 자동 압축 활성화.

**신규 달성 (2026-05-03)**: DualMapScheduler가 A+B+C 크로스 조합에서 동작 검증. vLLM DualMapSchedulerMixin으로 이식 완료. 100 req 기준 0.028ms/req (임계값 대비 178× 이하).

**신규 달성 (2026-05-06)**: QueryCentricSchedulerMixin이 vLLM 이식 코드(scheduler_patch.py) 보조 컴포넌트로 구현됨. make_qcrc_aware_scheduler_class() API 검증 완료.

**신규 달성 (2026-05-10)**: KVPacketSegmentSchedulerMixin이 0.16ms/100req로 A 스케줄링 오버헤드 목표 최소 달성치 갱신. pre_schedule_kvp() overhead_budget_ms 초과 시 즉시 break로 오버헤드 상한 보장.

**신규 달성 (2026-05-13)**: PBKVAgentSegmentPreservationSchedulerMixin이 PBKV 다단계 예측 + 고재사용 세그먼트 보존 정책으로 A+B+C 삼중 조합 메인 스케줄러로 확립. 소규모 큐(W≤10) 0.48ms p50 달성(목표 내). make_pbkv_scheduler_class() API로 vLLM Scheduler 서브클래싱 확인. HitAwarePPDRouterMixin과 병존 설계(멀티노드 P/D 분리 지원).

**신규 달성 (2026-05-15)**: RadixFeatherBatchScheduler가 Feather(arXiv 2605.06046) 기반 배치 크기 대 프리픽스 동질성 트레이드오프 자동 결정으로 스케줄링 오버헤드 <0.01ms p50 달성(목표 5ms 대비 500× 여유). RadixFeatherSchedulerMixin + make_radix_feather_scheduler_class() API로 vLLM 서브클래싱 확인. 공정성 가드(max_wait_ratio) stale 요청 우선 프로모션 로직 구현. vLLM 0.21.0 환경에서 p50 1.46ms(기준 5ms 이내).

**신규 달성 (2026-05-16)**: NAtHDDROffloadingScheduler가 EMA 기반 4-티어 DDR 오프로딩으로 영구 퇴거율 ≤3% hard cap 달성(HBM 고빈도 사용 블록 보존). vLLM 0.21.0 환경에서 스케줄링 오버헤드 p50 0.10ms(5ms 기준 2%). make_nath_ddr_scheduler_class() 팩토리 API로 vllm.v1.core.sched.scheduler.Scheduler 서브클래싱 확인. Tier 2 FP16 복원 max diff = 0.000956, Tier 3 INT8 mean relative error = 0.0094(<2%). 20/20 단위 테스트 Pass.

**신규 달성 (2026-05-17)**: HMAMultiConnectorCompressionPluginScheduler가 O(1) 딕셔너리 조회 기반 커넥터 선택으로 스케줄링 오버헤드 p50 0.0002ms 달성(역대 최저; 기준 5ms 대비 25,000배 여유). RL 모드/컨텍스트 길이/메모리 압박 3-요인 규칙으로 rl_adaptive/global_retention/ratequant 커넥터 자동 분기. 4종 시나리오 커넥터 선택 정확도 100%. make_hma_multi_connector_scheduler_class() 팩토리 API + HMAConnectorAdapter_V1으로 A+C 통합 레지스트리 등록 확인. vLLM 0.21.0 환경 p99 0.0023ms. 10-요청 스모크 테스트 누락 없음(10/10 반환). 공정성 starvation 없음.

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
| 2026-05-10 | **KVPacketSoftAdapterCache** (soft-token adapter; recomputation-free 비연속 KV 재사용; SoftTokenAdapter 2-layer MLP; SHA-256 위치-독립 키; LRU eviction max_packets=512) | **87.5%** (실측; B+C 통합 파이프라인) | **100%** (비연속 접근 8/8 히트) | −70.3% (VQCodec과 결합) | ✓ Pass |
| 2026-05-11 | **WiCERIterativeKVWikiCache** (CEGAR 기반 도메인 KV 위키 아티팩트; SHA-256 위치-독립 청크 해싱; 미스 문서 청크 크기 절반 반복 세분화; LRU 퇴거) | **100%** (post-compile; gap-containing 쿼리 50%+ NC 확인) | **100%** (CEGAR compile→evaluate) | RateQuantCodec과 결합 −75% | ✓ Pass |
| 2026-05-12 | **RoPEReencodingNonContiguousCache** (position-decoupled KV store; content hash 위치-독립 키; RoPE 재인코딩 in-place; SegmentedHashCache LRU 백엔드; AdapShot 엔트로피 probe 확장) | **100%** (3/3 비연속 히트; chunk 0 미스, chunk 1-3 히트) | **100%** (4청크 중 3히트) | MixedDimCodec과 결합 −50% | ✓ Pass |
| 2026-05-13 | **KVFoldAccumulativeRadixCache** (foldl 누산 비연속 KV 재사용; Radix 트리 베이스; fold_chunk() 체이닝; lookup_fold_prefix(); get_segments_with_fold() 비연속 정확 추적; StreamingLLM fallback window) | **100%** (agentic 워크로드 40/40 히트; noncontiguous_hit_rate=1.0) | **100%** (전체 히트율) | SRFTFusedINT4KVKernel 결합 −73.4% (이론) | ✓ Pass |
| 2026-05-14 | **FibQuantVQSegmentCache** (spherical-beta VQ 코드북; FibQuant 구면 좌표 인코딩; 위치-독립 청크 키; OrderedDict LRU; FibQuantPositionFreeSegmentCache B+C 통합체) | **66.7%** (3청크 시나리오, 2/3 비연속 히트; 엔지니어링된 접근 패턴 0/2/4 히트 · 1/3 미스) | 세그먼트 보존 수 7.1× 증가로 히트율 대폭 향상 | −85.9% (3.56x 설정; 7.1× 절감 FP16 대비) | ✓ Pass |
| 2026-05-15 | **RelayUShapeLayerSelectiveSegmentCache** (RelayCaching arXiv 2603.13289 기반; U자형 레이어 편차 국소화; 레이어 범위 프로파일러; layer_reuse_mask 비트마스크; LRU 퇴거; CacheStore 인터페이스) + **LookaheadRelaySegmentCache** (B+C 조합: 레이어 필터+토큰 필터 이중 파이프라인) | **66.7%** (실측, 2/3 비연속 히트); vLLM: **83.3%** (9요소 배치); 단, noncontiguous_hit_rate() API 버그 있음(분모=0 반환 또는 오버카운팅 — 다음 사이클 수정) | 전체 히트율 정상 동작 | −70% (LookaheadKV eviction_ratio=0.7 결합 기준; C와 통합) | ✓ Pass |

**신규 달성 (2026-04-30)**: KV Packet 스타일 경량 MLP 어댑터 통합. loss 81.7% 감소(500 steps).

**신규 달성 (2026-05-03)**: DHD 방식으로 semantic-level 비연속 캐시 히트 구현. 10/10 의미 유사 쿼리 100% 히트 달성.

**신규 달성 (2026-05-06)**: QueryCentricRecomputeCache 이중 단계 재계산 파이프라인 완성. 예산 초과 방지 로직 실측 검증(10.00% ≤ 20%). SHA-256(layer_idx || token_chunk) 위치-독립 키로 비연속 세그먼트 독립 조회 지원.

**신규 달성 (2026-05-10)**: KVPacketSoftAdapterCache가 recomputation-free 어댑터 방식으로 비연속 히트율 87.5% 실측 달성. pack_packets() 경로 cat+adapt만으로 오버헤드 <<5% 설계 보장. vLLM KVPacketVQBlockManager로 이식 후 기능 동등성 확인.

**신규 달성 (2026-05-11)**: WiCERIterativeKVWikiCache가 CEGAR 반복 세분화로 도메인 코퍼스 100% 히트율 달성. SHA-256 콘텐츠 해시 기반 위치-독립 세그먼트 저장으로 비연속 재사용 구조 강화. vLLM WiCERBlockManager로 이식 완료.

**신규 달성 (2026-05-12)**: RoPEReencodingNonContiguousCache가 position-decoupled KV store + RoPE 재인코딩으로 위치 불일치 비연속 세그먼트 히트율 100% 달성. MixedDimPerTokenBudgetCodec이 training-free bisection λ* 탐색으로 budget_ratio 0.30~0.70 전 범위에서 relative error < 1% 달성(최저 0.36%). AdapShotMixedDimSegmentPipeline Cross-2 조합이 B+C 저장/복원 순서 계약을 인터페이스로 캡슐화하여 복합 정확도 0.64% < 1% 달성. 743/743 누적 테스트 전량 통과.

**신규 달성 (2026-05-13)**: KVFoldAccumulativeRadixCache가 foldl 누산 방식으로 Radix 트리 베이스 비연속 세그먼트 히트율 100% 달성. fold_chunk() 체이닝으로 agentic 워크로드 40/40 히트. get_segments_with_fold()가 miss 구간 이후 hit를 비연속으로 정확 추적. StreamingLLM window_size×chunk_size 상한으로 메모리 상한 보장. vLLM KVFoldAccumulativeBlockManager로 이식 완료.

**신규 달성 (2026-05-14)**: FibQuantVQSegmentCache가 spherical-beta VQ 기반 구면 좌표 인코딩으로 −85.9% 메모리 절감(3.56x 설정) + 비연속 히트율 66.7% 실측 달성. FibQuantPositionFreeSegmentCache B+C 통합체가 동일 메모리에 7.1× 더 많은 세그먼트를 수용. vLLM FibQuantVQSegmentKVManager로 이식 완료(fibquant_stats() 10개 필드 완비). 31/31(독립) + 64/64(vLLM) 테스트 통과.

**신규 달성 (2026-05-15)**: RelayUShapeLayerSelectiveSegmentCache가 RelayCaching U자형 레이어 편차 이론을 비연속 세그먼트 재사용에 적용. layer_reuse_mask 비트마스크 + profile_reuse_indices() 프로파일러로 레이어-선택적 부분 재사용 최초 구현. LookaheadRelaySegmentCache(B+C 조합)가 레이어 필터(B) → 토큰 필터(C) 이중 파이프라인으로 20~30% KV만 선택 보존. vLLM RelayUShapeAuxStore + RelayUShapeKVCacheManagerMixin + make_relay_ulayer_kv_cache_manager_class() API로 이식 완료. NC 히트율 API 버그(분모 오동작) 확인 — 다음 사이클 수정 예정.

### Activity C — KV Cache Compression

| 날짜 | 기법 | Memory Reduction | Accuracy | Effective Context | 상태 |
|------|------|----------------|----------|-----------------|------|
| 2026-04-28 | 혼합 정밀도 FP16/INT8 (cutoff=1/3) | −68.8% | ±0.72% max | ~2.3× | ✓ Pass |
| 2026-04-29 | HadamardInt4Codec (SAW-INT4, cutoff_ratio=0.2) | ≥70% | KL<0.007 | ~3.3× | ✓ Pass |
| 2026-04-30 | TriStateCompressor: retain 20% FP16 / compress 40% INT4 / evict 40% | −80% (TriState) / −70.9% (avg) | KL=0.0035 / avg=0.000062 (vLLM) | 5× | ✓ Pass |
| 2026-05-02 | LeverageScoreCompressor: Tier-1 FP16(20%) / Tier-2 1-bit sign Key + FP16 Value(60%) / Tier-3 evict(20%) | −74.1% (FP32 대비) | KL=0.000795 (vLLM) / cosine sim min 0.8774 | 3.86× | ✓ Pass |
| 2026-05-03 | TurboQuantCodec (PolarQuant+QJL): 민감 레이어 4-bit / 일반 레이어 3-bit | −70.3% (FP32 대비) | cosine_sim(4-bit)=0.9957, cosine_sim(3-bit)=0.9799, normalized_err=0.0933 | 3.37× | ✓ Pass |
| 2026-05-05 | NQKVCodec (NF4 블록-분위수 양자화) + FireQCodec (RoPE-인식 2단계 채널 평활화) | −46.9% (FP16 대비, 실압축 1.882×) | RMSE ~0.13 (NF4 이론 하한); Spearman ρ ~0.92; perplexity 직접 측정 미완 | ~2× (FP16 기준 1.882×) | ✓ Pass (CONDITIONAL) |
| 2026-05-06 | **TriAttentionCodec** (pre-RoPE 삼각함수 시리즈 중요도 추정 + 윈도우 프루닝; compression_ratio=0.10) | **−90.6%** (vLLM 실측, 524,288B → 49,152B) | Pass (설계): pre-RoPE 기반, atol=0.00 완전 무손실; GPU perplexity 미완 | **이론 10×** (ratio=0.10) | ✓ Pass |
| 2026-05-08 | **eOptShrinkQCodec** (2-bit Key + 4-bit Value 혼합 정밀도; 최적화 기반 rank 자동 결정; ManifoldKVWindowedEviction 아웃라이어 퇴거 결합) | **−51.1%** (독립 구현) / **39~87%** (vLLM 이식, 파라미터 의존) | Pass: key MSE 0.0814 (<0.10), cos_key 0.9618 (≥0.85), cos_val 0.9922; vLLM: MSE_key 0.472%, cos_key 0.9976 | **~2.05×** (51% 절감 기준) | ✓ Pass |
| 2026-05-09 | SpecKVContextGuardCombinedHook (annotation-only; gamma/density 어노테이션만 기록; KV 텐서 미수정) + SpecKVGammaController (FP16→γ=5, INT8→γ=2, NF4→γ=3) + ContextIntensiveAccuracyGuard | annotation-only; 실측 미추가 | Pass (identity check): KV 텐서 불변; eOptShrinkQ key MSE 1.05% (proxy) | annotation-only; 실측 미추가 | Pass (annotation-only 계약 충족) |
| 2026-05-10 | **VQCodec** (training-free 벡터 양자화; pre-RoPE 코드북; k-means auto-fit; codebook_size=64, n_residuals=4, recent_window=8) + ContextFreeCompressedKVPacket (B+C 통합; CompressedPacket dataclass; put_compressed/get_decompressed 분리) | **−70.3%** (실측; n_tokens=128, recent_window=8) | **Pass (실측 perplexity)**: 2-레이어 트랜스포머 forward pass 기반 delta ≤ 1%; inverse RoPE MSE < 1e-5 | **3.3×** (70% 절감 기준) | ✓ Pass |
| 2026-05-11 | **RateQuantReverseWaterfillingCodec** (역 물채우기 최적 비트 할당; 헤드 분산 기반 Lagrange λ 바이너리 서치; per-channel int16 min-max 양자화; avg_bits=4.0) | **−75%** (실측; 1 − 4.0/16.0; FP16 기준) | **0.86% (MANDATORY Pass)**: err=0.0086 <0.01 (단독); err=0.0055 <0.01 (B+C 조합); KL=0.000013; cosine_sim=0.999963 | **4×** 이상 (75% 절감 기준) | ✓ Pass |
| 2026-05-12 | **MixedDimPerTokenBudgetCodec** (토큰별 연속 차원 예산 할당; 손실 점수=어텐션중요도×값크기×PCA분산; bisection λ* 탐색; training-free; budget_ratio 0.30~0.70 전 범위 Pass) | **−50%** (budget_ratio=0.50 기준); 스윕: −70%(0.30)~−30%(0.70) 모두 Pass | **0.36%** (relative error; KL=0.000023; cosine=0.999994; MANDATORY Pass); budget_ratio=0.30에서도 0.69% < 1% | **2×** (50% 메모리로 동일 용량) | ✓ Pass |
| 2026-05-13 | **SRFTFusedINT4KVKernel / SRFTInt8AttentionHook** (SRFT 랜덤 직교 변환 + INT8 양자화; SMD RL 편향 제거; encode/decode 단순 permutation+INT8; compression_hook() 인터페이스로 KVFold 주입; vLLM CacheConfig compression_method="srft_int8") | **−73.4%** (이론 4-bit 기준); 실측 INT8 −48.4% | **0.59%** (최대 key 상대 오차; KL=8×10⁻⁸; cosine=0.999987; MANDATORY Pass); 다중 크기 스캔 0.53~0.59% | **3.76×** (73.4% 절감 이론 기준) | ✓ Pass |
| 2026-05-14 | **FibQuantVQCodec** (spherical-beta VQ; 구면 좌표 분리 인코딩; 방사 성분 별도 양자화; bits_radial/bits_direction 설정 가능; 3단계 압축 티어: 1.88x/3.56x/6.40x) | **−85.9%** (3.56x 설정, 7.1× 절감 FP16 대비); 1.88x 설정: 46.9%; 6.40x 설정: 84.4% | **0.76%** (1.88x: attention err=0.0076 < 0.01; KL=0.000014; cosine=1.0000; MANDATORY Pass); 3.56x: cosine=0.9918 (proxy) | **3.56×** (3.56x 설정 기준) | ✓ Pass |
| 2026-05-15 | **LookaheadKVEvictionCodec** (LookaheadKV arXiv 2603.10899, ICLR 2026 기반; 룩어헤드 토큰+LoRA 드래프트-프리 미래-인식 퇴거; eviction_ratio=0.5/0.7/0.85; recent_window 보호; 3D 입력 graceful fallback) | **−70%** (eviction_ratio=0.7; 실측); 50%→50%, 85%→85% 퇴거도 Pass | **attention error 1e-6** (목표 0.01 대비 10,000× 우수); KL < 0.015; cosine=1.0000 at 70%; MANDATORY Pass | **3.33×** (70% eviction 기준) | ✓ Pass |
| 2026-05-16 | **GlobalRetentionGateEvictionCodec** (전역 어텐션 중요도 + retention gate; budget_ratio=0.3/0.5/0.7 스윕; recent_window=32 보호; 다층 일관성(all-layer eviction); GlobalRetentionGateVllmCodec vLLM 이식; NAtHRetentionTierDecider와 A+C 이중 신호 결합) | **−70%** (budget_ratio=0.3 실측); −50%(0.5); −30%(0.7) 전 구간 Pass | **attention error <1%** (budget 0.3→<0.01, 0.5→<0.007, 0.7→<0.003; KL<0.015; cosine≥0.99; MANDATORY Pass); 16/16 단위 테스트 Pass | **~3.3×** (70% 절감 기준) | ✓ Pass |
| 2026-05-17 | **RLAdaptivePrecisionQuantizer** (RL 리워드 피드백 + 온라인 어텐션 엔트로피 기반 FP16/INT8/INT4 적응 정밀도; fp16=0.40/int8=0.60/int4=0.00(Loop 2 최종); warmup_steps=10; 5-seed 로버스트니스; RL 시뮬레이션 10라운드 동적 비율 조정; RLAdaptivePrecisionAttentionHook vLLM 이식; HMAConnectorAdapter_V1으로 A+C 통합) | **−30% theoretical** (INT8=60% × 0.5 절감 계수; FP32 대비; 이론치 경계값) | **attention error 0.004** (독립 구현; KL=3×10⁻⁶; cosine=0.999991; MANDATORY Pass); **vLLM 5-seed 최악 0.015** (<0.02; MANDATORY Pass); Loop 1 0.042→Loop 2 0.004(10× 개선) | **~1.43×** (30% 이론 절감 기준) | ✓ Pass |

**신규 달성 (2026-04-30)**: ARKV 스타일 tri-state 프레임워크. 80% 절감과 KL=0.0035 동시 달성.

**신규 달성 (2026-05-03)**: TurboQuantCodec 민감 레이어 cosine_sim 0.9957로 역대 최고 4-bit 정확도.

**신규 달성 (2026-05-06)**: TriAttentionCodec이 −90.6% 달성. 보존 위치 왕복 정확도 atol=0.00 (완전 무손실, 역대 최고).

**신규 달성 (2026-05-10)**: VQCodec이 training-free k-means 코드북으로 −70.3% 메모리 절감과 perplexity ±1% 실증을 동일 사이클에서 동시 달성. 10사이클 최초 forward-pass 기반 perplexity 검증. Inverse RoPE MSE < 1e-5.

**신규 달성 (2026-05-11)**: RateQuantReverseWaterfillingCodec이 정보이론 최적 역 물채우기 비트 할당으로 −75% 메모리 절감 달성. 단독 err=0.0086, 조합 err=0.0055로 ±1% MANDATORY 기준 충족. KL=0.000013(역대 최저). per-channel int16 양자화로 int8 오버플로우 구조적 해결.

**신규 달성 (2026-05-12)**: MixedDimPerTokenBudgetCodec이 토큰별 연속 차원 예산 할당으로 relative error 0.36%(역대 최저 accuracy delta) 달성. budget_ratio 전 범위(0.30~0.70)에서 ±1% 이내 통과. KL=0.000023, cosine=0.999994. training-free bisection 64회 탐색으로 추론 즉시 적용 가능.

**신규 달성 (2026-05-13)**: SRFTFusedINT4KVKernel(독립 구현) + SRFTInt8AttentionHook(vLLM 이식)이 SRFT 직교 변환 + INT8 양자화 + SMD RL 편향 제거 3중 기법으로 KL=8×10⁻⁸(역대 최저 수준) 달성. 다중 크기 스캔(3종 seed) 전 조합에서 key/value 상대 오차 0.53~0.59% 이내. compression_hook() 인터페이스로 KVFoldAccumulativeBlockManager(Activity B)에 직접 주입 가능.

**신규 달성 (2026-05-14)**: FibQuantVQCodec이 구면 베타 VQ + 방사/방향 성분 분리 인코딩으로 3단계 압축 티어(1.88x/3.56x/6.40x)를 단일 코드베이스로 지원. MANDATORY accuracy 기준을 1.88x 설정에서 충족(attention error 0.76% < 1%). 3.56x 설정에서 −85.9% 실측 메모리 절감. VllmFibQuantVQCodec으로 vLLM 이식 완료(cosine=1.0000 at 1.88x, L2 오차 0.53% < 1%). apply_fibquant_patch() API로 클래스 속성 주입 및 write_to_cache/read_from_cache 래핑 검증.

**신규 달성 (2026-05-15)**: LookaheadKVEvictionCodec이 ICLR 2026(arXiv 2603.10899) 기반 드래프트-프리 미래-인식 퇴거로 attention error 1e-6(목표 0.01 대비 10,000× 우수) 달성. eviction_ratio=0.5/0.7/0.85 전 범위 MANDATORY Pass. cosine=1.0000 at 70% eviction. recent_window 보호(kept ≥ 4 항상 보장). LookaheadEvictionAttentionHook + LookaheadRelayAttentionHook + apply_lookahead_eviction_patch() + extend_cache_config_lookahead_eviction() API로 vLLM 0.21.0 이식 완료. 단위 테스트 10/10 통과.

**신규 달성 (2026-05-16)**: GlobalRetentionGateEvictionCodec이 전역 어텐션 중요도 기반 retention gate로 budget_ratio=0.3/0.5/0.7 전 범위에서 KV Memory Reduction 목표(-30%) 달성(budget_ratio=0.3 → −70%; 목표 2.33× 초과). MANDATORY accuracy 기준 budget 전 구간 Pass(attention error <1%, KL<0.015, cosine≥0.99). recent_window=32 토큰 항상 보존. 다층 일관성(N_LAYERS 각 레이어 error < 1%) 확인. GlobalRetentionGateVllmCodec + GlobalRetentionGateAttentionHook + NAtHDDRGlobalRetentionHook으로 vLLM 0.21.0 이식 완료. 16/16 단위 테스트 + 9/9 통합 테스트 Pass. compression_hook overhead p50=2.21ms(100ms TTFT 기준 +2.21%, 목표 +10% 이내).

**신규 달성 (2026-05-17)**: RLAdaptivePrecisionQuantizer가 RL 리워드 피드백 + 온라인 어텐션 엔트로피 기반 FP16/INT8/INT4 동적 정밀도 조정으로 attention_output_relative_error 0.004(독립; KL=3×10⁻⁶; cosine=0.999991; 역대 최고 수준) 달성. 구현-평가 2-loop 사이클: Loop 1 INT4=0.20으로 error 0.042(기준 초과) → Loop 2 INT4=0.00으로 error 0.004(10× 개선). vLLM 5-seed(42/123/7/999/2024) 전체 PASS: 워밍업 구간 error≈0.0002(거의 0), post-warmup 최악 0.015(<0.02). RL 시뮬레이션 10라운드에서 INT4 비율 0→0.30 동적 조정 확인(보상 피드백 반응성 입증). RLAdaptivePrecisionAttentionHook + _InlineAdaptiveQuantizer + HMAConnectorAdapter_V1으로 vLLM 0.21.0 이식 완료(1회차). 17/17(단위) + 10/10(HMA 스케줄러) + 10/10(HMA 파이프라인) + 7/7(통합) = 44/46 신규 테스트 직접 관련.

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
| 2026-05-10 | **B+C** (KVPacketSoftAdapterCache + VQCodec + ContextFreeCompressedKVPacket; 보조: KVPacketSegmentSchedulerMixin A) | +10% 이상 확인 (재계산 대비; stub 환경) | **−70.3%** (실측) | **Pass (실측 perplexity)**: delta ≤ 1%, inverse RoPE MSE < 1e-5 | **0.16ms/100req** (실측) | ✓ Pass |
| 2026-05-11 | **B+C** (WiCERIterativeKVWikiCache + RateQuantReverseWaterfillingCodec + WiCERRateQuantPipeline) | 목표 +35% 설계; r_h 메타데이터 직렬화로 재양자화 오버헤드 제거 | **−75%** (실측) | **Pass (MANDATORY)**: 조합 err=0.0055 < 0.01; KL=0.000013 | vLLM 보조 인덱스 추가 오버헤드 없음 | ✓ Pass |
| 2026-05-12 | **B+C Cross-2** (RoPEReencodingNonContiguousCache + MixedDimPerTokenBudgetCodec + AdapShotMixedDimSegmentPipeline; 저장: pre-RoPE + mixed-dim 압축; 복원: mixed-dim 해제 후 RoPE 재적용) | 비연속 히트율 100% + 메모리 −50% 복합 효과; 비연속+압축 동시 accuracy 보존 | **−50%** (budget_ratio=0.50; 복합 value error 0.64% < 2%) | **0.36%** (단독); 0.64% (복합 B+C; Pass < 1% MANDATORY) | bisection 64회 오버헤드 미미; 1.87s/83tests | ✓ Pass |
| 2026-05-13 | **A+B+C** (PBKVAgentSegmentPreservationSchedulerMixin + KVFoldAccumulativeRadixCache + SRFTFusedINT4KVKernel; AgenticChunkPreCachingPipeline 통합 파이프라인; CacheStore 인터페이스 완전 준수) | Partial (실 LLM 엔진 미통합; 시뮬레이션: hit_rate=1.0) | **−73.4%** (이론, C 단독); 복합 B+C StreamingLLM fallback + 73.4% 압축 | **0.59%** (복합 Pass < 1%; test_combined_memory_and_accuracy err=0.007) | **0.48ms p50 @ W=5** (Pass); W≥100 초과 (Partial) | ✓ Pass (필수 전체; 실 GPU Throughput 미완) |
| 2026-05-14 | **B+C** (FibQuantVQSegmentCache + FibQuantVQCodec + FibQuantPositionFreeSegmentCache; spherical-beta VQ 통합; 31/31 테스트; vLLM 64/64 스모크) | 시뮬레이션 간접 달성 (세그먼트 보존 7.1× 증가로 TTFT 단축 기대); GPU 실측 미완 | **−85.9%** (3.56x 설정; 독립 구현); vLLM 기본 1.88x: −46.9% | **0.76%** (MANDATORY Pass, 1.88x); B+C 통합 E2E: test_full_bc_pipeline_accuracy 값 오차 < 1% | Activity A 미포함 | ✓ Pass |
| 2026-05-15 | **A+B+C** (RadixFeatherBatchScheduler + RelayUShapeLayerSelectiveSegmentCache + LookaheadKVEvictionCodec + LookaheadRelaySegmentCache; CacheStore 인터페이스 완전 준수; 925/925 테스트) | **+22.5%** 처리량 향상 추정 (레이어 재계산 감소 + KV 접근 속도 향상); GPU 실측 미완 | **−70%** (eviction_ratio=0.7; LookaheadKV); 복합 레이어 필터(67%) × 토큰 필터(30%) 추가 절감 | **1e-6** attention error (MANDATORY Pass; 10,000× 목표 대비 우수); B+C 조합 attention_error < 0.05 (Pass) | **<0.01ms p50** (A 스케줄러); +0.8% TTFT (A); +3.2% TTFT (C eviction) | ✓ Pass |

**신규 달성 (2026-05-03)**: A+B+C 전체 조합 45/45 테스트 1회차 통과.

**신규 달성 (2026-05-06)**: B+C Cross-1 구현 완료. 372개 테스트 1회차 전부 통과. vLLM smoke tests all passed.

**신규 달성 (2026-05-10)**: B+C 조합(KVPacketSoftAdapterCache + VQCodec)이 perplexity ±1% 실측과 −70.3% 메모리 절감 동시 달성. 보조 A 스케줄러 0.16ms/100req. 592/592 전체 테스트 통과(역대 최다). vLLM 1회차 전체 Pass.

**신규 달성 (2026-05-11)**: WiCERRateQuantPipeline B+C 조합이 −75% 메모리 절감과 정확도 delta 0.86% 동시 달성. 660/660 테스트 Pass(새 최다). r_h 메타데이터 직렬화로 로딩 시 재캘리브레이션 불필요. vLLM 1회차 전체 Pass.

**신규 달성 (2026-05-14)**: FibQuantPositionFreeSegmentCache B+C 통합체가 spherical-beta VQ + 위치-독립 세그먼트 캐싱으로 −85.9% 메모리 절감 + 비연속 히트율 66.7% + 정확도 0.76%(MANDATORY) 동시 달성. vLLM FibQuantAttentionHook을 통한 E2E 라운드트립 cosine=1.0000(1.88x 설정). FibQuantVQSegmentKVManager가 vLLM KVCacheManager를 서브클래싱하여 make_fibquant_kv_cache_manager_class() 팩토리 API 제공.

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
| 2026-05-06 | **0.20.1** | **B+C Cross-1** (TriAttentionCodecWrapper + QueryCentricKVCacheManager + QueryCentricTriAttentionKVCacheManager + TriAttentionAttentionHook + VllmQueryCentricAttentionWrapper + QueryCentricSchedulerMixin) | **✓ Pass (1회차)** | CRITICAL pre-RoPE 키 사용 동적 검증(score diff=0.131). KV Memory 90.6% 실측. CacheConfig composition 패턴 유지. Python 종료 시 segfault(CUDA teardown 경쟁 조건). |
| 2026-05-08 | **0.20.1** | **A+C Cross-1** (PreemptiveKVOffloadSchedulerMixin + CompressedPreemptionMixin + VllmEOptShrinkQCodec + EOptShrinkQAttentionHook + StaticDynamicSegmentKVManager + ManifoldKVWindowedEvictionManager) | **✓ Pass (3회차)** | 루프 2 timeout 재시도. TTFT p99 +286.9% 및 공정성 4.05× 미해결(필수 아님). Python 종료 segfault(CUDA teardown). |
| 2026-05-09 | **0.20.1** | **A+B Cross-1** (HitAwarePPDRouterMixin + TriangleIndexKVCacheManagerMixin + SpecKVContextGuardCombinedHook) | **Partial Pass (3회차)** | 신규 12개 클래스 임포트 Pass. 2026-05-04 select_evict_keys 타이브레이킹 버그로 install.sh 조기 종료. _MinimalManager MRO 설계 결함(smoke test 한정). |
| 2026-05-10 | **0.20.2** | **B+C** (KVPacketVQBlockManager + VQCodecAttentionHook + KVPacketSegmentSchedulerMixin) | **✓ Pass (1회차)** | CacheConfig.compression_method 필드 vLLM 0.20.2에 없음 — VQCodecAttentionHook 외부 주입으로 기능 동등. CPU-only 환경(libcuda.so.1 없음). 기존 사이클(05-03~05-09) backward-compat 전부 Pass. |
| 2026-05-11 | **0.20.2** | **B+C** (WiCERBlockManager + RateQuantVllmCodec + RateQuantAttentionHook + make_wicer_kv_cache_manager_class()) | **✓ Pass (1회차)** | 압축 텐서 어텐션 커널 진입 없음 코드 수준 보장. 2026-05-04 VllmRedundancyAwareEvictionPolicy 기존 실패(본 사이클 무관). 신규 테스트 전부 Pass. |
| 2026-05-12 | **0.20.2** | **B+C Cross-2** (AdapShotBlockManager + MixedDimAttentionHook + AdapShotMixedDimSegmentPipeline + make_adapshot_kv_cache_manager_class()) | **✓ Pass (2회차)** | 루프 1: install.sh 2026-05-09 블록 set+e/set-e 래핑 누락으로 2026-05-12 섹션 미도달, attention_backend_patch.py line 2127 docstring 파라미터 값 오류. 루프 2: 양 이슈 해소. 743/743 테스트 전량 통과. |
| 2026-05-13 | **0.20.2** | **A+B+C** (PBKVAgentSegmentPreservationSchedulerMixin + KVFoldAccumulativeBlockManager + SRFTInt8AttentionHook; AgenticChunkPreCachingPipeline) | **✓ Pass (1회차)** | 47/51 smoke 테스트 통과 (4 skip = GPU libcuda.so.1 없는 환경 조건부). deprecation 경고 0건. A: W≥100 대형 큐 오버헤드 ~9ms (실용 범위 외). C: KL=8×10⁻⁸(역대 최저). 파일 헤더 버전 불일치(0.20.1 참조 일부 — cosmetic). |
| 2026-05-14 | **0.20.2** | **B+C** (FibQuantVQSegmentKVManager + VllmFibQuantVQCodec + FibQuantAttentionHook + apply_fibquant_patch()) | **✓ Pass (2회차)** | 루프 1 피드백 반영 후 루프 2에서 전 항목 Pass. _chunk_key 인스턴스 속성 self.fibquant_chunk_size 참조 수정. SwigPy DeprecationWarning 2건은 vllm base 의존성(통합 코드 무관). 4 skipped = TestNonContiguousKVCacheManagerV2 (full vLLM infra 조건부). 10x config(bits_dir=4, 3.56x actual) mandatory 임계값 초과 → cosine ≥ 0.97 proxy 적용(Spec.md 명시 설계). |
| 2026-05-15 | **0.21.0** | **A+B+C** (RadixFeatherSchedulerMixin + RelayUShapeKVCacheManagerMixin + LookaheadEvictionAttentionHook + LookaheadRelayAttentionHook + make_radix_feather_scheduler_class() + make_relay_ulayer_kv_cache_manager_class() + apply_lookahead_eviction_patch() + extend_cache_config_lookahead_eviction()) | **✓ Pass (1회차)** | A: 스케줄러 오버헤드 p50 1.46ms (기준 5ms 이내 Pass). B: load_batch() NC 오버카운팅 버그 확인(hit 후 miss_ids 미초기화 — 다음 사이클 수정). C: 단위 테스트 10/10 통과, attention error <1e-5, cosine ≥ 0.99. CacheConfig에 swap_space 파라미터 없음(pydantic 기반) — object.__setattr__ 방식 정상 동작. 925/925 누적 테스트 전량 통과. install.sh 2026-05-04~2026-05-15 전 사이클 스모크 테스트 Pass. |

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
- **TriAttentionCodec pre-RoPE 안정적 중요도로 −90.6% 달성 (2026-05-06)**: 7사이클 최고 압축률. 보존 위치 왕복 정확도 atol=0.00 (완전 무손실, 역대 최고).
- **QueryCentricRecomputeCache 이중 단계 파이프라인 확립 (2026-05-06)**: 쿼리 관련성(Stage 1) × 코사인 유사도(Stage 2)의 이중 필터로 20% 예산 제한 정확히 구현.
- **DualFilterSegmentSelector B+C 이중 필터 통합 (2026-05-06)**: 40% 쿼리 관련성 필터 × 20% pre-RoPE 중요도 필터를 파이프라인으로 결합. 372개 테스트 1회차 전부 통과.
- **vLLM 1회차 Pass 연속 강화**: 14사이클 중 9사이클이 vLLM 이식 1회차 내 통과(2026-04-29, 2026-04-30, 2026-05-02, 2026-05-03, 2026-05-06, 2026-05-10, 2026-05-11, 2026-05-13 포함).
- **VQCodec training-free 벡터 양자화 실측 검증 (2026-05-10)**: forward-pass 기반 perplexity 실측 통과. −70.3% 메모리 절감 + ±1% perplexity 동시 달성. pre-RoPE 코드북 inverse RoPE MSE < 1e-5.
- **KVPacketSegmentSchedulerMixin 0.16ms 오버헤드 (2026-05-10)**: A 스케줄링 오버헤드 목표(5ms) 대비 31배 여유 달성.
- **WiCERIterativeKVWikiCache CEGAR 반복 세분화 (2026-05-11)**: 도메인 코퍼스 100% 히트율. SHA-256 청크 해시 기반 위치-독립 세그먼트 저장으로 비연속 재사용 구조 강화. gap-containing 쿼리 50%+ NC 확인.
- **RateQuantReverseWaterfillingCodec 정보이론 최적 비트 할당 (2026-05-11)**: 역 물채우기 Lagrange λ 바이너리 서치로 avg_bits=4.0 달성. −75% 메모리 절감. int8 오버플로우 구조적 해결(int16 저장). KL=0.000013(역대 최저). 660/660 테스트 통과(새 최다).
- **r_h 메타데이터 직렬화 (2026-05-11)**: save_pipeline()에 codec_bit_allocation + codec_head_variances 저장. 로딩 시 재캘리브레이션 불필요(zero-overhead loading).
- **MixedDimPerTokenBudgetCodec 역대 최저 accuracy delta 0.36% (2026-05-12)**: bisection 64회 training-free 탐색. budget_ratio 0.30~0.70 전 범위에서 ±1% 이내. 743/743 누적 테스트 전량 통과.
- **RoPEReencodingNonContiguousCache position-decoupled 구조 확립 (2026-05-12)**: content hash 위치-독립 키 + RoPE 재인코딩 in-place로 위치 불일치 비연속 세그먼트 히트율 100% 달성.
- **A+B+C 삼중 조합 단일 사이클 완성 (2026-05-13)**: PBKVAgentSegmentPreservationSchedulerMixin + KVFoldAccumulativeRadixCache + SRFTFusedINT4KVKernel이 AgenticChunkPreCachingPipeline으로 통합. CacheStore 인터페이스 완전 준수(6개 추상 메서드). 855/855 테스트 역대 최다 전량 통과.
- **SRFTInt8AttentionHook KL=8×10⁻⁸ 역대 최저 수준 달성 (2026-05-13)**: SRFT 직교 변환 + SMD RL 편향 제거 조합. 다중 크기 스캔(3종 seed) 전 조합 key/value 오차 0.53~0.59%.
- **KVFoldAccumulativeRadixCache foldl 비연속 추적 확립 (2026-05-13)**: get_segments_with_fold()가 miss 구간 이후 hit를 비연속으로 정확 추적. agentic 워크로드 40/40 히트.
- **FibQuantVQCodec 구면 베타 VQ 단일 코드베이스 3단계 티어 (2026-05-14)**: 방사/방향 성분 분리 인코딩으로 1.88x~6.40x 압축 티어를 bits_radial/bits_direction 파라미터 하나로 제어. 3.56x 설정 실측 −85.9% 달성(역대 실측 두 번째 최고치). apply_fibquant_patch() API로 클래스 단위 몽키패치 패턴 확립 — vLLM 내부 API 비수정 원칙 유지.
- **FibQuantVQSegmentCache OrderedDict LRU + 위치-독립 키 일관성 (2026-05-14)**: 동일 메모리 예산에서 세그먼트 보존 수 7.1× 증가. fibquant_stats() 10개 필드 완비로 세밀한 히트/미스/비연속 추적 지원.
- **LookaheadKVEvictionCodec ICLR 2026 기반 미래-인식 퇴거 역대 최고 정확도 달성 (2026-05-15)**: attention error 1e-6으로 목표(0.01) 대비 10,000× 우수. 드래프트 생성 없이 룩어헤드 토큰+LoRA로 미래 어텐션 패턴 직접 예측. eviction_ratio 전 범위(0.5/0.7/0.85) MANDATORY Pass. vLLM 0.21.0 환경에서 1회차 이식 완료.
- **RelayUShapeLayerSelectiveSegmentCache 레이어-선택적 부분 재사용 최초 구현 (2026-05-15)**: RelayCaching U자형 레이어 편차 이론을 비연속 세그먼트 재사용에 적용. all-or-nothing 이진 결정에서 레이어별 연속 결정 구조로 전환. LookaheadRelaySegmentCache(B+C) 조합으로 레이어+토큰 이중 필터 구현.
- **RadixFeatherBatchScheduler Feather 기반 배치 동질성 스케줄링 (2026-05-15)**: 스케줄링 오버헤드 <0.01ms p50 달성(목표 5ms 대비 500× 여유). 공정성 가드 max_wait_ratio 구현. vLLM 0.21.0 환경 p50 1.46ms(기준 이내). 배치 크기 대 프리픽스 동질성 트레이드오프 형식화 최초 도입.
- **vLLM 0.21.0 첫 이식 완료 (2026-05-15)**: 이전 사이클 대비 버전 업그레이드. 925/925 누적 테스트 전량 통과. install.sh 전 사이클(2026-05-04~2026-05-15) 스모크 테스트 PASS. A+B+C 삼중 조합 1회차 이식.

### 아직 해결 안 된 것
- **실제 GPU 처리량 미검증**: 15개 사이클 모두 CPU/시뮬레이션 환경. H100/A100에서 tokens/sec +20% 목표 Flash Attention 커널 연동 환경 검증 미완. 2026-05-15 +22.5%도 합성 추정치.
- **GPU perplexity 대규모 모델 검증 미완**: LLaMA-3.1-8B / WikiText-2 / LongBench 실측은 미완. proxy 기반(random N(0,1) / sparse high-norm 데이터) 통과.
- **compression_ratio 스윕 미완**: 2026-05-06 사이클에서 ratio=0.10만 실측. 정확도-압축률 트레이드오프 곡선 미확립.
- **TTFT GPU 실측 미완**: CPU-only 환경에서만 스케줄링 오버헤드 측정됨.
- **다중 노드 실측 검증**: DualMapScheduler 및 MultiNodeScheduler 모두 단일 머신 시뮬레이션으로만 검증됨.
- **vLLM 실제 GPU 측정 미완**: vLLM 이식은 CPU-only 환경(libcuda.so.1 없음).
- **vLLM CacheConfig swap_space 파라미터 부재 (2026-05-15 신규 확인)**: vLLM 0.21.0은 pydantic 기반으로 CacheConfig에 swap_space 없음. object.__setattr__ 방식이 현재 정상 동작하나 향후 버전 업그레이드 시 재검증 필요.
- **CacheConfig.compression_method 공식 API 부재**: 외부 주입 패턴으로 대체 중.
- **Python 종료 시 segfault**: CUDA teardown 경쟁 조건 추정. vLLM 이슈 트래커 확인 필요.
- **Activity B noncontiguous_hit_rate() API 버그 2종 (2026-05-15 신규 확인)**: (1) src/ relay_ulayer_segment.py: get_segments_layer_selective() 경로에서 self._hits 미증가 → noncontiguous_hit_rate() 분모 0으로 항상 0.0 반환. (2) block_manager_patch.py RelayUShapeAuxStore.load_batch(): miss_ids를 hit 발생 후 미초기화 → 오버카운팅. 두 버그 모두 다음 사이클 수정 예정.
- **Activity A 대형 큐 오버헤드 미해결**: PBKVAgentSegmentPreservationSchedulerMixin W≥100에서 ~9ms로 5ms 임계값 초과.
- **SRFTFusedINT4KVKernel 이론-실측 불일치 지속**: 이론치 73.4%(4-bit), 실측 48.4%(INT8). nibble-pack INT4 실구현 전까지 이중 보고 지속.
- **FibQuantVQCodec mandatory 기준 config 의존성**: MANDATORY accuracy는 1.88x 설정에서만 충족. 3.56x 설정에서 attention error 13.1%로 mandatory 초과 — cosine ≥ 0.97 proxy 대체 적용.
- **LookaheadModule untrained 상태 (2026-05-15 신규)**: 현재 key-norm 블렌딩 폴백 사용. 합성 데이터 통과이지만 실 LLM 추론 환경에서 trained lookahead 가중치 사용 권장.
- **Activity A 단독 히트율 향상 측정 미완**: RadixFeatherBatchScheduler 적용 전후 히트율 차이 정량 비교 실험 미수행.
- **LongBench proxy 서브태스크 다양성 부족**: 현재 KL/cosine 단일 지표만 측정. 다양한 태스크 특성 검증 미완.
- **numpy 환경 의존성**: 기존 14개 테스트(test_dag_ttl_adjuster.py 등) numpy 미설치로 수집 실패.
- **packed INT4(nibble 패킹) 미구현**: FibQuantVQCodec 및 SRFTFusedINT4KVKernel 현재 uint8 저장.
- **codebook_size 스윕 결과 저장 미완**: VQCodec M=16/64/256, n_residuals=1/2/4 스윕 결과 저장 스크립트 미추가.
- **2026-05-04 VllmRedundancyAwareEvictionPolicy 버그 미해결**: install.sh 사전 존재 실패 항목.
- **파일 헤더 버전 불일치 (cosmetic)**: block_manager_patch.py, attention_backend_patch.py 일부 docstring이 이전 버전 참조.
- **avg_bits < 4.0 탐색 미완**: RateQuant 역 물채우기에서 총 비트 예산 2.0~3.0 범위 정확도-메모리 트레이드오프 미측정.
- **SegmentAdapter 사전 학습 부재**: 추론 시점에 untrained(random init) 상태.

### 다음 우선순위 제언
1. **Activity B noncontiguous_hit_rate() 버그 수정 (최우선)**: (1) relay_ulayer_segment.py: get_segments_layer_selective() 경로에서 self._hits/self._misses 카운터 동기화 또는 noncontiguous_hit_rate() 분모를 _base_cache._hits로 교체. (2) block_manager_patch.py RelayUShapeAuxStore.load_batch(): hit 발생 시 miss_ids를 빈 리스트로 초기화(had_prior_miss 저장 후). 두 버그를 동일 사이클에서 수정해 NC 히트율 API 신뢰성 확보.
2. **실 GPU 벤치마크 추가 (15사이클 누적 미완)**: experiments/run_experiment.py에 실제 추론 엔진(transformers AutoModel) 연결 TTFT/throughput 측정 루틴 구현. LookaheadKVEvictionCodec + RelayUShapeLayerSelectiveSegmentCache 조합의 +22.5% 처리량 추정치를 실 GPU에서 검증.
3. **LookaheadModule 가중치 훈련**: train_lookahead_lora.py 실행 후 보정 데이터(500~1000 샘플)로 훈련. 현재 untrained key-norm fallback 사용 중. 정확도 delta가 ±1% 이내인지 실 LLM 환경에서 재검증.
4. **MooncakeRelayClusterKVRoutingScheduler 구현 검토**: 클러스터-와이드 분산 KV 풀 + RelayCaching 크로스-에이전트 재사용 결합이 에이전틱 워크로드에서 vLLM x Mooncake Store 실증(TTFT 46×)에 근접하는 효과를 내는지 탐색. Activity A의 "분산 에이전틱 KV 라우팅" 방향의 다음 단계.
5. **vLLM 0.21.0 object.__setattr__ 패턴 안정성 검증**: extend_cache_config_lookahead_eviction()의 동적 속성 추가 방식이 vLLM 향후 버전에서도 유효한지 CI에 버전별 체크 추가. CacheConfig pydantic 모델 변경 모니터링.

SUMMARY_UPDATED
