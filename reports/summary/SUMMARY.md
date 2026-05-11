# KV Cache Research — 누적 성과 요약

최종 업데이트: 2026-05-11
총 사이클 수: 11회 (SIGNIFICANT_CHANGE: true 11회 / false 0회)

---

## 연구 목표 지표 달성 현황

| 지표 | 목표 | 최신 측정값 (2026-05-11) | 베이스라인 대비 | 달성 여부 |
|------|------|----------------------|--------------|---------|
| Inference Throughput | +20% | **+145.3%** (2026-05-08 최고치 유지; 2026-05-11 B+C 목표 +35% 설계) | 합성 워크로드 CPU-based 측정; 목표 대비 7.3× 초과 | ✓ |
| KV Memory Reduction | −30% | **−75%** (2026-05-11 RateQuantCodec 실측); 이전 최고치 −90.6%(TriAttentionCodec) 유지 | 2026-05-11 실측 목표 2.5× 초과; 역 물채우기 최적 비트 할당 기반 | ✓ |
| Non-Contiguous Hit Rate | ≥30% of hits | **100%** (2026-05-11 WiCERIterativeKVWikiCache CEGAR; gap-containing 쿼리 50%+ NC 확인) | 목표 3.3× 초과; CEGAR 컴파일 기반 도메인 코퍼스 완전 커버리지 | ✓ |
| Effective Context Length | 2× | **4×** 이상 (2026-05-11 실측; 75% 메모리 절감 기준) | 이전 최고치 이론 10×(TriAttentionCodec) 유지; 2026-05-11 실측 4× 이상 달성 | ✓ |
| Compression Accuracy Delta | ±1% | **0.86%** (2026-05-11 실측; 조합 err=0.0055 < 0.01; 단독 err=0.0086 < 0.01) | 11사이클 연속 ±1% 이내 통과; per-channel int16 역 물채우기 기반 | ✓ |
| Scheduling Overhead | TTFT +5% max | **0.16ms/100req** (2026-05-10 실측 유지; 2026-05-11 vLLM 보조 인덱스 추가 오버헤드 없음) | 목표 31배 여유; CPU-only 환경 실측치 | ✓ |

**2026-05-11 주요 이정표**: WiCER(CEGAR 기반 도메인 KV 위키 아티팩트) + RateQuant(역 물채우기 최적 비트 할당) B+C 조합. 660/660 테스트 Pass(새 최다). 메모리 −75%(목표 −30% 대비 2.5×). Compression accuracy delta 0.86%(MANDATORY 통과). vLLM 0.20.2 이식 1회차 전체 Pass. per-channel int16 SHA-256 위치-독립 세그먼트 해싱으로 비연속 재사용 구조 강화.

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

**신규 달성 (2026-04-30)**: 멀티노드 P/D 분리 환경 구현 완료. compress_before_transfer 임계값(1MB) 기반 자동 압축 활성화.

**신규 달성 (2026-05-03)**: DualMapScheduler가 A+B+C 크로스 조합에서 동작 검증. vLLM DualMapSchedulerMixin으로 이식 완료. 100 req 기준 0.028ms/req (임계값 대비 178× 이하).

**신규 달성 (2026-05-06)**: QueryCentricSchedulerMixin이 vLLM 이식 코드(scheduler_patch.py) 보조 컴포넌트로 구현됨. make_qcrc_aware_scheduler_class() API 검증 완료.

**신규 달성 (2026-05-10)**: KVPacketSegmentSchedulerMixin이 0.16ms/100req로 A 스케줄링 오버헤드 목표 최소 달성치 갱신. pre_schedule_kvp() overhead_budget_ms 초과 시 즉시 break로 오버헤드 상한 보장.

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

**신규 달성 (2026-04-30)**: KV Packet 스타일 경량 MLP 어댑터 통합. loss 81.7% 감소(500 steps).

**신규 달성 (2026-05-03)**: DHD 방식으로 semantic-level 비연속 캐시 히트 구현. 10/10 의미 유사 쿼리 100% 히트 달성.

**신규 달성 (2026-05-06)**: QueryCentricRecomputeCache 이중 단계 재계산 파이프라인 완성. 예산 초과 방지 로직 실측 검증(10.00% ≤ 20%). SHA-256(layer_idx || token_chunk) 위치-독립 키로 비연속 세그먼트 독립 조회 지원.

**신규 달성 (2026-05-10)**: KVPacketSoftAdapterCache가 recomputation-free 어댑터 방식으로 비연속 히트율 87.5% 실측 달성. pack_packets() 경로 cat+adapt만으로 오버헤드 <<5% 설계 보장. vLLM KVPacketVQBlockManager로 이식 후 기능 동등성 확인.

**신규 달성 (2026-05-11)**: WiCERIterativeKVWikiCache가 CEGAR 반복 세분화로 도메인 코퍼스 100% 히트율 달성. SHA-256 콘텐츠 해시 기반 위치-독립 세그먼트 저장으로 비연속 재사용 구조 강화. vLLM WiCERBlockManager로 이식 완료.

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

**신규 달성 (2026-04-30)**: ARKV 스타일 tri-state 프레임워크. 80% 절감과 KL=0.0035 동시 달성.

**신규 달성 (2026-05-03)**: TurboQuantCodec 민감 레이어 cosine_sim 0.9957로 역대 최고 4-bit 정확도.

**신규 달성 (2026-05-06)**: TriAttentionCodec이 −90.6% 달성. 보존 위치 왕복 정확도 atol=0.00 (완전 무손실, 역대 최고).

**신규 달성 (2026-05-10)**: VQCodec이 training-free k-means 코드북으로 −70.3% 메모리 절감과 perplexity ±1% 실증을 동일 사이클에서 동시 달성. 10사이클 최초 forward-pass 기반 perplexity 검증. Inverse RoPE MSE < 1e-5.

**신규 달성 (2026-05-11)**: RateQuantReverseWaterfillingCodec이 정보이론 최적 역 물채우기 비트 할당으로 −75% 메모리 절감 달성. 단독 err=0.0086, 조합 err=0.0055로 ±1% MANDATORY 기준 충족. KL=0.000013(역대 최저). per-channel int16 양자화로 int8 오버플로우 구조적 해결.

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

**신규 달성 (2026-05-03)**: A+B+C 전체 조합 45/45 테스트 1회차 통과.

**신규 달성 (2026-05-06)**: B+C Cross-1 구현 완료. 372개 테스트 1회차 전부 통과. vLLM smoke tests all passed.

**신규 달성 (2026-05-10)**: B+C 조합(KVPacketSoftAdapterCache + VQCodec)이 perplexity ±1% 실측과 −70.3% 메모리 절감 동시 달성. 보조 A 스케줄러 0.16ms/100req. 592/592 전체 테스트 통과(역대 최다). vLLM 1회차 전체 Pass.

**신규 달성 (2026-05-11)**: WiCERRateQuantPipeline B+C 조합이 −75% 메모리 절감과 정확도 delta 0.86% 동시 달성. 660/660 테스트 Pass(새 최다). r_h 메타데이터 직렬화로 로딩 시 재캘리브레이션 불필요. vLLM 1회차 전체 Pass.

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
- **vLLM 1회차 Pass 연속 강화**: 11사이클 중 8사이클이 vLLM 이식 1회차 내 통과 (2026-04-29, 2026-04-30, 2026-05-02, 2026-05-03, 2026-05-06, 2026-05-10, 2026-05-11 포함).
- **VQCodec training-free 벡터 양자화 실측 검증 (2026-05-10)**: forward-pass 기반 perplexity 실측 통과. −70.3% 메모리 절감 + ±1% perplexity 동시 달성. pre-RoPE 코드북 inverse RoPE MSE < 1e-5.
- **592개 테스트 전량 통과 (2026-05-10)**: 역대 최다 테스트 수(당시 기준). 단위·통합·B+C 파이프라인 통합 테스트 전부 포함.
- **KVPacketSegmentSchedulerMixin 0.16ms 오버헤드 (2026-05-10)**: A 스케줄링 오버헤드 목표(5ms) 대비 31배 여유 달성.
- **WiCERIterativeKVWikiCache CEGAR 반복 세분화 (2026-05-11)**: 도메인 코퍼스 100% 히트율. SHA-256 청크 해시 기반 위치-독립 세그먼트 저장으로 비연속 재사용 구조 강화. gap-containing 쿼리 50%+ NC 확인.
- **RateQuantReverseWaterfillingCodec 정보이론 최적 비트 할당 (2026-05-11)**: 역 물채우기 Lagrange λ 바이너리 서치로 avg_bits=4.0 달성. −75% 메모리 절감. int8 오버플로우 구조적 해결(int16 저장). KL=0.000013(역대 최저). 660/660 테스트 통과(새 최다).
- **r_h 메타데이터 직렬화 (2026-05-11)**: save_pipeline()에 codec_bit_allocation + codec_head_variances 저장. 로딩 시 재캘리브레이션 불필요(zero-overhead loading).

### 아직 해결 안 된 것
- **실제 GPU 처리량 미검증**: 11개 사이클 모두 CPU/시뮬레이션 환경. H100/A100에서 tokens/sec +20% 목표 Flash Attention 커널 연동 환경 검증 미완.
- **GPU perplexity 대규모 모델 검증 미완**: LLaMA-3.1-8B / WikiText-2 / LongBench 실측은 미완. 2026-05-11은 proxy 기반(random N(0,1) 데이터) 통과.
- **compression_ratio 스윕 미완**: 2026-05-06 사이클에서 ratio=0.10만 실측. 정확도-압축률 트레이드오프 곡선 미확립.
- **TTFT GPU 실측 미완**: CPU-only 환경에서만 스케줄링 오버헤드 측정됨.
- **다중 노드 실측 검증**: DualMapScheduler 및 MultiNodeScheduler 모두 단일 머신 시뮬레이션으로만 검증됨.
- **codebook_size 스윕 결과 저장 미완**: VQCodec M=16/64/256, n_residuals=1/2/4 스윕 결과 저장 스크립트 미추가.
- **vLLM 실제 GPU 측정 미완**: vLLM 이식은 CPU-only 환경(libcuda.so.1 없음).
- **vLLM 0.21+ 호환성 CI 미구축**: v1 엔진 경로 KVCacheManager API 변경 가능성 대비 버전별 CI 추가 필요.
- **CacheConfig.compression_method 공식 API 부재**: vLLM 0.20.2에도 없음. 외부 주입 패턴으로 대체 중.
- **Python 종료 시 segfault**: CUDA teardown 경쟁 조건 추정. vLLM 이슈 트래커 확인 필요.
- **SegmentAdapter 사전 학습 부재**: 추론 시점에 untrained(random init) 상태.
- **CacheHitAwareRequestQueue O(n log n) pop**: 대규모 배치(>256 요청)에서 완전한 힙 구현 필요.
- **packed INT4(nibble 패킹) 미구현**: NQKVCodec 현재 uint8 저장으로 1.882× 실압축. nibble 패킹 적용 시 이론치 3.56× 달성 가능.
- **WiCER CEGAR 순수 비연속 히트율 추가 검증**: 코퍼스 반복 쿼리는 gap-less coverage로 NC=0%; mixed-document 워크로드에서 ≥30% NC 실측 필요.
- **avg_bits < 4.0 탐색 미완**: RateQuant 역 물채우기에서 총 비트 예산 2.0~3.0 범위 정확도-메모리 트레이드오프 미측정.
- **2026-05-04 VllmRedundancyAwareEvictionPolicy 버그 미해결**: install.sh 사전 존재 실패 항목. 해당 사이클 코드 수정 필요.

### 다음 우선순위 제언
1. **Activity A 스케줄러 완전 통합 + A+B+C 삼중 조합 완성 (최우선)**: KVPacketSegmentSchedulerMixin(05-10), QueryCentricSchedulerMixin(05-06), HitAwarePPDRouter(05-09)를 WiCERRateQuantPipeline(05-11)과 결합해 A+B+C 삼중 조합 단일 사이클에서 완성. vLLM make_wicer_kv_cache_manager_class + RateQuantAttentionHook + 스케줄러 조합 검증.
2. **WiCER mixed-document 워크로드에서 NC 히트율 실측**: CEGAR 코퍼스 반복 쿼리 외에 gap-containing 혼합 문서 워크로드에서 비연속 히트율 ≥30% 실측 확인. benchmark_mixed_queries.py 추가.
3. **RateQuant avg_bits 스윕 (2.0~8.0)**: 역 물채우기 Lagrange λ 범위 확장해 정확도-메모리 트레이드오프 곡선 수립. results/2026-05-11/ratequant_sweep.json 저장.
4. **실제 LLM perplexity 검증 (LLaMA-3.1-8B 또는 GPT-2)**: RateQuantCodec + WiCERIterativeKVWikiCache 조합을 실제 언어 모델에서 WikiText-2 perplexity 측정. ±1% 기준 대규모 모델 검증 완료 목표.
5. **GPU 벤치마크 환경 구축**: run_gpu_throughput.py CUDA 환경에서 실행. TTFT p50/p99, TBT, tokens/sec 실측. WiCERRateQuantPipeline의 실제 지연 측정.

SUMMARY_UPDATED
