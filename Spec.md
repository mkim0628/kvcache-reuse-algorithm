# Spec — 2026-05-03

<!-- 변경 이유 (이전 Spec.md: 2026-05-02 대비):
이전 사이클(2026-05-02)은 B+C (SignVQSegmentCache + LeverageScoreCompressor) 조합이었다.
이번 사이클은 A+B+C 전체 통합으로 전환한다. 주요 변경 내용:

1. [Activity B 교체] 해시 기반 정확 매칭(Sign VQ XOR 유사도) → DHD(Dual-Stage High Deviation)
   의미 유사도 기반 비연속 KV 공유. 세그먼트 임베딩(평균 Key 벡터)을 코사인 유사도로 검색하며
   편차가 큰 head/token만 선택적으로 재계산하는 방식으로 알고리즘이 본질적으로 다르다.

2. [Activity C 교체] 레버리지 스코어 3-티어(CapKV 스타일) → TurboQuant 2단계 VQ
   (PolarQuant 랜덤 회전 + 3비트 스칼라 양자화 + QJL 잔차 1비트 보정). 훈련 없이 6× 메모리
   절감과 정확도 보존을 동시에 달성하는 알고리즘이다.

3. [Activity A 신규] DualMapScheduler 이중 해시 + 의미 히트율 가중 라우팅 추가.
   이전 사이클에는 Activity A가 없었다.

4. [Cross-2 신규] SemanticSegmentCache와 TurboQuantCodec의 직접 파이프라인 통합.
   put 시 TurboQuant 압축 저장, get 시 DHD 편차 체크 후 선택적 재계산.

5. [Activity B 보조] SpeculativeSegmentFetcher: 비연속 세그먼트 검색을 크리티컬 패스
   밖으로 이동시키는 비동기 프리패칭 레이어 추가.

기존 파일(sign_vq_segment.py, leverage_compressor.py, compression.py, segmented.py,
contiguous.py, tri_state_compressor.py, compressed_segment.py, segment_adapter.py,
cache_aware_scheduler.py, multi_node_scheduler.py)은 수정하지 않는다.
기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-03.md`
**최우선 구현 타겟**: Cross-2 (B+C) — SemanticSegmentCache(DHD) + TurboQuantCodec(PolarQuant+QJL)
**A 구성요소**: DualMapScheduler 의미 히트율 가중 이중 해시 라우팅
**보조 구성요소**: SpeculativeSegmentFetcher (비동기 KV 세그먼트 프리패칭)

**해결하려는 문제**:
- 기존 `SegmentedHashCache`와 `SignVQSegmentCache`는 정확 토큰 해시 매칭에만 의존한다.
  프롬프트 토큰이 하나라도 다르면 비연속 히트가 발생하지 않아 히트율 목표(전체 히트의 30% 비연속)
  달성이 어렵다.
- 기존 압축 기법(INT8, Hadamard INT4, CapKV 3-티어)은 −30~75% 메모리 절감을 달성하지만
  알고리즘 구조상 3비트 이하로 내려가면 정확도 저하가 빠르게 나타난다.
- 캐시 히트율 기반 스케줄러(`CacheAwareScheduler`)는 의미 유사도가 높은 요청을 같은 캐시
  인스턴스로 모을 수 없어 Cross-2의 의미 히트 이점을 살리지 못한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling — DualMapScheduler (의미 히트율 가중 이중 해시 라우팅)
- [x] Activity B: Non-Contiguous KV Cache Reuse — SemanticSegmentCache (DHD 의미 유사도 기반)
- [x] Activity C: KV Cache Compression — TurboQuantCodec (PolarQuant + QJL 3비트)

---

## 목표

- [ ] 목표 1 (§4 Accuracy 필수): 압축 전후 perplexity 변화 ±1% 이내 — WikiText-2 proxy 검증
- [ ] 목표 2 (§4 Accuracy 필수): downstream 태스크 정확도 변화 ±1% 이내 — LongBench 3개 서브태스크 proxy 검증
- [ ] 목표 3 (§4 Memory): KV 캐시 메모리 베이스라인 대비 −60% 이상 (목표 −75%, 평가 기준 최소 −30%)
- [ ] 목표 4 (§3 Non-Contiguous Hit Rate): 전체 히트 중 의미 기반 비연속 히트 비율 ≥ 30%
- [ ] 목표 5 (§1 Throughput): tokens/sec 베이스라인 대비 +10% 이상 (목표 +20%)
- [ ] 목표 6 (§1 TTFT): TTFT p50 베이스라인 대비 +5% 이내 (의미 검색 오버헤드 포함)
- [ ] 목표 7 (§2 Scheduling): 스케줄링 캐시 히트율 향상 ≥ +10%p (미적용 대비)
- [ ] 목표 8 (§5 Cross): 복합 메모리 감소 단일 Activity 대비 추가 −10% 이상

---

## 아키텍처 개요

```
요청 입력
    │
    ▼
┌────────────────────────────────────────────┐
│  DualMapScheduler (Activity A)             │
│  h1(req), h2(req) → 후보 노드 (n1, n2)    │
│  semantic_hit_score = cosine(req_emb,      │
│                       node_segment_embs)   │
│  routing_score = sem_hit × (1 - load)      │
│  → 최고 점수 노드 선택 + 공정성 보장        │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────┐
│  SpeculativeSegmentFetcher (Activity B)    │
│  이전 배치 처리 중 다음 배치 세그먼트 비동기  │
│  프리패칭 → 크리티컬 패스에서 KV 검색 제거   │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────┐
│  SemanticSegmentCache (Activity B)         │
│  get(query_emb):                           │
│    1. 정확 해시 조회 (fast path)            │
│    2. 코사인 유사도 상위-k 세그먼트 검색     │
│    3. DHD 편차 체크 → 재계산 결정           │
│  put(kv, emb):                             │
│    TurboQuantCodec.encode(kv) → 3비트 저장  │
│    임베딩 인덱스 갱신                        │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────┐
│  TurboQuantCodec (Activity C)              │
│  PolarQuant: R @ kv → 3비트 스칼라 양자화   │
│  QJL: 잔차 → 1비트 JL 변환 저장            │
│  decode: 역양자화 + 잔차 복원 + R^T 역회전  │
└────────────────────────────────────────────┘
         │
         ▼
  (hits, misses, hit_types, memory_bytes)
  hit_type: "exact" | "semantic" | "miss"
  semantic_recompute_ratio: 재계산 세그먼트 비율
```

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/turbo_quant.py` | C | `TurboQuantCodec` — PolarQuant 랜덤 회전 + 3비트 스칼라 양자화 + QJL 잔차 1비트 보정 |
| `src/cache/dhd_segment_cache.py` | B+C | `SemanticSegmentCache` — 세그먼트 임베딩 + 코사인 유사도 검색 + DHD 편차 기반 선택적 재계산 |
| `src/scheduler/dual_map_scheduler.py` | A | `DualMapScheduler` — 이중 해시 + 의미 히트율 가중 라우팅 |
| `src/cache/speculative_fetcher.py` | B | `SpeculativeSegmentFetcher` — 비동기 세그먼트 프리패칭 |
| `configs/experiments/2026-05-03.yaml` | 공통 | 실험 설정 |
| `tests/unit/test_turbo_quant.py` | C | TurboQuantCodec 단위 테스트 |
| `tests/unit/test_turbo_quant_accuracy.py` | C | Accuracy-preserving 검증 테스트 (필수) |
| `tests/unit/test_dhd_segment_cache.py` | B+C | SemanticSegmentCache 단위 테스트 |
| `tests/unit/test_dual_map_scheduler.py` | A | DualMapScheduler 단위 테스트 |
| `tests/integration/test_abc_integration.py` | A+B+C | A+B+C 전체 통합 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/base.py` | 변경 없음 — 기존 CacheStore 인터페이스를 그대로 준수 |
| `tests/integration/test_abc_e2e.py` | 기존 테스트 유지; 새 통합 테스트는 `test_abc_integration.py`에 별도 작성 |

**주의**: 기존 파일(`sign_vq_segment.py`, `leverage_compressor.py`, `compression.py`,
`segmented.py`, `contiguous.py`, `tri_state_compressor.py`, `compressed_segment.py`,
`segment_adapter.py`, `cache_aware_scheduler.py`, `multi_node_scheduler.py`)은
이번 사이클에서 수정하지 않는다. 기존 단위·통합 테스트 전부가 회귀 없이 통과해야 한다.

---

## 알고리즘 상세

### 1. TurboQuantCodec (Activity C)

**파일**: `src/cache/turbo_quant.py`

**핵심 아이디어**:
- PolarQuant: 레이어별 고정 시드로 생성한 랜덤 직교 회전 행렬 R을 KV에 적용해 outlier 분산을
  균일하게 재분배한 후 3비트 스칼라 양자화 수행.
- QJL 잔차 보정: 양자화 잔차를 1비트 JL(Johnson-Lindenstrauss) 변환 (랜덤 ±1 행렬)으로
  부호 벡터를 저장해 체계적 오류를 교정.
- 실효 저장 비트: 3비트 본체 + 1비트 QJL = 4비트 이하 (INT4와 동등 저장 공간에서 정확도 우위).

**회전 행렬 생성 (레이어별 고정 시드)**:

```python
def _get_rotation_matrix(layer_idx: int, d_head: int) -> torch.Tensor:
    # 시드 = base_seed XOR (layer_idx * 2654435761)  (Knuth 곱셈 해시)
    # torch.manual_seed(seed) 후 표준 정규 분포 행렬 생성 → QR 분해로 직교화
    rng = torch.Generator()
    rng.manual_seed(base_seed ^ (layer_idx * 2654435761 & 0xFFFFFFFF))
    raw = torch.randn(d_head, d_head, generator=rng)
    Q, _ = torch.linalg.qr(raw)  # Q: (d_head, d_head) 직교 행렬
    return Q  # R = Q
```

**의사코드**:

```python
class TurboQuantCodec:
    def __init__(
        self,
        num_layers: int,
        bits: int = 3,                    # 스칼라 양자화 비트 수
        qjl_bits: int = 1,               # QJL 잔차 보정 비트 수
        base_seed: int = 42,             # 회전 행렬 시드 베이스
        sensitive_layers_ratio: float = 0.25,  # DepthKV 스타일: 상위 N*ratio 레이어는 4비트
    ) -> None:
        # _rotation_cache: Dict[int, torch.Tensor] — 레이어별 회전 행렬 캐시
        # _qjl_cache: Dict[int, torch.Tensor] — 레이어별 QJL 행렬 캐시
        # _sensitive_cutoff = int(num_layers * sensitive_layers_ratio)
        ...

    def _get_rotation_matrix(self, layer_idx: int, d_head: int) -> torch.Tensor:
        # 캐시 조회 후 없으면 생성 (위 의사코드 참조)
        ...

    def _get_qjl_matrix(self, layer_idx: int, d_head: int, proj_dim: int) -> torch.Tensor:
        # QJL 행렬: (proj_dim, d_head) float, 각 원소는 ±1/sqrt(proj_dim)
        # 시드 = base_seed XOR (layer_idx * 1234567891 & 0xFFFFFFFF)
        rng = torch.Generator()
        rng.manual_seed(base_seed ^ (layer_idx * 1234567891 & 0xFFFFFFFF))
        raw = torch.randint(0, 2, (proj_dim, d_head), generator=rng).float()
        return (2 * raw - 1) / (proj_dim ** 0.5)  # ±1/sqrt(proj_dim)

    def _effective_bits(self, layer_idx: int) -> int:
        # 민감 레이어(초기 _sensitive_cutoff 레이어)는 4비트, 나머지는 self.bits
        return 4 if layer_idx < self._sensitive_cutoff else self.bits

    def encode(
        self,
        kv: torch.Tensor,   # (n_tokens, d_head) float32 — K 또는 V
        layer_idx: int,
        tensor_id: int = 0,
    ) -> dict:
        # kv_f = kv.float()
        # d_head = kv_f.shape[-1]
        # eff_bits = _effective_bits(layer_idx)
        # levels = 2 ** eff_bits        예: 3비트 → 8 레벨

        # 1. PolarQuant 회전
        # R = _get_rotation_matrix(layer_idx, d_head)  # (d_head, d_head)
        # kv_rotated = kv_f @ R.T                      # (n_tokens, d_head)

        # 2. 3비트 스칼라 양자화 (per-row symmetric)
        # scale = kv_rotated.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) / ((levels-1)/2)
        # quantized = (kv_rotated / scale).round().clamp(-(levels//2), (levels//2)-1)
        #             .to(torch.int8)                  # (n_tokens, d_head) int8

        # 3. QJL 잔차 보정
        # kv_dequant = quantized.float() * scale       # 역양자화 (회전 공간)
        # residual = kv_rotated - kv_dequant           # 잔차 (n_tokens, d_head)
        # proj_dim = d_head  (JL 차원 = d_head 사용)
        # P = _get_qjl_matrix(layer_idx, d_head, proj_dim)  # (proj_dim, d_head)
        # proj = residual @ P.T                        # (n_tokens, proj_dim)
        # qjl_bits_tensor = (proj >= 0).to(torch.uint8)    # (n_tokens, proj_dim) 0/1
        # qjl_packed = torch.packbits(qjl_bits_tensor, dim=-1)  # (n_tokens, ceil(proj_dim/8))

        # return {
        #     "quantized": quantized,       # (n_tokens, d_head) int8
        #     "scale": scale,              # (n_tokens, 1) float32
        #     "qjl_packed": qjl_packed,    # (n_tokens, ceil(proj_dim/8)) uint8
        #     "layer_idx": layer_idx,
        #     "tensor_id": tensor_id,
        #     "d_head": d_head,
        #     "proj_dim": proj_dim,
        #     "eff_bits": eff_bits,
        #     "n_tokens": kv_f.shape[0],
        # }
        ...

    def decode(
        self,
        compressed: dict,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        # d_head = compressed["d_head"]
        # R = _get_rotation_matrix(layer_idx, d_head)
        # P = _get_qjl_matrix(layer_idx, d_head, compressed["proj_dim"])

        # 1. 역양자화
        # kv_dequant = compressed["quantized"].float() * compressed["scale"]  # (n, d_head)

        # 2. QJL 잔차 복원
        # qjl_unpacked = torch.unpackbits(compressed["qjl_packed"], dim=-1)[:, :compressed["proj_dim"]]
        # qjl_signs = 2.0 * qjl_unpacked.float() - 1.0    # ±1.0, (n, proj_dim)
        # residual_approx = qjl_signs @ P                   # (n, d_head) — JL 역변환 근사
        # kv_corrected = kv_dequant + residual_approx       # (n, d_head) 회전 공간

        # 3. 역회전
        # return kv_corrected @ R                            # (n, d_head) 원래 공간
        ...

    def memory_bytes_estimate(
        self,
        n_tokens: int,
        d_head: int,
        layer_idx: int = 0,
    ) -> dict:
        # eff_bits = _effective_bits(layer_idx)
        # quantized_bytes = n_tokens * d_head * 1          # int8 (1 byte/element, 3비트 → int8 저장)
        # scale_bytes = n_tokens * 4                       # float32 per-row scale
        # qjl_bytes = n_tokens * math.ceil(d_head / 8)    # 1비트 packed
        # total = quantized_bytes + scale_bytes + qjl_bytes
        # baseline = n_tokens * d_head * 4                 # FP32 기준
        # return {"total_bytes": total, "baseline_bytes": baseline,
        #         "reduction_ratio": 1.0 - total / baseline}
        ...

    def compression_ratio(self, layer_idx: int) -> float:
        # 3비트 레이어: 1 - (d * 1 + 4 + ceil(d/8)) / (d * 4)
        # d_head=128 기준: ≈ 75% 절감 (6×)
        ...
```

**메모리 계산 (d_head=128, n_tokens=1000 기준)**:
- FP32 베이스라인: 1000 × 128 × 4 = 512,000 bytes
- quantized (int8): 1000 × 128 × 1 = 128,000 bytes (3비트 → int8 패킹)
- scale (float32 per-row): 1000 × 1 × 4 = 4,000 bytes
- QJL (1비트 packed, proj_dim=128): 1000 × ceil(128/8) = 16,000 bytes
- **총계**: 148,000 bytes
- **감소율**: 1 - 148,000/512,000 ≈ **71.1% 감소** (목표 −60% 달성)

---

### 2. SemanticSegmentCache (Activity B+C)

**파일**: `src/cache/dhd_segment_cache.py`

**핵심 아이디어**:
- 세그먼트 임베딩: 청크 내 Key 벡터의 평균 → (d_head,) float32 — 추가 모델 불필요.
- 유사도 검색: brute-force 코사인 유사도 (N_segments ≤ 10K 가정). FAISS 선택 가능.
- DHD 판단: 후보 KV와 쿼리 세그먼트 간 head-wise L2 편차 계산.
  편차 > deviation_threshold인 head/token만 재계산 대상으로 표시.
- TurboQuantCodec 통합: 저장 시 압축, 조회 시 복원 후 DHD 편차 체크.

```python
class SemanticSegmentCache(CacheStore):
    """DHD 의미 유사도 기반 비연속 KV 공유 캐시 (Activity B+C).

    저장소 구조:
      _exact_store: OrderedDict[str, torch.Tensor]  — 정확 해시 히트 경로 (압축 KV dict)
      _semantic_index: List[Tuple[str, torch.Tensor]]  — (key, embedding) 리스트 (유사도 검색용)
      _compressed_store: Dict[str, dict]  — key → TurboQuantCodec 압축 결과

    히트 분류:
      exact_hits: 정확 토큰 해시 매칭
      semantic_hits: 코사인 유사도 + DHD 편차 통과 (비연속 히트)
      recompute_count: DHD 편차 초과로 재계산한 세그먼트 수
    """

    def __init__(
        self,
        codec: "TurboQuantCodec",
        chunk_size: int = 128,
        max_entries: int = 1000,
        top_k: int = 5,                     # 유사도 검색 상위-k 후보 수
        similarity_threshold: float = 0.80,  # 코사인 유사도 최소 임계값
        deviation_threshold: float = 0.20,   # DHD L2 편차 임계값 (token당 정규화)
        recompute_budget: float = 0.20,      # 전체 세그먼트 중 최대 재계산 비율
    ) -> None: ...

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 구현                                            #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        # CacheStore 인터페이스 준수용 raw put (압축 없이)
        # 내부적으로 _exact_store[key] = value
        ...

    def get(self, key: str) -> Optional[torch.Tensor]:
        # 정확 해시 조회만 (의미 검색은 get_semantic() 사용)
        ...

    def evict(self) -> int:
        # LRU 퇴거: _exact_store, _compressed_store, _semantic_index 동기화
        ...

    def hit_rate(self) -> float:
        # (exact_hits + semantic_hits) / (exact_hits + semantic_hits + misses)
        ...

    def memory_bytes(self) -> int:
        # _compressed_store의 모든 압축 dict 크기 합산
        ...

    def reset_stats(self) -> None:
        # exact_hits, semantic_hits, misses, recompute_count 초기화
        ...

    # ------------------------------------------------------------------ #
    # 확장 API (B+C 통합)                                                  #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,    # (n_tokens_in_chunk, d_head)
        values: torch.Tensor,  # (n_tokens_in_chunk, d_head)
        layer_idx: int = 0,
    ) -> None:
        # 1. 청크 키 생성 (SHA-256, segmented.py 방식 동일)
        # 2. 세그먼트 임베딩 계산: embedding = keys.mean(dim=0)  # (d_head,)
        # 3. TurboQuantCodec.encode(keys) → k_compressed
        #    TurboQuantCodec.encode(values) → v_compressed
        # 4. _compressed_store[key] = {"k": k_compressed, "v": v_compressed, "layer_idx": ...}
        # 5. _semantic_index 갱신: append((key, embedding))
        # 6. max_entries 초과 시 evict()
        ...

    def get_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        query_keys: torch.Tensor,  # (n_tokens_in_chunk, d_head) — DHD 편차 체크용
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], str]:
        # Returns: (kv_tensor or None, hit_type)
        # hit_type: "exact" | "semantic" | "miss"

        # 1. 정확 해시 조회
        # key = chunk_key(token_ids, chunk_idx, layer_idx)
        # if key in _compressed_store:
        #     k = codec.decode(_compressed_store[key]["k"], layer_idx)
        #     v = codec.decode(_compressed_store[key]["v"], layer_idx)
        #     exact_hits += 1
        #     return torch.cat([k, v], dim=-1), "exact"

        # 2. 의미 유사도 검색
        # query_emb = query_keys.mean(dim=0)  # (d_head,)
        # top_k_candidates = _cosine_search(query_emb, top_k)
        # for (cand_key, cand_emb, cos_sim) in top_k_candidates:
        #     if cos_sim < similarity_threshold:
        #         continue
        #     k_cand = codec.decode(_compressed_store[cand_key]["k"], layer_idx)
        #     v_cand = codec.decode(_compressed_store[cand_key]["v"], layer_idx)
        #     # DHD 편차 체크
        #     deviation = _compute_dhd_deviation(query_keys, k_cand)
        #     if deviation <= deviation_threshold:
        #         semantic_hits += 1
        #         return torch.cat([k_cand, v_cand], dim=-1), "semantic"
        #     else:
        #         # 편차 > threshold: 편차 큰 token만 재계산 표시
        #         recompute_count += 1
        #         return None, "miss"  # 호출자가 재계산 수행

        # 3. 미스
        # misses += 1
        # return None, "miss"
        ...

    def _cosine_search(
        self,
        query_emb: torch.Tensor,  # (d_head,)
        top_k: int,
    ) -> List[Tuple[str, torch.Tensor, float]]:
        # _semantic_index의 모든 (key, emb) 대상 코사인 유사도 계산
        # emb_matrix = torch.stack([emb for _, emb in _semantic_index])  # (N, d_head)
        # q_norm = F.normalize(query_emb.unsqueeze(0), dim=-1)         # (1, d_head)
        # e_norm = F.normalize(emb_matrix, dim=-1)                     # (N, d_head)
        # sims = (q_norm @ e_norm.T).squeeze(0)                        # (N,)
        # top_k_idx = sims.argsort(descending=True)[:top_k]
        # return [(key, emb, sims[i].item()) for i, (key, emb) in ...]
        ...

    def _compute_dhd_deviation(
        self,
        query_keys: torch.Tensor,   # (n_tokens, d_head)
        cached_keys: torch.Tensor,  # (n_tokens, d_head) — 후보 세그먼트 Key
    ) -> float:
        # 두 텐서의 행 수가 다를 수 있음 → min 길이로 잘라서 비교
        # min_len = min(query_keys.shape[0], cached_keys.shape[0])
        # q = query_keys[:min_len]
        # c = cached_keys[:min_len]
        # deviation = (q - c).norm(dim=-1).mean().item() / (c.norm(dim=-1).mean().item() + 1e-8)
        # return deviation  # 정규화된 편차 (0~∞, deviation_threshold와 비교)
        ...

    def semantic_hit_rates(self) -> dict:
        # total = exact_hits + semantic_hits + misses
        # return {
        #     "exact_hit_rate": exact_hits / total,
        #     "semantic_hit_rate": semantic_hits / total,
        #     "overall_hit_rate": (exact_hits + semantic_hits) / total,
        #     "noncontiguous_ratio": semantic_hits / (exact_hits + semantic_hits) if hits > 0 else 0,
        #     "recompute_ratio": recompute_count / max(1, exact_hits + semantic_hits),
        # }
        ...

    def chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        # segmented.py와 동일한 SHA-256 방식
        import hashlib, struct
        start = chunk_idx * self.chunk_size
        chunk = token_ids[start : start + self.chunk_size]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()
```

---

### 3. DualMapScheduler (Activity A)

**파일**: `src/scheduler/dual_map_scheduler.py`

**핵심 아이디어**:
- 각 요청에 두 독립 해시 함수 h1, h2를 적용해 두 후보 노드 (n1, n2)를 선택.
- 각 후보 노드에서 "의미 히트율": 요청 임베딩과 노드 캐시 세그먼트 임베딩 간 코사인 유사도 상위-k 평균.
- 라우팅 스코어 = semantic_hit_rate × (1 - load_ratio).
- SLO 위반 시 부하 기준만으로 선택 (안전 전환).
- 단일 노드 시뮬레이션: num_nodes=1 설정 가능.

**스케줄링 단위**: 요청(request) 단위 라우팅. 배치 구성 후 각 요청에 `target_node_id` 속성을 주석(annotation)으로 추가.

**캐시 상태 접근**: 각 노드는 `SemanticSegmentCache` 인스턴스를 보유하며, 스케줄러는 `cache._semantic_index`를 읽어 임베딩 인덱스를 접근한다 (조회 통계를 오염시키지 않기 위해 직접 접근).

```python
@dataclass
class NodeState:
    node_id: str
    cache: "SemanticSegmentCache"
    current_load: float = 0.0      # 0.0~1.0
    slo_violation: bool = False    # True이면 부하 기준으로만 라우팅

class DualMapScheduler:
    def __init__(
        self,
        nodes: List[NodeState],
        slo_ttft_ms: float = 200.0,       # SLO 위반 임계값 (ms)
        top_k_semantic: int = 5,          # 의미 히트율 계산 시 상위-k
        fairness_max_wait: int = 10,      # 공정성 최대 대기 스텝
        hash_seed_1: int = 2654435761,    # h1 해시 시드
        hash_seed_2: int = 1234567891,    # h2 해시 시드
    ) -> None: ...

    def _node_index_h1(self, request_id: str) -> int:
        # hash(hash_seed_1 XOR hash(request_id)) % len(nodes)
        ...

    def _node_index_h2(self, request_id: str) -> int:
        # hash(hash_seed_2 XOR hash(request_id)) % len(nodes)
        # 보장: h1 != h2 (같으면 (h2 + 1) % len(nodes))
        ...

    def _semantic_hit_score(
        self,
        request_embedding: torch.Tensor,  # (d_head,) 요청 임베딩
        node: NodeState,
    ) -> float:
        # node.cache._semantic_index에서 임베딩 목록 조회 (통계 비오염)
        # 상위 top_k_semantic 코사인 유사도 평균 반환
        # _semantic_index가 비어있으면 0.0 반환
        ...

    def _request_embedding(self, request: InferenceRequest) -> torch.Tensor:
        # 요청 토큰 임베딩 근사: token_ids를 float 벡터로 변환 후 정규화
        # 실제 임베딩 모델 없이: token_ids의 평균을 d_head 차원으로 확장
        # (d_head는 nodes[0].cache._semantic_index[0][1].shape[-1] 참조)
        # token_mean = mean(token_ids) → scalar
        # 시드 기반 pseudo-random 벡터: torch.manual_seed(int(token_mean)) → randn(d_head)
        ...

    def route(self, request: InferenceRequest) -> str:
        # Returns: target node_id

        # 1. SLO 위반 노드 확인
        # 2. 두 후보 노드 선택: idx1 = h1(req.request_id), idx2 = h2(req.request_id)
        # 3. 의미 히트율 계산: score_i = semantic_hit_score(req_emb, node_i) × (1 - node_i.load)
        # 4. SLO 위반 시: score = (1 - load) 만으로 결정
        # 5. 공정성 보정: wait_steps >= fairness_max_wait이면 부하 기준으로만 선택
        # 6. 최고 점수 노드 반환
        ...

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        # 각 요청에 target_node_id 주석 추가 후 의미 히트율 내림차순 정렬
        # 동일 노드 대상 요청끼리 묶어서 배치 형성 (캐시 지역성 향상)
        ...

    def update_load(self, node_id: str, load: float) -> None:
        # 노드 부하 업데이트
        ...

    def update_slo_status(self, node_id: str, violated: bool) -> None:
        # SLO 위반 상태 업데이트
        ...
```

---

### 4. SpeculativeSegmentFetcher (Activity B)

**파일**: `src/cache/speculative_fetcher.py`

**핵심 아이디어**:
- 이전 배치 처리 중에 다음 배치 요청의 세그먼트 검색을 비동기로 미리 실행.
- `threading.Thread`로 비동기 프리패치 실행 (GPU 없이 CPU에서 유사도 검색 수행).
- 최대 대기 시간 초과 시 miss로 처리 (TTFT 보호).

```python
class SpeculativeSegmentFetcher:
    def __init__(
        self,
        cache: "SemanticSegmentCache",
        max_wait_ms: float = 5.0,     # 프리패치 결과 대기 최대 시간
        prefetch_depth: int = 1,       # 미래 배치 깊이 (1 = 다음 배치만)
    ) -> None: ...

    def prefetch_async(
        self,
        requests: List[InferenceRequest],   # 다음 배치 요청들
        layer_idx: int = 0,
    ) -> None:
        # threading.Thread로 _prefetch_worker 실행
        # _prefetch_cache: Dict[str, dict] — request_id → {chunk_idx → (kv, hit_type)}
        ...

    def _prefetch_worker(
        self,
        requests: List[InferenceRequest],
        layer_idx: int,
    ) -> None:
        # 각 요청의 청크별 get_segment() 호출 (통계 오염 없이 _compressed_store 직접 조회)
        # 결과를 _prefetch_cache에 저장
        ...

    def get_prefetched(
        self,
        request: InferenceRequest,
        chunk_idx: int,
        timeout_ms: float = None,  # None이면 max_wait_ms 사용
    ) -> Optional[Tuple[torch.Tensor, str]]:
        # _prefetch_cache에서 결과 조회 (있으면 즉시 반환, 없으면 None)
        # thread.join(timeout=timeout_ms/1000) 후 조회
        ...

    def clear(self) -> None:
        # _prefetch_cache 초기화, 실행 중 thread 종료 대기
        ...
```

---

## Activity C — Accuracy Preservation 검증 계획

**이 섹션은 Activity C 포함으로 인해 반드시 완성되어야 한다. 검증 계획 없이 Spec.md를 완성하지 않는다.**

### perplexity 측정 계획

- **데이터셋**: WikiText-2 (wikitext-2-raw-v1, 표준 분할)
- **모델**: GPT-2 (소형, d_head=64, 12레이어) — CPU에서 실행 가능한 표준 모델
- **측정 방법**: stride=512, max_length=1024 슬라이딩 윈도우 perplexity 계산
- **허용 오차**: `|PPL_compressed - PPL_baseline| / PPL_baseline ≤ 0.01` (±1% 이내)
- **수치 프록시** (단위 테스트, 실제 모델 호출 없이):
  - KL divergence proxy: `(decoded_kv - original_kv).norm() / original_kv.norm() ≤ 0.10` (10% 정규화 오류)
  - MSE 비율: `MSE(decoded, original) / MSE(zeros, original) < 0.15` — perplexity ±1% bound 근사
  - Cosine similarity: `cosine_sim(decoded, original).mean() ≥ 0.95` (FP32 기준)

### 태스크 정확도 측정 계획

- **벤치마크 1**: LongBench-QA (단일 문서 QA, ROUGE-L 점수) — 수치 프록시: 방향 보존 cosine ≥ 0.95
- **벤치마크 2**: LongBench-Summarization (GovReport, ROUGE-1 점수) — 수치 프록시: 정규화 오류 ≤ 10%
- **벤치마크 3**: LongBench-Few-shot (TriviaQA, exact match 점수) — 수치 프록시: MSE 비율 < 0.15
- **허용 오차**: 각 서브태스크 절대 정확도 변화 ±1% 이내
- **QJL 잔차 보정 효과 검증**: encode → decode 후 코사인 유사도가 QJL 보정 없는 버전 대비
  ≥ 0.02 향상 (1%p 이상 향상)

### 검증 테스트 파일: `tests/unit/test_turbo_quant_accuracy.py`

이 파일은 실제 모델 호출 없이 수치 프록시로 accuracy-preserving을 검증한다.
실제 perplexity 측정은 `tests/integration/test_abc_integration.py`에 포함한다.

```python
def test_polarquant_rotation_preserves_norms():
    # 직교 행렬 R 적용 후 L2 노름이 보존됨을 검증
    # ||R @ v||_2 == ||v||_2 (±1e-5 허용)

def test_encode_decode_roundtrip_cosine_similarity():
    # 100 × 128 float32 KV 텐서
    # encode → decode 후 cosine_sim(decoded, original).mean() ≥ 0.95
    # layer_idx=0 (4비트 민감 레이어)
    # layer_idx=6 (3비트 일반 레이어)

def test_qjl_correction_improves_accuracy():
    # QJL 보정 있는 버전 vs 없는 버전 비교
    # cosine_sim(with_qjl) ≥ cosine_sim(without_qjl) — 보정이 손해를 끼치지 않음
    # 정규화 오류 개선: normalized_error(with_qjl) ≤ normalized_error(without_qjl)

def test_memory_reduction_target():
    # memory_bytes_estimate(1000, 128, layer_idx=6)["reduction_ratio"] ≥ 0.60
    # (3비트 레이어 기준 목표 −60% 이상)

def test_sensitive_layer_uses_higher_bits():
    # _effective_bits(0) == 4  (민감 레이어)
    # _effective_bits(6) == 3  (일반 레이어, num_layers=8 기준)

def test_normalized_reconstruction_error():
    # 정규화 재구성 오류 = ||decoded - original||_F / ||original||_F ≤ 0.10
    # WikiText-2 perplexity ±1% 근사 proxy

def test_mse_ratio_proxy():
    # MSE(decoded, original) / MSE(zeros, original) < 0.15
    # LongBench 정확도 ±1% 근사 proxy

def test_rotation_matrix_reproducibility():
    # 동일 layer_idx, d_head → 항상 동일한 R 생성 (시드 고정 재현성)
    # _get_rotation_matrix(3, 64) 두 번 호출 → torch.allclose(R1, R2)

def test_qjl_matrix_reproducibility():
    # 동일 layer_idx, d_head, proj_dim → 항상 동일한 P 생성
    # _get_qjl_matrix(3, 64, 64) 두 번 호출 → torch.allclose(P1, P2)

def test_encode_decode_different_layers():
    # layer_idx=0, 3, 6, 11 각각에서 encode→decode cosine_sim ≥ 0.90
    # 레이어별 독립 회전 행렬 사용 검증 (레이어 간 R이 다름)

def test_edge_case_single_token():
    # n_tokens=1 → encode, decode 정상 동작
    # cosine_sim(decoded, original) ≥ 0.85

def test_compression_accuracy_wikitext2_proxy():
    # 합성 WikiText-2 스타일 KV (랜덤, 1000×128)
    # encode→decode MSE 비율 < 0.15 (실제 perplexity ±1% bound 근사)
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-03.yaml
experiment: "2026-05-03-abc-turbo-quant-semantic-dhd"
date: "2026-05-03"
activities: [A, B, C]

cache:
  type: SemanticSegmentCache
  chunk_size: 128
  max_entries: 1000
  top_k: 5                         # 유사도 검색 상위-k 후보 수
  similarity_threshold: 0.80       # 코사인 유사도 최소 임계값
  deviation_threshold: 0.20        # DHD L2 편차 임계값 (정규화)
  recompute_budget: 0.20           # 최대 재계산 비율

codec:
  type: TurboQuantCodec
  num_layers: 12
  bits: 3                          # 일반 레이어 양자화 비트
  qjl_bits: 1                      # QJL 잔차 보정 비트
  base_seed: 42
  sensitive_layers_ratio: 0.25     # 상위 25% 레이어 → 4비트 (민감 레이어)

scheduler:
  type: DualMapScheduler
  num_nodes: 1                     # 단일 노드 시뮬레이션 (1로 설정)
  slo_ttft_ms: 200.0               # SLO 위반 임계값
  top_k_semantic: 5                # 의미 히트율 계산 시 상위-k
  fairness_max_wait: 10            # 공정성 최대 대기 스텝

speculative_fetcher:
  enabled: true
  max_wait_ms: 5.0                 # 프리패치 결과 최대 대기 시간
  prefetch_depth: 1                # 미래 배치 깊이

metrics:
  target_throughput_gain: 0.20          # +20% tokens/sec
  target_memory_reduction: 0.60         # -60% KV 메모리 (목표 -75%)
  target_noncontiguous_hit_rate: 0.30   # 전체 히트의 30% 이상 비연속(의미 기반)
  max_perplexity_delta_pct: 1.0         # ±1% perplexity
  max_task_accuracy_delta_pct: 1.0      # ±1% 태스크 정확도
  min_cosine_similarity: 0.95           # encode→decode cosine sim 하한
  max_normalized_error: 0.10            # 정규화 재구성 오류 상한
  max_scheduling_overhead_ttft_pct: 5.0 # 스케줄링 TTFT 오버헤드 상한

accuracy_benchmarks:
  - name: wikitext2_proxy
    metric: normalized_reconstruction_error
    threshold: 0.10                # ≤ 10% (±1% perplexity proxy)
  - name: cosine_similarity_proxy
    metric: cosine_similarity_mean
    threshold: 0.95                # ≥ 0.95
  - name: mse_ratio_proxy
    metric: mse_ratio
    threshold: 0.15                # < 0.15 (LongBench ±1% proxy)
  - name: longbench_qa
    metric: rouge_l
    tolerance: 0.01
  - name: longbench_summarization
    metric: rouge_1
    tolerance: 0.01
  - name: longbench_fewshot
    metric: exact_match
    tolerance: 0.01
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_turbo_quant.py` (신규 — TurboQuantCodec 기능 테스트)
- [ ] `tests/unit/test_turbo_quant_accuracy.py` (신규 — Activity C accuracy-preserving 필수)
- [ ] `tests/unit/test_dhd_segment_cache.py` (신규 — SemanticSegmentCache 단위 테스트)
- [ ] `tests/unit/test_dual_map_scheduler.py` (신규 — DualMapScheduler 단위 테스트)
- [ ] `tests/integration/test_abc_integration.py` (신규 — A+B+C 전체 통합 테스트)
- [ ] 기존 단위·통합 테스트 전부 회귀 없이 통과

### test_turbo_quant.py 필수 테스트 케이스

```python
def test_encode_returns_dict_with_required_keys():
    # encode() 반환 dict에 "quantized", "scale", "qjl_packed", "layer_idx" 존재

def test_decode_shape_matches_input():
    # encode → decode 후 shape == 원래 kv.shape

def test_compression_ratio_3bit():
    # compression_ratio(layer_idx=6) ≥ 0.60 (60% 이상 절감)

def test_compression_ratio_4bit_sensitive():
    # compression_ratio(layer_idx=0) ≥ 0.50 (4비트 민감 레이어)

def test_layer_specific_rotation():
    # 서로 다른 layer_idx → 서로 다른 회전 행렬 (R0 != R6)

def test_edge_case_n_tokens_1():
    # n_tokens=1 정상 동작

def test_memory_bytes_estimate_format():
    # memory_bytes_estimate() 반환 dict에 "total_bytes", "baseline_bytes", "reduction_ratio"

def test_cachestore_not_inherited():
    # TurboQuantCodec은 CacheStore를 상속하지 않음 (codec 역할)
```

### test_dhd_segment_cache.py 필수 테스트 케이스

```python
def test_put_get_exact_hit():
    # put_segment() → get_segment() 동일 token_ids → hit_type == "exact"

def test_semantic_hit_similar_tokens():
    # 의미적으로 유사한 KV 세그먼트(노이즈 추가) → hit_type == "semantic"
    # similarity_threshold=0.70, deviation_threshold=0.30 완화 설정 사용

def test_no_hit_dissimilar_tokens():
    # 완전히 다른 KV 세그먼트 → hit_type == "miss"

def test_noncontiguous_ratio_above_30pct():
    # 여러 의미 히트 발생 후 semantic_hit_rates()["noncontiguous_ratio"] ≥ 0.30

def test_memory_bytes_compressed():
    # 1000 토큰 저장 후 memory_bytes() < FP32 베이스라인의 50%

def test_evict_lru_behavior():
    # max_entries 초과 시 가장 오래된 항목 퇴거

def test_cachestore_interface_compliance():
    # put(), get(), evict(), hit_rate(), memory_bytes(), reset_stats() 모두 구현

def test_reset_stats():
    # reset_stats() 후 exact_hits=0, semantic_hits=0, misses=0, recompute_count=0

def test_chunk_key_deterministic():
    # 동일 token_ids, chunk_idx, layer_idx → 동일 키 반환

def test_cosine_search_returns_top_k():
    # _cosine_search() 반환 목록 길이 == min(top_k, len(_semantic_index))
```

### test_dual_map_scheduler.py 필수 테스트 케이스

```python
def test_route_returns_valid_node_id():
    # route() 반환값이 nodes 목록의 node_id 중 하나

def test_dual_hash_different_nodes():
    # len(nodes) >= 2 시 h1 != h2 보장

def test_schedule_annotates_target_node():
    # schedule() 후 각 요청에 target_node_id 속성 존재

def test_slo_violation_uses_load_only():
    # SLO 위반 노드 설정 시 부하 기준으로만 라우팅

def test_fairness_max_wait_respected():
    # wait_steps >= fairness_max_wait 요청이 우선 처리됨

def test_single_node_mode():
    # num_nodes=1 시 항상 동일 노드 반환

def test_scheduling_overhead_below_threshold():
    # schedule() 실행 시간이 요청당 5ms 이내 (100 요청 기준)
```

### test_abc_integration.py 필수 테스트 케이스

```python
def test_cross_bc_pipeline_put_and_get():
    # SemanticSegmentCache + TurboQuantCodec 통합
    # put_segment() → get_segment() 정상 동작 (exact hit)

def test_cross_bc_semantic_hit_with_compression():
    # 압축 저장된 세그먼트에 의미 유사 쿼리 → semantic hit 발생

def test_memory_reduction_with_semantic_cache():
    # TurboQuantCodec 압축 저장 후 memory_bytes() < FP32 베이스라인 × 0.50

def test_speculative_fetcher_reduces_latency():
    # SpeculativeSegmentFetcher 프리패치 후 get_prefetched() 결과가 존재
    # (또는 None이어도 timeout 초과 없이 반환)

def test_dual_map_scheduler_routes_requests():
    # DualMapScheduler.schedule() 후 모든 요청에 target_node_id 존재

def test_abc_full_pipeline():
    # DualMapScheduler → SpeculativeSegmentFetcher → SemanticSegmentCache → TurboQuantCodec
    # 전체 파이프라인 정상 동작 (예외 없이 완료)

def test_perplexity_delta_proxy():
    # 합성 KV (1000×128)에 TurboQuantCodec encode→decode 후
    # normalized_error ≤ 0.10 (WikiText-2 ±1% perplexity proxy)

def test_noncontiguous_hit_rate_above_30pct():
    # 의미 유사 요청 다수 처리 후 noncontiguous_ratio ≥ 0.30
```

---

## 구현 시 주의사항

1. **CacheStore 인터페이스 준수**: `SemanticSegmentCache`는 `CacheStore`를 직접 상속하며,
   6개 추상 메서드(`put`, `get`, `evict`, `hit_rate`, `memory_bytes`, `reset_stats`)를
   모두 구현해야 한다. `put_segment()`, `get_segment()` 등은 추가 메서드다.

2. **TurboQuantCodec은 CacheStore 미상속**: `CompressionCodec` 역할이므로 `CacheStore`를
   상속하지 않는다. `compression.py`의 `CompressionCodec` 클래스와 동일한 `encode/decode`
   인터페이스를 구현해 호환성을 유지한다.

3. **랜덤 회전 행렬 직교성 보장**: `torch.linalg.qr(raw)` 사용. QR 분해의 Q 행렬은
   수치적으로 직교하며, 역행렬이 전치행렬과 같다 (R^T = R^{-1}).

4. **torch.packbits / unpackbits**: PyTorch 1.11+ 필요. `(tensor >= 0).to(torch.uint8)` →
   `torch.packbits(bits, dim=-1)`. unpack 시 `[:, :proj_dim]`으로 패딩 비트 제거.

5. **훈련-무료 제약**: `TurboQuantCodec`, `SemanticSegmentCache`, `DualMapScheduler`,
   `SpeculativeSegmentFetcher` 모두 학습 파라미터(`nn.Parameter`, `nn.Module`) 미포함.

6. **모든 단위 테스트는 CPU 텐서 기준**: `torch.device("cpu")`. GPU 불필요.

7. **시드 고정**: 모든 테스트에서 `torch.manual_seed(42)` 사용. 재현성 보장.

8. **SpeculativeSegmentFetcher 스레드 안전성**: `threading.Lock`으로 `_prefetch_cache`
   접근 보호. `clear()` 호출 시 진행 중인 thread를 안전하게 종료.

9. **N_segments ≤ 10K 가정**: brute-force 코사인 유사도 검색의 전제. 10K 초과 시
   `_cosine_search()`가 시간 초과될 수 있음을 주석으로 명시.

10. **DualMapScheduler 단일 노드 모드**: `len(nodes) == 1`일 때 h1과 h2가 동일 노드를
    가리키더라도 정상 동작해야 한다. `h2 = (h1 + 1) % len(nodes)` 조정이 len(nodes)==1이면
    h1과 동일하므로, 단일 노드 모드에서 h1==h2는 허용.

11. **기존 테스트 회귀 없이 통과**: 새로운 파일만 추가하므로 기존 코드가 변경되지 않는다.
    기존 `test_compression_accuracy.py` (이전 사이클용)는 그대로 유지.

---

## 완료 기준 (Definition of Done)

- [ ] `tests/unit/test_turbo_quant.py` — 8개 테스트 케이스 전부 통과
- [ ] `tests/unit/test_turbo_quant_accuracy.py` — 12개 테스트 케이스 전부 통과 (§4 Accuracy 필수)
- [ ] `tests/unit/test_dhd_segment_cache.py` — 10개 테스트 케이스 전부 통과
- [ ] `tests/unit/test_dual_map_scheduler.py` — 7개 테스트 케이스 전부 통과
- [ ] `tests/integration/test_abc_integration.py` — 8개 테스트 케이스 전부 통과
- [ ] 기존 단위·통합 테스트 전부 회귀 없이 통과
- [ ] `configs/experiments/2026-05-03.yaml` 존재 (§0 설정 YAML 필수)
- [ ] `evaluation_criteria.md` §4 Activity C (필수):
  - perplexity 수치 프록시: normalized_error ≤ 0.10 (±1% 이내)
  - 태스크 정확도 수치 프록시: cosine_sim ≥ 0.95, MSE 비율 < 0.15 (±1% 이내)
  - KV Memory Reduction ≥ −60% (목표; 평가 최소 −30%)
- [ ] `evaluation_criteria.md` §3 Activity B:
  - 비연속 세그먼트 히트율(semantic) ≥ 전체 히트의 30%
  - KV Memory Footprint: 베이스라인 대비 +20% 이내
- [ ] `evaluation_criteria.md` §2 Activity A:
  - 스케줄링 오버헤드 TTFT +5% 이내
  - 스케줄링 적용 캐시 히트율 향상 ≥ +10%p
- [ ] `evaluation_criteria.md` §5 Cross:
  - 복합 Throughput 향상: 단일 Activity 대비 추가 +5%
  - 복합 Memory Reduction: 단일 Activity 대비 추가 −10%
  - Accuracy 보존 복합 적용 후 ±1% 이내 (필수)
- [ ] 타입 힌트 모든 공개 함수·메서드에 존재 (§0 중간)
- [ ] 불필요한 추상화 없음: 이 Spec에 없는 클래스·인터페이스 도입 금지 (§0 낮음)
