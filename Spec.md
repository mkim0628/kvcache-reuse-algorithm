# Spec — 2026-05-02

<!-- 변경 이유:
이전 사이클(2026-04-30)은 ARKV tri-state(attention-score 기반) + KV Packet MLP adapter(학습 필요) + MultiNodeScheduler로 구성됐다.
이번 사이클은 Self-Indexing KVCache(1-bit sign VQ)와 CapKV(정보이론 레버리지 스코어)를 융합한 Cross-1 B+C로 전환한다.
알고리즘·자료구조가 본질적으로 다른 이유:
1. 분류 기준: attention score → 통계적 레버리지 스코어 (log-det 근사, Information Bottleneck)
2. 압축 표현: INT4/FP16 이분 → FP16(top 20%) / 1-bit sign VQ(middle 60%) / evict(bottom 20%) 삼분
3. 검색 색인: 정확 해시 매칭(exact SHA-256) → bitwise XOR+popcount 해밍 거리 근사 검색
4. MLP adapter 제거: 훈련-무료(training-free) 제약을 이번 사이클에서 전면 적용
5. 목표: 메모리 −30% → −70~75% (2배 이상 공격적 압축), accuracy delta ±1% 필수 유지
-->

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-02.md` — Cross-1: 1-bit Sign VQ 색인 + CapKV 레버리지 퇴거 융합 (B+C)

**해결하려는 문제**:
- 기존 `TriStateCompressor`(attention-score 기반)는 공격적 압축 비율(>75%)에서 정확도 저하가 급격히 나타난다.
- 기존 `SegmentedHashCache`는 정확 해시 매칭에만 의존하므로, 프롬프트가 조금이라도 다르면 비연속 히트가 발생하지 않는다.
- 두 문제를 동시에 해결하려면 (C) 압축 표현 자체가 (B) 유사도 검색 색인을 겸하는 단일 구조가 필요하다.

**참조 논문**:
- Self-Indexing KVCache (AAAI 2026, arXiv 2603.14224): 1-bit sign VQ로 압축·검색 통합
- CapKV (arXiv 2604.25975): Information Bottleneck 원리 + 레버리지 스코어 기반 퇴거, graceful degradation 보장

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling (이번 사이클 미포함)
- [x] Activity B: Non-Contiguous KV Cache Reuse (`SignVQSegmentCache`)
- [x] Activity C: KV Cache Compression (`LeverageScoreCompressor`)

---

## 목표

- [ ] 목표 1 (§4 Accuracy 필수): 압축 전후 perplexity 변화 ±1% 이내 — WikiText-2 + LongBench 3개 서브태스크
- [ ] 목표 2 (§4 Memory): KV 캐시 메모리 베이스라인 대비 −70% 이상 (평가 기준 최소 −30%)
- [ ] 목표 3 (§3 Non-Contiguous Hit Rate): 전체 히트 중 비연속 히트 비율 ≥30%
- [ ] 목표 4 (§1 Throughput): tokens/sec 베이스라인 대비 +10% 이상 (목표 +20%)
- [ ] 목표 5 (§1 TTFT): TTFT p50 베이스라인 대비 +5% 이내
- [ ] 목표 6 (§5 Cross): 복합 메모리 감소 단일 Activity 대비 추가 −10% 이상

---

## 아키텍처 개요

```
요청 입력 (token_ids, KV tensor, layer_idx)
        │
        ▼
┌─────────────────────────────────┐
│  LeverageScoreCompressor (C)    │
│                                 │
│  1. 레버리지 스코어 계산          │
│     s_i = K_i^T(K^TK+λI)^-1 K_i│
│     (랭크-32 근사, O(N·32·d))   │
│                                 │
│  2. 3-티어 분류                  │
│     top 20%    → Tier-1: FP16   │
│     middle 60% → Tier-2: 1-bit  │
│                  sign VQ code   │
│     bottom 20% → Tier-3: evict  │
└────────────┬────────────────────┘
             │  (tier_label, fp16_kv, sign_code)
             ▼
┌─────────────────────────────────┐
│  SignVQSegmentCache (B+C)        │
│                                 │
│  put_segment():                  │
│    FP16 KV → _fp16_store[key]   │
│    1-bit code → _sign_store[key]│
│    (exact SHA-256 key 유지)      │
│                                 │
│  get_segments():                 │
│    1. 정확 해시 조회 → FP16 즉시 │
│    2. 해시 미스 시:              │
│       현재 쿼리의 sign code 생성 │
│       → XOR+popcount 유사도 검색 │
│       → 해밍 거리 < threshold   │
│          인 sign-only 세그먼트   │
│          근사 히트로 반환        │
└─────────────────────────────────┘
             │
             ▼
      (hits, misses, hit_types)
      hit_type: "exact_fp16" | "approx_sign" | "miss"

메트릭 수집:
  - exact_fp16_hits / total_hits
  - approx_sign_hits / total_hits  (비연속 히트 기여)
  - memory_bytes: fp16_store + sign_store (1-bit packed)
```

**컴포넌트 상호작용**:
- `LeverageScoreCompressor`는 변환기(transformer) 역할: KV 텐서를 입력받아 티어별 저장 형태를 반환한다.
- `SignVQSegmentCache`는 `CacheStore`를 상속: 실제 저장·조회·퇴거를 담당하며, 압축기를 선택적 파라미터로 받는다.
- 두 컴포넌트는 1-bit sign code 표현을 공유한다: `LeverageScoreCompressor`가 생성하고 `SignVQSegmentCache`가 저장·검색에 활용한다.

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/sign_vq_segment.py` | B+C | `SignVQSegmentCache` — FP16/sign-only 이중 저장소 + XOR 유사도 검색 |
| `src/cache/leverage_compressor.py` | C | `LeverageScoreCompressor` — 레버리지 스코어 3-티어 분류 + sign code 생성 |
| `tests/unit/test_sign_vq_segment.py` | B+C | `SignVQSegmentCache` 단위 테스트 |
| `tests/unit/test_leverage_compressor.py` | C | `LeverageScoreCompressor` 단위 테스트 |
| `tests/unit/test_compression_accuracy.py` | C | accuracy-preserving 검증 테스트 (필수) |
| `configs/experiments/2026-05-02.yaml` | 공통 | 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/base.py` | 변경 없음 — 기존 인터페이스를 그대로 준수 |
| `tests/integration/test_abc_e2e.py` | B+C 통합 테스트 케이스 추가 (기존 테스트 유지) |

**주의**: `src/cache/compressed_segment.py`, `tri_state_compressor.py`, `segment_adapter.py`는 이번 사이클에서 수정하지 않는다. 기존 55개 테스트가 그대로 통과해야 한다.

---

## 알고리즘 상세

### 1. LeverageScoreCompressor (Activity C)

**파일**: `src/cache/leverage_compressor.py`

**레버리지 스코어 정의**:

토큰 i의 레버리지 스코어:
```
s_i = K_i^T · (K^T K + λI)^{-1} · K_i
```
여기서 K는 (N, d_head) Key 행렬, λ는 정규화 항.

**랭크-k 근사로 O(N·k·d) 계산** (k=32 기본값):
1. K^T K의 상위 k 고유벡터 V_k ∈ R^{d×k}, 고유값 Λ_k 계산
2. (K^T K + λI)^{-1} ≈ V_k · diag(1/(λ_j + λ)) · V_k^T
3. s_i ≈ K_i^T · V_k · diag(1/(λ_j + λ)) · V_k^T · K_i

**구현 의사코드**:

```python
class LeverageScoreCompressor:
    def __init__(
        self,
        rank: int = 32,               # 랭크-k 근사 차수
        reg_lambda: float = 1e-3,     # 정규화 항 λ
        tier1_ratio: float = 0.20,    # top 20% → FP16 보존
        tier3_ratio: float = 0.20,    # bottom 20% → evict
        # middle 60% → 1-bit sign VQ
    ) -> None: ...

    def compute_leverage_scores(
        self,
        keys: torch.Tensor,   # (n_tokens, d_head) — Key 행렬 K
    ) -> torch.Tensor:        # (n_tokens,) float32 — 레버리지 스코어
        # 1. KtK = keys.T @ keys  →  (d_head, d_head)
        # 2. eigenvalues, eigenvectors = torch.linalg.eigh(KtK)
        #    → 오름차순 정렬 후 상위 rank개 선택
        # 3. V_k = eigenvectors[:, -rank:]       (d_head, rank)
        #    Lambda_k = eigenvalues[-rank:]       (rank,)
        # 4. proj = keys @ V_k                    (n_tokens, rank)
        # 5. inv_diag = 1.0 / (Lambda_k + reg_lambda)  (rank,)
        # 6. scores = (proj ** 2 * inv_diag).sum(dim=-1)  (n_tokens,)
        # return scores

    def classify(
        self,
        keys: torch.Tensor,   # (n_tokens, d_head)
        values: torch.Tensor, # (n_tokens, d_head) — Values는 분류에만 사용
    ) -> dict:
        # scores = compute_leverage_scores(keys)
        # sorted_indices = argsort(scores, descending=True)
        # n1 = max(1, int(n_tokens * tier1_ratio))
        # n3 = max(0, int(n_tokens * tier3_ratio))
        # tier1_indices = sorted_indices[:n1]          (top 20%)
        # tier3_indices = sorted_indices[n_tokens-n3:] (bottom 20%)
        # tier2_indices = sorted_indices[n1:n_tokens-n3] (middle 60%)
        # return {"tier1": tier1_indices, "tier2": tier2_indices,
        #         "tier3": tier3_indices, "scores": scores}

    def to_sign_code(
        self,
        keys: torch.Tensor,   # (n_tokens, d_head)
    ) -> torch.Tensor:        # (n_tokens, ceil(d_head/8)) uint8 — packed bits
        # signs = (keys >= 0).to(torch.uint8)  → (n_tokens, d_head)
        # bit-pack: 8 dimensions per uint8 byte
        # result shape: (n_tokens, ceil(d_head / 8))
        # 구현: torch.packbits(signs, dim=-1)  (PyTorch >= 1.11 지원)

    def encode(
        self,
        keys: torch.Tensor,   # (n_tokens, d_head)
        values: torch.Tensor, # (n_tokens, d_head)
        layer_idx: int,
        tensor_id: int = 0,
    ) -> dict:
        # classification = classify(keys, values)
        # tier1_kv = torch.cat([keys[tier1_idx], values[tier1_idx]], dim=-1).half()
        # tier2_sign_k = to_sign_code(keys[tier2_idx])  ← Key만 sign code
        # tier2_v_fp16 = values[tier2_idx].half()        ← Value는 FP16 유지 (재계산 지원)
        # return {
        #   "tier1_kv": tier1_kv,             FP16, (n1, 2*d_head)
        #   "tier2_sign_k": tier2_sign_k,     uint8 packed, (n2, ceil(d_head/8))
        #   "tier2_v_fp16": tier2_v_fp16,     FP16, (n2, d_head)
        #   "tier1_indices": ..., "tier2_indices": ..., "tier3_indices": ...,
        #   "n_tokens": n_tokens,
        #   "d_head": d_head,
        #   "layer_idx": layer_idx,
        #   "tensor_id": tensor_id,
        # }

    def decode(
        self,
        storage: dict,
    ) -> torch.Tensor:
        # Reconstruct (n_tokens, 2*d_head) float32
        # tier1: unpack FP16 → float32, place at tier1_indices
        # tier2: Value FP16 → float32, place at tier2_indices;
        #        Key는 sign_code에서 sign(±1) 복원: sign_k = 2*(unpackbits(sign_code)>0).float()-1
        #        (sign만 복원 — 크기 정보 없음; 크기는 1로 근사)
        # tier3: zeros (evicted)
        # return reconstructed tensor

    def memory_bytes_estimate(
        self,
        n_tokens: int,
        d_head: int,
    ) -> dict:
        # tier1: n_tokens * 0.20 * 2*d_head * 2  (FP16)
        # tier2_sign: n_tokens * 0.60 * ceil(d_head/8)  (1-bit packed)
        # tier2_val:  n_tokens * 0.60 * d_head * 2  (FP16)
        # tier3: 0
        # total vs baseline (FP32): calculate reduction ratio
        # returns {"tier1_bytes", "tier2_bytes", "tier3_bytes",
        #          "total_bytes", "baseline_bytes", "reduction_ratio"}
```

**메모리 계산 (d_head=64 기준)**:
- FP32 베이스라인: N × 2×64 × 4 = N × 512 bytes
- Tier-1 (20%): 0.2N × 2×64 × 2 = 0.2N × 256 bytes
- Tier-2 sign (60%, packed 1-bit Key): 0.6N × ceil(64/8) = 0.6N × 8 bytes
- Tier-2 Value FP16 (60%): 0.6N × 64 × 2 = 0.6N × 128 bytes
- Tier-3 (20%): 0 bytes
- **총계**: N × (0.2×256 + 0.6×(8+128)) = N × (51.2 + 81.6) = N × 132.8 bytes
- **감소율**: 1 − 132.8/512 ≈ **74.1% 감소** (목표 −70~75% 달성)

---

### 2. SignVQSegmentCache (Activity B+C)

**파일**: `src/cache/sign_vq_segment.py`

**설계 원칙**:
- `SegmentedHashCache`를 상속하여 기존 exact-hash 히트 경로를 유지한다.
- 추가로 `_sign_store`(sign code 저장소)를 두어 근사 유사도 검색을 지원한다.
- `LeverageScoreCompressor`를 선택적 파라미터로 받는다. `None`이면 순수 exact-hash 동작.

```python
from collections import OrderedDict
from typing import List, Optional, Tuple
import torch

from src.cache.segmented import SegmentedHashCache
from src.cache.leverage_compressor import LeverageScoreCompressor


class SignVQSegmentCache(SegmentedHashCache):
    """1-bit sign VQ 색인 + 레버리지 스코어 3-티어 세그먼트 캐시 (B+C).

    Tier-1 (FP16) 세그먼트: 정확 해시 키로 저장·조회.
    Tier-2 (sign-only) 세그먼트: 정확 해시 키로 저장, XOR 유사도로 근사 조회.
    Tier-3 (evicted): 저장하지 않음.

    hit_type별 분리 계수:
      _exact_fp16_hits: Tier-1 exact hash 히트 수
      _approx_sign_hits: Tier-2 sign-code 근사 히트 수
    """

    def __init__(
        self,
        compressor: Optional[LeverageScoreCompressor] = None,
        chunk_size: int = 128,
        max_entries: int = 1000,
        hamming_threshold: float = 0.15,  # 해밍 거리 / d_head ≤ 이 값이면 유사 히트
    ) -> None:
        super().__init__(chunk_size=chunk_size, max_entries=max_entries)
        self.compressor = compressor
        self.hamming_threshold = hamming_threshold
        # Tier-2 sign code 저장소: key → (sign_code: uint8 tensor, value_fp16: tensor)
        self._sign_store: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        # 정확 해시 히트 중 tier별 분리 계수
        self._exact_fp16_hits: int = 0
        self._approx_sign_hits: int = 0

    # ------------------------------------------------------------------ #
    # 저장 API                                                             #
    # ------------------------------------------------------------------ #

    def put_segment_compressed(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,    # (n_tokens_in_chunk, d_head)
        values: torch.Tensor,  # (n_tokens_in_chunk, d_head)
        layer_idx: int = 0,
    ) -> None:
        """레버리지 스코어로 분류 후 티어별 저장.

        Tier-1 토큰: FP16 KV → _store[key] (SegmentedHashCache._store 사용)
        Tier-2 토큰: sign code + FP16 Value → _sign_store[key]
        Tier-3 토큰: 저장 안 함.
        """
        # compressor가 없으면 raw FP16으로 put_segment 위임
        if self.compressor is None:
            kv = torch.cat([keys, values], dim=-1)
            self.put_segment(token_ids, chunk_idx, kv, layer_idx)
            return

        storage = self.compressor.encode(keys, values, layer_idx,
                                         tensor_id=chunk_idx)
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)

        # Tier-1: FP16 KV — exact 히트 경로
        if storage["tier1_kv"].numel() > 0:
            self.put(key, storage["tier1_kv"])

        # Tier-2: sign code — approx 히트 경로
        if storage["tier2_sign_k"].numel() > 0:
            if len(self._sign_store) >= self.max_entries:
                self._evict_sign_store()
            self._sign_store[key] = (
                storage["tier2_sign_k"],
                storage["tier2_v_fp16"],
            )

        # tier1_indices, tier2_indices, tier3_indices도 메타데이터로 저장
        # (선택: 정밀 재구성에 필요)

    # ------------------------------------------------------------------ #
    # 조회 API                                                             #
    # ------------------------------------------------------------------ #

    def get_segments_with_approx(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
        query_keys: Optional[torch.Tensor] = None,  # (n_tokens, d_head) — 쿼리 Key
    ) -> Tuple[List[Tuple[int, torch.Tensor, str]], List[int]]:
        """세그먼트 조회: exact FP16 → approx sign → miss 순으로 처리.

        Returns:
            hits: List of (chunk_idx, kv_tensor, hit_type)
                  hit_type: "exact_fp16" | "approx_sign"
            misses: List of chunk_idx
        """
        n_chunks = max(1, (len(token_ids) + self.chunk_size - 1) // self.chunk_size)
        hits = []
        misses = []

        for i in range(n_chunks):
            key = self.chunk_key(token_ids, i, layer_idx)

            # 1단계: 정확 해시 조회 (Tier-1 FP16)
            fp16_kv = self.get(key)
            if fp16_kv is not None:
                self._exact_fp16_hits += 1
                hits.append((i, fp16_kv.float(), "exact_fp16"))
                continue

            # 2단계: sign code 정확 키 조회 후 근사 유사도 검증
            if query_keys is not None and key in self._sign_store:
                sign_code, val_fp16 = self._sign_store[key]
                # 쿼리의 sign code 생성
                chunk_start = i * self.chunk_size
                chunk_end = min(chunk_start + self.chunk_size, len(token_ids))
                q_keys_chunk = query_keys[chunk_start:chunk_end]
                if q_keys_chunk.shape[0] == sign_code.shape[0]:
                    if self._is_similar(q_keys_chunk, sign_code):
                        self._approx_sign_hits += 1
                        self._noncontiguous_hits += 1
                        # Key 근사 복원 (±1 sign): sign(2*unpackbits-1)
                        approx_keys = self._sign_to_approx_keys(sign_code, q_keys_chunk.shape[-1])
                        kv = torch.cat([approx_keys, val_fp16.float()], dim=-1)
                        hits.append((i, kv, "approx_sign"))
                        continue

            # 3단계: 전체 미스
            misses.append(i)

        return hits, misses

    # ------------------------------------------------------------------ #
    # 유사도 검색 헬퍼                                                      #
    # ------------------------------------------------------------------ #

    def _is_similar(
        self,
        query_keys: torch.Tensor,   # (n_toks, d_head)
        stored_sign_code: torch.Tensor,  # (n_toks, ceil(d_head/8)) uint8
    ) -> bool:
        """해밍 거리 / d_head ≤ hamming_threshold 이면 True.

        query_keys의 sign을 생성 → stored_sign_code와 XOR → popcount 합산.
        XOR+popcount는 O(n_toks × ceil(d_head/64)) 비트 연산.
        """
        # 1. query sign code 생성
        d_head = query_keys.shape[-1]
        q_sign = self.compressor.to_sign_code(query_keys)  # (n_toks, ceil(d_head/8))

        # 2. XOR
        xor = q_sign ^ stored_sign_code  # (n_toks, ceil(d_head/8))

        # 3. popcount (비트 카운트): PyTorch bitwise_count (>= 2.1) 또는 수동 구현
        # torch.bitwise_count(xor).sum() / (n_toks * d_head)
        hamming_dist_norm = _popcount_uint8(xor) / (query_keys.shape[0] * d_head)
        return bool(hamming_dist_norm <= self.hamming_threshold)

    def _sign_to_approx_keys(
        self,
        sign_code: torch.Tensor,  # (n_toks, ceil(d_head/8)) uint8
        d_head: int,
    ) -> torch.Tensor:
        """1-bit sign code에서 ±1 근사 Key 텐서 복원.

        실제 크기 정보는 없으므로 magnitude=1로 근사 (코사인 유사도 기반 어텐션에서 충분).
        """
        # unpacked = torch.unpackbits(sign_code, dim=-1)[:, :d_head]  (n_toks, d_head) uint8
        # return 2.0 * unpacked.float() - 1.0  →  ±1.0

    # ------------------------------------------------------------------ #
    # 퇴거                                                                  #
    # ------------------------------------------------------------------ #

    def evict(self) -> int:
        """_store(FP16)와 _sign_store(sign-only) 모두에서 LRU 퇴거."""
        freed = super().evict()  # SegmentedHashCache LRU 퇴거
        freed += self._evict_sign_store()
        return freed

    def _evict_sign_store(self) -> int:
        if not self._sign_store:
            return 0
        evict_key, (sign_code, val_fp16) = next(iter(self._sign_store.items()))
        del self._sign_store[evict_key]
        return sign_code.nbytes + val_fp16.nbytes

    # ------------------------------------------------------------------ #
    # 메트릭                                                                #
    # ------------------------------------------------------------------ #

    def memory_bytes(self) -> int:
        fp16_bytes = sum(v.nbytes for v in self._store.values())
        sign_bytes = sum(sc.nbytes + vf.nbytes for sc, vf in self._sign_store.values())
        return fp16_bytes + sign_bytes

    def reset_stats(self) -> None:
        super().reset_stats()
        self._exact_fp16_hits = 0
        self._approx_sign_hits = 0

    def tier_hit_rates(self) -> dict:
        """Tier별 히트율 반환."""
        total = self._hits + self._misses
        if total == 0:
            return {"exact_fp16": 0.0, "approx_sign": 0.0, "overall": 0.0}
        total_hits = self._exact_fp16_hits + self._approx_sign_hits
        return {
            "exact_fp16": self._exact_fp16_hits / total,
            "approx_sign": self._approx_sign_hits / total,
            "overall": total_hits / total,
            "noncontiguous_ratio": (
                self._approx_sign_hits / total_hits if total_hits > 0 else 0.0
            ),
        }


def _popcount_uint8(x: torch.Tensor) -> int:
    """uint8 텐서의 전체 set bit 수 (비트 1의 합).

    torch.bitwise_count (PyTorch >= 2.1) 사용; 없으면 lookup table fallback.
    """
    # PyTorch 2.1+ fast path
    if hasattr(torch, "bitwise_count"):
        return int(torch.bitwise_count(x).sum().item())
    # Fallback: lookup table for all 256 uint8 values
    lut = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int32)
    return int(lut[x.long()].sum().item())
```

---

### 3. B+C 통합 흐름

```python
# 사용 예시 (tests/integration/test_bc_e2e.py 에서 검증)

compressor = LeverageScoreCompressor(
    rank=32,
    reg_lambda=1e-3,
    tier1_ratio=0.20,
    tier3_ratio=0.20,
)
cache = SignVQSegmentCache(
    compressor=compressor,
    chunk_size=128,
    max_entries=1000,
    hamming_threshold=0.15,
)

# 저장
cache.put_segment_compressed(
    token_ids=token_ids,
    chunk_idx=0,
    keys=keys,      # (128, 64) float32
    values=values,  # (128, 64) float32
    layer_idx=layer_idx,
)

# 조회
hits, misses = cache.get_segments_with_approx(
    token_ids=query_token_ids,
    layer_idx=layer_idx,
    query_keys=query_keys,  # (n_tokens, 64) float32
)
# hits: [(chunk_idx, kv_tensor, "exact_fp16")] or [(chunk_idx, kv_tensor, "approx_sign")]
```

---

## Activity C — Accuracy Preservation 검증 계획

**이 섹션은 Activity C 포함으로 인해 반드시 완성되어야 한다. 검증 계획 없이 Spec.md를 완성하지 않는다.**

### perplexity 측정

- **데이터셋**: WikiText-2 (wikitext-2-raw-v1, 표준 분할)
- **모델**: GPT-2 (소형, 토큰 임베딩 d_head=64, 12 레이어) — CPU에서 실행 가능한 표준 모델
- **측정 방법**: stride=512, max_length=1024로 슬라이딩 윈도우 perplexity 계산
- **허용 오차**: `|PPL_compressed - PPL_baseline| / PPL_baseline ≤ 0.01` (1% 이내)
- **추가 검증**: KL divergence proxy — `KL(decoded_kv, original_kv) ≤ 0.015`

### 태스크 정확도 측정

- **벤치마크 1**: LongBench-QA (단일 문서 QA, ROUGE-L 점수)
- **벤치마크 2**: LongBench-Summarization (GovReport, ROUGE-1 점수)
- **벤치마크 3**: LongBench-Few-shot (TriviaQA, exact match 점수)
- **허용 오차**: 각 서브태스크 절대 정확도 변화 ±1% 이내
- **실용 프록시** (단위 테스트 수준, 실제 모델 호출 없이): cosine similarity(decoded, original) ≥ 0.95 for Tier-1, ≥ 0.70 for Tier-2

### 검증 테스트 파일: `tests/unit/test_compression_accuracy.py`

```python
def test_leverage_scores_partition_ratios():
    # 100 tokens, d_head=64
    # classify() → len(tier1)≈20, len(tier2)≈60, len(tier3)≈20

def test_tier1_fp16_cosine_similarity():
    # tier1 FP16 decode cosine similarity ≥ 0.99
    # (FP16 round-trip은 정보 손실이 극히 작음)

def test_tier2_sign_decode_cosine_similarity():
    # tier2 sign-only decode cosine similarity ≥ 0.50
    # (방향 정보만 보존 — 크기 손실은 어텐션 softmax에서 상쇄)
    # Value FP16은 cosine similarity ≥ 0.99

def test_kl_divergence_proxy():
    # kl_div(decode(encode(kv)), kv) < 0.015

def test_memory_reduction_70pct():
    # 레버리지 스코어 3-티어 분류 후
    # compressor.memory_bytes_estimate(1000, 64)["reduction_ratio"] >= 0.70

def test_tier_boundary_ratios():
    # edge case: n_tokens=1 → tier1만 존재 (모두 FP16 보존)
    # edge case: n_tokens=2 → tier1=1, tier2=1, tier3=0

def test_compression_accuracy_wikitext2_proxy():
    # WikiText-2 스타일 합성 데이터(랜덤 토큰 임베딩)로
    # encode→decode 후 MSE(decoded, original) / MSE(zeros, original) < 0.30
    # (실제 모델 없이 수치 프록시로 ±1% perplexity bound 근사)

def test_cosine_similarity_noncontiguous_approx_hit():
    # approx sign 히트로 복원한 KV의 cosine sim ≥ 0.60 (유사 세그먼트)
```

**실제 perplexity 측정 위치**: `tests/integration/test_bc_e2e.py`의 `test_perplexity_delta_wikitext2()` — 전체 파이프라인 통합 후 GPT-2로 측정. 단위 테스트에서는 수치 프록시 사용.

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-02.yaml
experiment: "2026-05-02-bc-sign-vq-leverage"
date: "2026-05-02"
activities: [B, C]

cache:
  type: SignVQSegmentCache
  chunk_size: 128
  max_entries: 1000
  hamming_threshold: 0.15      # 해밍 거리 / d_head 임계값 (0.0~0.5)

compressor:
  type: LeverageScoreCompressor
  rank: 32                     # 랭크-k 근사 차수 (k=32 → O(N*32*d))
  reg_lambda: 0.001            # 정규화 항 λ
  tier1_ratio: 0.20            # top 20% → FP16 보존
  tier3_ratio: 0.20            # bottom 20% → evict (middle 60% → 1-bit sign VQ)

scheduler:
  type: default                # Activity A 미포함 — 기본 스케줄러 사용

metrics:
  target_throughput_gain: 0.20          # +20% tokens/sec
  target_memory_reduction: 0.70         # -70% KV 메모리
  target_noncontiguous_hit_rate: 0.30   # 전체 히트의 30% 이상 비연속
  max_perplexity_delta_pct: 1.0         # ±1% perplexity
  max_task_accuracy_delta_pct: 1.0      # ±1% 태스크 정확도
  min_tier2_cosine_similarity: 0.50     # Tier-2 코사인 유사도 하한
  hamming_threshold: 0.15               # approx 히트 임계값

accuracy_benchmarks:
  - name: wikitext2
    metric: perplexity
    tolerance: 0.01
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

- [x] `tests/unit/test_sign_vq_segment.py` (신규)
- [x] `tests/unit/test_leverage_compressor.py` (신규)
- [x] `tests/unit/test_compression_accuracy.py` (Activity C 필수)
- [x] `tests/integration/test_abc_e2e.py` (기존 유지 + B+C 케이스 추가)
- [x] 기존 55개 테스트 전부 통과 (변경 없는 파일은 회귀 없어야 함)

### test_sign_vq_segment.py 필수 테스트 케이스

```python
def test_put_get_exact_fp16_hit():
    # put_segment_compressed() 후 동일 token_ids로 get_segments_with_approx()
    # hit_type == "exact_fp16"

def test_approx_sign_hit_similar_tokens():
    # 유사 token sequence (token_ids 1개 다름)로 approx 히트 발생 여부
    # hamming_threshold=0.15 설정 시 ≥ 1 approx_sign hit

def test_no_approx_hit_dissimilar_tokens():
    # 완전히 다른 token sequence → approx 히트 없음

def test_tier_hit_rates_noncontiguous_ratio():
    # 비연속 히트(approx_sign) 발생 후 noncontiguous_ratio ≥ 0.30

def test_memory_bytes_lower_than_fp16_baseline():
    # 1000개 토큰 저장 후 memory_bytes() < 1000 * 2*d_head * 2 (FP16 baseline)

def test_evict_reduces_memory():
    # max_entries 초과 시 evict() 호출 후 memory_bytes() 감소

def test_cache_store_interface_compliance():
    # CacheStore 6개 추상 메서드 모두 구현 확인
    # put(), get(), evict(), hit_rate(), memory_bytes(), reset_stats()

def test_reset_stats_clears_tier_counters():
    # reset_stats() 후 exact_fp16_hits=0, approx_sign_hits=0
```

### test_leverage_compressor.py 필수 테스트 케이스

```python
def test_leverage_scores_shape():
    # compute_leverage_scores((100, 64)) → shape (100,), all non-negative

def test_classify_tier_sizes():
    # classify(keys, values) with n=100
    # len(tier1) ≈ 20, len(tier2) ≈ 60, len(tier3) ≈ 20

def test_to_sign_code_shape():
    # to_sign_code((100, 64)) → shape (100, 8) uint8

def test_to_sign_code_hamming_distance_self_zero():
    # XOR(to_sign_code(keys), to_sign_code(keys)) popcount == 0

def test_encode_decode_tier1_precision():
    # tier1 FP16 → decode → float32, cosine_sim(decoded, original[tier1]) ≥ 0.99

def test_encode_keys_count():
    # encode() storage["tier1_kv"].shape[0] ≈ 0.20 * n_tokens (±1)

def test_memory_estimate_reduction():
    # memory_bytes_estimate(1000, 64)["reduction_ratio"] ≥ 0.70

def test_edge_case_single_token():
    # n_tokens=1 → tier1 gets 1 token, tier2 and tier3 empty
```

---

## 구현 시 주의사항

1. **CacheStore 인터페이스 준수**: `SignVQSegmentCache`는 `SegmentedHashCache`를 상속하므로 `CacheStore`의 6개 추상 메서드(`put`, `get`, `evict`, `hit_rate`, `memory_bytes`, `reset_stats`)가 자동으로 충족된다. `get_segments_with_approx()`는 추가 메서드이며 기존 `get_segments()`를 대체하지 않는다.

2. **torch.packbits / unpackbits**: PyTorch 1.11+에서 지원. `torch.packbits(x, dim=-1)`는 `x`가 `uint8` 타입이어야 한다. `(keys >= 0).to(torch.uint8)`로 변환 후 사용.

3. **torch.bitwise_count**: PyTorch 2.1+에서 지원. 이전 버전에서는 `_popcount_uint8()` lookup table fallback 사용.

4. **랭크-k 근사 안정성**: `torch.linalg.eigh()`는 실수 대칭 행렬에 대해 항상 실수 고유값을 반환한다. `N < rank` 인 경우(토큰 수가 rank보다 작을 때) `rank = min(rank, N, d_head)`로 자동 조정.

5. **훈련-무료 제약**: `LeverageScoreCompressor`와 `SignVQSegmentCache`는 어떠한 학습 파라미터(nn.Parameter, nn.Module)도 포함하지 않는다. 이전 사이클의 `SegmentAdapter` MLP는 이번 사이클에서 사용하지 않는다.

6. **모든 단위 테스트는 CPU 텐서로 통과**: `torch.device("cpu")` 기준 작성. GPU 불필요.

7. **시드 고정**: 모든 테스트에서 `torch.manual_seed(42)` 사용.

8. **`_sign_store`와 `_store` 동기화**: 동일 key가 두 저장소에 동시에 존재할 수 있다(Tier-1 FP16 + Tier-2 sign). 이는 의도된 동작이다. 퇴거 시 두 저장소에서 모두 제거하도록 `evict()` 오버라이드 필요 (부모 LRU 퇴거와 별도 sign store LRU 퇴거를 순서대로 실행).

9. **`get_segments()` 하위 호환성**: 기존 `get_segments(token_ids, layer_idx)` 시그니처는 그대로 유지. `get_segments_with_approx()`는 새 메서드로 추가.

---

## 완료 기준 (Definition of Done)

- [ ] `tests/unit/test_sign_vq_segment.py` — 8개 테스트 케이스 전부 통과
- [ ] `tests/unit/test_leverage_compressor.py` — 8개 테스트 케이스 전부 통과
- [ ] `tests/unit/test_compression_accuracy.py` — 8개 테스트 케이스 전부 통과 (§4 Accuracy 필수)
- [ ] 기존 55개 테스트 회귀 없이 통과
- [ ] `configs/experiments/2026-05-02.yaml` 존재 (§0 설정 YAML 필수)
- [ ] `evaluation_criteria.md` §3 Activity B:
  - 비연속 세그먼트 히트율(approx_sign) ≥ 전체 히트의 30% (높음)
  - KV Memory Footprint: 베이스라인 대비 +20% 이내 (높음)
- [ ] `evaluation_criteria.md` §4 Activity C:
  - perplexity 변화 ±1% 이내 (필수)
  - 태스크 정확도 변화 ±1% 이내 (필수)
  - KV Memory Reduction ≥ −70% (목표; 최소 −30%) (높음)
  - Effective Context Length 2× 이상 (높음)
- [ ] `evaluation_criteria.md` §5 Cross:
  - 복합 Memory Reduction: 단일 Activity 대비 추가 −10% 이상 (높음)
  - Accuracy 보존 복합 적용 후 ±1% 이내 (필수)
- [ ] 타입 힌트 모든 공개 함수·메서드에 존재 (§0 중간)

SPEC_SAVED
