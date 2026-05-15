<!-- 변경 이유 (이전 Spec.md: 2026-05-14 대비):
이전 사이클(2026-05-14)은 B+C (FibQuantVQCodec + FibQuantVQSegmentCache +
FibQuantPositionFreeSegmentCache) 조합이었다.
이번 사이클은 B+C를 유지하면서 알고리즘을 완전히 교체하고, Activity A(스케줄링)를
보조적으로 추가한다.

주요 변경:
1. [Activity C 교체] FibQuantVQCodec(구형-베타 방사상-각도 VQ) →
   LookaheadKVEvictionCodec(드래프트-프리 미래-인식 LoRA 퇴거).
   기존 코덱은 KV 텐서 자체를 VQ 압축하는 반면, 이번 코덱은 룩어헤드 토큰+LoRA로
   미래 응답 어텐션 패턴을 예측해 중요도 낮은 토큰의 KV를 퇴거한다. 퇴거(eviction)
   기반이므로 보존 KV는 FP16 원본이며 압축 왜곡 없음. ICLR 2026 실증 기반.

2. [Activity B 교체] FibQuantVQSegmentCache(VQ 압축 세그먼트) →
   RelayUShapeLayerSelectiveSegmentCache(U자형 레이어 편차 기반 레이어-선택적 세그먼트 재사용).
   기존 구현이 세그먼트 전체를 압축 저장하는 반면, 이번 구현은 RelayCaching(2603.13289)의
   U자형 레이어 편차 이론을 적용해 비동일 세그먼트에서도 중간 레이어 70~80%를 재사용한다.

3. [Cross B+C 교체] FibQuantPositionFreeSegmentCache(pre-RoPE+FibQuant) →
   LookaheadRelaySegmentCache(레이어 필터 + 토큰 미래-인식 필터 이중-필터).
   B-1 RelayUShapeLayerSelectiveSegmentCache(레이어 선택) +
   C-1 LookaheadKVEvictionCodec(토큰 선택)의 순차 파이프라인.

4. [Activity A 보조 추가] RadixFeatherBatchScheduler(Feather RL 배치 중단 정책 +
   Radix 트리 동질성 신호)를 보조 구성 요소로 추가.
   구현 난이도 상 B+C 완성 후 순차 구현. RL 학습 루프는 선택적(static 모드로도 평가 가능).

5. [보존 파일] 기존 모든 파일(fibquant_vq_codec.py, fibquant_vq_segment_cache.py,
   fibquant_position_free_segment.py 등)은 이번 사이클에서 수정하지 않는다.
   기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.

6. [인터페이스 유지] src/cache/base.py는 이번 사이클에서 수정하지 않는다.
   CacheStore 6개 추상 메서드를 모든 신규 구현체가 완전 구현한다.

7. [Activity C 필수] LookaheadKVEvictionCodec은 accuracy-preserving 검증 계획 없이 완성 불가.
   WikiText-2 perplexity proxy ±1% + LongBench 8개 서브태스크 proxy +
   퇴거 비율(0.5/0.7/0.85)별 정확도 곡선을 tests/unit/test_lookahead_kv_accuracy.py에 구현.
-->

# Spec — 2026-05-15

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-15.md`
**최우선 구현 타겟**:
- **C-1 (주)**: LookaheadKVEvictionCodec — 드래프트-프리 미래-인식 LoRA KV 퇴거 코덱
  (LookaheadKV, arXiv 2603.10899 / ICLR 2026, Samsung AI Research)
- **B-1 (주)**: RelayUShapeLayerSelectiveSegmentCache — U자형 레이어 편차 기반 레이어-선택적
  세그먼트 재사용 (RelayCaching, arXiv 2603.13289)
- **Cross-1 (주)**: LookaheadRelaySegmentCache — B-1 레이어 필터 + C-1 토큰 미래-인식 필터
  이중 필터 파이프라인
- **A-1 (보조)**: RadixFeatherBatchScheduler — Radix 트리 동질성 신호 기반 Feather RL 배치 중단 정책
  (Feather, arXiv 2605.06046)

**해결하려는 문제**:

- **Activity C**: 기존 KV 퇴거 기법(H2O, SnapKV, LaProx)은 현재 어텐션 통계로 토큰 중요도를
  추정해 미래 응답이 실제로 참조할 토큰을 정확히 예측하지 못한다. LookaheadKV는 학습 가능한
  룩어헤드 토큰(소프트 쿼리)과 경량 LoRA 어댑터로 미래 응답의 어텐션 패턴을 드래프트 생성 없이
  직접 예측한다. 퇴거된 토큰의 KV를 제거해 메모리 −70~85%를 달성하면서 accuracy delta ±1%
  이내를 보장한다.

- **Activity B**: 표준 비연속 세그먼트 캐시는 세그먼트가 byte-identical할 때만 재사용한다.
  RelayCaching(2603.13289)이 발견한 "프리픽스-유도 KV 편차가 U자형으로 국소화된다" — 첫 레이어와
  마지막 레이어에서 편차 크고 중간 레이어에서 작다 — 는 이론에 따라, 비동일 세그먼트에서도
  중간 레이어 70~80%의 KV를 재사용하고 경계 레이어만 재계산하는 레이어-선택적 그레이디드 재사용이
  가능하다.

- **Cross B+C**: B-1의 레이어 필터(U자형 프로파일)와 C-1의 토큰 필터(미래-인식 중요도)를 순차로
  적용하면 "이전 에이전트 KV 중 (i) 편차 작은 중간 레이어에서 (ii) 미래 응답이 높게 참조할
  토큰만" 이중 필터링으로 선택 재사용할 수 있다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling  (RadixFeatherBatchScheduler — 보조, RL 정책 단순화 버전)
- [x] Activity B: Non-Contiguous KV Cache Reuse  (RelayUShapeLayerSelectiveSegmentCache)
- [x] Activity C: KV Cache Compression  (LookaheadKVEvictionCodec)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — attention output relative error < 0.01 (WikiText-2 proxy)
      — 퇴거 비율 0.5/0.7/0.85 각각 측정
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 변화 ±1% 이내
      — LongBench 8개 서브태스크 proxy (KL divergence < 0.015, cosine >= 0.99)
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C): KV Cache Memory Reduction >= −30%
      — LookaheadKVEvictionCodec 퇴거 비율 0.7 시 −70% 목표
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C): Effective Context Length 동일 메모리 2× 이상
      — 70% 퇴거 시 보존 KV가 30%이므로 컨텍스트 길이 ~3× 증가 가능
- [ ] 목표 5 (evaluation_criteria.md §3 Activity B): 비연속 세그먼트 히트율 >= 30%
      — 비동일 세그먼트에서 중간 레이어 70~80% 재사용으로 달성
- [ ] 목표 6 (evaluation_criteria.md §3 Activity B): 전체 Cache Hit Rate 베이스라인 대비 +5%p
- [ ] 목표 7 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +20% 이상
      — TTFT 단축(레이어 재계산 20~30% 감소) + 퇴거로 KV 접근 속도 향상
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후에도 accuracy ±1% 이내
      — LookaheadRelaySegmentCache(Cross B+C) 기준 이중 필터 적용 후 측정
- [ ] 목표 9 (evaluation_criteria.md §4 Activity C): 압축(퇴거) 오버헤드 TTFT +10% 이내
      — 룩어헤드 토큰 어텐션 계산 추가 지연 측정 (n_la=5 토큰, LoRA rank=8)
- [ ] 목표 10 (evaluation_criteria.md §2 Activity A): 스케줄링 오버헤드 TTFT p50 +5% 이내
      — RadixFeatherBatchScheduler 동질성 점수 계산 지연 측정

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/lookahead_kv_eviction.py` | C | LookaheadKVEvictionCodec: 룩어헤드 토큰+LoRA 미래-인식 KV 퇴거 |
| `src/cache/relay_ulayer_segment.py` | B | RelayUShapeLayerSelectiveSegmentCache: U자형 레이어-선택적 세그먼트 재사용 |
| `src/cache/lookahead_relay_segment.py` | B+C | LookaheadRelaySegmentCache: 레이어 필터+토큰 필터 이중 파이프라인 |
| `src/scheduler/radix_feather_batch.py` | A | RadixFeatherBatchScheduler: Radix 트리 동질성 신호 + 배치 중단 정책 |
| `experiments/train_lookahead_lora.py` | C | 룩어헤드 토큰+LoRA 어댑터 훈련 스크립트 (≤1 GPU-hour) |
| `experiments/run_relay_layer_calibration.py` | B | 레이어 범위 프로파일러 (오프라인 1회, 100~200 세그먼트 쌍) |
| `configs/relay_ulayer_profile.yaml` | B | 레이어 재사용 범위 프로파일 저장 |
| `tests/unit/test_lookahead_kv_accuracy.py` | C | Activity C accuracy-preserving 검증 (필수) |
| `tests/unit/test_relay_ulayer_segment.py` | B | 비연속 히트율·레이어 마스크·메모리 단위 테스트 |
| `tests/unit/test_lookahead_relay_segment.py` | B+C | Cross B+C 이중 필터 단위 테스트 |
| `tests/unit/test_radix_feather_scheduler.py` | A | 배치 동질성 신호·스케줄링 오버헤드 단위 테스트 |
| `tests/integration/test_cross_bc_lookahead_relay.py` | B+C | E2E 통합 테스트 |
| `configs/experiments/2026-05-15.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/segmented.py` | `load_layers(segment_id, layer_mask)` 메서드 추가 — 특정 레이어만 선택 로드 인터페이스 |

---

## 알고리즘 상세

### LookaheadKVEvictionCodec (Activity C)

LookaheadKV(arXiv 2603.10899, ICLR 2026) 설계를 Activity C에 적용한 미래-인식 KV 퇴거 코덱.
기구현 FibQuantVQCodec(압축 왜곡 기반)과 LaProx(출력 행렬곱 근사 현재 통계 기반)와의 차이:
LookaheadKVEvictionCodec은 미래 응답의 어텐션 패턴을 직접 예측해 퇴거 결정하므로 보존 KV는
FP16 원본이며 양자화 왜곡이 없다.

```python
# src/cache/lookahead_kv_eviction.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class LookaheadKVConfig:
    n_layers: int = 12           # 모델 레이어 수
    n_heads: int = 8             # 어텐션 헤드 수
    d_head: int = 64             # 헤드 차원
    n_lookahead: int = 5         # 룩어헤드 토큰 수 (n_la)
    lora_rank: int = 8           # LoRA 어댑터 랭크 (8 또는 16)
    eviction_ratio: float = 0.7  # 퇴거 비율 (기본 70% 퇴거, 30% 보존)
    blend_ratio: float = 0.0     # LaProx 앙상블 비율 (0.0 = 순수 LookaheadKV)
    recent_window: int = 4       # 최근 N 토큰은 항상 보존 (퇴거 예외)
    seed: int = 42
    max_entries: int = 1000      # CacheStore용 최대 항목 수


class LookaheadModule(nn.Module):
    """학습 가능한 룩어헤드 토큰 + 경량 LoRA 어댑터.

    각 레이어·헤드당 독립 룩어헤드 쿼리 파라미터를 보유한다.
    모델 가중치를 동결한 채 이 모듈만 보정 데이터로 훈련한다.
    """

    def __init__(self, config: LookaheadKVConfig) -> None:
        super().__init__()
        self.config = config
        # 룩어헤드 토큰: [n_layers, n_heads, n_la, d_head]
        self.lookahead_tokens = nn.Parameter(
            torch.randn(
                config.n_layers,
                config.n_heads,
                config.n_lookahead,
                config.d_head,
            ) * 0.02
        )
        # LoRA 어댑터 (각 레이어용 Q 프로젝션 보정)
        # A: [n_layers, d_head, lora_rank], B: [n_layers, lora_rank, d_head]
        self.lora_A = nn.Parameter(
            torch.randn(config.n_layers, config.d_head, config.lora_rank) * 0.02
        )
        self.lora_B = nn.Parameter(
            torch.zeros(config.n_layers, config.lora_rank, config.d_head)
        )

    def forward(
        self,
        key: torch.Tensor,    # [n_tokens, n_heads, d_head] — 현재 레이어 K
        layer_idx: int,
    ) -> torch.Tensor:
        """룩어헤드 어텐션 점수 계산 → 토큰별 미래 참조 중요도.

        Algorithm:
          1. LoRA 보정 적용: la_q = lookahead_tokens[layer] + la_q @ lora_A @ lora_B
          2. 룩어헤드 쿼리와 K의 어텐션 점수: [n_la, n_tokens]
          3. max over (n_la, n_heads) → 토큰별 중요도 [n_tokens]

        Returns:
            importance: Tensor[n_tokens] — 값이 클수록 미래 응답이 많이 참조함
        """
        n_tokens, n_heads, d_head = key.shape
        # LoRA 보정된 룩어헤드 쿼리: [n_heads, n_la, d_head]
        la_q_base = self.lookahead_tokens[layer_idx]  # [n_heads, n_la, d_head]
        lora_delta = la_q_base.reshape(n_heads * self.config.n_lookahead, d_head)
        lora_delta = lora_delta @ self.lora_A[layer_idx] @ self.lora_B[layer_idx]
        la_q = la_q_base + lora_delta.reshape(n_heads, self.config.n_lookahead, d_head)

        # 어텐션 점수: [n_heads, n_la, n_tokens]
        scale = d_head ** -0.5
        k_t = key.permute(1, 2, 0)  # [n_heads, d_head, n_tokens]
        scores = torch.bmm(la_q, k_t) * scale  # [n_heads, n_la, n_tokens]

        # 토큰별 최대 점수 (heads × lookahead 중 최댓값)
        importance = scores.max(dim=1).values.max(dim=0).values  # [n_tokens]
        return importance


class LookaheadKVEvictionCodec(CacheStore):
    """Draft-free future-aware KV eviction using learnable lookahead tokens + LoRA.

    Activity C: KV Cache Compression via token eviction.
    보존 KV는 FP16 원본 — 양자화 왜곡 없음.
    퇴거 비율(eviction_ratio)만큼 중요도 하위 토큰 KV를 제거한다.

    CacheStore 인터페이스 완전 준수:
      put / get / evict / hit_rate / memory_bytes / reset_stats
    선택적 메서드 미구현 (pre-RoPE 불필요): store_pre_rope / load_with_rope는
    base.py 기본값(NotImplementedError) 사용.

    compression_hook() 오버라이드:
      put() 호출 전 LookaheadKV 중요도 점수로 토큰 퇴거 후 저장.
    """

    def __init__(self, config: LookaheadKVConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._lookahead = LookaheadModule(config)
        self._lookahead.eval()  # 추론 시 eval 모드; 훈련 시 train()으로 전환
        # 내부 저장소: key -> (kv_tensor, keep_mask)
        from collections import OrderedDict
        self._store: "OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]" = \
            OrderedDict()
        self._hits = 0
        self._misses = 0
        self._total_tokens_original = 0
        self._total_tokens_kept = 0

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store KV after LookaheadKV eviction via compression_hook().

        value shape: [n_tokens, 2, n_heads, d_head] or [n_tokens, d_head]
        """
        compressed = self.compression_hook(key, value)
        if len(self._store) >= self.config.max_entries:
            self.evict()
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (compressed, torch.ones(compressed.shape[0], dtype=torch.bool))

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve evicted-KV (returns only kept tokens)."""
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        kv, _ = self._store[key]
        return kv

    def evict(self) -> int:
        """LRU eviction. Returns freed bytes."""
        if not self._store:
            return 0
        oldest_key, (kv, _) = self._store.popitem(last=False)
        return kv.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(kv.nbytes for kv, _ in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_tokens_original = 0
        self._total_tokens_kept = 0

    # ------------------------------------------------------------------ #
    # Activity C 핵심: compression_hook 오버라이드                         #
    # ------------------------------------------------------------------ #

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,   # [n_tokens, 2, n_heads, d_head]
    ) -> torch.Tensor:
        """LookaheadKV 퇴거: 중요도 하위 eviction_ratio% 토큰 KV 제거.

        Algorithm:
          1. key tensor에서 layer_idx 추출 (key 포맷: "layer{i}:{content_hash}")
          2. LookaheadModule.forward(K, layer_idx) → importance [n_tokens]
          3. recent_window 토큰은 항상 보존 (importance += INF)
          4. eviction_ratio 비율로 하위 토큰 마스킹 → keep_mask
          5. value[keep_mask] 반환 (kept_tokens < n_tokens)

        Returns:
            Tensor[kept_tokens, 2, n_heads, d_head] — 퇴거 후 KV
        """
        ...

    # ------------------------------------------------------------------ #
    # 퇴거 메트릭                                                           #
    # ------------------------------------------------------------------ #

    def eviction_rate(self) -> float:
        """실제 퇴거 비율 = (원본 토큰 - 보존 토큰) / 원본 토큰."""
        if self._total_tokens_original == 0:
            return 0.0
        return 1.0 - self._total_tokens_kept / self._total_tokens_original

    def memory_reduction_ratio(self) -> float:
        """베이스라인 FP16 대비 메모리 감소율 (0.7 eviction → 0.7 감소)."""
        return self.eviction_rate()

    # ------------------------------------------------------------------ #
    # 훈련 지원                                                             #
    # ------------------------------------------------------------------ #

    def train_lookahead(
        self,
        calibration_data: List[torch.Tensor],  # List of [n_tokens, 2, n_heads, d_head]
        layer_idx: int,
        n_epochs: int = 5,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """보정 데이터로 룩어헤드 토큰 + LoRA 어댑터 파인튜닝.

        목표: 룩어헤드 어텐션 점수가 실제 다음 토큰 생성 시 어텐션 패턴과 정렬.
        손실: MSE(lookahead_score_i, normalized_future_attention_score_i)

        Returns: {"final_loss": float, "n_samples": int}
        """
        ...

    def save(self, path: str) -> None:
        torch.save(
            {
                "lookahead_state_dict": self._lookahead.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self._lookahead.load_state_dict(data["lookahead_state_dict"])
        self.config = data["config"]
```

---

### RelayUShapeLayerSelectiveSegmentCache (Activity B)

RelayCaching(arXiv 2603.13289)의 U자형 레이어 편차 이론을 비연속 세그먼트 재사용에 적용.
기존 SegmentedHashCache(all-or-nothing 이진 결정)와의 차이: 세그먼트가 byte-identical하지 않아도
중간 레이어(편차 작음)의 KV를 재사용하고 경계 레이어(편차 큼)만 재계산하는 연속 결정 구조.

`SegmentedHashCache`를 내부적으로 위임(composition)으로 사용하고, `CacheStore` 인터페이스를
완전히 구현한다. `segmented.py`의 `load_layers()` 확장 인터페이스를 활용한다.

```python
# src/cache/relay_ulayer_segment.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import yaml

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class RelayULayerConfig:
    chunk_size: int = 128         # 세그먼트 크기 (토큰 수)
    max_entries: int = 1000       # 최대 캐시 세그먼트 수
    n_layers: int = 12            # 모델 총 레이어 수
    n_heads: int = 8
    d_head: int = 64
    similarity_threshold: float = 0.95  # 레이어 재사용 임계값 τ_layer
    profile_path: str = "configs/relay_ulayer_profile.yaml"  # 레이어 범위 프로파일
    seed: int = 42


@dataclass
class LayerReuseProfile:
    """오프라인 프로파일링 결과: 레이어별 재사용 가능 여부."""
    n_layers: int
    reuse_layer_indices: List[int]   # τ_layer 이상 레이어 인덱스 (R_reuse)
    boundary_layer_indices: List[int]  # 재계산 필요 레이어 인덱스
    similarity_scores: List[float]   # 레이어별 평균 코사인 유사도

    @classmethod
    def from_yaml(cls, path: str) -> "LayerReuseProfile":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            n_layers=data["n_layers"],
            reuse_layer_indices=data["reuse_layer_indices"],
            boundary_layer_indices=data["boundary_layer_indices"],
            similarity_scores=data["similarity_scores"],
        )

    def to_bitmask(self) -> bytes:
        """n_layers 비트마스크: 재사용 가능 레이어=1."""
        mask = 0
        for idx in self.reuse_layer_indices:
            mask |= (1 << idx)
        return mask.to_bytes((self.n_layers + 7) // 8, byteorder="little")


class RelayUShapeLayerSelectiveSegmentCache(CacheStore):
    """Non-contiguous segment cache with U-shape layer-selective reuse.

    비동일 세그먼트에서도 편차 작은 중간 레이어(R_reuse)의 KV를 재사용하고
    경계 레이어만 재계산한다. 세그먼트 키에 layer_reuse_mask(비트마스크)를
    함께 저장해 로드 시 레이어별 선택적 반환을 지원한다.

    CacheStore 인터페이스 완전 준수:
      put / get / evict / hit_rate / memory_bytes / reset_stats
    """

    def __init__(self, config: RelayULayerConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._base_cache = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        # 레이어 프로파일 로드 (존재하지 않으면 기본값으로 중간 레이어 범위 사용)
        self._profile: Optional[LayerReuseProfile] = self._load_profile()
        # segment_id → layer_reuse_mask (bytes)
        self._layer_masks: Dict[str, bytes] = {}
        self._hits = 0
        self._misses = 0
        self._partial_reuse_hits = 0   # 비동일 세그먼트 레이어-선택 재사용 횟수
        self._noncontiguous_hits = 0

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store segment KV with full-layer mask (byte-identical path).

        value shape: [n_tokens, n_layers, 2, n_heads, d_head] 또는
                     [n_tokens, 2, n_heads, d_head] (단일 레이어)
        """
        self._base_cache.put(key, value)
        # 완전 일치 세그먼트: 모든 레이어 재사용 가능
        all_reuse_mask = ((1 << self.config.n_layers) - 1).to_bytes(
            (self.config.n_layers + 7) // 8, byteorder="little"
        )
        self._layer_masks[key] = all_reuse_mask

    def get(self, key: str) -> Optional[torch.Tensor]:
        result = self._base_cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        freed = self._base_cache.evict()
        return freed

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._base_cache.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._partial_reuse_hits = 0
        self._noncontiguous_hits = 0
        self._base_cache.reset_stats()

    # ------------------------------------------------------------------ #
    # 레이어-선택적 재사용 핵심 API                                         #
    # ------------------------------------------------------------------ #

    def put_with_layer_mask(
        self,
        key: str,
        value: torch.Tensor,         # [n_tokens, n_layers, 2, n_heads, d_head]
        layer_reuse_mask: bytes,     # n_layers 비트마스크
    ) -> None:
        """레이어 마스크와 함께 세그먼트 저장."""
        self._base_cache.put(key, value)
        self._layer_masks[key] = layer_reuse_mask

    def get_with_layer_selection(
        self,
        key: str,
        target_layer_indices: Optional[List[int]] = None,
    ) -> Optional[Tuple[torch.Tensor, List[int], List[int]]]:
        """레이어 마스크 기반 선택적 로드.

        Returns:
            (kv_tensor, reusable_layers, boundary_layers) 또는 None on miss.
            reusable_layers: 재사용 가능 레이어 인덱스 목록
            boundary_layers: 재계산 필요 레이어 인덱스 목록
        """
        kv = self._base_cache.get(key)
        if kv is None:
            return None
        mask_bytes = self._layer_masks.get(key)
        reusable, boundary = self._decode_layer_mask(mask_bytes)
        if target_layer_indices is not None:
            target_set = set(target_layer_indices)
            reusable = [l for l in reusable if l in target_set]
            boundary = [l for l in boundary if l in target_set]
        return kv, reusable, boundary

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,     # [n_tokens, n_layers, 2, n_heads, d_head]
        layer_idx: int = 0,
    ) -> None:
        """Content-hash 키 기반 세그먼트 저장 (레이어 전체)."""
        key = self._chunk_key(token_ids, chunk_idx)
        if self._profile is not None:
            mask = self._profile.to_bitmask()
        else:
            # 프로파일 없으면 중간 70% 레이어를 기본 재사용 대상으로 설정
            mask = self._default_middle_layer_mask()
        self.put_with_layer_mask(key, kv, mask)

    def get_segments_layer_selective(
        self,
        token_ids: List[int],
        request_kv: Optional[torch.Tensor] = None,  # 현재 요청 KV (유사도 계산용)
    ) -> Tuple[List[Tuple[int, torch.Tensor, List[int], List[int]]], List[int]]:
        """전체 청크 조회 + 레이어-선택적 반환.

        Returns:
            hits: [(chunk_idx, kv, reusable_layers, boundary_layers), ...]
            misses: [chunk_idx, ...]

        비연속 히트 추적:
            히트가 발생했는데 그 이전 청크 중 미스가 있으면 noncontiguous_hit 카운트.
        """
        n_chunks = max(1, (len(token_ids) + self.config.chunk_size - 1) //
                       self.config.chunk_size)
        hits = []
        misses = []
        for i in range(n_chunks):
            key = self._chunk_key(token_ids, i)
            result = self.get_with_layer_selection(key)
            if result is not None:
                kv, reusable, boundary = result
                hits.append((i, kv, reusable, boundary))
                if any(m < i for m in misses):
                    self._noncontiguous_hits += 1
                    self._partial_reuse_hits += 1
            else:
                misses.append(i)
        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        total = self._hits
        if total == 0:
            return 0.0
        return self._noncontiguous_hits / total

    def partial_reuse_rate(self) -> float:
        """비동일 세그먼트에서 레이어-선택 재사용이 발생한 비율."""
        total = self._hits
        if total == 0:
            return 0.0
        return self._partial_reuse_hits / total

    # ------------------------------------------------------------------ #
    # 내부 헬퍼                                                             #
    # ------------------------------------------------------------------ #

    def _load_profile(self) -> Optional[LayerReuseProfile]:
        import os
        if os.path.exists(self.config.profile_path):
            return LayerReuseProfile.from_yaml(self.config.profile_path)
        return None

    def _decode_layer_mask(
        self,
        mask_bytes: Optional[bytes],
    ) -> Tuple[List[int], List[int]]:
        """비트마스크 → (reusable_layers, boundary_layers) 인덱스 목록."""
        n = self.config.n_layers
        if mask_bytes is None:
            # 프로파일 없음: 중간 70% 재사용
            reusable = list(range(n // 6, n - n // 6))
            boundary = [i for i in range(n) if i not in reusable]
            return reusable, boundary
        mask_int = int.from_bytes(mask_bytes, byteorder="little")
        reusable = [i for i in range(n) if (mask_int >> i) & 1]
        boundary = [i for i in range(n) if not ((mask_int >> i) & 1)]
        return reusable, boundary

    def _default_middle_layer_mask(self) -> bytes:
        """프로파일 없을 때 기본 중간 레이어 마스크 (n_layers의 중간 70%)."""
        n = self.config.n_layers
        start = n // 6
        end = n - n // 6
        mask = 0
        for i in range(start, end):
            mask |= (1 << i)
        return mask.to_bytes((n + 7) // 8, byteorder="little")

    def _chunk_key(self, token_ids: List[int], chunk_idx: int) -> str:
        """content-hash 기반 세그먼트 키 (SegmentedHashCache와 동일 방식)."""
        return self._base_cache.chunk_key(token_ids, chunk_idx, layer_idx=0)
```

---

### SegmentedHashCache 확장 — load_layers() (Activity B 지원)

기존 `src/cache/segmented.py`에 다음 메서드를 추가한다. 기존 메서드는 변경하지 않는다.

```python
# src/cache/segmented.py 에 추가할 메서드

def chunk_key(
    self,
    token_ids: List[int],
    chunk_idx: int,
    layer_idx: int = 0,
) -> str:
    """공개 청크 키 계산 메서드 (기존 _chunk_key 래퍼).

    RelayUShapeLayerSelectiveSegmentCache 등 외부 클래스가 동일한
    content-hash 키를 생성할 때 사용한다.
    """
    # 기존 _chunk_key 로직과 동일하게 구현
    ...

def load_layers(
    self,
    segment_id: str,
    layer_mask: Optional[bytes] = None,
) -> Optional[Tuple[torch.Tensor, List[int]]]:
    """특정 레이어만 선택 로드.

    Args:
        segment_id: 세그먼트 키
        layer_mask: 비트마스크 bytes (None이면 전 레이어 반환)

    Returns:
        (kv_tensor, reusable_layer_indices) 또는 None on miss
    """
    kv = self.get(segment_id)
    if kv is None:
        return None
    if layer_mask is None:
        return kv, list(range(kv.shape[1] if kv.dim() >= 2 else 0))
    # 비트마스크 기반 레이어 인덱스 추출
    mask_int = int.from_bytes(layer_mask, byteorder="little")
    n_layers = kv.shape[1] if kv.dim() >= 3 else 1
    reusable = [i for i in range(n_layers) if (mask_int >> i) & 1]
    # 레이어 차원이 있으면 선택; 없으면 전체 반환
    if kv.dim() >= 3:
        selected_kv = kv[:, reusable, ...]
    else:
        selected_kv = kv
    return selected_kv, reusable
```

---

### LookaheadRelaySegmentCache (Cross B+C)

B-1 RelayUShapeLayerSelectiveSegmentCache(레이어 필터)와
C-1 LookaheadKVEvictionCodec(토큰 필터)의 순차 이중-필터 파이프라인.

```python
# src/cache/lookahead_relay_segment.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.relay_ulayer_segment import (
    RelayUShapeLayerSelectiveSegmentCache,
    RelayULayerConfig,
)
from src.cache.lookahead_kv_eviction import (
    LookaheadKVEvictionCodec,
    LookaheadKVConfig,
)


@dataclass
class LookaheadRelayConfig:
    """B+C 통합 파이프라인 설정."""
    # B-1 설정
    relay_config: RelayULayerConfig = None   # type: ignore
    # C-1 설정
    lookahead_config: LookaheadKVConfig = None  # type: ignore
    # Cross 전용 파라미터
    token_importance_threshold: float = 0.3   # τ_token: 이 이상인 토큰만 재사용
    max_entries: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.relay_config is None:
            self.relay_config = RelayULayerConfig()
        if self.lookahead_config is None:
            self.lookahead_config = LookaheadKVConfig()


class LookaheadRelaySegmentCache(CacheStore):
    """B+C dual-filter: U-shape layer filter (B-1) + future-aware token filter (C-1).

    에이전트 KV 수신 시 처리 파이프라인:
      Step 1 (레이어 필터): RelayUShapeLayerSelectiveSegmentCache로
                            U자형 레이어 범위 프로파일 → R_reuse 결정
      Step 2 (토큰 필터): LookaheadKVEvictionCodec의 LookaheadModule로
                          R_reuse 레이어 KV에서 토큰별 미래-인식 중요도 계산
      Step 3 (재사용 결정): 중요도 > τ_token(0.3) 토큰만 캐시에 보존·재사용

    전체 KV 대비 20~30%만 보존해 최대 메모리 효율을 달성한다.

    CacheStore 인터페이스 완전 준수.
    """

    def __init__(self, config: LookaheadRelayConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._relay_cache = RelayUShapeLayerSelectiveSegmentCache(config.relay_config)
        self._eviction_codec = LookaheadKVEvictionCodec(config.lookahead_config)
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """이중 필터 적용 후 저장.

        Step 1: 레이어 선택 (프로파일 기반 중간 레이어만 보존)
        Step 2: 보존 레이어의 토큰 선택 (LookaheadKV 중요도 > τ_token)
        Step 3: 선택된 KV만 relay_cache에 저장
        """
        layer_filtered = self._apply_layer_filter(value)
        token_filtered = self._apply_token_filter(layer_filtered, key)
        self._relay_cache.put(key, token_filtered)

    def get(self, key: str) -> Optional[torch.Tensor]:
        result = self._relay_cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self._relay_cache.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._relay_cache.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._relay_cache.reset_stats()
        self._eviction_codec.reset_stats()

    # ------------------------------------------------------------------ #
    # 이중 필터 파이프라인                                                  #
    # ------------------------------------------------------------------ #

    def _apply_layer_filter(
        self,
        kv: torch.Tensor,   # [n_tokens, n_layers, 2, n_heads, d_head]
    ) -> torch.Tensor:
        """Step 1: U자형 프로파일 기반 중간 레이어만 선택.

        프로파일이 없으면 기본 중간 70% 레이어 사용.
        Returns: [n_tokens, n_reuse_layers, 2, n_heads, d_head]
        """
        profile = self._relay_cache._profile
        if profile is not None:
            reuse_idx = profile.reuse_layer_indices
        else:
            n = self.config.relay_config.n_layers
            reuse_idx = list(range(n // 6, n - n // 6))
        if kv.dim() >= 3:
            return kv[:, reuse_idx, ...]
        return kv

    def _apply_token_filter(
        self,
        kv: torch.Tensor,   # [n_tokens, ...]
        key: str,
    ) -> torch.Tensor:
        """Step 2: LookaheadKV 중요도 > τ_token 토큰만 보존.

        layer_idx는 key에서 추출 (기본 0).
        Returns: [kept_tokens, ...]
        """
        if kv.dim() < 3:
            return kv
        n_tokens = kv.shape[0]
        # K 텐서 추출: 단일 레이어 또는 첫 번째 레이어
        if kv.dim() == 5:  # [n_tokens, n_layers, 2, n_heads, d_head]
            k = kv[:, 0, 0, :, :]  # [n_tokens, n_heads, d_head]
        elif kv.dim() == 4:  # [n_tokens, 2, n_heads, d_head]
            k = kv[:, 0, :, :]     # [n_tokens, n_heads, d_head]
        else:
            return kv

        with torch.no_grad():
            importance = self._eviction_codec._lookahead.forward(k, layer_idx=0)
        # recent_window 토큰은 항상 보존
        rw = self.config.lookahead_config.recent_window
        if rw > 0:
            importance[-rw:] = float("inf")
        keep_mask = importance >= self.config.token_importance_threshold
        return kv[keep_mask]

    # ------------------------------------------------------------------ #
    # 세그먼트 API                                                          #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """이중 필터 적용 후 세그먼트 저장."""
        key = self._relay_cache._chunk_key(token_ids, chunk_idx)
        self.put(key, kv)

    def get_segments(
        self,
        token_ids: List[int],
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """전체 청크 이중 필터 조회.

        Returns: (hits[(chunk_idx, kv)], misses[chunk_idx])
        """
        n_chunks = max(
            1,
            (len(token_ids) + self.config.relay_config.chunk_size - 1)
            // self.config.relay_config.chunk_size,
        )
        hits = []
        misses = []
        for i in range(n_chunks):
            key = self._relay_cache._chunk_key(token_ids, i)
            kv = self.get(key)
            if kv is not None:
                hits.append((i, kv))
                if any(m < i for m in misses):
                    self._noncontiguous_hits += 1
            else:
                misses.append(i)
        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        total = self._hits
        if total == 0:
            return 0.0
        return self._noncontiguous_hits / total
```

---

### RadixFeatherBatchScheduler (Activity A — 보조)

Feather(arXiv 2605.06046)의 배치 크기 대 프리픽스 동질성 트레이드오프를
기존 `src/cache/radix.py` Radix 트리에서 추출한 실시간 동질성 신호로 결정.
RL 정책은 단순화된 threshold-based 정적 모드로 우선 구현하고,
PPO-Clip 2층 MLP 모드를 선택적으로 추가한다.

```python
# src/scheduler/radix_feather_batch.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import torch

from src.cache.radix import RadixAttentionCache  # 기구현 Radix 트리


@dataclass
class RadixFeatherConfig:
    chunk_size: int = 256          # 동질성 체크 단위 토큰 수
    target_batch_size: int = 8     # 목표 최대 배치 크기
    homogeneity_threshold: float = 0.6  # 동질성 임계값 (정적 모드)
    scheduler_mode: str = "static"  # "static" | "rl" (RL은 선택적)
    max_wait_ratio: float = 2.0    # 최대 대기 시간 배수 (공정성)
    seed: int = 42


class RadixFeatherBatchScheduler:
    """Prefix-homogeneity-aware batch scheduler based on Feather (2605.06046).

    배치 구성 결정 단위: 요청(request) 단위
    캐시 상태 접근: src/cache/radix.py RadixAttentionCache의 공유 프리픽스 조회
    스케줄링 오버헤드: 동질성 점수 계산 < 0.5ms (Radix 트리 순회 O(prefix_len))

    배치 중단 정책 (정적 모드):
      homogeneity_score = (공유 프리픽스 총 토큰) / (배치 총 입력 토큰)
      homogeneity_score >= threshold → STOP_BATCH
      배치 크기 >= target_batch_size → STOP_BATCH
      그 외 → ADD_REQUEST

    Activity A 평가 기준 (evaluation_criteria.md §2):
      - 스케줄링 오버헤드: TTFT p50 증가 +5% 이내
      - 캐시 히트율 향상: 스케줄링 미적용 대비 +10%p
      - 요청 공정성: 최대 대기 시간 2× 초과 금지
    """

    def __init__(
        self,
        config: RadixFeatherConfig,
        radix_cache: Optional[RadixAttentionCache] = None,
    ) -> None:
        self.config = config
        self._radix = radix_cache
        self._request_queue: List[Dict[str, Any]] = []
        self._scheduling_times: List[float] = []  # 오버헤드 측정

    def add_request(self, request: Dict[str, Any]) -> None:
        """요청 큐에 추가. request = {"id": str, "token_ids": List[int], "arrival_time": float}"""
        request.setdefault("arrival_time", time.monotonic())
        self._request_queue.append(request)

    def form_batch(self) -> List[Dict[str, Any]]:
        """동질성 신호 기반 배치 구성.

        Returns: 현재 배치에 포함될 요청 목록
        """
        t0 = time.monotonic()
        batch = self._form_batch_static()
        overhead_ms = (time.monotonic() - t0) * 1000
        self._scheduling_times.append(overhead_ms)
        return batch

    def _form_batch_static(self) -> List[Dict[str, Any]]:
        """정적 threshold 기반 배치 구성 (RL 미사용 모드)."""
        if not self._request_queue:
            return []
        batch: List[Dict[str, Any]] = []
        for req in list(self._request_queue):
            candidate_batch = batch + [req]
            score = self._homogeneity_score(candidate_batch)
            should_stop = (
                len(candidate_batch) >= self.config.target_batch_size
                or (len(batch) > 0 and score < self.config.homogeneity_threshold)
            )
            if should_stop and len(batch) > 0:
                break
            batch.append(req)
            self._request_queue.remove(req)
        return batch

    def _homogeneity_score(self, requests: List[Dict[str, Any]]) -> float:
        """동질성 점수 = 공유 프리픽스 총 토큰 / 배치 총 입력 토큰.

        Radix 트리가 없으면 토큰 집합 교집합으로 근사.
        """
        if not requests:
            return 0.0
        total_tokens = sum(len(r["token_ids"]) for r in requests)
        if total_tokens == 0:
            return 0.0
        if self._radix is not None:
            shared = self._count_shared_prefix_tokens_radix(requests)
        else:
            shared = self._count_shared_prefix_tokens_naive(requests)
        return shared / total_tokens

    def _count_shared_prefix_tokens_radix(
        self, requests: List[Dict[str, Any]]
    ) -> int:
        """Radix 트리 순회로 공유 프리픽스 토큰 수 계산."""
        if not requests:
            return 0
        # 최단 공유 프리픽스 = 모든 요청에서 Radix 트리 히트가 발생한 토큰 수
        min_hit = min(
            self._radix.prefix_match_length(r["token_ids"])
            for r in requests
        )
        return min_hit * len(requests)

    def _count_shared_prefix_tokens_naive(
        self, requests: List[Dict[str, Any]]
    ) -> int:
        """Radix 트리 없을 때 토큰 리스트 공통 프리픽스 길이로 근사."""
        if len(requests) < 2:
            return len(requests[0]["token_ids"]) if requests else 0
        ref = requests[0]["token_ids"]
        shared = 0
        for i, tok in enumerate(ref):
            if all(len(r["token_ids"]) > i and r["token_ids"][i] == tok
                   for r in requests[1:]):
                shared += 1
            else:
                break
        return shared * len(requests)

    def scheduling_overhead_ms_p50(self) -> float:
        """스케줄링 오버헤드 p50 (ms)."""
        if not self._scheduling_times:
            return 0.0
        sorted_times = sorted(self._scheduling_times)
        return sorted_times[len(sorted_times) // 2]

    def fairness_max_wait(self) -> float:
        """현재 대기 중인 요청의 최대 대기 시간 (초)."""
        if not self._request_queue:
            return 0.0
        now = time.monotonic()
        return max(now - r["arrival_time"] for r in self._request_queue)
```

---

### 오프라인 레이어 범위 프로파일러 (Activity B 보정)

```python
# experiments/run_relay_layer_calibration.py

"""
RelayUShapeLayerSelectiveSegmentCache용 레이어 범위 프로파일러.
100~200개 "유사하지만 비동일" 세그먼트 쌍을 사용해 레이어별 KV 코사인 유사도를 측정하고,
임계값 τ_layer(기본 0.95) 이상인 레이어를 R_reuse로 분류한다.

출력: configs/relay_ulayer_profile.yaml

의사코드:
  1. 세그먼트 쌍 생성: (base_segment, perturbed_segment) — 토큰 ID 일부를 치환
  2. 각 쌍에 대해 레이어별 KV를 합성 어텐션 계산으로 생성 (실제 모델 없이 synthetic)
  3. layer_similarity[l] = mean(cosine_sim(kv_base_l, kv_perturbed_l)) over all pairs
  4. reuse_layers = [l for l if layer_similarity[l] >= tau_layer]
  5. 결과를 yaml로 저장

Usage:
  python experiments/run_relay_layer_calibration.py \
    --n_pairs 100 --n_layers 12 --tau_layer 0.95 \
    --output configs/relay_ulayer_profile.yaml --seed 42
"""

def run_calibration(
    n_pairs: int = 100,
    n_layers: int = 12,
    n_heads: int = 8,
    d_head: int = 64,
    chunk_size: int = 128,
    tau_layer: float = 0.95,
    perturbation_ratio: float = 0.1,  # 세그먼트 토큰의 10% 치환
    output_path: str = "configs/relay_ulayer_profile.yaml",
    seed: int = 42,
) -> None:
    ...
```

---

### 룩어헤드 LoRA 훈련 스크립트 (Activity C)

```python
# experiments/train_lookahead_lora.py

"""
LookaheadKVEvictionCodec의 룩어헤드 토큰 + LoRA 어댑터 훈련 스크립트.
보정 데이터 500~1000 샘플, ≤1 GPU-hour.

훈련 목표:
  룩어헤드 어텐션 점수 → 미래 응답 어텐션 패턴과 정렬.
  손실: MSE(lookahead_score, future_attention_score)

Usage:
  python experiments/train_lookahead_lora.py \
    --n_samples 500 --n_epochs 5 --lr 1e-3 \
    --n_lookahead 5 --lora_rank 8 \
    --output configs/lookahead_lora_weights.pt --seed 42
"""

def train(
    n_samples: int = 500,
    n_epochs: int = 5,
    lr: float = 1e-3,
    n_layers: int = 12,
    n_heads: int = 8,
    d_head: int = 64,
    n_lookahead: int = 5,
    lora_rank: int = 8,
    output_path: str = "configs/lookahead_lora_weights.pt",
    seed: int = 42,
) -> Dict[str, float]:
    """Returns {"final_loss": float, "n_samples": int, "training_time_sec": float}"""
    ...
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(LookaheadKVEvictionCodec)를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-2 proxy (실 데이터셋 없을 경우 synthetic 토큰 시퀀스로 대체)
- **측정 방법**: `src/metrics/perplexity.py`의 `attention_output_relative_error(q, k_orig, v_orig, k_kept, v_kept)` < 0.01 (1%)
  - `k_kept`, `v_kept`: 퇴거 후 남은 토큰으로 구성된 K, V (패딩 없이 kept_tokens 기준 계산)
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)
- **퇴거 비율별 측정**: eviction_ratio = 0.5, 0.7, 0.85 각각 독립 측정

### 태스크 정확도 측정

- **벤치마크**: LongBench 8개 서브태스크 proxy (KL divergence, cosine similarity)
- **측정 방법**: `attention_kl_divergence(q, k_orig, k_kept)` < 0.015,
  `cosine_similarity_output(q, k_orig, v_orig, k_kept, v_kept)` >= 0.99
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)

### 퇴거 비율 — 정확도 곡선

| 퇴거 비율 | 목표 메모리 감소 | attention error 한계 | cosine 최소값 |
|----------|----------------|---------------------|--------------|
| 0.5 (50% 퇴거) | −50% | < 0.005 (0.5%) | >= 0.995 |
| 0.7 (70% 퇴거) | −70% | < 0.01  (1.0%) | >= 0.99  |
| 0.85 (85% 퇴거) | −85% | < 0.02  (2.0%) | >= 0.98 (경고, non-mandatory) |

### 기존 LaProx와의 동일 설정 비교

- LaProx(기구현, `src/cache/`)가 있으면 동일 eviction_ratio=0.7 설정에서 정확도 비교 포함.
- LookaheadKV cosine >= LaProx cosine - 0.01 조건을 테스트로 명시.

### 검증 테스트 파일

`tests/unit/test_lookahead_kv_accuracy.py`

**테스트 케이스 목록**:

```python
# tests/unit/test_lookahead_kv_accuracy.py

"""Activity C — LookaheadKVEvictionCodec accuracy-preserving verification.

Mandatory per evaluation_criteria.md §4:
  - perplexity change ±1% (proxied by attention output relative error < 0.01)
  - downstream task accuracy ±1% (KL < 0.015, cosine >= 0.99)
  - eviction_ratio = 0.5 / 0.7 / 0.85 각각 측정
All tests use synthetic data (no real model API calls).
Seed 42 고정으로 재현성 보장.
"""

import pytest
import torch
from src.cache.lookahead_kv_eviction import LookaheadKVEvictionCodec, LookaheadKVConfig
from src.metrics.perplexity import (
    attention_output_relative_error,
    attention_kl_divergence,
    cosine_similarity_output,
)

SEED = 42
D_HEAD = 64
N_HEADS = 4
N_LAYERS = 4
N_TOKENS = 64

@pytest.fixture
def codec_ratio_50() -> LookaheadKVEvictionCodec:
    """eviction_ratio=0.5 (50% 퇴거)."""
    cfg = LookaheadKVConfig(
        n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD,
        n_lookahead=5, lora_rank=8, eviction_ratio=0.5, seed=SEED,
    )
    return LookaheadKVEvictionCodec(cfg)

@pytest.fixture
def codec_ratio_70() -> LookaheadKVEvictionCodec:
    """eviction_ratio=0.7 (70% 퇴거) — 기본 설정."""
    cfg = LookaheadKVConfig(
        n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD,
        n_lookahead=5, lora_rank=8, eviction_ratio=0.7, seed=SEED,
    )
    return LookaheadKVEvictionCodec(cfg)

@pytest.fixture
def codec_ratio_85() -> LookaheadKVEvictionCodec:
    """eviction_ratio=0.85 (85% 퇴거)."""
    cfg = LookaheadKVConfig(
        n_layers=N_LAYERS, n_heads=N_HEADS, d_head=D_HEAD,
        n_lookahead=5, lora_rank=8, eviction_ratio=0.85, seed=SEED,
    )
    return LookaheadKVEvictionCodec(cfg)


def _make_kv_and_query(seed_offset: int = 0):
    torch.manual_seed(SEED + seed_offset)
    kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
    q = torch.randn(N_TOKENS, D_HEAD)
    return kv, q


def _apply_eviction(codec, kv):
    """compression_hook을 통해 퇴거 적용 후 K/V 반환."""
    kept_kv = codec.compression_hook("test_key", kv)
    k_kept = kept_kv[:, 0, 0, :]  # 첫 번째 헤드 기준
    v_kept = kept_kv[:, 1, 0, :]
    return k_kept, v_kept


# ------------------------------------------------------------------ #
# 1. eviction_ratio=0.5: attention error < 0.005 (MANDATORY ±1%)    #
# ------------------------------------------------------------------ #
def test_50pct_eviction_attention_error(codec_ratio_50):
    kv, q = _make_kv_and_query(0)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    k_kept, v_kept = _apply_eviction(codec_ratio_50, kv)
    # kept 토큰이 N_TOKENS * 0.5 이상이어야 함 (recent_window 포함)
    assert k_kept.shape[0] >= int(N_TOKENS * 0.14), "Too few tokens kept"
    err = attention_output_relative_error(q, k_orig, v_orig, k_kept, v_kept)
    assert err < 0.01, f"50% eviction attention error {err:.4f} exceeds 1% limit"


# ------------------------------------------------------------------ #
# 2. eviction_ratio=0.7: attention error < 0.01 (MANDATORY ±1%)     #
# ------------------------------------------------------------------ #
def test_70pct_eviction_attention_error(codec_ratio_70):
    kv, q = _make_kv_and_query(1)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    k_kept, v_kept = _apply_eviction(codec_ratio_70, kv)
    err = attention_output_relative_error(q, k_orig, v_orig, k_kept, v_kept)
    assert err < 0.01, f"70% eviction attention error {err:.4f} exceeds 1% limit"


# ------------------------------------------------------------------ #
# 3. eviction_ratio=0.85: error < 0.02, 경고 only (non-mandatory)   #
# ------------------------------------------------------------------ #
def test_85pct_eviction_attention_error(codec_ratio_85):
    kv, q = _make_kv_and_query(2)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    k_kept, v_kept = _apply_eviction(codec_ratio_85, kv)
    err = attention_output_relative_error(q, k_orig, v_orig, k_kept, v_kept)
    import warnings
    if err >= 0.01:
        warnings.warn(f"85% eviction error {err:.4f} exceeds 1% (expected for high ratio)")
    assert err < 0.05, f"85% eviction error {err:.4f} exceeds 5% hard limit"


# ------------------------------------------------------------------ #
# 4. KL divergence < 0.015 at 70% (LongBench proxy, MANDATORY)     #
# ------------------------------------------------------------------ #
def test_70pct_kl_divergence(codec_ratio_70):
    kv, q = _make_kv_and_query(3)
    k_orig = kv[:, 0, 0, :]
    k_kept, _ = _apply_eviction(codec_ratio_70, kv)
    kl = attention_kl_divergence(q, k_orig, k_kept)
    assert kl < 0.015, f"70% eviction KL divergence {kl:.6f} >= 0.015"


# ------------------------------------------------------------------ #
# 5. Cosine similarity >= 0.99 at 70% (MANDATORY)                   #
# ------------------------------------------------------------------ #
def test_70pct_cosine_similarity(codec_ratio_70):
    kv, q = _make_kv_and_query(4)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
    k_kept, v_kept = _apply_eviction(codec_ratio_70, kv)
    cos = cosine_similarity_output(q, k_orig, v_orig, k_kept, v_kept)
    assert cos >= 0.99, f"70% eviction cosine {cos:.4f} < 0.99"


# ------------------------------------------------------------------ #
# 6. Recent window 토큰 보존 검증                                     #
# ------------------------------------------------------------------ #
def test_recent_window_preserved(codec_ratio_70):
    """최근 recent_window 토큰은 항상 보존되어야 한다."""
    kv, _ = _make_kv_and_query(5)
    kept_kv = codec_ratio_70.compression_hook("test_key", kv)
    rw = codec_ratio_70.config.recent_window
    if rw > 0:
        # 마지막 rw 토큰이 항상 포함되어 있는지 확인
        assert kept_kv.shape[0] >= rw, f"Recent {rw} tokens not preserved"


# ------------------------------------------------------------------ #
# 7. eviction_rate() 정확성                                           #
# ------------------------------------------------------------------ #
def test_eviction_rate_matches_ratio(codec_ratio_70):
    """실제 퇴거율이 eviction_ratio와 일치해야 한다 (±5%p)."""
    kv, _ = _make_kv_and_query(6)
    codec_ratio_70.put("key1", kv)
    actual_rate = codec_ratio_70.eviction_rate()
    expected = codec_ratio_70.config.eviction_ratio
    assert abs(actual_rate - expected) <= 0.1, (
        f"eviction_rate {actual_rate:.3f} differs from target {expected:.3f} by >10%p"
    )


# ------------------------------------------------------------------ #
# 8. memory_reduction_ratio() >= 0.30 at 70%                        #
# ------------------------------------------------------------------ #
def test_memory_reduction_30pct(codec_ratio_70):
    """메모리 감소율 >= 30% (evaluation_criteria.md §4)."""
    kv, _ = _make_kv_and_query(7)
    codec_ratio_70.put("key_mem", kv)
    reduction = codec_ratio_70.memory_reduction_ratio()
    assert reduction >= 0.30, f"Memory reduction {reduction:.3f} < 30%"


# ------------------------------------------------------------------ #
# 9. CacheStore 인터페이스 준수                                        #
# ------------------------------------------------------------------ #
def test_cachestore_interface(codec_ratio_70):
    """put/get/evict/hit_rate/memory_bytes/reset_stats 동작 검증."""
    kv, _ = _make_kv_and_query(8)
    codec_ratio_70.put("if_key", kv)
    result = codec_ratio_70.get("if_key")
    assert result is not None, "get after put must return tensor"
    assert codec_ratio_70.hit_rate() > 0.0
    assert codec_ratio_70.memory_bytes() > 0
    freed = codec_ratio_70.evict()
    assert freed > 0
    codec_ratio_70.reset_stats()
    assert codec_ratio_70.hit_rate() == 0.0


# ------------------------------------------------------------------ #
# 10. 다중 레이어 일관성                                               #
# ------------------------------------------------------------------ #
def test_multilayer_consistency(codec_ratio_70):
    """N_LAYERS 레이어 각각에서 attention error < 1%."""
    for layer_idx in range(N_LAYERS):
        torch.manual_seed(SEED + 100 + layer_idx)
        kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
        q = torch.randn(N_TOKENS, D_HEAD)
        k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]
        key = f"layer{layer_idx}:test"
        kept_kv = codec_ratio_70.compression_hook(key, kv)
        k_kept, v_kept = kept_kv[:, 0, 0, :], kept_kv[:, 1, 0, :]
        err = attention_output_relative_error(q, k_orig, v_orig, k_kept, v_kept)
        assert err < 0.01, f"Layer {layer_idx}: error {err:.4f} exceeds 1%"
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-15.yaml
experiment:
  date: "2026-05-15"
  activity: "A+B+C"
  description: >
    C-1 LookaheadKVEvictionCodec (미래-인식 드래프트-프리 LoRA KV 퇴거) +
    B-1 RelayUShapeLayerSelectiveSegmentCache (U자형 레이어-선택적 비연속 세그먼트 재사용) +
    Cross-1 LookaheadRelaySegmentCache (이중-필터 B+C 통합) +
    A-1 RadixFeatherBatchScheduler (보조, Radix 트리 동질성 신호 기반 배치 중단)

lookahead_kv_eviction:
  n_layers: 12
  n_heads: 8
  d_head: 64
  n_lookahead: 5             # 룩어헤드 토큰 수 (n_la)
  lora_rank: 8               # LoRA 어댑터 랭크
  eviction_ratio: 0.7        # 기본 퇴거 비율 (sweep: 0.5/0.7/0.85)
  blend_ratio: 0.0           # LaProx 앙상블 비율 (0.0=순수 LookaheadKV)
  recent_window: 4           # 최근 N 토큰 항상 보존
  max_entries: 1000
  seed: 42

relay_ulayer_segment:
  chunk_size: 128
  max_entries: 1000
  n_layers: 12
  n_heads: 8
  d_head: 64
  similarity_threshold: 0.95   # τ_layer
  profile_path: "configs/relay_ulayer_profile.yaml"
  seed: 42

lookahead_relay_segment:
  token_importance_threshold: 0.3   # τ_token
  max_entries: 1000
  seed: 42

radix_feather_scheduler:
  chunk_size: 256
  target_batch_size: 8
  homogeneity_threshold: 0.6
  scheduler_mode: "static"   # 초기 구현; "rl" 선택적
  max_wait_ratio: 2.0
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    proxy_tolerance: 0.01           # 1% attention output error limit
    kl_tolerance: 0.015
    cosine_min: 0.99
    perplexity_dataset: "wikitext-2"
    perplexity_tolerance_pct: 1.0
    task_accuracy_tolerance_pct: 1.0
  eviction_sweep:
    ratios: [0.5, 0.7, 0.85]        # 퇴거 비율 sweep
  layer_calibration:
    n_pairs: 100                    # 보정 세그먼트 쌍 수
    perturbation_ratio: 0.1         # 세그먼트 토큰 치환 비율
    tau_layer: 0.95                 # 레이어 재사용 임계값
  hit_rate:
    target_noncontiguous_fraction: 0.30   # 비연속 히트율 목표 >= 30%
  memory_reduction:
    target_ratio: 0.30              # 최소: −30%
    target_ratio_goal: 0.70         # 목표: −70% (70% 퇴거)
  throughput:
    target_improvement_pct: 20      # +20% 이상
  effective_context:
    target_multiplier: 2.0          # 2× 이상 (70% 퇴거 시 ~3×)
  scheduling:
    ttft_overhead_limit_pct: 5.0    # TTFT p50 +5% 이내

lookahead_training:
  n_samples: 500
  n_epochs: 5
  lr: 1.0e-3
  output: "configs/lookahead_lora_weights.pt"

layer_calibration_output: "configs/relay_ulayer_profile.yaml"
seed: 42
results_dir: "results/2026-05-15"
```

---

## 테스트 요구사항

- [x] `tests/unit/test_lookahead_kv_accuracy.py` — Activity C 필수 accuracy 검증 (10개 테스트, 위 코드 참조)
- [ ] `tests/unit/test_relay_ulayer_segment.py` — Activity B 레이어 마스크·비연속 히트율·메모리 단위 테스트
- [ ] `tests/unit/test_lookahead_relay_segment.py` — Cross B+C 이중 필터 파이프라인 단위 테스트
- [ ] `tests/unit/test_radix_feather_scheduler.py` — Activity A 동질성 점수·스케줄링 오버헤드 단위 테스트
- [ ] `tests/integration/test_cross_bc_lookahead_relay.py` — E2E 통합: 다중 요청 이중 필터 재사용 흐름

### 단위 테스트 최소 요구 사항 (test_relay_ulayer_segment.py)

```
- test_put_get_full_layer: put/get 라운드트립 정확성
- test_layer_mask_encoding_decoding: 비트마스크 인코딩/디코딩 정확성
- test_default_middle_layer_coverage: 기본 중간 레이어 범위 70% 이상
- test_noncontiguous_hit_rate_target: 비연속 히트율 >= 30% (동일 메모리)
- test_partial_reuse_tracking: _partial_reuse_hits 카운터 정확성
- test_layer_selective_get: get_with_layer_selection 반환 형식 검증
- test_profile_from_yaml: LayerReuseProfile.from_yaml 로드 검증
- test_lru_eviction: max_entries 초과 시 LRU 퇴거 동작
- test_cachestore_interface: CacheStore 6개 추상 메서드 완전 동작
```

### 단위 테스트 최소 요구 사항 (test_lookahead_relay_segment.py)

```
- test_dual_filter_layer_then_token: 레이어 필터 → 토큰 필터 순서 확인
- test_token_threshold_respected: τ_token=0.3 이하 토큰 퇴거 확인
- test_memory_below_baseline: 이중 필터 후 메모리 < 원본 30%
- test_accuracy_within_1pct: 이중 필터 후 attention error < 0.01
- test_noncontiguous_hit_tracking: 비연속 히트 카운터 정확성
- test_cachestore_interface: CacheStore 6개 추상 메서드 완전 동작
```

### 단위 테스트 최소 요구 사항 (test_radix_feather_scheduler.py)

```
- test_homogeneity_score_identical_prefix: 완전 동일 프리픽스 → score=1.0
- test_homogeneity_score_no_prefix: 공유 프리픽스 없음 → score=0.0
- test_batch_stops_at_target_size: target_batch_size 초과 시 배치 중단
- test_batch_stops_below_threshold: homogeneity_score < threshold 시 중단
- test_scheduling_overhead_under_5pct: scheduling_overhead_ms_p50 < 5ms
- test_fairness_max_wait: 최대 대기 요청이 2× 초과하지 않음
```

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 5개 파일 + 기존 회귀 없음)
- [ ] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - perplexity 변화 ±1% 이내 (attention error < 0.01, eviction_ratio 0.5/0.7 각각)
      - downstream 태스크 정확도 ±1% 이내 (KL < 0.015, cosine >= 0.99)
- [ ] `evaluation_criteria.md` §3 Activity B 항목 충족:
      - 비연속 세그먼트 히트율 >= 30%
      - KV Memory Footprint: 이중 필터로 베이스라인 대비 대폭 감소
- [ ] `evaluation_criteria.md` §2 Activity A 항목 충족 (보조):
      - 스케줄링 오버헤드 TTFT p50 +5% 이내
      - 캐시 히트율 향상 +10%p (RadixFeatherBatchScheduler 동질 배치 효과)
- [ ] `evaluation_criteria.md` §5 크로스 조합 C 포함: 복합 적용 후 accuracy ±1% 이내
- [ ] `evaluation_criteria.md` §0 공통 필수:
      - CacheStore 인터페이스 모든 추상 메서드 구현
      - 시드 42 고정 재현성
      - `configs/experiments/2026-05-15.yaml` 존재
      - 모든 공개 함수·메서드 타입 힌트
- [ ] 목표 지표 수치 `results/2026-05-15/metrics.json`에 JSON 기록:
      ```json
      {
        "inference_throughput_improvement_pct": ...,
        "kv_memory_reduction_ratio_50pct_eviction": ...,
        "kv_memory_reduction_ratio_70pct_eviction": ...,
        "kv_memory_reduction_ratio_85pct_eviction": ...,
        "noncontiguous_hit_rate": ...,
        "compression_accuracy_delta_50pct": ...,
        "compression_accuracy_delta_70pct": ...,
        "compression_accuracy_delta_85pct": ...,
        "effective_context_length_multiplier": ...,
        "eviction_overhead_ttft_pct": ...,
        "scheduling_overhead_ttft_p50_pct": ...,
        "layer_reuse_rate": ...,
        "token_filter_retention_rate": ...
      }
      ```
- [ ] `src/cache/base.py` CacheStore 인터페이스 깨지지 않음 (수정 없음)
- [ ] 기존 모든 단위·통합 테스트 회귀 없이 통과
- [ ] `configs/relay_ulayer_profile.yaml` 생성 (`experiments/run_relay_layer_calibration.py` 실행 후)
