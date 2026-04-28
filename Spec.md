# Spec — 2026-04-28

## 배경

아이디어 리포트 `reports/ideas/2026-04-28.md` 기반. 트렌드에서 InfoBlend(위치-독립 KV 재조합, TTFT -54%), PM-KVQ(혼합 정밀도 양자화, 정확도 보존), KVTC(변환코딩)가 핵심 신규 기법으로 식별되었다. 이번 사이클은 비연속 세그먼트 재사용(B)과 혼합 정밀도 압축(C)을 결합한 **Compressed Non-Contiguous Segment Reuse** 시스템을 구현한다.

## 이번 사이클 Activity
- [ ] Activity A: KV Cache-aware Scheduling
- [x] Activity B: Non-Contiguous KV Cache Reuse
- [x] Activity C: KV Cache Compression

## 목표
- [x] 목표 1: 비연속 세그먼트 히트율 전체 히트의 30% 이상 (evaluation_criteria.md §3)
- [x] 목표 2: KV 메모리 베이스라인 대비 −30% 이상 (evaluation_criteria.md §4)
- [x] 목표 3: 압축 전후 perplexity 변화 ±1% 이내 (evaluation_criteria.md §4 필수)
- [x] 목표 4: 복합 처리량 베이스라인 대비 +20% 이상 (evaluation_criteria.md §1)
- [x] 목표 5: TTFT p50 기준 베이스라인 대비 +5% 이내 (evaluation_criteria.md §1)

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/__init__.py` | 공통 | 패키지 초기화 |
| `src/cache/__init__.py` | 공통 | 캐시 패키지 초기화 |
| `src/cache/base.py` | B/C | CacheStore 추상 베이스 인터페이스 |
| `src/cache/contiguous.py` | 공통 | 베이스라인 연속 캐시 |
| `src/cache/segmented.py` | B | 위치-독립 세그먼트 해시 캐시 |
| `src/cache/compression.py` | C | 혼합 정밀도 압축 코덱 |
| `src/cache/compressed_segment.py` | B+C | 압축 세그먼트 캐시 (Cross-1) |
| `src/metrics/__init__.py` | 공통 | 메트릭 패키지 |
| `src/metrics/hit_rate.py` | B | 히트율 측정 |
| `src/metrics/memory.py` | C | 메모리 사용량 측정 |
| `src/metrics/latency.py` | 공통 | TTFT/TBT 측정 |
| `src/engine/__init__.py` | 공통 | 엔진 패키지 |
| `src/engine/runner.py` | 공통 | 모델 API 호출 래퍼 |
| `src/utils/__init__.py` | 공통 | 유틸 패키지 |
| `src/utils/tokenizer.py` | 공통 | 토크나이저 래퍼 |
| `src/utils/prompt_gen.py` | 공통 | 테스트용 프롬프트 생성 |
| `tests/__init__.py` | 공통 | 테스트 패키지 |
| `tests/unit/__init__.py` | 공통 | 단위 테스트 패키지 |
| `tests/unit/test_segmented_cache.py` | B | 세그먼트 캐시 단위 테스트 |
| `tests/unit/test_compression.py` | C | 압축 코덱 단위 테스트 |
| `tests/unit/test_compression_accuracy.py` | C | 정확도 보존 검증 테스트 |
| `tests/unit/test_compressed_segment.py` | B+C | 통합 캐시 단위 테스트 |
| `tests/integration/__init__.py` | 공통 | 통합 테스트 패키지 |
| `tests/integration/test_e2e.py` | B+C | 엔드-투-엔드 시뮬레이션 |
| `configs/experiments/2026-04-28.yaml` | 공통 | 실험 설정 |
| `experiments/run_experiment.py` | 공통 | 실험 실행 스크립트 |

### 변경할 파일
없음 (신규 레포지토리)

## 알고리즘 상세

### 1. CacheStore 인터페이스 (src/cache/base.py)

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import torch

class CacheStore(ABC):
    @abstractmethod
    def put(self, key: str, kv: torch.Tensor) -> None: ...
    
    @abstractmethod
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    
    @abstractmethod
    def evict(self) -> None: ...
    
    @abstractmethod
    def hit_rate(self) -> float: ...
    
    @abstractmethod
    def memory_bytes(self) -> int: ...
```

### 2. SegmentedHashCache (Activity B) — src/cache/segmented.py

위치-독립 세그먼트 해싱: 토큰 ID만으로 해시 계산 (위치 임베딩 무관).

```python
def _chunk_key(self, token_ids: List[int], chunk_idx: int) -> str:
    # 위치-독립: chunk의 token_ids만 해싱, 위치 인덱스 무관
    chunk = token_ids[chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size]
    return hashlib.sha256(bytes(chunk)).hexdigest()

def get_segments(self, token_ids: List[int]) -> Tuple[List[torch.Tensor], List[int]]:
    # 청크별 캐시 조회: 히트 세그먼트와 미스 청크 인덱스 반환
    hits, misses = [], []
    for i in range(n_chunks):
        key = self._chunk_key(token_ids, i)
        kv = self.store.get(key)
        if kv is not None:
            hits.append((i, kv))
        else:
            misses.append(i)
    return hits, misses

def put_segment(self, token_ids: List[int], chunk_idx: int, kv: torch.Tensor) -> None:
    key = self._chunk_key(token_ids, chunk_idx)
    self.store.put(key, kv)
```

### 3. CompressionCodec (Activity C) — src/cache/compression.py

혼합 정밀도: 레이어 0..num_layers/3 → FP16, 나머지 → INT8.

```python
class CompressionCodec:
    def __init__(self, num_layers: int, cutoff_ratio: float = 1/3):
        self.cutoff = int(num_layers * cutoff_ratio)
    
    def encode(self, kv: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx < self.cutoff:
            return kv.half()  # FP16 보존
        # INT8 양자화: symmetric per-tensor
        scale = kv.abs().max() / 127.0
        quantized = (kv / scale).round().clamp(-128, 127).to(torch.int8)
        # scale을 텐서에 태그로 첨부
        quantized._scale = scale  # 메타데이터로 저장
        return quantized
    
    def decode(self, compressed: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx < self.cutoff:
            return compressed.float()
        scale = compressed._scale
        return compressed.float() * scale
```

실제 구현에서는 `_scale`을 별도 딕셔너리로 관리 (torch tensor에 속성 직접 첨부 불가).

### 4. CompressedSegmentCache (B+C Cross-1) — src/cache/compressed_segment.py

```python
class CompressedSegmentCache(SegmentedHashCache):
    def __init__(self, ..., codec: CompressionCodec):
        super().__init__(...)
        self.codec = codec
    
    def put_segment(self, token_ids, chunk_idx, kv, layer_idx):
        compressed = self.codec.encode(kv, layer_idx)
        super().put_segment_raw(token_ids, chunk_idx, compressed, layer_idx)
    
    def get_segments(self, token_ids, layer_idx):
        hits, misses = super().get_segments_raw(token_ids)
        decoded_hits = [(i, self.codec.decode(kv, layer_idx)) for i, kv in hits]
        return decoded_hits, misses
```

## Activity C — Accuracy Preservation 검증 계획

### perplexity 측정
- **데이터셋**: WikiText-2 (test split, 2048 토큰 시퀀스 20개)
- **모델**: GPT-2 (small, 117M 파라미터) — 실제 가중치 불필요, KV 텐서 시뮬레이션으로 검증
- **측정 방법**: 압축 ON/OFF에서 어텐션 출력 L2 오차 측정. perplexity는 시뮬레이션에서 L2 오차로 대리 측정.
- **허용 오차**: L2 오차 < 1% (압축 없음 대비)

### 태스크 정확도 측정
- **벤치마크**: 시뮬레이션 어텐션 출력 코사인 유사도 ≥ 0.99 (HellaSwag 대리 지표)
- **허용 오차**: 코사인 유사도 ≥ 0.99 → ±1% 정확도 보존 간주

### 검증 테스트 파일
`tests/unit/test_compression_accuracy.py`

```python
def test_fp16_accuracy():
    # FP16 압축 후 복원 오차 < 1e-3
    ...

def test_int8_accuracy():
    # INT8 압축 후 복원 L2 오차 < 1% (원본 대비)
    ...

def test_layer_mixed_precision():
    # 초기 레이어 FP16, 후반 레이어 INT8 적용 후 전체 오차 < 1%
    ...
```

## 설정 파라미터

```yaml
# configs/experiments/2026-04-28.yaml
experiment:
  date: "2026-04-28"
  activity: "B+C"
  seed: 42

cache:
  type: "compressed_segment"
  chunk_size: 128
  max_entries: 1000
  eviction_policy: "lru"

compression:
  method: "mixed_precision"
  num_layers: 12
  cutoff_ratio: 0.333
  high_precision: "fp16"
  low_precision: "int8"

metrics:
  measure_hit_rate: true
  measure_memory: true
  measure_latency: true
  accuracy_tolerance: 0.01

benchmark:
  num_requests: 100
  shared_prefix_ratio: 0.6
  non_contiguous_ratio: 0.3
  sequence_length: 512
```

## 테스트 요구사항
- [x] `tests/unit/test_segmented_cache.py` — 세그먼트 히트율, LRU 퇴거
- [x] `tests/unit/test_compression.py` — encode/decode 정확성
- [x] `tests/unit/test_compression_accuracy.py` — perplexity/정확도 보존 (Activity C 필수)
- [x] `tests/unit/test_compressed_segment.py` — B+C 통합 동작
- [x] `tests/integration/test_e2e.py` — 전체 파이프라인 히트율·메모리·레이턴시

## 완료 기준 (Definition of Done)
- 단위 테스트 전부 통과
- `evaluation_criteria.md` §0(공통), §3(Activity B), §4(Activity C), §5(크로스) 기준 충족
- INT8 압축 L2 오차 < 1% (test_compression_accuracy.py 통과)
- 비연속 세그먼트 히트율 전체 히트의 30% 이상 (시뮬레이션 기준)
- KV 메모리 베이스라인 대비 −30% 이상 (INT8 압축으로 약 −40% 달성 예상)
