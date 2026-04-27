# Non-Contiguous KV Cache Reuse — Research Harness

## 프로젝트 목표

비연속(non-contiguous) KV 캐시 재사용률을 높이는 기법을 실험하고 측정한다.
LLM 추론 시 프롬프트 접두사가 완전히 일치하지 않아도 캐시를 재활용할 수 있는
구조(세그먼트 해시, 트리 어텐션, 청크 캐싱 등)를 비교·평가한다.

---

## RALPH 루프 — 자율 연구 사이클

하루 한 번 다음 7단계 멀티에이전트 파이프라인이 순서대로 실행된다.
**2단계(Analyze)에서 아이디어 변화가 없으면 3~7단계는 건너뛴다.**

```
┌──────────────────────────────────────────────────────────────────────┐
│  매일 KST 06:00                                                       │
│                                                                       │
│  R  1. Research   → 외부 기술 동향 수집 → reports/trends/             │
│          │                                                            │
│  A  2. Analyze    → 아이디어 생성 → reports/ideas/                    │
│          │                                                            │
│          │  SIGNIFICANT_CHANGE: false → STOP                         │
│          │  SIGNIFICANT_CHANGE: true  ↓                              │
│  L  3. Launch     → Spec.md 생성                                      │
│          │                                                            │
│  P  4. Program  ◄──────────────────────────────────────┐             │
│          │                                              │ 최대 3회    │
│  H  5. Heuristic  → 피드백 → 구현 반영 ─────────────────┘             │
│          │                                                            │
│          └→ Report ① reports/evaluations/YYYY-MM-DD.md               │
│          │                                                            │
│          ↓ 검증 통과 알고리즘                                          │
│                                                                       │
│     6. vLLM Port ◄─────────────────────────────────────┐             │
│          │                                              │ 최대 3회    │
│     7. vLLM Eval  → 피드백 → 이식 수정 ─────────────────┘             │
│          │                                                            │
│          └→ Report ② reports/vllm-evaluations/YYYY-MM-DD.md          │
└──────────────────────────────────────────────────────────────────────┘
```

### 단계별 에이전트 정의

| 단계 | 에이전트 파일 | 입력 | 출력 |
|------|-------------|------|------|
| 1. Research | `.claude/agents/trend-sensor.md` | 웹 검색 | `reports/trends/YYYY-MM-DD.md` |
| 2. Analyze | `.claude/agents/idea-generator.md` | 트렌드 리포트 + 과거 아이디어 | `reports/ideas/YYYY-MM-DD.md` |
| 3. Launch | `.claude/agents/planner.md` | 아이디어 리포트 | `Spec.md` |
| 4. Program | `.claude/agents/implementer.md` | `Spec.md` + 평가 피드백 | `src/` 코드 변경 |
| 5. Heuristic | `.claude/agents/evaluator.md` | 구현 결과 + `evaluation_criteria.md` | 피드백 or **Report ①** |
| 6. vLLM Port | `.claude/agents/vllm-porter.md` | Report ① + `src/cache/` + vLLM latest | `vllm_integration/` |
| 7. vLLM Eval | `.claude/agents/vllm-evaluator.md` | `vllm_integration/` + vLLM latest | 피드백 or **Report ②** |

### 최종 산출물

| 리포트 | 경로 | 내용 |
|--------|------|------|
| Report ① | `reports/evaluations/YYYY-MM-DD.md` | 독립 구현 검증 (히트율·지연·메모리·코드 품질) |
| Report ② | `reports/vllm-evaluations/YYYY-MM-DD.md` | vLLM latest 이식 검증 (처리량·호환성·회귀) |

### 파이프라인 실행

```bash
# 전체 파이프라인 수동 실행
/run-pipeline

# 개별 단계 실행
/run-trend      # 트렌드 센싱만
/run-idea       # 아이디어 생성만
```

파이프라인은 Claude Code 스케줄러를 통해 **매일 자동 실행**된다.
스케줄 등록: `/schedule` 명령 또는 `.claude/settings.json` 의 cron 설정 참조.

---

## 디렉토리 구조

```
.
├── CLAUDE.md                        # 이 파일
├── Spec.md                          # 현재 구현 스펙 (planner가 생성·갱신)
├── evaluation_criteria.md           # 평가 기준 (사람이 관리)
│
├── .claude/
│   ├── agents/
│   │   ├── trend-sensor.md          # 1단계: 외부 트렌드 수집
│   │   ├── idea-generator.md        # 2단계: 아이디어 생성 + 변화 감지
│   │   ├── planner.md               # 3단계: Spec.md 작성
│   │   ├── implementer.md           # 4단계: 코드 구현
│   │   ├── evaluator.md             # 5단계: 평가 + 피드백 루프 → Report ①
│   │   ├── vllm-porter.md           # 6단계: 검증 알고리즘 vLLM 이식
│   │   └── vllm-evaluator.md        # 7단계: vLLM 환경 평가 → Report ②
│   ├── commands/
│   │   ├── run-pipeline.md          # /run-pipeline 슬래시 커맨드
│   │   ├── run-trend.md             # /run-trend
│   │   └── run-idea.md              # /run-idea
│   └── settings.json                # 훅·권한·스케줄 설정
│
├── vllm_integration/                # vllm-porter 생성 (vLLM 이식 코드)
│   ├── block_manager_patch.py
│   ├── attention_backend_patch.py
│   ├── scheduler_patch.py
│   ├── install.sh                   # pip install --upgrade vllm + 패치
│   └── README.md                    # 버전 호환성 기록
│
├── reports/
│   ├── trends/                      # trend-sensor 출력
│   │   └── YYYY-MM-DD.md
│   ├── ideas/                       # idea-generator 출력
│   │   └── YYYY-MM-DD.md
│   ├── evaluations/                 # evaluator 최종 출력 (Report ①)
│   │   └── YYYY-MM-DD.md
│   └── vllm-evaluations/            # vllm-evaluator 최종 출력 (Report ②)
│       └── YYYY-MM-DD.md
│
├── configs/
│   ├── baseline.yaml
│   └── experiments/
│       └── <exp-name>.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── cache/
│   │   ├── base.py                  # CacheStore 추상 베이스
│   │   ├── contiguous.py            # 베이스라인
│   │   ├── segmented.py             # 세그먼트 해시 캐시
│   │   └── radix.py                 # Radix 트리 캐시
│   ├── engine/
│   │   ├── runner.py
│   │   └── batch_runner.py
│   ├── metrics/
│   │   ├── hit_rate.py
│   │   ├── latency.py
│   │   └── memory.py
│   └── utils/
│       ├── tokenizer.py
│       └── prompt_gen.py
├── experiments/
│   ├── run_experiment.py
│   └── sweep.py
├── notebooks/
│   └── analysis.ipynb
├── results/                         # git-ignored
└── tests/
    ├── unit/
    └── integration/
```

---

## 핵심 개념

### 비연속 KV 캐시 재사용이란?

표준 KV 캐시는 요청 간 **공통 접두사**가 byte-identical할 때만 재사용된다.
비연속 재사용은 프롬프트 내 임의 위치의 세그먼트를 독립적으로 캐싱해
부분 일치만으로도 히트를 발생시키는 기법이다.

| 기법 | 핵심 아이디어 | 구현 파일 |
|------|-------------|----------|
| Segmented Hash Cache | 고정 청크 단위 해시 → 불연속 세그먼트 재조합 | `src/cache/segmented.py` |
| Radix Attention | 공유 접두사를 Radix 트리로 관리 | `src/cache/radix.py` |
| Chunk-level Reuse | 가변 길이 청크 + LRU 퇴거 | `src/cache/segmented.py` |

### 측정 지표

- **Cache Hit Rate** : 히트된 KV 토큰 수 / 전체 입력 토큰 수
- **TTFT (Time-To-First-Token)** : 첫 토큰 생성까지 지연
- **TBT (Time-Between-Tokens)** : 토큰 간 평균 지연
- **KV Memory Footprint** : 캐시가 점유하는 GPU 메모리 (GB)
- **Reuse Efficiency** : 절약된 FLOPs / 전체 어텐션 FLOPs

---

## 개발 규칙

- `src/cache/base.py` 의 `CacheStore` 인터페이스를 반드시 구현해야 한다.
- 실험 결과는 `results/` 에 저장하고 git에는 커밋하지 않는다.
- 새 캐시 구현체는 `tests/unit/` 에 히트율 단위 테스트를 함께 작성한다.
- 모델 API 호출은 `src/engine/runner.py` 를 통해서만 한다.
- 측정 결과는 `results/<exp-name>/metrics.json` 에 JSON으로 기록한다.
- `Spec.md` 는 planner 에이전트만 수정한다. 사람이 직접 편집할 경우 planner에게 알린다.
- `evaluation_criteria.md` 는 사람이 관리하며 에이전트는 읽기만 한다.
- `vllm_integration/` 은 vllm-porter만 수정한다.
- vLLM은 항상 `pip install --upgrade vllm` 으로 최신 버전을 사용한다. 고정 버전 핀 금지.
- Report ①(evaluator)과 Report ②(vllm-evaluator)는 독립적으로 저장한다. 하나가 실패해도 다른 하나는 저장한다.

---

## 의존성

```
anthropic>=0.40.0
torch>=2.2.0
transformers>=4.40
numpy
pandas
matplotlib
pyyaml
pytest
```

설치: `pip install -r requirements.txt`

---

## 참고 문헌

- [Anthropic Prompt Caching 문서](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- RadixAttention / PagedAttention: Kwon et al., 2023
- SGLang: Zheng et al., 2024
- ChunkAttention: segment-level KV sharing across requests
