# /deep-dive

특정 기술 하나를 심층 분석한 리포트를 생성한다.

## 사용법

```
/deep-dive <기술 이름>
/deep-dive <기술 이름> <URL 또는 arXiv ID>
```

예시:
- `/deep-dive RadixAttention`
- `/deep-dive H2O`
- `/deep-dive Mooncake`
- `/deep-dive CacheBlend https://arxiv.org/abs/2405.16444`
- `/deep-dive MLA Multi-head Latent Attention`

## 동작

`deep-dive-analyzer` 에이전트를 호출한다. 인자(`$ARGUMENTS`)를 그대로 TECH_NAME으로 전달한다.

인자가 비어 있으면 사용자에게 어떤 기술을 분석할지 되묻고 종료한다.

## 출력

`reports/deep-dive/YYYY-MM-DD-<slug>.md`

리포트는 다음을 포함한다.
- 1차 출처 식별 (논문 / 공식 구현)
- 알고리즘 및 아키텍처 상세
- 복잡도 분석
- 보고된 실험 결과 + 본 연구 목표 지표와의 매핑
- 관련/경쟁 기술 비교
- 본 연구(`src/cache/`, `vllm_integration/`)로의 적용 가능성
- vLLM 이식 관점
- 열린 질문 및 후속 실험 제안

완료 시 다음 형식의 마커를 출력한다.

```
DEEP_DIVE_REPORT_SAVED: reports/deep-dive/YYYY-MM-DD-<slug>.md
```

## 주의

- 이 커맨드는 daily 파이프라인(`/run-pipeline`)과 독립적이다. 트렌드/아이디어/스펙/평가 단계를 거치지 않는다.
- 같은 날 같은 기술을 다시 분석하면 `-v2`, `-v3` 접미사가 붙는다. 과거 리포트가 있으면 이를 갱신·확장하는 형태로 작성된다.
