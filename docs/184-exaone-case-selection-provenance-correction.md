# EXAONE case-selection provenance correction

> 작성일: 2026-08-15
>
> 상태: **첫 EXAONE retrieval candidate 실행 전 발견·고정한 claim correction**

## 무엇이 잘못 서술되었나

`exaone-retrieval-data-v1`은 case rank key를 만들 때 compatibility result artifact SHA-256과
`result_summary_sha256`를 사용한다. Compatibility summary는 synthetic deterministic/forced generation의
model-output hashes를 포함한다. 따라서 data contract의 `historical_model_output=false`와 “model-output을
입력으로 사용하지 않은 case selection”이라는 문장은 literal하게 거짓이다.

정확한 causal chain은 다음과 같다.

1. EXAONE compatibility pass가 commit `fe37bf5a00286b7f8b0c3ef620f4057440652c04`에서 먼저 공개됐다.
2. 그 결과 artifact
   `results/large-model-retrieval-preflight-v4/summary.json`의 SHA-256은
   `670fed1737cd439413c162c3611075206c04f54a29e84e59892bd209d8f975af`다.
3. Data protocol은 약 42분 뒤 commit `cbaa7b9a307e8ddbdbb4dfd8e01233e7a545bb43`에서 작성됐다.
4. Data plan과 seal의 artifact SHA-256은 각각
   `a2ef4594a289a32a6fa8bbdb3181c6f08560734e15140470ff0d8097ffcd8f97`,
   `6a04814416edc58ca06ef717a9e9d3ef5457b9dd058518c14e420387783805dc`다.

따라서 compatibility output을 알기 전에 case rank seed가 고정됐다는 시간적 주장도 할 수 없다.

## 무엇은 여전히 관찰되지 않았나

이 교정을 발견한 시점에는 sealed 72 cases에 대해 다음 값이 한 번도 실행·관찰되지 않았다.

- 실제 200,000-entry Korean retrieval table을 사용한 candidate output
- corpus/prompt proposal coverage와 acceptance
- candidate latency와 baseline 대비 latency ratio
- actual primary gate 결과

Baseline-only resource calibration의 model output과 latency는 알려져 있다. Evaluation documents도 과거 품질
평가에 사용한 pool이다. 그러므로 이번 실험은 case-selection-blind confirmation이 아니라, candidate actual
timing을 열기 전에 비교·통계·gate를 고정한 exploratory scale-transfer test다.

## 왜 case를 다시 고르지 않는가

문제를 발견한 뒤 다른 seed나 pool로 case를 다시 고르면, candidate 결과를 보지 않았더라도 연구자가 또
하나의 dataset 선택권을 행사하게 된다. 현재 primary estimand는 case-selection method의 품질이 아니라 이미
고정된 Korean-heavy workload에서 output-identical actual latency가 줄어드는지다. 따라서 기존 case artifact를
폐기·재선정하지 않고 다음을 택한다.

1. 잘못된 blind claim을 명시적으로 철회한다.
2. Actual plan의 machine claim에
   `case_rank_seed_includes_compatibility_model_output_hash=true`와
   `case_selection_model_output_blind=false`를 봉인한다.
3. Candidate output·acceptance·latency를 처음 열기 전 actual protocol과 gate를 그대로 commit한다.
4. 결과를 exploratory Korean-centric scale-transfer evidence로만 해석한다.

이 선택은 case provenance 결함을 없애지 않는다. 결함을 숨기지 않고 claim 범위를 줄이는 것이다.

## 후속 confirmatory workload의 의무

Generic retrieval이 actual primary gate를 통과해 한국어 형태·띄어쓰기·자모 proposal을 비교할 가치가 생기면,
후속 case set은 다음 조건을 모두 만족해야 한다.

- 현재 train/evaluation/case documents와 exact·normalized disjoint
- long-shingle/MinHash near-duplicate와 selected 256-token span overlap audit를 case 선택 전에 고정
- candidate 및 comparator output·latency·acceptance를 열기 전에 case contract commit
- rank key를 raw source hash, exclusion-set commitment, 고정 case contract 같은 pre-output identity에서만 유일 도출
- compatibility result, model output, latency summary, 임의 salt를 rank seed에 사용하지 않음
- 새 case seal 이후에는 failure를 이유로 seed/pool을 교체하지 않음

이 조건을 만족하기 전에는 후속 결과를 final-blind confirmatory Korean-specific evidence라고 부르지 않는다.
