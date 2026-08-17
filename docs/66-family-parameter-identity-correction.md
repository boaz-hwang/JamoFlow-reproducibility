# Family parameter-identity correction

> 작성일: 2026-08-12
> 상태: **family-aware preflight 구현·실행 전 결과맹 교정**
> 상위 protocol: [family-aware scale-feasibility correction](./64-family-aware-scale-feasibility-correction.md)

## 1. 발견한 self-attestation 경로

`FamilyScaleFeasibility`는 worker가 보고한 `parameter_count`와
`expected_parameter_count`가 같으면 family 단독 gate를 통과시켰다. Final
campaign 타입은 candidate와 raw reference에 대해서만 기대값을 독립적인
publication-scale 상수와 대조했다. 16K와 32K BPE는 두 필드를 같은 잘못된
숫자로 기록해도 통과할 수 있었다.

기존 unit test도 이 약점을 그대로 반영해 BPE 기대값을 각각
`candidate_parameters - 1000`, `candidate_parameters - 100`이라는 임의 값으로
만들고 passing campaign으로 취급했다. 이는 runner artifact가 자신의 정답을
스스로 선언하는 구조이며 parameter-matched/body-matched comparator의 lineage
gate로 사용할 수 없다.

이 감사 시점에는 family-aware runner, tokenizer, preflight report와
publication-scale checkpoint가 존재하지 않았다. 따라서 실제 memory/time 또는
quality 결과에 맞춘 변경이 아니다.

## 2. Target×family main-graph 봉인 표

Final scale lock은 다음 정확한 **main graph** 기대 파라미터 수를 코드 상수로 검증한다.

| Target | Candidate | Raw byte reference | Body-matched BPE 16K | Parameter-matched BPE 32K |
|---:|---:|---:|---:|---:|
| 50M | 49,823,488 | 49,823,488 | 42,617,792 | 49,785,792 |
| 75M | 76,492,480 | 76,492,480 | 66,710,368 | 76,438,368 |
| 100M | 98,403,360 | 98,403,360 | 86,975,680 | 98,239,680 |

16K는 32K Transformer body를 그대로 유지하고 tied embedding/output rows만
줄이므로 candidate total parameter 수와 맞지 않는 것이 의도된 stress
control이다. 32K만 candidate total parameter 수의 1% 이내다.

Raw reference가 structural F/C/W/S 또는 selected-C이면 auxiliary parameter는 0이다.
E/EC이면 compact selection descriptor에서 `entropy_router`를 파생하고 다음 별도
router graph를 반드시 더한다. Worker가 `none`을 선택할 수 없다.

| Target | Main raw graph | Entropy router | Entropy raw total resident/trainable system |
|---:|---:|---:|---:|
| 50M | 49,823,488 | 3,541,248 | 53,364,736 |
| 75M | 76,492,480 | 5,491,520 | 81,984,000 |
| 100M | 98,403,360 | 6,626,400 | 105,029,760 |

논문 표에는 main-graph match와 실제 total system parameter를 둘 다 보고한다. Entropy
router의 추가 용량을 숨긴 채 raw reference를 candidate와 parameter-matched라고 부르지
않는다.

## 3. Machine-checkable 교정

`PUBLICATION_FAMILY_EXPECTED_PARAMETERS[target][family]`를 final identity의
단일 표로 추가했다.

- 각 `FamilyScaleFeasibility`는 자신의 target을 명시한다.
- `actual == worker_expected`뿐 아니라
  `worker_expected == sealed_target_family_expected`를 요구한다.
- `CampaignScaleFeasibility`는 네 family의 target이 campaign target과 모두
  같은지 검사한다.
- BPE graph derivation도 같은 봉인 표와 analytical/Transformers parameter
  count가 일치하는지 교차 검증한다.
- Raw family의 `auxiliary_kind`는 compact selection artifact의 concrete policy에서
  파생한다. E/EC이면 target별 router actual count가 위 표와 같아야 하고 structural이면
  auxiliary count와 모든 auxiliary workload field가 0이어야 한다.
- `total_runtime_parameter_count = main + auxiliary`는 입력값이 아니라 파생값이다.

Candidate, raw 또는 어느 BPE family든 actual과 expected를 함께 같은 오답으로
바꾸면 family gate와 campaign gate가 모두 실패한다. 향후 family-aware runner는
이 표를 report 입력으로 받지 않고 현재 코드에서 직접 재구성해야 한다.

## 4. 영향 범위

이 교정은 graph geometry, memory/time threshold, scale order와 campaign-hour
식을 바꾸지 않는다. Family-aware runner를 구현할 수 있는 무결성 전제만
강화한다. Candidate-only provisional preflight도 final scale lock으로 승격되지
않는다.
