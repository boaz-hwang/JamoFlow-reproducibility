# Phase 3 primary-family inference correction

> 고정일: 2026-08-10  
> 상태: **첫 W primary contrast와 모든 OOD 결과 확인 전 고정**  
> 영향: Gate J/M의 effect size, sign count, crossed-bootstrap 95% upper, OOD 기준은 유지; 다중비교 판정만 표준 Holm 검정으로 명확화

## 1. 발견한 문제

초기 Phase 3 요약기는 두 primary contrast를 one-sided paired-seed Student-$t$ p-value 순서로 정렬한 뒤, 각 순서에 `1 - 0.05/(m-rank)` bootstrap 분위수를 적용했다. 이를 “Holm-adjusted bootstrap upper bound”라고 불렀다.

이 구성은 설명 가능해 보이지만 표준 Holm 검정이 아니다. Holm의 step-down family-wise error control은 유효한 개별 p-value를 순서화하고 임계값과 비교하거나, 동등하게 Holm-adjusted p-value를 계산하는 절차다. 서로 다른 통계량으로 순서를 정한 뒤 percentile-bootstrap bound를 순차 비교한 혼합 절차에는 여기서 주장할 수 있는 명시적 오류율 보장이 없다.

Top-tier 심사에서 이 값을 정식 multiplicity control로 제시하는 것은 부적절하므로 결과 전에 제거한다.

## 2. 최종 판정 정의

W−C와 W−F 각각에 대해 다음을 모두 보고한다.

1. seed별 BPB 차이와 평균
2. seed-level two-sided paired-$t$ 95% interval
3. seed × 공통 test sequence crossed-bootstrap 95% interval
4. one-sided paired-seed Student-$t$ raw p-value
5. 두 raw p-value에 대한 Holm-adjusted p-value
6. add-one bootstrap nonnegative tail diagnostic

Gate J의 각 contrast는 다음을 모두 만족해야 한다.

- mean `<= −0.003 BPB`
- 최소 4/5 seed negative
- crossed-bootstrap 95% upper `< 0`
- Holm-adjusted one-sided paired-seed Student-$t$ p-value `<= 0.05`

그리고 public OOD guard를 통과해야 Gate J 전체가 통과한다. Gate M의 final-five 두 mechanism contrast에도 같은 추론 구조를 적용한다. Initial-three Gate I/M은 더 큰 실험 실행 여부를 정하는 screening gate이므로 기존 effect·sign·integrity·OOD 조건만 사용한다.

## 3. 각 통계량의 역할

| 통계량 | 역할 | 하지 않는 주장 |
|---|---|---|
| paired-seed $t$ + Holm | 두 primary hypothesis의 정식 family-wise 판정 | seed 모집단의 비정규성에 강건하다는 주장 |
| crossed bootstrap 95% interval | 공통 held-out sample 구성에 대한 민감도 | 독립 seed 수를 5보다 늘렸다는 주장 |
| bootstrap nonnegative tail | 방향 진단 | calibrated p-value라는 주장 |
| effect threshold와 sign count | 실질 크기와 반복성 | 그 자체만으로 sampling uncertainty가 해결된다는 주장 |

독립 학습 seed가 다섯 개뿐이라는 제한은 남는다. Exact paired sign-flip test는 가능한 부호 배치가 32개뿐이어서 최소 단측 p-value가 1/32이고, 두 가설을 Bonferroni/Holm 보정하면 단독으로 0.05를 통과할 수 없다. 따라서 이를 확증 gate로 가장한 채 사용하는 대신, seed effect의 분포 가정을 명시한 Student-$t$ 검정과 원자료 다섯 값을 함께 공개한다.

## 4. 결과 열람 시점과 비대칭 방지

교정 시점에 seed 1,729의 F와 C scalar 결과만 존재했고 W는 학습 중이었다. 따라서 W−C, W−F, OOD, mechanism contrast의 부호·크기·p-value는 관측되지 않았다. 동일 규칙은 두 primary contrast와 두 mechanism contrast에 대칭적으로 적용된다.

이미 확인한 F와 C의 절대 BPB는 이번 family test의 방향이나 통과 여부를 결정할 수 없다. 이 교정은 유리한 p-value 정의를 고른 조치가 아니라, 비표준 혼합 절차를 표준 검정과 독립 sensitivity check로 분리한 조치다.

## 5. 구현 규약

Summary JSON의 `holm_primary_family`와 `holm_mechanism_family`에는 다음만 multiplicity 결과로 저장한다.

- family 내 rank와 크기
- raw one-sided paired-seed Student-$t$ p-value
- Holm-adjusted p-value
- family-wise alpha 0.05 기각 여부

Crossed-bootstrap interval과 tail diagnostic은 contrast 자체의 별도 field에 둔다. `step_upper_quantile`, `step_upper_bpb`, `step_rejects_nonnegative_effect`는 출력에서 삭제한다. Gate 코드는 crossed-bootstrap 95% upper와 Holm-adjusted seed p-value를 서로 다른 조건으로 직접 검사한다.
