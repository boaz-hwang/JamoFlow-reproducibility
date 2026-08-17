# Phase 3 addendum: crossed bootstrap and summary integrity

> 작성일: 2026-08-10
> 상태: **W primary contrast와 OOD 결과 확인 전 고정**
> 상위 protocol: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md)
> 영향: policy, seed 수, effect·sign·interval threshold는 변경하지 않음; Gate M에서 잘못 이름 붙인 bootstrap tail을 paired-seed Student-$t$ p-value로 교정
> 후속 정정: 이 문서의 shared-window bootstrap은 document dependence를 충분히 반영하지 못했다. HPLT primary와 downstream gates에는 [document-cluster correction](./52-document-cluster-inference-integrity-addendum.md)이 우선하며, shared-window interval은 diagnostic으로만 남는다.

## 1. 발견한 통계 설계 불일치

Phase 3의 모든 seed는 동일한 held-out test window를 평가한다. 따라서 seed와 sequence는 nested 관계가 아니라 **crossed 관계**다.

기존 bootstrap 구현은 다음처럼 동작했다.

1. seed를 replacement로 표집
2. 선택된 seed occurrence마다 sequence를 서로 독립적으로 replacement 표집

이 방식은 각 seed가 서로 다른 test sample을 가진 nested design에는 맞지만, Phase 3처럼 같은 31,250개 window를 공유하는 설계에서는 공통 예제 난이도의 seed 간 상관을 깨뜨린다.

## 2. 고정한 crossed paired bootstrap

각 bootstrap replicate는 다음을 수행한다.

1. seed index를 replacement로 표집한다.
2. test sequence index를 replacement로 **한 번** 표집한다.
3. 같은 sequence index vector를 모든 선택된 seed에 적용한다.
4. 선택된 seed × sequence crossing의 paired policy NLL 차이를 평균한다.
5. 511 predicted bytes와 `ln 2`로 나눠 BPB contrast로 변환한다.

모든 seed의 loss vector 길이가 같지 않으면 summary를 중단한다. 이 구현은 policy pairing뿐 아니라 seed가 공유한 test sample도 보존한다. HPLT3 primary, Leipzig OOD, conditional mechanism control, private ecology aggregate에 같은 helper를 사용한다.

Bootstrap 명칭은 기존 문서와의 연결을 위해 `hierarchical`을 일부 field에 유지하지만, output metadata에 다음 design을 명시한다.

```text
crossed seeds x shared test sequences
```

## 3. bootstrap tail의 통계적 명칭 정정

기존 코드는 bootstrap estimate 중 0 이상인 비율에 add-one correction을 적용한 값을 `pvalue`라고 불렀다. 이 값은 percentile-bootstrap tail diagnostic이지 exact 또는 일반적으로 calibrated된 hypothesis-test p-value라고 단정할 수 없다.

따라서 다음처럼 정정한다.

- 이름: `bootstrap_nonnegative_tail`
- 역할: bootstrap distribution의 방향을 보여주는 diagnostic
- 금지: exact p-value 또는 독립적인 유의성 증거라고 표기
- Holm 순서·보정값: seed-level paired effect의 one-sided Student-$t$ p-value
- Gate 근거: 사전 고정 effect size, negative seed count, crossed-bootstrap 95% upper, paired-seed Holm-adjusted p-value, OOD guard

Student-$t$ CDF는 SciPy 의존성을 추가하지 않도록 regularized incomplete beta로 계산하고, tabulated critical value(df=4, one-sided 0.05/0.025)에 대한 회귀 테스트를 고정한다. Primary Gate J와 mechanism Gate M의 다중비교 조건은 오표기된 bootstrap tail이나 비표준 순차 분위수 대신 이 seed-level paired-$t$ p-value의 Holm 보정을 사용한다. Effect·sign·crossed-bootstrap 95% upper threshold는 바뀌지 않는다. 상세한 판정 정의는 [후속 교정](./35-phase3-primary-family-inference-correction.md)에 고정했다.

## 4. summary integrity 강화

Primary summary는 이제 aggregate를 쓰기 전에 다음을 검증한다.

- seed가 사전등록 initial 3 또는 final 5의 정확한 순서인지
- F/C/W primary가 모두 있고 policy가 중복되지 않는지
- full-run manifest인지, model/optimization spec과 byte limit가 맞는지
- 재구성한 test stream SHA-256이 manifest와 같은지
- train/calibration/test example·predicted-byte·training-step 수가 manifest와 맞는지
- 각 test NLL vector가 finite·nonnegative이고 절대 BPB를 `1e-7` 이내로 재구성하는지
- 각 `.pt` state dict를 직접 다시 hash한 값이 training report와 같은지
- F/C/W의 모든 row가 각 split에서 정확히 86 patch이고 padding slot이 0인지
- structural patch matrix가 seed-independent인지
- OOD summary가 primary와 정확히 같은 seed set을 사용하고 checkpoint/rate/hash integrity를 통과했는지

Leipzig OOD summarizer도 loss shape, finite/nonnegative 값, 절대 BPB 재구성, row-level exact rate, manifest seed/policy 일치를 추가로 검증한다.

## 5. 결과 열람 시점 공개

이 수정 시점에는 seed 1,729의 F와 C scalar report가 존재했고 W는 학습 중이었다. W−C, OOD, E/EC/S, mechanism control, cost 결과는 존재하지 않았다. 수정은 통계 자료구조와 provenance 검증에서 발견된 문제에 의해 이루어졌으며 모든 contrast에 대칭적으로 적용된다.

## 6. 남는 제한

- initial 3와 final 5 seed는 여전히 작은 상위 sampling unit이다.
- bootstrap 반복 수를 늘려도 seed 다양성이 늘지는 않는다.
- percentile bound는 작은 seed 수에서 불안정할 수 있으므로 paired seed effects와 paired-$t$ interval을 함께 공개한다.
- descriptive strata는 multiplicity-controlled discovery test가 아니다.
- Gate threshold 통과는 practical importance와 publication scale을 자동 보장하지 않는다.

이 수정은 유리한 결과를 만들기 위한 분석 변경이 아니라, 실제 crossed experiment와 resampling unit을 일치시키고 artifact corruption·seed mismatch가 gate에 들어오는 것을 막기 위한 결과 전 교정이다.
