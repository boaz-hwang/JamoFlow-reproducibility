# Publication evidence-identity correction

> 작성일: 2026-08-12
> 상태: **publication-scale prediction·timing artifact 생성 전 결과맹 교정**
> 적용 범위: downstream, learning-curve, comparator actual-inference, Final Value Gate

## 1. 발견한 결함

기존의 개별 통계 함수는 prediction, document loss와 latency array에서 지표를
계산했지만, 상위 gate 사이에서는 일부 결과를 단순 boolean으로 전달했다.

- data-adequacy gate는 `downstream_informativeness_pass=True`를 직접 받을 수 있었다.
- comparator inference gate는 `downstream_noninferiority_pass=True`를 직접 받을 수
  있었다.
- Final Value Gate는 `data_adequacy_pass=True`를 직접 받을 수 있었다.
- BPB, downstream과 latency gate에 candidate/comparator identity가 완전히 연결되지
  않아 서로 다른 실험의 passing result를 한 final result로 이어 붙일 수 있었다.

이는 통계식 자체의 오류와 별개인 **evidence stitching** 결함이다. 정직한 runner가
정상 값을 넣는다는 가정 아래서는 드러나지 않지만, publication artifact의 일부가
stale하거나 다른 checkpoint family를 가리킬 때 false pass를 만들 수 있다.

## 2. 교정된 증거 그래프

상위 판정기는 이제 하위 pass boolean을 직접 받지 않는다.

```text
task별 paired predictions
  -> PublicationDownstreamGate(candidate identity)
       |-> comparator actual-inference gates
       `-> PublicationDataAdequacy(candidate identity)

document losses(candidate, comparator identities)
  -> PublicationBPBNoninferiority
       -> PublicationComparatorInferenceGate

raw + BPE-16K + BPE-32K comparator gates
  + PublicationDataAdequacy
  -> PublicationFinalValueGate
```

구체적으로 다음을 강제한다.

1. 모든 downstream task가 동일한 candidate identity를 가져야 한다.
2. data-adequacy curve의 candidate와 downstream candidate가 같아야 한다.
3. BPB result의 candidate/comparator가 comparator inference gate와 같아야 한다.
4. raw, 16K, 32K actual gate의 candidate가 모두 같아야 한다.
5. Final Value Gate의 raw/BPE family key와 data-adequacy curve key가 정확히
   일치해야 한다.
6. seed order는 모든 단계에서 봉인된 publication seeds와 같아야 한다.

허용 model-family key와 16K/32K vocabulary 대응은 후속
[publication comparator-role lock](./69-publication-comparator-role-lock.md)에서 exact
set으로 봉인한다.

여기서 model key는 training-budget variant가 아니라 stable model-family identity다.
Data-matched와 compute-matched checkpoint의 구체적 차이는 향후 lock artifact의
checkpoint hash와 budget metadata로 표현하고, family key 자체를 사후 변경하지
않는다.

## 3. Runtime 경계 후속 교정

위 문서를 작성할 당시 남아 있던 runtime boolean 경계는
[publication runtime evidence correction](./74-publication-runtime-evidence-correction.md)에서
제거했다. Comparator gate는 이제 다음 원시 증거에서 재구성한 단일
`PublicationRuntimeEvidence`만 받는다.

- seed별 checkpoint·config와 tokenizer/UTF-8 transition hash
- 공통 raw prompt/continuation 및 모델별 실제 unitization hash
- full-forward 대 incremental logit 배열 비교
- warm-up, seed/trial 실행 순서, 환경 snapshot과 timing 배열
- emitted byte/model-unit/forward-step/DFA/overshoot 항등식

다만 합성 배열 unit test는 여전히 publication 결과가 아니다. 실제 checkpoint와
private numeric artifact를 읽는 final runner가 같은 builder를 호출한 뒤에만
Final Value Gate를 실행하며, BPB/downstream까지 concrete checkpoint lock을 확장하는
작업도 별도로 남는다.

## 4. 검증

당시 Downstream/data-adequacy/publication-inference focused test 23개가 통과했다.
후속 runtime 교정의 검증 수와 경계는 문서 74에 기록한다. 전체 suite는 진행 중인
단일 MPS training family가 종료된 뒤 실행한다.
