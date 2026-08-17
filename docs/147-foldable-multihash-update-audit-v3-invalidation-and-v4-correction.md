# Foldable multi-hash update audit v3 무효화와 v4 alignment 교정

> 작성일: 2026-08-15
>
> 상태: v3는 result artifact 생성 전 중단; v4에서 zero-gradient 정의역 교정

## 문제 정의

- 기대: 첫 고정 batch에서 dense와 multi-hash를 각각 한 번 업데이트하고, direct token-row
  gradient와 해당 hash-bucket aggregate gradient의 방향 정렬을 진단한다.
- 실제: forward, backward와 optimizer step 뒤 input collision-alignment 계산에서
  `bucket-alignment row gradient is zero`로 중단됐다.
- 결과 노출: V3 summary 파일과 stdout 정량 metric은 생성되지 않았다. Stack trace로 확인한 사실은
  신규 input row 집합에 zero-gradient 행이 하나 이상 있다는 것뿐이다.

## 재현과 원인

고정 batch는 32×512 token이다. 신규 vocabulary에는 6,144개 행이 있지만 한 batch에 모든 신규
token이 input으로 등장하지 않는다. 따라서 미등장 input embedding 행의 direct gradient는 정확히
0이다. 기존 구현은 6,144개 전부가 nonzero라고 가정해 cosine의 정의역과 실제 sparse exposure를
혼동했다.

Output head는 softmax 때문에 target으로 직접 등장하지 않은 행도 일반적으로 gradient를 갖는다.
그러므로 input과 output의 active row 집합은 동일하다고 가정하지 않고 각각 독립적으로 계산해야
한다.

## V4의 단일 수정

V4 collision alignment는 다음처럼 정의한다.

1. matrix별 direct lexical row norm이 정확히 0보다 큰 행만 기본 정의역으로 삼는다.
2. 각 slot에서 선택된 bucket-gradient norm도 0보다 큰 행만 cosine에 포함한다.
3. 전체 신규 행 수, nonzero direct-row 수, 제외된 zero-row 수, slot별 aligned row 수와
   nonzero direct row 중 zero selected-bucket 수를 모두 기록한다.
4. nonzero direct row 또는 aligned row가 하나도 없으면 계속 fail-closed한다.

이는 모델, batch, checkpoint, optimizer, update geometry, projection multiplier 또는 control 선택
규칙을 바꾸지 않는다. Zero를 작은 epsilon으로 바꾸거나 임의 값을 채우지 않으며, 수학적으로
정의되지 않는 cosine만 제외하고 그 제외 규모를 공개한다.

## 예방

- synthetic zero-row regression test를 추가한다.
- 전 행이 zero인 경우의 fail-closed test를 추가한다.
- V1/V2/V3 plan은 삭제하거나 재사용하지 않고 V4 namespace를 새로 봉인한다.
- V3가 optimizer step까지 수행했다는 사실을 명시하되, 정량 결과가 출력·파일화되지 않았으므로
  V4의 multiplier나 이후 quality gate는 V3 관측값으로 조정하지 않는다.
