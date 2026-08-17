# Foldable multi-hash AdamW update audit 결과와 mechanism 결정

> 작성일: 2026-08-15
>
> 상태: V4 complete; development mechanism evidence only
>
> result: `results/foldable-multihash-update-audit-v4/summary.json`
>
> canonical summary SHA-256:
> `f1d9c0082f539932f7ef9df79273cf62940abda11df3671cb9a3aac5aedd347f`

## 결론

Generic multi-hash의 B1 이득을 단순한 “새 vocabulary row의 learning rate 증가”로 설명할 수
없다. 정렬된 dense-update 성분은 input `1.4854×`, output `2.1706×`였지만 multi-hash effective
update의 전체 norm은 각각 `3.4868×`, `3.3513×`였다. 후보 update 중 dense update와 직교하는
비율도 input `90.47%`, output `76.19%`였다.

따라서 다음 실험에는 두 control이 모두 필요하다.

1. 고정 projection multiplier만 재현하는 `update_matched_dense`
2. 같은 slot·bucket·parameter budget으로 collision diffusion을 보존하되 surface 의미를 제거한
   `balanced_random_multihash`

첫 control만으로 generic multi-hash를 설명할 수 있으면 optimizer-scale artifact로 판정한다. 두
번째 control이 generic surface와 같거나 더 좋으면 Jamo뿐 아니라 surface-semantic claim도 버리고,
효과를 generic shared-hash reparameterization으로만 해석한다.

## 무결성 확인

- 시작 effective input/output weight: bitwise identical
- 네 microbatch loss의 maximum absolute difference: `0.0`
- 동일 step-0 physical checkpoint, 32×512 first batch, optimizer와 clipping 계약 사용
- audit은 BPB, final test, latency를 읽지 않고 multiplier를 고정
- 이 결과는 development train batch의 mechanism audit이며 model-quality 또는 publication 증거가
  아니다.

## Update geometry

| matrix | projection multiplier | norm ratio | cosine | orthogonal fraction |
|---|---:|---:|---:|---:|
| input | 1.4854 | 3.4868 | 0.4260 | 0.9047 |
| output | 2.1706 | 3.3513 | 0.6477 | 0.7619 |

Input row별 norm ratio의 median `588.68×`는 전체 학습률 증폭으로 읽으면 안 된다. 첫 batch에
직접 등장하지 않은 신규 input row는 ordinary dense graph에서 주로 작은 weight-decay update만
받지만, multi-hash graph에서는 다른 token이 같은 bucket을 업데이트하면 effective weight가 함께
움직인다. 작은 dense denominator와 collision diffusion이 결합해 큰 비율을 만든다.

이 해석은 alignment 진단과도 일치한다.

- 신규 input 6,144행 중 direct gradient nonzero: 2,506행
- zero direct-gradient input row: 3,638행
- output direct gradient nonzero: 6,144행 전부
- input bucket-gradient mean cosine: `0.2045`
- output bucket-gradient mean cosine: `0.4299`
- nonzero direct row가 선택한 bucket gradient는 모든 13 slot에서 nonzero

즉 residual branch는 같은 token의 gradient를 단순 복제하는 것이 아니라, bucket을 공유한 여러
token의 gradient를 섞어 각 effective vocabulary row로 확산한다.

## Gradient clipping

| graph | pre-clip total norm | clip coefficient |
|---|---:|---:|
| ordinary dense | 3.1141 | 0.3211 |
| multi-hash | 3.6066 | 0.2773 |

Multi-hash는 더 강하게 clipping됐다. 그럼에도 effective update norm이 더 컸으므로, residual
parameterization과 AdamW의 per-parameter normalization이 clipping 차이 이후에도 독립적인 update
geometry를 만든다. `1 + sqrt(13)` 같은 이상화한 첫-step 배수는 실제 graph의 설명값으로 사용할 수
없다.

## 다음 control의 고정값

`update_matched_dense`는 quality 결과를 보지 않고 다음 값을 고정한다.

- 신규 input row post-AdamW update multiplier: `1.485414522979104`
- 신규 output row post-AdamW update multiplier: `2.170601418278963`
- old vocabulary row와 Transformer body: ordinary dense AdamW 그대로
- moment state: ordinary dense optimizer 그대로

이는 multi-hash optimizer와 동등한 control이 아니다. Dense update와 정렬된 first-step 성분만
재현하는 강한 단순 control이다. 이후 step에서 moment와 collision-coupled direction이 달라지는
것이 바로 비교하려는 잔여 mechanism이다.

## 연구 방향에 주는 영향

기존 계획을 폐기할 이유는 없다. 다만 positive-paper 후보의 표현을 더 좁힌다.

- 더 이상 “Jamo-aware residual”을 주 후보로 삼지 않는다.
- 단순 “새 token을 더 빨리 학습시킨다”는 주장도 control 통과 전에는 하지 않는다.
- 현재 후보는 training-time shared-hash vocabulary reparameterization이며 inference 때 exact dense
  graph로 fold된다는 조합이다.
- 한국어 연구 가치는 fresh Korean corpus에서 matched-quality actual E2E 속도가 실제로 개선될 때만
  열린다. Mechanism screen의 BPB 이득만으로 논문 성공을 선언하지 않는다.

다음 screen이 실패하면 이 positive branch는 중단하고, W72 boundary-placement의 primary-negative
연구와 total-cost Pareto 정리로 돌아간다. 통과하면 fresh disjoint Korean data, multi-seed,
strong vocabulary-adaptation control과 실제 batch-1 inference 비교로 승격한다.
