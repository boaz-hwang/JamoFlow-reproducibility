# Learning-curve noninferiority correction

> 작성일: 2026-08-11
> 상태: **publication-scale 학습곡선 결과 전 고정**
> 교정 대상: [publication comparator protocol](./48-publication-comparator-and-downstream-protocol.md) §11
> 영향 범위: publication data-adequacy gate만 변경; 진행 중 compact Phase 3는 불변

## 1. 기존 규칙의 estimand 오류

기존 gate는 마지막 두 training budget에서 `candidate − comparator` BPB의 부호가 바뀌면 실패시키고, 최종 절대 gap이 두 model의 학습 진행량 대비 작아도 실패시켰다. 이 규칙은 superiority 순위가 안정적인지를 묻는 데는 쓸 수 있지만 현재 연구 질문과 맞지 않는다.

JamoFlow의 quality estimand는 candidate가 comparator보다 더 좋아야 한다는 superiority가 아니라 `+0.010 BPB` 이내의 **noninferiority**다. 따라서 다음처럼 바람직한 결과도 기존 규칙에서는 실패한다.

- `−0.001 → +0.001 BPB`: 거의 동률인 채 부호만 바뀜
- `+0.001 → 0.000 BPB`: 학습이 진행되며 두 model이 수렴함
- 두 model의 BPB가 크게 개선되지만 차이는 계속 0에 가까움

실제 속도가 개선되고 품질 차이가 margin 안에 있는 연구에서 gap이 작다는 사실을 undertraining의 증거로 취급하는 것은 estimand 오류다.

## 2. 교정된 질문

Data adequacy는 다음 두 질문만 판정한다.

1. Reference가 downstream informativeness floor를 넘어 실제 capability를 보이는가?
2. Candidate가 마지막 두 matched-data budget 모두에서 raw, 16K BPE, 32K BPE 각각에 대해 BPB noninferiority를 유지하며 두 model의 학습이 계속 진행되는가?

Candidate와 comparator의 어느 쪽 BPB가 더 낮은지는 gate가 아니다. 부호 반전도 두 시점이 모두 margin 안이면 허용한다.

## 3. Last-two-budget gate

기본 learning curve는 64M, 128M, 256M matched raw bytes다. 각 comparator와 마지막 두 budget `B/2`, `B`에서 paired model seed 차이를 계산한다.

\[
d_s(B) = \operatorname{BPB}_{candidate,s}(B)
         - \operatorname{BPB}_{comparator,s}(B)
\]

각 budget에서 model seed를 paired resampling한 one-sided 97.5% bootstrap upper bound를 계산한다. Pair 하나가 통과하려면 다음을 모두 만족해야 한다.

1. `B/2`와 `B` 각각의 bootstrap upper bound가 `+0.010 BPB`보다 작음
2. 두 budget 각각에서 세 seed 중 최소 두 seed가 `<= +0.010 BPB`
3. Candidate와 comparator의 mean BPB가 `B/2 → B`에서 증가하지 않음
4. Candidate와 comparator 각각 세 seed 중 최소 두 seed의 BPB가 `B/2 → B`에서 증가하지 않음

Raw, 16K data-matched BPE, 32K data-matched BPE 세 pair가 모두 통과해야 한다. Compute-matched BPE는 본 raw bytes가 다르므로 matched-data learning-curve gate에 넣지 않는다. Current-budget의 document-clustered BPB gate는 별도로 유지되며 이 seed-level trend guard가 대신하지 않는다.

## 4. 왜 미래 gap을 외삽하지 않는가

세 checkpoint만으로 difference scaling law를 적합하거나 다음 doubling gap을 선형 외삽하지 않는다. 개별 model loss는 power-law 형태일 수 있지만 두 architecture loss의 차이는 같은 함수형을 따른다는 보장이 없고, 세 seed·세 budget으로 추정한 extrapolation은 protocol의 정밀도보다 model assumption이 더 커진다.

따라서 claim은 실제로 측정한 최대 budget과 hardware에만 한정한다. 미래 scale에서도 유지된다고 말하려면 그 scale을 직접 학습하고 같은 gate를 다시 실행해야 한다.

## 5. Extension rule

256M에서 downstream floor 또는 last-two-budget gate가 실패하면 positive broad claim을 내지 않는다. 자원이 허용되어 data extension을 수행할 경우 결과에 맞춰 임의의 종점을 고르지 않고 512M과 1.024B matched-data checkpoint를 모두 만든다. 그때는 512M–1.024B가 새로운 last-two-budget pair다.

1.024B에서도 downstream floor 또는 세 pair 중 하나가 실패하면 Mac-only broad claim을 중단한다. 이는 1.024B가 충분한 data라는 뜻이 아니라, 이 연구에서 결과를 보지 않고 고정한 최대 local extension이다.

## 6. Claim 영향

이 교정은 quality gate를 느슨하게 만들어 superiority를 주장하려는 것이 아니다. 오히려 연구 질문에 없는 순위 부호를 제거하고, 두 실제 checkpoint에서 직접 검증한 noninferiority와 capability만 남긴다.

- BPB가 margin 밖이면 actual latency와 무관하게 실패한다.
- Downstream이 uninformed floor면 BPB가 좋아도 capability-undertrained다.
- Near-tie 또는 sign reversal은 두 budget 모두 noninferior일 때만 허용한다.
- 통과해도 256M/1.024B, 50–100M, Apple MPS 범위를 넘어 scaling claim을 하지 않는다.
