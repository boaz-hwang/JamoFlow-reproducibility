# Fresh vocabulary trained actual-inference 결과와 16K 전환

> 작성일: 2026-08-15
>
> 결과 커밋: `3d63e44`
>
> 결과 payload SHA-256: `75e8a661706b34e054050607a31491e6fe37edd65a5425078d153e95a4b3f861`
>
> 결과 file SHA-256: `33b59bc1cb4daf6e721cc5765f04a81aadfc9c4d744bfca2d6e74009410e07bc`
>
> 판정: 8K trained actual gate 실패; 현재 8K multi-seed 확대 중단

## 결론

Fresh one-seed에서 품질이 더 좋았던 `dense8k_update_geometry`는 실제 trained
controlled replay에서 `dense2k_joint`보다 **20.131% 빨랐다**. 그러나 strict-UTF8
free-running에서는 point reduction이 **8.843%**였고 bootstrap lower가 음수였으며, 빠른
문서도 `44/64`에 그쳤다. 사전 계약은 두 mode 모두 point `>=10%`, bootstrap lower `>0`,
빠른 문서 `>=48/64`를 요구했다. 따라서 공동 gate는 실패했고 multi-seed confirmation은
승인되지 않는다.

이 결과는 실제 속도 개선이 전혀 없었다는 뜻은 아니다. 두 mode의 관측 point는 모두
양수였다. 다만 사용자가 정한 논문 성공 기준과 결과 전에 봉인한 uncertainty/stability 기준을
충족하지 못했다. Free gate를 8%로 낮추거나 controlled만 primary로 바꾸지 않는다.

핵심 원인은 tokenizer 시간이 아니라 **자유 생성 token-length 분포**다. 같은 128 raw-byte
continuation을 강제로 replay하면 8K는 35개 token을 27개로 줄였고 latency도 거의 같은 비율로
줄었다. 각 모델이 자기 출력을 생성하면 중앙 token 수 차이가 36.5 대 33.5로 작아졌고,
문서별 latency reduction과 token-count reduction의 상관은 `0.9996`이었다. 현재 8K의 병목은
추가 optimizer tuning이 아니라 자유 생성에서도 충분한 step headroom을 안정적으로 확보하는
것이다.

## 사전 고정 gate 결과

| mode | candidate E2E | reference E2E | reduction | paired-prompt 95% CI | candidate faster | 판정 |
|---|---:|---:|---:|---:|---:|---|
| controlled replay | 66.728 ms | 83.547 ms | **20.131%** | [16.524%, 22.723%] | 64/64 | pass |
| free-running UTF-8 greedy | 79.944 ms | 87.699 ms | **8.843%** | [-0.168%, 16.685%] | 44/64 | fail |

두 checkpoint는 warm-up case의 full no-cache와 parallel-prefill/incremental-cache 경로에서 모든
argmax가 exact 일치했다. 모든 640 free output은 strict UTF-8, stop rule과 repetition 간
determinism을 통과했고 summary가 checkpoint와 token trace를 독립 replay했다. 실패를
correctness 문제로 돌릴 근거는 없다.

## 원인 분해

### 1. Controlled speed는 token-step 감소가 만들었다

| diagnostic | 8K candidate | 2K reference | candidate reduction |
|---|---:|---:|---:|
| controlled output tokens | 27.0 | 35.0 | 22.86% |
| controlled decode | 63.132 ms | 79.850 ms | 20.94% |
| controlled E2E | 66.728 ms | 83.547 ms | 20.13% |
| tokenizer encode | 0.060 ms | 0.061 ms | 2.22% |
| TTFT | 3.491 ms | 3.444 ms | **-1.37%** |

Tokenizer encode는 전체의 0.1%보다 작다. 8K가 빨라진 이유는 긴 token으로 autoregressive
호출을 줄였기 때문이며 TTFT나 prompt encoding 개선이 아니다.

### 2. Free-running은 서로 다른 출력 분포가 지배했다

Free output은 역할별 128--137 raw bytes로 거의 같은 길이였지만 token 수 분산이 컸다.

- candidate token count: 20--91, median 33.5, mean 35.38
- reference token count: 25--128, median 36.5, mean 43.31
- 문서별 paired token reduction median: 14.12%
- 문서별 paired latency reduction median: 11.13%
- latency/token-reduction Pearson correlation: 0.99958

극단 사례에서 candidate는 91 token, reference는 45 token을 생성해 103.9% 느렸다. 반대로
reference가 128-token ceiling까지 간 세 사례에서는 candidate가 71--74% 빨랐다. 이 차이는
kernel noise보다 각 작은 모델이 생성한 문장의 tokenizability 차이다. Free-running은 실제 경로를
재지만, 서로 다른 model output이라는 필연적 confound 때문에 controlled causal contrast보다
분산이 크다는 점을 이후 통계와 주장에 명시해야 한다.

Aggregate script 진단에서도 두 역할 모두 출력의 약 59--60%가 완성형 Hangul이었고 strict-valid였다.
그러나 이는 언어 품질 평가가 아니다. 한 seed의 free output 길이 분포를 근거로 model이나 prompt를
다시 선택하지 않는다.

### 3. 큰 lexical head 비용은 작지만 불리한 방향이다

Candidate는 8K input embedding과 output head를 untie해 각 `8,192 x 384` matrix를 가진다.
Reference는 `2,048 x 384` 한 matrix를 tie한다.

- parameters: 25,172,352 vs 19,667,328, candidate +27.99%
- checkpoint bytes: 100,713,026 vs 81,838,658, candidate +23.06%
- free E2E/output-token diagnostic: 약 2.440 vs 2.381 ms/token

따라서 8K의 token 하나는 약간 더 비싸다. 다만 controlled에서 20% 개선이 남았으므로 full-vocabulary
head가 주 병목이라고 단정할 수는 없다. 현재 결과에서 더 큰 문제는 free output에서 step reduction이
안정적이지 않았다는 것이다. Memory 개선은 측정하지도 주장하지도 않는다.

## Fable 5 검토에 대한 최신 판정

`fable5-연구-중간-검토.md`에서 다음은 계속 수용한다.

1. analytical patch/FLOP/token 감소를 실제 E2E와 분리한다.
2. per-step local/body/head 비용 때문에 proxy 절감이 그대로 wall time이 되지 않는다고 본다.
3. rate, placement, output length와 per-step cost를 각각 분해한다.
4. compact Apple-MPS 결과를 production/CUDA/대형 모델로 일반화하지 않는다.

이번 결과는 첫 원칙을 vocabulary branch에서도 재확인했다. Controlled에서는 token 감소가 실제
속도로 이어졌지만, free-running은 출력 분포 때문에 같은 결론을 안정적으로 재현하지 못했다.

다음 제안은 현재 우선순위로 수용하지 않는다.

- speed 실패를 quality/방법론 중심 소논문으로 간주해 연구를 종료하지 않는다.
- S rate-placement 분해는 boundary negative paper의 보조 질문이며 새 효율 후보보다 앞세우지 않는다.
- compact joint gate를 통과하지 못한 상태에서 CUDA나 큰 캠페인으로 바로 확대하지 않는다.

Fable 문서의 `컴퓨트 개선 확립`도 wall-clock과 구분되는 좁은 analytical claim으로만 유지한다.
현재 사용자의 성공 기준에서 실제 효율 개선이 확립됐다고 부를 수 있는 branch는 아직 없다.

## 왜 다음은 16K인가

8K threshold를 사후 완화하거나 같은 checkpoint의 prompt를 다시 고르는 대신, 이번 결과 전에
봉인·측정된 same-body systems frontier에서 **다음으로 작은 vocabulary**를 사용한다.

| trained 전 systems role | controlled steps vs 2K | controlled E2E vs 2K |
|---|---:|---:|
| dense 8K | -22.98% | -19.81% |
| **dense 16K** | **-30.59%** | **-24.96%** |
| dense 32K | -36.80% | -24.00% |

32K는 16K보다 step이 적지만 큰 output head 때문에 실제 E2E가 더 느렸고, 앞선 one-pass quality
frontier에서도 vocabulary가 커질수록 BPB 손상이 커졌다. 16K는 8K보다 약 5%p의 latency 여유를
추가하면서 32K보다 quality/head 위험이 작다. 이는 결과에 맞춘 임계값 변경이 아니라, 실패 원인이
step headroom 부족임을 확인한 뒤 기존 Pareto surface에서 한 단계 이동하는 새 가설이다.

최신 선행과의 중복 때문에 `vocabulary expansion` 자체를 신규성으로 주장하지 않는다.
In-Place Tokenizer Expansion은 source BPE의 계속 학습, compositional initialization과 two-stage
adaptation을 이미 제안했고, Beyond Initialization Loss는 input/output 비대칭 subword composition과
짧은 CPT selection을 강한 기준선으로 만들었다. Lifecycle-Optimal Tokenization은 batch-1에서
unembedding 비용을 포함한 vocabulary optimum이 deployment regime에 따라 달라짐을 보였다.
JamoFlow가 추가로 입증해야 할 것은 다음의 좁은 교집합이다.

> compact Korean vocabulary expansion에서 결과를 보지 않고 고정한 new-row AdamW update geometry가
> strong ordinary/two-stage controls보다 raw-byte quality를 회복하고, 더 큰 16K compression의
> 비용까지 포함한 trained controlled/free batch-1 E2E를 모두 10% 이상 줄이는가?

## 수정된 다음 fail-fast

새 disjoint Korean train/calibration stream을 만들고 historical Phase 3, sealed final, fresh-v1의
exact 및 normalized document identity를 모두 제외한다. 동일한 ordered 128MB train과 8MB
calibration에서 다음 다섯 역할을 한 seed로 비교한다.

1. `dense2k_joint_v2`: source checkpoint continuation baseline
2. `dense8k_update_geometry_v2`: 현재 양성 quality recipe의 cross-dataset replication
3. `dense16k_standard_joint`: ordinary all-parameter joint AdamW
4. `dense16k_inplace_two_stage`: published in-place recipe의 compact analogue
5. `dense16k_update_geometry`: **8K에서 이미 고정된 동일 input/output multiplier를 무조정 재사용**

16K의 initialization은 source 2K BPE merge frontier에서 input uniform mean, output
byte-length-weighted mean을 쓰고 input/output을 untie한다. 16K 결과를 보고 multiplier, stage fraction,
LR, data budget 또는 role pool을 바꾸지 않는다.

Quality progression은 다음을 요구한다.

- 16K geometry가 2K와 8K 중 더 낮은 document BPB anchor의 `+0.010 BPB` 이내
- document bootstrap upper도 `+0.010` 이하
- geometry가 16K ordinary와 in-place control을 각각 최소 `0.002 BPB` 이기고 upper `<=0`
- 모든 checkpoint NLL의 independent bitwise replay

Quality가 실패하면 16K actual timing을 열지 않는다. 통과하면 exact selected 16K checkpoint와
같은 run의 2K/8K anchors를 사용해 이번과 동일한 controlled/free joint actual gate를 새 plan으로
봉인한다. 두 mode 모두 10%를 통과한 경우에만 multi-seed/multi-session, 더 큰 model, downstream,
CUDA 및 Hugging Face 공개로 간다.

## 현재 주장 경계

- 말할 수 있음: one seed에서 trained 8K는 같은 raw continuation의 실제 E2E를 20.13% 줄였다.
- 말할 수 있음: free-running point도 8.84% 양수였지만 사전 gate와 uncertainty/stability를 실패했다.
- 말할 수 없음: 8K가 publication-grade inference technique다, 자유 생성에서 안정적으로 10% 이상
  빠르다, parameter/memory efficient하다, 다른 seed/hardware/scale에 일반화된다.
- 다음 16K 실험도 새 quality와 trained actual 결과가 나오기 전에는 opportunity일 뿐이다.

