# Vocabulary-transfer probe 결과와 strong-baseline closure

> 작성일: 2026-08-15
>
> 상태: one-seed development probe 통과; Korean method·actual-efficiency 주장은 아직 없음

## 결론

Trained dense BPE-2K checkpoint의 body와 기존 lexical rows를 보존하고, continued-BPE의 merge
tree를 source vocabulary frontier에서 잘라 새 8K rows를 초기화하는 경로는 살아 있다. 사전 고정한
512-step joint gate를 통과한 역할은 두 untied composition 역할이었다.

- 최선 `untied_uniform_in_byte_weighted_out`: `1.465715 BPB`
- 같은 graph의 `untied_random_norm`: `1.556707 BPB`
- composition advantage: `0.090992 BPB`
- dense-2K anchor `1.429662`와의 gap: `+0.036053 BPB`
- 사전 기준: advantage `>=0.010`, anchor gap `<=+0.050`

7개 역할의 4개 checkpoint, 총 28개 state를 새 모델에 strict-load하고 8M-byte calibration 전 구간을 다시
forward한 NLL은 저장 float32 배열과 모두 bitwise 동일했다. 따라서 positive 판정은 worker가
기록한 scalar가 아니라 독립 재계산본에 근거한다.

이 결과가 말하는 범위는 좁다. **Generic vocabulary transfer가 현재 compact Korean setting에서
large-vocabulary cold start를 크게 줄였다.** Jamo, 형태론 또는 한국어 규칙은 어느 역할에도
들어가지 않았으므로 한국어 고유 기법의 증거가 아니며, 실제 생성이 빨라졌다는 증거도 아니다.

## 전체 recovery curve

| role | params | step 0 | step 32 | step 128 | step 512 | anchor gap |
|---|---:|---:|---:|---:|---:|---:|
| tied random+mean norm | 22,026,624 | 2.21802 | 2.03674 | 1.79641 | 1.60024 | +0.17058 |
| tied uniform+mean norm | 22,026,624 | 2.20698 | 1.82442 | 1.58611 | 1.48685 | +0.05719 |
| tied byte-weighted+mean norm | 22,026,624 | 2.23469 | 1.82487 | 1.58479 | 1.48482 | +0.05516 |
| tied last subpiece | 22,026,624 | 2.51099 | 2.00808 | 1.69315 | 1.53215 | +0.10249 |
| untied random+mean norm | 25,172,352 | 2.21896 | 2.01440 | 1.74591 | 1.55671 | +0.12705 |
| untied uniform input / uniform output | 25,172,352 | 2.04555 | 1.76308 | 1.56793 | 1.46780 | +0.03814 |
| **untied uniform input / byte-weighted output** | **25,172,352** | **2.03072** | **1.75156** | **1.56365** | **1.46572** | **+0.03605** |

Tied composition도 random control을 `0.11339`--`0.11542 BPB` 이겼지만 anchor gap이 각각
`+0.05719`, `+0.05516`이라 고정 gate를 `0.00719`, `0.00516 BPB`만큼 실패했다. Threshold를
결과에 맞춰 완화하지 않는다. 다만 추가 lexical storage가 없는 tied route가 아주 근접했다는
사실은 다음 strong-baseline closure에서 tied deployment branch를 유지할 충분한 이유다.

## 이 결과에서 새로 확인된 것

### 1. 초기 loss만으로 initializer를 고르면 틀릴 수 있다

Tied byte-weighted role은 step 0에서 random보다 `0.01667 BPB` 나빴다. 그러나 step 512에서는
오히려 `0.11542 BPB` 앞서 tied 역할 중 최선이 됐다. 이는
[Beyond Initialization Loss](https://arxiv.org/abs/2608.03494)의 핵심 관찰, 즉 initialization
BPB보다 짧은 CPT가 수렴 전략을 더 잘 고른다는 결과와 같은 방향이다. 이후 선택은 step 0이
아니라 사전 고정한 short-CPT checkpoint에서만 한다.

### 2. Untying은 품질과 capacity를 함께 바꾼다

Untied 8K는 tied 8K보다 input/output을 공유하지 않아 `3,145,728` parameters, 즉 tied target
대비 약 14.28%가 더 많다. Logit vocabulary geometry는 같지만 resident weight bytes와 lexical
capacity는 같지 않다. 따라서 untied의 `0.01910 BPB` 최종 우위를 initializer 효과로만 부를 수
없다. 모든 후속 비교는 architecture 안 random control을 유지하고, tied와 untied를 하나의
무차별 순위로 합치지 않는다.

### 3. Pure codebook 실패 원인은 transfer로 상당 부분 우회된다

이전 direct dense-8K는 `1.51963 contiguous BPB`, Hangul codebook은 `1.63792`였다. 이번
transfer best는 동일 8K tokenizer에서 `1.46572`까지 회복했다. 이는 8K vocabulary 자체가
근본적으로 불가능했던 것이 아니라, source lexical knowledge를 버린 one-pass cold start와 pure
shared-code constraint가 주 병목이었다는 설명을 강화한다.

### 4. 아직 publication result가 아닌 이유

- model seed 하나와 이미 알려진 calibration stream을 사용했다.
- source dense-2K가 학습한 first 128M train bytes를 short CPT에서 다시 사용했다.
- 최선 역할은 최신 initializer를 정확히 재현한 것이 아니라 근사했다.
- dense-2K와 direct-8K에 같은 추가 history를 주는 full-CPT 비교가 아니다.
- trained checkpoint의 동일-output batch-1 wall-clock을 측정하지 않았다.

따라서 이 결과는 expensive next step을 여는 development gate이지 논문의 효율 결론이 아니다.

## 사후 문헌 감사: 현재 역할은 strongest baseline의 정확한 재현이 아니다

[Beyond Initialization Loss](https://arxiv.org/abs/2608.03494)는 2026년 8월 4일 공개된
20개 이상 vocabulary-extension initializer의 체계적 비교다. 최선 설정은 untied input에
source-subword uniform mean과 **target-script token row L2의 median norm**, output에
**decoded Unicode code-point 길이 가중 mean**, 그리고 output norm 없음이다. 현재 probe는
input에 전체 source row의 **mean norm**, output에 constituent raw-byte 길이를 썼다.

[In-Place Tokenizer Expansion](https://arxiv.org/abs/2607.15232)은 continued BPE, carried-row
copy와 새 row source-subtoken mean을 사용하고, 새 embedding row만 학습한 뒤 full model CPT로
넘어가는 two-stage recipe를 제시한다. 현재 probe는 모든 parameter를 첫 step부터 학습했으므로
이 baseline도 아직 닫지 못했다.

[EEVE](https://arxiv.org/abs/2402.14714)는 Korean vocabulary expansion에서 input new row를
source subword mean, output new row를 첫 source subword로 초기화하고 여러 parameter-freezing
stage를 사용했다. 현재 `last_subpiece`는 첫 subword가 아니라 마지막을 복사했고 tied graph라
EEVE initializer의 정확한 대조군도 아니다.

또 [Thunder-Tok](https://arxiv.org/abs/2506.15138)은 Korean fertility를 약 9% 줄이면서 성능을
유지하는 tokenizer와 실제 target-sentence generation time을 이미 보고했다. 따라서
“Korean tokenizer가 token 수와 시간을 줄인다” 또는 “continued-BPE transfer가 된다”만으로는
신규성이 없다.

## 필요한 계획 수정

기존 protocol은 joint gate 통과 시 곧바로 equal-history full CPT를 허용했다. Gate 자체는
통과했지만, 위 최신 문헌으로 인해 full CPT 전에 **strong generic baseline closure**를 한 단계
삽입한다. 이는 결과에 맞춘 threshold 변경이 아니라, 약한 baseline 위에서 Korean 기여를
과장하지 않기 위한 보강이다.

### B0 — 9-role strong-baseline closure

모든 역할은 같은 source checkpoint, 8K tokenizer, target token sequence, order, 512-update
budget과 checkpoint grid `0/32/50/128/307/512`를 사용한다. Step 50은 BIL의 lightweight-CPT
selection claim을 직접 확인하고, step 307은 two-stage 경계다.

Untied/BIL branch:

1. `untied_random_hangul_median_input_native_output`
2. `untied_bil_hangul_median_char_out`
3. `untied_bil_global_median_char_out`
4. `untied_bil_hangul_median_uniform_out`
5. `untied_eeve_uniform_in_first_out`

Tied/continued-BPE branch:

6. `tied_random_native_all`
7. `tied_uniform_no_norm_all`
8. `tied_random_native_two_stage`
9. `tied_uniform_no_norm_two_stage`

여기서 BIL exact 역할은 tokenizer의 `decode([source_id])` 결과 Unicode code-point 수를
`max(length,1)`로 사용한다. Hangul norm subset은 decoded source token에 U+1100--11FF,
U+3130--318F, U+A960--A97F, U+AC00--D7A3 또는 U+D7B0--D7FF가 하나라도 포함된 row로
고정한다. Byte-BPE의 incomplete piece는 decode 시 U+FFFD 한 code point가 될 수 있으며, 이는
논문의 decode 정의를 따른 결과이자 별도 limitation으로 공개한다.

`two_stage`는 512 steps를 307+205로 나눈 compact 60:40 **비율 analogue**다. Stage 1은 body와
기존 2,048 rows를 bitwise 고정하고 새 6,144 tied rows만 갱신한다. AdamW decay가 frozen row를
바꾸는 우회를 막기 위해 매 step old rows exact equality를 검사한다. Stage 2 시작 시 optimizer를
재생성하고 모든 parameters를 연다. 논문의 600B+400B token scale을 재현했다고 부르지 않는다.

EEVE role은 initializer를 정확히 맞추되, 원 논문의 convergence-driven seven-stage schedule을
512 fixed budget 안에서 재현하지 않으므로 **EEVE initializer analogue**라고만 부른다.

각 composed role은 같은 architecture와 schedule의 random control보다 step 512에서
`>=0.010 BPB` 좋아야 하고 dense-2K anchor gap이 `<=+0.050`이어야 한다. Step-0/50을 보고
fallback하거나 role을 추가하지 않는다. Tied와 untied가 모두 통과하면 다음 단계에 lowest-BPB
untied와 qualified tied Pareto role을 함께 보존한다.

### B1 — foldable Korean contribution

B0가 통과할 때만 training-time Jamo shared residual/reparameterization을 연다. 목표는 SCRIPT나
KOMBO처럼 online Jamo channel을 더하는 것이 아니라, 학습 후 ordinary dense 8K rows로 **fold**해
배포 tokenizer·parameter count·FLOPs를 generic role과 정확히 같게 만드는 것이다.

필수 대조군은 Jamo feature 수, token byte-length, 노출 빈도와 residual cardinality를 맞춘 shuffled
assignment다. Korean 역할은 same-graph strong generic initializer와 shuffled control을 모두
이기고, 그 차이가 추가 training compute를 정당화해야 한다. 그렇지 않으면 Korean branch를
종료한다.

### B2 — fresh-data equal-history full CPT

Dense-2K continuation, direct dense-8K continuation, architecture-matched random transfer, strong
generic transfer와 qualified foldable-Jamo role에 동일한 **새 continuation raw-byte budget**을
준다. 가능한 경우 source checkpoint가 보지 않은 disjoint train documents를 사용한다. 기존
128M 반복 결과는 이 단계의 final quality evidence로 재사용하지 않는다.

### B3 — 실제 효율 gate

품질 noninferiority를 통과한 exact checkpoint만 같은 prompt와 같은 raw output bytes의
controlled replay 및 자체 strict-valid free-running batch-1 generation으로 비교한다. Token count,
fertility 또는 synthesized time을 actual latency로 대신하지 않는다. End-to-end 10% 개선과 CI
gate를 통과한 뒤에만 multi-seed, 더 큰 Mac-feasible scale, Korean downstream, Hugging Face 공개와
논문 positive claim으로 확장한다.

## 신규성의 현재 경계

Generic continued-BPE, merge-tree initialization, input/output 비대칭, freezing curriculum,
Korean vocabulary expansion과 fertility 감소는 모두 선행연구가 있다. 현재 가능한 기여는 다음
교집합으로 좁혀야 한다.

> Strong generic tokenizer-expansion baselines와 동일한 deployed dense graph 아래에서,
> training-only Korean orthographic sharing이 adaptation compute/quality를 추가 개선하고, 그 결과
> matched-quality 실제 batch-1 Korean generation을 유의미하게 빠르게 만드는가?

이 질문에 음성이면 vocabulary-transfer engineering 결과는 공개할 수 있어도 사용자가 정한
“실제 추론 효율이 개선된 논문”의 성공으로 간주하지 않는다.

## Artifacts

- sealed protocol: `docs/136-vocabulary-transfer-probe-protocol.md`
- plan: `data/manifests/vocabulary-transfer-probe-v1.json`
- tracked summary: `results/vocabulary-transfer-probe-v1/summary.json`
- ignored checkpoints/NLL/worker receipts: `artifacts/vocabulary-transfer-probe-v1/`
- implementation commit: `fb474cb`
- plan commit: `824d451`
- result commit: `a446364`
- summary file SHA-256: `7bb6a596e11cead7ce3fdbe906804faa0fc26d10b81a8c6a8f7471b71fddeb5e`
- canonical summary payload SHA-256: `e33a25baf536ddcfb169538e5b985eb5fbf4e2b95d9cb2ad8ec81d63ce818dd7`
