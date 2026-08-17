# Trained static-geometry one-seed screen

> 작성일: 2026-08-13
>
> 상태: **candidate training 전 봉인 예정**
>
> protocol: `jamoflow-static-geometry-one-seed-v1`

## 1. 목적과 범위

Random-weight preflight에서 사전 고정 gate를 통과한
`thin160_e1_d1_g384x9`가 실제 Korean language modeling quality를 유지하는지 한 seed로
screen한다. 이 architecture는 BLT의 알려진 local/global allocation axis를 재현하는
generic static control이며 JamoFlow의 novelty가 아니다.

이 단계의 질문은 하나다.

> Original W72와 같은 Korean data, W72 patch matrix, seed, batch order와 optimizer로
> 처음부터 학습했을 때 calibration quality가 0.010 BPB margin 안에 있고, 학습된 실제
> checkpoint의 controlled/free E2E latency가 모두 최소 15% 줄어드는가?

한 seed 결과는 publication claim이 아니며 후속 multi-seed replication과 orthographic
conditional-compute 연구에 비용을 쓸지 결정하는 screen이다.

## 2. 고정 비교

| 항목 | candidate | baseline |
|---|---|---|
| seed | 1729 | 1729 |
| patch policy | W72 | W72 |
| local width | 160 | 192 |
| local encoder/decoder | 1 / 1 | 2 / 2 |
| global width/layers/FFN | 384 / 9 / 1128 | 384 / 8 / 1152 |
| parameters | 19,571,872 | 19,596,096 |
| train stream | 같은 128,000,000 Korean bytes | 기존 봉인 checkpoint |
| optimizer/order | Phase 3 exact spec / seed-1729 permutation | 동일 |

Candidate는 baseline checkpoint를 이식하지 않고 같은 seed로 새 graph를 초기화해 처음부터
학습한다. Baseline은 기존 compute-conversion W72 seed-1729 checkpoint, report와
per-sequence calibration NLL을 exact artifact/state hash로 고정한다.

Test, sealed final, historical test NLL, downstream와 기존 final timing 수치는 읽지 않는다.
기존 primary summary는 source/stream authority에만 사용한다.

## 3. 품질 screen

- evaluation split: 8,000,000-byte Korean calibration stream
- sequences: 15,625 x 512 bytes
- targets: sequence마다 511 bytes
- paired estimand: candidate minus baseline per-sequence NLL
- margin: 0.010 BPB
- document map: 386 documents, 15,240/15,625 eligible windows (97.536%)
- bootstrap: whole document를 target-byte weighting으로 10,000회 resample
- seed: 20,261,001

Quality pass는 전체-stream mean difference <= 0.010 BPB, document bootstrap one-sided
95% upper <= 0.010 BPB, eligible-window coverage >= 95%를 모두 요구한다. 하나라도 실패하면
정적 control은 품질 비보존으로 판정한다.

## 4. 학습 checkpoint actual timing

Prompt와 controlled continuation은 model-free calibration bottom-hash order의 첫 64개를
학습 전에 고정한다.

- prompt 128 bytes, minimum output 128 bytes
- controlled replay: 같은 source continuation의 127 feedback forwards
- free running: strict RFC 3629 mask를 포함한 greedy output, 첫 >=128-byte scalar boundary에서
  stop, 최대 131 bytes
- warmup: 8 prompts
- repetitions: prompt/mode/role마다 5회, repetition은 prompt median으로 접음
- role order: prompt x repetition x mode 안에서 exact balance
- timing scope: fresh incremental runtime, parallel prefill, mask/argmax/DFA/feedback,
  two MPS synchronizations
- AC power 및 shared publication MPS lock 필수

Sequential-prefill runtime과 parallel-prefill runtime은 두 role 각각 첫 8 prompt의
1,024 logit 위치, argmax, boundary/cache trace로 대조한다. 모든 free output은 strict UTF-8로
재검증하고 repetitions 사이 byte-exact deterministic이어야 한다.

각 controlled/free mode가 모두 다음을 만족해야 actual-latency pass다.

1. E2E point reduction >= 15%
2. paired prompt bootstrap 95% lower >= 10%
3. positive prompts >= 48/64
4. correctness와 strict-output gate 전부 pass

## 5. 결정 규칙

Quality와 두 actual mode가 모두 pass할 때만 다음을 허가한다.

1. static control의 나머지 네 seed replication
2. 같은 backbone을 쓰는 generic UTF-8 state-conditioned control 구현
3. 동일 평균 compute의 Hangul-specific conditional local-depth candidate 구현

Quality가 실패하면 static thinning은 최종 control로 쓰지 않는다. 다만 easy orthographic
state에서만 얕은 path를 쓰고 hard state에서는 original local depth를 보존하는 conditional
candidate는 별도 가설로 남는다. 속도가 실패하면 이 MPS graph에서 geometry allocation
branch를 종료한다. 어느 경우에도 결과를 보고 margin, prompt, geometry 또는 seed를 바꾸지
않는다.

## 6. Claim 경계

Pass는 한 seed calibration screen일 뿐이다. 정적 geometry의 새 방법 claim, 일반 hardware
속도, held-out quality, downstream 또는 publication-scale 효율을 뜻하지 않는다. 최종 성공은
다중 seed matched quality, generic UTF-8 대비 Hangul-specific 효과, actual generation과
후속 hardware replication을 모두 요구한다.
