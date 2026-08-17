# 동일 2K 예산 compositional vocabulary systems preflight 결과

> 작성일: 2026-08-14
>
> 상태: sealed random-weight systems opportunity pass; trained quality 미검증

## 결론

사전 고정한 13-role gate에서 **8K generic code와 8K Hangul code가 모두 통과**했다. 따라서 가장
작은 허용 vocabulary인 8K를 one-seed 학습 단계로 올린다.

8K Hangul code는 BPE-2K dense baseline과 trainable parameter가 정확히 같은 19,667,328개이면서,
같은 128-byte continuation의 token step을 22.98% 줄였고 controlled end-to-end median을 19.56%
줄였다. Paired-prompt bootstrap 95% lower bound는 16.88%였고, 36 prompt 중 34개에서 빨랐다.
Generic code도 각각 19.00%, 16.39%, 35/36으로 같은 gate를 통과했다.

이 결과는 연구를 계속할 충분한 systems 여유를 보여 주지만, compositional code의 독자적 speed
우위를 보여 주지는 않는다. 같은 8K tokenizer의 dense와 low-rank control도 각각 19.81%,
19.27% 빨랐다. 현재 효과의 주원인은 **8K tokenization의 짧은 autoregressive sequence**이며,
codebook의 역할은 그 이득을 2K와 같은 trainable head budget에서 유지하는 것이다. 다음 단계의
핵심 질문은 속도가 아니라 이 압축된 head가 품질을 보존하는가, 그리고 Hangul assignment가
generic/shuffled assignment보다 실제로 더 나은가이다.

## 봉인된 결과

모든 수치는 같은 session, 같은 36 independent-document prompt, prompt별 3 repetitions의 median을
사용한다. Tokenizer 시간은 model timer 밖이며, continuation은 동일 raw bytes를 보장하기 위한
gold-token controlled replay다.

| role | params | E2E median | BPE-2K 대비 | 95% lower | 빠른 prompts | continuation steps | step 감소 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense 2K | 19,667,328 | 79.002 ms | 기준 | — | — | 1,288 | 기준 |
| dense 8K | 22,026,624 | 63.354 ms | 19.806% | 17.162% | 35/36 | 992 | 22.981% |
| low-rank 8K | 19,669,888 | 63.776 ms | 19.273% | 16.171% | 35/36 | 992 | 22.981% |
| generic code 8K | 19,667,328 | 63.988 ms | 19.005% | 16.388% | 35/36 | 992 | 22.981% |
| **Hangul code 8K** | **19,667,328** | **63.548 ms** | **19.561%** | **16.883%** | **34/36** | **992** | **22.981%** |
| dense 16K | 25,024,896 | 59.283 ms | 24.960% | 21.696% | 36/36 | 894 | 30.590% |
| Hangul code 16K | 19,667,328 | 58.990 ms | 25.330% | 22.524% | 36/36 | 894 | 30.590% |
| dense 32K | 31,168,896 | 60.037 ms | 24.005% | 20.552% | 34/36 | 814 | 36.801% |
| Hangul code 32K | 19,667,328 | 58.162 ms | 26.379% | 22.696% | 36/36 | 814 | 36.801% |

16K/32K의 generic와 low-rank도 모두 10% gate를 통과했다. 그러나 결과 전에 고정한 규칙은
generic/Hangul이 공동 통과한 가장 작은 size를 택하도록 했다. 이전 one-seed BPE frontier에서
vocabulary가 커질수록 128M-byte 학습 품질이 단조 악화됐기 때문에, 더 빠른 16K/32K를 결과를 본
뒤 선택하는 것은 품질 위험과 선택 편향을 키운다. 8K 선택을 유지한다.

## 무엇을 배웠는가

### 1. 10% 이상의 실제 decode 여유는 존재한다

Same-2K LengthGain의 최적 분할은 continuation step을 5.67%밖에 줄이지 못했다. 반면 이미 학습된
8K BPE tokenizer는 같은 cases에서 22.98%를 줄였고, codebook gather overhead를 포함해도 E2E
19.56%가 남았다. 따라서 연구 방향을 same-2K tokenizer 최적화에서 large-vocabulary
constant-budget head로 바꾼 결정은 systems 관점에서 타당했다.

### 2. Codebook은 현재 speed mechanism이 아니라 capacity-preserving mechanism이다

8K dense, low-rank, generic code, Hangul code의 E2E는 63.35--63.99 ms로 매우 가깝다. 이
preflight에는 이들 사이의 superiority test도 고정하지 않았다. 그러므로 `Hangul code가 dense나
low-rank보다 빠르다`고 주장할 수 없다.

Codebook의 확인된 장점은 8K dense head의 추가 2,359,296 trainable parameter를 쓰지 않고도 같은
short sequence를 실행한다는 것이다. 반대로 assignment buffer는 숨은 비용이다. 8K code role의
runtime buffer는 1,048,832 bytes로, baseline/dense/low-rank의 256 bytes보다 약 1 MiB 크다. 32K는
약 4 MiB다. 후속 memory 표는 trainable parameters와 nontrainable assignment bytes를 분리해
항상 함께 공개해야 한다.

### 3. Per-step cost는 공짜가 아니다

Median decode milliseconds per continuation step은 2K dense 2.150, 8K Hangul code 2.234였다.
즉 codebook의 per-step overhead가 약 3.9% 있었지만 22.98% 적은 step이 이를 상쇄했다. TTFT도
3.368 ms에서 3.488 ms로 개선되지 않았다. 최종 주장은 긴-enough generation의 E2E에 한정하고,
짧은 요청이나 TTFT 개선을 과장하면 안 된다.

### 4. Hangul assignment의 가치는 아직 전혀 입증되지 않았다

Random initialization에서 generic/Hangul code의 graph, parameter 수, token steps는 같다. 두 role의
작은 timing 차이는 linguistic prior의 증거가 아니다. Hangul contribution은 같은 8K tokenizer,
same body, same training order에서 다음 세 역할의 raw-byte BPB와 downstream Korean behavior로만
판단한다.

- generic Unicode-surface code
- byte-length-stratified shuffled-Hangul code
- true Hangul onset/vowel/coda code

True Hangul이 generic와 shuffled 모두를 넘지 못하면 연구 결과는 `한국어 compositional prior`가
아니라 standard low-rank/factorized vocabulary의 systems tradeoff로 축소된다. Low-rank가 같은
품질을 보존하면 더 단순한 표준 control이 우선이며, JamoFlow의 한국어 특화 논문 축은 중단한다.

## Correctness와 artifact

13 role의 6 warmup case에서 full no-cache, one-token sequential cache, parallel prefill cache를
비교했다. 총 5,987 logit position에서 argmax가 모두 같았고, 가장 큰 normalized tolerance ratio는
0.02779로 고정 한계 1보다 충분히 작았다. Continuation step 배열도 tokenizer에서 독립 재구성한
값과 exact 일치했다.

- v2 plan payload SHA-256: `4bda6b70c1e5d01172571e48125957f7f5b1b1a2fa8a212a288fc5a8044ff563`
- v2 plan file SHA-256: `f4f0f0da980acc3069fff3e0dbd75dfee7b18287cd8612b20dab25d7fff8b6d4`
- report file SHA-256: `a836113b940515cd1d43106b78849ebe592d6c13464f36e043657bf962c7cb89`
- timing file SHA-256: `54cd9bf38080603cf7f2b0ecceffdbb2a98e8b8d221bf80b56356f8078d35f99`
- result payload SHA-256: `e18c017fd27dff49ce47148d432fd8f52bc3cfe2af0aca76d8965fcd828aa091`
- result file SHA-256: `5cbbf2b39af0b8efda52e04d6a8fc2b8bee58983892e224ac0afbc2d9f75c72d`

V1 plan은 audit tuple/list JSON round-trip 오류로 model construction 전에 fail-closed됐다. Timing과
result artifact는 생성되지 않았고, role/case/gate를 바꾸지 않은 v2가 위 결과의 유일한 evidence다.

## 다음 단계

8K의 여섯 역할을 동일 128M Korean raw-byte stream에서 one seed로 학습한다.

1. dense 2K matched-total baseline
2. dense 8K same-body quality ceiling (+2.36M head parameters)
3. low-rank 8K near-matched-total control
4. generic code 8K
5. shuffled-Hangul code 8K
6. Hangul code 8K

먼저 result-free train/evaluation memory·time preflight로 microbatch와 24-hour campaign feasibility를
고정한다. 그 다음 one-seed 학습과 independent checkpoint replay를 수행한다. Quality gate는 2K
baseline 대비 raw-byte BPB noninferiority와 8K controls 사이의 사전 고정 contrast를 함께 사용한다.
이 단계에서 positive하면 그때만 trained actual-inference를 측정한다. Random-weight 19.56%를 trained
model의 실제 10% 개선으로 간주하지 않는다.
