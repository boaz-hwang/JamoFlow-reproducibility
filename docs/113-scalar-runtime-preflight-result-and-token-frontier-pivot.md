# Scalar runtime preflight result and token-frontier pivot

> 작성일: 2026-08-14
>
> plan commit: `2b03d49295d8c75ef85b3de3127be32de2160cbd`
>
> authoritative aggregate:
> `results/scalar-runtime-preflight-v1/summary.json`

## 1. 판정

두 scalar 후보는 기존 byte W72보다 크게 빨랐지만, parameter-matched BPE16K/32K에 명확하게
뒤졌다. Hangul hybrid도 generic Unicode scalar보다 느렸다. 결과 전에 고정한 모든 training
gate를 적용하면 승인 후보는 **0개**이며 scalar/hybrid branch는 여기서 종료한다.

| candidate comparison | paired median E2E reduction | prompt bootstrap 95% | positive prompts |
|---|---:|---:|---:|
| generic scalar vs byte W72 | +46.843% | [+46.245%, +47.934%] | 32/32 |
| Hangul hybrid vs byte W72 | +42.540% | [+41.477%, +43.517%] | 32/32 |
| generic scalar vs BPE32K | -105.985% | [-115.257%, -93.780%] | 0/32 |
| generic scalar vs BPE16K | -181.243% | [-190.244%, -168.007%] | 0/32 |
| Hangul hybrid vs BPE32K | -122.659% | [-133.801%, -110.179%] | 0/32 |
| Hangul hybrid vs BPE16K | -204.009% | [-215.130%, -190.327%] | 0/32 |
| Hangul hybrid vs generic scalar | -8.095% | [-10.404%, -7.036%] | 0/32 |

양수는 앞 모델의 latency 감소, 음수는 slowdown이다. Repetition을 독립 표본으로 세지 않고
각 document prompt 안에서 세 회 median으로 축약한 뒤 32 prompt를 paired bootstrap했다.

Summary file SHA-256은
`7cf8e90164634f737f3df325c2f1b3bc12e8047f4d3a8c38565b6efa209c4a56`, 내부 canonical
summary SHA-256은
`514806cde290937ccf49c6eb4103369488731fcf9c3c310c41347d442e40d238`이다.

## 2. 실제 graph 결과

전체 repetition을 합친 단순 latency 중앙값은 다음과 같다.

| role | TTFT ms | decode ms | E2E ms | continuation step median |
|---|---:|---:|---:|---:|
| byte W72 | 6.072 | 350.271 | 356.479 | 128 |
| generic scalar | 6.286 | 183.315 | 189.797 | 52 |
| Hangul hybrid | 13.582 | 191.368 | 205.622 | 53 |
| BPE32K | 5.603 | 86.423 | 92.887 | 22 |
| BPE16K | 4.363 | 63.155 | 67.634 | 24 |

Sequential step 수가 실제 decode latency를 강하게 설명한다. Generic scalar는 byte step을
128에서 52로 줄여 W72 latency의 거의 절반을 없앴다. 그러나 BPE는 22--24 token으로 더 짧고
global BLT patch hierarchy 없이 compact token Transformer를 사용해 scalar보다 다시 2--3배
빨랐다.

BPE16K가 BPE32K보다 token은 조금 많지만 더 빨랐다는 점도 중요하다. Total parameter를
맞추면 vocabulary가 작은 16K 모델은 head/embedding 밖의 capacity를 다른 geometry에 배분할
수 있고, 여기서는 9-layer width-320 graph가 13-layer width-256 graph보다 빨랐다. Random
weights이므로 어느 geometry가 같은 quality를 내는지는 아직 모른다. 다만 `token 수만 비교`도,
`total parameter만 비교`도 latency frontier를 충분히 설명하지 않는다는 직접 증거다.

## 3. Correctness와 파라미터

다섯 graph의 full forward, sequential cache, parallel prefill+incremental decode 대조가 모두
통과했다. MPS normalized worst tolerance ratio의 최댓값은 byte W72 0.0517, generic 0.0415,
hybrid 0.0418, BPE32K 0.0139, BPE16K 0.0179로 고정 maximum 1보다 충분히 작았다.

| role | parameters | W72 대비 |
|---|---:|---:|
| byte W72 | 19,596,096 | 0% |
| generic scalar | 19,632,960 | +0.1881% |
| Hangul hybrid | 19,609,152 | +0.0666% |
| BPE32K | 19,593,984 | -0.0108% |
| BPE16K | 19,595,200 | -0.0046% |

따라서 이번 실패를 obvious parameter mismatch나 cache bug로 설명할 수 없다.

## 4. Fable 검토와 기존 가설에 대한 최종 반영

Fable 5가 강조한 `actual speed가 아직 없다`, `rate와 placement를 분리하라`, `scale과 강한
baseline을 확인하라`는 지적은 이 결과에서도 타당했다. 더 나아가 다음을 알게 됐다.

1. W72의 2.5% actual speed가 작았던 직접 원인은 byte local step이었다.
2. 정보를 삭제하지 않고 scalar로 묶으면 그 병목을 실제로 크게 줄일 수 있다.
3. 그러나 그 아이디어는 강한 BPE token graph라는 기존 해법보다 systems frontier에서 멀다.
4. Hangul hybrid는 generic scalar보다 head가 작아도 route/conditional overhead와 1.85% 더 긴
   sequence를 상쇄하지 못했다.

즉 Fable의 분석을 받아들여 BLT 안에서 placement만 더 조정하는 것은 더 이상 합리적이지
않다. 동시에 scalar 결과의 W72 대비 큰 개선만 골라 성공으로 발표하는 것도 잘못이다. 사용자의
기준은 **강한 기준선 대비 matched-quality actual inference efficiency**이기 때문이다.

## 5. 연구 계획 수정

다음 단계에서 하지 않는 것:

- generic scalar 또는 Hangul hybrid one-seed LM 학습
- 더 느슨한 BPE competitiveness threshold로 같은 branch 재해석
- W72만 기준선으로 삼은 42--47% speed claim
- Hangul-specific efficiency claim

다음 단계의 최소 수정은 token-level frontier로 이동하는 것이다.

1. 먼저 byte BPE16K/32K와 Korean-aware reversible tokenizer들의 model-free token count,
   vocabulary allocation, Unicode/OOV robustness를 같은 train/calibration stream에서 비교한다.
2. Korean candidate는 byte BPE보다 sequence가 실제로 짧거나, 같은 길이에서 더 작은
   vocabulary/head를 제공해야만 학습 단계로 간다.
3. 후보에는 generic scalar-aware BPE를 반드시 control로 넣고, Hangul/Jamo 구조가 generic
   Unicode 처리보다 추가 이득을 내는지 분리한다.
4. Token count만 통과시켜서는 안 된다. 이번 결과처럼 vocabulary와 body geometry가 latency를
   바꾸므로, 다음 actual preflight도 total parameter와 적어도 한 body-matched control을 함께
   둔다.
5. 최종 가치는 one-seed quality noninferiority를 통과한 뒤 실제 free-running E2E에서 강한
   BPE baseline을 이겼을 때만 인정한다.

이 피벗은 “한글 규칙을 버린다”가 아니다. 규칙을 Transformer layer skip이나 scalar BLT에
넣는 대신, 이미 가장 빠른 token-level graph의 **가역적 vocabulary/encoding allocation**에
넣어 검증한다. Korean-specific candidate가 generic scalar-aware tokenizer를 이기지 못하면
방법적 기여를 Korean-specific이라고 부르지 않는다.

## 6. Claim 경계

이번 결과는 단일 Apple MPS session, 단일 random seed, 32 calibration-development documents의
controlled fixed-route graph timing이다. 다음만 말할 수 있다.

> Parameter-matched reversible scalar BLTs cut controlled Korean decode latency
> by roughly 43--47% relative to byte W72, confirming that byte-local sequential
> updates were the dominant bottleneck. Yet both scalar variants were more than
> twice as slow as a 32K byte-BPE Transformer and roughly three times as slow as
> a 16K control; the Hangul-specific hybrid was also about 8% slower than the
> generic scalar control. We therefore stopped scalar-model training and moved
> the research frontier to Korean-aware token-level encodings.

Matched quality, free-running generation, memory, multi-seed stability 및 새로운 tokenization의
효율은 아직 증명되지 않았다.
