# EXAONE 7.8B retrieval actual 결과와 core-scale 결정

> 작성일: 2026-08-16
>
> 실제 추론 plan commit: `83bc287`
>
> 다섯 번째 session receipt commit: `5aa9dac`
>
> summary commit: `aff8747`
>
> plan payload SHA-256:
> `a43cdc6fb9502e4ec84cf25e65894b2d5f5991afd0865f0a573085908fab21b0`
>
> summary payload SHA-256:
> `473efd073372f4cf6064e8e4e96e5a1a05caa88eca335514082ede2c110448f2`
>
> 판정: **generic retrieval 7.8B scale transfer 실패; retrieval 및 형태론 확장 종료**

## 1. 결론

고정한 `mlx-community/EXAONE-3.5-7.8B-Instruct-4bit` revision과 Apple M4 Pro에서
train-only corpus n-gram + prompt/self-output hybrid retrieval은 ordinary cached greedy보다
빠르지 않았다. Token ID와 decoded bytes는 exact했지만 실제 end-to-end generation은
오히려 **14.938% 느려졌다**.

| 항목 | ordinary AR | hybrid retrieval | 차이/판정 |
|---|---:|---:|---:|
| cell median E2E | 3.224539 s | 3.706212 s | **-14.938% reduction** |
| crossed session×prompt bootstrap 95% |  |  | **[-17.442%, -11.637%]** |
| faster prompts |  |  | **7/64** |
| positive sessions |  |  | **0/5** |
| target forward-call median | 128 | 95.5 | 25.391% 감소 |
| exact output/correctness | pass | pass | pass |
| memory safety | pass | pass | pass |

사전 고정 primary gate의 correctness만 통과했고, point `>=10%`, bootstrap lower `>0`,
`48/64` prompts faster, 5/5 sessions positive는 모두 실패했다. Summary status는
`fail_generic_retrieval_scale_transfer_actual`이며
`korean_specific_followup_authorized=false`다.

이 결과를 보고 hybrid threshold, table size, draft length, source order 또는 prompt subset을
같은 pool에서 다시 고르지 않는다. `docs/171`과 `docs/176`의 stop rule에 따라 generic
retrieval branch를 종료하고 morphology-normalized retriever도 열지 않는다.

## 2. 증거 무결성

다섯 fresh process session을 순차 실행했다. 각 session은 완료 뒤 metric-free receipt를 별도
Git commit으로 고정한 후에만 다음 session을 열었다.

| session | receipt commit | correctness |
|---:|---|---|
| 0 | `c89971f` | pass |
| 1 | `7443776` | pass |
| 2 | `abe4d66` | pass |
| 3 | `31d34f5` | pass |
| 4 | `5aa9dac` | pass |

공식 summarizer는 성능 통계 전에 다음을 독립 재구성했다.

- 고정 EXAONE checkpoint와 tokenizer, compressed retrieval table, 8 warmup + 64 measured cases
- 1,920 stored measured trials의 target output와 source별 proposal/counter
- ordinary AR와 retrieval candidate의 128-token output ID 및 decoded UTF-8 bytes exact equality
- 다섯 warmup session root
- session/receipt Git chronology, environment와 memory-safety contract

Summary를 commit한 뒤 public read-only verifier를 별도 process에서 다시 실행했다.

```text
summary_full_forward_verification=pass
summary_sha256=473efd073372f4cf6064e8e4e96e5a1a05caa88eca335514082ede2c110448f2
```

따라서 실패는 output divergence, 잘못된 cache, stale table, invalid counter 또는 한 세션의
일시적 변동으로 설명되지 않는다.

## 3. 왜 forward call 감소가 wall time 악화로 뒤집혔는가

핵심은 **호출 수와 target이 실제 계산한 token-position 수가 다르다**는 점이다.

다섯 session × 64 prompts × 3 repetitions, 총 960 measured trial에서 ordinary AR은
매 output마다 한 position을 계산했다.

```text
ordinary target positions = 960 × 128 = 122,880
```

Hybrid는 proposal cycle마다 `last_token + proposed_tokens` 전체를 target block으로 계산했다.

```text
candidate target calls     = 36,210 no-proposal
                           + 7,545 corpus proposal
                           + 45,345 prompt proposal
                           = 89,100

candidate proposed tokens  = 9,090 corpus + 134,640 prompt
                           = 143,730

candidate target positions = 89,100 + 143,730
                           = 232,830
```

즉 호출 수는 평균 27.49% 줄었지만 target token-position은 **89.48% 증가**했다. Block forward의
병렬성이 이 추가 계산 대부분을 숨겼기 때문에 wall time은 89%가 아니라 14.9%만 악화됐지만,
대형 target에서는 작은 모델에서처럼 kernel-launch 절감이 이기지 못했다.

Proposal 143,730개 중 accepted draft token은 34,005개뿐이었다.

| source | calls | proposed | accepted | token acceptance |
|---|---:|---:|---:|---:|
| corpus n-gram | 7,545 | 9,090 | 7,890 | **86.799%** |
| prompt/self-output | 45,345 | 134,640 | 26,115 | **19.396%** |
| total | 52,890 | 143,730 | 34,005 | **23.659%** |

Prompt fallback이 proposal token의 93.68%를 만들었지만 acceptance는 19.40%였다. 이 낮은
precision block work가 large target에서 주된 손실이다. Corpus source는 정확하지만 coverage와
span이 짧다. 현재 hybrid trace만으로 corpus-only의 실제 경로를 정확히 재구성할 수 없으며,
같은 measured pool 결과를 보고 source를 바꾸는 것은 새 exploratory candidate 선택이다.
Generic retrieval의 선행연구 중복과 현재 effect margin을 고려하면 이를 publication primary로
계속 최적화하지 않는다.

## 4. 작은 모델 결과와의 scale inversion

Fresh-v2 16K target에서는 같은 계열 hybrid가 free-running E2E를 26.244% 줄였다. 당시
baseline target-call median은 32.5, candidate는 22였고 lookup overhead는 약 0.5ms였다.
반면 현재 7.8B target에서는 block 내부의 rejected token compute가 훨씬 비싸다.

따라서 다음 넓은 주장은 기각된다.

- 작은 target에서 call count가 줄면 큰 target에서도 빨라진다.
- acceptance scalar 하나만으로 speculative latency를 예측할 수 있다.
- launch-bound compact model의 positive retrieval 결과를 8B compute-bound target에 외삽할 수 있다.

현재 결과가 지지하는 더 정확한 systems 명제는 다음이다.

> Retrieval speculation의 scale transfer는 accepted tokens/cycle뿐 아니라 target이 평가한 전체
> draft token-position, block-length별 target cost, correction/bonus와 no-proposal coverage를 함께
> 회계해야 한다. Small-model launch amortization이 large-model wall-clock speedup을 보장하지 않는다.

이 명제는 유용한 음성 결과이지만, 사용자가 정한 `실제 추론 개선` 기준을 충족한 새 기법은 아니다.

## 5. Claim boundary

이 결과는 다음 범위에만 해당한다.

- 한 EXAONE 3.5 7.8B 4-bit revision
- 한 Apple M4 Pro / MLX 환경
- 64개의 Hangul-heavy public raw-completion prompt
- 정확히 128 output tokens, EOS 무시
- historically used evaluation pool에서의 exploratory scale-transfer timing
- compatibility model-output hash가 rank seed에 들어간 case set

Actual retrieval candidate output, acceptance, latency는 plan 전에 보지 않았고 actual timing plan은
prospectively Git-sealed했다. 그러나 case selection은 model-output blind가 아니며 이 결과를
untouched final, public preregistration, generic hardware 또는 chat-serving 결과로 부르지 않는다.

## 6. 연구 방향 수정

### 종료

1. 동일 pool에서 corpus-only/prompt-only/threshold/draft-length를 사후 winner로 고르지 않는다.
2. Generic retrieval의 한국어 morphology 변형을 열지 않는다.
3. Qwen fallback이나 다른 7--8B checkpoint를 성능 결과를 보고 교체하지 않는다.
4. Forward-call 감소를 실제 효율 개선으로 표현하지 않는다.

### 보존

Core W72 연구는 별도다. 19.6M matched-quality 다섯-seed 결과에서 W72는 C86보다 controlled
2.628%, free 2.531% 실제 E2E를 일관되게 줄였다. 이는 작지만 **양의 실제 추론 개선**이며,
현재 paper의 정직한 중심 증거다. Retrieval 실패가 이 결과를 무효화하지 않는다.

### 다음 한 번의 저비용 결정 실험

대규모 training을 곧바로 시작하지 않는다. 먼저 기존 봉인 family geometry 50M/75M/100M에서
같은 random weights와 같은 controlled Korean byte continuation을 사용해 W72와 C86 schedule만
교차하는 **scale-sensitivity actual runtime preflight**를 새 namespace에 봉인한다.

이 preflight의 목적은 quality를 주장하는 것이 아니라, global trunk가 커질 때 W72의 16.3%
patch-event 감소가 실제 E2E 10% margin으로 커질 가능성이 있는지 판정하는 것이다.

- 50M, 75M, 100M을 결과와 무관한 순서로 모두 실행
- target별 같은 weight, 같은 bytes, schedule만 W72/C86
- cached incremental path와 각 schedule의 full/parallel oracle 검증
- 16 measured prompts, 3 inner repetitions, balanced order
- 100M primary: median controlled E2E reduction `>=10%`, prompt bootstrap lower `>=8%`,
  `15/16` prompts faster
- 실패 시 publication-scale W72/C86 training 종료
- 통과 시에만 100M one-seed matched-quality training/actual feasibility를 별도 봉인

이 단계는 7.8B retrieval 실패를 구제하지 않는다. 이미 실제 양성인 core W72가 larger
byte-latent geometry에서 논문 가치가 커질 여지가 있는지만, 비싼 training 전에 실제 runtime으로
검사하는 최소 실험이다.

## 7. 논문에 미치는 영향

현재 paper의 허용 결론은 바뀌지 않는다.

1. Korean same-rate boundary placement는 quality에 영향을 준다.
2. W72는 C86 대비 작지만 재현 가능한 matched-quality actual speedup을 보인다.
3. 그 효과는 사전 10% 목표를 실패하므로 강한 efficiency technique claim은 불가하다.
4. Generic retrieval은 16K development target에서 양성이었지만 7.8B EXAONE에서 명확히
   느려져 scale-transfer 기법이 되지 못했다.

Scale-sensitivity preflight까지 실패하면 현재 연구는 이 empirical result와 negative systems
analysis를 논문 초안의 최종 범위로 삼고, 더 큰 training·CUDA·Hugging Face model release는
열지 않는다. 코드·plan·summary와 compact checkpoints의 reproducibility release만 준비한다.
