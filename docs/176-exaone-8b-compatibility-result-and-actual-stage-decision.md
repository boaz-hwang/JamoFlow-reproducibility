# EXAONE 8B compatibility result and actual-stage decision

> 작성일: 2026-08-15
>
> V4 plan commit: `cde4914cf94c54468186a9387aaaefed9b111671`
>
> V4 plan SHA-256: `47f45d390fca779554c46e1b6e7fcac1d11fc359b74cc486667adb84bf1c2b0b`
>
> V4 plan artifact SHA-256:
> `3e7a73aa30109127524e24a2a8416f56cf725f886d2a585dd0f9488feaaac8fc`
>
> Result commit: `fe37bf5a00286b7f8b0c3ef620f4057440652c04`
>
> Result summary SHA-256: `b88595d1046ea5e89882664bb892af0e220a0eee661e337a547d56d4877d1a28`
>
> Result artifact SHA-256:
> `670fed1737cd439413c162c3611075206c04f54a29e84e59892bd209d8f975af`
>
> 판정: **EXAONE 3.5 7.8B 4-bit greedy transaction compatibility pass; actual efficiency untested**

## 결과

고정한 `mlx-community/EXAONE-3.5-7.8B-Instruct-4bit` revision에서 V4 compatibility gate를
통과했다.

| 항목 | 결과 |
|---|---:|
| model parameters | 7,818,448,896 |
| compatibility prompt tokens | 13 |
| full/cache greedy-decision positions | 28 |
| argmax exact | 28 / 28 |
| finite logits | pass |
| all-logit numeric diagnostic | fail, descriptive only |
| maximum absolute logit error | 0.080078125 |
| maximum normalized logit error | 1.3894497156143188 |
| rollback argmax | exact |
| rollback maximum normalized error | 0.8595988750457764 |
| repeated 16-token cached greedy | exact |
| forced full-accept cycles | 4 |
| forced immediate-reject cycles | 16 |
| forced partial-accept cycles | 8 |
| forced output vs ordinary greedy | all exact |
| MLX peak allocated bytes | 4,580,557,168 |
| recommended working-set fraction | 0.1139416671 |
| memory safety | pass |

Generated text와 token IDs는 result에 저장하지 않았고 domain-separated sequence hash만 봉인했다.
Candidate-vs-baseline lookup, acceptance, latency, throughput은 실행하지 않았다.

## 해석

이 결과가 증명하는 것은 좁다. 현재 Mac/MLX에서 고정 EXAONE target의 ordinary cached greedy와
forced speculative verification/rollback을 같은 token sequence로 구현할 수 있고, 실제 7.8B model을
메모리 안전 범위에서 실행할 수 있다는 것이다.

V3에서 실패했던 all-vocabulary numeric bound도 V4 result에서 `numeric_tolerance_pass=false`로 그대로
남았다. 이것을 삭제하거나 1.389보다 큰 threshold로 고치지 않았다. Greedy decoding의 관찰 가능한
decision과 output exactness가 모두 통과했기 때문에 compatibility만 승인했다. 따라서 다음 actual
experiment에서도 quality equivalence는 평균 logit 유사도가 아니라 case별 output token IDs와 decoded
bytes exact로 검증한다.

## 연구 방향 결정

Qwen technical fallback은 사용하지 않는다. Fallback으로 바꾸는 것은 호환성 문제를 해결하지 않고
한국어 중심 primary를 약화한다. EXAONE revision을 actual-stage target으로 영구 고정한다.

다음 단계는 method novelty 탐색이 아니라 다음 단일 질문에 답한다.

> Public Korean prompts에서 compact generic corpus+prompt retrieval이 ordinary EXAONE greedy와 exact한
> output을 유지하면서 free-running end-to-end wall time을 10% 이상 줄이는가?

Actual-stage plan을 timing 전에 새 namespace로 봉인한다. Primary는 hybrid retrieval free-running이고,
controlled same-output replay와 prompt-only/corpus-only는 secondary다. Compatibility prompt와 이번
diagnostic sequence는 actual case selection에 쓰지 않는다.

## 다음 단계에서 고정할 것

1. public Korean train-only source와 previously used evaluation pool의 exact/normalized
   digest-disjoint split
2. EXAONE tokenizer로 만든 fixed-byte generic token n-gram table
3. latency/model output을 보기 전에 고른 one-document-per-case 64 Korean raw-completion prompts와
   exact prompt hashes
4. proposal cap 3, corpus→prompt/self-output fallback 순서, deterministic tie rule
5. baseline과 candidate의 동일 stop rule 및 128-token minimum / UTF-8-safe stop window
6. lookup, verification, rollback, correction, sync를 포함한 end-to-end timer
7. baseline-only resource calibration 뒤 고정할 fresh-process session/repetition 수
8. primary gate: median reduction ≥10%, paired 95% lower bound >0, 48/64 prompts faster,
   every session positive, token IDs와 bytes exact

이 gate가 실패하면 table 크기나 proposal length를 같은 case의 latency를 본 뒤 바꾸지 않는다. Generic
retrieval branch를 scale-transfer negative로 종료한다. 통과할 때만 Qwen3-8B replication과
equal-memory morphology-normalized extension을 별도 연구로 연다.

이 첫 timing pool은 runtime development/replication용이며 untouched final로 부르지 않는다. 실제 효과가
확인되면 동일 방법을 미사용 public Korean raw-completion 및 chat-template workload에서 별도로 확증한다.

## claim boundary

현재 허용:

> A fixed 7.8B Korean-centric MLX target supports exact greedy retrieval-verification transactions on one
> Apple Silicon configuration.

현재 금지:

- EXAONE retrieval이 ordinary AR보다 빠르다.
- 한국어가 retrieval acceptance를 높인다.
- 이 generic retriever가 새 speculative-decoding method다.
- 11.39% memory fraction이 candidate memory 개선이다.
- 다른 model, runtime, hardware에도 일반화된다.
