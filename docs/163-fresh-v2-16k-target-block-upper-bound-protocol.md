# Fresh-v2 16K target-block upper-bound protocol

> 작성일: 2026-08-15
>
> 상태: **구현·검증 후 timing 전에 봉인할 result-blind protocol**
>
> 선행 결과: `docs/162-fresh-v2-16k-trained-actual-result-and-block-pivot.md`

## 1. 질문

Fresh-v2에서 학습한 16,000-token 모델은 2K baseline보다 quality가 좋았고, controlled
same-output 생성에서는 24.93% 빨랐다. 그러나 free-running faster-prompt 수가 43/64로 사전
48/64 gate를 통과하지 못했고, 8K frontier보다도 free path가 8.29% 느렸다. Dense vocabulary
확대만으로는 사용자가 요구한 안정적인 actual inference efficiency를 입증하지 못했다.

후속 질문은 더 좁다.

> 같은 trained 16K target의 exact greedy output을 보존하면서 여러 known-correct target token을
> 한 forward로 검증하면, 실제 Apple-MPS target kernel만으로 learned draft의 비용과 오차를
> 감당할 만큼 큰 E2E 여유가 생기는가?

이 단계는 perfect draft upper bound다. Draft를 학습하거나 실행하지 않고 target-side 가능한
최대 headroom을 먼저 재서, 상한 자체가 낮으면 learned draft 연구를 시작하지 않는다.

## 2. 선행연구와 novelty 경계

Block verification과 speculative decoding은 이미 확립된 일반 기법이다. 따라서 다음을 novelty로
주장하지 않는다.

- known future tokens를 한 target forward로 검증하는 것
- perfect-draft upper bound를 측정하는 것
- target과 다른 tokenizer를 쓰는 speculative decoding
- draft output vocabulary를 줄이는 것

[Block Verification](https://arxiv.org/abs/2403.10444)은 autoregressive verification을 block으로
바꾸는 일반 방법을 다룬다. ACL 2026의
[TokenTiming](https://aclanthology.org/2026.acl-long.1983/)은 서로 다른 tokenizer를 쓰는 target과
draft 사이의 speculative decoding을 제시한다. ACL 2026 Findings의
[SpecVocab](https://aclanthology.org/2026.findings-acl.2000/)은 draft output head의 vocabulary 비용을
직접 줄이고 EAGLE-3 대비 최대 8.1% throughput 개선을 보고한다. 또한
[Speculative Decoding Across Languages](https://arxiv.org/abs/2605.30580)는 multilingual learned
draft가 약할 수 있고, acceptance가 낮아도 값싼 n-gram draft가 더 유리할 수 있음을 보고한다.

따라서 이번 단계의 연구 가치는 새로운 speculative algorithm이 아니라 다음의 **실측
의사결정 근거**에 있다.

1. 한국어 중심 fresh-v2에서 quality-qualified된 실제 16K target을 쓴다.
2. dense-vocabulary actual 실패가 token-step 지배적이었다는 관측을 target block kernel로 직접
   검증한다.
3. draft 학습 전에 exact-output target-only ceiling을 고정 gate로 판정한다.
4. 상한이 통과할 때만 한국어/same-tokenizer draft의 비용·acceptance 연구를 연다.

이 upper bound 자체는 publication efficiency claim이 아니다.

## 3. 고정된 물리 target

- role: `dense16k_update_geometry`
- vocabulary: 정확히 16,000
- parameters: 31,168,896
- checkpoint: fresh-v2 16K quality 결과와 actual 결과가 봉인한 동일 checkpoint
- quality: document BPB 1.3934744561
- tokenizer, strict UTF-8 transition table, 72 inference cases: 이전 16K actual plan과 동일

모든 timing role은 같은 model object와 weights를 사용한다. 모델 크기나 quality 차이가 speed
차이에 섞이지 않는다.

## 4. 역할과 block 알고리즘

고정 역할은 다음 네 개다.

| role | target block size | 용도 |
|---|---:|---|
| `baseline_ar` | 1 | ordinary cached target AR |
| `perfect_block_2` | 2 | 진단 |
| `perfect_block_4` | 4 | **사전 고정 primary** |
| `perfect_block_8` | 8 | 진단 |

Prompt prefill 뒤 cache가 prompt 전체를 관찰하고 첫 output logit을 낸다. Output token 수를 `n`,
block size를 `b`라 하면:

- baseline target forward calls: `n`
- perfect block target forward calls: `1 + ceil((n - 1) / b)`

Block runtime은 현재 cache가 마지막 committed token 직전까지가 아니라 **마지막 committed token을
아직 consume하지 않은 상태**라는 기존 AR invariant를 유지한다. 다음 output index가 `i`일 때
target input은 `output[i-1 : i+b-1]`이고, 나온 `b`개 logits는 `output[i : i+b]`를 검증한다.
마지막 cache length는 baseline과 똑같이 `prompt tokens + output tokens - 1`이다.

## 5. Exactness 계약

각 warmup/측정 case와 controlled/free mode에서 다음을 요구한다.

1. no-cache full target forward와 cached baseline/block logits를 모든 output 위치에서 비교한다.
2. MPS 허용치는 `atol=1e-4`, `rtol=2e-5`와 같은 reference-relative normalized ratio `<=1`이다.
3. 모든 위치 target argmax가 exact해야 한다.
4. controlled output token/bytes가 기존 128-byte continuation과 exact해야 한다.
5. free output은 ordinary target AR의 strict-UTF-8-masked greedy output과 token/byte 단위로 exact해야
   한다.
6. 5 repetitions와 네 role의 free output이 모두 동일하고 strict DFA replay를 통과해야 한다.
7. target forward-call counter와 최종 cache-observed token 수가 수식과 exact해야 한다.

Summary는 측정 report의 boolean을 신뢰하지 않는다. 같은 checkpoint를 다시 load해 64 measured
cases 전부에 대해 full/cache/block replay와 free greedy regeneration을 독립 수행한다.

## 6. Timing 범위

Timer 안에는 다음이 들어간다.

- raw prompt strict UTF-8 decode와 16K tokenizer encode
- fresh target runtime/KV cache 구성
- parallel prompt prefill
- 모든 target block forward
- verifier argmax, strict UTF-8 mask와 device-host readback
- token-byte reconstruction, DFA transition, stop와 strict decode
- final MPS synchronization

Checkpoint load, tokenizer/transition compile, case 선택, perfect-draft token **생성 비용**은 timer 밖이다.
Known-correct token block slicing과 target verification은 timer 안이다. 따라서 측정값은 target-side
kernel upper bound이지 실제 speculative runtime이 아니다.

## 7. 고정 workload와 schedule

- modes: `controlled_replay`, `free_running_utf8_greedy`
- prompt: 128 raw bytes
- continuation/stop minimum: 128 raw bytes
- warmup: 8 distinct documents
- measured: 64 distinct documents
- repetitions: 5
- target 하나만 resident

네 role의 순서는 8-row balanced/reversed cycle을 사용한다. 전체 `64 × 5 × 2 = 640` cell은
정확히 80 cycle이라 각 role은 각 temporal position에 160번씩 놓인다. Block role별 timing을
몰아서 재는 순서 편향을 피한다.

## 8. 사전 gate

Primary는 결과와 무관하게 `perfect_block_4`다. Controlled와 free 각각 다음을 모두 만족해야 한다.

- 모든 correctness pass
- median-of-prompt-medians E2E reduction `>=35%`
- paired 64-prompt bootstrap 95% interval lower `>0`
- faster prompts `>=48/64`

두 mode가 모두 통과해야 전체 pass다. Block 2/8이 아무리 좋아도 block 4 실패를 구제하거나
primary를 대체하지 않는다. Gate를 35%로 둔 이유는 target-only upper bound에서 적어도 이 정도
headroom이 없으면 learned draft forward, proposal selection, rejection/rollback 비용과 불완전
acceptance를 포함해 최종 10%+ actual improvement를 안정적으로 남기기 어렵기 때문이다.

## 9. 결과에 따른 고정 행동

### Fail

- learned draft를 학습하지 않는다.
- block size를 결과를 보고 다시 고르지 않는다.
- dense vocabulary 확장과 이 target-block branch를 종료한다.
- 기존 negative actual 결과와 함께 token compression만으로는 충분하지 않았다는 systems evidence로
  보존한다.

### Pass

- 허용되는 것은 **same-tokenizer learned-draft fail-fast 한 단계뿐**이다.
- draft parameter/FLOPs/resident memory, acceptance, rollback, tokenizer/verification overhead를 모두
  포함한 actual runtime을 새 result-blind gate로 측정한다.
- generic n-gram/cheap draft를 mandatory control로 둔다.
- 한국어 구조를 쓰는 draft는 같은 비용의 generic control을 이겨야 한국어-specific 기여로 인정한다.
- exact trained target output과 quality를 보존해야 한다.

Pass여도 speculative efficiency, publication readiness, multi-seed/hardware generalization은 주장하지
않는다.

## 10. Fable 5 중간 검토 반영

수용한 핵심은 다음과 같다.

1. analytical token/call 감소와 actual E2E를 분리한다.
2. upper bound와 deployable runtime을 분리한다.
3. aggregate median 외 prompt direction과 uncertainty를 gate에 넣는다.
4. 실제 target quality와 exact output을 먼저 고정한다.
5. Apple-MPS one-seed 결과를 일반 LLM/한국어 전체/CUDA로 확대하지 않는다.

수용하지 않은 해석은 negative 결과만으로 quality/methodology paper를 완성됐다고 간주하는 것이다.
사용자의 기준은 실제 효율 개선이므로, 이 단계 역시 양성 actual target-kernel headroom을 다음 연구의
필요조건으로 둔다.

## 11. 주장 경계

이 결과로 말할 수 있는 것:

- trained 16K target의 exact output에서 perfect-draft target block kernel이 ordinary AR보다 얼마나
  빠르거나 느린지
- block 2/4/8별 target invocation 감소와 실제 wall-time 사이의 차이
- learned draft가 감당할 수 있는 target-only headroom이 있는지

말할 수 없는 것:

- 실제 draft를 포함한 speculative decoding이 빠르다
- acceptance/rollback을 포함해 35% 또는 10%가 남는다
- 새 speculative algorithm을 제안했다
- memory가 개선됐다
- 다른 seed, model scale, GPU, 언어에서도 일반화된다
- publication-grade 최종 효율 결과다
