# Fresh-v2 16K trained actual-inference preflight protocol

> 작성일: 2026-08-15
>
> 상태: actual timing 전 사전 계약
>
> primary pair: trained 16K update geometry vs trained 2K joint
>
> mandatory diagnostic: trained 16K update geometry vs trained 8K update geometry

## 이 단계에서 답할 질문

Fresh-v2 품질 gate를 통과한 16K candidate가 더 큰 output head와 더 많은 resident weight를 포함하고도
실제 batch-1 생성 wall time을 줄이는지 검증한다. Token count, optimizer step, training time, analytical
FLOPs, random-weight timing은 이 질문의 대체 증거가 아니다.

비교 역할은 calibration quality 결과만으로 고정한다.

| actual role | quality role | vocabulary | parameters | 역할 |
|---|---|---:|---:|---|
| `candidate_16k` | `dense16k_update_geometry` | 16,384 | 31,168,896 | 유일한 candidate |
| `baseline_2k` | `dense2k_joint_v2` | 2,048 | 19,667,328 | primary expansion baseline |
| `frontier_8k` | `dense8k_update_geometry_v2` | 8,192 | 25,172,352 | mandatory frontier diagnostic |

16K는 2K보다 parameter가 58.48%, 8K보다 23.82% 많다. 따라서 vocabulary 확대가 단순히 step 수를
줄이는 것과 전체 시스템 지연을 줄이는 것은 별도 가설이다. 이 실험은 세 모델을 동시에 resident한
상태로 측정하고 parameter·checkpoint 비용을 결과에 함께 남긴다. Memory 개선은 주장하지 않는다.

## 왜 primary는 2K이고 8K는 diagnostic인가

사용자가 정한 가치 기준은 quality-qualified 모델의 실제 추론 효율 개선이다. 2K는 이 연구가 출발한
표준 dense vocabulary anchor이므로 16K-vs-2K가 vocabulary expansion의 핵심 검정이다. 반면 8K는 이전
actual preflight에서 controlled는 통과했지만 free-running joint gate를 실패했고, fresh-v2 quality에서는
16K보다 약했다. 그럼에도 현재의 가장 강한 중간 Pareto point이므로 같은 session에서 반드시 측정한다.

하지만 16K-vs-8K 결과는 16K-vs-2K 실패를 구제하지 못한다. 결과를 본 뒤 comparator를 바꾸거나
10% 기준을 낮추지 않는다.

## 두 공동 primary generation mode

1. `controlled_replay`: 동일한 strict-UTF8 128 raw-byte continuation을 각 tokenizer로 lossless encode한다.
   매 output step의 model forward, argmax, device-host readback은 수행하되 다음 KV input은 고정된
   continuation token을 사용한다. 출력 내용이 같을 때 autoregressive step 구조를 비교한다.
2. `free_running_utf8_greedy`: 각 모델의 logits에 strict-UTF8 token mask를 적용하고 실제 greedy token을
   feedback한다. 최소 128 raw bytes를 생성한 뒤 처음 scalar boundary에서 끝낸다. 모델마다 생성 token
   수와 내용이 달라지는 실제 경로를 비교한다.

두 mode 모두 `raw prompt -> tokenizer -> prefill -> cached decode -> strict-valid bytes`의 E2E를 잰다.
한 mode만 빠르면 primary gate 실패다.

## Cases와 result blindness

Fresh-v2 calibration stream에서 checkpoint·loss·latency를 입력받지 않는 기존 Hangul-heavy selector를
사용한다.

- 서로 다른 문서에서 한 case만 선택
- 128-byte prompt와 그 뒤의 128-byte continuation
- Hangul alphabetic ratio 80% 이상
- warm-up 8문서, measured 64문서
- 세 tokenizer 모두 동일 raw bytes를 lossless round-trip

이 실험은 development one-seed systems preflight이며 final held-out quality test가 아니다. Case selection,
비교 역할, gate, role order는 timing 전에 plan으로 봉인한다.

## Timed scope

Primary `end_to_end_ms`는 다음을 포함한다.

```text
raw prompt UTF-8 decode
  -> tokenizer encode
  -> runtime/KV-cache construction
  -> parallel prefill
  -> every argmax and device-host token readback
  -> incremental cached decode
  -> token-byte reconstruction
  -> strict UTF-8 transition and stop check
  -> final strict decode
```

Checkpoint/tokenizer load, transition-table compilation, case selection은 timing 밖이다. 각 trial 직전과
model loop 끝에 MPS synchronize를 둔다. `tokenizer_ms`, TTFT, decode, `model_loop_ms`를 진단으로
분리하지만 gate는 `end_to_end_ms`만 사용한다.

## 독립 correctness replay

Warm-up 전에는 모든 role/mode에 대해 cache path를 full `use_cache=False` logits와 대조한다. Summary는
더 강하게 64 measured cases 전부에서 checkpoint를 다시 load해 다음을 재구성한다.

- controlled continuation의 exact tokenizer trace와 모든 full/cache logits
- free mode의 masked-greedy token trace를 처음부터 다시 생성
- 저장된 5회 token/output trace와 독립 greedy trace의 exact equality
- free trace 전 위치의 full/cache logits tolerance 및 argmax equality
- strict RFC 3629 transition replay, scalar-boundary stop, repetition determinism

따라서 runtime report의 `pass=true`나 저장된 valid bit만으로 결과를 만들 수 없다. Controlled는 greedy
생성이 아니라 고정 continuation 계약이므로, 공통 증거 필드는 정확히 `trace_contract_exact`라고 부른다.

## 측정 순서와 통계

- 64 measured documents × 5 repetitions × 2 modes × 3 roles
- repetition은 독립 표본으로 세지 않고 먼저 prompt×role cell median으로 접음
- 세 역할의 6개 permutation을 고정 순환하여 first/second/third order를 균형화
- 10,000회 paired-prompt bootstrap
- 세 모델은 같은 process에 동시에 resident

### Primary 16K vs 2K gate

각 mode가 모두 만족해야 한다.

- E2E median point reduction `>= 10%`
- paired-prompt 95% bootstrap lower `> 0`
- candidate가 빠른 문서 `>= 48 / 64`
- 모든 independent correctness pass

두 mode가 모두 통과할 때만 `multiseed_confirmation_authorized=true`다.

### Mandatory 16K vs 8K frontier diagnostic

각 mode에 대해:

- E2E point reduction `> 0`
- paired-prompt 95% bootstrap lower `> 0`
- candidate가 빠른 문서 `>= 33 / 64`
- 모든 independent correctness pass

이 diagnostic은 incremental 8K→16K frontier 해석만 지지하며 primary를 대체하거나 완화하지 않는다.

## 실행 전 fail-fast와 상태 전이

1. 코드·문서·테스트를 clean commit에 고정한다.
2. 결과/plan을 만들지 않는 1-case, 3-role MPS preflight로 checkpoint load, cache/full, free strict decode가
   유한하게 동작하는지만 확인한다. Loss와 latency 수치는 출력하지 않는다.
3. plan을 no-clobber로 생성하고 별도 commit한다.
4. exact plan commit에서 한 timing session을 실행한다.
5. summary가 64-case 독립 replay 후 결과를 no-clobber로 생성한다.

Plan이나 result path에 Git history가 있거나 runtime namespace가 비어 있지 않으면 새 실험을 조용히
만들지 않고 중단한다.

## 결과별 의사결정

### Primary joint gate 통과

- 16K update-geometry recipe를 새 model seeds에 고정
- quality를 seed-level로 복제
- actual timing을 fresh process·multi-session으로 반복
- role-isolated memory와 parameter/checkpoint 비용을 별도 측정
- 그 뒤에만 publication claim과 Hugging Face 공개 artifact를 준비

8K diagnostic 실패는 16K가 2K 대비 가치 있다는 primary 결론을 무효화하지 않지만, monotonic vocabulary
frontier 주장은 금지한다.

### Primary joint gate 실패

- 8K diagnostic이 좋아도 comparator fallback을 하지 않음
- threshold를 낮추거나 controlled-only 결과로 성공을 선언하지 않음
- dense-vocabulary expansion branch의 multi-seed scale-up을 중단
- 다음 연구는 output-head/kernel 최적화나 한 forward에서 sequential step 자체를 줄이는 architecture로
  새 protocol부터 설계

## 주장 경계

통과해도 이 단계가 말할 수 있는 것은 exact Apple-MPS, one model seed, one timing session의 trained
checkpoint 비교뿐이다. 한국어 LLM 일반, 다른 hardware, memory efficiency, publication-grade 재현성은
후속 multi-seed·multi-session evidence 없이는 주장하지 않는다.
