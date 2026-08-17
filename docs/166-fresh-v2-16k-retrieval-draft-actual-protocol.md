# Fresh-v2 16K retrieval-draft actual protocol

> 작성일: 2026-08-15
>
> 상태: **구현·검증 뒤 실제 timing 전에 봉인할 개발용 프로토콜**
>
> 선행 근거: `docs/164-fresh-v2-16k-target-block-upper-bound-result.md`,
> `docs/165-retrieval-draft-literature-audit-and-fail-fast-direction.md`

## 1. 연구 질문

Perfect block-4 target kernel은 동일한 16K target의 ordinary cached AR보다 controlled와 free
mode 모두에서 60%가 넘는 실제 E2E headroom을 보였다. 그러나 perfect future token은 현실에서
공짜로 얻을 수 없다. 이번 단계의 질문은 하나다.

> 128MB train split에서만 만든 compact token n-gram과 prompt/self-output suffix lookup을 실제
> draft로 실행하고, lookup·target 검증·accept/reject·rollback을 모두 포함해도 exact target
> output에서 ordinary AR보다 Apple-MPS E2E가 10% 이상 빨라지는가?

이 질문에 실패하면 현재 target에서 retrieval speculative branch를 종료한다. 성공하더라도 이
결과만으로 새 한국어 방법이나 출판 수준의 효율 기여를 주장하지 않는다. 성공은 disjoint test에서
cost-matched generic control과 한국어-aware draft를 비교할 다음 설계만 허용한다.

## 2. 선행연구와 novelty 경계

Prompt/output retrieval, corpus n-gram draft, target-probability confidence, hardware-aware draft sizing,
compact dictionary는 이미 선행연구다. 특히 UniSpec, SSSD, DictSpec, Cacheback과 multilingual
speculative-decoding 연구가 이 축을 직접 다룬다. 따라서 아래는 novelty가 아니다.

- prompt에서 가장 긴 suffix를 찾아 다음 토큰을 제안하는 것
- train corpus n-gram dictionary에서 다음 토큰을 제안하는 것
- corpus 제안이 없을 때 prompt lookup으로 fallback하는 것
- target model이 block을 검증하고 첫 mismatch에서 correction하는 것
- acceptance와 hardware cost가 속도를 좌우한다는 관찰

이번 실험은 필수 generic baseline의 **실제 시스템 feasibility screen**이다. 통과 뒤에도 한국어
기여는 동일한 target·table budget·draft 길이·timing 범위에서 generic hybrid를 이겨야 인정한다.

## 3. 고정 target과 retrieval table

- target: quality-qualified fresh-v2 16,000-token dense model, 31,168,896 parameters
- checkpoint/tokenizer/UTF-8 transition: upper-bound 실험과 exact 동일
- corpus table source: filtered 128MB train split의 24,722,642 token만 사용
- calibration/test/model logits/latency 입력: 없음
- table: 1/2/3-token context, 최대 200,000 entries
- minimum context count = 5, minimum best-next probability = 0.8
- 실제 seal 기준 order별 entry 수: 454 / 73,364 / 126,182
- table artifact: 1,341,550 bytes
- 최대 proposal: 4 tokens

Table은 timing 전에 이미 별도 seal과 commit으로 고정돼 있다. 결과를 보고 entry budget, threshold,
context order, precedence를 바꾸지 않는다.

## 4. 고정 역할

| role | proposal source | 지위 |
|---|---|---|
| `baseline_ar` | 없음 | ordinary target AR baseline |
| `prompt_lookup_block_4` | prompt+self-output suffix | diagnostic |
| `corpus_ngram_block_4` | train-only compact n-gram | diagnostic |
| `hybrid_retrieval_block_4` | corpus 우선, 없으면 prompt fallback | **primary** |

Diagnostic role이 더 빨라도 primary 실패를 구제하거나 대체하지 않는다. Corpus-first precedence도
결과와 무관하게 고정한다.

## 5. Exact speculative transaction

Target cache는 마지막 emitted token을 아직 consume하지 않은 one-token-lag invariant를 유지한다.
현재 pending token이 `p`, 최대 네 draft token이 `d1..dk`이면 target 입력은
`[p,d1,...,dk]`다.

- no proposal: `[p]` 한 step ordinary AR
- mismatch at draft index `j`: 앞의 accepted draft와 target correction을 emit하고, rejected suffix를
  `DynamicCache.crop`으로 제거
- full accept: 모든 draft를 emit하고 마지막 verifier logit의 bonus token까지 emit
- UTF-8 stop이 block 중간에 생기면 terminal emitted token을 cache에서 제외하도록 crop

모든 role은 같은 resident target model과 같은 expected target trace를 사용한다. Controlled mode는
고정 continuation token을 verifier target으로 쓰고, free mode는 strict-UTF-8-masked target greedy
argmax를 쓴다. 어느 경우든 baseline과 token·raw byte·최종 DFA state·cache length가 exact해야 한다.

## 6. Timing 범위

Timer 안:

- raw prompt strict UTF-8 decode와 16K tokenizer encode
- fresh target runtime/KV cache, parallel prefill
- train-only dictionary 및 prompt/self-output lookup
- 모든 target block forward와 verifier argmax/device-host readback
- accept/reject, correction, bonus, `DynamicCache.crop`
- token-byte reconstruction, strict DFA transition/stop/decode
- final MPS synchronization

Timer 밖:

- checkpoint와 compact table 파일 load
- tokenizer/UTF-8 transition compile
- case selection 및 ordinary-target expected trace 생성

따라서 lookup이 싸다는 분석적 가정이 아니라 실제 Python lookup, synchronization과 rollback까지
포함한 E2E를 판정한다.

## 7. Workload와 schedule

- modes: controlled replay, free-running strict-UTF-8 greedy
- prompt 128 raw bytes, continuation/stop minimum 128 raw bytes
- maximum free output 131 tokens, maximum raw output는 sealed target trace의 155 bytes
- warmup 8 documents, measured 64 distinct documents, 5 repetitions
- 네 role 순서는 8-row balanced/reversed cycle
- 총 timed trials: `64 × 5 × 2 × 4 = 2,560`
- 한 Apple-MPS process/session, target one seed

Repetition은 독립 표본으로 세지 않는다. 각 prompt에서 다섯 repetition median을 만든 뒤 64 prompt를
통계 단위로 사용한다.

## 8. 사전 고정 gate

Primary hybrid role은 controlled와 free 각각 다음을 모두 만족해야 한다.

1. output/cache/full-vs-cache correctness 전체 pass
2. median-of-prompt-medians E2E reduction `>=10%`
3. paired 64-prompt bootstrap 95% lower bound `>0`
4. faster prompts `>=48/64`

두 mode 모두 통과해야 전체 pass다. Target forward-call 감소나 높은 acceptance만으로 통과하지 않는다.

## 9. 독립 검증과 결과에 따른 행동

측정 report는 timing arrays와 counters, free token/byte traces를 고정한다. Summary는 report의 pass
boolean을 신뢰하지 않고 같은 checkpoint/table를 다시 load해 measured 64 cases의 네 role·두 mode를
모두 재실행한다. Ordinary full/cache target logits, strict greedy trace, output bytes, cache lag와 table
identity를 다시 대조한 뒤에만 통계를 계산한다.

- fail: block size/table threshold를 튜닝하지 않고 retrieval branch 종료
- pass: 새 disjoint Korean test와 고정 cost budget을 먼저 봉인한 뒤 generic hybrid 대 한국어-aware
  draft 비교 설계만 허용

## 10. 주장 경계

말할 수 있는 것:

- 이 exact target/checkpoint/cases/Apple-MPS에서 generic retrieval draft의 실제 E2E feasibility
- prompt/corpus/hybrid별 proposal coverage, acceptance와 target-call 감소
- 분석적 call 감소와 실제 wall-time 사이의 차이

말할 수 없는 것:

- retrieval speculative decoding 자체가 새 방법이라는 주장
- 한국어 구조가 효율을 개선했다는 주장
- final-blind 또는 confirmatory 결과
- 여러 seed/model scale/CUDA/다른 언어·hardware 일반화
- memory 개선 또는 publication-ready 최종 효율 결과

이 범위는 Fable 5 검토의 핵심 지적—실제 E2E와 분석량 분리, generic control 필수, 상한과 deployable
runtime 분리, one-device 결과의 일반화 금지—를 그대로 반영한다.
