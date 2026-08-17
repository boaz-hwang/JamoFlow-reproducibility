# Phase 3 generation provenance addendum

> 작성일: 2026-08-10  
> 상태: **Phase 3 generation 결과 생성 전 고정**  
> 수정 시점의 정보: primary seed 1,729 F/C/W 완료, generation 결과 0개  
> 영향: generation condition·decoding·metric·분석·gate 불변; evidence reconstruction 강화

## 1. 정적 감사에서 확인한 문제

[기존 generation addendum](./27-phase3-generation-addendum.md)은 raw prompt와 continuation을 저장하지 않는 대신 aggregate JSON만 남기도록 했다. 이 방식은 원문 비공개에는 유리하지만, 다음 무결성 공백이 있었다.

1. Aggregate count와 rate의 내부 산술만 맞으면 실제 continuation 판정에서 나온 값인지 재구성할 수 없었다.
2. Summarizer가 현재 HPLT3 artifact에서 test stream과 deterministic prompt selection을 다시 만들지 않았다.
3. Generation report 안의 checkpoint state hash 두 값만 비교했으며, 현재 primary checkpoint와 training report artifact를 직접 열어 확인하지 않았다.
4. 기존 report가 있으면 source, prompt metadata, checkpoint 또는 metric이 바뀌어도 완료된 것으로 간주했다.
5. Append-only manifest에 요청한 모든 seed/policy pair의 실제 invocation이 있는지 강제하지 않았다.

Generation validity는 natural-text decision gate가 아니지만, 논문에서 teacher-forced BPB와 자유 생성 validity의 차이를 제한점으로 제시하려면 그 집계도 독립 검산할 수 있어야 한다.

## 2. 보존한 실험 설계

다음 사전등록 요소는 바꾸지 않았다.

- HPLT3 Korean test 16M bytes
- 256개의 deterministic Hangul-heavy prompt
- prompt 256 bytes + continuation 256 bytes
- F/C/W, greedy와 temperature 0.8/top-p 0.95
- seed 1,729의 모든 policy/mode에만 UTF-8 hard-mask control
- strict UTF-8, U+FFFD, broad conjoining-Jamo, bytes/codepoint와 DFA failure taxonomy
- initial 3 seeds와 조건부 final 5 seeds의 분리 집계
- 별도 decision gate 없음
- full-prefix elapsed time은 latency evidence가 아님

수정 시점에 Phase 3 generation은 한 조건도 실행되지 않았다. 알려져 있던 primary seed 1,729 F/C/W의 teacher-forced BPB는 generation effect의 방향이나 크기를 알려주지 않는다.

## 3. Source, prompt와 primary lineage 재구성

Runner manifest는 processed `ko.jsonl`의 file size와 SHA-256, selected test stream hash, model/optimization spec과 global-position limit을 invariant로 기록한다.

Summarizer는 current filesystem에서 다음을 다시 수행한다.

```text
processed HPLT3 ko.jsonl
  -> 16M-byte Korean test stream
  -> 512-byte rows와 UTF-8 boundary mask
  -> 등록된 Hangul-heavy/dedup/hash-order prompt selection
  -> 공개 가능한 candidate/unique/selected count
```

재구성한 stream은 primary manifest의 test limit, 전체 test metadata와 selected-stream hash에도 정확히 일치해야 한다. Prompt bytes, prompt hash와 source row index는 여전히 저장하지 않는다. 따라서 검증 가능한 것은 deterministic code path와 공개 metadata까지이며, 비공개 prompt bytes 자체의 사후 cryptographic proof는 의도적으로 포기한다. 이 한계를 tracked summary의 guardrail에 명시한다.

각 seed/policy에는 현재 primary artifact를 직접 대조한다.

- training report의 seed/policy, parameter count, model spec과 optimization spec
- serialized checkpoint artifact SHA-256
- checkpoint state-dict SHA-256
- training report artifact SHA-256
- training report의 `trained_state_sha256`

Generation report는 이 네 lineage hash와 model/global-position/source/prompt metadata를 모두 포함한다.

## 4. 비내용 per-prompt 진단 artifact

Raw continuation 없이 aggregate를 재구성하기 위해 ignored NPZ에 다음 숫자 벡터만 저장한다. 배열의 한 원소는 prompt rank 하나에 대응한다.

### 4.1 Structural family

- strict-valid 여부
- U+FFFD-free 여부
- broad conjoining-Jamo transition-valid 여부
- strict-valid일 때 bytes/codepoint, 아니면 `NaN`

### 4.2 UTF-8 DFA family

- failure category: valid / illegal transition / incomplete terminal scalar
- legal prefix byte 수
- 마지막 완결 codepoint prefix byte 수
- 첫 illegal byte position, 해당 없으면 `-1`

Key는 `mode__variant__family__metric`으로 고정한다. Initial seed 1,729에는 unconstrained와 hard-mask key가 모두 있어야 하고, 다른 seed에는 unconstrained key만 허용한다. 예상하지 않은 배열 key는 실패 처리한다.

이 벡터들은 생성문을 복원할 수 없지만, prompt별 validity signature라는 제한된 파생정보는 담는다. 따라서 `artifacts/` 아래 ignored evidence로만 유지하고 tracked result에는 aggregate와 artifact hash만 남긴다. 원문, prompt, prompt hash, continuation byte, decoded generation은 NPZ와 JSON 어느 쪽에도 넣지 않는다.

## 5. 독립 aggregate 검산과 safe resume

Runner는 continuation에서 직접 계산한 aggregate와 위 진단 배열에서 다시 계산한 aggregate가 완전히 같아야만 artifact를 기록한다. 기존 report와 NPZ를 건너뛸 때도 다음을 모두 재검증한다.

1. current source stream과 prompt metadata
2. current primary checkpoint/report lineage
3. report와 NPZ artifact SHA-256
4. mode/variant별 exact key set과 256×256-byte geometry
5. binary/category/range/partition invariant
6. 진단 배열에서 재구성한 모든 aggregate metric
7. hard-mask 256/256 strict-valid invariant

Report와 artifact 중 하나만 있거나 어느 검증이든 실패하면 stale/incomplete result로 중단하고, 의도적 재실행에만 `--force`를 요구한다.

Summarizer도 같은 계산을 별도로 수행한다. JSON에 적힌 elapsed time만 재구성에서 제외하되 유한한 비음수인지 확인한다. 이 값은 계속 diagnostic only이며 quality contrast나 latency claim의 근거가 아니다.

## 6. 분석과 주장에 미치는 영향

이 보강은 metric, seed, contrast, hard-mask control이나 Phase 3 Gate I/J/K를 바꾸지 않는다. Generation 결과는 semantic quality가 아니며 natural-text gate에도 들어가지 않는다.

증거 강도는 다음처럼 제한된다.

1. Source/test stream과 prompt-selection algorithm 및 공개 count는 독립 재구성된다.
2. 실제 checkpoint와 모든 aggregate validity 수치는 직접 검산된다.
3. Raw prompt를 저장하지 않으므로 실행 당시 각 prompt byte의 사후 직접 증명은 할 수 없다.
4. 숫자 진단은 output content나 coherence를 증명하지 않는다.
5. Full-prefix 실행시간은 incremental decoding 성능을 증명하지 않는다.

따라서 이 실험이 지지할 수 있는 가장 강한 결론은 동일한 공개 source-selection code path와 checkpoint lineage 아래에서 F/C/W의 byte-level encoding failure가 어떻게 달랐는지뿐이다.

## 7. 회귀 검증

추가한 검사는 다음과 같다.

- structural 및 UTF-8 DFA 진단의 aggregate round trip
- fractional category, non-binary flag와 cross-family tampering 거부
- 모든 seed/policy invocation coverage 강제
- completed report/NPZ의 independent aggregate reconstruction
- diagnostic artifact 변조와 stale result 거부
- 기존 prompt selection, causal generation patch, UTF-8 mask와 paired contrast 검증 유지

전체 test suite **165개**가 통과했다.
