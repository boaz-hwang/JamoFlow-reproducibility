# Fresh vocabulary trained actual-inference preflight protocol

> 작성일: 2026-08-15
>
> 상태: actual timing 전 사전 계약
>
> candidate: fresh `dense8k_update_geometry`
>
> reference: fresh `dense2k_joint`

## 목적

Fresh one-seed quality screen에서 선택된 실제 trained checkpoint가 batch-1 생성의 전체 wall time을
줄이는지 검증한다. Token count, analytical FLOPs, random-weight timing, training time은 이 gate의
대체 지표가 아니다.

두 공동 primary mode가 모두 통과해야 한다.

1. `controlled_replay`: 같은 128 raw-byte continuation을 각 tokenizer로 따로 encode하고, 매 step
   model forward와 argmax를 실행하되 다음 cache input은 고정 continuation token을 사용한다.
2. `free_running_utf8_greedy`: 각 모델이 strict UTF-8 token mask 아래 실제 greedy token을 feedback하고,
   최소 128 raw bytes를 생성한 뒤 처음 Unicode scalar boundary에서 멈춘다.

Controlled는 같은 출력 내용에서 tokenizer별 autoregressive step cost를 비교한다. Free-running은 실제
모델 출력과 token-length 분포가 만드는 경로를 잰다. 한 mode만 빠르면 성공이 아니다.

## 고정 물리 모델

Quality result의 exact artifact/state hash를 plan에 고정한다.

| timing role | trained role | vocab | parameters | checkpoint bytes |
|---|---|---:|---:|---:|
| candidate | `dense8k_update_geometry` | 8,192 | 25,172,352 | 100,713,026 |
| reference | `dense2k_joint` | 2,048 | 19,667,328 | 81,838,658 |

Candidate는 parameter가 27.99% 많다. 이 preflight는 latency 개선만 판정하고 memory 개선을 주장하지
않는다. Checkpoint load는 timing 밖이며 두 모델은 같은 process에 동시에 resident한 상태에서
role order를 교차한다.

## Cases

Fresh calibration stream에서 결과와 무관한 고정 selector를 사용한다.

- 서로 다른 document에서 한 case만 선택
- Hangul alphabetic ratio 80% 이상인 128-byte prompt
- 이어지는 strict-UTF8 128-byte continuation
- warm-up 8문서, measured 64문서
- 동일 prompt/continuation raw bytes를 두 tokenizer가 각각 lossless encode

Case 선택은 checkpoint, loss, latency를 입력으로 받지 않는다. Fresh calibration을 사용하므로 이
단계는 development systems preflight이며 sealed final quality 결과가 아니다.

## Timed scope

Primary `end_to_end_ms`는 다음을 모두 포함한다.

```text
raw prompt
  -> strict UTF-8 text conversion
  -> tokenizer encode
  -> runtime/KV-cache construction
  -> parallel prompt prefill
  -> every output-token argmax + device-to-host token readback
  -> every incremental cached decode
  -> token-byte reconstruction
  -> strict UTF-8 state/stop check
  -> output strict decode
```

Checkpoint loading, tokenizer JSON loading, strict-token transition compilation, case selection은 timing 밖이다.
각 trial 직전과 model loop 끝에 MPS synchronize를 둔다. `tokenizer_ms`, TTFT, decode,
`model_loop_ms`를 진단으로 분리하지만 gate는 raw-input-to-valid-output `end_to_end_ms`만 사용한다.

Controlled에서도 매 output position에 argmax와 host readback을 실행한다. Gold token을 cache에 넣는다는
이유로 sampling 비용을 생략하지 않는다. Free mode에서는 state별 valid token mask, argmax, token-byte
append, DFA transition, stop check가 전부 timer 안이다.

## UTF-8 token constraint

각 ByteLevel BPE token을 raw bytes로 복원하고, strict RFC 3629 DFA의 reachable state마다
`state x token -> next state/invalid` table을 plan 전에 컴파일한다. Free mode는 invalid token을
`-inf` mask한다. 최소 128 bytes 이후 state 0에서만 종료한다.

Token 하나가 여러 bytes를 내므로 최대 output은 역할별
`128 + maximum_token_bytes - 1`이다. 모든 repetition의 token trace와 raw output을 저장하고 summary가
transition table로 독립 replay한다. 동일 prompt/role의 5회 output은 bitwise 같아야 한다.

## Cache correctness

Timing 전에 warm-up 8 case의 controlled token sequence와 free-generated token sequence를 각각
full `use_cache=False` forward와 비교한다.

- parallel prefill final logit
- 이후 모든 incremental cached logit
- active tolerance 안의 normalized maximum error
- 모든 비교 위치의 argmax exact equality

Summary도 checkpoint를 다시 load해 같은 correctness를 독립 수행한다. Timing artifact가 `pass=true`만
자기 선언해서는 결과를 만들 수 없다.

## 측정과 통계

- 64 measured documents
- prompt마다 5 repetitions
- repetition은 독립 표본으로 세지 않고 먼저 cell median으로 접는다
- candidate-first/reference-first를 case×repetition×mode parity로 정확히 교차
- 10,000회 paired-prompt bootstrap

각 mode gate:

- median E2E point reduction `>= 10%`
- paired-prompt bootstrap 95% lower `> 0`
- candidate가 빠른 document `>= 48 / 64`
- 모든 cache/full 및 strict-output correctness pass

두 mode가 모두 통과해야 `multiseed_confirmation_authorized=true`다. 이것은 one-seed/one-session
fail-fast gate이며 publication CI가 아니다.

## 결과별 다음 단계

### 둘 다 통과

- update geometry recipe와 2K/8K model family를 새 model seeds에 고정
- fresh training을 multi-seed로 복제
- quality를 seed-level로 확인
- 실제 timing을 fresh process/multi-session으로 반복
- parameter, checkpoint, isolated memory를 latency와 함께 공개
- 그 뒤에만 larger Mac-feasible graph, Korean downstream, CUDA replication을 검토

### 하나라도 실패

- 현재 dense-vocabulary inference-efficiency branch는 scale-up하지 않음
- geometry의 quality-adaptation 관측은 별도 결과로 남기되 사용자의 핵심 효율 성공으로 부르지 않음
- token head/kernel 또는 architecture-level sequential-step reduction을 다시 설계할 때도 새 protocol을
  먼저 봉인

## 주장 경계

통과하더라도 말할 수 있는 것은 이 exact Apple-MPS, one model seed, one timing session에서 trained
8K가 trained 2K보다 두 실제 generation path에서 빨랐다는 것뿐이다. 한국어 전반, 다른 hardware,
더 큰 모델, memory efficiency, publication-grade 재현성은 후속 multi-seed/multi-session 증거가
필요하다.
