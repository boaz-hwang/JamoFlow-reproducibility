# BPE token UTF-8 transition correction

> 작성일: 2026-08-12
> 상태: **publication BPE tokenizer·free-running timing artifact 생성 전 결과맹 구현**
> 상위 교정: [valid-output actual-inference correction](./65-valid-output-actual-inference-correction.md)

## 1. Byte mask를 token logit에 그대로 쓸 수 없다

Candidate/raw byte model의 output unit은 byte 하나지만 standard BPE의 unit은 가변
길이 raw-byte string이다. ByteLevel BPE token은 UTF-8 scalar 여러 개를 포함할 수도
있고 scalar 중간의 continuation byte에서 시작하거나 lead byte로 끝날 수도 있다.

따라서 다음 구현은 잘못이다.

- token ID를 byte 값으로 간주해 256-way DFA mask를 적용
- `tokenizer.decode([token_id])`의 Unicode string을 다시 encode해 raw bytes 복구
- initial UTF-8 state에서 valid한 token만 고정 허용하고 현재 partial state를 무시

Singleton decode는 불완전 byte sequence를 U+FFFD로 바꿀 수 있어 원래 token bytes를
잃는다. State-independent mask는 이전 token이 남긴 partial scalar를 처리하지 못한다.

## 2. Raw token bytes 복구

`byte_bpe_token_bytes`는 tokenizer의 contiguous ID마다 `id_to_token`을 읽고,
GPT-2 ByteLevel의 고정 256-byte-to-Unicode alphabet을 역변환한다. Unicode decoder를
거치지 않으므로 partial scalar도 그대로 보존한다.

Tokenizer audit는 full 256-byte base alphabet을 요구한다. 따라서 어떤 reachable
UTF-8 state에서도 적어도 하나의 legal token이 존재하며, vocabulary가 state를
strand하면 transition compilation 자체가 실패한다.

## 3. 8 × vocabulary transition table

Strict RFC 3629 DFA의 reachable state 8개와 모든 token의 Cartesian product를
결과와 무관하게 한 번 컴파일한다.

```text
next_state[state, token] =
  strict-DFA가 token raw bytes 전체를 소비한 뒤의 state
  또는 illegal transition이면 -1
```

Table은 state별 allowed token IDs, token byte length, raw-token-table SHA-256과
transition-table SHA-256을 포함한다. Compilation은 timing 밖이다. 실제 trial에서는
현재 state row 선택, logit mask 적용, argmax token의 next-state lookup, raw-byte
append, stop/failure 검사를 timing 안에 둔다.

## 4. 종료와 overshoot

BPE token은 atomic하므로 최소 128 bytes를 token 중간에서 넘더라도 token 전체를
emit한다. 그 token이 partial UTF-8 state로 끝나면 다음 token까지 생성하고 accept
state에서만 멈춘다. 따라서 byte model의 0--3 byte bound를 BPE에 적용하지 않는다.

- emitted raw bytes, token steps, bytes/token, overshoot를 그대로 기록한다.
- 512 sequential token units 안에 accept-state completion이 없으면 structural
  failure이며 latency 표본에서 조용히 제외하지 않는다.
- 모든 publication seed에서 completion 100%가 아니면 해당 comparator actual gate는
실패한다.
- Raw time-to-minimum-valid-bytes가 primary이고 bytes/s·codepoints/s는 diagnostic이다.

공통 DFA는 validity-preserving decoder 비교를 정의할 뿐 candidate가 intrinsic UTF-8
문법을 더 잘 학습했다는 증거가 아니다.

## 5. 검증

Synthetic vocabulary에서 split 3-byte scalar, complete scalar, illegal continuation과
illegal lead transition을 확인했다. 256 single-byte token과 multi-byte token 전체에
대해 table 결과를 bytewise DFA와 exhaustive 대조했다. 실제 Tokenizers 0.22.2로
학습한 300-token ByteLevel BPE에서는 raw token table이 256개 single-byte base token을
정확히 포함하고 모든 reachable state가 legal token을 가짐을 확인했다.

