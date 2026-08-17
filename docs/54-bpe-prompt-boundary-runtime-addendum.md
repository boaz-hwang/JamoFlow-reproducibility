# BPE prompt-boundary runtime addendum

> valid-output token-DFA 후속 교정: [BPE token UTF-8 transition correction](./72-bpe-token-utf8-transition-correction.md)

> 작성일: 2026-08-11
> 상태: **publication-scale tokenizer·model·timing 결과 전 고정**
> 적용: standard byte-level BPE controlled replay와 free-running benchmark

## 1. 문제

Raw prompt `p`와 continuation `c`를 한 번에 BPE encode하면 merge가 `p|c` 경계를 가로지를 수 있다. 그러나 실제 API는 continuation이 오기 전에 prompt tokenization과 KV cache를 이미 고정한다. 나중에 continuation이 도착했다고 prompt 마지막 token을 다시 쪼개거나 합치는 joint tokenization은 deployed incremental runtime과 다르다.

## 2. Primary replay semantics

Controlled replay의 primary BPE 경로는 다음으로 고정한다.

1. prompt를 단독 encode하고 prompt token IDs를 고정한다.
2. held-out continuation을 단독 encode한다.
3. prompt prefill final logit이 continuation token 1을 채점한다.
4. continuation token 1…T−1을 cache에 넣어 T개 truth tokens를 모두 채점한다.
5. 사용하지 않을 T+1번째 logit은 계산하지 않는다.

두 token sequence를 이어 decode했을 때 raw `p+c`가 byte-for-byte 복원되어야 한다. Prompt와 continuation 각각의 round trip, concatenated round trip, joint round trip 중 하나라도 실패하면 benchmark를 중단한다.

## 3. Joint encoding은 sensitivity

`encode(p+c)`는 다음을 보고하는 sensitivity일 뿐 primary timing에 쓰지 않는다.

- joint token count
- joint sequence 안에 raw prompt와 정확히 일치하는 token prefix가 존재하는지
- separate token IDs와 joint token IDs가 같은지
- 경계를 가로지르는 merge가 발생한 prompt 비율

Joint encoding이 더 짧더라도 이미 고정된 prompt cache를 소급 수정할 수 없으므로 API-realistic speedup으로 계산하지 않는다.

## 4. Cross-architecture output horizon

Controlled replay는 동일 raw prompt와 동일 raw continuation을 각 architecture의 실제 sequential unit으로 완전히 채점한다. Candidate는 byte unit, BPE는 separate continuation token unit을 사용한다. Raw completion bytes, token/byte steps, TTFT, decode, end-to-end를 모두 보고한다.

Free-running BPE는 공통 strict UTF-8 transition constraint 아래에서 최소 128 valid raw bytes에 처음 도달한 accept-state token에서 멈춘다. Token 하나가 threshold를 넘을 수 있으므로 overshoot와 allowed-token 부재/cap failure를 기록한다. Byte candidate도 같은 constraint와 stop 검사를 timing에 포함하며 0--3 byte overshoot가 가능하다. Static constraint compilation만 trial 밖에 두고 mask 적용, tokenizer·detokenizer와 stop 검사는 end-to-end timing에 포함한다. Prepared-token model-only timing은 diagnostic이며 이 결과를 대체하지 않는다. 상세 통계 이유는 [valid-output actual-inference correction](./65-valid-output-actual-inference-correction.md)을 따른다.

## 5. 구현 gate

`prepare_byte_bpe_replay()`는 private token IDs와 content-free audit만 반환한다. Tracked 결과에는 token count, byte count, boolean audit, tokenizer artifact hash만 남기고 prompt text와 token IDs는 남기지 않는다.
