# Large-model preflight V2 invalidation and V3 oracle correction

> 작성일: 2026-08-15
>
> V2 plan SHA-256: `81ed3957443c3fc935391886c37f3549f8f5f3b70dcd53194c4546ac57c52a09`
>
> V2 plan artifact SHA-256:
> `f69f5b36ffd442bea6ae8c1840a6a9b2067f3044150383bb2fa083ecb57ea8c9`
>
> 판정: **V2 cache oracle invalid; no generation or efficiency result**

## V2가 실패한 위치

V2는 pinned custom config/tokenizer와 model load를 통과했다. 그 뒤 monolithic 13-token prompt
forward를, 같은 prompt를 처음부터 끝까지 one-token call 13회로 처리한 cache replay와 비교했다.
이 비교는 다음 aggregate에서 실패했다.

- finite logits: pass
- argmax mismatch: 13개 위치 중 1개
- mismatch position: 첫 input position
- monolithic reference top-2 margin: exact `0.0`
- maximum absolute error: `0.294921875`
- maximum normalized error: `5.458246231079102`

V2는 여기서 중단됐고 generated token, forced proposal, candidate-vs-baseline, latency, throughput,
acceptance를 관측하지 않았다. Result artifact도 생성되지 않았다.

## oracle이 잘못된 이유

실제 MLX generation은 prompt를 one-token call로 처음부터 재생하지 않는다. Prompt의 대부분을 한
parallel prefill로 처리하고 마지막 token부터 cached decode를 시작한다. 4-bit quantized matmul은
matrix shape에 따라 작은 수치 차이가 날 수 있으므로, 사용하지 않는 13회 one-token prompt path를
필수 compatibility oracle로 둔 것은 actual runtime과 맞지 않았다.

실패 후 aggregate diagnostic에서 실제 첫-step 경로인 `parallel prompt[:-1] + cached last token`은:

- 13/13 position argmax exact
- maximum absolute error `0.03125`
- maximum normalized error `0.14919807016849518`
- final-position argmax exact

로 **V1부터 고정한 동일 tolerance**를 통과했다. 이 진단은 candidate latency나 retrieval 결과를
포함하지 않는다.

## V3의 유일한 의미 변경

V3는 tolerance를 넓히지 않는다. Cache oracle만 실제 runtime과 맞춘다.

1. `prompt[:-1]` parallel prefill
2. final prompt token cached call
3. cached greedy token을 한 step씩 진행
4. 각 step의 cached logits를 같은 전체 prefix의 monolithic final logits와 비교
5. prompt 13 positions와 생성 16 tokens의 후속 15 positions, 총 28 position에서 기존
   `0.05 + 0.01 * abs(reference)` 및 exact argmax 요구

그 뒤 independent cached greedy 두 회와 forced full-accept/immediate-reject/partial-accept가 모두
oracle token sequence와 같아야 한다.

Model/revision/weight, tokenizer texts, tolerance, 16-token horizon, proposal cap 3, memory 75%, fallback
rule, no-timing claim boundary는 바꾸지 않는다. V1/V2 namespace는 보존한다.

