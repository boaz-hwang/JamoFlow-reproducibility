# Large-model preflight V3 invalidation and V4 decision contract

> 작성일: 2026-08-15
>
> V3 plan SHA-256: `f8b4b32afbd8883ce427a9fc71ca8e28cfc4386c3ecad955cade0dd0d2ff62ad`
>
> V3 plan artifact SHA-256:
> `70856fd8edced5ba72bab1acee70be25ac26158fc82b780620b7167c007bf964`
>
> 판정: **V3 all-vocabulary numeric bound invalid; no retrieval or timing result**

## V3에서 실제로 관측한 것

V3 공식 runner는 pinned EXAONE 3.5 7.8B 4-bit model을 load한 뒤 실제 MLX generation 경로인
parallel prefill과 cached decode를 monolithic full-prefix forward와 비교했다. 13-token prompt와 16-token
greedy horizon에서 비교한 28개 position의 argmax는 모두 exact였지만, decode 한 step의
all-vocabulary numeric diagnostic이 고정 bound를 넘어서 fail-closed로 멈췄다.

- argmax mismatch: `0 / 28`
- maximum absolute logit error: `0.080078125`
- maximum normalized error: `1.3894497156143188`
- numeric bound를 넘은 decode step: `1 / 15`
- NaN/Inf: 없음

V3 result artifact는 생성되지 않았다. Official runner는 retrieval lookup, acceptance, candidate-vs-baseline,
latency, throughput을 실행하지 않았다. 그 뒤 timing-silent 원인 진단에서는 rollback argmax가 exact였고
maximum normalized error는 `0.8595988750457764`였다. Forced full-accept, immediate-reject,
partial-accept path도 모두 ordinary 16-token greedy sequence와 exact였으며 각각 4, 16, 8 cycle을
실행했다. Raw token ID와 text는 출력하거나 문서화하지 않았다.

## 왜 Qwen fallback을 바로 선택하지 않는가

V3 문면상 full/cache numeric failure는 technical fallback을 열었다. 그러나 여기서 Qwen으로 바꾸면
실제 greedy output과 rollback은 전부 같은 한국어 중심 primary를, 생성 결정에 쓰이지 않은 vocabulary
logit 하나의 사전 bound 때문에 버리게 된다. 아직 어느 model의 retrieval acceptance나 latency도 보지
않았으므로 primary 유지가 speed-based model shopping은 아니다.

MLX-LM의 공식 `generate_step`도 prompt를 parallel prefill하고 이후 token을 cache에 넣으며 기본 sampler로
argmax를 사용한다. 이 연구가 구현할 exact greedy speculative verification의 관찰 가능한 의미 역시 매
position의 target argmax와 최종 token sequence다.

- MLX-LM generation source:
  <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py>

따라서 V3 실패는 model/runtime incompatibility라기보다 4-bit quantized model의 모든 비선택 logit까지
동일한 수치 bound에 넣은 preflight estimand 오류로 해석한다.

## V4 hard gate

V4는 tolerance를 넓히거나 새 threshold를 고르지 않는다. 기존 `0.05 + 0.01 * abs(reference)`로
maximum absolute/normalized error와 numeric-bound pass 여부를 계속 공개한다. 단, 이 수치는 descriptive
diagnostic으로 내리고 다음 observable greedy semantics만 hard gate로 둔다.

1. prompt와 decode 전 position의 logits가 finite
2. monolithic full prefix와 실제 parallel-prefill/cached path의 argmax가 전부 exact
3. cache trim 뒤 fresh-cache argmax exact
4. independent cached greedy 두 회와 oracle의 16-token sequence exact
5. forced full-accept, immediate-reject, partial-accept의 전체 output exact
6. cache offset/counter invariant와 75% memory safety gate pass

한 position이라도 argmax가 다르거나 output sequence가 다르면 primary는 실패한다. Numeric diagnostic은
값이 크더라도 숨기지 않으며 result schema에 `numeric_tolerance_pass`와 최대 오차를 보존한다.

## 연구 방향에 미치는 영향

이 수정은 compatibility 단계에만 적용한다. V4가 통과해도 retrieval이 빠르거나 한국어에서 유리하다는
증거는 아니다. 실제 논문의 primary estimand는 이후 별도 봉인할 public Korean cases의 free-running
end-to-end latency이며, candidate와 ordinary target이 exact output을 내는 경우에만 효율을 비교한다.
V1~V3 실패 기록은 삭제하지 않고 V4 plan에 전부 결속한다.
