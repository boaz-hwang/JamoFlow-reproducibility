# Fresh-v2 16K target-block upper-bound result

> 작성일: 2026-08-15
>
> 상태: **target-only upper bound 통과; same-tokenizer draft fail-fast만 허가**
>
> authoritative result: `results/fresh-vocabulary-16k-target-block-v1/summary.json`

## 결론

실제 학습된 fresh-v2 16K target의 ordinary cached AR과 perfect-draft target block 2/4/8을
Apple MPS에서 비교했다. 사전 고정 primary인 block 4는 controlled와 free 모두 35% E2E gate를
큰 폭으로 통과했다.

- controlled E2E reduction: **63.927%**
- controlled 95% prompt-bootstrap interval: **[62.677%, 64.959%]**
- controlled faster prompts: **64/64**
- free E2E reduction: **65.683%**
- free 95% prompt-bootstrap interval: **[63.061%, 66.521%]**
- free faster prompts: **64/64**

따라서 target block kernel은 이 trained 16K model에서 후속 draft 연구를 막는 병목이 아니다.
다만 perfect-draft 생성 비용과 imperfect acceptance/rollback은 측정하지 않았으므로, 이것은 실제
speculative decoding 효율 결과도 논문 양성 결과도 아니다.

## 봉인과 identity

- implementation commit: `d6d2a4c`
- result-blind plan commit: `7a1fcea`
- plan payload SHA-256: `f4be11e53a90c916695b11e105402b06a1ad6d3dda683d056491b04e7feb8fd2`
- plan file SHA-256: `35c26841a31be032336db63d2c58a5c13b819ad093020fa5bf3fcdcb9363fc35`
- runtime report payload SHA-256: `2c90fbfeb2cc9de98667b61678bd66027f591ddc09330c5de222db20e79b047f`
- runtime report file SHA-256: `d7969f03f8b077eac590983b165b5b12e9df01ce1f3daa084f1b111d47c6be7d`
- timing artifact SHA-256: `9f252f96937eebe0320f9688ed2458bbc532ed3010aa3d919387429d14934790`
- result payload SHA-256: `ad7db5d5699735eb27d798c1b716060cb5670a5b721eec1e5f82154d8f81185e`
- tracked result file SHA-256 before commit: `c751b58612f97cf68c2e509deffe404ba9cb15358cf9e607dd01898c83329914`

Target는 31,168,896-parameter `dense16k_update_geometry` checkpoint 하나다. 네 timing role은 같은
model object와 weights를 사용했고 role별로 fresh KV cache만 만들었다.

## 전체 결과

| target role | controlled reduction | controlled CI | free reduction | free CI | faster prompts |
|---|---:|---:|---:|---:|---:|
| block 2 | 38.148% | [37.073%, 39.284%] | 37.921% | [34.508%, 39.177%] | 64/64, 64/64 |
| **block 4 primary** | **63.927%** | **[62.677%, 64.959%]** | **65.683%** | **[63.061%, 66.521%]** | **64/64, 64/64** |
| block 8 diagnostic | 77.893% | [76.102%, 78.818%] | 78.677% | [76.173%, 79.641%] | 64/64, 64/64 |

Block 2/8은 사전 계획대로 진단일 뿐이다. Primary가 이미 통과했지만, 후속 actual runtime의
block size는 결과가 가장 큰 8로 바꾸지 않고 계획대로 4를 유지한다.

## Target 호출과 wall time

Median target forward calls와 E2E는 다음과 같다.

| mode | AR calls/time | block-4 calls/time | call reduction | E2E reduction |
|---|---:|---:|---:|---:|
| controlled | 25 / 66.917 ms | 7 / 24.139 ms | 72.0% | 63.9% |
| free | 32.5 / 81.550 ms | 9 / 27.985 ms | 72.3% | 65.7% |

호출 감소가 wall-time 감소로 대부분 이어졌지만 완전히 같지는 않다. Block forward는 여러 위치의
vocabulary logits를 동시에 계산하고, verifier mask/argmax/vector readback 비용도 있다. 반대로 TTFT는
동일 target prefill이므로 개선되지 않았다.

- controlled TTFT: 3.528 ms → 3.551 ms
- free TTFT: 3.477 ms → 3.463 ms
- controlled decode: 63.364 ms → 20.126 ms
- free decode: 78.071 ms → 24.555 ms

효과는 decode sequentiality 감소에서 왔으며 tokenizer나 TTFT 개선으로 설명되지 않는다.

## Correctness

Official summary는 checkpoint를 다시 load해 64 measured documents를 전부 독립 재생했다.

- controlled comparisons: role마다 1,634/1,634 exact argmax
- free comparisons: role마다 2,330/2,330 exact argmax
- maximum normalized tolerance ratio: 모든 role/mode `<=0.167` (`<=1` 요구)
- strict-valid free traces: 1,280/1,280
- 네 role·5 repetitions free token/byte output: exact
- target-call 수와 final observed-cache length: exact

따라서 speed 차이에 output/quality 변경이나 잘못 정렬된 KV cache가 섞이지 않았다.

## Draft가 감당할 수 있는 여유

Primary 35% upper-bound gate를 median 수준에서 유지하려면 block runtime 외에 남는 대략적 시간은:

- controlled: `66.917 × 0.65 - 24.139 = 19.36 ms`
- free: `81.550 × 0.65 - 27.985 = 25.02 ms`

이 값은 draft budget 보장이 아니다. 실제 rejection은 block target call 수를 늘리고 rollback/correction
비용도 만든다. 단지 target-only ceiling이 지나치게 낮아 learned draft를 시작할 이유가 없는 경우를
배제했다.

## 과거 W72 speculation과의 관계

W72 byte runtime의 exact learned speculation은 target invocation을 23.2% 줄였지만 E2E는 9.98%만
줄여 20% gate를 실패했다. 이번 16K target은 128 raw bytes를 중앙 25--32.5 token으로 표현하고,
perfect block 4가 target calls를 약 72% 줄였다. Byte-local W72보다 block kernel geometry가 훨씬
유리하다는 직접 증거다.

그러나 이번에는 acceptance가 100%이고 draft cost가 0이다. W72 실패에서 얻은 교훈은 그대로
유효하다. Upper bound를 실제 개선으로 오독하지 않고 proposal acceptance, draft head/vocabulary
비용, cache rollback, correction과 bonus를 모두 포함해야 한다.

## 연구 계획 수정

Dense-vocabulary branch 중단 결정은 바꾸지 않는다. 이번 통과는 16K vocabulary 자체가 stable
actual winner였다는 뜻이 아니라, 이미 quality-qualified된 16K target에서 **sequential target calls를
줄일 kernel headroom**이 크다는 뜻이다.

사전 계획에 따라 다음 한 단계만 연다.

1. block 4를 고정한다. 한 cycle은 최대 3 draft tokens를 제안하고 target block이 세 proposal과
   한 bonus/correction position을 평가한다.
2. 동일 16K tokenizer를 쓰는 값싼 generic n-gram/copy draft를 mandatory control로 먼저 구현한다.
3. 작은 learned draft 또는 multi-token head는 total parameters, resident bytes, draft forward time과
   16K output-vocabulary 비용을 전부 보고한다.
4. Proposal feasibility는 calibration-only로 고정된 후보/rule을 선택할 수 있지만, actual speed
   성공은 새로운 sealed timing protocol에서만 판정한다.
5. exact target greedy bytes, target cache rollback, correction/bonus와 strict UTF-8 stop을 보존한다.
6. target-only upper bound 수치를 actual speedup으로 사용하지 않는다.
7. generic cheap draft를 이기지 못한 한국어-specific draft에는 novelty를 부여하지 않는다.

최신 SpecVocab 결과처럼 draft vocabulary projection 자체가 병목일 수 있으므로 16K full draft head를
당연한 기본값으로 두지 않는다. 그러나 dynamic vocabulary restriction 역시 선행연구이므로, 그것만
구현해 새 방법이라고 주장하지 않는다.

## 주장 경계

말할 수 있음:

- one seed/one Apple-MPS session의 trained 16K target에서 exact perfect block 4 target-side E2E가
  controlled 63.9%, free 65.7% 줄었다.
- target-only headroom이 사전 gate를 통과해 same-tokenizer draft fail-fast를 실행할 근거가 생겼다.

말할 수 없음:

- 실제 learned/n-gram speculative runtime이 빠르다.
- draft cost, acceptance, rollback을 포함해 효율이 개선됐다.
- block 8이 최종 선택이다.
- memory가 줄었다.
- 다른 model seed, scale, hardware나 일반 한국어 LLM에 재현된다.
- publication-ready actual efficiency claim이 완성됐다.
