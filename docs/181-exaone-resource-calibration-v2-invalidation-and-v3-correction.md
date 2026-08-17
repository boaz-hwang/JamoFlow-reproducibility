# EXAONE resource calibration V2 invalidation and V3 correction

> 작성일: 2026-08-15
>
> 상태: **V2 무효화, V3 재봉인 전 correctness 교정 기록**

## V2에서 드러난 문제

V2는 model load와 baseline generation 경로에 진입했으나 tracked NPZ나 summary를 공개하기 전에
`generated token/text round trip differs`로 중단됐다. Candidate와 retrieval table은 실행되지 않았다.
Baseline 숫자 latency와 output도 파일이나 stdout에 노출되지 않았다.

원인은 generated text의 유효성 문제가 아니라 correctness 조건의 정의 오류였다. BPE tokenizer에서
임의의 token sequence를 문자열로 decode한 뒤 다시 encode하면 동일 문자열의 canonical segmentation으로
합쳐질 수 있다. 따라서 다음 두 명제는 다르다.

1. token sequence의 detokenization이 결정적이고 strict UTF-8 문자열을 만든다.
2. 그 문자열을 다시 tokenize했을 때 원래의 비정규 token segmentation까지 복원한다.

모델 추론 correctness와 candidate 비교에 필요한 것은 1번 및 candidate와 baseline의 generated token-ID
exact equality다. 2번은 일반 BPE 출력에 요구할 수 없는 과도한 조건이다.

## V3 교정

V3는 fixed 128-token workload와 timer boundary를 바꾸지 않는다.

- prompt text의 `decode → encode` exactness는 유지한다. Timed tokenization input identity를 지킨다.
- timed region의 full prompt+output detokenization과 final synchronize를 유지한다.
- timer 밖에서 같은 token IDs를 한 번 더 decode하여 detokenization 결정성을 검사한다.
- resulting Python string은 strict UTF-8 encode가 가능해야 한다.
- generated full sequence의 `decode → encode` token-ID identity는 요구하지 않는다.
- 이후 actual comparison에서는 candidate와 ordinary baseline의 128 generated token IDs 및 decoded hash가
  exact하게 같아야 한다.

즉 correctness를 완화해 서로 다른 모델 출력을 허용한 것이 아니라, tokenizer의 비정규 segmentation을
model-output 오류로 오인하던 조건을 제거한 것이다.

## 봉인과 해석 경계

V2 active payload와 hash, V2 plan, V2 invalidation record를 V3 dependency로 추가한다. V1/V2 namespace는
그대로 보존하고 V3는 새 namespace만 사용한다.

V2에서는 baseline generation 경로가 한 번 이상 진입했으므로 V3 수정은 완전한 baseline-output-blind
설계라고 부르지 않는다. 다만 어떤 숫자 latency나 output도 공개·저장되지 않았고, candidate 결과는 여전히
전혀 관측되지 않았다. 변경 범위는 traceback이 직접 드러낸 tokenizer correctness 조건 하나뿐이며,
schedule 후보, 8시간 budget, memory gate, case set, output-token count는 바꾸지 않았다.
