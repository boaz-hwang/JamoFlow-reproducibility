# EXAONE resource calibration result and actual decision

> 작성일: 2026-08-15
>
> 상태: **baseline resource gate 통과, actual candidate comparison 진행**

## 결론

EXAONE-3.5-7.8B-Instruct 4-bit ordinary greedy baseline은 현재 M4 Pro 48GB 환경에서 자원 gate를
통과했다. 봉인된 schedule rule이 선택한 첫 조합은 다음과 같다.

- independent fresh-process sessions: **5**
- measured case별 inner repetitions: **3**
- warmup/measured cases: **8 / 64**
- prompt/output: **128 / 128 EXAONE tokens**
- candidate가 baseline보다 2배 느리다는 상한 가정 아래 projected campaign: **2.740시간**

따라서 8B actual branch를 중단할 자원 근거는 없다. 다음 단계에서 candidate와 baseline을 같은 case,
session, repetition, order-balanced schedule로 직접 비교한다.

## baseline-only 관측값

64 measured cases aggregate는 다음과 같다.

- median per-case end-to-end: **3.0859초**
- p95 per-case end-to-end: **3.1051초**
- measured total: **197.4231초**
- aggregate generation throughput: **41.4946 output tokens/s**
- model identity/load 검증: **2.0957초**
- warmup total: **24.5542초**

Timer에는 prompt tokenization, fresh-cache prefill, 128 cached greedy decode calls, full detokenization, final
MLX synchronize가 포함됐다. Case/model load와 post-timer correctness replay는 제외됐다.

## 메모리

- exact model parameters: **7,818,448,896**
- retrieval table loaded: **false**
- MLX active before/after: **4.3983 / 4.3983GB**
- MLX peak active: **4.6951GB**
- process peak RSS: **4.9432GB**
- conservative observed working set: **4.9432GB**
- MLX recommended working set: **40.2009GB**
- observed fraction: **12.30%**
- 75% safety gate: **pass**

최댓값은 model-load 직후와 종료 뒤 `active+cache`, MLX peak active, process peak RSS 중에서 골랐다.
Candidate table과 draft block이 추가되는 actual session에서는 role별 memory를 다시 측정한다.

## 무결성

- V3 plan SHA-256: `436bc4a3ae2974f5ca692cd6ea91a846ee080872d95e8d1b4ba3bf477f93f382`
- baseline NPZ SHA-256: `74134eecf00b10f7ebd99c70de52752baaaa0f947e7eca514cd466d12d210ddc`
- result summary SHA-256: `e0738309b50931eab9721ae9729e266218d6695dd1792043117c38bb596d06a3`
- output token hashes were recomputed from all 72×128 token IDs
- timing summary was independently reconstructed from the ignored NPZ
- exact compatibility model-file hashes and parameter count were revalidated
- candidate/table path was not loaded or executed

## 두 번의 사전 실패가 의미하는 것

V1은 first trial 전에 MLX config key alias 오류로 중단됐다. V2는 candidate를 열기 전에 generated BPE
sequence의 과도한 `decode→encode` identity gate로 중단됐다. 두 실패는 각각 별도 plan, active-marker
payload, tracked invalidation record로 보존했다. V3에서 workload·threshold·case·schedule 후보를 바꾸지
않았고 loader schema와 correctness 정의만 교정했다.

V2가 baseline generation에 진입했으므로 V3를 완전한 baseline-output-blind 설계라고 부르지 않는다.
그러나 V2에서는 숫자 latency·output aggregate가 저장 또는 출력되지 않았고 candidate 결과는 전혀
관측되지 않았다.

## 다음 actual stage의 고정 방향

다음 plan은 resource result를 exact dependency로 사용하되 다음을 새로 봉인한다.

- 5 fresh-process sessions × 3 inner repetitions
- session×case 안에서 candidate/reference order balance
- ordinary greedy와 retrieval candidate의 exact generated token-ID equality
- candidate proposal/call/acceptance trace의 독립 재구성
- primary: free-running 128-token end-to-end latency reduction
- secondary: target forward-call reduction, acceptance, TTFT/decode decomposition, role별 memory
- 통계 단위: repetitions를 cell median으로 접은 session×case paired cells
- candidate 결과를 본 뒤 case, draft cap, table, threshold, session 수를 바꾸지 않음

이 단계까지는 효율 개선을 보였다는 결론이 아니다. 논문 가치 여부는 다음 actual paired comparison이
실제로 유의하고 재현 가능한 end-to-end 개선을 보일 때만 열린다.
