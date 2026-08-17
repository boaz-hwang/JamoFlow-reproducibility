# Fresh-v2 16K retrieval mechanism audit protocol

> 작성일: 2026-08-15
>
> 상태: **효율 주장이 아닌 고정 development mechanism screen**
>
> 선행 결과: `docs/167-fresh-v2-16k-retrieval-actual-result-and-free-path-correction.md`

## 1. 왜 이 단계를 여는가

Generic hybrid retrieval은 봉인된 joint gate를 실패했지만 exact target-greedy free inference에서는
26.244% 빨랐다. Controlled gold continuation과 free target trajectory의 차이는 prompt-copy proposal
acceptance에서 가장 컸다. 이 결과만 보고 한국어 router를 만들면 feature shopping이 된다. 따라서
한 가지 언어학적 가설만 먼저 고정해 descriptive event replay로 검사한다.

이 단계는 latency를 다시 재거나 이전 실패를 구제하지 않는다. 같은 closed 64-case development
set을 오직 mechanism 분석에 재사용한다.

## 2. 사전 고정 가설

Free-running hybrid role의 `prompt_lookup` proposal cycle만 본다.

> 현재 committed token이 한글 어절 내부에서 끝날 때는 공백 직후보다 철자·형태 제약이 강하므로,
> accepted tokens per proposal cycle이 더 높다.

고정 contrast:

`within_hangul_eojeol - after_whitespace`

Boundary는 proposal과 target outcome을 보기 전에 현재 committed raw prefix만으로 분류한다.

- `within_hangul_eojeol`: UTF-8 DFA accept state이고 마지막 scalar가 Hangul syllable/Jamo
- `after_whitespace`: UTF-8 DFA accept state이고 마지막 scalar가 whitespace
- `inside_utf8_scalar`, `after_other`: secondary descriptive strata

## 3. 고정 screen

다음을 모두 요구한다.

- 두 primary stratum 각각 proposal cycles `>=32`
- 두 stratum을 모두 가진 cases `>=16`
- case별 accepted-tokens/cycle 차이의 mean `>=0.25`
- 10,000회 paired-case bootstrap 95% lower `>0`

하나라도 실패하면 Hangul boundary router 가설을 종료한다. Secondary boundary, proposal-crossing,
table confidence 또는 prompt-match length가 좋아 보여도 primary를 대체하지 않는다.

## 4. Replay와 integrity

Checkpoint에서 measured 64 cases의 controlled/free target trace를 다시 생성한다. 각 role의 proposal
state machine을 model forward 없이 exact token equality로 재생한다. 이는 speculative acceptance가
proposal과 이미 고정된 target output의 equality로 결정되기 때문이다.

각 event에서 원문/token IDs는 저장하지 않고 다음 aggregate용 값만 메모리에 둔다.

- mode, role, proposal source
- pre-proposal boundary class와 current eojeol Hangul length
- proposal/accepted token counts와 outcome
- whitespace crossing, prompt match length
- corpus first-order/minimum confidence

Event replay의 proposal attempts/tokens/accepted/source totals는 봉인된 actual timing counters의 정확히
1/5이어야 한다. Timed run의 다섯 repetitions과 한 번의 deterministic event replay를 연결하는
integrity check다. Tracked result에는 aggregate와 ordered event commitment만 남기며 raw text/token
sequence는 남기지 않는다.

## 5. 결과에 따른 행동

### Fail

- 이 development set에서 다른 Korean feature를 탐색해 primary를 바꾸지 않는다.
- Hangul-boundary conditional prompt router 분기를 종료한다.
- generic free-path improvement는 prior-work development evidence로만 유지한다.

### Pass

- 새 disjoint Korean document/case set을 먼저 봉인할 수 있다.
- 허용되는 candidate는 사전 가설과 직접 연결된 Hangul-boundary-aware proposal-depth policy 하나다.
- generic hybrid와 target/checkpoint/table bytes/max draft/timing 범위를 exact match한다.
- actual free target-greedy E2E 추가 개선이 primary이고 controlled replay는 secondary diagnostic이다.

Pass는 한국어 효율 개선이나 publication readiness를 뜻하지 않는다.

## 6. 주장 경계

말할 수 있는 것:

- closed development target trajectories에서 pre-proposal Hangul/space boundary와 accepted span의 관계
- 그 관계가 고정 coverage/effect/uncertainty screen을 통과했는지

말할 수 없는 것:

- latency 또는 actual efficiency가 추가 개선됐음
- boundary feature가 causal함
- 새 disjoint documents에서도 관계가 유지됨
- Korean-aware algorithm이 generic retrieval보다 나음
- 논문 기여가 확립됨
