# Retrieval mechanism result and boundary-router closure

> 작성일: 2026-08-15
>
> V2 plan SHA-256: `dcd9a7c301bf28943cabef350a73d9821b8ceeae9587e7366071efb782734093`
>
> Summary SHA-256: `73044422788ee992bc76c31e2ac25965d903d8846137cc3fa0b00e2f66fd601c`
>
> 판정: **사전 고정 Hangul-boundary 가설 실패**

## 1. Integrity

- checkpoint에서 controlled/free target traces 재생성
- 9,335 proposal/no-proposal events deterministic replay
- proposal attempts, proposal tokens, accepted tokens, corpus/prompt source totals이 actual timing
  counters의 정확히 1/5과 일치
- raw text와 token sequence 미공개; ordered event commitment만 기록
- V1은 serialization 전에 실패했고 V2에서 scalar normalization만 교정

따라서 결과는 timing artifact의 aggregate boolean을 재사용한 것이 아니라 target trace와 retrieval
state machine을 별도로 재구성한 mechanism evidence다.

## 2. Primary result

Free hybrid의 prompt-lookup proposal cycles에서:

| pre-proposal boundary | cycles | accepted tokens/cycle |
|---|---:|---:|
| within Hangul eojeol | 244 | 1.406 |
| after whitespace | 101 | 1.752 |

두 stratum의 cycle coverage는 통과했지만 둘을 모두 가진 case는 13개로 고정 최소 16에 못 미쳤다.
더 중요하게 paired-case mean contrast는:

`within Hangul eojeol - after whitespace = -0.246 tokens/cycle`

로 예상한 `>=+0.25`와 방향부터 반대다. Coverage 부족 때문에 bootstrap interval은 계산하지 않고 JSON
`null`로 기록했다. Primary gate는 실패다.

## 3. 해석

“공백 직후는 semantic uncertainty가 높으므로 prompt copy가 덜 맞는다”는 직관은 이 target
trajectory에서 성립하지 않았다. Prompt lookup은 공백 직후에도 이미 나온 구절이나 문장 패턴을
복사할 수 있고, 실제 accepted span은 오히려 더 길었다. 단순 boundary class만으로 proposal depth를
줄이는 router는 generic hybrid의 좋은 free-path cycle을 제거할 위험이 있다.

Secondary descriptive row에서는 free hybrid가 Hangul eojeol 내부에 있으면서 proposal을 전혀 얻지
못한 cycle이 508개였다. 이는 eojeol completion coverage의 잠재적 빈 공간이지만, 이번 protocol은
secondary fallback을 명시적으로 금지했다. 따라서 이 수치로 boundary-router 실패를 구제하거나 곧바로
새 method를 승인하지 않는다.

## 4. 결정

1. `within_hangul_eojeol`에서 full prompt draft, `after_whitespace`에서 short/disabled draft를 쓰는
   conditional router 분기를 종료한다.
2. 같은 64 cases에서 다른 boundary, prompt-match length, whitespace-crossing threshold를 탐색해
   primary를 교체하지 않는다.
3. Generic retrieval의 free 26.2% development speedup은 유지하지만 Korean-specific evidence로
   승격하지 않는다.
4. 508 no-proposal cycles는 오직 새 연구 질문을 만들 수 있는 exploratory observation이다. 이를
   사용하려면 먼저 Korean eojeol/character completion이 speculative decoding에서 이미 다뤄졌는지
   최신 선행연구를 다시 확인하고, generic character/word dictionary 대비 독립 기여를 정의해야 한다.
5. Novelty와 cost-matched control이 명확하지 않으면 retrieval branch 전체를 종료한다. 명확해도
   새 method는 train-only build와 새 disjoint development/confirmation split을 결과 전에 봉인해야 한다.

## 5. 주장 경계

말할 수 있는 것:

- 이 closed target trajectory에서 prompt proposal의 Hangul-inside accepted span이 whitespace-after보다
  높지 않았음
- 단순 Hangul/space boundary depth router의 사전 가설이 실패했음
- 많은 Hangul-inside cycles가 generic proposal을 얻지 못했다는 descriptive count

말할 수 없는 것:

- no-proposal cycles를 Korean completion으로 정확히 채울 수 있음
- eojeol dictionary가 generic character/token dictionary보다 나음
- Korean-aware speculative decoding이 실제 E2E를 개선함
- 새 disjoint experiment 또는 publication claim이 승인됨
