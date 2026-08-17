# Fresh-v2 16K retrieval actual result and free-path correction

> 작성일: 2026-08-15
>
> 상태: **봉인된 joint gate 실패; generic retrieval 개발 결과**
>
> Plan SHA-256: `77e9d7e3589e5622c2f91af8ab960b2abc8cd24a13d9ce0ade3ec95794e0f929`
>
> Summary SHA-256: `3083986fc9fe3d2da8800143bd6d2aee89817206d24dcd51d31cd97e8967281b`

## 1. 결론

고정 primary `hybrid_retrieval_block_4`는 free-running actual inference에서는 강한 양성 결과를
보였지만 controlled same-output에서는 사전 고정 10% gate를 넘지 못했다. 두 mode 모두 통과해야
하는 joint gate이므로 이 개발 실험의 공식 판정은 실패다.

| mode | E2E reduction | paired-prompt 95% interval | faster prompts | gate |
|---|---:|---:|---:|---|
| controlled replay | 5.310% | [0.683%, 10.189%] | 45/64 | fail |
| free-running UTF-8 greedy | 26.244% | [13.877%, 32.095%] | 61/64 | pass |

임계값을 낮추지 않고, diagnostic prompt-only/corpus-only role로 primary를 교체하지 않는다. 이
결과는 한국어-specific follow-up을 원래 계약 아래 자동 승인하지 않는다.

## 2. Correctness와 evidence integrity

- 2,560 timed trials 완료
- warmup 8 cases와 independent measured 64-case checkpoint replay pass
- target full/cache argmax 및 normalized tolerance pass
- 네 역할·두 mode output token/raw bytes/cache one-token lag exact
- free output 1,280 traces가 role/repetition 전체에서 exact하고 strict UTF-8 replay pass
- checkpoint, tokenizer, table, upper-bound result와 implementation identity는 sealed plan과 exact

따라서 controlled/free 차이는 output 또는 cache 오류로 설명되지 않는다.

## 3. Diagnostic roles

| role | controlled reduction | free reduction | controlled acceptance | free acceptance |
|---|---:|---:|---:|---:|
| prompt lookup | -1.519% | 22.578% | 11.49% | 48.31% |
| corpus n-gram | 5.006% | 7.883% | 86.57% | 69.77% |
| hybrid primary | 5.310% | 26.244% | 28.93% | 51.66% |

Corpus-only controlled interval은 [0.395%, 9.856%]이고 51/64 prompts가 빨랐지만 point 10%를
넘지 못했다. Free에서는 interval lower가 -0.308%이고 41/64뿐이다. Prompt-only는 free에서
강하지만 controlled에서 regression이다. 어느 diagnostic도 joint gate를 통과하지 않으므로
fallback으로 결과를 구제할 수 없다.

## 4. 실제 비용과 mechanism

### Controlled replay

- baseline target calls median: 25
- corpus-only: 23, E2E 65.940 → 62.639 ms
- hybrid: 22, E2E 65.940 → 62.439 ms
- hybrid lookup median: 0.597 ms
- hybrid proposal tokens: 2,955, accepted 855
- hybrid source cycles: corpus 470, prompt 780

Corpus proposal 자체는 gold continuation과 잘 맞았지만 한 cycle에서 accepted token은 평균 1.184에
불과해 target call을 2회만 줄였다. Corpus miss 뒤 prompt fallback은 fixed continuation에서 자주
틀려 hybrid 전체 acceptance를 28.9%로 낮췄다. Target block forward가 single-token forward보다
싸지 않으므로 세 call 감소만으로는 10%를 남기지 못했다.

### Free-running greedy

- baseline target calls median: 32.5
- prompt-only: 24, E2E 88.189 → 68.278 ms
- hybrid: 22, E2E 88.189 → 65.045 ms
- hybrid lookup median: 0.529 ms
- hybrid proposal tokens: 8,905, accepted 4,600
- hybrid source cycles: corpus 460, prompt 2,820

실제 target-greedy trajectory에서는 prompt/self-output copy가 훨씬 잘 맞았다. Hybrid는 accepted
tokens/cycle 1.402를 얻어 target calls를 약 32.5에서 22로 줄였고 lookup 비용보다 훨씬 큰 decode
절감을 남겼다.

## 5. 무엇을 배웠는가

첫째, perfect-draft 64%+ headroom의 상당 부분은 deployable generic retrieval로 실제 전환될 수
있다. 적어도 이 one-seed Apple-MPS free-generation workload에서 exact output을 유지하면서 26.2%
E2E improvement가 관측됐다.

둘째, acceptance scalar만으로 효율을 예측할 수 없다. Corpus-only controlled acceptance는 86.6%지만
proposal coverage와 accepted span이 짧아 5.0% improvement에 그쳤다. 반대로 free hybrid는 더 낮은
51.7% token acceptance로 26.2%를 얻었다. Proposal cycle당 accepted span과 target-call reduction이
더 직접적인 변수다.

셋째, controlled gold continuation과 deployable target-greedy trajectory는 retrieval draft에 서로 다른
분포다. Controlled replay는 같은 output 길이에서 kernel robustness를 보는 좋은 stress test지만,
실제 greedy model output에 존재하는 prompt-copy/repetition 구조를 제거한다. 사용자의 핵심 기준인
“모델이 실제 추론할 때의 효율”과 완전히 같은 estimand는 아니다.

## 6. 계획 수정 여부

기존 결과의 판정은 수정하지 않는다. Joint gate는 실패했고 이 dataset에서 threshold/source/block
size를 재튜닝하지 않는다. 다만 원래 fail 행동의 전제였던 “joint fail이면 deployable free path에도
headroom이 없다”는 가정은 실제 결과와 맞지 않는다. 따라서 연구 계획은 다음처럼 최소 수정한다.

1. 이 결과는 generic prior-work baseline의 **development evidence**로만 보존한다.
2. 현재 64 cases에서는 추가 timing 또는 policy selection을 하지 않는다.
3. 먼저 저장된 aggregate와 별도 instrumented development replay로 source, accepted-prefix length,
   Hangul/whitespace/eojeol boundary별 mechanism을 분석한다. 이 분석은 효율 claim이 아니다.
4. 한국어 구조가 generic prompt/corpus retrieval보다 더 나은 proposal span 또는 rejection 회피를
   만들 수 있다는 명시적 가설이 생길 때만 새 disjoint case set을 봉인한다.
5. 새 비교의 actual primary estimand는 exact target-greedy free generation으로 두고, controlled
   replay는 secondary robustness/mechanism diagnostic으로 둔다. 이 변경은 새 data 전에 봉인한다.
6. Generic hybrid를 그대로 primary baseline으로 유지하고 동일 table bytes/draft length/resident
   memory/timer 범위에서 한국어-aware 방법이 E2E를 추가 개선해야 novelty 후보로 인정한다.

이는 실패 gate를 사후에 통과로 바꾸는 것이 아니다. 실패가 드러낸 estimand 불일치를 다음 disjoint
실험에서 사전에 교정하는 것이다.

## 7. 주장 경계

현재 말할 수 있는 것:

- 이 exact model/seed/cases/Apple-MPS의 free greedy inference에서 generic hybrid가 ordinary AR보다
  26.2% 빨랐고 61/64 prompts에서 빨랐음
- exact target output과 strict UTF-8 validity를 유지했음
- controlled same-output에서는 고정 joint gate를 실패했음
- retrieval utility가 target trajectory의 copy/repetition 구조에 강하게 의존함

말할 수 없는 것:

- joint 개발 protocol을 통과했다는 주장
- 한국어 구조가 효율을 개선했다는 주장
- generic retrieval 자체의 novelty
- unseen Korean documents, 다른 decoding, seed, scale, hardware에서 같은 speedup
- publication-grade 최종 결과
