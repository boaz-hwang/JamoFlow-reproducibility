# Trained 16K retrieval-draft literature audit and fail-fast direction

> 작성일: 2026-08-15
>
> 상태: **target-block upper bound 통과 뒤의 실제-draft 방향 수정**
>
> 적용 대상: trained fresh-v2 16K target, one-seed Apple-MPS development preflight

## 결론

Perfect-draft block 4가 controlled 63.9%, free 65.7%의 target-only E2E headroom을
보였으므로 실제 draft를 시험할 시스템 여유는 충분하다. 그러나 2026년 선행연구를 다시 대조하면
단순 prompt lookup, corpus n-gram, hardware-aware draft length, reduced/dynamic draft vocabulary 중
어느 하나도 새 기여가 아니다.

따라서 다음 단계의 목적은 새 방법을 주장하는 것이 아니라 다음 질문을 가장 싸게 판정하는 것이다.

> 같은 trained 16K target에서 draft 조회, target verification, rejection, cache rollback,
> correction/bonus, UTF-8 제약을 모두 포함해도 강한 generic retrieval draft가 ordinary AR보다
> 실제로 빠른가?

이 단계에서 actual E2E가 개선되지 않으면 learned draft나 한국어-specific draft를 학습하지 않는다.
개선되더라도 generic retrieval baseline의 재현에 불과하며, 한국어 기여는 같은 비용의 generic
baseline보다 추가 actual gain을 보여야만 열린다.

## 최신 직접 선행연구가 닫은 주장

### Prompt와 self-output lookup

[SSSD](https://aclanthology.org/2026.acl-long.1530/)는 prompt와 이미 생성한 output을 하나의
n-gram source로 취급하고 외부 corpus datastore와 결합한다. Prefix 길이 1부터 4까지의
continuation tree를 조회하고 hardware-aware speculation length를 사용하며, 별도 학습 없이 최대
2.9배 latency 감소를 보고한다. 그러므로 다음은 novelty가 아니다.

- prompt-local suffix copy
- generated-history cache
- prompt와 corpus lookup의 결합
- 장치별 block/draft 크기 선택

### 비라틴 문자를 위한 corpus dictionary

[DictSpec](https://aclanthology.org/2026.unlp-1.15/)은 비라틴 문자 언어에서 tokenizer fertility가
높을수록 corpus dictionary speculative decoding의 이득이 커진다는 가설을 직접 연구했다.
Unlabeled corpus의 frequent 1/2/3-gram을 tokenizer ID prefix-to-continuation table로 압축하고,
200K entries, minimum probability 0.8의 compact live configuration과 prompt-n-gram fallback을
평가했다. 실제 vLLM hybrid는 최대 1.76배를 보고한다.

따라서 다음 역시 그 자체로는 새 기여가 아니다.

- 한국어 corpus에서 static n-gram dictionary를 만드는 것
- non-Latin/high-fertility 언어라서 retrieval draft가 잘 된다는 것
- dictionary-first, prompt-lookup fallback hybrid
- 2--5 MB 수준의 compact tokenizer-specific table

다만 DictSpec은 Ukrainian/Crimean Tatar를 평가했고 Korean은 평가하지 않았다. 이것은 Korean
replication의 필요성은 만들지만, 단독 method novelty를 만들지는 않는다.

### Training-free multilingual/hardware adaptation

[UniSpec](https://aclanthology.org/2026.acl-long.285/)은 target probability로 n-gram confidence를
갱신하고, device 측정으로 draft tree 크기를 정하며, seven-language benchmark에서 최대 2.6배를
보고한다. [Speculative Decoding and the Curse of Multilinguality](https://arxiv.org/abs/2605.30580)는
11개 언어에서 small learned draft가 multilingual generation에 약할 수 있고, 낮은 acceptance라도
매우 싼 n-gram draft가 더 큰 actual speedup을 낼 수 있음을 보인다.

그러므로 acceptance만 높이는 learned draft는 충분하지 않다. Draft cost와 target verification을
합친 wall time이 최종 기준이다.

### Draft output head와 vocabulary

[SpecVocab](https://aclanthology.org/2026.findings-acl.2000/)은 최신 learned drafter에서 output
embedding/projection이 drafting time을 지배할 수 있음을 보이고, decoding step별 vocabulary
subset을 고르는 방법을 제시했다. 따라서 16K full-vocabulary head를 무비판적으로 쓰지 않으며,
reduced/dynamic vocabulary만으로 novelty를 주장하지도 않는다.

### 더 넓은 retrieval 계열

[Cacheback](https://aclanthology.org/2025.emnlp-main.1581/)은 LRU token n-gram cache만으로,
[SAM-Decoding](https://arxiv.org/abs/2411.10666)은 suffix automaton으로 corpus와 dynamic sequence의
longest suffix를 조회한다. Retrieval data structure 자체를 조금 바꾸는 수준도 본 논문의 충분한
기여가 아니다.

## 고정할 generic primary

실제-draft development preflight의 primary는 다음 hybrid다.

1. target과 동일한 fresh-v2 16K tokenizer를 사용한다.
2. maximum proposal은 고정 block 4에 맞춘 3 tokens다.
3. train split에서만 compact token n-gram table을 만든다.
4. 문맥 order 3, 2, 1 순으로 back off하며 최대 세 proposal을 재귀적으로 낸다.
5. static table이 현재 prefix에 proposal을 내지 못할 때 prompt+self-output suffix lookup으로
   fallback한다.
6. prompt lookup은 suffix length 4, 3, 2, 1 순으로 찾고, 같은 길이에서는 deterministic한
   고정 match order를 쓴다.
7. proposal이 없으면 ordinary one-token AR로 진행한다.

Diagnostic ablation으로 prompt-only와 dictionary-only를 함께 측정할 수 있지만, 이들이 hybrid
primary를 대체하거나 실패 뒤 fallback winner가 될 수는 없다.

## exact speculative transaction

현재 cache에는 마지막 emitted token 직전까지만 들어 있다. 세 proposal `d0,d1,d2`가 있을 때
target에는 다음 네 tokens를 한 번에 넣는다.

```text
[last emitted, d0, d1, d2]
```

네 logit rows는 각각 proposal 세 개와 bonus/correction 위치를 검증한다.

- proposal `j`에서 mismatch면 그 앞 proposal만 accept하고 row `j`의 target token을 correction으로
  emit한다.
- target cache는 `last emitted + accepted prefix`까지만 남기고 rejected suffix를 crop한다.
- 세 proposal이 모두 맞으면 세 개를 accept하고 마지막 row의 target bonus token을 emit한다.
- UTF-8 byte quota가 accepted proposal 중간에서 닫히면 마지막 emitted token 직전까지만 cache를
  남기고 더 이상 correction/bonus를 emit하지 않는다.
- free-running output은 strict-mask greedy AR과 token/byte 단위로 exact해야 한다.

Controlled replay는 sealed gold continuation의 동일 token sequence에서 acceptance와 systems cost를
재는 진단이고, free-running이 실제 target greedy speculative execution이다.

## development gate

이 단계는 기존 64-case development set과 one model seed, one Apple-MPS session을 재사용한다.
이미 target/upper-bound 결과가 알려진 뒤 설계되므로 confirmatory 또는 final-blind라고 부르지 않는다.

Hybrid primary가 다음을 모두 만족해야 다음 단계로 간다.

- controlled와 free E2E median reduction 각각 `>= 10%`
- paired-prompt bootstrap 95% lower bound 각각 `> 0`
- 각 mode에서 `>= 48/64` prompts faster
- free output token IDs와 bytes가 ordinary strict greedy AR과 전부 exact
- full-forward/cache logits, rollback length, target-call/proposal/acceptance accounting 전부 통과
- dictionary lookup, prompt lookup, rollback, argmax/readback, UTF-8 stop을 timer 안에 포함

통과는 generic retrieval baseline이 이 small target/hardware에서도 실제로 작동한다는 뜻만 가진다.
실패하면 retrieval/learned/Korean draft branch를 중단한다. 통과하면 새 disjoint held-out protocol에서
generic hybrid와 parameter/cost-matched Korean-aware extension을 비교할 수 있다.

## 한국어-specific 후속의 최소 요건

Generic hybrid 통과 뒤에도 한국어 방향은 자동으로 열리지 않는다. 후보는 다음 조건을 동시에
만족해야 한다.

1. 한글 음절/어절/형태 경계를 쓰는 부분이 generic token n-gram과 기능적으로 달라야 한다.
2. 동일 table bytes 또는 동일 draft compute budget을 지켜야 한다.
3. acceptance length뿐 아니라 actual E2E를 generic hybrid보다 개선해야 한다.
4. Korean 외 contrast 또는 boundary-ablation으로 이득이 단순 frequency memorization이 아님을 보여야
   한다.
5. UniSpec/SSSD/DictSpec/SpecVocab보다 좁고 명확한 새 mechanism 또는 Korean systems finding을
   제시해야 한다.

현재 가장 그럴듯한 가설은 형태·어절 경계로 table budget을 배분하거나 proposal length/confidence를
조절하는 것이다. 그러나 이는 아직 가설일 뿐이며 generic actual gate 전에는 구현하지 않는다.

## 주장 경계

이번 문헌 검토로 말할 수 있는 것:

- target-only upper bound 때문에 actual retrieval-draft 실험을 할 이유가 있다.
- 최신 선행연구상 generic retrieval hybrid가 mandatory baseline이다.
- 단순 한국어 dictionary/n-gram은 논문 novelty가 아니다.

아직 말할 수 없는 것:

- retrieval draft가 JamoFlow target에서 실제로 빠르다.
- 한국어 구조가 generic retrieval보다 낫다.
- learned draft를 학습할 가치가 있다.
- publication-ready efficiency claim이 있다.
