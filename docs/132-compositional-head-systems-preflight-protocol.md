# 동일 2K 예산 compositional vocabulary systems preflight

> 작성일: 2026-08-14
>
> 상태: 구현·게이트 선고정 단계; full-grid 결과 미관측

## Protocol v1 무효화 기록

첫 v1 plan을 commit한 뒤 benchmark는 model construction 전에 plan을 JSON에서 다시 읽어
검증했다. 이때 assignment audit의 Python tuple이 JSON list로 바뀌어 deterministic rebuild와
동등 비교되지 않는 직렬화 오류가 발견됐다. Benchmark는 바로 중단됐고 full-grid timing, model
construction, 결과 artifact는 생성되지 않았다. v1 plan을 삭제하거나 고쳐 쓰지 않고 감사 흔적으로
보존한다. v2는 audit 필드를 처음부터 JSON list로 canonicalize하며, 아래 role·case·gate·통계
계약은 v1과 동일하다.

## 결론부터

Byte-LengthGain-2K의 최적 분할조차 calibration token 수를 4.153%밖에 줄이지 못했으므로,
same-2K tokenizer를 더 튜닝하는 경로는 종료했다. 다음 질문은 vocabulary를 크게 만들면서 생기는
짧은 sequence를 유지하되, 작은 모델에서 큰 dense embedding/unembedding이 차지하는 parameter와
매 decoding-step 비용을 없앨 수 있는가이다.

이 단계는 학습 품질을 주장하지 않는다. 동일한 2K dense head parameter budget과 동일한
Transformer body에서 8K/16K/32K vocabulary의 **exact full-vocabulary logits**를 계산하는
codebook head가 Apple MPS batch-1 controlled decoding에서 10% 이상의 실제 systems 여유를
만드는지만 검사한다. 이 게이트를 통과해야만 학습 실험을 허용한다.

## 연구 가설과 신규성 경계

큰 vocabulary, 저차원 embedding, product/compositional code, factorized output은 각각 기존
연구가 있는 구성요소다. 따라서 이 연구가 주장할 수 있는 기여는 다음 교집합으로 제한한다.

1. BPE-2K와 같은 Transformer body 및 같은 trainable head parameter 수
2. 한국어 byte-BPE 8K 이상이 제공하는 실제 autoregressive step 감소
3. approximate retrieval 없이 standard cross-entropy에 사용할 수 있는 exact logits
4. dense-large-vocabulary 및 같은 예산 low-rank control보다 나은 실제 inference efficiency
5. 학습 뒤 Hangul assignment가 compute-identical generic·shuffled control보다 품질을 보존한다는
   별도 증거

이번 preflight는 1--3의 systems 가능성과 4의 random-weight runtime만 본다. 5는 후속 학습을
통과해야 하며, 그 전에는 `Korean-specific contribution`을 주장하지 않는다.

## 모델 계약

공통 body는 기존 BPE-2K×8L 모델과 동일하다.

- hidden size: 384
- intermediate size: 1,536
- layers: 8
- attention heads / KV heads: 6 / 6
- body parameters excluding tied vocabulary: 18,880,896
- BPE-2K tied vocabulary budget: `2,048 × 384 = 786,432`
- baseline total parameters: 19,667,328

Compositional head는 16개 독립 codebook, codebook당 128개 row를 사용한다.

```text
trainable code rows = 16 × 128 = 2,048
head parameters     = 2,048 × 384 = 786,432

E(token) = (1 / sqrt(16)) × Σ_m C[m, code(token,m)]
logit(token | h) = <h, E(token)>
```

출력은 먼저 `h`와 2,048 code row의 dot product를 계산한 뒤, vocabulary token마다 16개 값을
gather-add한다. 이 값은 합성한 dense tied weight로 계산한 logit과 수치 허용오차 안에서 정확히
같다. 후보 pruning, ANN/MIPS, invalid-token mask는 쓰지 않는다. `V×16` int64 assignment buffer와
gather 비용은 runtime과 memory에서 제외하지 않는다.

### 13개 preflight role

| vocabulary | dense | 동일 예산 low-rank | generic code | Hangul code |
|---:|---:|---:|---:|---:|
| 2,048 | 19,667,328 | — | — | — |
| 8,192 | 22,026,624 | 19,669,888 (rank 92) | 19,667,328 | 19,667,328 |
| 16,000 | 25,024,896 | 19,667,328 (rank 48) | 19,667,328 | 19,667,328 |
| 32,000 | 31,168,896 | 19,658,112 (rank 24) | 19,667,328 | 19,667,328 |

모든 role은 같은 seed로 같은 Transformer body를 만든다. Dense-large role은 vocabulary head가
얼마나 비싼지 보여 주는 ceiling control이고, low-rank role은 표준 factorization 계열의
compute-matched control이다. 작은 정수 rank 때문에 8K/32K low-rank 총 parameter는 baseline과
각각 +2,560/-9,216 차이가 난다. 그 차이는 결과에 그대로 공개한다.

## 결정적인 token code

모든 token은 16개 `[0,127]` index를 갖는다. 마지막 세 slot은 base-128 token ID라 전체 tuple의
유일성을 보장한다. 따라서 linguistic surface가 같거나 UTF-8 fragment가 불완전해도 token
collision은 없다.

| slot | 의미 |
|---:|---|
| 0--2 | 첫 Unicode scalar 또는 invalid-byte pseudo scalar의 base-127 digits |
| 3--5 | 마지막 scalar의 base-127 digits |
| 6--8 | 첫 scalar의 auxiliary code |
| 9--11 | 마지막 scalar의 auxiliary code |
| 12 | byte length, 127에서 포화 |
| 13--15 | token ID의 base-128 digits |

Generic assignment는 auxiliary slot에 domain-separated SHA-256 code를 쓴다. Hangul assignment는
완성형 음절이면 onset/vowel/coda index를 쓰고, 나머지는 같은 generic fallback을 쓴다. 이후
학습 단계의 shuffled-Hangul control은 byte-length strata 안에서 6개 auxiliary slot을 함께
permutation하여 slot histogram과 graph를 유지한다. Shuffled role은 runtime graph가 Hangul role과
동일하므로 이번 13-role random-weight timing에는 중복 추가하지 않고, 품질 ablation에서 반드시
추가한다.

## 실제 systems 측정

기존 Korean BPE systems frontier와 같은 calibration document cases를 재사용한다.

- 6 warmup document cases
- 36 measured document cases
- case마다 strict-UTF-8 128-byte prompt + 128-byte continuation
- document/cluster는 case 사이에서 서로 다름
- role마다 자기 2K/8K/16K/32K tokenizer로 같은 raw bytes를 encode
- batch size 1, Apple MPS, float32, KV-cache 사용
- measured case마다 3 repetitions
- 13 role을 cyclic Latin schedule로 회전하고 완전 cycle마다 첫 role을 보존한 채 나머지 순서를 반전

Timer에는 model prefill, cached decode, exact full-vocabulary head, argmax kernel, MPS synchronization이
포함된다. Tokenizer encode와 model construction은 제외한다. Continuation은 모든 role에 같은 raw
bytes가 되도록 gold token IDs를 feedback하는 controlled replay다. 따라서 이것은 free-running
generation latency가 아니며, 최종 publication efficiency 증거가 아니다. 후속 trained model이
품질을 통과하면 별도의 free-running actual-inference protocol에서 host feedback, stop rule,
selector/head/cache/synchronization을 모두 포함해 다시 측정한다.

### correctness gate

각 13 role의 6 warmup case에서 다음을 모두 확인한다.

- full no-cache logits 대 one-token sequential cache logits
- full logits 대 parallel-prefill + continuation cache logits
- normalized tolerance ratio `<= 1` (`atol=1e-4`, `rtol=2e-5`)
- 모든 비교 위치에서 exact argmax equality
- 실제 parameter count가 sealed role spec과 일치
- continuation step array가 tokenizer로 독립 재구성한 값과 exact 일치

하나라도 실패하면 timing 결과와 무관하게 branch를 중단한다.

## 사전 고정 opportunity gate

각 non-baseline role은 BPE-2K dense baseline과 같은 36 prompt에서 paired comparison한다. 먼저 prompt
안의 세 repetition을 median으로 접고, prompt를 10,000회 bootstrap한다.

한 role의 pass 조건은 모두 다음과 같다.

1. cache/full correctness 전부 통과
2. aggregate end-to-end median reduction `>= 10%`
3. paired-prompt bootstrap 95% lower bound `> 0`
4. continuation step reduction `>= 10%`
5. 36 prompt 중 candidate가 빠른 prompt `>= 24`

후속 학습에 사용할 vocabulary size는 **generic code와 Hangul code가 모두 통과한 가장 작은
8K→16K→32K size**다. 한쪽만 통과했을 때 다른 쪽을 버리거나 차순위 gate를 완화하지 않는다.
어느 size도 공동 통과하지 못하면 compositional-head branch를 종료한다.

Dense와 low-rank control도 같은 gate로 보고하지만 size 선택의 필수조건은 아니다. 이 controls는
후속 결과가 단순한 vocabulary 길이, parameter 증가, 표준 저차원 factorization 중 무엇으로
설명되는지 판단하기 위한 것이다.

## Evidence DAG와 결과 관측 경계

1. implementation, tests, 이 문서, gate를 한 commit에 고정한다.
2. clean tree에서 plan을 생성하고 별도 commit한다.
3. plan commit에서 13-role MPS benchmark를 한 번 실행한다.
4. raw timing NPZ와 correctness report를 `results/.../evidence`에 publish하고 별도 commit한다.
5. 다음 clean commit에서 summarizer가 plan/tokenizer/case/parameter/correctness/array hash와 exact
   continuation steps를 재구성한 뒤 summary를 생성한다.
6. summary를 별도 commit한다.

Evidence report와 timing은 각각 Git history가 한 번뿐인 current-HEAD blob이어야 한다. 이 구조는
요약 전에 ignored timing을 바꾸는 실수를 막지만, 로컬 실행자가 최초 evidence commit 전에 여러
번 실행하고 선택적으로 하나만 남기는 행위를 암호학적으로 막지는 못한다. 따라서 이 단계는
`prospectively Git-sealed systems opportunity test`로만 부르며 one-shot/public preregistration이라고
주장하지 않는다.

봉인 전에는 13개 role을 각각 MPS에 설치해 two-token input의 one-step forward shape와 parameter
contract가 동작하는지만 확인했다. 앞선 구현 과정에서 2K/8K 일부 role의 짧은 engineering timing도
관측했지만 해당 smoke 수치는 evidence로 보존하지 않았고, role grid, 10% threshold, size order를
바꾸는 데 사용하지 않았다. Full 13-role×36-case 결과는 plan 뒤에 처음 생성한다.

## 통과 후 학습 방향

게이트가 통과하면 선택된 한 size에서 먼저 one-seed quality 실험을 선고정한다.

- BPE-2K dense baseline
- selected-size same-body dense ceiling
- selected-size tied low-rank
- selected-size generic code
- selected-size shuffled-Hangul code
- selected-size Hangul code

동일 raw-byte train stream, optimization token/byte budget, initialization policy 및 evaluation split을
고정하고 BPB를 비교한다. Hangul 기여는 `Hangul > shuffled-Hangul` 및 `Hangul > generic`에서만
판단한다. 단순히 BPE-2K보다 품질이 좋거나 codebook이 빠르다는 사실을 한국어 특화 효과로
해석하지 않는다.

One-seed에서 matched-quality와 실제 latency 여유가 없으면 scale·multi-seed로 가지 않는다.
통과할 때만 가능한 가장 큰 Mac-feasible model, 다중 seed, 새 sealed final Korean test, free-running
actual inference로 확장한다. 최종 논문 기준은 random-weight preflight가 아니라 **품질을 보존한
trained model의 실제 end-to-end inference가 10% 이상 개선되는가**이다.
