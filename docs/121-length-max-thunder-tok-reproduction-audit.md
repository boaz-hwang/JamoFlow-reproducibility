# Length-MAX·Thunder-Tok 재현 감사와 same-2K 실험 보정

> 작성일: 2026-08-14
>
> 상태: primary-source audit; 결과를 보지 않은 다음 실험 설계 입력

## 결론

BPE one-seed frontier에서 2K×8L만 품질 자격을 통과했기 때문에, 다음 질문은 큰 vocabulary를
더 만드는 것이 아니라 **같은 2,048 output classes와 같은 19.67M-parameter graph에서 어떤
vocabulary construction과 segmentation이 한국어 byte 수당 decode step을 줄이면서 BPB를
보존하는가**다.

이 단계에서 Length-MAX를 단순히 `DP로 최소 토큰 수를 만드는 공개 tokenizer`라고 구현하면
안 된다. 논문과 현재 확인 가능한 동명 PyPI 패키지는 다음 핵심 지점에서 다르다.

- 논문은 frozen vocabulary 적용을 left-most-longest trie/DFA로 명시한다.
- PyPI 패키지는 global minimum-token DP를 기본 적용 경로로 제공한다.
- 논문은 raw corpus에 pre-tokenization 없이 적용한다고 서술한다.
- 패키지는 `split_whitespace()`로 모든 공백을 단어 끝 표식 하나로 정규화한다.
- 논문은 single UTF-8 characters에서 시작한다. 패키지도 관측 Unicode character를 기본
  alphabet으로 쓰므로, 한국어 2K에서는 byte fallback과 고정 2K budget을 동시에 보장하지
  못한다.

따라서 JamoFlow는 논문, 패키지, 자체 byte-exact 변형을 서로 다른 artifact와 이름으로
분리한다. Thunder-Tok의 한국어 결과 역시 구조 제약의 존재 이유는 지지하지만, 그 규칙을
그대로 옮기는 것만으로 novelty를 주장하지 않는다.

## 1. 확인한 primary artifacts

### Length-MAX

- 논문: [Length-MAX: Tokenization with an Average Length Objective](https://arxiv.org/abs/2511.20849),
  arXiv v2, 2026-08-08. arXiv metadata는 TMLR accepted로 표시한다.
- 내려받은 arXiv source tar SHA-256:
  `f376b7fd9afd26aba7ec4857eb40fd730f802832f1c4bc8778faee37bb29fb6c`
- 논문은 10K--50K에서 BPE보다 14--18% 적은 token, 64K에서 13.0% 적은 token을 보고한다.
  GPT-2 124M/355M/1.3B의 추론 latency 감소는 각각 13.7/12.7/13.7%이며, 각 model scale은
  다섯 seed로 학습했다.
- 논문 source에는 실행 가능한 tokenizer source나 repository URL이 포함되지 않았다. 본문은
  complete source 공개를 주장하지만, 이번 감사에서 논문과 직접 결속된 code artifact는
  찾지 못했다.

### 동명 PyPI package

- package: [`length-tokenizer-rs==0.1.10`](https://pypi.org/project/length-tokenizer-rs/),
  2025-12-31 release.
- sdist SHA-256:
  `33ce705974113845f6c49398fc4eb0ce39fadebad4feb45c1ba8493785855946`
- package metadata는 author, homepage, repository URL을 제공하지 않는다. README가 자신을
  `Length-MAX Tokenizer`라고 부르지만, 논문이 이 package를 링크하지 않으므로 **논문 저자의
  공식 reference implementation이라고 단정하지 않는다.**

### Thunder-Tok

- 논문: [Less Is More: Reducing Token Counts Without Compromising Performance](https://arxiv.org/abs/2506.15138),
  arXiv v2, 2026-07-09.
- Korean 560M 결과에서 BPE 대비 fertility는 1.50→1.37, 1,000개 생성의 token 수는
  22,451→20,572, wall time은 222.00→204.15초다. token-step 약 8.4%, wall time 약 8.0%
  감소이며 ms/token은 9.89→9.92로 거의 같다.
- 560M 실험은 다섯 run이지만 2.5B는 단일 run이다. 또한 fixed raw-byte exposure가 아니라
  fixed token budget에 맞추어 같은 corpus를 반복하므로 JamoFlow의 학습 계약과 직접 비교할
  수 없다.

## 2. Length-MAX 논문에서 실제로 정의된 것

논문의 vocabulary construction loop는 다음과 같다.

1. single UTF-8 character와 special token으로 초기화한다.
2. 현재 tokenized shard에서 길이 2부터 `L_max`까지 모든 n-gram을 센다.
3. 후보를 `frequency × token length`로 순위화한다.
4. 최고 후보를 vocabulary에 추가하고 corpus의 해당 token sequence를 in-place 치환한다.
5. target vocabulary에 도달할 때까지 반복한다.

학습 corpus는 이 in-place merge 결과가 곧 segmentation이다. 그러나 **새 corpus에 frozen
vocabulary를 적용할 때는 길이 내림차순의 left-most-longest prefix trie/DFA**를 쓴다고
명시한다. 따라서 논문 자체의 재현 역할을 minimum-token DP라고 부르면 틀리다.

논문 안에도 재현 시 고정해야 할 모호성과 불일치가 있다.

- Appendix 한 곳은 scoreboard `M=50,000`, `L_max=64`라고 하지만 다른 reproducibility
  단락은 `L_max=16`이라고 한다.
- graph objective를 `min`으로 적고 split이 objective를 감소시킨다고 설명하지만, 제시된
  delta 식은 prefix 길이가 증가할 때 양수가 된다. 부등호/목적함수 부호 중 하나가 뒤집힌
  것으로 보인다. 이 형식적 보장을 JamoFlow 방법의 근거로 재사용하지 않는다.
- single-character alphabet은 영어에서는 작지만 완성형 한글을 포함한 Korean corpus에서는
  2,048 vocabulary budget의 대부분 또는 전부를 소모할 수 있다.

이 때문에 JamoFlow가 구현할 수 있는 것은 exact paper reproduction과 byte-compatible
adaptation을 명시적으로 나눈 것이다.

## 3. PyPI 구현과 논문의 의미 차이

`length-tokenizer-rs` source를 직접 감사한 결과는 다음과 같다.

| 항목 | 논문 | PyPI 0.1.10 |
|---|---|---|
| 입력 단위 | single UTF-8 character | 관측 Unicode char + `Ġ` word-end |
| 공백 | raw corpus, no pre-tokenization | `split_whitespace()`, 공백 종류·개수 collapse |
| merge score | `f(t) × |t|` | 현재 n-gram 기준 `(n-1) × frequency` 누적 |
| frozen inference | left-most-longest trie/DFA | global minimum-token DP, longest tie-break |
| unknown | single-character fallback | `unk` fallback 가능 |
| 제약 | 논문 핵심에는 일반적인 raw cross-word | punctuation/word mix, max words/chars, incomplete cross-word 등 다수 옵션 |

둘 사이의 일부 차이는 구현 최적화일 수 있지만, 공백 보존과 segmentation objective는 model이
보는 sequence를 바꾸는 실험적 차이다. 특히 JamoFlow의 raw-byte BPB 및 byte-exact roundtrip
계약에서는 package를 그대로 baseline으로 사용할 수 없다. 2K Korean vocabulary에서는 관측
Unicode alphabet도 크기 제약을 깨므로 더더욱 그렇다.

## 4. Thunder-Tok에서 수용할 것과 수용하지 않을 것

수용할 부분은 세 가지다.

1. compression만 최대화하면 quality가 나빠질 수 있으므로 candidate pruning을 quality prior와
   함께 설계해야 한다.
2. 한국어는 영어보다 cross-word token 기회가 작고, 조사·어미가 어절 안에 붙는 구조 때문에
   eojeol 내부의 완결된 조각이 중요하다.
3. 실제 speedup은 token당 kernel 개선이 아니라 생성 token 수 감소에서 왔으므로, 동일 graph
   actual batch-1 wall time을 반드시 측정해야 한다.

그대로 수용하지 않을 부분도 분명하다.

- byte-decodability, incomplete cross-word 금지, word-boundary truncation, likelihood/entropy
  pruning은 이미 Thunder-Tok의 주된 공간이다. 같은 필터를 한국어에 적용한 것만으로는 새
  방법이 아니다.
- 128K/560M의 약 8% wall-time 결과를 2K/19.6M에 그대로 외삽할 수 없다.
- fixed token-budget 결과는 fixed raw-byte exposure의 학습 효율을 증명하지 않는다.

따라서 Korean constraint는 독립 novelty가 아니라 **generic long-token objective가 만드는
잘못된 압축을 줄이는 품질 제약**으로 먼저 취급한다. Generic 대비 추가적인 matched-quality
속도 이득이 실제로 남을 때만 주 방법의 일부로 승격한다.

## 5. 수정된 same-2K 역할 분해

한 번에 `Korean Length-MAX` 하나만 BPE와 비교하면 vocabulary construction과 segmentation
효과가 섞인다. 다음 역할을 순서대로 분리한다.

1. `BPE-2K-greedy`: 봉인된 exact ByteLevel BPE 2K 기준선.
2. `Byte-Unigram-2K`: 같은 256-byte fallback과 no-whitespace-pretokenization을 쓰는 generic
   likelihood/Viterbi control.
3. `Byte-LengthGain-2K-LML`: 256 mandatory bytes에서 시작해 length-weighted current-token
   n-gram merge로 vocabulary를 만들고 논문 정의의 left-most-longest로 적용한다.
4. `Byte-LengthGain-2K-DP`: 3과 **같은 vocabulary**를 minimum-token DP로 적용한다. 이는
   논문 reproduction이 아니라 segmentation-only ablation이다.
5. `Korean-Complete-2K`: 같은 byte budget에서 multi-byte 후보가 UTF-8/Hangul/eojeol
   completeness 조건을 만족하도록 제한한다. vocabulary construction과 application algorithm은
   generic 대응 role과 하나씩 맞춘다.

여기서 `Byte-LengthGain`이라는 이름은 paper-faithful claim을 피하기 위한 의도적인 명칭이다.
Mandatory byte alphabet, raw whitespace 보존, 한국어 2K budget은 논문의 character setup과
다르기 때문이다.

### 먼저 통과해야 하는 token-only opportunity gate

- 모든 byte string의 decode roundtrip과 256-byte fallback
- train/calibration document 순서와 raw-byte hash 동일
- exact vocabulary size 2,048 및 special-token accounting 동일
- calibration token 수가 BPE보다 최소 10% 낮음
- 64개 고정 Korean prompt의 predicted continuation step 수가 최소 10% 낮음
- tokenizer encode throughput과 peak memory 공개
- vocabulary utilization, token byte-length, UTF-8 완결성, Hangul syllable/eojeol 경계 crossing 공개

`Korean-Complete`는 generic보다 token 수가 약간 많아도 즉시 탈락시키지 않는다. 같은 2K
one-seed model에서 generic의 BPB가 무너지고 Korean constraint가 품질을 회복하면서 BPE 대비
10% 이상의 step headroom을 유지할 수 있기 때문이다. 다만 token-only 단계에서 BPE 대비
10% headroom 자체가 없으면 full training으로 보내지 않는다.

## 6. 연구 방향에 미치는 결론

현재 근거는 원래 계획을 폐기하지는 않지만, 다음처럼 좁힌다.

- `Length-MAX/DP control`이라는 단일 역할을 사용하지 않는다.
- vocabulary construction과 frozen segmentation을 별도 ablation으로 둔다.
- 2K에 맞는 byte alphabet adaptation은 재현 변형으로 명시한다.
- generic likelihood control(Byte-Unigram)을 추가해 length gain이 단순한 Unigram 최적화보다
  나은지 확인한다.
- Korean 제약은 Thunder-Tok과 다른 이름만 가진 필터가 아니라, 동일 generic vocabulary
  objective에서 압축–품질 frontier를 개선하는지로 평가한다.
- token count만으로 성공을 선언하지 않는다. 최종 기준은 여전히 fastest quality-qualified
  BPE 대비 raw-byte quality noninferiority와 trained-model batch-1 E2E 최소 10%다.

이 보정은 불필요한 방향 전환이 아니다. 2K quality frontier 결과와 최신 primary-source
구현 감사가 직접 요구한 confound 제거다. 반대로 generic 역할이 충분한 headroom을 보이고
Korean 제약이 quality를 회복하지 못한다면, 한국어 특화 방법을 억지로 유지하지 않고 generic
tokenizer systems 연구 또는 negative result로 결론을 수정한다.
