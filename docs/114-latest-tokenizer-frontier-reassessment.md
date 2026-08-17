# 최신 tokenizer frontier 재검토와 연구 방향 보정

> 작성일: 2026-08-14
>
> 상태: scalar runtime 반증 뒤의 선행연구·연구 방향 재검토

## 1. 결론

Scalar/Hangul-hybrid branch를 중단하고 token-level encoding으로 이동한 결정은 맞다. 다만
`한국어-aware tokenizer를 만들면 새롭다`는 수준으로는 논문이 되지 않는다. 2026년 8월
현재 다음 세 결과가 이미 존재한다.

1. [Thunder-Tok](https://arxiv.org/abs/2506.15138)은 한국어 560M decoder에서 BPE 대비
   fertility를 1.50에서 1.37로 낮추고, 같은 1,000개 target 문장 생성 시간을
   222.00초에서 204.15초로 줄였다. 실제 wall-clock 개선은 약 **8.0%**다.
2. [Length-MAX](https://arxiv.org/abs/2511.20849)는 영어에서 같은 vocabulary size의
   BPE보다 token 수를 14--18% 줄이고, 124M/355M/1.3B GPT-2에서 각각
   13.7%/12.7%/13.7% inference latency 감소를 보고했다.
3. [Lifecycle-Optimal Tokenization](https://arxiv.org/abs/2608.11361)은 vocabulary와
   unembedding 비용의 최적점이 batch·hardware·model scale에 따라 바뀐다고 직접
   측정했다. 100M English training에서는 16K, A100 batch 1 inference에서는 32K를
   최적으로 보고한다. 이는 2026-08-11 공개된 최신 preprint이므로 결과의 독립 재현과
   peer-review 상태는 별도로 구분해야 하지만, 단순 vocabulary-size sweep 자체의
   신규성은 이미 약해졌음을 뜻한다.

따라서 JamoFlow의 남은 가치 있는 질문은 다음과 같다.

> 한국어 단일 사용자·batch 1 환경에서, 강하게 최적화한 BPE vocabulary/geometry
> frontier보다 더 짧고 실제로 더 빠르면서 raw-byte BPB와 생성 품질을 유지하는
> language-aware tokenizer 또는 output factorization을 만들 수 있는가?

이 질문에 대한 답이 `아니오`이면 tokenization branch도 중단한다. Byte W72보다 빠른지만
보거나 token count만 줄었다고 성공으로 부르지 않는다.

## 2. 선행연구가 이미 닫은 주장

### 2.1 한국어 morphology-aware tokenization

[한국어 tokenization 비교 연구](https://aclanthology.org/2020.aacl-main.17/)는 형태소 분석
후 BPE가 여러 한국어 NLU/MT task에서 강하다는 것을 보였다.
[Morpheme Matters](https://aclanthology.org/2026.eacl-short.22/)는 32K vocabulary와 41M
BERT encoder에서 형태소의 surface/base form과 어절 내·외 prefix를 이용해 일반적으로
더 좋은 downstream 성능과 MoA 대비 평균 19% 적은 token instance를 보고했다. 그러나
decoder-only 생성 latency는 측정하지 않았다.

[MorphBPE](https://aclanthology.org/2026.findings-acl.2068/)는 형태소 경계를 넘는 merge를
금지해 300M/1B decoder의 품질을 개선했다. Korean은 평가하지 않았고, token length는
BPE와 거의 같거나 약간 길었다. 따라서 morphology가 quality prior가 될 수 있다는 근거이지,
그 자체가 speed mechanism이라는 근거는 아니다.

결론적으로 `형태소 경계를 tokenizer에 넣었다`, `한국어 token 수가 줄었다`, `downstream이
좋아졌다`는 각각 이미 독립적으로 선행된다. JamoFlow는 이 중 어느 하나도 단독 novelty로
주장하지 않는다.

### 2.2 token count와 실제 latency

Thunder-Tok의 Table 4는 중요한 강한 기준선이다.

| Korean tokenizer | fertility | generated tokens | time | ms/token |
|---|---:|---:|---:|---:|
| BPE | 1.50 | 22,451 | 222.00 s | 9.89 |
| BPE-Mecab | 1.97 | 29,402 | 292.92 s | 9.96 |
| SuperBPE | 1.35 | 20,295 | 201.83 s | 9.94 |
| Thunder-Tok | 1.37 | 20,572 | 204.15 s | 9.92 |

이 표는 같은 graph에서는 token step 감소가 거의 그대로 wall-clock 감소로 이어짐을 보여준다.
동시에 가장 짧은 SuperBPE가 quality를 희생했고, Thunder-Tok은 약간 더 긴 대신 품질을
보존했다. 즉 compression 최대화와 matched quality가 별개의 constraint다.

Length-MAX 논문은 `freq(token) × length(token)`에 가까운 length-weighted vocabulary
objective와 **frozen-vocabulary left-most-longest segmentation**으로 영어의 실제 latency를
더 크게 줄였다. 별도로 배포된 동명 PyPI 패키지는 minimum-token DP를 제공하지만, 논문과
패키지의 정확한 구현 동일성은 확인되지 않았고 공백 정규화 등 의미 차이도 있다. 이 둘은
재현 실험에서 같은 방법으로 취급하지 않는다.
따라서 `BPE보다 긴 문자열을 vocabulary에 넣는다`도 신규 주장이 아니다. Korean-specific
후보는 Length-MAX 같은 generic long-token control과 Thunder-Tok의 공개 수치 둘 다 넘어야
설득력이 생긴다.

### 2.3 vocabulary와 model geometry

[Compute Optimal Tokenization](https://arxiv.org/abs/2605.01188)은 988개 50M--7B BLT
실험으로 optimal compression rate가 compute budget과 언어에 따라 달라진다고 보고했다.
Lifecycle-Optimal Tokenization은 output head가 batch 1에서 memory-bound이고 vocabulary가
커질수록 token 수는 줄지만 token당 head 비용은 커진다는 점을 실측했다.

JamoFlow의 scalar runtime에서도 같은 현상이 더 직접적으로 나타났다. BPE16K는 BPE32K보다
continuation token이 조금 많았지만, 우연히 선택된 9-layer graph가 13-layer graph보다 빨라
E2E 중앙값이 67.6ms 대 92.9ms였다. 이 결과로는 16K tokenizer가 본질적으로 우월하다고
말할 수 없다. Vocabulary, depth, width, FFN, output head가 함께 바뀌었기 때문이다.

따라서 다음 candidate를 하나의 BPE16K와만 비교하면 약한 comparator를 고를 위험이 있다.

## 3. 수정된 연구 순서

### 단계 A — Korean BPE systems frontier

새 방법을 보기 전에 같은 19.6M total-parameter budget에서 다음을 calibration-development
data로 측정한다.

- vocabulary: 2K, 4K, 8K, 16K, 32K, 64K
- depth: 8, 12, 16
- 각 `(vocabulary, depth)`에서 hidden/FFN/head 수를 결과를 보지 않는 deterministic grid로
  parameter-match
- 동일 raw prompt/continuation, batch 1, Apple MPS, tied embedding, actual output head,
  parallel prefill와 cached incremental decode
- tokenization은 timer 밖이되 tokenizer encode throughput은 별도 diagnostic

이 단계의 산출물은 `새 기법 성공`이 아니라 가장 빠른 BPE systems comparator와
vocabulary–geometry Pareto surface다. Random weights이므로 quality comparator를 확정하지
않는다.

### 단계 B — candidate opportunity

Frontier가 정해진 뒤 동일 vocabulary와 동일 model graph에서 최소 세 tokenizer를 비교한다.

1. exact byte BPE
2. generic length-maximizing/likelihood tokenizer
3. Korean-aware constrained variant

Korean variant가 진입하려면 다음을 모두 만족해야 한다.

- raw bytes에 대해 완전 가역적이며 byte fallback을 보존
- calibration token count가 같은-V BPE보다 최소 10% 감소
- generic long-token control보다 추가 이득 또는 명확한 robustness/quality prior를 제공
- tokenizer runtime과 model output head를 포함한 random-weight E2E에서 가장 빠른 BPE
  systems role보다 최소 10% 개선 가능성을 보임

형태소 분석기를 쓰면 analyzer version·license·오분석 fallback을 고정한다. 형태소 경계는
quality prior이지 speed 결과로 간주하지 않는다.

### 단계 C — one-seed matched-quality training

단계 B를 통과한 후보만 같은 clean train bytes와 raw-byte BPB metric으로 학습한다. 비교는
최소한 다음을 포함한다.

- fastest quality-qualified BPE frontier model
- same-vocabulary BPE
- generic long-token control
- Korean-aware candidate

Model-token budget이 아니라 raw-byte budget을 고정한다. Tokenizer가 corpus를 더 많이 반복해
보는 혼입을 막고, BPB는 tokenizer vocabulary에 독립적으로 계산한다.

### 단계 D — publication confirmation

One seed에서 quality noninferiority와 actual E2E 10% gate를 모두 통과한 경우에만 3--5 seeds,
새 sealed final split, 실제 free-running 생성, 독립 timing session, memory 및 tokenizer encode
cost로 확장한다.

## 4. 후보 기법에 대한 현재 판단

지금 당장 `Jamo-aware BPE`를 주 방법으로 고정하지 않는다. Jamo BPE, morphology+BPE,
Morpheme Matters, Thunder-Tok이 이미 가까운 공간을 차지한다. 반대로 다음 결합은 아직
실험 가치가 있다.

> fixed-size long-token vocabulary의 step-saving objective에 Korean eojeol/morpheme validity를
> quality constraint로 넣고, vocabulary/head/model geometry를 실제 batch-1 latency로 공동
> 선택하는 방법

그러나 이 표현도 Length-MAX, Thunder-Tok, lifecycle-optimal tokenization의 단순 합성에
그칠 수 있다. 단계 A/B 결과가 generic control을 유의미하게 넘지 못하면 새 방법이라고
밀지 않는다. Hierarchical/factorized large-vocabulary head는 큰 vocabulary의 token-step 이득을
작은 output cost로 얻을 가능성이 있지만 adaptive/hierarchical softmax와 compositional output
선행이 넓다. BPE frontier와 fixed-V tokenizer만으로 10%를 만들 수 없을 때 별도 novelty
감사 후에만 확장한다.

## 5. 논문 claim의 최소 기준

최종 positive paper에 필요한 주장은 다음 하나다.

> At matched raw-byte quality and total parameter budget, the proposed Korean-aware method
> reduces batch-1 end-to-end generation latency by at least 10% relative to the fastest
> quality-qualified BPE vocabulary/geometry baseline, with the improvement replicated across
> model seeds, documents, and fresh timing sessions.

이를 못 만족하면 negative engineering evidence로 보존하되 `더 효율적인 한국어 LLM`이라고
발표하지 않는다. Thunder-Tok의 약 8% Korean result 때문에 10% gate는 임의로 높은 숫자가
아니라, 이미 공개된 실제 시스템 개선을 분명히 넘어서는 최소 효과 크기다.
