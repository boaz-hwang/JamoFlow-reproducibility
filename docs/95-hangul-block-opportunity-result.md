# Hangul block opportunity 결과

> 작성일: 2026-08-13
>
> 상태: **calibration-only data oracle 완료; learned-draft preflight 허가**
>
> authoritative aggregate:
> `results/hangul-block-opportunity-v1/summary.json`

## 1. 결론

한글 scalar block은 순차 target invocation을 줄일 이론적 여유가 충분하다. 그러나 한글
조합규칙만으로 미래 byte가 결정되지는 않는다. 따라서 다음 단계는 rule decoder가 아니라
**문맥을 조건으로 한 cheap dependence-aware Hangul draft**여야 한다.

사전 gate 두 개는 모두 통과했다.

| gate | 기준 | 관측 | 판정 |
|---|---:|---:|---|
| perfect Hangul-only target-call reduction | >=20% | **57.593%** | pass |
| 전체 scalar savings 중 Hangul 비율 | >=90% | **98.681%** | pass |

이 pass는 learned-draft acceptance preflight만 허가한다. Speed, acceptance, quality 또는
novelty 증거가 아니다.

## 2. Calibration stream 구성

Selection-v2와 같은 8,000,000-byte calibration stream을 exact SHA-256으로 재구성했다.
마지막 1 byte가 유효한 3-byte scalar의 잘린 prefix라 7,999,999 complete-scalar bytes만
oracle 분모에 넣었다.

| 항목 | 값 |
|---|---:|
| complete scalar bytes | 7,999,999 |
| complete scalars | 3,330,976 |
| precomposed Hangul syllables | 2,303,716 |
| Hangul scalar rate | 69.160% |
| Hangul byte rate | **86.389%** |
| ASCII scalars | 994,596 |
| Jamo-block scalars | 802 |
| other scalars | 31,862 |

UTF-8 길이는 1-byte 994,596개, 2-byte 3,848개, 3-byte 2,332,421개, 4-byte
111개였다. 즉 이 Korean workload에서 byte AR의 긴 sequence 대부분은 3-byte Hangul
scalar가 만든다.

## 3. Perfect call oracle

| oracle | target calls | byte AR 대비 감소 |
|---|---:|---:|
| byte AR | 7,999,999 | 0% |
| one call per complete scalar | 3,330,976 | 58.363% |
| Hangul scalar만 block, 나머지 bytewise | 3,392,567 | **57.593%** |
| perfect fixed 2-byte block | 4,000,000 | 50.000% |
| perfect fixed 3-byte block | 2,666,667 | 66.667% |
| perfect fixed 4-byte block | 2,000,000 | 75.000% |
| perfect fixed 8-byte block | 1,000,000 | 87.500% |

Hangul-only grouping은 전체 scalar grouping이 절약하는 4,669,023 calls 중 4,607,432를
설명한다. 이는 Korean-specific opportunity가 크다는 뜻이다. 그러나 fixed block oracle이
더 큰 상한을 갖는다는 사실도 중요하다. Scalar alignment의 가치는 호출 수 자체가 아니라
learned acceptance와 invalid/rejected proposal 비용에서 generic fixed-byte draft를 이길
때만 성립한다.

## 4. 규칙만으로 continuation을 결정할 수 없는 이유

완성형 한글 11,172자는 onset 19, vowel 21, coda 28의 전 조합과 일대일 대응한다. Flat
syllable head의 11,172 logits를 독립 component head 68 logits로 줄이면 output-logit 수는
99.391% 감소한다. 그러나 이는 같은 분포를 표현한다는 뜻이 아니다.

| empirical diagnostic | 값 |
|---|---:|
| joint syllable entropy | 8.098 bits |
| onset entropy | 3.507 bits |
| vowel entropy | 3.518 bits |
| coda entropy | 2.337 bits |
| component total correlation | **1.264 bits** |
| context-free independent component mode exact rate | **0.563%** |

세 component의 주변분포는 독립이 아니며, 문맥을 무시한 독립 최빈 조합은 실제 syllable의
0.563%밖에 맞히지 못한다.

UTF-8 continuation pair에서도 같은 문제가 더 직접적으로 보인다.

| first Hangul lead byte를 조건으로 한 diagnostic | 값 |
|---|---:|
| joint `(b2,b3)` entropy | 6.305 bits |
| `b2` entropy | 4.244 bits |
| `b3` entropy | 4.469 bits |
| `I(b2;b3 | b1)` | **2.409 bits** |
| joint pair mode exact rate | 8.375% |
| independent byte modes exact-pair rate | 6.952% |

첫 byte `EA..ED`는 뒤 두 byte를 결정하지 않는다. Validity rule은 후보공간을 제한할 뿐
semantic/lexical choice를 대신할 수 없다. 두 continuation byte도 강하게 의존하므로
fully-factorized byte head가 충분하다고 미리 가정해서는 안 된다.

이 수치는 context-free corpus diagnostic이다. Target hidden state를 조건으로 하면
acceptance는 크게 달라질 수 있으므로 다음 neural experiment가 필요하다.

## 5. 다음 설계에 주는 제약

다음 preflight는 한 seed의 frozen W72 target에서 같은 training contexts를 사용해 최소
세 draft를 비교한다.

1. **generic FF byte-MTP**: future offsets의 독립 byte heads
2. **scalar-aligned generic joint control**: UTF-8 scalar 끝에 맞춘 block이지만 Hangul
   component를 쓰지 않는 dependence-aware head
3. **Hangul factorized/joint draft**: onset/vowel/coda 구조와 유효 composition을 사용하는
   head

Target의 첫 next-byte argmax는 그대로 authority로 쓰고, draft는 그 뒤의 future bytes를
제안한다. Acceptance는 target AR logits로 exact 검증한다. 최소 보고 단위는 다음과 같다.

- proposal exact-prefix length와 complete-block acceptance
- Hangul/ASCII/other 및 scalar-boundary start별 결과
- accepted bytes per verification
- head parameter 수와 isolated forward time
- cached block target verification time
- draft+verification을 합친 measured calibration E2E

Generic FF보다 joint control이 낫지만 Hangul head가 joint control을 넘지 못하면
`dependence-aware MTP`의 효과일 뿐 한국어 구조의 기여가 아니다. Hangul head가 acceptance를
높여도 추가 head latency가 이득을 지우면 method는 실패다.

## 6. Claim 경계

- 57.593%는 perfect future oracle의 **call-count opportunity**이지 speedup이 아니다.
- `68 vs 11,172 logits`는 output factorization 크기이며 전체 model parameter/FLOP 감소가
  아니다.
- context-free entropy는 neural acceptance가 아니다.
- 이 결과는 development calibration의 사후 exploratory 분석이다.
- final test, checkpoint, model loss, latency를 읽지 않았다.

따라서 현재 허가되는 결론은 “Korean calibration에는 neural draft를 시험할 충분한
opportunity가 있고, 독립 head보다 dependence-aware 구조를 비교해야 한다”까지다.

결과 확인 전 고정한 후속 비교는
[`96-hangul-draft-acceptance-preflight.md`](96-hangul-draft-acceptance-preflight.md)에
있다. Generic independent/joint UTF-8와 parameter-matched Hangul parallel/conditional head를
같은 frozen W72 hidden에서 비교하며, 한국어-specific prototype은 joint generic control을
통계적으로 넘어야만 허가된다.
