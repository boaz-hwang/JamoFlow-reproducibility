# Hangul block opportunity preflight v1

> 작성일: 2026-08-13
>
> 상태: **실행 전 고정한 post-result exploratory protocol**

## 목적

Component profile은 매-byte local path가 실제 병목임을 보였다. 다음 neural draft를
학습하기 전에, calibration stream의 UTF-8/한글 구성만으로 multi-byte block이 제거할 수
있는 target invocation의 상한과 한글 구조가 그 상한에서 차지하는 비율을 계산한다.

이 분석은 acceptance나 speed를 증명하지 않는다. Perfect oracle은 미래 byte를 이미
안다고 가정하므로 실제 draft 비용, rejection, verification, cache rollback을 모두
무시한다. 역할은 비싼 prototype을 시작할 최소한의 opportunity가 있는지 반증하는 것이다.

## 입력 경계

- 기존 HPLT 3.0 Korean calibration stream 정확히 8,000,000 bytes
- selection-v2가 봉인한 stream SHA-256과 exact match
- model checkpoint, model loss, historical/final test, latency artifact를 읽지 않음
- aggregate count, entropy, theoretical call count만 tracked result로 기록
- raw text, codepoint sample, document identifier는 출력하지 않음

Calibration은 이미 연구에 사용된 development split이므로 이 분석을 confirmatory 또는
held-out이라고 부르지 않는다.

## UTF-8 call oracle

완전한 scalar에 속한 byte 수를 `N`, scalar 수를 `C`, precomposed Hangul syllable 수를
`H`라 한다. 현재 target은 byte마다 한 번의 sequential consume을 수행한다고 단순화한다.

- byte AR: `N` calls
- perfect one-scalar block: `C` calls
- perfect Hangul-only adaptive block: `N - 2H` calls
- perfect fixed-k byte block: `ceil(N/k)` calls

Hangul-only oracle은 각 3-byte Hangul scalar를 한 번에 처리하고 나머지는 bytewise로
남긴다. 이는 구현 속도 상한이 아니라 제거 가능한 sequential call count 상한이다.
Generic fixed block은 scalar를 가로지를 수 있으므로 구조-aware 방식보다 call 상한이
더 좋을 수 있다. 한국어 구조의 가치는 call count 자체가 아니라 learned proposal의
acceptance/cost에서 generic control을 넘어야 한다.

## 분포 diagnostic

Precomposed Hangul `U+AC00..U+D7A3`를 표준식으로 onset 19, vowel 21, coda 28 index로
분해한다. 다음을 aggregate로 계산한다.

- component 및 joint empirical entropy
- `H(L)+H(V)+H(T)-H(L,V,T)` total correlation
- 첫 UTF-8 byte가 주어졌을 때 continuation pair entropy
- 두 continuation byte의 conditional mutual information
- context-free joint-pair mode와 independent-byte mode의 exact pair accuracy

이 값은 target hidden state를 조건으로 하지 않으므로 neural draft acceptance 예측치가
아니다. 다만 independent future-byte head가 구조적으로 충분하다고 미리 가정할 수 있는지,
Hangul factorization이 output space를 얼마나 줄이는지 판단하는 진단이다.

## 사전 decision rule

다음을 모두 만족할 때에만 learned-draft acceptance preflight를 연다.

1. perfect Hangul-only target-call reduction `>=20%`
2. 전체 scalar grouping savings 중 Hangul이 설명하는 비율 `>=90%`

통과해도 orthography-aware method가 유효하다는 뜻은 아니다. 다음 단계에서 같은 hidden,
training bytes, parameter/latency envelope의 generic byte-MTP를 반드시 control로 둔다.
실패하면 Hangul-specific draft branch를 중단한다.

## 산출물

- plan: `data/manifests/hangul-block-opportunity-v1.json`
- implementation: `scripts/analyze_hangul_block_opportunity.py`
- aggregate: `results/hangul-block-opportunity-v1/summary.json`

Plan과 구현/tests를 먼저 commit하고 clean HEAD에서 한 번 실행한다. 결과는 exploratory이며
새 final-quality/timing protocol을 직접 authorize하지 않는다.
