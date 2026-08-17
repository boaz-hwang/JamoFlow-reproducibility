# Stage 2 Public Corpus Results and Research Decision

> 실행일: 2026-08-10  
> 상태: **Phase 0 공개 corpus 검증 완료**  
> 사전 등록: [06-stage2-public-corpus-protocol.md](./06-stage2-public-corpus-protocol.md)  
> 기계 판독 요약: [results/stage2-public/summary.json](../results/stage2-public/summary.json)  
> 사람이 읽는 집계: [results/stage2-public/summary.md](../results/stage2-public/summary.md)

## 1. 결론부터

Stage 2 결과는 최초의 “한글 음절 규칙으로 entropy patcher를 대체한다”는 방향을 지지하지 않는다. 오히려 다음처럼 연구 문제를 수정해야 한다.

> **한글 규칙 기반 patching**이 아니라, **multi-byte UTF-8 언어에서 learned byte patcher의 평가 위치를 causal codepoint boundary로 제한했을 때 detector의 총비용과 LM 품질 사이에 어떤 trade-off가 생기는가**를 연구한다.

핵심 근거는 네 가지다.

1. codepoint candidate restriction은 score 평가 위치를 한국어에서 59.1%, 중국어에서 64.7% 줄였지만 영어에서는 0.9%만 줄였다. 한국어 고유 신호가 아니라 multi-byte UTF-8 효과라는 설명이 더 강하다.
2. 한글 음절·구두점으로 후보를 더 좁힌 policy는 generic codepoint 후보보다 모든 proxy quality 지표가 낮았다. 중국어의 대칭 CJK policy에서도 같은 방향이었다.
3. pure Hangul-syllable rule은 matched-rate learned proxy 및 단순 rule control보다 약했다. rule-only architecture의 중심 가설로 남길 근거가 없다.
4. oracle capture와 top-decile recall의 절대값은 n-gram order와 smoothing에 크게 흔들렸다. Phase 0 entropy proxy로 neural architecture 품질을 결론내리면 안 된다.

따라서 Phase 1은 **generic UTF-8-aware candidate restriction의 실제 neural 품질과 detector 포함 총비용**만 검증한다. Hangul/Jamo/FST/multi-jamo generation을 한 번에 묶은 큰 architecture는 보류한다.

## 2. 데이터와 재현성

세 corpus는 Leipzig Corpora Collection의 100,000-sentence Wikipedia 표본이다. 제공처는 normed-size corpus를 무작위 선택 문장으로 설명하며, 다운로드 text corpus를 CC BY로 제공한다고 명시한다. 원천 Wikipedia text의 CC BY-SA/GFDL 및 page별 예외도 별도로 기록했다.

- [Leipzig corpus documentation](https://wortschatz.uni-leipzig.de/en/documentation)
- [Leipzig terms of usage](https://www.wortschatz.uni-leipzig.de/en/usage)
- [Wikimedia dump licensing](https://dumps.wikimedia.org/legal.html)
- [Goldhahn, Eckart, and Quasthoff, LREC 2012](https://aclanthology.org/L12-1154/)

Archive URL, compressed byte length, archive SHA-256, 파생 JSONL SHA-256은 [데이터 manifest](../data/manifests/leipzig-wikipedia-100k.json)에 고정했다. corpus text와 record identifier는 Git에 넣지 않았다.

| Language | Snapshot | Records | Raw text bytes | Codepoints | Bytes/codepoint |
|---|---:|---:|---:|---:|---:|
| Korean | 2021 | 100,000 | 14,272,643 | 5,947,059 | 2.400 |
| Chinese | 2018 | 100,000 | 10,576,852 | 3,824,271 | 2.766 |
| English | 2016 | 100,000 | 12,743,071 | 12,723,553 | 1.002 |

snapshot 연도가 다른 것은 명시적 위협 요인이다. 언어 간 절대 entropy를 직접 비교하지 않고, 각 언어에서 candidate restriction 전후의 normalized 지표를 비교했다.

## 3. 분석 설정

Primary configuration은 다음과 같다.

- deterministic exact-text deduplication
- content-hash train/calibration/test split 80/10/10
- causal byte 4-gram, additive smoothing `alpha=0.1`
- target fixed stride 6의 patch rate에 calibration threshold를 맞춤
- test split에서 threshold 고정
- 500회 record-level percentile bootstrap, seed 1729
- score evaluation 수는 algorithmic opportunity count
- Python runtime은 production latency로 해석하지 않음

Sensitivity는 다음을 추가했다.

- n-gram order 2와 4
- `alpha=0.01, 0.1, 1.0`
- fixed stride target 4, 6, 8
- whitespace/cap 12, 24, 48

Primary 전체 보고서는 [Korean](../results/stage2-public/ko-order4/report.md), [Chinese](../results/stage2-public/zh-order4/report.md), [English](../results/stage2-public/en-order4/report.md)에 있다.

## 4. Primary result

### 4.1 Generic codepoint candidate

아래는 fixed-byte-6 rule의 calibration rate에 맞춘 결과다.

| Language | Bytes/patch | Oracle capture | Top-budget overlap | Top-decile recall | Score eval/byte | 평가 위치 감소 |
|---|---:|---:|---:|---:|---:|---:|
| Korean | 5.910 | 0.692 | 0.300 | 0.418 | 0.409 | 59.1% |
| Chinese | 5.828 | 0.884 | 0.475 | 0.478 | 0.353 | 64.7% |
| English | 5.844 | 0.997 | 0.991 | 0.988 | 0.991 | 0.9% |

Bootstrap 95% interval은 다음처럼 sampling uncertainty가 작음을 보였다.

| Language | Top-decile recall 95% | Score eval/byte 95% |
|---|---:|---:|
| Korean | [0.413, 0.424] | [0.409, 0.410] |
| Chinese | [0.474, 0.482] | [0.352, 0.354] |
| English | [0.987, 0.989] | [0.990, 0.991] |

이 결과의 정확한 해석은 제한적이다.

- 영어에서는 거의 모든 byte 위치가 codepoint boundary이므로 restriction의 계산 이득이 없다.
- 한국어와 중국어에서는 3-byte 문자가 많아 candidate 평가 위치가 약 35–41%만 남는다.
- 중국어의 proxy capture가 한국어보다 높다. 따라서 현재 신호를 Hangul 고유 장점으로 부를 수 없다.
- 한국어의 raw-byte top-decile 위치 중 상당수가 UTF-8 continuation byte에 있다. “그 위치에 global boundary가 반드시 필요하다”는 것은 n-gram oracle의 가정이지 neural LM에서 검증된 사실이 아니다.

즉 **측정 가능한 detector-work opportunity**는 존재하지만 **품질 보존**은 아직 미판정이다.

### 4.2 더 좁은 Hangul/CJK + delimiter candidate

| Language | Candidate | Bytes/patch | Oracle capture | Top-budget overlap | Top-decile recall | Score eval/byte |
|---|---|---:|---:|---:|---:|---:|
| Korean | codepoint | 5.910 | 0.692 | 0.300 | 0.418 | 0.409 |
| Korean | Hangul+delimiter | 5.788 | 0.595 | 0.190 | 0.251 | 0.388 |
| Chinese | codepoint | 5.828 | 0.884 | 0.475 | 0.478 | 0.353 |
| Chinese | CJK+delimiter | 5.844 | 0.833 | 0.366 | 0.348 | 0.315 |

한국어에서는 score 평가를 추가로 2.1%p만 줄이면서 oracle capture가 9.7%p, top-decile recall이 16.7%p 낮아졌다. 중국어에서도 평가 위치 3.8%p 추가 절감에 비해 proxy 손실이 컸다.

이 policy는 Phase 1 주력 비교군에서 제외한다. 다만 neural pilot에서 generic codepoint restriction이 성공한 뒤 feature ablation으로 한 번 확인할 수는 있다.

### 4.3 Pure script rule

| Language | Rule | Bytes/patch | Oracle capture | Top-budget overlap | Top-decile recall | Mean lag |
|---|---|---:|---:|---:|---:|---:|
| Korean | Hangul syllable | 3.380 | 0.443 | 0.127 | 0.162 | 3.910 |
| Chinese | CJK ideograph | 3.437 | 0.716 | 0.353 | 0.318 | 3.693 |

두 언어의 script rule은 같은 정의에서도 결과가 크게 다르다. 이는 writing-system structure만이 아니라 corpus 분포와 entropy proxy가 결과를 좌우한다는 뜻이다. 특히 Hangul rule은 Korean public corpus에서도 learned proxy를 대체할 수준의 boundary signal을 보이지 않았다.

**판정: pure Hangul-syllable patcher는 no-go다.**

### 4.4 SpaceByte-compatible structural diagnostic

| Language | Bytes/patch | UTF-8 내부 경계 | Hangul 내부 | CJK 내부 |
|---|---:|---:|---:|---:|
| Korean | 3.277 | 69.1% | 68.4% | 0.4% |
| Chinese | 3.013 | 98.3% | 0.0% | 87.0% |
| English | 5.974 | 0.5% | 0.0% | 0.0% |

ASCII letter/digit가 아닌 non-continuation byte를 spacelike로 보는 cadence는 multi-byte script에서 영어와 전혀 다른 patch rate와 phase를 만든다. 이것은 중요한 architecture diagnostic이지만 SpaceByte/BLT의 품질 오류를 직접 뜻하지 않는다. local byte encoder가 내부 경계를 처리할 수 있기 때문이다.

Phase 1에서는 “같은 이름의 rule”만 비교하지 않고 **실제 achieved patch rate를 맞춘 codepoint-aligned control**을 반드시 둔다.

## 5. Sensitivity 결과

### 5.1 Proxy specification sensitivity

fixed-rate codepoint candidate의 결과는 n-gram 설정에 크게 의존했다.

| Language | Setting | Oracle capture | Top-budget overlap | Top-decile recall |
|---|---|---:|---:|---:|
| Korean | order 2, alpha 0.1 | 0.607 | 0.110 | 0.133 |
| Korean | order 4, alpha 0.01 | 0.637 | 0.155 | 0.186 |
| Korean | order 4, alpha 0.1 | 0.692 | 0.300 | 0.418 |
| Korean | order 4, alpha 1.0 | 0.803 | 0.420 | 0.455 |
| Chinese | order 2, alpha 0.1 | 0.712 | 0.091 | 0.133 |
| Chinese | order 4, alpha 0.01 | 0.772 | 0.255 | 0.308 |
| Chinese | order 4, alpha 0.1 | 0.884 | 0.475 | 0.478 |
| Chinese | order 4, alpha 1.0 | 0.987 | 0.479 | 0.480 |

이는 sampling noise가 아니라 **oracle definition uncertainty**다. 같은 corpus와 candidate set에서 smoothing을 바꾸자 “보존률”이 크게 변한다. 따라서 Stage 2의 entropy quality 수치를 neural result의 사전 예측치로 사용하지 않는다.

### 5.2 Patch-rate sensitivity

| Language | Target stride | Achieved bytes/patch | Oracle capture | Top-budget overlap |
|---|---:|---:|---:|---:|
| Korean | 4 | 3.957 | 0.650 | 0.251 |
| Korean | 6 | 5.910 | 0.692 | 0.300 |
| Korean | 8 | 7.824 | 0.729 | 0.359 |
| Chinese | 4 | 3.913 | 0.820 | 0.470 |
| Chinese | 6 | 5.828 | 0.884 | 0.475 |
| Chinese | 8 | 7.661 | 0.920 | 0.476 |
| English | 4 | 3.960 | 0.997 | 0.994 |
| English | 6 | 5.844 | 0.997 | 0.991 |
| English | 8 | 7.814 | 0.996 | 0.989 |

rate가 희소해질수록 capture ratio가 올라가는 현상은 denominator인 top-`M` oracle entropy sum도 함께 달라지는 metric 특성의 영향을 받는다. 절대 수치보다 한국어 < 중국어 < 영어의 방향과 영어에서 비용 절감이 거의 없다는 사실이 안정적이다.

## 6. Stage 1과의 불일치가 의미하는 것

private Markdown 표본에서 codepoint candidate는 proxy oracle capture 95–97%를 보였다. 공개 Korean Wikipedia에서는 65–80% 범위가 n-gram 설정에 따라 나타났고 primary는 69.2%였다.

이 차이는 다음을 보여 준다.

1. 개인 Markdown vault는 code·영어·markup 비율이 높아 자연 한국어의 대리 corpus가 아니다.
2. candidate coverage는 corpus composition에 민감하다.
3. 하나의 convenience sample에서 나온 높은 수치를 연구 방향 확정 근거로 사용하면 안 된다.
4. Stage 1 결과를 “promising signal”로만 둔 판단이 맞았고, 공개 control이 없었다면 잘못된 결론을 내릴 위험이 컸다.

## 7. Gate 판정

| Gate | 판정 | 조치 |
|---|---|---|
| Data integrity | Pass | 3개 archive 및 파생 JSONL hash 고정 |
| Prefix causality | Pass | primary 언어별 240–400 policy-prefix checks 통과 |
| Rule-only Hangul | No-go | 중심 architecture에서 제거 |
| Hangul/CJK narrow candidate | No-go for primary | generic codepoint보다 작은 추가 절감에 큰 proxy 손실 |
| Generic codepoint cost opportunity | Pass | Korean 59.1%, Chinese 64.7% evaluation-position reduction |
| Proxy quality preservation | Inconclusive | oracle가 specification-sensitive; neural loss로 판정 |
| Korean-specific novelty | Not supported | Chinese 효과가 더 큼; generic UTF-8 framing으로 전환 |
| Phase 1 neural pilot | Go, scoped | full entropy vs codepoint candidate vs cheap aligned rule만 비교 |

## 8. 폐기·보류·유지할 아이디어

### 폐기

- “Hangul syllable boundary가 learned entropy patcher를 무손실로 대체한다”는 중심 가설
- output head 후보 자소 수 감소를 주요 latency source로 보는 주장
- 이번 proxy 수치만으로 BLT quality 또는 end-to-end speedup을 예측하는 주장

### 보류

- 자소 atomic representation
- 형태소 FST routing
- spacing-aware adaptive depth
- multi-jamo parallel generation

이들은 각각 독립적인 연구 질문이다. 지금 모두 결합하면 어느 요소가 효과를 냈는지 식별할 수 없고 engineering failure surface만 커진다.

### 유지

- full learned entropy detector의 비용을 end-to-end accounting에 넣어야 한다는 문제 제기
- causal UTF-8 codepoint candidate restriction
- Korean/Chinese/English control
- equal achieved patch-rate comparison
- rule-only, learned-only, hybrid의 분해된 ablation

## 9. Phase 1에 넘길 정확한 연구 질문

> At matched global-patch rates, does restricting a learned byte-level patcher to causal UTF-8 codepoint boundaries reduce detector and end-to-end inference cost without materially degrading bits-per-byte, and is the trade-off specific to multi-byte scripts rather than Korean?

Phase 1의 최소 비교군은 세 개다.

1. **Full-byte learned router**: 모든 byte position에서 learned boundary score 평가
2. **Codepoint-candidate learned router**: causal UTF-8 codepoint boundary에서만 같은 score head 평가
3. **Codepoint-aligned fixed rule**: learned detector가 없는 cheap control

필수 계측은 다음과 같다.

- validation/test BPB와 byte-normalized NLL
- achieved bytes/patch 및 patch-length distribution
- router FLOPs와 호출 횟수
- global/local model FLOPs
- tokens/bytes per second, first-token latency, steady-state latency
- peak memory와 KV/cache traffic proxy
- Korean, Chinese, English별 결과
- 최소 3 seeds와 paired confidence interval

**성공 기준은 detector 호출 감소 자체가 아니다.** 같은 quality tolerance 안에서 detector를 포함한 end-to-end latency 또는 총 연산이 줄어야 한다.

## 10. 남은 한계

- Leipzig 문장은 문서 순서를 제거한 sentence corpus라 long-context modeling을 평가하지 못한다.
- 세 snapshot 연도가 다르다.
- content-hash dedup은 near-duplicate를 제거하지 않는다.
- record bootstrap은 corpus sampling uncertainty만 나타내며 model seed uncertainty가 아니다.
- n-gram score head는 neural entropy model과 다르다.
- Python policy runtime은 kernel implementation을 대표하지 않는다.
- downstream Korean morphology·generation quality는 아직 측정하지 않았다.

이 한계 중 neural architecture와 직접 관련된 것은 Phase 1에서 해소한다. long-context와 downstream 평가는 pilot이 성공한 뒤에만 확장한다.

## 11. 재현 명령

Primary Korean 예시는 다음과 같다. Chinese는 `--script cjk`, English는 `--script none`으로 바꾼다.

```bash
PYTHONPATH=src python3 -m jamoflow audit \
  data/processed/leipzig-wikipedia-100k-controls/ko.jsonl \
  --format jsonl \
  --text-field text \
  --script hangul \
  --order 4 \
  --alpha 0.1 \
  --stride 6 \
  --eojeol-cap 24 \
  --orthographic-cap 24 \
  --runtime-repeats 1 \
  --bootstrap-repeats 500 \
  --bootstrap-seed 1729 \
  --output-dir results/stage2-public/ko-order4
```

Aggregate summary:

```bash
python3 scripts/summarize_stage2.py \
  --primary-root results/stage2-public \
  --sensitivity-root results/private/stage2-sensitivity \
  --output-dir results/stage2-public
```
