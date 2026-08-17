# Stage 1 Local Audit Results

> 실행일: 2026-08-10  
> 상태: **Stage 1 완료 — 도구 검증 및 비대표 표본의 가설 선별**  
> 상위 프로토콜: [04-phase0-research-protocol.md](./04-phase0-research-protocol.md)  
> 주의: 이 문서는 neural LM 성능 결과나 논문 결론이 아니다.

## 1. 이번 단계가 답하는 것과 답하지 않는 것

Stage 1의 목적은 대규모 학습 전에 측정 코드의 인과성·Unicode 처리·비교 방식이 작동하는지 확인하고, 명백히 약한 가설을 제거하는 것이다. 다음은 이번 단계가 답할 수 있다.

- 각 rule이 UTF-8 또는 완성형 한글 음절 내부에 경계를 두는가?
- 작은 causal byte n-gram proxy가 높게 평가한 위치를 각 policy가 얼마나 포착하는가?
- entropy 평가 위치를 UTF-8 codepoint boundary로 제한했을 때 평가 횟수가 얼마나 줄어드는가?
- rule-only 후보 중 어느 것이 다음 단계에서 더 검토할 가치가 있는가?

다음은 답할 수 없다.

- BLT 또는 decoder-only LM의 BPB/perplexity가 개선되는가?
- GPU end-to-end latency, KV cache, memory traffic이 줄어드는가?
- downstream 한국어 능력이 유지되는가?
- 관찰된 신호가 한국어 고유 구조 때문인가, 일반적인 UTF-8 정렬 때문인가?

## 2. 읽기 전용 표본 사용

사용자가 미리 작성한 `../assist-creator/vault`의 Markdown 문서를 **읽기 전용 convenience sample**로 사용했다.

- directory scan은 `--include-suffix .md`로 제한했다.
- 파일을 수정·이동·정규화하거나 저장소 안으로 복사하지 않았다.
- 원문, 파일명 목록, record별 결과는 커밋하지 않았다.
- private 실행 보고서는 `.gitignore`가 적용되는 `results/private/`에만 두었다.
- 이 문서에는 corpus 전체의 aggregate만 기록했다.

실행 시점에 `.md` 1,265개를 발견했고, exact-byte hash 중복 제거 뒤 1,227개 record가 남았다. deterministic content-hash split은 train 962, calibration 146, test 119개였다. 전체 크기는 4,958,377 bytes, 3,103,752 Unicode codepoints다.

이 표본은 한국어 산문만으로 이루어진 corpus가 아니다. Markdown 문법, 영어, code, URL, 숫자와 개인 메모가 섞여 있다. 따라서 실제 사용 환경의 stress sample로는 유용하지만 모집단을 대표하는 한국어 corpus로 간주하지 않는다.

## 3. 구현 검증

- 14개 unit test 통과
- 5개 rule과 각 matched entropy/hybrid policy 조합에 대해 300회 prefix-causality check 통과
- 1,227개 record 모두 strict UTF-8 decode 성공
- 모든 record가 NFC exact match
- 완성형 한글 음절 908,512개, ASCII Latin 971,251개
- Hangul/CJK/Latin 중 둘 이상을 포함한 mixed-script record 1,089개

`b_t`가 byte `x_t` 앞의 새 patch 시작을 뜻하도록 boundary convention을 통일했다. entropy policy의 `b_t`는 `x_<t`에서 계산한 predictive entropy만 사용한다. observed byte의 surprisal은 online decision에 사용하지 않았다.

## 4. 핵심 결과

아래 표는 test split 결과다. 서로 같은 group 안에서 calibration split의 평균 bytes/patch에 맞추었지만, n-gram entropy 값의 tie 때문에 test의 rate는 완전히 같지 않다. 따라서 작은 차이를 순위로 해석하지 않는다.

| Policy | Role | Bytes/patch | Boundary H | Oracle capture | Top-budget overlap | Top-decile recall | Mean lag | UTF-8 split | Score eval/byte |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed byte 6 | rule | 5.997 | 3.838 | 0.517 | 0.169 | 0.168 | 2.488 | 0.364 | 0.000 |
| codepoint stride 6 | rule | 6.439 | 3.681 | 0.492 | 0.150 | 0.151 | 2.726 | 0.000 | 0.000 |
| SpaceByte-compatible | rule | 4.407 | 3.540 | 0.499 | 0.146 | 0.135 | 4.057 | 0.510 | 0.000 |
| Hangul syllable | rule | 5.652 | 2.934 | 0.398 | 0.063 | 0.047 | 212.394 | 0.000 | 0.000 |
| eojeol + 24-byte cap | rule | 4.778 | 4.069 | 0.565 | 0.245 | 0.238 | 3.334 | 0.000 | 0.000 |
| candidate entropy, fixed-rate group | hybrid | 5.858 | 7.172 | 0.969 | 0.794 | 0.817 | 1.835 | 0.000 | 0.640 |
| candidate entropy, SpaceByte-rate group | hybrid | 4.556 | 6.786 | 0.951 | 0.780 | 0.817 | 1.180 | 0.000 | 0.640 |
| candidate entropy, eojeol-rate group | hybrid | 4.842 | 6.892 | 0.956 | 0.784 | 0.817 | 1.305 | 0.000 | 0.640 |

`entropy_matched`의 oracle capture, overlap, recall은 같은 n-gram entropy를 policy와 oracle 양쪽 정의에 사용하므로 구성상 1이다. 이는 learned entropy patcher의 우수성을 독립적으로 검증한 결과가 아니어서 표에서 제외했다.

### 4.1 순수 Hangul-syllable policy는 다음 단계의 주력 가설이 아니다

완성형 한글 음절 뒤에만 경계를 두는 rule은 이번 표본에서 가장 약했다.

- oracle entropy capture: 0.398
- top-budget overlap: 0.063
- top-decile recall: 0.047
- high-entropy position에서 직전 patch boundary까지 평균 거리: 212.394 bytes

긴 lag의 주원인은 영어·code·URL 구간에서 새 boundary를 거의 만들지 못하는 정의 자체다. 이 결과는 “한글 음절 신호가 전혀 쓸모없다”를 증명하지 않는다. 다만 **순수 syllable-only patcher가 혼합 텍스트용 기본 architecture가 될 수 있다**는 현재 형태의 주장은 기각한다. 향후에는 Hangul을 단독 결정 규칙이 아니라 generic Unicode candidate set 안의 feature 또는 prior로만 검토한다.

### 4.2 단순 rule 중에는 eojeol+capped가 가장 낫지만 learned uncertainty를 대체하지 못한다

whitespace·punctuation과 causal 24-byte cap을 결합한 rule은 rule-only 중 boundary entropy, oracle capture, overlap, recall이 가장 높았다. 그러나 각각 4.069 bits, 0.565, 0.245, 0.238에 그쳐 같은 proxy의 uncertainty 위치를 대부분 대체하지는 못했다. 따라서 “형태·띄어쓰기 규칙만으로 learned patcher를 제거한다”는 주장도 현재 증거로 지지되지 않는다.

### 4.3 Unicode candidate restriction은 다음 단계로 가져갈 가치가 있다

UTF-8 codepoint가 완성된 위치에서만 entropy를 평가한 hybrid는 다음 신호를 보였다.

- entropy score evaluation: 1.000에서 0.640 per byte로 감소, 즉 약 36.0%의 위치 제거
- oracle entropy capture: rate group에 따라 0.951–0.969
- top-budget overlap: 0.780–0.795
- top-decile recall: 0.817
- UTF-8 및 완성형 한글 음절 내부 boundary: 0

이것은 **candidate restriction이 learned detector를 없앤다**는 결과가 아니다. codepoint 내부의 entropy score를 생략해도 이 n-gram proxy가 중요하다고 본 신호 대부분을 보존할 수 있다는 Stage 2 진입 신호다. 실제 절약량은 score head 구현, batching, kernel fusion, global block 비용에 따라 달라진다.

### 4.4 SpaceByte-compatible 경계의 phase를 품질 문제로 오해하면 안 된다

현재 boundary-start convention으로 옮긴 SpaceByte-compatible cadence는 non-initial boundary의 51.0%를 UTF-8 codepoint 내부, 50.3%를 완성형 한글 음절 내부에 둔다. 이는 multi-byte UTF-8에서 ASCII 중심 spacelike rule의 phase가 어떻게 나타나는지를 보여 준다.

그러나 BLT/SpaceByte의 local byte encoder는 byte 내부 경계를 처리할 수 있다. 따라서 이 비율만으로 원 architecture의 오류나 품질 저하를 주장하지 않는다. 다음 단계에서 동일 patch rate의 codepoint-aligned control과 실제 neural loss를 비교해야 한다.

## 5. 위협 요인과 측정 부채

1. **표본 편향**: 한 사용자의 Markdown vault이며 공개 한국어 분포를 대표하지 않는다.
2. **자기참조 proxy**: boundary와 oracle이 같은 n-gram predictive entropy를 사용한다.
3. **proxy capacity**: 4-gram은 장거리 의미·형태 불확실성을 표현하지 못한다.
4. **rate mismatch**: discrete entropy tie 때문에 일부 calibration/test group의 bytes/patch가 완전히 일치하지 않는다.
5. **record 크기**: 파일 단위 record라 큰 문서가 byte-weighted 결과를 지배할 수 있다.
6. **runtime 비현실성**: Python reference runtime은 production GPU latency를 나타내지 않는다. 현재 candidate scan이 반복되어 오히려 Python에서는 느리다.
7. **control 부재**: 중국어·영어·NFD·compatibility jamo별 통제 실험을 아직 하지 않았다.
8. **통계 부재**: confidence interval, bootstrap, seed variation이 아직 없다.

Stage 2에서는 최소한 1, 4, 5, 7, 8을 해소하고, neural phase에서 2, 3, 6을 직접 다룬다.

## 6. Gate 판정

| Gate | 판정 | 근거 |
|---|---|---|
| 구현·causality | Pass | unit test 14개 및 prefix check 300회 통과 |
| Candidate coverage | Provisional pass | 36.0% score 위치 감소에서 proxy capture 95.1–96.9% |
| Trivial baseline | Pure Hangul rule no-go | codepoint stride와 eojeol rule보다 boundary-quality proxy가 낮음 |
| Code-mixing robustness | 미판정 | mixed sample 신호는 있으나 mixture별 stratification이 없음 |
| Neural Phase 1 진입 | 보류 | 공개 corpus·언어 control·confidence interval이 먼저 필요 |

## 7. 수정된 연구 방향

Stage 1 뒤의 주력 질문은 다음과 같다.

> **Can causal Unicode/orthographic candidate restriction reduce the total cost of learned byte-level patching while preserving language-model quality across Korean and multilingual text?**

비교 축은 다음처럼 정리한다.

1. fixed byte / codepoint stride / SpaceByte-compatible / eojeol-capped rule
2. full byte-wise learned entropy patcher
3. codepoint-candidate learned entropy patcher
4. codepoint + Hangul/whitespace feature를 쓰는 learned hybrid
5. 같은 global patch budget에서 detector를 포함한 total FLOPs, latency, memory, BPB

한국어 특화 주장은 generic codepoint control과 중국어 control을 이긴 경우에만 유지한다. 그렇지 않으면 결과를 “UTF-8-aware cost-constrained patching”으로 일반화하고, Jamo/Hangul은 분석 대상 언어 중 하나로 낮춘다.

## 8. 재현 명령

저장소 문서 smoke report:

```bash
PYTHONPATH=src python3 -m jamoflow audit \
  docs/00-topic-selection.md \
  docs/01-verification-report.md \
  docs/02-critical-research-direction-review.md \
  docs/03-citation-verification.md \
  docs/04-phase0-research-protocol.md \
  --format plain \
  --corpus-label "JamoFlow repository documents" \
  --interpretation-note "Repository research notes are not a representative Korean corpus." \
  --output-dir results/stage1-local \
  --runtime-repeats 5
```

읽기 전용 Markdown convenience sample:

```bash
PYTHONPATH=src python3 -m jamoflow audit ../assist-creator/vault \
  --format plain \
  --plain-record-unit file \
  --include-suffix .md \
  --corpus-label "Assist Creator vault Markdown convenience sample" \
  --interpretation-note "User-authored Markdown mixes Korean prose, English, code, and notes; it is read-only and not representative of a population corpus." \
  --output-dir results/private/vault-stage1 \
  --runtime-repeats 3
```

## 9. 실행 승인 변경 기록

초기 프로토콜은 각 단계 전에 사용자 승인을 받도록 정했다. 2026-08-10 사용자가 중간 단계 승인을 일괄 허용하고 논문 수준의 연구 및 논문 초안까지 자율적으로 진행하도록 범위를 확장했다. 이후에도 방법론적 gate는 유지하지만 승인 대기를 이유로 연구를 멈추지 않는다. 유료 외부 자원이나 별도 권한이 필요한 행위는 실제 권한 범위 안에서만 수행한다.
