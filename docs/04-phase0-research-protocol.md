# Phase 0 Research Protocol

> 작성일: 2026-08-10  
> 상태: **Stage 1 — 저비용 측정 프로토콜 확정**  
> 상위 검토: [02-critical-research-direction-review.md](./02-critical-research-direction-review.md)  
> 인용 검증: [03-citation-verification.md](./03-citation-verification.md)

## 0. 목적

이 문서는 JamoFlow의 첫 번째 실제 연구 단계를 사전 고정한다. 목적은 특정 방법의 우월성을 미리 주장하는 것이 아니라 다음 세 요소를 분리해 측정하는 것이다.

1. **Boundary quality**: 어느 위치에 global compute를 배치하는 것이 좋은가?
2. **Boundary detector cost**: 그 위치를 결정하는 데 얼마가 드는가?
3. **End-to-end value**: detector 비용을 포함해도 품질–지연–메모리 trade-off가 개선되는가?

Phase 0은 대규모 LLM을 학습하지 않는다. corpus 통계, causal policy 검증, 작은 byte n-gram proxy, 실제 boundary detector runtime을 사용해 어떤 가설이 작은 모델 학습으로 넘어갈 가치가 있는지 판정한다.

## 1. 연구 질문

### RQ1 — Boundary quality

같은 평균 bytes per patch에서 다음 policy들의 경계가 predictive uncertainty가 높은 위치에 얼마나 잘 compute를 배치하는가?

- fixed byte stride
- UTF-8 codepoint-aligned stride
- SpaceByte-compatible rule
- Hangul syllable boundary
- whitespace/eojeol boundary with a causal cap
- byte entropy boundary
- orthographic candidate-restricted entropy boundary

### RQ2 — Detector cost

같은 corpus와 hardware에서 다음 detector의 비용은 얼마인가?

- integer stride/state machine
- UTF-8/Hangul deterministic automaton
- byte n-gram lookup proxy
- small learned entropy predictor
- integrated learned boundary head

Phase 0에서는 앞의 세 항목을 직접 측정한다. learned predictor와 integrated head는 Phase 1에서 측정한다.

### RQ3 — Hangul-specific value

관찰된 이득이 다음 중 어디에서 오는가?

- generic UTF-8 codepoint alignment
- precomposed Hangul syllable boundary
- whitespace/eojeol structure
- learned context-sensitive uncertainty

한국어와 중국어 control을 사용해 generic 3-byte UTF-8 효과와 Hangul-specific structure를 분리한다.

### RQ4 — Robustness

다음 조건에서 rule 또는 hybrid policy의 이득이 유지되는가?

- 영어·숫자·URL·code 혼입
- compatibility jamo(`ㅋㅋ`, `ㅠㅠ` 등)
- NFC/NFD 혼재
- emoji 및 combining mark
- malformed UTF-8

## 2. 검증할 가설

결과를 선취하지 않도록 양방향으로 작성한다.

### H1 — Rule-only trade-off

- 지지 결과: rule-only가 같은 patch rate에서 entropy boundary와 유사한 uncertainty coverage를 보이며 detector 비용은 더 낮다.
- 반증 결과: detector는 싸지만 high-entropy position을 놓치거나 patch lag가 커진다.

### H2 — Hybrid value

- 지지 결과: orthographic candidate에서만 entropy score를 평가해 full byte-wise entropy policy의 boundary quality 대부분을 보존한다.
- 반증 결과: 중요한 high-entropy position이 candidate set 밖에 많아 품질 상한이 낮다.

### H3 — Hangul specificity

- 지지 결과: codepoint-aligned baseline을 통제한 뒤에도 Hangul/eojeol-aware policy가 추가 이득을 보인다.
- 반증 결과: 한국어와 중국어의 결과가 유사하며 대부분 generic UTF-8 alignment로 설명된다.

### H4 — Runtime realization

- 지지 결과: parameter-free automaton의 낮은 연산량이 실제 latency와 memory traffic 감소로 이어진다.
- 반증 결과: Python/CPU branch, dispatch, synchronization 오버헤드로 FLOPs 차이가 wall-clock에 나타나지 않는다.

## 3. 용어와 경계 방향

### 3.1 Patch boundary 정의

byte sequence를 `x_0, ..., x_(N-1)`이라 할 때 `b_t=1`은 **byte `x_t`를 새 patch의 첫 byte로 처리한다**는 뜻이다. 각 record는 항상 `b_0=1`이다.

### 3.2 Incremental causality

generation-time policy는 다음 조건을 만족해야 한다.

```text
b_t = f(x_<t, state_<t)
```

`b_t`는 `x_t` 또는 그 이후 byte를 볼 수 없다. offline evaluation에서도 같은 streaming state machine으로 boundary를 계산한다.

다음 정보는 causal하다.

- 직전까지의 UTF-8 continuation state
- 직전에 완성된 codepoint의 script/category
- 이미 출력된 whitespace·punctuation
- prefix로 계산한 predictive entropy
- 현재 patch의 누적 byte 수

다음 정보는 그대로 사용하면 non-causal하다.

- 아직 출력되지 않은 다음 문자의 종류
- 완성된 전체 어절을 본 형태소 분석
- 오른쪽 문맥을 사용한 segmentation
- true future surprisal을 사용한 online boundary

true future surprisal은 oracle upper bound 분석에만 사용한다.

### 3.3 Patch rate

policy의 계산량을 통제하기 위해 다음을 보고한다.

```text
average bytes per patch = evaluated bytes / number of patches
```

Boundary quality 비교는 가능한 한 동일한 average bytes per patch에서 수행한다. rule policy의 고유 rate와 정확히 일치하지 않을 경우 calibration split에서 가장 가까운 entropy threshold 또는 rule parameter를 선택하고 test split에서는 고정한다.

## 4. 비교 policy

### P0 — Fixed byte stride

매 `k` bytes마다 boundary를 둔다. 가장 싼 비언어학적 baseline이다.

### P1 — Causal codepoint-aligned stride

현재 patch가 `k` bytes 이상이고 UTF-8 parser가 다음 byte 직전에 codepoint boundary state라면 새 patch를 시작한다. 다음 codepoint 길이를 미리 보지 않으므로 strict maximum이 아니라 **첫 causal codepoint boundary after budget**이다.

### P2 — SpaceByte-compatible rule

SpaceByte의 spacelike byte 정의와 global-block cadence를 boundary-start convention으로 옮긴다.

- ASCII letter/digit가 아님
- UTF-8 continuation byte가 아님
- 연속 spacelike byte의 중복 trigger 방지

원 논문과 boundary index convention 차이가 생길 수 있으므로 구현 보고서에 정확한 byte index 예시를 포함한다.

### P3 — Hangul syllable boundary

직전에 완성된 codepoint가 precomposed Hangul syllable `U+AC00..U+D7A3`이면 다음 byte에서 새 patch를 시작한다. compatibility jamo, conjoining jamo, old Hangul은 별도 category로 보고하며 자동으로 완성형 음절로 간주하지 않는다.

### P4 — Causal eojeol/capped rule

이미 출력된 whitespace·punctuation 뒤에서 boundary를 시작한다. 지나치게 긴 patch는 codepoint boundary에서 causal byte budget cap으로 자른다.

### P5 — Byte entropy policy

train split에서 학습한 byte n-gram model의 predictive entropy가 calibration threshold보다 높으면 boundary를 시작한다. Phase 0 proxy일 뿐 BLT의 100M entropy LM과 동등하다고 주장하지 않는다.

### P6 — Candidate-restricted entropy policy

predictive entropy를 모든 byte 위치가 아니라 다음 causal candidate에서만 평가한다고 가정한다.

- UTF-8 codepoint boundary
- whitespace·punctuation 이후
- maximum patch budget 도달 후 첫 codepoint boundary

이 policy는 hybrid learned router가 byte-wise router evaluation을 얼마나 줄일 수 있는지에 대한 upper-bound proxy다.

## 5. 데이터 분할

record 단위 deterministic hash split을 사용한다.

- train: 80%
- calibration: 10%
- test: 10%

동일 record가 여러 파일이나 corpus에 중복될 수 있으므로 향후 실제 dataset에서는 normalized text hash deduplication을 분할 전에 적용한다. Phase 0 도구는 record hash를 사용해 입력 순서와 무관하게 같은 split을 재현해야 한다.

### 5.1 Stage 1 입력

대용량 다운로드 전에는 다음으로 구현을 검증한다.

- synthetic Korean/English/code-mixed fixtures
- 저장소 문서의 한국어 text
- malformed UTF-8 fixture

이 결과는 연구 결론으로 사용하지 않고 tool validation으로만 사용한다.

### 5.2 Stage 2 후보 데이터

Stage 1 결과와 사용자 승인 후 다음 조건을 만족하는 공개 corpus sample을 선택한다.

- 재배포·연구 사용 가능한 라이선스
- raw 또는 normalization 상태 확인 가능
- 한국어와 중국어 control을 같은 pipeline으로 처리 가능
- domain별 분리 가능
- deduplication 방법 기록 가능

데이터 다운로드 규모와 저장 위치는 Stage 2 시작 전에 별도 승인받는다.

## 6. Phase 0 entropy proxy

### 6.1 모델

표준 라이브러리만 사용하는 byte n-gram backoff model을 기본 proxy로 사용한다.

- order: CLI parameter, 기본 4
- vocabulary: 256 raw bytes
- additive smoothing
- record boundary에서 context reset
- unseen context는 더 짧은 suffix context로 backoff

### 6.2 측정값

각 byte 위치 `t`에서 다음을 계산한다.

- predictive entropy `H(X_t | x_<t)`
- observed surprisal `-log2 p(x_t | x_<t)`

predictive entropy는 causal boundary policy에 사용한다. observed surprisal은 사후 진단과 oracle 분석에만 사용한다.

### 6.3 한계

byte n-gram 결과는 다음을 증명하지 않는다.

- BLT 학습 후 BPB
- neural entropy router의 정확도
- downstream task quality
- global/local Transformer interaction

Phase 0의 역할은 명백히 열등한 policy를 제거하고 Phase 1 config 수를 줄이는 것이다.

## 7. 핵심 지표

### 7.1 Compression 및 구조

- total bytes
- number of patches
- mean/median/p95/max bytes per patch
- boundaries per KiB
- boundary-inside-UTF-8-codepoint rate
- boundary-inside-precomposed-Hangul-syllable rate
- script/category별 patch length

### 7.2 Boundary quality

- mean predictive entropy at selected boundaries
- mean observed surprisal at selected boundaries
- top-`M` entropy boundary precision/recall, `M`은 policy boundary budget과 동일
- top-decile entropy recall
- high-entropy patch lag: high-entropy 위치에서 직전 boundary까지 byte 거리
- oracle entropy-capture ratio

### 7.3 Router-effort proxy

- score evaluations per byte
- score evaluations per codepoint
- candidate positions / all byte positions
- rule state transitions per byte

이 값은 FLOPs나 latency 자체가 아니다. runtime microbenchmark와 별도로 보고한다.

### 7.4 Unicode 및 corpus audit

- valid/invalid UTF-8 records와 bytes
- NFC/NFD exact-match records
- normalization으로 변하는 codepoint 수
- precomposed Hangul syllables
- modern conjoining jamo
- compatibility jamo
- Hangul extended/old jamo
- CJK characters
- ASCII Latin/digit
- whitespace/punctuation
- other/emoji/combining marks
- record-level mixed-script rate

## 8. Runtime microbenchmark 원칙

Phase 0 Python 구현은 과학적 정확성과 reproducibility가 목적이다. Python runtime 결과를 production kernel 성능으로 일반화하지 않는다.

다음 두 층으로 나눈다.

1. **Algorithmic operation count**
   - byte당 state transition
   - entropy score evaluation 빈도
   - table lookup 수

2. **Reference wall-clock**
   - 동일 process·동일 input·warm-up 후 반복
   - median, p95, standard deviation
   - throughput bytes/sec
   - Python version과 hardware 기록

Phase 1에서 GPU-integrated implementation을 별도로 측정한다.

## 9. Phase 0 판정 규칙

최종 숫자를 보기 전에 다음 논리적 gate를 고정한다. 절대 latency 임계값은 Stage 1 microbenchmark로 measurement noise를 확인한 뒤 Stage 2 실행 전에 확정한다.

### Gate A — Candidate coverage

candidate-restricted policy가 full entropy policy의 high-entropy boundary를 충분히 포함하지 못하면 hybrid candidate restriction을 중단한다.

판정은 동일 boundary budget에서 confidence interval을 사용한다. 임의의 `5%p/15%p` 기준을 재사용하지 않는다.

### Gate B — Trivial baseline

Hangul-specific rule은 최소한 다음 중 하나보다 의미 있는 이득을 보여야 한다.

- fixed byte stride
- causal codepoint-aligned stride
- SpaceByte-compatible rule

이기지 못하면 Hangul-specific architecture claim을 중단하고 generic UTF-8 measurement result로 재분류한다.

### Gate C — Code-mixing robustness

한글 비율이 낮아질수록 성능이 변하는 곡선을 보고한다. 특정 임의 비율 하나에서 실패했다고 fatal 판정을 내리지 않고 break-even region을 측정한다.

### Gate D — Phase 1 진입

100~300M neural pilot은 다음을 모두 만족할 때만 제안한다.

1. 정책 구현과 causality test 통과
2. trivial baseline 대비 개선 가능성 확인
3. candidate restriction 또는 detector-cost 절감의 측정 가능한 여지 확인
4. code-mixed control에서 즉시 붕괴하지 않음
5. 필요한 데이터·compute·라이선스 계획 완성

Phase 1은 별도 사용자 승인 후 시작한다.

## 10. 재현성 산출물

Stage 1 완료 시 다음을 커밋한다.

- corpus reader
- Unicode audit
- byte n-gram entropy proxy
- causal boundary policies
- prefix-causality tests
- boundary metrics
- JSON 및 Markdown report generator
- synthetic fixtures와 unit tests
- 실행 명령과 환경 정보

Stage 2 완료 시 추가한다.

- dataset manifest와 license record
- content hash와 split manifest
- corpus별 결과 JSON/Markdown
- calibration parameter
- runtime raw measurements
- Go/No-Go decision record

## 11. 단계별 승인 지점

| 단계 | 범위 | 완료 산출물 | 다음 단계 전 승인 |
|---|---|---|---|
| Stage 1 | 로컬 구현·synthetic/local smoke test | protocol, audit tool, tests, 비대표 초기 결과 | 공개 corpus download 승인 |
| Stage 2 | 제한된 공개 corpus Phase 0 | corpus audit, matched-rate 결과, runtime proxy | 100~300M pilot 승인 |
| Phase 1 | 100~300M controlled pretraining | multi-seed BPB 및 실제 inference 측정 | 1B scaling 승인 |
| Phase 2 | 최소 1B scaling + downstream | scale result와 paper draft | 추가 scale/제출 전략 승인 |

이 문서의 Stage 1 범위를 넘어서는 데이터 다운로드, 유료 GPU 사용, 대규모 pretraining은 자동으로 진행하지 않는다.
