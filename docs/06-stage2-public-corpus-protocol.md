# Stage 2 Public Corpus Protocol

> 사전 고정일: 2026-08-10  
> 상태: **공개 corpus 다운로드 전 고정**  
> Stage 1 결과: [05-stage1-local-audit-results.md](./05-stage1-local-audit-results.md)

## 1. 목적

Stage 2는 private Markdown convenience sample에서 관찰한 신호가 공개 corpus와 언어 control에서도 유지되는지 검증한다. neural LM 우월성을 주장하는 단계가 아니라 Phase 1의 실험 공간을 줄이는 단계다.

주요 질문은 다음과 같다.

1. UTF-8 codepoint candidate restriction의 entropy 평가 절감률과 proxy coverage가 한국어·중국어·영어에서 어떻게 다른가?
2. 한국어에서 관찰되는 효과가 generic multi-byte UTF-8 정렬로 설명되는가?
3. 순수 Hangul boundary, whitespace/cap rule, learned-proxy hybrid 중 무엇을 neural pilot에 남길 것인가?
4. normalization, jamo, code mixing에 대한 별도 stress transformation에서 결론이 유지되는가?

## 2. 데이터 선택

Leipzig Corpora Collection의 normed-size Wikipedia sentence corpus를 사용한다.

| 언어 | Archive | Snapshot | Sentences | Compressed bytes |
|---|---|---:|---:|---:|
| Korean | `kor_wikipedia_2021_100K.tar.gz` | 2021 | 100,000 | 21,536,996 |
| Chinese | `zho_wikipedia_2018_100K.tar.gz` | 2018 | 100,000 | 24,202,817 |
| English | `eng_wikipedia_2016_100K.tar.gz` | 2016 | 100,000 | 25,566,797 |

선택 이유는 세 가지다.

- 제공처가 normed-size corpus를 무작위 선택 문장으로 설명한다.
- 언어별 형식과 문장 수가 같아 parser 및 record-size confound를 줄일 수 있다.
- 한국어와 중국어를 비교해 3-byte UTF-8 효과와 Hangul 고유 효과를 분리하고, 영어로 ASCII-heavy control을 둔다.

snapshot 연도가 같지 않고 sentence selection/GDEX 처리의 세부 차이가 있을 수 있다. 따라서 언어 간 절대 entropy를 직접 순위화하지 않고, 각 언어 내부의 normalized capture와 상대 변화량을 비교한다.

## 3. 라이선스 및 데이터 보존

- Leipzig download terms는 다운로드 text corpus를 CC BY로 제공한다고 명시하지만 버전을 표시하지 않는다.
- 원천 Wikipedia text에는 CC BY-SA/GFDL 및 page별 예외가 있을 수 있다.
- 논문과 저장소에서 Leipzig Corpora Collection과 Wikimedia를 모두 attribution한다.
- archive와 파생 JSONL은 Git에 넣지 않는다.
- URL, byte length, SHA-256, preparation code, aggregate statistics만 커밋한다.
- corpus 문장을 논문 부록이나 test fixture로 재배포하지 않는다.

정확한 URL과 terms는 [`data/manifests/leipzig-wikipedia-100k.json`](../data/manifests/leipzig-wikipedia-100k.json)에 기록한다.

## 4. 전처리

1. archive의 byte length와 SHA-256을 검증한다.
2. tar를 filesystem에 풀지 않고 정확히 하나의 `*[-_]sentences.txt` member만 읽는다.
3. 각 UTF-8 line을 첫 tab에서 `sentence_id`와 `text`로 나눈다.
4. newline만 제거하고 Unicode normalization, whitespace 정리, case folding은 하지 않는다.
5. JSONL에는 `id`, `language`, `text`만 저장한다.
6. exact text bytes로 중복 제거하고 content hash로 80/10/10 split한다.

원문 상태 분석과 별도로 생성한 NFD/compatibility-jamo/code-mixed stress set은 원 corpus와 혼동되지 않도록 별도 결과로 기록한다.

## 5. 사전 고정 비교

각 언어에서 최소 다음 policy를 비교한다.

- fixed byte stride: 4, 6, 8
- causal UTF-8 codepoint-aligned stride: 4, 6, 8
- SpaceByte-compatible cadence
- whitespace/punctuation + causal byte cap: 12, 24, 48
- full byte-wise predictive-entropy proxy
- UTF-8 codepoint-candidate predictive-entropy proxy
- Korean에서 Hangul-syllable rule 및 Hangul/whitespace candidate hybrid
- Chinese에서 대칭적인 CJK-character rule/control

모든 online policy는 prefix-causal test를 통과해야 한다. threshold는 calibration split에서만 선택하고 test split에서 고정한다.

## 6. 분석과 판정

언어별로 다음을 보고한다.

- Unicode/category audit와 bytes per codepoint
- achieved bytes per patch와 calibration mismatch
- boundary entropy, oracle capture, top-budget overlap, top-decile recall, patch lag
- UTF-8/codepoint 및 해당 script character 내부 boundary
- candidate positions와 entropy score evaluations per byte
- record bootstrap 95% confidence interval
- n-gram order 및 additive-smoothing sensitivity

해석 규칙은 다음과 같다.

1. pure Hangul rule이 generic codepoint/eojeol baseline을 이기지 못하면 rule-only architecture는 폐기한다.
2. codepoint candidate 절감과 coverage가 Korean과 Chinese에서 유사하면 generic UTF-8 효과로 분류한다.
3. Hangul feature를 더한 candidate가 generic candidate보다 독립적인 개선을 보일 때만 Korean-specific claim을 Phase 1에 남긴다.
4. proxy 결과가 entropy definition에 자기참조한다는 사실을 유지하고 neural quality로 표현하지 않는다.
5. Python runtime은 appendix diagnostic으로만 두고 총비용 주장은 Phase 1의 integrated implementation까지 보류한다.

## 7. Stage 2 완료 조건

- archive hash가 manifest에 고정됨
- 3개 언어 각 100K record 준비 및 audit 완료
- confidence interval과 sensitivity 결과 생성
- private Stage 1과 public Korean 결과의 방향 일치 여부 보고
- neural pilot에 남길 policy가 최대 3개로 축소됨
- Phase 1 compute budget, model size, seed 수, stopping rule을 결과 보기 전에 문서화함
