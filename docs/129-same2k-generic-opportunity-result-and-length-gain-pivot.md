# Same-2K generic opportunity 결과와 Length-Gain vocabulary 전환

> 작성일: 2026-08-14
>
> 상태: independently replayed calibration-development token-only result;
> model quality 또는 publication latency evidence 아님
>
> 후속 결과: Byte-LengthGain-2K도 10% gate를 실패해 same-2K tokenizer branch를
> 종료했다. `docs/131-length-gain-result-and-compositional-head-pivot.md` 참조.

## 결론

같은 2,048 vocabulary에서 deterministic Byte-Unigram vocabulary를 scored Viterbi,
left-most-longest, global minimum-token DP로 각각 적용했지만 세 역할 모두 exact BPE-2K보다
token sequence가 **11.7--15.7% 더 길었다.** 10% 감소 gate와 방향이 반대다.

따라서 이 vocabulary로 model을 학습하지 않는다. 다음 단계는 “긴 piece가 많은 vocabulary”가
아니라 train split의 실제 비중첩 token 절감량을 직접 최적화하는 Byte-LengthGain-2K와,
동일 objective에 한국어 음절·어절 completeness를 적용한 짝지은 대조군이다.

이 전환은 한국어 특화 아이디어를 억지로 유지하기 위한 것이 아니다. 이번 결과가 명확히
보인 실패 원인, 즉 **vocabulary 구성 목적과 실제 sequential-step 절감의 불일치**를 교정한다.

## 봉인된 결과

Calibration complete UTF-8 prefix는 7,999,999 bytes이고 1-byte incomplete suffix는 모든
역할에서 동일하게 제외했다. Continuation은 사전 고정한 36 measured cases의 별도 128-byte
continuation token 수 합이다.

| role | calibration tokens | bytes/token | BPE 대비 | continuation tokens | BPE 대비 | encode MB/s |
|---|---:|---:|---:|---:|---:|---:|
| Byte-BPE-2K | **2,263,476** | **3.534** | 기준 | **1,288** | 기준 | 6.372 |
| Byte-Unigram scored | 2,602,473 | 3.074 | **+14.98%** | 1,441 | **+11.88%** | 5.688 |
| same vocab, left-most-longest | 2,619,138 | 3.054 | **+15.71%** | 1,453 | **+12.81%** | **7.114** |
| same vocab, minimum-token DP | 2,595,380 | 3.082 | **+14.66%** | 1,439 | **+11.72%** | 5.788 |

표의 encode throughput은 independent replay median이다. Worker replication에서도 각각
6.281, 5.754, 7.130, 5.673 MB/s로 방향이 같았다. Selection에는 timing을 쓰지 않았다.

Minimum-token DP는 같은 vocabulary의 left-most-longest보다 calibration token을 약 0.91%
줄였고 scored Viterbi보다 약 0.27% 줄였다. 즉 segmentation choice가 일부 영향을 주지만,
BPE와의 약 15% 격차를 설명하거나 회복하지 못한다. 실패의 주된 위치는 vocabulary다.

## 구조 분석

| diagnostic | BPE-2K | Byte-Unigram vocabulary |
|---|---:|---:|
| used vocabulary | 1,956 | 1,952--1,960 |
| maximum used piece bytes | 13 | 48 |
| multibyte pieces | 1,792 | 1,792 |
| strict-UTF-8 multibyte fraction | 84.82% | 97.43% |
| Hangul-containing multibyte pieces | 1,433 | 1,637 |
| cross-eojeol multibyte pieces | 0 | 227 |
| boundary-complete cross-eojeol pieces | 0 | 4 |

여기서 중요한 부정 결과는 다음과 같다.

1. Maximum piece를 13→48 bytes로 늘려도 실제 corpus compression은 좋아지지 않았다.
2. UTF-8로 완결된 piece와 Hangul-containing piece를 더 많이 갖는 것만으로는 충분하지 않다.
3. 공백을 넘는 piece 227개도 step을 줄이지 못했다. Cross-eojeol 허용 자체가 해법이 아니다.
4. Global minimum-token segmentation도 약한 vocabulary를 구하지 못했다.
5. BPE가 학습한 짧은 hierarchical merge vocabulary는 이 2K regime에서 likelihood-based
   direct pieces보다 훨씬 높은 실제 사용 효율을 보였다.

따라서 “한국어 완결 token을 많이 넣자”는 단순 규칙은 기각한다. 후보 piece는 구조적으로
그럴듯할 뿐 아니라, 기존 vocabulary에서 잃는 coverage 비용까지 포함한 **순 token 절감량**을
보여야 한다.

## 탐색 HF Unigram과 방향이 달라진 이유

docs/122의 unsealed HF Unigram no-regex는 BPE보다 token 수가 3.97% 적었지만, sealed
SentencePiece control은 약 15% 많았다. 두 결과를 재현 실패로 섞어 해석하지 않는다.

- vocabulary trainer와 objective implementation이 다르다.
- SentencePiece에는 exact 256-byte fallback projection과 deterministic ordering이 적용됐다.
- HF exploration은 seed를 봉인할 수 없는 비결정적 API였고 selection evidence가 아니다.
- 두 경우 모두 10% gate를 통과하지 못한다는 연구 결정은 동일하다.

이 차이는 tokenizer algorithm 이름보다 exact trainer/runtime/artifact identity를 보고해야 한다는
systems 교훈이다. HF exploration의 더 나은 3.97% 값을 골라 model training으로 보내지 않는다.

## 독립 검증과 protocol iteration

v6 summarizer는 저장 tokenizer JSON, SentencePiece model, ordered pieces/scores를 hash로 확인하고,
실제 `Tokenizer.from_file()` scored runtime과 bounded trie/DP runtime을 재구성했다. 8MB token
stream, token-ID hash, 42개 prompt/continuation/joint count, vocabulary structure를 모두 다시
계산해 worker와 exact 비교했다. Wall-clock만 별도 replication으로 보존했다.

- plan payload SHA-256: `5e12f58ba45fb3c730b32dcb8d6456a3a4dd559393080033990c9f835f9d1675`
- plan file SHA-256: `154d2249694a152b17680cfb431350aca1082c222dd17232835d016659aef37d`
- result payload SHA-256: `bd76af9ff95213fbff20e4df49d7a719745fee7b6e3e2002dbbf947b006d5299`
- result file SHA-256: `1e9e57e76998c8729c68cee4f5a3f8dd8eb4528eda933597bf6eb4060ab09126`
- worker SHA-256: `8eb233a700c1d0ae77483e0cce20dfa37f8b8ba47e2ba2f62b1d89f61e01f759`

v3--v5의 교정은 결과 수치를 본 뒤 이루어진 것이 아니다. v3은 결과 artifact 공개 전 runtime
병목, v4/v5는 summarizer가 metric을 읽기 전 tokenizer identity gate에서 중단됐다. 실패 marker와
artifact hashes는 docs/126--128에 보존했다. 최종 v6 tokenizer/model/pieces hashes가 v4/v5와
동일해 deterministic trainer도 확인됐다.

## 필요한 계획 수정

### 중단할 것

- 이 Byte-Unigram vocabulary의 model training
- same vocabulary에서 segmentation만 더 탐색하는 일
- 긴 UTF-8/Hangul/cross-space piece 수 자체를 optimization target으로 삼는 일
- non-deterministic HF Unigram exploration의 3.97% 결과를 사후 후보로 부활시키는 일

### 다음 sealed gate: Byte-LengthGain-2K

다음 vocabulary construction은 train split만 사용하고, 후보 하나의 점수를 raw 길이나
likelihood가 아니라 현재 segmentation에서의 비중첩 token 절감으로 정의한다.

1. mandatory byte IDs 0--255와 exact 2,048 budget을 유지한다.
2. generic 후보 pool은 train-only byte substrings이며 maximum 48 bytes로 고정한다.
3. paper-adapted `frequency × current-token length`와 실제 immediate saving
   `frequency × (current-token length - 1)`을 구분해 기록한다.
4. 후보 채택 뒤 current train segmentation을 갱신해 중복·중첩 gain을 다시 계산한다.
5. frozen application은 bounded-trie left-most-longest, 같은 vocabulary의 minimum-token DP를
   segmentation ablation으로 둔다.
6. Korean-complete variant는 같은 construction budget과 score를 쓰되 UTF-8 codepoint 경계,
   Hangul syllable completeness, eojeol-boundary 상태를 candidate eligibility에만 적용한다.
7. Generic과 Korean variant 모두 BPE 대비 calibration/continuation 10% 감소를 먼저 요구한다.

구현 비용 때문에 train corpus나 후보 pool을 결과를 보며 줄이지 않는다. 먼저 resource-only
preflight로 exact full-train 알고리즘과 candidate cap을 고정한다. 만약 full Length-Gain 자체가
10%를 못 넘으면 same-2K tokenizer branch를 model training 전에 종료한다.

### model 단계 진입 조건

통과 역할만 exact 2K×8L, 128M raw bytes one-seed training으로 보낸다. 이후에도 성공은 아니다.

- BPE-2K 대비 raw-byte document BPB noninferiority
- generic 대비 Korean constraint의 quality--latency Pareto 기여
- tokenizer를 포함한 batch-1 trained-model E2E 최소 10% 개선 가능성
- 이후 multi-seed, sealed final quality, fresh-process timing replication

이번 결과는 논문 핵심 가설을 입증하지 않는다. 대신 약한 generic 후보를 학습하는 비용을 막고,
다음 실험이 실제 sequential-step objective를 직접 최적화해야 함을 강하게 뒷받침한다.

## Artifacts

- plan: `data/manifests/same2k-generic-opportunity-v6.json`
- result: `results/same2k-generic-opportunity-v6/summary.json`
- ignored worker/tokenizer/model/pieces: `artifacts/same2k-generic-opportunity-v6/`
