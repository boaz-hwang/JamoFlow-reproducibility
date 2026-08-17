# Scalar representation and BPE opportunity protocol

> 작성일: 2026-08-14
>
> 상태: **BPE token count를 보기 전에 고정하는 exploratory protocol**

## 1. 왜 방향을 바꾸는가

W72는 quality-matched actual inference에서 일관되게 빨랐지만 controlled 2.628%,
free-running 2.531%에 그쳐 고정 10% gate를 실패했다. Component profile은 줄어든 네
boundary update가 약 10.1ms를 설명하고, 127개의 순차 local byte step은 그대로 남는다는
것을 보였다. Exact speculation은 9.983%에 그쳤고, static thin local graph는 22.8--24.3%
빨랐지만 +0.0956 BPB, frozen conditional-local skip은 최소 +0.1988 BPB로 품질을 잃었다.

이 결과는 다음 제약을 준다.

- byte 정보를 버리면 안 된다.
- global boundary만 더 줄여서는 목표에 부족하다.
- 실제로 줄여야 하는 것은 정보량이 아니라 **순차 local state update 횟수**다.

따라서 다음 저비용 질문은 UTF-8의 여러 encoding byte를 하나의 가역적인 단위로 묶을 때
한국어 stream에서 BPE를 포함한 실현 가능한 cost frontier가 남는가다.

## 2. 재현할 두 표현

### Generic conditional UTF-8 scalar control

- strict Unicode scalar 하나를 main autoregressive step 하나로 처리한다.
- 첫 byte 256-way head 뒤 필요한 continuation을 최대 세 개의 64-way conditional
  micro-head로 factorize한다.
- invalid 또는 truncated byte는 raw-byte fallback으로 보존한다.
- resident output rows의 단순 합은 `256 + 3×64 = 448`이다.

### Hangul scalar / otherwise raw-byte hybrid

- canonical precomposed Hangul `U+AC00..U+D7A3`만 main step 하나로 묶는다.
- 첫 head는 raw byte 256개와 onset 19개를 함께 선택하고, Hangul이면 vowel 21개와
  coda 28개를 조건부로 선택한다.
- resident output rows의 단순 합은 `(256+19) + 21 + 28 = 324`이다.
- 한글 이외의 모든 것은 계속 raw byte이므로 임의 byte fallback을 유지한다.
- precomposed Hangul byte를 raw-byte route로도 만들 수 있게 두지 않는다. 추후 구현은
  canonical transducer로 text당 encoding 하나만 허용해야 한다.

두 표현 모두 component prediction이 공짜라는 뜻이 아니다. `한 main state update` 안의
conditional micro-head는 추가 kernel, synchronization, cache 및 rejection 비용을 낳을 수
있다. 이 protocol의 FLOP 식은 그 실제 latency를 증명하지 않는다.

## 3. 선행연구와 novelty 경계

[Cognetta et al. (EACL 2023)](https://aclanthology.org/2023.eacl-main.172/)은 이미
한글 음절을 한 timestep으로 유지하면서 초성·중성·종성을 조건부 three-hot decoder로
예측했다. 그러므로 다음은 JamoFlow의 새 기여가 아니다.

- 한글 음절을 `L/V/T`로 분해하는 것
- independent three-head 대신 conditional chain을 쓰는 것
- 음절 vocabulary보다 embedding/output parameter를 줄이는 것

남을 수 있는 기여는 Korean decoder-only LM에서 raw-byte fallback을 가진 hybrid를 BLT의
local sequential bottleneck에 연결하고, generic scalar와 BPE를 상대로 matched-quality
actual wall time을 개선하는지 보이는 것이다. [MYTE (ACL 2024)](https://aclanthology.org/2024.acl-long.804/)와
[From Bytes to Subwords (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.530/)도
UTF-8 자체가 비라틴 script에 비효율적이며 alternative encoding이 유효한 연구 축임을
보인다. 반면 [Fast BLT (2026)](https://arxiv.org/abs/2605.08044)는 generic multi-byte
generation을 이미 다루므로 “여러 byte를 한 번에 낸다”도 독립 novelty가 아니다.

## 4. 입력과 누출 경계

- BPE 학습: 고정 HPLT 3.0 Korean `train` split 5,791 documents만 사용한다.
- opportunity 평가: 기존 8,000,000-byte calibration stream만 사용한다.
- historical test, new final test, downstream, checkpoint, model loss, latency artifact를 읽지
  않는다.
- source text에 normalizer를 적용하지 않는다. 한 train document가 non-NFC라는 기존
  integrity 사실도 그대로 보존한다.
- tokenizer JSON은 corpus substring을 포함할 수 있으므로 ignored `artifacts/`에만 둔다.
- tracked result에는 aggregate count, hash, entropy 및 cost estimate만 기록한다.

Calibration은 이미 여러 개발 결정에 사용됐다. 이 audit은 confirmatory 또는 blind가
아니다. 또한 아래 값은 protocol 작성 전에 이미 model-free exploration 또는 기존 tracked
result로 알려져 있었다.

- 8MB stream: 7,999,999 complete bytes, 3,330,976 scalars
- generic scalar step reduction 약 58.36%
- Hangul-only grouping reduction 약 57.59%
- scalar savings 중 precomposed Hangul 설명 비율 약 98.68%
- train unique scalar 약 7,006
- calibration에서 train-vocabulary unseen occurrence 약 0.0045%
- W72 대비 scalar local-path dense-matmul 절감의 초기 근사 약 36.5%

따라서 이 수치들에 대한 gate는 새 발견을 가장한 가설 검정이 아니라 implementation
sanity/resource-allocation check다. 아직 보지 않은 주요 산출물은 exact train-only
ByteLevel-BPE 16K/32K token count다.

## 5. BPE control

두 tokenizer는 `tokenizers==0.22.2`에서 다음 설정으로 학습한다.

- normalizer 없음
- ByteLevel pre-tokenizer, `add_prefix_space=False`, `use_regex=True`
- full 256-byte initial alphabet
- no special/added token, no dropout, no unknown token
- `min_frequency=2`
- exact fixed train document order와 join separator
- vocabulary 16,000 및 32,000

각 tokenizer를 독립적으로 두 번 학습해 compact JSON bytes가 같은지 확인한다. Calibration
complete-scalar prefix에서 decode roundtrip과 raw token-byte concatenation을 모두 exact
검증한다. Token count가 scalar보다 작더라도 scalar branch를 자동 통과시키거나 중단하지
않는다. BPE는 다른 graph와 큰 head를 가지므로 다음 actual construction에서 반드시 같은
parameter/hardware 조건으로 비교한다.

## 6. 측정값

1. raw byte, generic scalar, Hangul-hybrid, BPE16K, BPE32K sequential units
2. UTF-8 length 및 Hangul composition counts
3. train scalar vocabulary와 calibration unseen occurrence
4. train/calibration의 `H(L)`, `H(V|L)`, `H(T|L,V)`, joint entropy 및 independent-head
   total correlation
5. local width 192에서 각 output parameterization의 row/한 projection parameter 수
6. 같은 512 raw-byte horizon과 W72 72 data patches를 가정한 dense-matmul opportunity
   estimate

FLOP estimate는 local sequence length만 empirical unit count로 바꾸고 global patch count는
72로 유지한다. UTF-8 parsing, conditional dispatch, kernel launch, memory movement, cache,
softmax, capacity reallocation 및 quality recovery는 제외한다. 따라서 analytical lower-cost
candidate를 찾는 용도이며 actual speed 결과가 아니다.

## 7. 사전 decision rule

아래를 모두 만족해야 다음 **random-weight construction/timing feasibility**만 연다.

1. 모든 표현과 두 BPE가 exact reversible
2. generic scalar sequential-step reduction `>=50%`
3. Hangul hybrid sequential-step reduction `>=50%`
4. Hangul hybrid counted dense-matmul reduction versus W72 `>=20%`
5. train scalar vocabulary `<=8,192`
6. calibration unseen scalar occurrence `<=0.1%`

통과는 학습, 품질, 실제 속도 또는 한국어 고유 우위를 승인하지 않는다. 다음 단계는
parameter-matched graph를 실제로 만들고 random weights에서 다음 네 경로의 runtime 구조만
검증한다.

- byte W72
- generic conditional UTF-8 scalar
- Hangul scalar / raw-byte hybrid
- train-only BPE16K/32K control

그 단계에서 hybrid가 generic scalar보다 의미 있게 빠르지 않으면 한국어 고유
efficiency claim을 중단한다. Random-weight 구조가 유망해도 별도 calibration에서 from-scratch
matched-budget quality를 통과해야 actual inference 평가를 연다.

## 8. 산출물

- plan: `data/manifests/scalar-representation-opportunity-v1.json`
- implementation: `scripts/scalar_representation_core.py`
- runner: `scripts/analyze_scalar_representation_opportunity.py`
- ignored tokenizers: `artifacts/scalar-representation-opportunity-v1/`
- aggregate: `results/scalar-representation-opportunity-v1/summary.json`

Protocol, code, tests, manifest를 먼저 commit하고 clean HEAD에서 실행한다. 결과에 따라 다음
단계를 수정할 수 있지만, 이미 실패한 skip/static 후보의 threshold를 되돌려 살리지 않는다.
