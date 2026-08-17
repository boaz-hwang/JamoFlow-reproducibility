# Scalar / Hangul-hybrid actual-runtime preflight protocol

> 작성일: 2026-08-14
>
> 상태: **random-weight MPS timing을 보기 전에 고정할 exploratory protocol**

## 1. 연구 질문

8 MB Korean calibration audit에서 generic Unicode scalar와 Hangul hybrid는 raw byte
main step을 약 58% 줄였지만, train-only ByteLevel BPE는 두 후보보다 약 2.4배 더 짧았다.
따라서 다음 질문은 scalar가 raw byte보다 짧다는 자명한 사실이 아니라 아래 두 가지다.

1. W72의 local/global hierarchy와 작은 conditional head를 유지한 실제 MPS graph가 기존
   byte W72보다 의미 있게 빠른가?
2. 같은 약 19.6M resident parameter에서 그 graph가 훨씬 짧은 BPE16K/32K token
   Transformer와 경쟁 가능한가?

이 단계는 무작위 가중치만 사용한다. 언어모델 품질이나 최종 생성 성능은 답하지 않으며,
비싼 one-seed 학습을 열 가치가 있는 graph인지 선별한다.

## 2. 고정 graph

| role | graph | total parameters |
|---|---|---:|
| byte W72 | 기존 72-patch BLT | 19,596,096 |
| generic scalar | W72 backbone + conditional UTF-8 heads | 19,632,960 |
| Hangul hybrid | W72 backbone + conditional L/V/T heads | 19,609,152 |
| byte BPE32K | tied Llama, d=256, FFN=800, 13 layers, 4 heads | 19,593,984 |
| byte BPE16K | tied Llama, d=320, FFN=1,248, 9 layers, 5 heads | 19,595,200 |

모든 graph는 W72 total parameter의 ±0.25% 안이다. BPE는 vocabulary를 포함한 total
parameter match이므로 body geometry가 서로 다르다. 이 단계의 목적은 같은 parameter budget의
실제 latency frontier이지, tokenizer만 바꾼 동일 body ablation이 아니다.

Scalar 입력은 기존 BLT의 8,192-row resident hash table을 그대로 유지하고 causal
3-unit n-gram을 hash한다. Hybrid는 128 row를 초·중·종성 component에 예약하고 나머지를
unit trigram에 쓴다. 따라서 입력 표현을 위해 큰 scalar embedding table을 추가하지 않는다.

## 3. 출력과 timing 의미

Generic scalar는 첫 UTF-8 byte를 256-way head로 예측하고, 고정 target scalar 길이에 따라
최대 세 64-way continuation head를 순차 실행한다. Hybrid는 raw-byte/Hangul-onset 결합
275-way head와, Hangul route에서 21-way vowel 및 28-way coda head를 실행한다. 각 후속
head는 앞 head의 **device-side argmax** embedding에 의존한다.

무작위 모델은 의미 있는 Korean continuation을 만들 수 없으므로 target의 값은 teacher-force하고
route 종류와 길이만 고정한다. 이를 `controlled_fixed_route_sampling`이라 부른다. 각 head의
linear, argmax, conditional dependency는 timing 안에 있지만 다음은 timing 밖이다.

- tokenizer 또는 scalar transducer 실행
- case 및 patch schedule 사전 계산
- Python에서 target route를 고르는 작업

따라서 이는 실제 graph latency의 강한 feasibility test지만 free-running 생성 결과가 아니다.
유망 후보는 학습 후 strict/canonical free-running timing을 별도로 통과해야 한다.

모든 role에서 runtime/cache 생성, parallel prompt prefill, 첫 target sampling, 나머지
controlled continuation의 incremental cache update와 sampling, 마지막 device sync를 end-to-end
구간에 포함한다. TTFT는 첫 sampling과 sync까지, decode는 그 이후다. 정확히 128 raw bytes에
해당하는 continuation을 처리한다.

## 4. case와 BPE 경계

- source: 기존 HPLT 3.0 Korean 8,000,000-byte calibration stream
- 128-byte prompt + 128-byte controlled continuation
- Hangul-heavy, strict UTF-8, one case per source document
- 기존 deterministic content-hash order의 330개 document case pool
- 두 sealed tokenizer 모두에서 raw offset 128과 256이 token boundary인 case만 유지
- 첫 8개 warmup, 다음 32개 measurement
- text와 token IDs는 tracked artifact에 쓰지 않고 aggregate hash만 봉인

Prompt/continuation 경계를 BPE token이 가로지르는 case를 제외함으로써 BPE가 prompt token을
사후 재분절하는 비현실적 이점을 갖지 않게 한다. 이 필터는 model output이나 latency를 읽지
않는다.

## 5. correctness gate

첫 8개 case에서 모든 role의 full forward와 incremental cache path를 MPS에서 비교한다.

- byte/scalar/hybrid: fixed W72 horizon 512의 causal whitespace-grid schedule
- scalar/hybrid: full unit graph, sequential cache, parallel prefill+incremental decode
- BPE: full canonical token sequence와 parallel prefill+incremental decode
- `atol=1e-4`, `rtol=2e-5`의 normalized worst error `<=1`
- 모든 tensor finite, cache 길이와 observed unit/token 수 exact
- scalar/hybrid encoding 및 BPE token-byte concatenation이 원문 byte와 exact 일치

하나라도 실패하면 timing 결과와 무관하게 중단한다.

## 6. measurement와 통계

- Apple MPS, AC power, 정상 thermal/power mode, machine-global JamoFlow MPS lock
- model seed `20260814`
- 32 independent-document prompts × 3 repetitions
- repetition을 독립 표본으로 세지 않고 prompt별 median으로 먼저 축약
- 고정 Latin-style role rotation으로 order를 균형화
- paired prompt bootstrap 10,000회, seed `20260814`
- primary component: end-to-end; TTFT/decode는 진단

단일 random model, 단일 session, calibration development case이므로 confidence interval은
resource-allocation diagnostic이다. Publication-level 일반화나 confirmatory p-value가 아니다.

## 7. 사전 decision rule

각 scalar 후보를 독립적으로 판정한다. 아래를 모두 만족하면 one-seed matched-budget quality
학습 후보가 된다.

1. 모든 correctness와 reversibility check 통과
2. 모든 graph가 W72 parameter의 ±0.25% 이내
3. byte W72 대비 median E2E reduction `>=10%`
4. byte W72 paired prompt bootstrap 95% lower bound `>0`
5. 32 prompts 중 최소 28개에서 byte W72보다 빠름
6. BPE32K 및 BPE16K 각각에 대해 reduction의 95% lower bound `>=-10%`

마지막 조건은 scalar가 BPE보다 반드시 빠르다는 뜻이 아니라, 이 저비용 graph 단계에서조차
10%를 넘는 명백한 열세이면 품질 학습 비용을 쓰지 않겠다는 기준이다.

Hangul-specific branch는 추가로 hybrid 대 generic reduction의 95% lower bound가 `>=-5%`여야
runtime-competitive로 남긴다. Hybrid가 이를 실패하고 generic만 통과하면 generic 표현 연구만
남고, 한국어 고유 효율 기여 주장은 중단한다. 둘 다 실패하면 scalar/hybrid branch를 학습하지
않는다. Gate를 결과 뒤 완화하지 않는다.

## 8. claim 경계와 다음 단계

통과가 의미하는 것은 “random-weight graph가 학습을 시도할 만큼 빠르다”뿐이다. 다음 단계는
고정 train/calibration budget의 한 seed에서 candidate와 강한 BPE control을 from scratch로
학습하고 BPB noninferiority를 먼저 평가한다. 품질을 통과한 checkpoint만 strict free-running
actual inference로 간다.

이 단계는 다음을 주장하지 않는다.

- matched quality 또는 유용한 Korean generation
- memory improvement
- BPE보다 더 짧은 sequence
- conditional three-hot 또는 multi-byte generation 자체의 novelty
- 다른 hardware, model scale 또는 corpus로의 일반화

## 9. 산출물과 봉인

- plan: `data/manifests/scalar-runtime-preflight-v1.json`
- ignored raw timing: `artifacts/scalar-runtime-preflight-v1/`
- aggregate result: `results/scalar-runtime-preflight-v1/summary.json`

Protocol, implementation, tests, exact cases와 threshold를 먼저 commit한다. Clean plan commit에서
runner를 한 번 실행하고, 별도 고정 summarizer가 ignored timing arrays를 검증한 뒤 aggregate만
tracked result로 낸다.
