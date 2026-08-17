# 2K→8K vocabulary-transfer short-CPT probe protocol

> 작성일: 2026-08-14
>
> 상태: 2026-08-14 결과 전 봉인된 historical protocol; 결과와 사후 baseline 정정은 `docs/137` 참조

> 2026-08-15 사후 정정: 이 protocol의 비대칭 역할은 당시 최신 방향을 근사했지만
> Beyond Initialization Loss의 exact strongest setting은 아니다. 현재 구현은 source mean-L2와
> raw-byte-length output weighting을 사용했고, 논문 최선 설정은 target-script subset median-L2와
> decoded Unicode-character-length weighting을 사용한다. 이 차이는 결과를 본 뒤 역할을 바꾸는 데
> 사용하지 않았으며, 별도 strong-baseline closure로 사전 고정해 검증한다.

## 목적

8K dense model은 random-weight generation에서 BPE-2K보다 19.81% 빨랐지만 같은 128M raw-byte
one-pass 학습에서는 document BPB가 `+0.08978` 나빴다. Pure compositional codebook은 이 격차를
줄이지 못하고 추가로 약 `+0.107`--`+0.119 BPB`를 잃었다.

이 단계는 새 Korean method를 시험하지 않는다. 기존 trained dense-2K body와 lexical rows를
8K tokenizer로 옮기는 generic initialization이 large-vocabulary cold start를 충분히 줄이는지
512 update의 짧은 continued-pretraining(CPT) curve로 판단한다. [Beyond Initialization
Loss](https://arxiv.org/abs/2608.03494)가 input uniform+norm / output character-length weighting의
untied 비대칭 초기화를 가장 강한 설정으로 보고했으므로, tied-only 음성 결과로 generic transfer
전체를 기각하지 않도록 tied와 untied 경로를 함께 넣는다. 결과는 후속 full CPT 비용을 정당화하는
development probe이며 publication quality나 actual-latency claim이 아니다.

## 고정 입력

- source model: sealed compositional-quality run의 `dense_v2048` checkpoint
- source tokenizer: exact ByteLevel BPE 2,048
- target tokenizer: exact ByteLevel BPE 8,192
- target Transformer: dense 8K, hidden 384, FFN 1,536, 8 layers
- tied target parameters: 22,026,624
- untied target parameters: 25,172,352; Transformer/attention/logit 연산 geometry는 같고 input
  embedding storage 3,145,728 parameters만 추가된다
- train stream: 기존 clean Korean train stream의 first 128M bytes와 exact 8K token inventory
- evaluation: 기존 8M contiguous calibration stream 전체
- dense-2K anchor: 독립 checkpoint replay의 contiguous BPB `1.4296615772178647`

2K와 8K token byte table은 각각 unique하고, 2K의 모든 2,048 token이 8K table에 exact 포함된다.
나머지 6,144개 8K token은 8K BPE merge genealogy를 재귀적으로 펼치되, exact 2K vocabulary
frontier에 닿으면 멈춘다. 두 tokenizer에서 2K vocabulary ID가 완전히 같고 1,792개 2K merge가
7,936개 8K merge의 exact prefix임을 먼저 검증한다. 이 canonical source-BPE cut은 target piece의
임의 minimum cover가 아니라 source model이 실제로 학습한 merge frontier를 보존한다. 모든 target
piece의 byte reconstruction을 검사하며, 최대 constituent 수는 8, 전체 평균은
`1.884033203125`다.

## 초기화 역할과 architecture control

일곱 역할은 Transformer body를 source checkpoint에서 exact copy한다. 2K와 byte string이 같은
8K input/output row도 모두 exact copy한다. 새 6,144 row만 다르다. 각 composed role은 같은
tied/untied graph의 random control과 비교하므로 비대칭 head의 추가 parameter를 initializer 이득으로
오인하지 않는다.

| role | tying | new input row | new output row |
|---|---|---|---|
| `tied_random_norm` | tied | 고정 random 방향 + source mean-L2 | input과 공유 |
| `tied_uniform_norm` | tied | constituent uniform mean + norm | input과 공유 |
| `tied_byte_weighted_norm` | tied | raw-byte-length weighted mean + norm | input과 공유 |
| `tied_last_subpiece` | tied | 마지막 constituent exact copy | input과 공유 |
| `untied_random_norm` | untied | 독립 random + norm | 독립 random + norm |
| `untied_uniform_in_uniform_out` | untied | uniform mean + norm | uniform mean, output norm 없음 |
| `untied_uniform_in_byte_weighted_out` | untied | uniform mean + norm | byte-length weighted mean, output norm 없음 |

비대칭 설정은 위 최신 선행연구의 strongest internal-composition 방향을 근사하도록 설계됐다.
사후 exact-method 감사에서 norm statistic과 output-length 정의가 다름을 확인했으므로 exact
reproduction이라고 부르지 않는다. Jamo feature는 넣지 않으므로 이번 결과가 positive여도 한국어
특화 기여가 아니다. Untied가 선택되면
추가 parameter와 resident memory를 숨기지 않고 full-CPT 및 actual-inference 단계에서 tied 2K와
별도로 공정하게 계상한다.

## 짧은 CPT 계약

- model/order seeds: 20,260,824 / 20,260,827
- effective batch: 32×512 target tokens
- microbatch: 8, gradient accumulation 4
- checkpoints: step `0`, `32`, `128`, `512`
- body LR: constant `3e-5`
- lexical input/output LR: 26-step warmup 뒤 `3e-4`→`3e-5` cosine; tied는 한 matrix,
  untied는 두 matrix에 같은 schedule
- AdamW betas `(0.9,0.95)`, eps `1e-8`
- matrix weight decay 0.1, vector weight decay 0
- gradient clipping 1.0
- 각 checkpoint에서 full 8M contiguous raw-byte BPB 평가

일곱 역할은 exact target token sequence, order와 batch를 공유한다. Tied/untied는 lexical parameter
sharing만 다르고 Transformer/attention/logit 연산 shape는 같다. 총 512 updates는 약 24% one-pass에
해당하며, full convergence나 target-quality 증거가 아니라 initializer/architecture route ranking과
초기 recovery slope를 보기 위한 상한이다.

## 결과 전에 고정한 판정

각 composed role은 자기 architecture의 random control과 비교한다. Step 512에서 아래 두 조건을
모두 만족한 role만 qualified pool에 넣고, 그중 BPB가 가장 낮은 role을 선택한다. Exact tie는
`tied_uniform`, `tied_byte_weighted`, `tied_last`, `untied_uniform/uniform`,
`untied_uniform/byte_weighted` 순서다.

1. `corresponding_random_control − composed >= 0.010 BPB`
2. `selected − dense2K_anchor <= 0.050 BPB`

첫 조건은 subpiece composition이 같은 tied/untied graph의 단순 pretrained-body transfer보다 실제
초기화 이득을 주었는지 묻는다. 둘째는 512-step 뒤에도 남은 격차가 너무 커서 full 128M CPT를
무의미하게 실행하는 것을 막는다. Qualified role이 없으면 중단한다. Threshold를 결과에 맞춰
낮추거나 초기 checkpoint로 역할을 다시 고르지 않는다.

통과하더라도 즉시 actual timing으로 가지 않는다. 새 sealed full-CPT one-seed 실험은 선택된
transfer, 같은 architecture의 random-row transfer, 기존 direct-8K checkpoint continuation,
dense-2K checkpoint continuation에 동일한 추가 raw-byte budget을 준다. 즉 candidate만 두 번째
학습 budget을 받는 불공정 비교를 하지 않는다. 이 equal-history dense-2K 대비
contiguous/document/bootstrap upper가 모두 `+0.010 BPB` 이내여야 한다. 그 다음에만 generic best와
SCRIPT/Jamo-aware·shuffled control을 비교한다.

## 무결성과 실행 경계

- 구현·문서·source/tokenizer·parent result·source checkpoint artifact/state·결정적으로 계산한 초기
  row/decomposition/state/order를 첫 loss 계산 전에 plan에 봉인한다. 따라서 “초기화를 실행하기 전”이
  아니라 “초기화 정의와 exact state를 알고 있지만 loss는 모르는 시점”의 봉인이다.
- parent dense-8K resource evidence가 tied graph/microbatch의 memory feasibility를 통과했다. Untied는
  activation geometry가 같고 lexical storage만 3,145,728 parameters 늘어난다. Worker는 exact
  parameter count와 실제 MPS 실행을 fail-closed 검증하며, resource failure로 role을 사후 제거하지
  않고 campaign 전체를 중단한다.
- 각 역할은 fresh process와 shared publication MPS lock에서 실행한다.
- 역할별 네 checkpoint와 NLL을 worker 완료 전에 메모리에서 직렬화·검증하고 no-clobber로 공개한다.
- summary는 28개 checkpoint를 strict-load해 full calibration NLL을 독립 재계산하고 저장 배열과
  bitwise 비교한 뒤 step-512 재계산본만으로 판정한다.
- 한 model seed, 알려진 calibration stream, 작은 model의 development evidence다. 일반 수렴·Korean
  우월성·training efficiency·actual inference를 주장하지 않는다.

## 사후 결과

독립 28-checkpoint replay가 모두 bitwise 일치했고, 두 untied composition 역할이 joint gate를
통과했다. 최선 역할은 `untied_uniform_in_byte_weighted_out`, step-512 BPB는 `1.465715`, 대응
random control 대비 advantage는 `0.090992 BPB`, dense-2K anchor gap은 `+0.036053 BPB`였다.

Historical gate는 full CPT를 허용했지만, 결과 후 확인한 exact BIL·In-Place Tokenizer Expansion·
EEVE baseline gap 때문에 full CPT 전에 strong-baseline closure를 추가했다. Threshold나 현재 결과는
바꾸지 않는다. 정량 해석과 수정된 실행 순서는 `docs/137-vocabulary-transfer-probe-result-and-baseline-closure.md`가 지배한다.
