# Balanced 200M trained scale screen protocol

> 작성일: 2026-08-16
>
> 상태: **첫 batch-8 optimizer preflight와 model training 전에 고정**
>
> 선행 결과: [Global-heavy failure and pivot](./194-global-heavy-result-and-trained-scale-pivot.md)

## 1. 연구 질문

Random-weight balanced family에서 W72의 C86 대비 controlled E2E 감소는 19.6M-trained setting의
약 2.5%보다 model scale과 함께 증가했다. 특히 exact 188,639,808-parameter graph에서는
7.218%였다. 이제 같은 200M geometry를 실제 Korean bytes로 학습했을 때 다음을 묻는다.

1. W72가 C86 calibration BPB를 `+0.010` 이내로 보존하는가?
2. 서로 다른 trained weights가 된 뒤에도 actual controlled inference 개선이 증가하는가?

이것은 한 seed mechanism-scale screen이며 충분히 학습된 200M frontier claim이 아니다.

## 2. 고정 model/data

- exact parameters: 188,639,808
- spec/state: sealed post-100M 200M graph와 같은 initialization seed `20260816`
- roles: C86 causal codepoint grid / W72 causal whitespace grid
- source: canonical HPLT Korean Phase3 train split
- available train stream: 128,000,000 bytes = 250,000 sequences
- fixed usable training set: shuffled order의 첫 249,984 sequences
- exact used bytes: 127,991,808
- dropped tail: 16 sequences, 결과와 무관하게 batch divisibility 때문에 고정
- calibration: 기존 8,004,309 available bytes 중 exact 8,000,000-byte prefix
- historical test/final loss: training/selection gate에 사용하지 않음

두 model은 initial weights, source sequences와 shuffled order가 같고 patch matrix만 다르다.

## 3. 고정 optimizer와 batch

- float32 Apple MPS
- microbatch 8 sequences
- gradient accumulation 4
- effective batch 32 sequences = 16,384 source bytes/update
- total optimizer updates: 7,812
- AdamW: LR `3e-4`, minimum LR `3e-5`, betas `(0.9,0.95)`, eps `1e-8`,
  weight decay `0.1`
- cosine schedule, 100 warmup updates
- gradient clip `1.0`

Batch 8은 timing 결과가 아니라 batch-1 resource result의 낮은 200M memory fraction과 기존
batch-32 optimization contract를 근거로 미리 정했다. 첫 단계에서 batch-8 actual update를 두
role 모두 fresh subprocess로 측정한다.

## 4. Preflight gate

각 role은 1 warmup + 2 measured effective-batch-32 optimizer updates를 수행한다. 다음을 모두
요구한다.

1. 실제 AdamW state 초기화, finite forward/backward/update
2. per-process MPS 75% cap 아래 완료
3. 127,991,808-byte projection role당 `<=12 h`
4. pair projection `<=24 h`

둘 중 하나라도 실패하면 full training을 시작하지 않는다. 성공 receipt를 별도 tracked summary로
commit한 뒤에만 training implementation을 실행한다.

Preflight는 학습 corpus의 canonical 첫 96개 sequence만 사용한다. 이는 세 effective update에
필요한 정확한 개수이며, 학습 순서나 quality metric을 보지 않는 resource-only workload다.

## 5. Quality screen

Training이 끝나면 고정 calibration stream에서 per-sequence NLL을 저장하고 checkpoint에서 다시
재구성한다.

- primary quality: `BPB(W72) - BPB(C86) <= +0.010`
- finite loss/NLL, exact input/patch/checkpoint identities 필수
- one seed이므로 confidence interval이나 multiseed claim을 하지 않음
- quality 실패 시 actual timing으로 favorable systems claim을 만들지 않음

Quality가 통과하면 exact 두 trained checkpoint를 기존 controlled cases에서 3 fresh sessions로
timing한다. Random-weight 200M의 7.218%는 descriptive expectation일 뿐 trained gate를 정하지
않는다. Trained systems 성공 기준은 point `>0`, 95% lower `>0`, 16 prompt 중 15개 이상 W72
방향이다. 10%는 요구하지 않는다.

Timing은 기존 scale-schedule plan의 4 warmup + 16 measured independent-document prompts,
128-byte prompt + 128-byte controlled continuation, repetition 3회, fresh session 3개를 그대로
사용한다. repetition은 독립 표본으로 세지 않고 session×prompt cell median으로 collapse한 뒤
두 축 crossed bootstrap 10,000회(seed `20260902`)를 사용한다. C86/W72는 같은 초기화에서
출발했지만 각각 학습된 서로 다른 checkpoint이므로 random-weight 측정처럼 한 model object의
schedule만 교체했다고 주장하지 않는다.

## 6. Claim boundary와 확장

성공 시 허용되는 가장 강한 문장:

> In a one-seed 188.6M Korean byte-LM screen, W72 retained calibration quality
> within 0.010 BPB and reduced actual controlled cached-inference latency versus
> the same-initialization C86 model.

128M bytes는 `0.68 bytes/parameter`에 불과하다. 성공 후에도 충분히 학습된 LLM이라고 부르지
않는다. 다음 확장은 model outcome과 무관하게 새 disjoint Korean train stream을 먼저 봉인한 뒤
512M/1.024B continuation checkpoint를 만드는 별도 protocol이어야 한다.
