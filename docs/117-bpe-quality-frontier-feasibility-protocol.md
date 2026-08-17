# BPE quality-frontier feasibility protocol

> 작성일: 2026-08-14
>
> 상태: 새 training/evaluation step time, memory, loss를 보기 전 고정

## 목적

Random-weight systems frontier에서 vocabulary별 fastest graph 여섯 개가 정해졌다. 작은
vocabulary의 더 큰 model core가 품질상 이점을 가질 수 있으므로, latency 상위 세 개만
남기지 않고 2K/4K/8K/16K/32K/64K를 모두 one-seed quality frontier에 포함한다.

그러나 128M Korean raw bytes를 여섯 모델에 학습하는 실제 시간과 peak memory는 아직 모른다.
이 preflight는 loss나 BPB를 기록하지 않고 실제 MPS forward/backward/optimizer/evaluation
경로만 재서, 결과와 무관한 resource rule로 공통 train-byte budget을 정한다.

## 고정 role

- 2K×8L
- 4K×12L
- 8K×8L
- 16K×8L
- 32K×8L
- 64K×8L

모두 systems-frontier에서 해당 vocabulary의 fastest random graph이며 19.6M parameters의
±0.5% 안이다.

## 데이터와 token inventory

- train: 동일 HPLT3 Korean clean train stream 128,000,000 raw bytes
- calibration: 동일 clean calibration stream 8,000,000 raw bytes
- sequence length: 512 tokens
- tokenizer: systems-frontier에서 두 번 학습·왕복 검증한 exact ByteLevel BPE

Plan을 쓰기 전에 각 raw stream을 document separator를 보존해 tokenize하고, token stream hash,
full sequence count, dropped tail, predicted target raw-byte count, fixed first-batch hash를
봉인한다. Token count는 feasibility projection에는 쓰지만 quality 결과는 아니다.

## 실제 MPS 측정

각 role은 fresh Python subprocess에서 단독으로 실행한다.

- float32 model/optimizer
- effective batch 32
- vocabulary별 train microbatch: 2K 32, 4K 16, 8K 8, 16K 4, 32K 2, 64K 1
- gradient accumulation으로 항상 effective batch 32
- AdamW `lr=3e-4`, betas `(0.9,0.95)`, eps `1e-8`, matrix weight decay 0.1
- gradient clipping 1.0
- warmup 1 effective step, measured 3 effective steps
- vocabulary별 evaluation batch: 64/32/16/8/4/2
- evaluation warmup 1 batch, measured 3 batches

Loss는 finite인지 확인하는 데만 쓰고 값은 report/stdout에 기록하지 않는다. Train timing은
zero-grad, 모든 accumulated forward/backward, gradient clipping, optimizer step, MPS synchronize를
포함한다. Evaluation timing은 tensor transfer, forward/loss, synchronize를 포함한다.

Native resettable MPS peak가 없으므로 각 fresh process에서 baseline, 단계 후 sampled
current/driver allocation, process `ru_maxrss`, release 값을 보존한다. 이는 exact instantaneous
peak가 아니라 conservative isolated-process diagnostic임을 명시한다.

## 결과와 무관한 budget 선택

후보 raw-byte budget 순서는 `128M → 64M → 32M`이다. Role별 full token sequence 수에 budget
비율을 곱하고 effective batch 32로 나눠 optimizer step을 올림한다. Calibration 8M 전체 평가
projection은 모든 budget에 더한다.

가장 큰 다음 조건 통과 budget을 선택한다.

1. 여섯 role train+calibration projected sum ≤ 24 hours
2. 모든 role의 sampled MPS driver/recommended memory ≤ 75%
3. 모든 role의 process RSS/physical memory ≤ 75%
4. 모든 timed value finite positive, start/end AC/default-power/no-thermal-warning

32M도 실패하면 임의 role을 삭제하거나 budget을 더 낮추지 않고 model scale 또는 protocol을
다시 설계한다. 128M이 통과하면 그대로 full one-pass quality frontier를 진행한다.

## Claim 경계

- actual Apple MPS train/eval path의 feasibility evidence
- random initialization에서의 timing/memory projection
- model quality, convergence, BPB, generation efficiency 증거 없음
- 실제 full campaign time을 보장하지 않으며 measured medians의 deterministic projection
- 선택은 loss/BPB를 입력받지 않음
