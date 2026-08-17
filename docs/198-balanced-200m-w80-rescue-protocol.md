# Balanced 200M W80 quality-rescue protocol

> 작성일: 2026-08-16
>
> 상태: **W80 preflight, training, calibration replay와 actual timing 전에 고정**
>
> 선행 결정: [W72 quality failure and W80 pivot](./197-balanced-200m-quality-failure-result-and-w80-pivot.md)
>
> 봉인 후 결과: [W80 quality and actual-inference result](./199-balanced-200m-w80-quality-and-actual-result.md)

## 1. 단일 가설과 후보

Candidate는 exact 188,639,808-parameter W80 하나다. Baseline은 이미 학습하고 독립 replay한
exact C86 checkpoint다. W72, W82, W84를 후보 pool로 두지 않는다.

- graph/model seed: 기존 balanced-200M screen과 exact 동일
- candidate policy: `causal_whitespace_grid`, patch count `80`
- baseline policy: `causal_codepoint_grid`, patch count `86`
- candidate와 historical baseline의 initial-state hash는 동일
- train/calibration source, train sequence permutation, AdamW와 update 수는 동일
- 새 final/test metric은 사용하지 않음

W80은 새 namespace에서만 생성한다. 기존 C86/W72 plan, checkpoint, report, NLL, verification과
failure analysis는 immutable upstream evidence로 hash-lock한다.

## 2. Candidate resource preflight

W80만 1 warmup + 2 measured effective-batch-32 optimizer updates로 확인한다.

1. actual AdamW state initialization과 finite update
2. Apple MPS recommended memory의 75% cap 아래 완료
3. 전체 7,812-update projection `<=12 h`

Preflight summary를 별도 commit한 뒤에만 candidate training을 시작한다. C86은 동일 geometry와
더 많은 global patches로 이미 같은 batch에서 학습을 완료했으므로 다시 resource preflight하지
않는다.

## 3. Candidate training

- 249,984 sequences, 127,991,808 source bytes
- seed/order: 기존 plan의 exact initial state와 permutation
- microbatch 8, accumulation 4, effective batch 32
- AdamW LR `3e-4` to `3e-5`, cosine, warmup 100 updates
- betas `(0.9, 0.95)`, eps `1e-8`, weight decay `0.1`, clip `1.0`
- exact 7,812 optimizer updates

Candidate report는 checkpoint file/state hash, W80 train/calibration matrix hash, environment,
training history와 per-sequence calibration NLL을 봉인한다.

## 4. Co-primary quality gate

15,625 calibration sequences의 paired effect를

`(NLL_W80 - NLL_C86) / (511 * ln 2)`

로 정의한다. 다음 두 조건을 모두 만족해야 한다.

1. aggregate `BPB(W80) - BPB(C86) <= +0.010`
2. contiguous 64-sequence block 244개를 resampling unit으로 한 10,000-repetition
   percentile bootstrap(seed `20260904`)의 97.5% upper `<= +0.010`

마지막 9 sequences는 point estimate에는 포함하고 block bootstrap에서만 미리 고정한 방식으로
제외한다. Candidate checkpoint에서 전체 calibration forward를 별도 verifier가 다시 수행해 저장
float32 NLL과 bitwise equality를 확인해야 한다. Mean/upper/replay 중 하나라도 실패하면 actual
timing은 실행하지 않는다.

## 5. Actual inference gate

Quality와 독립 replay가 통과한 경우에만 exact trained W80와 immutable trained C86을 비교한다.

- modes: 128-byte prompt + 128-byte controlled replay, strict-valid UTF-8 greedy free-running
- cases: 기존 scale-schedule의 outcome-independent 4 warmup + 16 measured documents
- sessions: 5 fresh processes
- repetitions: prompt/role/mode당 3, cell median으로 collapse
- timing scope: runtime construction, structural selector, parallel prefill, cached incremental
  decode, argmax/UTF-8 DFA/stop, final MPS synchronization을 포함
- primary unit: session x prompt; repetitions은 독립 표본으로 세지 않음
- interval: session과 prompt를 같은 draw에서 독립 resample하는 10,000-repetition crossed
  bootstrap, seed `20260905`

각 mode는 다음을 모두 만족해야 research-value actual success다.

1. candidate/reference E2E point reduction이 compact matched-quality point보다 큼
   - controlled: `> 0.026283464474602614`
   - free-running: `> 0.025305234146383637`
2. crossed-bootstrap 95% lower `> 0`
3. 16 prompts 중 최소 15 prompts에서 candidate direction
4. 5 sessions 모두 aggregate direction이 양수
5. incremental parallel/sequential logits, argmax, boundary/cache trace와 strict UTF-8 output 검증 통과

별도의 더 강한 **scale-amplification support**는 각 mode의 bootstrap lower까지 위 compact point를
넘을 때만 true다. Point만 넘으면 “이 screen의 관측 개선이 compact reference보다 컸다”고만
쓴다.

## 6. 중단 규칙

- W80 quality fail: actual timing 금지, W82/W84 자동 탐색 금지, density-rescue path 종료
- quality pass + actual positive지만 compact point 이하: 실제 효율 개선은 기록하되 scale 증가
  근거로 사용하지 않음
- point가 compact보다 크지만 lower가 compact 이하면: descriptive larger point만 허용
- 두 mode 모두 strong threshold까지 통과: larger-model density-adjusted replication을 다음
  multiseed/data-scaling 단계로 확장

## 7. 허용되는 가장 강한 문장

Quality와 actual primary가 통과한 경우:

> In a one-seed, severely undertrained 188.6M Korean byte-LM screen, a
> density-relaxed W80 policy retained calibration quality against the exact
> C86 baseline and produced larger observed controlled and free-running
> end-to-end reductions than the prior compact reference.

Strong scale-amplification support까지 통과하지 않으면 `statistically larger` 또는 `model scale
caused the increase`라고 쓰지 않는다.
