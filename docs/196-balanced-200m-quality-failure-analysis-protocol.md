# Balanced-200M quality-failure analysis protocol

## 1. 왜 이 분석을 하는가

사전 봉인한 188,639,808-parameter one-seed screen에서 W72는 C86보다
calibration BPB가 `+0.024200478` 높았다. 고정 noninferiority margin `+0.010`을
넘었으므로 actual incremental timing은 열지 않는다. 이 결과는 model scale만
키우면 compact W72의 약 2.5% actual speedup이 자동으로 커진다는 가설을 지지하지
않는다.

aggregate quality 결과는 이미 알려져 있다. 아래 분석은 **post-outcome exploratory
diagnosis**이며 confirmatory evidence가 아니다. 다만 per-sequence NLL pattern과
구조적 feature는 이 protocol을 commit하기 전에는 열지 않는다.

## 2. 고정 입력

- sealed plan: `data/manifests/balanced-200m-trained-screen-v1.json`
- committed result: `results/balanced-200m-trained-screen-v1/training-summary.json`
- exact C86/W72 checkpoint, report, calibration NLL identities in that result
- exact 8,000,000-byte calibration stream and sealed patch matrices
- independent full-checkpoint replay receipt produced by the originally sealed verifier

Historical test, sealed final-test loss, actual latency, downstream task metric은 입력하지
않는다.

## 3. 사전 고정 분석

각 512-byte sequence의 paired effect를

`(NLL_W72 - NLL_C86) / (511 * ln 2)` BPB

로 정의한다.

1. paired effect의 mean, median, standard deviation, 5/95 percentile, positive-rate
2. contiguous 64-sequence block 244개(마지막 9 sequence 제외)를 독립 resampling
   unit으로 한 10,000-repetition percentile interval; seed `20260903`
3. positive excess의 상위 1/5/10% sequence 집중도
4. 다음 outcome-free structural feature와 paired effect의 Spearman correlation 및
   equal-count quintile summary
   - whitespace event count
   - Hangul syllable count after per-window strict-prefix-safe UTF-8 ignore decoding
   - W72 maximum patch length
   - W72 whitespace-trigger count
   - W72/C86 internal-boundary overlap count
5. C86 및 W72/W76/W78/W80/W82/W84의 patch-length, trigger, displacement profile
6. training/calibration throughput, resident high-water memory, raw-byte/parameter 및
   global-patch-token/parameter accounting

Correlation과 quintile 결과는 원인 증명이 아니다. 문서 경계, serial correlation,
한 seed, 0.68 raw byte/parameter의 severe undertraining을 명시한다.

## 4. 다음 실험을 고르는 규칙

W72의 관측 delta와 C86의 zero delta 사이를 removed-patch count에 대해 선형으로
보간한 값은 **design heuristic**일 뿐 quality estimate가 아니다. 이 heuristic과
구조 분석을 이용해 후속 candidate를 정하되, candidate training 전에 별도 plan을
commit한다.

우선 고려 대상은 W80이다. W72가 제거한 14 patches 중 6개만 제거하므로 선형
heuristic은 약 `+0.01037 BPB`로 gate 경계에 놓이고, 188.6M random-weight schedule
sensitivity의 단순 비례치는 약 3.1%이다. 따라서 W80은 “품질을 유지하면서 compact
2.5%보다 큰 actual gain이 가능한가”를 가장 직접적으로 판별하는 단일 next screen이다.
W80이 실패하면 W82/W84의 예상 headroom은 기존 compact result보다 작을 가능성이
높으므로, 자동으로 더 조밀한 모델을 계속 학습하지 않는다.

## 5. 주장 경계

- W72 quality failure는 independent replay 뒤 확정할 수 있다.
- calibration evaluator throughput 차이는 actual incremental generation speed가 아니다.
- random-weight large-graph timing은 trained-model quality를 증명하지 않는다.
- 이 분석만으로 failure가 patch density 때문인지, low data/parameter ratio 때문인지
  식별할 수 없다.
- 후속 W80도 one seed이면 mechanism screen이며 publication-scale generalization이 아니다.

