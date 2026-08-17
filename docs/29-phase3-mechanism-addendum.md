# Phase 3 addendum: scale replication of whitespace mechanism controls

> 상태: **Phase 3 C/W 비교 결과 확인 전 고정**
> 고정 시점: 2026-08-10
> 실행 조건: Gate I 통과 시에만
> 선행: [Phase 2b protocol](./12-phase2b-mechanism-control-protocol.md), [Phase 2b results](./13-phase2b-control-results.md), [novelty/identification audit](./28-novelty-and-identification-audit.md)
> 실행 전 provenance 보강: [mechanism provenance addendum](./39-phase3-mechanism-provenance-addendum.md)

## 1. 목적

Phase 2에서는 W가 generic codepoint grid뿐 아니라 delayed phase와 rate-matched causal hash placebo보다 낮은 BPB를 보였다. Phase 3 primary는 compact signal의 scale/domain 재현부터 검사하기 위해 F/C/W만 먼저 학습하지만, W가 재현될 경우 primary만으로는 scale에서 whitespace association이 원인인지 식별할 수 없다.

이 addendum은 W의 정의를 바꾸지 않고 두 alternative mechanism을 검사한다.

## 2. 실행 stopping rule

다음 둘 중 하나면 새 control을 학습하지 않는다.

- Gate I quality component 실패
- public OOD guard 실패로 Gate I 최종 실패

Gate I가 실패한 뒤 D/P를 W 구조 수정이나 method rescue로 실행하지 않는다. Gate I가 통과하면 first 3 seeds의 D/P를 모두 실행한다. Gate J까지 통과해 confirmation W가 생성됐으면 D/P도 같은 두 seed를 추가한다.

## 3. 공통 고정 조건

Phase 3 primary와 다음을 공유한다.

- HPLT3 train/calibration/test stream과 128M/8M/16M byte limits
- 512-byte arbitrary windows
- exact 86 data patches
- 19,596,096-parameter BLT graph
- optimization schedule와 one-pass training
- seed별 initialization과 shuffled order
- checkpoint/report/loss hash integrity
- test BPB와 per-sequence NLL

Control마다 별도 checkpoint를 처음부터 학습한다. W checkpoint를 fine-tune하지 않는다.

## 4. D — delayed grid

`causal_grid_delayed2`는 whitespace를 보지 않는다.

1. scheduled target은 `ceil(j × 512 / 86)`이다.
2. 마지막 target을 제외하고 `target + 2` 이후 처음 관측된 UTF-8 codepoint boundary에서 patch를 연다.
3. 마지막 target은 C와 같은 unshifted target 이후 처음 codepoint boundary를 사용한다.
4. 모든 결정은 observed prefix만 사용한다.

D는 W에서 early whitespace event가 한 번도 발동하지 않는 phase/lag control이다.

## 5. P — causal rolling-hash placebo

`causal_placebo_grid`는 W와 같은 target window와 minimum patch length를 쓰지만 whitespace 대신 prefix hash event를 사용한다.

- 64-bit FNV-1a
- 각 512-byte row 시작에서 fixed offset basis로 reset
- position `t` event는 byte `t−1`까지 소비한 hash만 사용
- complete UTF-8 codepoint boundary에서만 event 허용
- `target−2`부터 early event 허용
- event가 없으면 `target+2` 이후 첫 codepoint boundary
- minimum patch length 2
- final target은 unshifted C rule
- exact 86 patches

16-bit low hash threshold 하나만 Korean calibration에서 고정한다. Target은 calibration W의 nonfinal early-event trigger fraction이다. 가장 가까운 threshold를 absolute error, 그다음 낮은 threshold 순으로 선택한다. Test loss와 test event rate는 calibration에 쓰지 않는다.

P는 event frequency를 맞추지만 target displacement와 patch-length distribution까지 완전히 같게 만든다고 주장하지 않는다.

## 6. 고정 diagnostics

Split·policy마다 다음을 보고한다.

- exact patch count
- early event / deadline / final count
- event trigger fraction
- target displacement mean/median/p05/p95/min/max
- patch length median/p95/max
- UTF-8 internal boundary rate
- selected event의 whitespace hit rate
- P calibration target/realized fraction, threshold, absolute error

D/P/W의 distribution 차이는 quality 해석 전에 표로 제시한다.

## 7. Mechanism contrasts와 Gate M

Primary contrasts는 두 개다.

1. W − D
2. W − P

Negative가 W 우위다. Initial 3-seed Gate M은 각 contrast가 모두 다음을 만족해야 통과한다.

- mean `<= −0.002 BPB`
- 최소 2/3 seed negative
- exact-rate, initialization/order, loss reconstruction integrity 통과

Final 5-seed 결과가 존재할 때 Gate M을 다음으로 강화한다.

- mean `<= −0.003 BPB`
- 최소 4/5 seed negative
- 각 crossed bootstrap 95% upper `< 0`
- 두 contrast의 one-sided paired-seed Student-$t$ Holm-adjusted p-value `<= 0.05` ([추론 교정](./35-phase3-primary-family-inference-correction.md))

Gate M은 Gate J/K의 숫자를 바꾸거나 Gate L을 단독으로 열지 않는다. 이는 claim attribution gate다.

## 8. 결과 해석

- **Gate J/K/M 통과:** “observed whitespace association beyond matched phase/event alternatives” 허용
- **Gate J/K 통과, M 실패:** efficiency Pareto는 보고할 수 있으나 W를 deterministic relocation heuristic으로만 부름
- **Gate J 통과, K 실패:** boundary-quality observation; efficiency claim 금지
- **Gate I 실패:** D/P 미실행; scale failure branch

어느 경우에도 Korean morphology, morpheme boundary, optimal segmentation을 원인으로 주장하지 않는다.

## 9. 고정 구현과 실행 순서

구현은 `src/jamoflow/phase3_mechanism.py`, `scripts/run_phase3_mechanism.py`, `scripts/summarize_phase3_mechanism.py`에 둔다. Full runner는 primary summary를 먼저 읽고 다음을 강제한다.

1. initial seeds를 요청하면 Gate I `overall_pass == true`
2. confirmation seed를 하나라도 요청하면 Gate J `overall_pass == true`
3. primary와 control의 model/optimization spec, byte limits, stream SHA-256 일치
4. 독립 재구성한 W matrix와 primary W report의 split별 matrix SHA-256 일치
5. D/P cache를 현재 input/boundary/whitespace에서 독립 재구성하고 diagnostics 전체와 일치
6. W/D/P checkpoint state, D/P checkpoint artifact, loss와 seeded order의 독립 재검증

`--quick`은 pipeline 검증만 위해 gate 없이 실행할 수 있지만 manifest에 `evidence_eligible: false`를 기록한다. Quick 결과는 full summary가 승격을 거부한다.

Gate I 통과 뒤 initial controls:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase3_mechanism.py \
  --seeds 1729 2718 31415

PYTHONPATH=src .venv/bin/python scripts/summarize_phase3_mechanism.py \
  --seeds 1729 2718 31415
```

Gate J 통과 뒤 confirmation controls와 final summary:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase3_mechanism.py \
  --seeds 57721 65537

PYTHONPATH=src .venv/bin/python scripts/summarize_phase3_mechanism.py \
  --seeds 1729 2718 31415 57721 65537
```
