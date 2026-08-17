# Phase 3 primary evidence provenance addendum

> 작성일: 2026-08-11
> 상태: **initial 3-seed primary 분석 단위 완성 전 고정**
> 수정 시점의 정보: seeds 1,729와 2,718의 F/C/W 완료, seed 31,415 실행 중, initial OOD 결과 0개
> 영향: policy·model·data split·loss·contrast·통계·Gate I/J/K 불변; source와 evidence reconstruction 강화

## 1. 정적 감사에서 확인한 문제

[Patch-cache provenance addendum](./37-phase3-cache-provenance-addendum.md)은 final summarizer가 current stream에서 F/C/W matrix를 독립 재구성한다고 적었다. 그러나 결과 파일을 만들기 전 실제 코드를 다시 대조하니 그 구현이 빠져 있었다. 기존 summarizer는 다음까지만 했다.

1. HPLT3 test stream hash 하나를 manifest와 비교했다.
2. Report가 주장하는 matrix hash와 patch count를 읽었다.
3. Test per-sequence NLL에서 BPB를 다시 계산했다.
4. Checkpoint state-dict hash와 report의 hash를 비교했다.

이것만으로는 source→patch matrix→checkpoint→loss의 전체 사슬을 독립적으로 검증하지 못한다. 추가로 다음 공백이 있었다.

- train/calibration stream을 current processed source에서 다시 만들지 않았다.
- processed `ko.jsonl` 전체 artifact와 `integrity.json`을 primary manifest가 고정하지 않았다.
- Checkpoint와 training-report의 serialized artifact hash를 tracked summary에 남기지 않았다.
- Existing report/checkpoint/loss가 있으면 current matrix와 일치하는지 검사하지 않고 완료 처리했다.
- Cache loader는 provenance metadata가 맞아도 cache content와 diagnostics의 exact 일치를 다시 확인하지 않았다.
- Append-only manifest가 requested seed/policy pair의 invocation을 실제로 포함하는지 final summary가 강제하지 않았다.

이 문제는 3-seed contrast를 계산하거나 열람하기 전에 코드 감사로 발견했다. 일부 개별 seed report는 이미 존재했지만, 수정 과정에서 새 threshold를 고르거나 policy/metric/gate를 바꾸지 않았다.

## 2. Processed source와 legacy manifest upgrade

새 invocation부터 primary manifest invariant에 다음을 추가한다.

- `ko.jsonl` filename, byte size, SHA-256
- `integrity.json` filename, byte size, SHA-256

Runner는 `integrity.json`의 dataset ID, output size와 output SHA-256이 실제 `ko.jsonl`과 일치해야 시작한다. 초기 F/C/W process는 이 필드가 추가되기 전 commit에서 시작했으므로 기존 manifest에는 없다. 다음 invocation에서 legacy manifest를 upgrade하되, 이미 고정된 train/calibration/test stream metadata와 selected-stream hash가 current reconstruction과 모두 같은 경우에만 evidence로 사용한다.

Full processed file의 후행 비사용 record가 달라졌더라도 기존 결과에 직접 영향을 주는 것은 선택된 세 stream이다. 그럼에도 재현 가능한 공개 artifact를 하나로 고정하기 위해 upgrade 시점의 full file과 processed integrity hash를 함께 남긴다. 세 selected stream 중 하나라도 다르면 upgrade가 아니라 실패다.

## 3. Final summarizer의 독립 재구성

Summarizer는 다음 경로를 current filesystem에서 다시 계산한다.

```text
processed ko.jsonl + integrity.json
  -> train / calibration / test byte streams
  -> input, UTF-8 boundary, whitespace, SpaceByte event matrices
  -> F / C / W / S structural patch matrices
  -> seed별 E / EC cache lineage와 patch diagnostics
  -> training report + checkpoint + test per-sequence NLL
  -> absolute BPB + paired contrasts + gates
```

### 3.1 Structural policies

F/C/W/S는 세 split의 모든 row에서 다시 만든다. 다음이 모두 exact match여야 한다.

- cache NPZ의 exact key set과 `uint16` dtype
- reconstructed matrix 전체 배열
- matrix SHA-256
- variable patch diagnostics
- source boundary/whitespace/SpaceByte event hash에 묶인 cache provenance
- 각 seed report가 기록한 split별 matrix hash와 diagnostics

따라서 F/C/W exact-rate 주장은 report의 self-assertion이 아니라 current source에서 재구성된 matrix에 근거한다.

### 3.2 Learned E/EC policies

E/EC는 다음 lineage를 직접 검증한다.

- deterministic router initialization과 training order
- router report의 model/optimization spec과 parameter count
- actual router checkpoint state-dict 및 serialized artifact hash
- current split input/boundary matrix hash
- threshold-cache provenance의 router-state hash
- cache의 exact keys, dtype, coverage, matrix hash와 full diagnostics
- E/EC main report의 split별 matrix hash와 diagnostics

Quality summarizer는 모든 byte의 router logits를 다시 추론하지 않는다. 즉 E/EC 검증은 current router·source에 묶인 cache lineage와 matrix 내용의 완전 검산이며, 독립적인 full router re-inference는 아니다. 후속 cost benchmark가 사전등록된 timing subset에서 online selector와 cache의 일치를 별도로 확인한다. 이 범위를 summary에 명시한다.

## 4. Main checkpoint와 loss lineage

각 seed/policy report에 대해 다음을 다시 확인한다.

1. exact report field set과 seed/policy identity
2. 19,596,096 parameter model/optimization spec
3. deterministic initialization hash와 shuffled training-order hash
4. current matrix hash와 full patch diagnostics
5. checkpoint state-dict hash와 report의 trained-state hash
6. checkpoint artifact SHA-256과 training-report artifact SHA-256
7. test loss NPZ의 exact key, `float32` dtype, shape, 유한성·비음수성
8. per-sequence NLL 합에서 `nll_nats`, predicted bytes와 absolute BPB 재구성

Tracked summary에는 raw text, checkpoint, cache와 per-sequence loss를 넣지 않고 각 ignored artifact의 SHA-256과 aggregate만 남긴다.

## 5. Safe resume 강화

향후 S/E/EC와 confirmation run에서 기존 policy result를 건너뛰기 전에 runner도 다음을 검증한다.

- current deterministic initialization/order
- current split별 matrix hash와 diagnostics
- checkpoint state hash
- test-loss shape/dtype와 reconstructed BPB

불일치하면 stale result로 중단하고 `--force`를 요구한다. Structural/threshold cache를 재사용할 때도 provenance뿐 아니라 cache content에서 다시 계산한 diagnostics와 stored diagnostics가 같아야 한다.

Active 초기 F/C/W process는 시작된 Python code가 바뀌지 않으므로 학습 자체에는 영향이 없다. 완료 뒤 current runner로 provenance upgrade와 structural cache 재구성을 한 번 수행하고 나서만 final summarizer를 실행한다.

## 6. 분석과 주장에 미치는 영향

이 보강은 어떤 quality 결과도 살리거나 탈락시키지 않는다. Gate I은 여전히 initial 3 seeds와 OOD guard가 모두 있어야 판정하며, 실패 시 S/E/EC·confirmation·scale-up을 실행하지 않는 원칙도 같다.

강화 뒤에도 남는 한계는 다음과 같다.

1. Serialized artifact hash는 accidental mix-up과 사후 변경을 탐지하지만 외부 공증은 아니다.
2. Test per-sequence loss를 checkpoint에서 다시 forward하지는 않는다.
3. E/EC full router logits를 final quality summary에서 재계산하지 않는다.
4. 19.6M/MPS evidence는 publication-scale 또는 CUDA serving evidence가 아니다.

이 한계를 포함해도 primary quality endpoint와 F/C/W causal matrix 정의는 source에서 독립 재구성되므로, Gate I을 실행하기 위한 최소 evidence chain은 충족한다.

## 7. 회귀 검증

추가·강화한 검사는 다음과 같다.

- Legacy manifest의 source artifact field upgrade와 이후 exact invariant
- Processed source와 integrity metadata의 상호 hash 검증
- 요청한 모든 seed/policy invocation coverage
- Structural/threshold cache provenance와 content mismatch 거부
- Completed policy의 initialization/order/matrix/checkpoint/loss stale detection
- 실제 checkpoint tensor 변경 감지와 OOD-primary checkpoint binding 유지

전체 test suite **167개**가 통과했다.
