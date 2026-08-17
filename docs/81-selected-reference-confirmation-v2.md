# 선택된 Phase 3 비교군 확인 실험 v3

## 목적

초기 세 seed의 calibration NLL을 체크포인트에서 재계산해 고른 최강 비교군이
`S`, `E`, `EC` 중 하나라면, 그 정책도 확인용 두 seed에서 새로 학습해야 한다.
기존 Gate I 승인은 F/C/W 세 정책만 허가하므로 이 권한을 재사용하지 않는다.

`selection_lock_selected_phase3_reference_confirmation_v3`는 다음 조건을 모두
만족할 때 정확히 하나의 선택된 정책만 허가한다.

- seed는 `57721, 65537` 순서 그대로다.
- 정책은 selection-v2 lock이 고른 `S`, `E`, `EC` 중 정확히 하나다.
- selection lock, selection plan, calibration evidence, sealed final-test seal은 모두
  현재 `HEAD`의 정확한 blob이어야 한다.
- 입력 용도는 `calibration_selection=true`,
  `historical_screening_test=false`, `final_test=false`, `latency=false`로 고정한다.
  기존 Gate I/J/OOD summary를 직접 authorization 입력으로 사용하지 않는다.
- `S`는 보조 모델이 없어야 한다. `E`와 `EC`는 seed별 entropy router와 threshold
  patch cache가 동일한 승인 레코드에 결속돼야 한다.
- Candidate−strongest-reference initial calibration gap이 사전 고정한 `+0.010`
  broad futility screen을 통과해야 한다. 실패하면 이 승인은 발급되지 않으며 C86
  대비 primary within-family confirmation만 계속한다.
- Selection lock 전에 봉인한 confirmation implementation manifest와 현재 HEAD가
  byte-for-byte 같아야 한다. 후속 post-authorization은 selection-lock commit,
  training run commit, calibration evaluator commit과 evidence commit의 순서를 검증한다.

## 실행 계약

선택 lock이 커밋된 뒤 다음 형태로만 실행한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase3.py \
  --seeds 57721 65537 \
  --policies <locked-S-or-E-or-EC> \
  --selection-lock results/phase3-inference-selection-v2/selection-lock.json \
  --device mps
```

`--authorization-summary`, `--force`, `--no-checkpoints`, 다른 selection 경로, 다른
seed/policy 조합은 거부한다. 첫 실행 전에 같은 seed/policy의 report, checkpoint,
loss, router/cache 산출물이 하나라도 있으면 승인 없는 선행 실행으로 간주해
중단한다.

실행 manifest는 학습 전에 승인을 기록한다. main report에는 승인 전체, `device=mps`,
clean-start와 실행 commit을 담은
`selected_phase3_reference_training_evidence_v4`(schema 4)가 들어간다. E/EC의 router
report 및 threshold-cache provenance에도 같은 binding이 들어간다. 재개는 현재
commit에서 생성된 동일 승인·active attempt에서만 허용하며, 부분 산출물이나 `.part`
파일은 자동 덮어쓰기 없이 forensic recovery를 요구한다. 정상 종료 시 clean-end를
확인하고 checkpoint/report/router/cache hash 전체를 담은 fixed-path
`phase3-reference-training-completion.json`을 한 번 publish한다. 이 receipt를 별도
commit하기 전에는 confirmation calibration evaluator가 실행되지 않는다.

## 해석 경계

이 단계는 비교군 체크포인트를 공정하게 추가하는 절차일 뿐 최종 품질이나 속도
우위를 주장하지 않는다. Final unique roles는 candidate, C86 matched-efficiency
baseline, same-rate C control, 그리고 broad screen이 통과한 경우에만 selected
reference다. Broad futility 실패면 v3 authorization 자체가 `null`이며 narrow roles는
계속된다. C86 confirmation seed checkpoint는 새로 prospective 학습한 것이 아니라
plan에 봉인된 historical five-seed summary의 artifact/state/report hash로 검증한다.
모든 두 confirmation seed의 calibration evidence를 재구성·commit하고
`post-confirmation-authorization.json`을 별도 commit한 뒤에만 새 sealed final test를
연다. 실제 추론 효율 주장은 그 품질 gate를 통과한 뒤 별도의 actual-inference
protocol에서만 판정한다.
