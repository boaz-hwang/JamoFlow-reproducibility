# Phase 3 confirmation authorization lock

> 작성일: 2026-08-12  
> 상태: **confirmation seed 학습 전 고정**  
> 수치 영향: 없음; execution/provenance fail-closed 보강

## 1. 발견한 공백

기존 `run_phase3.py`는 seed가 사전등록 집합에 속하는지와 model/data design은
검사했지만, confirmation seeds 57,721 / 65,537를 실행한 이유가 corrected Gate I
summary에 직접 결속되지 않았다. Manifest에는 invocation의 seed, policy와 commit은
남았지만 다음과 같은 잘못된 실행도 schema상 구분하기 어려웠다.

- Gate I 실패 또는 아직 미평가 상태에서 confirmation 실행
- historical pre-document-cluster summary로 confirmation 실행
- 현재 pre-confirmation run manifest와 다른 summary를 authorization으로 연결
- primary training과 OOD confirmation에 서로 다른 authorization summary 사용

현재 Gate I 결과가 pass라는 사실은 실제 수치를 올바르게 만들지만, 논문의 실행
DAG를 재현 가능하게 증명하기에는 부족하다. 이 공백은 confirmation 결과가 생기기
전에 수정했다.

## 2. 봉인 계약

`src/jamoflow/phase3_confirmation.py`는 authorization summary에 다음을 요구한다.

1. seed가 정확히 1,729 / 2,718 / 31,415
2. policy가 정확히 F/C/W
3. corrected Gate I `overall_pass == true`
4. composite integrity `all_integrity_checks_pass == true`
5. summary artifact SHA-256가 유효함
6. summary가 가리키는 source manifest SHA-256가 유효함
7. primary confirmation 시작 시 그 source hash가 현재 pre-confirmation
   `runs/phase3/manifest.json`의 실제 hash와 일치함

Primary confirmation request 자체도 두 seed와 F/C/W를 정확한 사전등록 순서로 한
번에 요청해야 한다. Runner는 다음 identity를 invocation에 저장한다.

- authorization kind와 gate/status
- corrected summary artifact SHA-256
- pre-confirmation source-manifest SHA-256
- summary commit, seeds와 policies

첫 invocation 뒤 중단된 작업은 current manifest hash가 이미 달라진다. 재개는 새
summary를 허용하는 방식이 아니라, current manifest에 같은 seed/policy request와
exact authorization record가 이미 존재할 때만 허용한다. 다른 seed subset, policy
subset 또는 summary hash로의 재개는 거부한다.

Runner는 긴 data/cache 재구성 뒤의 HEAD가 아니라 **process 시작 시점**의 clean
Git commit을 고정한다. 시작 시 tracked/untracked 변경이 있으면 실행하지 않으며,
종료 시 HEAD가 달라졌으면 evidence completion을 거부한다. 따라서 학습 중에는
문서·코드 commit도 만들지 않는다.

Five-seed primary summarizer는 같은 summary를 다시 읽고 모든 confirmation
seed/policy invocation이 그 exact record와 결속됐는지 확인한다.

OOD confirmation도 같은 record를 요구한다. OOD runner는 먼저 primary training
manifest의 confirmation invocation을 재검증한 뒤 같은 record를 OOD invocation에
복제한다. Five-seed OOD summarizer는 training/OOD 두 manifest를 모두 확인하고,
최종 primary summarizer는 primary와 OOD summary의 authorization identity가 완전히
같은지 검사한다.

추가로 OOD runner도 process 진입 직후 clean Git commit을 고정하고 종료 시 HEAD와
worktree가 그대로 clean인지 확인한다. 각 confirmation OOD report에는 그 commit,
clean-start attestation과 exact authorization record를 직접 기록한다. 따라서 같은
checkpoint/stream scalar만 가진 과거 report를 새 authorization invocation 아래에서
재사용할 수 없다.

## 3. 고정 실행 명령

Primary confirmation:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase3.py \
  --seeds 57721 65537 \
  --policies fixed_byte_6 causal_codepoint_grid causal_whitespace_grid \
  --authorization-summary results/phase3-primary-clustered/summary.json
```

Leipzig OOD confirmation:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase3_ood.py \
  --seeds 57721 65537 \
  --authorization-summary results/phase3-primary-clustered/summary.json
```

Five-seed OOD와 primary summary:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_phase3_ood.py \
  --seeds 1729 2718 31415 57721 65537 \
  --confirmation-authorization-summary \
    results/phase3-primary-clustered/summary.json \
  --output results/phase3-ood-confirmation/summary.json

PYTHONPATH=src .venv/bin/python scripts/summarize_phase3.py \
  --seeds 1729 2718 31415 57721 65537 \
  --policies fixed_byte_6 causal_codepoint_grid causal_whitespace_grid \
  --ood-summary results/phase3-ood-confirmation/summary.json \
  --confirmation-authorization-summary \
    results/phase3-primary-clustered/summary.json \
  --output-root results/phase3-primary-confirmation
```

## 4. 해석

이 수정은 seed, model, optimizer, corpus, patch policy, loss, bootstrap 또는 gate
threshold를 바꾸지 않는다. Confirmation training 결과를 보지 않은 상태에서 실행
권한의 provenance만 강화했다. 이후 S/E/EC comparator confirmation에는 Gate I가
아니라 calibration-only comparator descriptor가 필요하므로, 이 primary-confirmation
authorization을 재사용하지 않는다.
