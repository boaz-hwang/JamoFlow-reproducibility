# Phase 3 five-seed confirmation results

> 작성일: 2026-08-12
> 상태: **historical test 기반 mechanism replication 완료; actual-inference 결과 아님**
> 모델: 19,596,096-parameter BLT, Apple MPS

## 판정

F/C/W의 확인용 seed `57721`, `65537` 학습과 Leipzig OOD 평가가 모두 완료됐다.
초기 세 seed와 합친 corrected Gate J는 통과했다.

| 비교 (`left − right`) | mean test BPB | paired-seed 95% CI | crossed bootstrap 95% upper | document-cluster 95% upper | 부호 재현 |
|---|---:|---:|---:|---:|---:|
| W86 − C86 | -0.009347 | [-0.010970, -0.007725] | -0.008382 | -0.008206 | 5/5 negative |
| W86 − F86 | -0.014471 | 보고 artifact 참조 | -0.013809 | -0.013624 | 5/5 negative |

W86, C86, F86의 다섯-seed mean test BPB는 각각 `1.637117`, `1.646464`,
`1.651589`다. W86 − C86의 Holm-adjusted one-sided paired-seed p-value는
`4.47e-05`이고 document-window coverage는 `97.6544%`다.

Leipzig OOD guard도 통과했다.

| OOD 비교 | mean BPB | 허용 최대 regression | 판정 |
|---|---:|---:|---:|
| W86 − C86 | -0.013711 | +0.020 | pass |
| W86 − F86 | -0.016718 | +0.020 | pass |

## 증거 무결성

- 확인 학습 전에 corrected Gate-I summary와 pre-confirmation manifest hash를
  authorization으로 고정했다.
- 여섯 main report/checkpoint/per-sequence NLL artifact가 모두 존재하고 `.part`는
  없다.
- 각 checkpoint의 loaded state hash가 report의 trained-state hash와 일치한다.
- NLL은 정책마다 31,250개의 finite nonnegative float32 값이며, 동일 source/test
  stream과 document layout을 공유한다.
- OOD와 primary summary는 같은 다섯 checkpoint state를 직접 대조한다.
- 초기 세-seed authorization summary는 덮어쓰지 않았고, 다섯-seed 결과는 별도
  고정 경로에 생성했다.

Tracked result identities:

- `results/phase3-ood-five-seed/summary.json`:
  `864367ea95f122a63578a2b2df7bcab0532ebd7c6a9a6a95023cbc7916f0f149`
- `results/phase3-primary-five-seed/summary.json`:
  `c374c06303334f06577c118d37c7e703a031c2a91af12b10ffcd384bb803bd33`
- `results/phase3-primary-five-seed/observations.csv`:
  `28bc464671c6eb8a8847e09a4bf9cee78e4323b56f19407c9852609148933e54`

## 의미와 다음 단계

이 결과는 Korean whitespace-informed boundary가 같은 86-patch compute에서 plain
codepoint/fixed grid보다 품질이 좋아진다는 mechanism replication이다. 다섯 seed,
paired NLL, document clustering과 OOD 방향이 모두 일치하므로 reduced-rate study를
진행할 근거는 충분하다.

하지만 기존 HPLT test는 이미 development에 노출됐고 W86/C86의 patch 수가 같으므로,
이 결과만으로 실제 추론 효율이 개선됐다고 말할 수 없다. 다음 순서를 유지한다.

1. 새 disjoint final test를 model loss 없이 생성·검증하고 aggregate seal을 commit한다.
2. Calibration-only selection plan을 commit한다.
3. W64/W72와 same-rate C를 학습하고 checkpoint에서 calibration NLL을 재구성한다.
4. Candidate−C86 calibration rate gate가 통과한 정확한 한 rate만 확인한다.
5. 새 final test에서 candidate−C86 quality와 candidate−same-rate-C mechanism을
   one-shot으로 확인한다.
6. 둘 다 통과한 경우에만 batch-1 incremental actual timing을 연다.

Strongest raw reference S는 initial calibration에서 W86보다 약 0.091 BPB 좋지만
global patch가 약 153.5 대 86이다. Candidate가 S와의 broad calibration futility
screen을 통과하지 못하면 S를 약한 비교군으로 교체하지 않고 broad replacement
주장을 사전에 포기한다. 그 경우 C86 대비 within-family actual speedup만 별도이고,
publication-scale/BPE frontier를 통과하기 전에는 일반적인 “faster Korean LLM” 주장을
하지 않는다.
