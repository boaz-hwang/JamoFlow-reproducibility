# Hangul draft acceptance 결과와 systems cost model 교정

> 작성일: 2026-08-13
>
> 상태: **한국어-specific draft 가설 실패; target block-kernel upper-bound만 후속 허가**
>
> authoritative aggregate:
> `results/hangul-draft-acceptance-v1/summary.json`

## 1. 결론

Frozen W72 hidden에서 parameter-matched 네 draft를 세 initialization으로 비교했지만,
사전 feasibility gate를 통과한 architecture는 없었다. 가장 좋은 head도 한글 구조 head가
아니라 **generic independent UTF-8 continuation head**였다.

| architecture | params | free complete pair | first continuation | accepted suffix / 2 | median head ms | gate |
|---|---:|---:|---:|---:|---:|---|
| generic independent UTF-8 | 41,728 | **24.379%** | **42.373%** | **0.668** | 0.995 | fail |
| Hangul parallel components | 42,468 | 20.767% | 28.977% | 0.499 | 1.044 | fail |
| Hangul conditional components | 39,604 | 17.702% | 25.614% | 0.433 | **0.887** | fail |
| generic joint UTF-8 | 42,733 | 16.267% | 26.695% | 0.433 | 0.977 | fail |

Free-running W72 trace에는 `EA..ED` activation 14,422개가 있었고, 이들은 100% 실제
precomposed Hangul scalar였다. 따라서 실패 원인은 activation precision이 아니라 future
continuation prediction이다.

현재 허가되는 결론은 다음과 같다.

1. 조합규칙은 proposal validity를 보장하지만 이 target hidden과 작은 head budget에서
   acceptance를 개선하지 않았다.
2. Dependence-aware라고 해서 자동으로 강하지 않다. 9-rank joint pair head는 independent
   주변분포 head보다 크게 낮았다.
3. Jamo factorization 또는 conditional composition을 positive technique으로 계속
   최적화하지 않는다.
4. 다만 acceptance gate가 speculative correction/bonus byte를 계산하지 않았다는 systems
   오류가 드러났으므로, multi-byte branch 전체를 즉시 폐기하기 전에 **perfect-draft target
   block kernel의 실제 비용**만 별도 측정한다.

## 2. 실행과 provenance

- target: quality-authorized W72 seed 1729, weights frozen
- train/calibration hidden: 각 100,000 Hangul contexts
- free prompts: calibration-only 128개, prompt 128 bytes, output 380--383 bytes
- free draft attempts: 14,422
- head seeds: 20260813, 20260817, 20260819
- elapsed: 201.26 seconds
- result artifact SHA-256:
  `0e31dada00ca04835432f8f35b2e438b225dbed58e4ee12f262aff081fcd3591`
- internal canonical summary SHA-256:
  `02306d6f06012659f34fcd48b7c6394d9a24fa6e5eebc5ece2078746517b7924`

첫 실행은 model/data 전 authorization API 누락으로, 두 번째 실행은 artifact publish 전
80%-prompt feasibility로 중단됐다. Calibration 구조만 확인해 가능한 최소 변경인 79%로
낮춘 뒤 세 번째 실행에서 결과를 생성했다. Head 수, parameter budget, training, output
horizon, acceptance gate는 바꾸지 않았다.

## 3. Seed 안정성과 한국어-specific 판정

### Complete-pair acceptance

| architecture | seed 20260813 | seed 20260817 | seed 20260819 |
|---|---:|---:|---:|
| generic independent | 24.379% | 25.600% | 23.235% |
| generic joint | 16.191% | 16.267% | 18.319% |
| Hangul parallel | 21.918% | 19.519% | 20.767% |
| Hangul conditional | 17.702% | 19.727% | 16.419% |

Conditional Hangul은 사전 primary generic-joint control보다 prompt-paired 평균
`+1.059 percentage points`였지만 95% CI가 `[-0.706, +2.763] points`라 specificity gate를
통과하지 못했다. 더 중요한 결과는 사후 가장 강한 generic independent와의 비교다.
Conditional Hangul의 median은 6.677 points 낮았고 세 seed 모두 낮았다. Parallel Hangul도
independent보다 세 seed 모두 낮았다. 따라서 comparator 선택이 joint control에 불리하게
고정됐다는 사실을 이용해 한국어-specific 성공을 주장할 수 없다.

Teacher-forced complete-pair accuracy도 independent 16.192%, conditional 15.595%, parallel
14.117%, joint 12.106%였다. Free target trace의 수치가 더 높은 것은 작은 W72가 생성한
분포가 corpus label보다 반복적/자기일관적일 수 있음을 보여 주는 descriptive 차이이지,
품질 개선이 아니다.

## 4. 왜 사전 stop rule의 범위를 교정하는가

사전 gate는 complete-pair 40%, accepted suffix 0.90/2를 요구했다. 네 architecture 모두
실패했으므로 **learned Hangul draft 가설은 계획대로 종료**한다. Threshold를 낮추거나
epoch/rank/beam을 결과를 보고 재튜닝하지 않는다.

하지만 gate는 exact speculative decoding의 correction token을 비용모형에서 빠뜨렸다.
현재 target next-byte `b1`이 이미 있을 때 draft가 `d2,d3`를 제안하고 target이
`[b1,d2,d3]`를 한 block으로 검증한다고 하자.

- `d2` mismatch: `b1`과 target correction `b2`를 확정 → 2 bytes
- `d2` accept, `d3` mismatch: `b1,d2`와 correction `b3` 확정 → 3 bytes
- 둘 다 accept: `b1,d2,d3`와 verifier bonus `b4` 확정 → 4 bytes

따라서 verifier당 기대 확정 bytes는

`2 + P(d2 accepted) + P(d2,d3 accepted)`

이다. 가장 강한 independent head의 median에서는

`2 + 0.423728 + 0.243794 = 2.667522 bytes/verification`

이다. 이는 `accepted suffix < 0.90`만으로 speed 불가능을 결론 내릴 수 없음을 뜻한다.
Acceptance 결과를 positive로 바꾸는 것이 아니라 **target block kernel의 비용을 아직
측정하지 않았다는 결손**을 수정하는 것이다.

Component profile의 non-boundary byte step 2.36 ms와 현재 unoptimized head 0.995 ms를
진단적으로 대입하면, 2-draft verifier가 대략 5.30 ms보다 빨라야 break-even이다.
이 값은 production bound가 아니다. Patch boundary, rollback, cache crop, block-shape MPS
kernel이 모두 포함된 실제 측정만 판정할 수 있다.

## 5. 수정된 다음 단계

결과를 보고 head를 더 튜닝하지 않는다. 먼저 draft와 무관한 target-side upper bound를
사전 봉인한다.

1. Sequential W72와 같은 checkpoint/cache에서 1/2 future-byte block의 logits,
   boundaries, cache를 exact 비교한다.
2. Perfect continuation을 입력해 non-boundary와 patch-boundary strata의 target block
   latency를 측정한다.
3. 기존 independent head seed 20260813의 **이미 고정된** acceptance/latency를 결합한
   conservative projected E2E가 sequential bytewise보다 20% 이상 낮을 때만 rollback과
   exact speculative runtime을 구현한다.
4. Upper bound가 실패하면 multi-byte branch를 완전히 종료한다.
5. Upper bound가 통과해도 한국어-specific claim은 복원되지 않는다. 실제 system은
   Hangul-heavy UTF-8 scalar-aligned speculative decoding이며, generic all-byte MTP와 standard
   W72 AR이 필수 comparator다.

개발은 calibration에서 수행한다. 새 method의 publication claim에는 이미 개봉한 final을
재사용하지 않고 별도의 disjoint held-out replication이 필요하다.

## 6. Claim 경계

- 네 draft 모두 preflight gate fail이다.
- 조합형 Hangul draft가 generic independent를 이겼다는 증거는 없다. 실제로 반대다.
- 2.6675 bytes/verification은 acceptance로부터 계산한 work opportunity이지 speedup이 아니다.
- 0.995 ms는 현 Python/MPS proposal implementation의 isolated median이며 최종 fused cost가
  아니다.
- 다음 target block 측정은 post-result exploratory upper-bound다.
- Exact full runtime이 실제 E2E를 개선하기 전에는 새 positive efficiency claim이 없다.
