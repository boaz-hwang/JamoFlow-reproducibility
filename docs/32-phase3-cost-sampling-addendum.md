# Phase 3 addendum: direct-cost input sampling correction

> 작성일: 2026-08-10
> 상태: **W primary 및 모든 direct-cost 결과 확인 전 고정**
> 상위 protocol: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md)
> 영향: policy, quality endpoint, Gate I/J/K threshold, analytical FLOPs는 변경하지 않음

## 1. 발견한 문제

기존 `benchmark_phase3.py`는 test sequence의 seeded permutation에서 batch size별로 단 하나의 nested batch를 선택한 뒤 같은 입력을 반복 측정했다. 이 방식은 device timing noise를 측정할 수 있지만 input sampling이 약하다.

- batch 1은 한 sequence에만 의존한다.
- SpaceByte와 E/EC는 입력에 따라 realized patch width가 달라지므로 한 batch의 width가 대표적이지 않을 수 있다.
- timing p95가 input variability까지 포함하는 것처럼 오해될 수 있다.

이 문제는 quality 결과나 selector 정의와 무관하지만 Gate K의 direct-latency 근거를 약하게 만든다.

## 2. 결과 열람 시점 공개

이 수정 시점에는 seed 1,729의 F와 C scalar report가 생성돼 있었고 W는 학습 중이었다. W−C primary contrast, E/EC/S 결과, cost benchmark 결과는 존재하지 않았다. 수정 사유는 부분 quality 방향이 아니라 cost code의 single-batch sampling 구조다. 변경은 모든 policy에 대칭적으로 적용하며 Gate K 기준은 그대로 둔다.

## 3. 고정한 수정

각 batch size 1/8/32/64에 대해 다음을 사용한다.

1. HPLT3 test sequence의 하나의 seeded permutation을 만든다.
2. 이를 **8개의 서로 겹치지 않는 timing batch row**로 재구성한다.
3. 각 batch size는 모든 row의 nested prefix를 사용한다.
4. 한 timing round 안에서는 모든 policy가 같은 row를 사용한다.
5. 50회 measurement에서 8개 row를 가능한 균등하게 배정하고, policy 실행 순서는 round마다 무작위화한다.
6. warmup 10회도 같은 방식으로 row를 순환한다.

따라서 direct benchmark가 접하는 고유 test sequence 수는 다음과 같다.

| Batch size | Timing batches | Unique sequences |
|---:|---:|---:|
| 1 | 8 | 8 |
| 8 | 8 | 64 |
| 32 | 8 | 256 |
| 64 | 8 | 512 |

8개 row는 batch size 사이에서도 nested하다. 예를 들어 batch 1은 각 row의 첫 sequence, batch 8은 같은 row의 첫 8개 sequence를 사용한다.

## 4. 추가 무결성

Benchmark는 다음을 기계적으로 확인·기록한다.

- timing index matrix hash
- timing batch 수와 batch별 measurement count
- 각 batch size × timing row × policy에서 online selector가 cached evaluation matrix를 정확히 재구성하는지
- 모든 policy가 같은 balanced input-batch schedule을 사용했는지 나타내는 protocol field
- checkpoint, router, stream, patch-matrix hash

`--timing-batches`는 evidentiary run에서 최소 8이어야 하며 기본값도 8이다. 반복 수 최소 30 조건은 유지한다.

## 5. 해석 제한

보고하는 median과 p95는 고정된 8개 held-out batch를 순환했을 때의 device/runtime measurement 분포다. 전체 Korean input population의 latency quantile은 아니다. Analytical cost는 timing subset이 아니라 전체 test patch-count 분포로 계속 계산한다.

MPS teacher-forced timing이 incremental CUDA serving을 대표하지 않는다는 기존 제한도 유지한다. 이 수정은 local systems evidence의 input sampling을 개선할 뿐 production speed claim을 허용하지 않는다.
