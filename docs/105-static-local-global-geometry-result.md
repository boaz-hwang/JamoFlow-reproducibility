# Static local/global geometry preflight result

> 작성일: 2026-08-13
>
> plan commit: `b8ad8e999fc867bcbc7d5ba7eb6706f3bb926ac9`
>
> protocol: `jamoflow-static-geometry-preflight-v1`
>
> result: `results/static-geometry-preflight-v1/summary.json`

## 1. 판정

사전 고정한 세 candidate 중 두 번째인
`thin160_e1_d1_g384x9`가 모든 feasibility gate를 통과했다.

| geometry | params 차이 | counted FLOPs 감소 | actual E2E 감소 | 95% prompt bootstrap | 양수 prompt | 판정 |
|---|---:|---:|---:|---:|---:|---|
| thin128 E1/D2 | +0.050% | 29.347% | 5.773% | [2.296%, 14.299%] | 24/32 | fail |
| thin160 E1/D1 | -0.124% | 31.047% | **24.417%** | **[19.202%, 29.112%]** | **32/32** | **pass** |
| thin128 E1/D1 | -0.104% | 36.393% | 19.610% | [15.411%, 26.407%] | 31/32 | fail |

Baseline median은 592.524 ms, 선택 candidate median은 447.849 ms였다. 네 geometry 모두
128개 sequential/parallel logit 위치의 argmax, boundary trace와 cache diagnostics가
일치했고 정규화 최대 logit error도 1 이하였다.

규칙에 따라 이 결과가 허가하는 것은 `thin160_e1_d1_g384x9`의 **Korean 한 seed
train/calibration 비교 하나**뿐이다. Random-weight timing은 BPB noninferiority, 실제
학습 안정성 또는 publication efficiency를 증명하지 않는다.

## 2. 결과가 새로 알려 준 것

첫째, compact W72에서 local path를 줄이고 parameter를 global trunk로 옮기는 방향에는
실제 Apple MPS latency potential이 있다. Counted dense FLOPs 31.047% 감소가 실제
controlled generation 24.417% 감소로 이어졌으므로, 기존 2.5% W72 schedule 효과와
9.983% exact speculation보다 큰 다음 단계 후보가 처음 생겼다.

둘째, FLOPs와 latency는 단조롭지 않았다. 가장 공격적인 thin128 E1/D1은 counted FLOPs가
36.393% 줄었지만 actual point reduction은 19.610%로 gate를 0.390 percentage point
놓쳤다. 반대로 더 넓은 thin160이 더 빨랐다. 이 compact MPS graph에서는 작은 hidden
shape의 낮은 kernel utilization, projection shape, launch/memory overhead가 FLOP 절감 일부를
상쇄한다는 해석이 가장 타당하다. 이는 관측 결과에 근거한 시스템 가설이지 아직 kernel
profiler로 식별한 인과 결론은 아니다.

셋째, decoder 2층을 보존한 thin128 E1/D2는 5.773%에 그쳤다. 이 결과는 local decoder의
반복 depth가 실제 병목에서 중요하다는 기존 component profile 해석과 일치한다. 다만
width와 depth를 동시에 바꾼 세 geometry 비교이므로 decoder depth의 독립 효과로 주장하지
않는다.

## 3. 연구 방향에 주는 수정

전체 방향을 바꿀 필요는 없지만 우선순위는 더 선명해졌다.

1. 다음 학습 대상은 고정 순서상 첫 통과 후보인 thin160 E1/D1 하나다. thin128 E1/D1을
   19.610%라는 사후 근접성 때문에 함께 학습하지 않는다.
2. 같은 Korean byte budget과 seed에서 original W72 대비 calibration BPB noninferiority와
   actual controlled/free latency를 본다.
3. 이 정적 control이 품질을 잃으면 local capacity를 무조건 줄이는 방식은 폐기하고,
   같은 평균 compute에서 UTF-8/Hangul state별로 local depth를 보존·생략하는 conditional
   candidate로 넘어간다.
4. 정적 control이 matched quality와 actual speed를 모두 유지하면 그 geometry를 이후
   generic UTF-8 및 Hangul-specific conditional-depth 실험의 공통 backbone/control로 쓴다.
5. 정적 control 자체는 BLT의 알려진 local/global allocation axis이므로 novelty claim을
   만들지 않는다.

핵심 성공 기준은 그대로다. 학습된 모델이 matched quality에서 실제 추론을 유의미하게
줄여야 한다. 이번 결과는 그 검증에 학습 비용을 쓸 만하다는 feasibility 근거이지 최종
논문 결론이 아니다.

## 4. 다음 단계의 고정 원칙

한 seed screen은 train byte budget, optimizer, batch order, W72 patch matrix와 평가 stream을
baseline과 공유해야 한다. 모델은 처음부터 새 geometry로 학습하며 기존 W72 checkpoint를
부분 이식하지 않는다. Primary quality screen은 calibration BPB의 paired document/block
noninferiority이고, test/final-quality artifact는 열지 않는다.

속도는 학습 결과를 본 뒤 임의 workload를 고르지 않도록 지금 사용한 32개 calibration
prompt의 controlled replay와 별도 deterministic calibration free-running protocol을
학습 전에 봉인한다. 최소 효율 gate는 이번 random-weight preflight의 20%를 그대로
publication claim으로 재사용하지 않고, 품질을 통과한 실제 checkpoint pair에 대해 새로
사전 고정한다.
