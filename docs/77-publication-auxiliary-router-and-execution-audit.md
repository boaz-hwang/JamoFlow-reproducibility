# Publication auxiliary-router and execution audit

> 작성일: 2026-08-12
> 상태: **publication artifact 생성 전 결과맹 설계 감사 및 protocol v3 교정**
> 적용 범위: publication-scale raw reference, model lock, family-aware feasibility, actual inference
> 주의: 이 문서는 품질·속도 양성 결과가 아니라 false-pass 차단 계약이다.

## 1. 왜 이 감사를 했는가

Compact comparator는 calibration-only selection에서 F/C/W/S/E/EC와 selected same-rate
codepoint 중 하나로 고정된다. 따라서 publication의 stable role
`raw_byte_reference`는 structural policy일 수도 있고 learned entropy policy E/EC일 수도
있다. 기존 publication schema는 이 차이를 alias 뒤에서 잃었다.

E/EC인데도 worker가 auxiliary 종류를 `none`으로 선언하면 다음을 모두 생략할 수
있었다.

- seed별 entropy-router checkpoint와 calibration threshold/cache
- router parameter
- router 학습과 256M train stream 전체 scoring
- main+router 동시 inference 시간 및 memory

이 상태에서는 main-only 속도·memory로 scale을 승인하고 runtime에서는 router를 실제로
실행하지 않은 채 “router-inclusive”라고 주장할 수 있다. Publication 결과가 아직 없고
family-aware/actual runner도 없으므로 관측 결과와 무관하게 계약을 먼저 교정했다.

## 2. Authoritative raw-reference descriptor

`src/jamoflow/publication_reference.py`는 compact selection artifact를 다음 concrete
descriptor로 재구성한다.

- `policy`, `runtime_policy`, `model_family`, `patch_count`
- `requires_entropy_router`
- selection artifact와 Phase 3/compute-conversion initial summary SHA-256
- descriptor protocol version과 canonical identity

정책별 예상 runtime/family/rate를 코드가 파생해 selection artifact와 대조한다.
`entropy_threshold_full`과 `entropy_threshold_codepoint`만
`auxiliary_kind=entropy_router`가 되고 나머지는 `none`이다. Model, runtime 또는
feasibility worker가 auxiliary 종류를 직접 고를 수 없다. Publication test나 latency를
본 뒤 raw policy를 다시 선택하는 경로도 없다.

## 3. Structured entropy-router bundle

E/EC의 각 seed는 opaque calibration hash 하나가 아니라 다음 structured bundle을
가진다.

- selection-descriptor identity, seed, E/EC policy와 runtime policy
- router checkpoint file hash와 loaded state hash
- router training report와 공통 architecture-config hash
- router train, calibration, test stream hash
- finite scalar threshold와 maximum patch length
- all-byte 또는 UTF-8-codepoint candidate-position 규칙의 canonical hash
- threshold cache와 diagnostics artifact hash
- train/calibration/test policy-specific patch-matrix hash

같은 seed의 E와 EC는 같은 learned router를 쓸 수 있지만 policy definition, threshold와
patch lineage는 구분한다. Seed 사이에는 checkpoint/report/cache/patch bundle을 공유하지
않는다. Config와 source streams는 같은 budget의 세 seed에서 같아야 한다. Bundle의
identity는 model snapshot, BPB, downstream, learning curve와 runtime lineage에 그대로
들어간다.

Learning curve에서는 main checkpoint뿐 아니라 router checkpoint, report, cache,
diagnostics, split별 patch matrix와 bundle identity를 budget×seed 전체에서 재사용할 수
없다. 이전 budget의 seed 0 artifact를 다음 budget의 seed 1로 회전시키는 우회도
거부한다. Config와 compact raw descriptor만 budget invariant다.

## 4. Main graph와 total system parameter

기존 target 표는 main BLT graph만 센다.

| Target | Main raw graph | Entropy router | E/EC total system |
|---:|---:|---:|---:|
| 50M | 49,823,488 | 3,541,248 | 53,364,736 |
| 75M | 76,492,480 | 5,491,520 | 81,984,000 |
| 100M | 98,403,360 | 6,626,400 | 105,029,760 |

Structural raw이면 auxiliary count는 0이다. E/EC이면 target별 router count가 정확히
일치해야 하며 `total_runtime_parameter_count = main + auxiliary`를 파생한다. Main-graph
parameter match와 실제 resident/trainable total을 논문에서 모두 공개한다.

## 5. Family-aware 시간·memory 계약

Entropy raw의 model당 core-pretraining 투영은 다음 합이다.

```text
T_raw = T_main_train + T_router_train + T_router_score
component_steps = ceil(component_total_raw_bytes / measured_raw_bytes_per_step)
component_hours = median_step_seconds * component_steps / 3600
```

Main과 router train은 각각 256M clean train bytes를 처리한다. Offline router score는
최소 같은 256M train stream 전체를 한 번 덮고 cache를 만든다. 각 component는 독립
subprocess/workload hash, 완료 flag, finite steady step 3개, median time, 실제 raw
bytes/step와 exact total source bytes를 기록한다. Calibration/test scoring의 고정
overhead는 common data manifest가 확정한 실제 bytes로 별도 기록하며 core 120시간 표에
포함하지 않았다면 이를 전체 campaign 시간이라고 부르지 않는다.

Memory gate는 main-only high-water를 받지 않는다. 다음 stage별 MPS driver-memory
high-water의 최댓값을 recommended maximum의 75%와 비교하며 모든 값은 양수여야 한다.

1. main workload
2. router training
3. full-source router scoring/cache
4. main과 router를 동시에 resident하게 둔 cached incremental runtime

각 family의 safety-adjusted time은 12시간 이하, 세 paired seed와 네 family의 합은
120시간 이하여야 한다. Candidate-only preflight는 계속 provisional이며 final scale을
승인하지 못한다.

## 6. 실제 timed path에서 router 실행 증명

Runtime lineage에 router bundle을 넣었다는 사실만으로는 실제 timing 경로가 router를
호출했다는 증거가 아니다. Runtime protocol v3는 timing과 같은 private trial artifact에
다음 정수 diagnostic을 추가한다.

- router observed model units
- router cached model units
- router scored model units
- router forward calls

E/EC reference에서는 앞의 세 값이 reference runtime-observed units와 같고 forward
calls가 `decode_forward_steps + 1`이어야 한다. Candidate, BPE와 structural raw에서는
모두 0이어야 한다. 이 counter, output trace audit와 latency array가 같은 trial artifact
hash를 공유해야 하므로 lineage만 router-inclusive이고 실제 측정은 main-only인 경로를
허용하지 않는다.

## 7. Correct publication execution DAG

다음 순서는 병렬화하지 않는다. 뒤 단계의 입력 identity가 앞 단계 산출물에 의존한다.

1. Compact calibration-only comparator descriptor와 Final Value Gate 봉인
2. Pinned benchmark/HPLT revision 및 file hash 확인
3. Contamination reference-equivalence fixture 통과 후 full-corpus scan
4. Candidate/raw/BPE 공통 clean train/calibration/test stream, document order와 hash 봉인
5. Filtered train split만으로 16K/32K tokenizer 학습 및 round-trip/transition audit 봉인
6. Candidate, concrete raw descriptor, BPE graph와 entropy auxiliary config instantiate
7. 실제 common batches/tokenizers/patch/auxiliary path로 50/75/100M four-family preflight
8. Largest passing scale와 campaign inputs를 pre-training lock으로 commit
9. E/EC이면 seed별 router train→score/cache→calibrate, 이어 네 family main training
10. 실제 checkpoint/router/calibration bundle을 post-training model-lock graph에 추가
11. Sealed BPB, learning curve, downstream과 actual inference를 한 번만 실행

독립적인 corpus 메타데이터 감사나 CPU 합성 테스트는 MPS 학습과 겹칠 수 있지만,
MPS evidence job과 latency job은 하나씩 실행한다. Model 선택, confirmation, scale lock과
sealed evaluation은 위 의존 순서를 지킨다.

## 8. 아직 해결되지 않은 실행 blocker

이 교정 시점에는 다음 실제 publication artifact가 없다.

- contamination-filtered common publication stream과 pinned tokenizer
- descriptor-aware four-family feasibility runner/report
- publication-scale router/main checkpoints와 learning curves
- structured router bundle을 실제 파일에서 재구성하는 evidence runner
- router counter를 같은 trial trace에서 생성하는 publication actual-inference runner

따라서 contract unit test가 통과해도 효율 개선을 증명하지 않는다. 위 runner와 실제
artifact가 model-lock protocol v3를 통과하기 전에는 publication-scale 학습을 시작하거나
논문 표에 양성 결과를 기록하지 않는다.

## 9. 결과 비의존 검증

초기 3-seed S/E/EC 학습이 완전히 종료되고 9개 report/checkpoint/NLL artifact와
`.part` 부재를 확인한 뒤에만 검증을 실행했다. Reference/model-lock/runtime/scale 및
연결 gate 집중 test 66개와 전체 회귀 test 371개가 통과했다. 여기에는 실제
50M/75M/100M router graph의 exact parameter count, entropy→structural 위장, router
counter 누락, main-only memory, auxiliary train/score 누락과 budget×seed artifact 회전
재사용 음성 검증이 포함된다. 이 검증은 contract correctness일 뿐 actual efficiency
결과가 아니다.
