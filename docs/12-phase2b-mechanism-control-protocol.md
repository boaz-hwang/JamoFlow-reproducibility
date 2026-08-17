# Phase 2b protocol addendum: mechanism and artifact controls

> 작성일: 2026-08-10  
> 상태: **아래 control 결과 확인 전 고정**  
> 기준 primary 결과: [Phase 2 primary results](./11-phase2-primary-results.md)  
> 성격: **post-primary, prospectively specified mechanism study**

## 0. 이 addendum이 필요한 이유

Phase 2 primary 결과를 확인한 뒤 다음 현상을 알게 됐다.

- `causal_codepoint_grid − fixed_byte_6`: −0.00654 BPB
- `causal_eojeol_grid − causal_codepoint_grid`: −0.00708 BPB
- 두 contrast 모두 5/5 seed에서 negative

첫 contrast는 원 protocol의 causal replication 가설을 직접 검사한다. 둘째 contrast는 통계적으로 강하지만 “어절 경계의 언어학적 가치”만 바꾼 실험은 아니다. Eojeol policy는 동시에 다음을 바꾼다.

1. target보다 이른 delimiter에서 boundary를 낼 수 있음
2. delimiter가 없으면 target `+2`까지 기다림
3. codepoint grid보다 patch-length distribution의 tail이 김
4. BLT patch lag에 유리한 위상으로 우연히 이동할 수 있음

따라서 primary 결과만으로 eojeol semantics를 원인이라고 말하면 과도하다. 이 문서는 결과를 본 사실을 숨기지 않고, alternative mechanism을 구분할 다음 실험을 **실행 전에** 고정한다. Phase 2 primary의 confirmatory label은 유지하되, Phase 2b 결과는 독립적인 post-primary evidence로 보고한다.

## 1. 고정되는 부분

다음은 Phase 2 primary와 같다.

- Korean Wikipedia 2021 hash split과 byte caps
- compact BLT graph와 1,251,136 parameters
- seeds 1,729 / 2,718 / 31,415 / 57,721 / 65,537
- seed 안에서 동일 initial state와 train order
- 256 bytes, 정확히 43 data patches
- one-pass optimization schedule
- calibration/test bytes와 평가 BPB
- 정책별 test selection이나 early stopping 없음

기존 C1/C2 checkpoint와 per-sequence NLL은 재사용한다. 새 control만 같은 조건으로 학습한다.

## 2. Original-protocol artifact controls

### A0 — exact duplicate

seed 1,729의 C1을 같은 initial state, order, inputs, patch matrix로 한 번 더 학습한다.

기록:

- test BPB difference
- per-sequence NLL 최대·평균 절대차
- 최종 state tensor 전체의 최대 절대차
- checkpoint SHA-256

해석:

- BPB 차이 >0.001이면 그 이하 policy 차이는 noise floor 아래로 처리
- duplicate 차이가 C1−C0 primary effect의 50%를 넘으면 scale-up 중단

### A1 — aligned packing

원 stream에서 각 row 시작과 끝을 complete codepoint boundary로 맞춘다.

- 한 row에 가능한 최대 raw prefix ≤256 bytes
- 마지막 complete codepoint 뒤를 newline으로 pad
- 일반 row의 padding은 0–3 bytes
- 256 bytes보다 짧게 남은 마지막 fragment는 버림
- raw bytes, inserted newline bytes, dropped tail, insertion rate 기록

C0/C1을 seeds 1,729 / 2,718 / 31,415로 다시 학습한다. `C1−C0`의 3-seed 평균이 negative여야 Gate D를 통과한다. Primary arbitrary-packing 추정치와 합치지 않는다.

## 3. Phase/lag mechanism controls

모든 target은 `τ_j = ceil(j × 256 / 43)`이다. 마지막 target은 window 끝에서 patch를 잃지 않도록 C1처럼 offset 0을 쓴다.

### M0 — `causal_grid_early2`

마지막을 제외한 각 target `τ_j−2` 이후 첫 complete codepoint에서 boundary를 낸다. Delimiter를 전혀 보지 않는다.

### M1 — `causal_grid_delayed2`

마지막을 제외한 각 target `τ_j+2` 이후 첫 complete codepoint에서 boundary를 낸다. 이는 C2에서 delimiter가 한 번도 발동하지 않는 경우와 같다.

M0/M1은 fixed grid phase만 바꿔 primary C2의 이득이 단순 boundary shift 또는 patch-lag 효과인지 검사한다.

## 4. Causal placebo-event control

### M2 — `causal_placebo_grid`

C2와 같은 `target−2` 시작, `target+2` deadline, minimum patch length 2를 사용하되 delimiter 대신 관측된 prefix의 deterministic rolling-hash event를 사용한다.

- 64-bit FNV-1a state
- 각 256-byte row 시작에서 fixed offset basis로 reset
- byte `t−1`까지 소비한 hash만 position `t` 결정에 사용
- complete codepoint boundary에서만 event 평가
- normalized low bits가 calibration에서 고정한 threshold 아래일 때 event
- final target은 C1과 같이 offset 0

Hash threshold는 Korean calibration split에서 **C2의 nonfinal delimiter-trigger fraction**에 가장 가까워지도록 고정한다. 품질 label과 test split은 쓰지 않는다. Calibration target은 event fraction이며 patch 수는 모든 threshold에서 정확히 43이다.

다음 matching diagnostics를 함께 보고한다.

- early event trigger fraction
- target-relative displacement mean/median/p05/p95
- patch-length median/p95/max
- whitespace·punctuation boundary hit rate

Placebo가 C2와 완전히 같은 분포를 보장한다고 주장하지 않는다. 남은 분포 차이는 명시적으로 보고한다.

## 5. Delimiter decomposition

### M3 — `causal_whitespace_grid`

C2와 동일하지만 Unicode whitespace 뒤만 early trigger로 허용한다.

### Exploratory M4 — `causal_punctuation_grid`

Unicode punctuation 뒤만 허용한다. 원 protocol의 punctuation addendum 조건을 retroactively primary로 바꾸지 않는다. Calibration C2 selected early triggers 중 punctuation share가 50%를 넘을 때만 학습하고, 그렇지 않으면 구조 통계만 보고한다.

Whitespace와 punctuation이 동시에 성립하는 codepoint는 whitespace로 분류한다.

## 6. Primary Phase 2b contrasts

이 addendum의 두 mechanism contrast는 다음이다.

1. C2 `causal_eojeol_grid` − M1 `causal_grid_delayed2`
2. C2 `causal_eojeol_grid` − M2 `causal_placebo_grid`

각 contrast에서 다음을 모두 만족해야 delimiter-aware linguistic signal을 유지한다.

- mean difference ≤ −0.003 BPB
- 5 seeds 중 최소 4개 negative
- paired-t 95% upper bound < 0

둘 중 하나라도 실패하면 “eojeol prior improves quality”를 중심 claim으로 쓰지 않는다. 이 경우 C2는 좋은 heuristic일 수 있지만 linguistic causal attribution은 철회한다.

Secondary diagnostics:

- M0/M1 − C1: grid phase sensitivity
- M3 − M1: whitespace signal beyond delay
- C2 − M3: punctuation의 추가 효과
- 모든 policy의 Korean test strata

Secondary 결과에는 새로운 gate를 만들지 않는다.

## 7. Gate 업데이트 규칙

### Gate D

Primary C1−C0 조건은 이미 충족했다. Aligned packing 3-seed 평균이 negative이고 duplicate noise stop condition이 없을 때 최종 통과한다.

### Gate E

기존 primary C2−C1 조건, exact patch count, ecological/external no-regression 외에 이 addendum의 두 mechanism contrast를 모두 요구한다. 이는 원 protocol보다 엄격한 사후 방어 규칙이다.

### Gate H

Scale-up은 다음 모두가 확인될 때만 연다.

- Gate D 또는 강화된 Gate E
- duplicate noise 허용 범위
- aligned direction 유지
- cost 또는 validity stop condition 없음

## 8. Claim boundary

Phase 2b가 모두 성공해도 입증되는 것은 compact Korean byte-BLT에서의 boundary mechanism이다. 다음은 여전히 입증되지 않는다.

- 대형 모델 scaling law
- 한국어 형태론 FST의 추가 가치
- 자소 단위 병렬 생성 속도
- production CUDA autoregressive latency
- 영어·중국어보다 한국어에서 특별히 큰 eojeol 효과

이 항목들은 scale-up 설계의 질문이지 Phase 2b의 결론이 아니다.
