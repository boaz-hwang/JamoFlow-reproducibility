# Family-aware scale feasibility correction

> 작성일: 2026-08-11
> 상태: **publication-scale preflight 실행 전 고정**
> 교정 대상: [Mac feasibility addendum](./47-publication-scale-feasibility-addendum.md)
> 영향 범위: final publication scale 선택; compact Phase 3와 quality gate는 불변
> parameter-lineage 후속 교정: [family parameter-identity correction](./66-family-parameter-identity-correction.md)
> time-projection 후속 교정: [family time-projection correction](./67-family-time-projection-correction.md)
> auxiliary 후속 교정: [publication auxiliary-router and execution audit](./77-publication-auxiliary-router-and-execution-audit.md)

## 1. Candidate 한 graph의 시간을 12배 하면 충분하지 않다

기존 feasibility runner는 byte-latent candidate의 한 train step을 측정하고 그 model당 투영 시간을 core run 수만큼 곱했다. Core family가 candidate/raw/BPE-16K/BPE-32K로 늘어난 뒤에는 이 값을 final campaign time으로 사용할 수 없다.

- Raw reference는 compact calibration-only selection이 잠근 concrete policy descriptor를 그대로 승계한다. Structural F/C/W/S/selected-C일 수도 있고 learned E/EC일 수도 있으며, E/EC이면 별도 entropy router가 필요하다. Main graph parameter 수가 candidate와 같아도 patch rate, global positions 및 auxiliary 비용은 다를 수 있다.
- BPE는 512 raw bytes가 가변 token 수로 바뀌며 output projection과 attention sequence length의 비용 구성이 다르다.
- 16K와 32K는 같은 body지만 embedding/output rows와 token sequence length가 다르다.
- 어느 family가 가장 느리고 memory가 큰지는 graph parameter 수만으로 보장되지 않는다.

Candidate time × 12는 provisional upper-bound 가정으로도 실제 측정 없이 증명되지 않는다. 큰 model을 안전하게 고르는 scale lock에는 각 runtime family의 실제 preflight가 필요하다.

## 2. 두 단계 feasibility

### 2.1 Candidate-only preflight

기존 `benchmark_publication_scale_feasibility.py`의 candidate graph 측정은 세 target이 MPS에서 대략 실행 가능한지 확인하는 빠른 preflight로 유지한다. 그 결과의 `pass`는 **provisional**이며 publication scale을 잠그지 않는다.

### 2.2 Family-aware campaign lock

Final scale 선택 전에 각 50M/75M/100M target에서 다음 네 family를 독립 subprocess로 측정한다.

1. selected whitespace candidate
2. locked strongest raw-byte reference의 concrete policy descriptor와, E/EC일 때의 entropy-router auxiliary path
3. body-matched 16K byte-BPE
4. parameter-matched 32K byte-BPE

모든 family가 완료된 뒤에만 `select_largest_campaign_feasible_scale`이 가장 큰 passing target을 반환할 수 있다. Candidate-only selector의 값을 publication runner에 직접 전달하지 않는다.

## 3. Family별 workload

Quality score를 계산하거나 scale 선택에 사용하지 않는다. 그러나 runtime shape는 실제 campaign과 같아야 한다.

- 같은 clean publication train stream에서 raw source bytes가 동일한 batch를 사용한다.
- Byte-latent family는 실제 selected/reference patch matrices를 사용한다.
- Raw descriptor가 E/EC이면 main step만 재지 않는다. Router 학습, 공통 clean train stream 전체의 offline scoring/cache 구축, calibration threshold 구축, 그리고 main+router를 함께 둔 cached incremental prefill/decode를 각각 실제 workload로 실행한다.
- BPE family는 고정된 16K/32K tokenizer로 그 source batch를 실제 encode하고 padding/loss mask를 campaign과 동일하게 만든다.
- 각 family에서 warmup 1회, steady train step 3회, evaluation step, cached incremental prefill/decode를 수행한다.
- BPE tokenizer 학습은 train split만 사용하며 downstream/test text나 quality 결과를 사용하지 않는다.
- Family마다 별도 process를 사용하고 AC power, default power mode와 thermal eligibility를 확인한다.
- Entropy auxiliary의 train/score/runtime stage마다 완료 여부, finite measurement 수, workload/config hash와 MPS driver-memory high-water를 기록한다. Runtime high-water는 main과 router가 동시에 resident인 실제 경로에서 잰다.

Projected hours는 batch 수가 아니라 **관측 raw source bytes/step**을 분모로 계산한다. BPE가 같은 raw bytes를 더 적은 token으로 처리하는 효과와 padding overhead가 이 투영에 실제로 들어가야 한다.

Worker가 `projected_hours_per_model`을 자체 선언하지 않는다. Final result는
steady-step median seconds와 관측 raw source bytes/step만 받아 256M-byte budget의
step 수와 시간을 코드에서 재계산한다.

## 4. Final campaign 조건

Family 하나의 preflight가 통과하려면 다음을 모두 만족해야 한다.

1. main train/eval/incremental step이 완료되고 loss/time이 finite
2. 실제 main-graph parameter count가 preregistered family graph와 일치
3. E/EC이면 router train/score/runtime stage가 모두 완료되고 router parameter/config/workload identity가 봉인값과 일치
4. main, router train, router score, main+router runtime의 stage별 driver-memory high-water 중 최댓값이 recommended maximum의 75% 이하이며 0보다 큼
5. safety factor 1.20을 적용한 model당 projected core-pretraining time이 12시간 이하

Target 전체의 campaign 시간은 family별 시간을 먼저 합산한 뒤 paired seed 수를 곱한다.

\[
T_{core\ campaign}(m)
= 3 \times 1.20 \times
\sum_{f \in \{W,R,B16,B32\}} T_{m,f}
\]

여기서 entropy raw family의 `T_{m,R}`는 main training뿐 아니라 router training과 sealed source scoring을 포함한다. 이 값이 120시간 이하여야 한다. 이는 12개 core pretraining run의 예산이며 downstream fine-tuning·최종 BPB·timing은 별도 실행 예산으로 보고한다. `max(T_f) × 12`나 `T_candidate × 12`로 대체하지 않는다. 네 family 중 하나라도 실패하거나 parameter/descriptor identity가 다르면 해당 target은 실패다.

## 5. 선택과 downshift

50M/75M/100M의 네-family preflight를 모두 결과와 무관한 고정 순서로 완료한다. 그 뒤 100M, 75M, 50M 순으로 검사해 가장 큰 passing target을 고른다.

- 100M의 BPE만 OOM이어도 candidate가 통과했다는 이유로 100M을 선택하지 않는다.
- 실패한 family만 batch/precision을 바꿔 구조하지 않는다.
- 어느 target도 통과하지 못하면 baseline을 생략하거나 120시간 상한을 사후 완화하지 않고 외부 compute 필요로 판정한다.

## 6. Machine-checkable boundary

`src/jamoflow/publication_scale.py`는 두 결과 타입을 분리한다.

- `ScaleFeasibility`: candidate-only provisional preflight, `family_aware_campaign_lock=false`
- `CampaignScaleFeasibility`: compact selection에서 재구성한 raw-reference descriptor, 네 family의 main/auxiliary parameter·stage별 memory/time 결과와 합산 core-campaign time, `family_aware_campaign_lock=true`

Final publication runner는 두 번째 artifact의 hash와 selected target만 받아야 한다. Family-aware runner 자체는 아직 실행되지 않았으며, 구현이 완성되기 전에는 publication-scale model training을 시작하지 않는다.

네 family의 `expected_parameter_count`는 worker가 자체 선언하지 않는다. Final
lock은 target×family 봉인 표와 actual graph count를 직접 대조하며, 상세 값과
self-attestation 차단은 후속 교정 문서 66을 따른다.
