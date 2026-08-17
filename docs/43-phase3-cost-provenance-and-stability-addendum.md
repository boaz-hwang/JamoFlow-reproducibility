# Phase 3 addendum: cost provenance and input-batch stability

> 작성일: 2026-08-11  
> 상태: **모든 S/E/EC 및 cost 결과 생성 전 고정**  
> 상위 protocol: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md)  
> 선행 보강: [direct-cost input sampling correction](./32-phase3-cost-sampling-addendum.md)  
> 영향: policy, quality endpoint, analytical FLOP 식, 10% latency threshold는 유지하고 evidence reconstruction과 latency 판정을 강화함

## 1. 수정 시점과 알려진 정보

이 addendum을 고정할 때 initial seed 1,729와 2,718의 F/C/W 학습 산출물은 존재했고 seed 31,415의 F가 학습 중이었다. 세 seed의 완결된 primary analysis unit, OOD 결과, S/E/EC 결과, cost benchmark 결과는 존재하지 않았다. 이 수정은 부분 quality 값의 방향이 아니라 다음 두 측정 취약점을 해결하기 위해 이루어졌다.

1. benchmark JSON이 quality summary, patch cache, checkpoint와 같은 evidence lineage에 속한다는 결박이 충분하지 않았다.
2. 전체 timing sample의 median만으로는 특정 input batch에 집중된 속도 이득을 안정적인 이득으로 오판할 수 있었다.

부분 primary 숫자는 이 수정 과정에서 추가로 열람하거나 threshold 선택에 사용하지 않는다.

## 2. benchmark 실행 전 provenance 결박

`benchmark_phase3.py`는 계산을 시작하기 전에 다음을 모두 확인한다.

- shared-seed quality summary가 모든 여섯 policy와 최소 세 seed를 포함하고 integrity를 통과했는가
- 현재 primary manifest의 model, optimization, stream, source artifact가 quality summary와 같은가
- 현재 `ko.jsonl`과 `integrity.json`의 byte size와 SHA-256이 manifest와 같은가
- seed 1,729의 모든 training report와 checkpoint artifact/state hash가 quality evidence와 같은가
- router report/checkpoint와 threshold diagnostics/cache가 quality evidence와 같은가
- structural/threshold NPZ의 artifact hash, exact key set, dtype, row 수, patch coverage가 같은가
- timing subset의 online selector가 cached evaluation matrix를 정확히 재구성하는가

Benchmark 결과에는 이 lineage와 각 raw timing sample이 사용한 input-batch ID를 함께 기록한다. Raw source text나 sequence index 자체는 기록하지 않는다.

## 3. summary의 독립 재구성

`summarize_phase3_cost.py`는 benchmark의 aggregate field를 신뢰하지 않고 다음을 다시 계산한다.

1. seed와 protocol에서 8개 input-batch measurement schedule을 재생성한다.
2. raw millisecond samples에서 repetitions, median, p05, p95, mean, sample standard deviation을 다시 계산한다.
3. raw sample과 batch ID의 길이·유한성·양수성, batch balance, 모든 method의 shared schedule을 검사한다.
4. test patch cache에서 policy별 patch count를 다시 얻고 batch 1/8/32/64의 ideal 및 implemented batch-max FLOPs를 다시 계산한다.
5. throughput과 모든 learned-versus-structural comparison을 다시 계산한다.
6. Gate J의 공통 F/C/W checkpoint lineage가 shared six-policy quality summary와 같은지 검사한다.

Tracked summary에서는 raw timing sample과 raw measurement batch-ID vector를 제거한다. 재구성에 사용한 ignored benchmark artifact의 SHA-256과 compact aggregate만 남긴다.

## 4. paired input-batch stability guard

Batch 1과 8에서 W와 quality로 사전 선택된 learned policy를 비교할 때, 각 8개 input batch 안에서 method별 latency median을 먼저 계산한다. Input batch `i`의 paired reduction은 다음과 같다.

\[
r_i = 1 - \frac{\operatorname{median}(T_{W,i})}
                 {\operatorname{median}(T_{L,i})}.
\]

보고 estimand는 여덟 `r_i`의 산술평균이다. Input batch를 cluster 단위로 10,000회 복원추출하고, 고정 seed `20,260,811 + batch_size`를 사용한 percentile 95% interval을 계산한다. 여덟 batch라는 작은 표본 때문에 이 interval은 보수적 안정성 guard이며 전체 Korean input population에 대한 정밀한 confidence interval로 해석하지 않는다.

Gate K의 latency component는 이제 같은 batch size에서 다음을 동시에 만족해야 한다.

- 전체 raw timing sample 기준 median latency reduction `>= 10%`
- paired input-batch bootstrap 95% lower bound `> 0`

Batch 1 또는 8 중 하나가 두 조건을 모두 만족해야 한다. Analytical 10%, quality noninferiority, SpaceByte 포함 nondominance 조건은 그대로 유지한다.

## 5. 해석 한계와 최종 inference 기준

이 보강 뒤의 Gate K도 512-byte teacher-forced forward에 대한 local systems screen이다. 다음을 입증하지 않는다.

- incremental KV-cache correctness
- autoregressive decode latency 또는 bytes/s
- TTFT와 decode를 합친 end-to-end serving speed
- CUDA production serving 성능

따라서 Gate K 통과만으로 `faster generation`, `inference speedup`, 또는 사용자가 요구한 최종 연구 가치 판정을 허용하지 않는다. 실제 자동회귀 경로와 publication-scale 판정은 별도 결과 전 protocol에서 더 강한 최종 gate로 고정한다.
