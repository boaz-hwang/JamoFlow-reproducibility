# Fable 5 중간 검토에 대한 재검토와 연구 방향 보완

> 작성일: 2026-08-13
>
> 대상: [`fable5-연구-중간-검토.md`](../fable5-%EC%97%B0%EA%B5%AC-%EC%A4%91%EA%B0%84-%EA%B2%80%ED%86%A0.md)
>
> 상태: sealed-final quality 고정 후 작성; 아래 12--17절에 v5r3, component
> profile, learned draft, exact speculative runtime 및 최신 문헌 재검토를 반영

> 읽는 순서: 2--11절은 결과 개봉 전 검토 기록이다. 현재 최종 판정은
> 15--24절과 `docs/142-fable5-final-retrospective-and-current-direction.md`가 우선한다.
>
> 원칙: 세션 receipt의 correctness·환경 metadata만 확인했으며 latency array와
> latency aggregate는 열지 않았다.

## 1. 총평

외부 검토의 중심 판단은 타당하다. 연구 질문은 살아 있고, W72의 sealed-final
quality 결과와 same-rate C72 대비 기전 대비는 논문에 쓸 수 있는 수준으로 잘
식별되어 있다. 특히 `actual inference`가 analytical patch/FLOP 절감을 따라오지
않을 수 있다는 경고는 반드시 수용해야 한다.

다만 다음 네 문장은 그대로 수용할 수 없다.

1. 현재 확립된 것을 포괄적인 `컴퓨트 개선`이라고 부르는 것
2. 측정 전에 `메모리 개선 없음`이라고 결론짓는 것
3. `16.3% × global trunk 시간 비중`을 이론적 속도 상한이라고 부르는 것
4. actual speed가 실패해도 사용자가 정한 핵심 연구 목표가 달성된다고 보는 것

따라서 연구의 주 가설이나 v5r3 gate는 바꾸지 않는다. 대신 analytical claim을
정밀화하고, actual 결과가 실패할 때에는 작은 quality paper로 종료하지 않고 병목을
분해한 뒤 실제 효율을 만드는 새 구조로 연구를 이어가는 outcome branch를 명시한다.

## 2. 항목별 수용 판정

| 외부 검토 항목 | 판정 | 근거와 조치 |
|---|---|---|
| 연구 질문이 유효하고 최근 연구와 맞닿아 있음 | 수용 | BLT, SpaceByte, AU-Net, H-Net, ByteFlow는 모두 byte hierarchy/chunking과 compute allocation을 다룬다. JamoFlow의 좁은 차별점은 한국어에서 관측 가능한 causal whitespace 경계를 detector-free policy로 검증하고 실제 생성 비용까지 닫는 것이다. |
| sealed-final W72−C86/C72 수치 | 수용 | 논문 초안 및 quality lock과 일치한다. W72−C86 `+0.003682 BPB`는 고정 `+0.010` margin에서 noninferior이고, W72−C72 `−0.010781 BPB`는 superiority gate를 통과했다. |
| S가 강하지만 rate가 훨씬 높아 broad replacement 주장을 못 함 | 수용 | calibration에서 W72−S는 `+0.103950 BPB`, margin 안 seed는 0/3이다. S의 평균 data patches `153.313`은 C/W86보다 78.3% 많다. |
| E/EC 결과가 테스트한 learned router에 불리함 | 범위를 좁혀 수용 | E/EC는 W86보다 나쁘고 각각 2,016,960 auxiliary parameters가 필요했다. 이는 이 compact 설정의 이 router를 기각할 뿐 learned routing 일반을 기각하지 않는다. |
| 10% actual-speed gate가 빡빡함 | 수용 | 계수된 total dense-matmul 감소는 8.33%다. 10% wall-clock 감소는 가능하더라도 자동으로 기대할 수 있는 결과가 아니다. |
| 16.3%×global share가 이론적 상한 | 기각 | projection, cross-attention, quadratic attention, cache, dispatch, memory movement, kernel shape 효과를 생략한다. 이는 heuristic risk model이지 upper bound가 아니다. |
| 메모리 개선 없음 | 기각 | weight memory는 동일하지만 global/cross-attention cache와 allocator high-water는 patch 수에 따라 달라질 수 있다. role-isolated 10-unit 측정 전에는 `동일 parameter memory, runtime peak 미결`만 허용한다. |
| speed 실패 시 작은 quality/method paper 가능 | 학술적으로 수용, 핵심 목표로는 불수용 | 별도 소논문은 가능하지만 이 프로젝트의 성공 기준은 실제 추론 효율 개선이다. 실패하면 speed claim을 제거하고 profiler와 구조 재설계로 계속 간다. |
| S 우위 분해 실험 | 조건부 수용 | 중요한 reviewer question이지만 actual efficiency 결과보다 우선하지 않는다. 단일 `S downsample`보다 C/W rate grid와 authentic S를 함께 두는 설계가 낫다. |
| total-cost Pareto 완결 | 수용 | 단, 아직 authoritative summary가 아니라 diagnostic Gate K다. actual v5r3를 대체하지 않는다. |
| CUDA replication 검토 | 조건부 수용 | compact actual result가 양성이거나 scaling preflight가 타당성을 보일 때 별도 protocol로 연다. Mac 결과를 CUDA 결과로 대체하지 않는다. |
| `docs/87-*` 중복 번호 수정 | 수용 | claim–evidence matrix를 `docs/88-*`로 이동했다. |

## 3. 확립된 analytical workload의 정확한 수치

현재 `src/jamoflow/cost.py`가 계수하는 것은 구현 그래프의 dense matrix
multiplication이다. embedding lookup, RMSNorm, RoPE, activation, softmax, selector,
Python/framework dispatch와 memory movement는 명시적으로 제외된다.

| 항목 | C86 | W72 | W72의 C86 대비 감소 |
|---|---:|---:|---:|
| data patches / 512 bytes | 86 | 72 | 16.279% |
| HF global positions, dummy 포함 | 87 | 73 | 16.092% |
| total dense-matmul FLOPs / sequence | 6,152,810,496 | 5,640,155,136 | **8.332%** |
| global Transformer FLOPs / sequence | 2,761,371,648 | 2,304,454,656 | 16.547% |
| main parameters | 19,596,096 | 19,596,096 | 0% |
| auxiliary router parameters | 0 | 0 | 0% |

따라서 허용되는 표현은 다음과 같다.

> 이 19.6M BLT 그래프와 한국어 설정에서 W72는 C72보다 sealed-final BPB가
> 낮았고, C86 대비 품질 noninferiority를 유지하면서 data patches를 16.28%,
> dummy를 포함한 global positions를 16.09%, 계수된 dense-matmul workload를
> 8.33% 줄였다. 실제 end-to-end generation latency와 runtime memory는 별도
> 실측 결과가 나올 때까지 미결이다.

반대로 `C86이 W72보다 16.3% 더 많은 global position을 쓴다`는 문장은 분모가
뒤집혀 틀리다. data patch 기준으로 C86은 W72보다 19.44% 많고, W72가 C86보다
16.28% 적다.

## 4. 10% gate에 대한 판단

외부 검토가 위험을 짚은 방향은 맞다. local encoder/decoder와 byte head는 두
정책에서 동일한 byte horizon을 처리하며, dense-matmul 회계에서도 전체 감소는
8.33%에 그친다. 따라서 두 co-primary mode에서 10% 이상, 5/5 session 양수,
3/5 session 10% 이상을 동시에 요구하는 v5r3는 강한 gate다.

그러나 이를 실패 예측이나 이론적 상한으로 승격하지 않는다.

- incremental cache에서는 teacher-forced FLOP 식과 실제 호출 구조가 다르다.
- 줄어든 global call이 제거하는 kernel launch와 cache movement는 FLOP 비율보다
  비쌀 수도, 더 쌀 수도 있다.
- MPS의 shape별 utilization과 synchronization 비용은 analytical 식에 없다.
- controlled replay와 free running은 서로 다른 경로 민감도를 가진다.

결론적으로 현재 gate는 변경하지 않는다. 결과가 0–10%라면 `positive trend`로
gate를 소급 완화하지 않고 primary inference-efficiency claim은 실패로 기록한다.

## 5. 메모리 판단의 수정

W72와 C86의 parameter graph가 같으므로 persistent weight bytes가 같은 것은
확정이다. 하지만 다음은 아직 측정 전이다.

- global KV/cache state의 patch-count 의존 부분
- cross-attention state 및 allocator high-water
- role별 process baseline을 뺀 peak increment
- 실행 종료 뒤 release residual

따라서 E/EC의 2.02M router 절감은 learned-router 대비의 보조 관찰일 뿐,
primary W72−C86 memory 결론이 아니다. v5r3 memory는 resettable native MPS peak가
없어 descriptive evidence로만 보고한다.

## 6. 방법론과 논문 가치의 현실적 평가

방법론은 강하지만 `흠잡을 데가 거의 없다`는 평가는 과하다. 강점은 calibration-only
selection, physical-checkpoint lock, independent full-loss replay, document-cluster
inference, exact incremental correctness, five fresh timing sessions이다. 동시에 다음
한계가 남는다.

- 19.6M model, 128M training bytes의 compact setting
- 한 Apple MPS machine과 작은 session cluster 수
- historical development test와 일부 pre-lock confirmation의 존재
- local Git seal은 public preregistration이나 trusted execution이 아님
- final efficiency 비교는 strongest S가 아니라 quality-matched C86에 한정

따라서 compact speed가 통과하더라도 곧바로 top-tier scaling/general systems
claim이 되는 것은 아니다. ACL Findings/TMLR와 같은 집중된 실증 논문의 가능성은
있지만 venue는 결과·scaling·BPE/CUDA replication을 본 뒤 판단한다. 반대로 speed가
실패해도 quality/geometry 및 reproducibility paper는 가능하나, 그것을 이 연구의
최종 성공으로 간주하지 않는다.

## 7. S 우위 분해 실험의 보완 설계

리뷰어가 물을 핵심은 `S의 우위가 rate인지 boundary placement인지`다. 다만
`S를 72로 downsample`한 한 조건은 authentic SpaceByte가 아니고 thinning rule에
새 임의성이 생긴다. 더 해석 가능한 compact 후속은 다음 factorial이다.

1. C/W를 최소 72, 86, 약 153의 고정 rate에서 비교한다.
2. authentic S를 자연 rate `153.313`의 별도 점으로 유지한다.
3. 가능하면 각 sequence의 S patch count를 그대로 쓰되 boundary placement만
   C/W 규칙으로 재배치하는 per-example rate-matched control을 추가한다.
4. rate, policy, rate×policy interaction을 seed-paired BPB로 분석한다.
5. thinned S를 쓰면 `S72`가 아니라 `deterministically thinned-S72`로 명명하고
   thinning rule을 결과 전에 고정한다.

이 실험은 기전 논문을 강화하지만 실제 효율이라는 핵심 목표보다 우선하지 않는다.
v5r3 summary 뒤에 실행 여부를 결정한다.

## 8. 보완된 연구 진행 분기

### 지금 즉시

1. v5r3 gate, case, pair, statistic을 바꾸지 않는다.
2. repository drift로 무효화된 session 4를 evidence로 사용하지 않고 재실행한다.
3. session 4–5, role-isolated memory 10 units, immutable summary를 끝낸다.
4. 기존 Phase 3 total-cost summary는 provenance를 다시 검증해 diagnostic
   Main Table 3로만 승격한다.

### A — controlled와 free-running 모두 v5r3 통과

1. compact within-family actual-efficiency positive로 선언한다.
2. 50M/75M/100M family-aware preflight로 가장 큰 Mac-feasible scale을 고정한다.
3. candidate/raw/BPE16K/BPE32K, Korean downstream, learning curve를 수행한다.
4. 별도 CUDA protocol로 exact workload replication을 검토한다.
5. S rate×placement 분해를 mechanism extension으로 추가한다.

### B — 방향은 양수지만 10% gate 또는 한 co-primary mode 실패

1. primary actual-efficiency 결과는 실패로 기록하고 threshold를 낮추지 않는다.
2. summary의 TTFT/decode, mode, seed, role-order sensitivity와 새 profiler에서 local
   byte path, global trunk, cache, synchronization 비중을 분해한다.
3. 동일 shape의 19.6M/50M/75M/100M inference-only systems preflight로 global
   share가 커질 때 효과가 증가하는지 탐색한다. 이는 exploratory이며 quality
   evidence가 아니다.
4. 병목이 byte-sequential local path이면 whitespace routing만 미세 조정하지 않고
   multi-byte/block decoding 또는 local self-speculation을 결합한 새 architecture를
   사전 고정한다.
5. 새 구조에서 다시 matched-quality actual inference gate를 통과해야 핵심 목표를
   달성한 것으로 본다.

### C — 속도 방향도 불안정하거나 음수

1. W72를 speed technique으로 홍보하지 않는다.
2. compact quality/geometry 결과는 negative systems result와 함께 보존한다.
3. S 분해보다 runtime architecture redesign을 우선한다.
4. multi-byte output, speculative local decoder, 더 큰 global/local compute ratio 중
   실제 병목을 직접 줄이는 가설만 다음 단계로 연다.

이 분기는 `속도 실패 → 논문 종료`도 아니고 `속도 실패 → 품질 논문으로 성공 선언`도
아니다. 결과가 요구하는 만큼만 연구 방향을 바꾸면서 실제 추론 효율이라는 원래
성공 기준을 유지한다.

## 9. 외부 검토 문서의 사실 정정

검토 문서가 추가된 시점에 실행 중이던 session 4는 결과 publish 직전
`repository changed during actual timing session`으로 폐기됐다. 문서가 모델 계산이나
타이밍 값을 바꾼 것은 아니지만 clean-worktree provenance gate를 작동시켰으므로
`진행 중인 세션 4를 방해하지 않았다`는 문장은 운영적으로 사실이 아니다. 타이밍 및
output artifact는 하나도 publish되지 않았고 latency 값도 열지 않았다.

또한 session 1–3은 모두 correctness gate를 통과했지만 `tolerance-tie 0건`은 틀리다.
각 session의 free-path receipt에 동일한 허용된
`main_parallel_tolerance_tie_argmax_comparisons=1`이 기록되어 있다. 이는 v5r3의
사전 고정 tie-safe contract 안의 pass이고 speed 방향을 노출하지 않는다.

## 10. 관련 1차 문헌

- [SpaceByte: Towards Deleting Tokenization from Large Language Modeling](https://arxiv.org/abs/2404.14408)
- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)
- [From Bytes to Ideas: Language Modeling with Autoregressive U-Nets](https://arxiv.org/abs/2506.14761)
- [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955)
- [ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer](https://arxiv.org/abs/2603.03583)
- [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [Fast and Expressive Multi-Byte Prediction with Probabilistic Circuits](https://arxiv.org/abs/2511.11346)

## 11. 최종 판정

Fable 5 검토는 연구의 유효성, 강한 quality evidence, actual-speed risk, S-rate
confounding을 정확히 짚었다. 따라서 방향성 검토 자료로 수용할 가치가 있다. 다만
analytical compute와 actual efficiency의 구분, memory의 미결 상태, exact position/FLOP
분모, 핵심 성공 기준을 수정해야 한다.

현재 연구 계획은 즉시 갈아엎지 않는다. 먼저 봉인된 v5r3를 끝낸다. 결과가 실패를
가리킬 때에만 그 실패 양상에 맞춰 scale preflight 또는 multi-byte/speculative
architecture로 전환한다. 이것이 결과를 존중하면서도 불필요한 계획 변경을 피하는
가장 타당한 방향이다.

## 12. v5r3 개봉 후 후속 판정

다섯 세션을 끝내고 immutable summary를 commit한 뒤 결과를 열었다. Controlled E2E는
2.628% 감소(95% CI [2.026%, 3.526%]), strict-valid free-running E2E는 2.531%
감소([1.687%, 3.127%])했다. 두 mode 모두 5/5 session과 5/5 seed에서 양수였지만
10%에 도달한 session은 0/5였으므로 사전 고정 gate는 실패했다.

따라서 이 문서의 **B 분기**를 채택한다. Fable 5의 핵심 위험 진단은 결과로 지지됐지만,
효과가 0이라는 해석은 틀리다. W72는 global patch event 43개를 36개로 줄였으나
controlled decode의 127개 local byte step은 그대로 유지했다. 다음 연구는 W72의
scale-up이 아니라 component profiling 뒤 multi-byte/block generation 또는 local
self-speculation으로 순차 byte step 자체를 줄이는 방향이다. 상세 수치와 claim 경계는
`docs/91-v5r3-actual-inference-result-and-research-pivot.md`를 따른다.

## 13. Component profile 이후 최종 보완

2×2 checkpoint×schedule profile은 B 분기의 병목 가설을 직접 지지했다. 같은 candidate
weights에서 W72는 C86 schedule보다 decode를 2.852%, 같은 reference weights에서는
2.842% 줄였고 두 contrast 모두 5/5 seed가 양수였다. C86의 22개와 W72의 18개
decode-new patch 차이에 synchronized boundary increment 약 2.54ms를 곱한 10.16ms가
실제 same-checkpoint decode gap 10.12--10.14ms와 거의 일치했다. 양쪽에 공통인 local
byte step은 약 2.36ms로 127번 남았다.

따라서 Fable 5의 `local path가 이득을 삼킬 수 있다`는 진단은 수용하되, synchronized
component 합을 production share나 엄밀한 상한으로 바꾸지 않는다. Whole-trial 2×2
교차가 schedule effect를 식별하고 synchronized timing은 그 경로를 설명하는
diagnostic이다.

후속 방향도 한 단계 더 좁힌다. Generic multi-byte prediction, Medusa head, BLT
self-speculation, dependence-aware joint byte-MTP는 이미 선행연구가 있다. 다음 후보는
Hangul composition과 UTF-8 scalar alignment가 같은-cost generic byte-MTP보다 draft
acceptance와 실제 wall time을 추가 개선하는지를 검증해야 한다. 상세 architecture,
comparator, kill rule은
`docs/93-exploratory-component-profile-result-and-architecture-decision.md`를 따른다.

## 14. Learned-draft 결과에 따른 추가 판정

Parameter-matched frozen-W72 draft preflight에서 generic independent UTF-8가 complete pair
24.379%로 가장 높았고 Hangul conditional은 17.702%였다. 네 architecture 모두 사전
acceptance/cost gate를 실패했다. 따라서 Fable 5 검토에서 출발한 “한국어 구조를 넣으면
local sequential bottleneck도 싸게 줄일 수 있다”는 구체 가설 중 **Jamo/composition
factorization 부분은 지지되지 않았다**. Threshold를 낮추거나 head를 사후 튜닝해 살리지
않는다.

다만 exact speculative verifier는 mismatch에서도 correction byte를 확정하므로, 단순
accepted-suffix 수만으로 target-call speed를 판정한 preflight cost model은 불완전했다.
가장 강한 independent head의 관측값은 verifier당 기대 2.667522 bytes에 해당한다. 따라서
연구 방향을 불필요하게 다시 넓히지 않고, draft와 무관한 perfect target block-kernel의
exact wall-time upper-bound 한 번만 추가한다. 이 upper-bound가 실패하면 multi-byte branch도
종료한다. 상세 결과는 `docs/97-hangul-draft-acceptance-result-and-cost-model-correction.md`에
기록한다.

## 15. Exact speculative runtime 이후의 최종 사후검증

Perfect-draft block kernel은 inference mode에서 target block 자체를 정확하게 실행하며
perfect-Hangul whole path를 44.044% 줄였다. 그러나 실제 frozen head, correction,
rollback, retry, strict UTF-8 masking을 모두 넣은 exact speculative W72는 128/128
prompt에서 baseline output과 cache를 재현하면서도 E2E 감소가 **9.983%**
(prompt-bootstrap 95% interval **[7.579%, 11.695%]**)에 그쳤다. 110/128 prompt에서
방향은 양수였지만 사전 고정한 point 20% 및 lower-bound 10% gate를 모두 실패했다.

이 결과는 Fable 5의 중심 위험 진단을 두 번 지지한다.

1. W72 patch schedule만 바꾼 v5r3는 실제 E2E를 2.5--2.6% 줄였지만 10% gate를
   실패했다.
2. 순차 target invocation을 23.233% 줄인 exact speculative runtime도 모든 overhead를
   넣으면 9.983%에 머물렀다.

다만 이 사후 결과도 Fable 5의 `16.3% x global share` 식을 이론적 상한으로 만들지는
않는다. Perfect block 경로의 44.044% 감소가 보여주듯 block execution, correction,
kernel shape와 amortization은 단순 global-position 식 밖에 있다. 그 식은 W72 schedule
변경의 위험을 잘 짚은 heuristic이었고, architecture 전체의 상한은 아니었다.

사전 stop rule에 따라 learned multi-byte/draft 분기는 여기서 종료한다. 9.983%를 보고
head, activation, retry 또는 threshold를 다시 고르지 않는다.

## 16. 외부 검토의 최종 수용 범위

시간이 지난 뒤 실제 결과까지 포함해 보면 외부 검토는 **방향은 상당히 정확했지만,
증거와 전략의 몇 지점은 과도하게 단정했다**고 평가하는 것이 맞다.

| 항목 | 최종 판정 | 현재 근거 |
|---|---|---|
| 연구 질문이 살아 있음 | 수용 | BLT·SpaceByte 이후에도 byte hierarchy와 generation 병목은 활발한 문제다. |
| W72 quality 결과가 논문 가치 있음 | 수용 | C86 noninferiority와 C72 superiority가 sealed final 5 seeds에서 재현됐다. |
| actual speed 위험 | 강하게 수용 | v5r3 2.5--2.6%, exact speculative 9.983%로 두 번 관측됐다. |
| `컴퓨트 개선 확립` | 표현 교정 후 수용 | 16.28% data-patch 및 8.33% counted dense-matmul workload 감소다. 포괄적 compute 또는 speed 개선이 아니다. |
| `메모리 개선 없음` | 결과 이후에만 수용 | 당시에는 성급했다. 이후 role-isolated v5r3에서 parameter/MPS increment가 같고 RSS 방향도 섞였다. |
| 방법론적으로 거의 흠이 없음 | 부분 수용 | 최종 evidence chain은 강하지만, 실제 진행 중 selection·resume·runtime provenance의 여러 결함을 추가 감사와 수정으로 닫았다. 제한된 문서 검토만으로 무결성을 인증할 수는 없다. |
| speed 실패여도 작은 논문 완성 | 학술적 fallback만 수용 | negative/diagnostic paper는 가능하나 사용자가 정한 성공 기준은 실제 추론 효율이므로 연구 종료 조건은 아니다. |
| S rate/placement 분해가 다음 최우선 | 과학적으로 수용, 우선순위는 기각 | reviewer 질문에는 답하지만 실제 병목을 줄이지 않는다. 현재는 효율 candidate 뒤의 mechanism extension이다. |
| total-cost Pareto 완결 | 수용 | compact paper 표에 필요하되 teacher-forced diagnostic으로만 둔다. |
| 즉시 CUDA로 확대 | 현 시점 기각 | compact primary gate와 speculative gate가 모두 실패했다. 새 candidate가 실제 효율을 통과한 뒤에만 scale/hardware replication을 연다. |

특히 `S가 왜 강한가`는 미해결이지만 이미 W72--C72가 rate 72에서 boundary placement
효과를 식별한다. S 자체를 72로 억지 downsample하면 authentic SpaceByte가 아니며 새로운
thinning confound가 생긴다. 후속으로 수행한다면 C/W rate grid, authentic S, per-example
rate-matched placement control을 함께 써야 한다.

## 17. 연구 방향의 필요한 수정

Fable 5가 짚지 못한 가장 중요한 점은 **단순한 local-to-global capacity 이동 자체도
새 아이디어가 아니라는 것**이다. [BLT 원 논문](https://arxiv.org/abs/2412.09871)은
patch가 길어져 global Transformer가
덜 호출될 때 그 절약분으로 global model을 키우는 scaling axis를 명시적으로 제안하고
실험한다. 따라서 local encoder/decoder를 얇게 하고 global trunk를 키우는 정적 geometry는
다음 가설의 타당성을 검사하는 control일 수는 있지만 JamoFlow의 최종 novelty가 될 수 없다.

그럼에도 이 control은 필요하다. 현재 19.6M compact graph의 병목이 연구 아이디어가 아니라
부적절한 local/global allocation에서 생겼는지를 가장 싸게 반증할 수 있기 때문이다. 실제
HF graph를 이용한 비봉인 construction check에서도 local width/depth를 줄이고 global
capacity를 늘리면서 약 19.6M parameter 근처를 유지하고 counted dense FLOPs를 크게 줄이는
geometry가 존재한다. 이 값은 후보 선택 결과가 아니라 protocol 설계를 위한 feasibility
확인에만 사용한다.

보완된 순서는 다음과 같다.

1. 현재 exact speculative 9.983%/gate-fail 결과를 그대로 고정하고 multi-byte branch를
   종료한다.
2. 정적 local-to-global geometry를 **novel method가 아닌 generic control/falsification**으로
   사전 고정한다. Random-weight actual timing은 latency potential만 평가하며 quality
   evidence로 쓰지 않는다.
3. 최소 20% actual-latency potential이 없으면 이 geometry branch를 즉시 닫는다. 있으면
   한 seed를 같은 Korean train/calibration budget으로 학습해 W72 baseline 대비 BPB
   noninferiority와 actual latency를 함께 본다.
4. 정적 geometry가 통과해도 그 결과만으로 새 architecture paper라고 부르지 않는다.
   다음 주가설은 prefix에서 이미 알려진 UTF-8/Hangul composition state를 이용해 쉬운
   continuation 위치의 **local compute depth를 조건부로 줄이는 것**이다. Future byte를
   미리 맞히지 않으므로 실패한 speculative head와 다른 가설이다.
5. 필수 대조는 original W72, parameter-matched 정적 thin-local control, generic UTF-8
   state-conditioned control, Hangul-specific state-conditioned candidate다. 한국어 고유
   주장은 Hangul candidate가 같은-cost generic UTF-8 control을 넘고 Korean-vs-control
   script interaction이 재현될 때만 허용한다.
6. 새 compact candidate가 matched quality에서 사전 고정 actual E2E gate를 통과한 뒤에만
   50M/75M/100M, BPE16K/32K, downstream, S rate-by-placement, CUDA replication을 연다.

[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)가 token별 learned depth allocation을
이미 제안했으므로 `conditional depth` 일반도 novelty가 아니다. 새 기여가 성립하려면
**학습 라우터 없이 prefix에서 완전히 결정되는 orthographic state**, **BLT의 반복 local-byte
module에 대한 적용**, **generic UTF-8와 Hangul-specific route의 같은-cost 인과 대비**,
**matched-quality actual generation**이 함께 필요하다. 이 네 조건 중 하나라도 빠지면
결과는 기존 BLT/Mixture-of-Depths 계열의 Korean replication 또는 engineering ablation으로
범위를 낮춘다.

이 수정은 Fable 5의 제안을 버리는 것이 아니다. 가장 정확했던 경고인 `global event만
줄여서는 local byte bottleneck을 못 없앤다`를 실제 결과까지 반영해 한 단계 더 밀어붙인
것이다. 반대로 S 분해, negative-paper 골격, total-cost 표는 보존하되 핵심 효율 후보보다
앞세우지 않는다.

현재 논문에 안전하게 쓸 수 있는 이야기는 다음까지다.

> Korean BLT에서 whitespace-aware boundary placement는 같은 patch rate의 codepoint grid보다
> 품질이 좋고, 더 촘촘한 C86과 matched quality에서 global workload와 실제 latency를
> 일관되게 조금 줄였다. 그러나 patch scheduling만으로는 per-byte local bottleneck을
> 제거하지 못했고, Hangul-specific drafts는 generic control보다 약했으며, exact
> speculation도 9.983%에서 사전 gate를 실패했다.

이는 정직하고 유용한 negative/diagnostic contribution이지만 아직 최종 성공은 아니다.
논문 가치가 높은 positive 결론은 새 conditional-local-compute 후보가 정적 generic control을
넘어 실제 E2E 효율을 재현할 때 생긴다.

## 18. 정적 geometry falsification 결과

사전 봉인한 random-weight actual-timing control에서 `thin160 E1/D1 + global 384x9`가
original W72 대비 parameter 차이 -0.124%, counted dense FLOPs -31.047%를 유지하면서
controlled E2E latency를 **24.417%** 줄였다. Prompt bootstrap 95% interval은
**[19.202%, 29.112%]**, 방향은 32/32 prompt에서 양수였다. 따라서 고정 gate에 따라 이
geometry의 Korean 한 seed 학습을 허가한다.

이 결과는 Fable 5의 경고를 부정하지 않는다. 기존 W72가 줄인 것은 global patch event뿐이라
2.5--2.6%에 머물렀지만, 이번 control은 매 byte 반복되는 local encoder/decoder의 폭과
깊이를 직접 줄였다. 즉 `global event 절감만으로는 부족하다`는 진단을 더 직접적인
architecture allocation으로 해결할 latency potential이 확인된 것이다.

동시에 가장 얇고 counted FLOPs가 가장 적은 thin128 E1/D1은 19.610%로 point gate를
실패했다. MPS wall time이 FLOP 수에 단조롭지 않으므로 이후 geometry 선택은 analytical
compute가 아니라 실제 runtime을 함께 보아야 한다. 고정 순서의 첫 통과 후보만 학습하며,
근접 실패 후보를 사후 추가하지 않는다.

상세 판정과 claim 경계는
`docs/105-static-local-global-geometry-result.md`를 따른다. 아직 random-weight feasibility이므로
quality 또는 publication efficiency 결론은 아니다.

## 19. 학습된 정적 geometry 결과와 조건부 계산 피벗

봉인된 one-seed screen에서 `thin160 E1/D1 + global 384x9`는 controlled E2E를
**24.307%**(prompt bootstrap 95% **[23.770%, 24.630%]**), strict free-running E2E를
**22.841%**(**[22.370%, 23.164%]**) 줄였다. 두 mode 모두 64/64 prompt에서 양수였고
incremental correctness도 통과했다. 즉 random-weight preflight의 latency potential은
학습된 checkpoint에서도 거의 그대로 재현됐다.

그러나 Korean calibration BPB는 W72보다 **0.095601** 나빠졌고 document-bootstrap
one-sided 95% upper도 **0.096740**이었다. 사전 margin 0.010을 명확하게 실패했으므로 이
모델은 사용자가 요구한 matched-quality 효율 개선이 아니다. 정적 geometry의 나머지 네
seed를 학습하지 않고 해당 branch를 종료한다.

이 결과로 계획을 필요한 만큼만 수정한다. Local path를 줄이는 시스템 목표는 맞았지만,
모든 위치에서 width/depth를 제거하는 방식은 품질을 훼손했다. 따라서 다음 실험은 original
local capacity를 hard state에서 보존하고 easy UTF-8/Hangul state에서만 compute를 줄이는
position-conditional local path다. 이는 failed static result를 성공으로 재해석하는 것이
아니며, 새 prospective feasibility·correctness·quality gate를 별도로 통과해야 한다.

정적 모델은 이후 논문의 matched-quality comparator가 아니라 speed--quality trade-off를
보여 주는 negative control로만 남긴다. Conditional candidate도 같은-cost generic UTF-8
control을 이기지 못하면 한국어 고유 기여를 주장하지 않는다. 상세 수치와 다음 단계의
claim 경계는 `docs/107-static-geometry-one-seed-result-and-pivot.md`를 따른다.

## 20. 현 conditional protocol에 적용한 추가 교정

Fable 5가 S에 대해 제기한 `rate와 placement를 분리하라`는 원칙은 현재 conditional-local
후보에도 적용해야 한다. 다만 모델 결과를 열기 전 route mask만 계산한 결과, 8MB Korean
calibration stream에서 generic `utf8_incomplete`는 58.3054875%, `hangul_prefix`는
57.5361125%를 easy로 분류했다. Hangul mask는 generic mask의 98.6804415%를 차지한다.

따라서 현재 2×2×2 frozen screen은 계산 노출이 거의 맞는 두 route의 compatibility를 함께
보는 데는 타당하지만, Hangul-specific superiority를 식별하지는 못한다. 두 처치가 거의
같기 때문이다. 이 점을 반영해 다음 경계를 고정한다.

1. frozen failure는 dense W72 checkpoint의 perturbation risk이며, 처음부터 conditional
   graph로 학습한 방법 일반의 기각이 아니다.
2. screen에 노출된 8MB calibration stream은 후속 trained candidate의 confirmatory quality
   평가에 재사용하지 않는다. 한-seed training 전에 disjoint Korean validation을 봉인한다.
3. 거의 겹치는 UTF-8/Hangul mask만으로 한국어 고유 기여를 주장하지 않는다. 같은 계산량의
   generic control과 Hangul/non-Hangul coverage interaction 또는 script-stratified evidence가
   추가로 필요하다.
4. 그 증거가 없으면 positive 범위는 `Korean data에서 검증된 generic UTF-8 structural
   conditional compute`로 낮춘다. 이는 여전히 한국어 추론 효율 연구이지만 Hangul 고유
   architecture claim은 아니다.

이 보완은 conditional frozen screen 자체를 폐기하지 않는다. Static result가 보여 준
`local path는 큰 latency 병목이면서 quality-critical하다`는 양면성을 가장 싼 단계에서
검사하는 목적은 유지한다. 다만 frozen pass/fail과 Korean-specific novelty를 서로 다른
질문으로 분리한다.

## 21. Conditional frozen screen 결과와 최종 수용 판정

보완된 2×2×2 screen의 여덟 candidate는 모두 실패했다. 가장 덜 손상된
`hangul_prefix / decoder / second_mlp`도 W72 대비 +0.198832 BPB, document one-sided upper
+0.199967로 +0.020 risk margin에서 멀었다. 따라서 현재 candidate 집합의 runtime
prototype은 열지 않는다.

이 결과는 Fable 5의 위험 진단을 더 일반적인 원칙으로 확장한다. 규칙이 후보 공간을
제한한다는 사실과 그 위치의 neural representation이 불필요하다는 주장은 동일하지 않다.
UTF-8/Hangul continuation은 문법적으로 제한되어도 어느 scalar인지를 결정하는 정보 비트를
담고 있었고, frozen W72는 local layer에 크게 의존했다.

따라서 다음 효율 가설은 그 정보를 삭제하지 않고 한 autoregressive decision에 묶는
reversible scalar representation이다. 다만 Korean three-hot 및 alternative byte encoding
선행연구와의 중복이 크므로, 새 학습보다 먼저 generic Unicode-scalar control, Hangul
factorization, BPE의 step/parameter/head-cost frontier와 component dependence를 감사한다.
이론적 이점이 generic control에도 남지 않으면 더 모델을 학습하지 않고 negative systems
study로 정리한다. 상세 결과와 경계는
`docs/109-conditional-local-frozen-sensitivity-result-and-pivot.md`를 따른다. 그 다음
model-free scalar/BPE audit의 exact representation, known-anchor disclosure, gate는
`docs/110-scalar-representation-and-bpe-opportunity-protocol.md`에 결과 전에 고정했다.

## 22. Scalar/BPE audit 결과에 따른 비교 시점 수정

고정 audit에서 generic scalar와 Hangul hybrid는 raw byte step을 각각 58.363%, 57.593%
줄였고 W72 대비 counted dense-matmul opportunity는 36.622%, 36.252%였다. 모든 가역성,
vocabulary/OOV 및 opportunity gate가 통과해 random-weight construction은 허가된다.

그러나 train-only reversible ByteLevel BPE16K와 BPE32K는 같은 calibration prefix를 각각
1.534M, 1.389M token으로 표현했다. Scalar/hybrid 3.331M/3.393M보다 훨씬 짧다. 따라서
scalar 방향의 유효한 주장은 `BPE보다 sequence가 짧다`가 아니다. 작은 conditional head와
BLT local/global hierarchy가 BPE의 짧은 token sequence를 parameter-matched actual runtime에서
상쇄하는지가 핵심이다.

이에 따라 BPE 비교를 publication-scale 마지막으로 미루지 않고 바로 다음 random-weight
runtime에 포함한다. Hybrid는 generic보다 step이 1.849% 많지만 resident output rows가
448에서 324로 작다. 한국어 고유 효과는 이 head/capacity 차이가 generic scalar보다 더 나은
actual speed--quality frontier로 이어질 때만 인정한다. 상세 결과는
`docs/111-scalar-representation-opportunity-result.md`를 따른다.

## 23. Parameter-matched runtime 질문의 사전 고정

다음 MPS preflight는 결과 전에 다섯 graph를 W72 19.596M parameters의 ±0.25%로 맞춘다.

- byte W72 19,596,096
- generic conditional scalar BLT 19,632,960
- Hangul conditional hybrid BLT 19,609,152
- BPE32K tied Transformer 19,593,984
- BPE16K tied Transformer 19,595,200

Random weights에서는 model-generated Korean이 의미 없으므로 free-running 성공을 흉내 내지
않는다. 대신 같은 128-byte controlled continuation에서 target route와 scalar/token 길이만
고정하고, 실제 output projection, device-side argmax, conditional dependency, cache update와
synchronization을 실행한다. 이 단계는 graph feasibility만 고른다.

Scalar 후보는 W72 대비 E2E median 10% 이상, bootstrap lower>0, 28/32 prompt positive인
동시에 BPE16K와 BPE32K 각각보다 95% lower가 -10% 아래로 떨어지지 않아야 one-seed quality
학습으로 간다. Hybrid는 generic 대비 lower>=-5%도 만족해야 Hangul-specific branch로 남는다.
정확한 case-selection, correctness와 claim 경계는
`docs/112-scalar-runtime-preflight-protocol.md`를 따른다.

## 24. Scalar runtime 결과와 BLT branch 종료

다섯 parameter-matched graph의 MPS 결과에서 generic scalar와 Hangul hybrid는 byte W72보다
각각 46.843%, 42.540% 빨랐다. 이는 127개의 순차 byte-local update가 실제 병목이라는 앞선
component 진단을 강하게 확인한다.

그러나 generic은 BPE32K보다 105.985%, BPE16K보다 181.243% 느렸고, hybrid는 각각
122.659%, 204.009% 느렸다. Hybrid 자체도 generic보다 8.095% 느려 Korean-specific runtime
gate를 실패했다. Correctness와 ±0.25% parameter match는 모두 통과했으므로 결과는 graph
bug나 obvious capacity mismatch로 설명되지 않는다.

사전 gate에 따라 승인된 scalar 후보는 없으며 이 branch를 학습하지 않는다. 다음 연구는
강한 token Transformer를 기준으로 Korean-aware reversible tokenizer가 byte BPE보다 더 나은
sequence/vocabulary/head frontier를 제공하는지 model-free로 먼저 본다. Generic scalar-aware
BPE를 필수 control로 두고, 실제 Korean-specific 추가 이득이 없으면 한국어 고유 architecture
claim을 하지 않는다. 상세 수치와 claim 경계는
`docs/113-scalar-runtime-preflight-result-and-token-frontier-pivot.md`를 따른다.
