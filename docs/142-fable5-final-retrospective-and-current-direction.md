# Fable 5 중간 검토의 최종 사후검증과 현재 연구 방향

> 작성일: 2026-08-15
>
> 대상: [`fable5-연구-중간-검토.md`](../fable5-%EC%97%B0%EA%B5%AC-%EC%A4%91%EA%B0%84-%EA%B2%80%ED%86%A0.md)
>
> 상태: v5r3 actual inference부터 foldable-Jamo residual까지의 판정; 이후 mechanism-control
> 결과는 `docs/150`이 대체

> **2026-08-15 후속 교정:** `docs/150`의 봉인 실험에서 generic multi-hash는
> `update_matched_dense`보다 약 `0.008 BPB` 나빠 mechanism guard를 실패했다. 따라서 이 문서 아래의
> foldable multi-hash 양성 가능성과 fresh-stage 실행 순서는 더 이상 current plan이 아니다.
> Multi-hash/Jamo branch는 종료하고 ordinary dense new-row optimization만 별도 신규 가설로 남긴다.

## 결론

Fable 5 검토는 당시 연구의 가장 큰 위험을 정확하게 짚었다. 분석적 patch/FLOP 절감은 실제
batch-1 end-to-end 속도와 다르며, W72가 줄이지 못한 per-byte local path가 10% gate를 막을 수
있다는 경고는 이후 실측으로 지지됐다. 따라서 이 문서는 단순한 외부 의견이 아니라 연구 방향을
교정한 유효한 사후 예측으로 보존할 가치가 있다.

그러나 모든 제안을 현재 우선순위로 받아들이는 것은 잘못이다. `속도 실패여도 작은 논문으로
종료`, `S rate/placement 분해를 다음 최우선으로 실행`, `CUDA로 즉시 확대`는 사용자가 정한
성공 기준과 후속 병목 증거에 맞지 않는다. 현재는 W72를 더 세밀하게 설명하는 것보다, 품질을
유지하면서 실제 token step을 줄일 수 있는 새 compact candidate를 먼저 검증해야 한다.

최종 판정은 다음과 같다.

1. Fable의 핵심 systems 경고와 인과분리 원칙은 수용한다.
2. W72는 작은 actual speed effect를 가진 primary-negative boundary result로 고정한다.
3. S 분해와 total-cost 표는 boundary manuscript의 보조 과제로 남기되 새 efficiency candidate보다
   앞세우지 않는다.
4. 당시 current foldable generic residual은 Jamo 방법의 fallback이 아니라 별도
   `training-time foldable vocabulary adaptation` 가설로 분리했다.
5. 이후 이 가설은 optimizer/self-update control을 실패했다. 현재 남은 candidate는 hash가 아니라
   별도 ordinary dense new-row optimization이며, fresh matched quality와 trained actual E2E 10%
   gate를 새 protocol에서 통과해야 연구 성공이다.

## 당시 문서에서 정확했던 것

### 1. analytical workload와 actual latency를 구분한 점

W72는 C86보다 data patch를 16.28%, 계수된 dense-matmul workload를 8.33% 줄였다. 하지만
다섯-session actual inference에서 controlled E2E는 2.628%, strict-valid free-running은 2.531%만
줄었고, 사전 고정 10% gate는 0/5 session으로 실패했다. Fable의 위험 진단은 방향뿐 아니라
실제 연구 의사결정에도 유효했다.

다만 Fable 문서의 `16.3% × global trunk share`는 이론적 상한이 아니다. 이후 perfect block
kernel은 전혀 다른 실행구조에서 44.044%를 줄였고, exact speculative path는 overhead를 포함해
9.983%에 머물렀다. 이 식은 W72 schedule에 대한 유용한 heuristic이지 architecture 전체의
upper bound가 아니다.

### 2. local path 병목을 의심한 점

2×2 checkpoint×schedule profile에서 네 번의 decode boundary update 제거가 약 10.1ms를
절약했고, 두 checkpoint에서 같은 schedule effect가 재현됐다. 반면 127개의 local byte update는
그대로 남았다. W72의 small speed effect가 품질 문제가 아니라 execution structure의 문제라는
점을 식별했다.

후속 실험도 같은 결론을 강화했다.

- static thin-local geometry는 trained E2E를 22.8--24.3% 줄였지만 BPB가 0.0956 나빠졌다.
- conditional local skip의 여덟 frozen candidate는 모두 큰 quality 손상을 냈다.
- generic/Hangul scalar BLT는 byte W72보다 빨랐지만 parameter-matched BPE보다 훨씬 느렸다.

즉 local path는 큰 비용이면서 동시에 quality-critical했다. 단순히 규칙상 쉬운 위치라고 neural
compute가 불필요한 것은 아니었다.

### 3. rate와 placement를 분리하라고 한 점

이 원칙은 W72--C72 same-rate 비교에서 이미 핵심 기여를 만들었고, 이후 모든 Korean-specific
후보에도 적용됐다. True Hangul/Jamo assignment는 항상 같은-cost generic 및 shuffled control과
비교했다. 그 결과 Hangul/Jamo 신호가 shuffle보다 작게 좋은 경우에도 generic control을 넘지
못하면 Korean-specific claim을 중단할 수 있었다.

### 4. 작은 compact 결과를 과장하지 말라고 한 점

19.6M/MPS 결과를 CUDA, production serving, general hardware 또는 대형 LLM에 일반화하지 않는
경계는 유지됐다. 후속 random-weight opportunity나 one-seed development pass도 final claim으로
승격하지 않았다. 이 원칙은 현재 foldable multi-hash 결과에도 그대로 적용한다.

## 수정해서 받아들여야 했던 것

### `컴퓨트 개선 확립`

포괄적 compute improvement가 아니라 아래 세 항목으로 분리해야 한다.

- data patch count 감소
- 구현 그래프에서 사전 정의한 dense-matmul FLOP 감소
- 실측 end-to-end latency 감소

W72는 앞의 두 항목과 작은 세 번째 항목을 보였지만 primary 10% efficiency gate는 실패했다.

### `메모리 개선 없음`

측정 전에는 성급한 판단이었다. 이후 role-isolated 결과에서 persistent parameter와 MPS increment는
사실상 같았고 RSS 방향도 섞였다. 그러므로 최종적으로는 `memory improvement 없음`이 맞지만,
그 결론은 parameter count만이 아니라 실제 측정 뒤에만 허용된다.

### `방법론적으로 흠잡을 데가 거의 없음`

최종 evidence chain은 강하지만 진행 중 selection, resume, router, final-test, actual-runtime
provenance에서 여러 결함이 발견되어 수정됐다. 제한된 결과 문서와 receipt만 읽은 외부 검토가
코드 무결성까지 인증할 수는 없다. 정확한 평가는 `좋은 causal design과 강한 최종 재현 계약을
가졌지만, 그 계약은 반복적인 adversarial audit로 완성됐다`이다.

## 현재는 우선하지 않는 제안

### S rate×placement 분해

과학적으로 타당하고 boundary paper의 reviewer question에 답한다. 다만 authentic S의 자연 rate는
약 153이고 W72와 크게 다르며, S를 72로 downsample하면 새 thinning confound가 생긴다. 실행한다면
C/W rate grid, authentic S, per-example rate-matched placement를 함께 써야 한다.

현재는 우선하지 않는다. W72--C72가 rate 72의 placement effect를 이미 식별했고, S 분해는 실제
local/token-step 병목을 고치지 않는다. 새 compact candidate가 actual 10%를 통과하거나 boundary
manuscript를 제출 직전 완결할 때 실행한다.

### 즉시 CUDA 또는 larger-scale 확대

W72와 speculative branch가 primary gate를 실패했으므로 확대는 정보 대비 비용이 나쁘다. 새
candidate가 Mac에서 matched-quality actual 10%를 재현한 뒤에만 CUDA와 50M 이상을 연다.

### speed 실패 후 작은 논문으로 종료

학술적 fallback으로는 가능하다. W72의 quality, same-rate placement, stable small latency, exhaustive
correctness와 negative systems result는 집중된 empirical paper가 될 수 있다. 그러나 사용자가
정한 성공 기준은 실제 효율이므로 이것을 연구 완료로 간주하지 않는다.

## 아직 수용해야 하는 미완 과제

Fable의 `total-cost Pareto 표를 완결하라`는 제안은 여전히 맞다. 현재 paper draft 5.5절은 아직
pending이다. 이는 W72를 positive speed method로 살리는 작업이 아니라, learned router와 authentic
S를 포함한 compact boundary study를 정직하게 닫는 보조 evidence다. 새 candidate 실험과 독립적으로
재구성할 수 있지만, boundary manuscript 제출 전에는 반드시 완료한다.

Scenario-B 논문 골격도 유지한다. 다만 최종 원고는 다음 두 트랙을 혼합하지 않는다.

1. **Boundary-placement diagnostic manuscript:** W72 same-rate quality, analytical workload,
   stable 2.5% actual effect, local bottleneck 및 실패한 10% gate.
2. **Foldable vocabulary-adaptation manuscript 후보:** optimizer-confound를 이긴 training-only
   reparameterization이 fresh matched quality를 보존하면서 dense 8K의 실제 token-step speed를
   회복하는 경우에만 별도 positive paper.

한 원고에 모든 후속 피벗을 넣으면 결과 후 탐색과 contribution drift가 과도해진다. 두 번째가
실패하면 첫 번째만 정리하고, 성공하면 development history는 supplement/limitations로 공개하되
main causal question은 분리한다.

## foldable multi-hash 결과에 적용한 Fable 원칙

최신 B1 결과에서 true Jamo residual은 matched shuffle보다 untied 0.000309, tied 0.000831 BPB
좋았지만 고정 0.002 minimum을 실패했다. 같은-cost generic multi-hash는 두 architecture 모두
Jamo보다 약 0.0013 BPB 좋고 no-residual base보다 0.0156--0.0255 BPB 좋았다. 따라서 Korean-specific
branch는 종료했다.

Generic result에는 다음 confound가 있다.

- 13개의 zero-initialized residual branch가 AdamW에서 새 row의 effective update를 크게 바꾼다.
- high-entropy hash assignment는 semantic Unicode feature보다 low-interference multi-hash code에 가깝다.
- 추가 optimizer state와 collision-coupled gradient가 더 좋은 basin을 만들었을 수 있다.

Plain SGD의 `약 2× self update` 직관만으로 control을 고르면 부족하다. 첫 AdamW step에서는 gradient
scale normalization 때문에 collision-free 이상화의 residual contribution이 `sqrt(13)`배에 가까울 수
있고, bucket aggregate의 sign, clipping, weight decay와 이후 moment가 이를 바꾼다. 다음 단계는
loss를 더 보는 것이 아니라 실제 fixed first-batch update geometry를 먼저 감사하고, 그 결과로 단
하나의 optimizer-equivalent control을 고정해야 한다.

필수 최소 대조는 다음이다.

1. ordinary dense EEVE-transfer base
2. current 13-way foldable multi-hash
3. new-row effective update를 맞춘 dense/diagonal optimization control
4. surface dependence를 제거한 same-budget balanced random multi-hash control

Multi-hash가 3을 이기지 못하면 기여는 새로운 representation이 아니라 vocabulary adaptation용
optimization recipe로 낮춘다. 4와 차이가 없으면 `surface` 또는 Unicode semantic claim을 제거하고
generic hash reparameterization으로만 부른다. 이 mechanism guard를 통과하지 못하면 fresh-data
campaign을 열지 않는다.

## 현재 실행 결정

연구 계획을 불필요하게 되돌리지 않는다. 다음 순서를 채택한다.

1. vocabulary adaptation, hash embedding, training-time overparameterization의 최신 직계 선행을
   재검증한다.
2. 기존 dev batch에서 model-loss-free one-step AdamW update audit을 수행한다.
3. 분석적으로 고정한 optimizer-equivalent control과 same-budget random-hash control을 봉인한다.
4. 최소 one-seed mechanism screen에서 current multi-hash가 두 control을 의미 있게 이길 때만 fresh
   disjoint Korean equal-history 3-seed 이상으로 간다.
5. fresh quality-qualified folded dense-8K checkpoint만 dense-2K와 controlled/free actual inference로
   비교한다.
6. 두 mode 모두 point reduction 10% 이상과 uncertainty/seed/session stability를 통과한 뒤에만
   larger model, downstream, CUDA와 Hugging Face release를 연다.

## 최종 판정

Fable 5 검토는 수용 가치가 높았다. 특히 실제 speed 위험은 후속 결과가 강하게 확인했다. 하지만
그 문서를 현재 계획의 체크리스트로 기계적으로 적용해서는 안 된다. 가장 타당한 반영은 `proxy를
성공으로 부르지 않기`, `같은-cost generic control로 언어학적 기여를 분리하기`, `compact actual
gate 전에 scale하지 않기`라는 세 원칙을 유지하는 것이다.

현재 foldable multi-hash 방향은 이 원칙을 만족할 가능성이 있으므로 조사할 가치가 있다. 다만
문헌 중복과 optimizer confound를 이기기 전에는 새로운 Korean LLM 기법도, 논문 성공도 아니다.

## 2026-08-15 mechanism-control 이후 재검토

`docs/150`에서 multi-hash가 `update_matched_dense`보다 약 `0.008 BPB` 나빠 primary mechanism
guard를 실패했다. 따라서 위 마지막 문단의 multi-hash 양성 가능성은 이제 폐기됐고, Fable 원칙을
적용한 현재 판단은 다음과 같다.

1. Korean/Jamo/hash representation 분기는 더 탐색하지 않는다.
2. ordinary dense new-row optimization은 기존 candidate의 fallback이 아니라 별도 신규 가설이다.
3. 이미 알려진 B1 corpus의 양성 수치는 발견 근거일 뿐이므로, 기존 및 sealed-final 문서를 exact와
   normalized 형태로 제외한 fresh Korean train/calibration에서 다시 검증한다.
4. dense-2K 대비 matched-quality를 통과한 trained dense-8K checkpoint만 actual controlled/free
   inference로 보내며, 두 mode의 `>=10%` point gate를 유지한다.
5. 이 gate 전에는 S 분해, larger scale, CUDA, downstream 또는 Hugging Face 공개를 열지 않는다.

이 수정은 Fable 검토가 제안한 모든 후속을 채택한 것이 아니다. proxy를 실제 효율로 부르지 않고,
같은-cost/강한 optimization control을 두며, compact actual gate 전에 확대하지 않는 세 원칙만
현재 증거에 맞게 수용한 것이다. Fresh-data 계약은 `docs/151`이 지배한다.
