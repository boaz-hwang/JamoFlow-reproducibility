# Foldable-Jamo residual 결과와 multi-hash vocabulary-adaptation 전환

> 작성일: 2026-08-15
>
> 상태: Korean-specific B1 gate 실패로 Jamo residual branch 종료; 같은 배포 graph의 generic foldable residual은 별도 신규 가설로만 후속 검증

## 결론

봉인된 6-role B1 screen은 의도한 Korean-specific gate를 통과하지 못했다. `jamo` residual은 두
architecture에서 no-residual base보다 좋아졌고 `shuffled_jamo`보다도 통계적으로 일관되게
좋았지만, 효과가 사전 최저선 `0.002 BPB`에 미치지 못했다. 더 결정적으로 같은 training-only
parameter와 lookup budget을 쓴 `generic_surface`가 두 architecture 모두에서 `jamo`보다 좋았다.

따라서 다음 결론을 고정한다.

1. Jamo-specific foldable residual은 여기서 종료한다. 결과를 보고 threshold, slot, shuffle 또는
   checkpoint를 바꾸지 않는다.
2. 이번 결과는 “자모 정보가 전혀 없다”는 뜻은 아니다. True Jamo는 matched shuffle보다 untied
   `0.000309`, tied `0.000831 BPB` 좋았고 두 document bootstrap interval도 0 아래였다. 다만
   publication-worthy Korean-specific 최소 효과가 아니었다.
3. 반면 high-entropy generic residual은 untied base를 `0.015562`, tied base를 `0.025471 BPB`
   개선했다. Fold 뒤 배포 parameter, tokenizer, graph와 NLL은 architecture base와 정확히 같은
   형태이며 residual runtime은 0이다.
4. 이 generic 역할을 곧바로 Korean method라고 부르지 않는다. Auxiliary slot은 Unicode 의미
   표현이라기보다 domain-separated multi-hash에 가까워, training-time overparameterization 또는
   optimizer preconditioning 효과일 가능성이 크다.
5. 별도 신규 protocol에서 이 confound를 먼저 분리한 뒤, strongest foldable role만 fresh Korean
   equal-history quality와 trained batch-1 actual inference로 보낸다. 최종 성공 기준은 그대로 두
   co-primary E2E mode에서 matched-quality latency `>=10%`다.

## 봉인 결과

모든 수치는 step 512의 independently replayed float32 NLL에서 계산했다.

| architecture | no residual | generic surface | shuffled Jamo | true Jamo | Jamo−generic | Jamo−shuffle | Jamo gate |
|---|---:|---:|---:|---:|---:|---:|---|
| untied | 1.454530 | **1.438968** | 1.440638 | 1.440328 | +0.001360 | −0.000309 | fail |
| tied | 1.495260 | **1.469789** | 1.471952 | 1.471120 | +0.001331 | −0.000831 | fail |

차이는 왼쪽 역할에서 오른쪽 comparator를 뺀 contiguous BPB다. 음수이면 candidate가 더 좋다.

Document BPB도 같은 순서였다.

| architecture | no residual | generic surface | shuffled Jamo | true Jamo |
|---|---:|---:|---:|---:|
| untied | 1.453841 | **1.438181** | 1.439839 | 1.439528 |
| tied | 1.494595 | **1.469098** | 1.471222 | 1.470420 |

### 사전 gate별 판정

Untied `jamo`:

- base 대비 contiguous/document: `−0.014202 / −0.014312`, pass
- generic 대비 contiguous/document: `+0.001360 / +0.001348`, fail
- shuffle 대비 contiguous/document: `−0.000309 / −0.000311`, 효과 방향은 pass지만 `0.002` minimum fail
- generic document bootstrap 95% CI: `[+0.001244,+0.001446]`, Jamo가 유의하게 나쁨
- shuffle document bootstrap 95% CI: `[−0.000384,−0.000241]`, Jamo가 유의하게 좋지만 작음
- dense-2K anchor gap: `+0.010667`, pass

Tied `jamo`:

- base 대비 contiguous/document: `−0.024140 / −0.024175`, pass
- generic 대비 contiguous/document: `+0.001331 / +0.001321`, fail
- shuffle 대비 contiguous/document: `−0.000831 / −0.000802`, 효과 방향은 pass지만 `0.002` minimum fail
- generic document bootstrap 95% CI: `[+0.001198,+0.001436]`, Jamo가 유의하게 나쁨
- shuffle document bootstrap 95% CI: `[−0.000887,−0.000718]`, Jamo가 유의하게 좋지만 작음
- dense-2K anchor gap: `+0.041459`, pass

두 architecture 모두 같은 이유로 joint gate를 실패했다. Qualified Jamo role은 0개이고,
`fresh_equal_history_stage_authorized=false`다. 이 false는 기존 B1 plan 아래에서 generic 역할을
사후 fallback으로 승격할 수 없다는 뜻이다.

## 학습 곡선이 말하는 것

True Jamo는 초기에 generic보다 약간 빨랐지만 이 우위가 유지되지 않았다.

| architecture | step 32 Jamo−generic | step 128 | step 512 |
|---|---:|---:|---:|
| untied | −0.000188 | +0.001122 | +0.001360 |
| tied | −0.001852 | +0.000753 | +0.001331 |

Jamo 공유가 초기 최적화에 약한 inductive bias를 준 정황은 있다. 그러나 128 steps부터
high-entropy generic assignment가 앞섰다. True Jamo의 component entropy는 일부 slot에서 3–6 bits로
낮고 여러 token이 같은 component를 공유하지만, generic auxiliary slots는 거의 7-bit uniform이다.
현재 budget에서는 의미 있는 공유의 이득보다 collision/interference 또는 token-specific adaptation
capacity 손실이 더 컸다는 해석이 가장 직접적이다.

이는 `jamo`가 shuffle을 이긴 작은 차이와도 양립한다. 올바른 component alignment는 무작위 정렬보다
조금 낫지만, 더 독립적인 high-entropy hash path보다 낫지는 않았다.

## Generic 역할의 정확한 의미

`generic_surface`라는 기존 이름을 의미론적 Unicode 표현으로 과장하면 안 된다. Slots 0–5와 12는
first/last scalar digits와 byte length지만, 승부를 가른 slots 6–11은 domain-separated surface hash다.
이들은 실질적으로 13-way foldable multi-hash residual의 일부다.

Plain SGD 근사에서 token `i`의 effective row를 `e_i = w_i + A_i r`라 쓰면 한 update의 자기 항은
대략 다음과 같다.

`Δe_i ≈ −η (I + A_i A_i^T) g_i − η Σ_{j≠i} A_i A_j^T g_j`

13개 slot을 `1/sqrt(13)`로 합쳤으므로 자기 residual 경로의 squared norm은 1이다. Dense 경로까지
합치면 collision이 없는 자기 update는 plain SGD에서 약 `2×`가 된다. 다른 token과 code가 겹칠 때만
cross-token coupling이 추가된다. 실제 optimizer는 AdamW라 이 식이 exact하지 않지만 다음 confound를
명확히 제시한다.

- 효과가 surface information 때문이 아니라 새 row의 effective learning-rate/preconditioner 때문일 수 있다.
- high-entropy hash가 Jamo보다 좋은 이유는 sharing이 유용해서가 아니라 interference가 적어서일 수 있다.
- training-only extra optimizer state와 redundant parameter path가 더 좋은 basin을 찾게 했을 수 있다.

그러므로 generic 결과만 보고 새로운 Unicode representation 기법이라고 주장하지 않는다.

## 그래도 실제 효율 연구 후보가 되는 이유

Generic residual 자체는 강했다.

- untied: base 대비 contiguous `−0.015562`, document `−0.015660 BPB`
- tied: base 대비 contiguous `−0.025471`, document `−0.025497 BPB`
- untied generic의 dense-2K anchor gap: `+0.009307 BPB`
- tied generic의 dense-2K anchor gap: `+0.040128 BPB`
- 여섯 역할 모두 residual zero-init, copied-row mask와 fold state/NLL exact equality pass
- 배포 residual module: 없음
- deployed parameters: untied `25,172,352`, tied `22,026,624`로 architecture base와 exact 동일

특히 untied generic은 알려진 development stream에서 dense-2K anchor의 `+0.010 BPB` 근방까지
회복했다. Dense BPE-8K 경로는 이전 random-weight system probe에서 BPE-2K보다 약 `19.8%` 빠른
가능성을 보였다. 따라서 fresh data에서 품질이 유지되면 **더 큰 Korean vocabulary의 token-step
감소를 품질 손실 없이 실제 latency로 전환**할 현실적 경로가 있다.

다만 이는 아직 actual trained-model speed 결과가 아니다. 동일 개발 데이터를 반복 사용한 one-seed
screen이고, source가 보지 않은 데이터에서 quality noninferiority도 확인하지 않았다.

## 학습 비용

Fold가 추론 비용을 없애더라도 학습 비용은 무료가 아니다.

| architecture | training-only params | deployed 대비 | residual session | no-residual session | descriptive overhead |
|---|---:|---:|---:|---:|---:|
| untied | 1,277,952 | 5.08% | 958.77s | 730.76s | 31.20% |
| tied | 638,976 | 2.90% | 939.04s | 735.36s | 27.70% |

Session은 별도 실행의 elapsed diagnostic이라 paired performance gate는 아니다. Memory도 checkpoint 뒤
sampled allocator diagnostic이며 resettable peak가 아니다. 후속 논문은 inference saving과 추가
training cost의 amortization 조건을 함께 보고해야 한다.

## 독립 재계산과 artifact

Summarizer는 worker scalar를 신뢰하지 않았다.

- 6 roles × 4 checkpoints = 24 unfolded checkpoints를 strict-load하고 contiguous NLL을 재계산
- 6 final checkpoints의 document NLL 재계산
- 6 independently materialized dense checkpoints의 contiguous/document NLL 재계산
- B0 untied/tied no-residual controls의 contiguous/document NLL 재계산
- 모든 저장 float32 NLL 배열과 bitwise equality
- 모든 fold replay pass
- independent replay elapsed: `1824.802 s`

Artifact:

- plan commit: `847c824`
- result commit: `0a7c049`
- tracked summary: `results/foldable-jamo-residual-v1/summary.json`
- canonical summary SHA-256: `e9145e70537ec04b8df7575f98d80e4c330fb83cbdc8439cac052ea5353b34ec`

## 수정된 다음 연구 방향

### 종료하는 것

- Jamo residual slot/scale/table/seed 추가 탐색
- `0.002 BPB` threshold 완화
- step 32의 일시적 우위를 final candidate로 선택
- shuffled control을 제거한 Korean-specific claim
- BPB만으로 inference efficiency를 주장

### 새 별도 가설로 여는 것

다음 단계는 B1의 fallback이 아니라 별도 **foldable multi-hash vocabulary-adaptation** 가설이다.

첫째, 알려진 development data에서 최소 mechanism guard를 봉인한다.

1. ordinary dense no-residual
2. current 13-way foldable multi-hash residual
3. 새 row의 diagonal/self-update를 맞춘 dense optimization control
4. collision/sharing을 분리하는 assignment control

이 screen은 multi-hash 이득이 단순 `2×` self update, 추가 optimizer state, collision pattern 중 어디에서
오는지 식별해야 한다. 결과를 보고 learning rate를 연속 sweep하지 않으며, 분석적으로 고정한 control만
쓴다.

둘째, mechanism guard를 통과한 최소 recipe만 source가 보지 않은 disjoint Korean continuation으로
보낸다. Dense-2K continuation, ordinary dense-8K transfer, foldable dense-8K와 필요한 optimization
control에 동일 raw-byte history를 부여하고 적어도 3 model seeds를 사용한다. 역할 선택은 calibration
only로 고정하고 final quality를 한 번 연다.

셋째, quality-qualified exact folded dense-8K checkpoints만 dense-2K와 trained batch-1 actual inference로
비교한다. Controlled same-output와 strict-valid free-running 두 mode에서 E2E latency point reduction
`>=10%`, uncertainty와 seed/session stability를 모두 통과해야 positive efficiency result다. Residual,
hash lookup 또는 auxiliary branch는 timed graph에 존재하면 안 된다.

문헌상 novelty와 가장 강한 baseline은 이 신규 protocol을 봉인하기 전에 다시 검증한다. Hash
embedding, vocabulary expansion, Net2Net/structural reparameterization, training-time overparameterization,
optimizer preconditioning과 겹치는 부분을 분리하지 못하면 독립 기법 claim을 축소한다.

## Claim boundary

현재 강하게 말할 수 있는 것은 다음뿐이다.

> 한 compact Korean vocabulary-transfer development setting에서, foldable true-Jamo residual은
> matched generic multi-hash residual을 이기지 못해 사전 Korean-specific gate를 실패했다. 같은 실험의
> generic multi-hash residual은 추가 inference module 없이 ordinary dense 8K checkpoint로 정확히
> fold되었고, no-residual adaptation quality를 크게 개선했다.

Publication-quality 일반화, fresh-data quality, multi-seed uncertainty, trained actual inference speed,
CUDA/general hardware, larger model과 Korean downstream 개선은 아직 주장하지 않는다.
