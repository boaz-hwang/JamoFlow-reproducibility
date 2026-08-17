# Fresh-v2 16K vocabulary quality result

> 작성일: 2026-08-15
>
> 상태: one-seed quality gate 통과; trained actual-inference preflight만 허가

## 결론

결과를 보기 전에 봉인한 다섯 역할을 fresh-v2의 동일한 128MB 한국어 학습 stream에서 모두
학습했다. `dense16k_update_geometry`는 두 품질 anchor보다 나쁘지 않은 수준에 머물지 않고 둘 다
앞섰으며, 동일한 16K tokenizer와 초기 상태를 쓰는 두 강한 adaptation control도 사전 고정된
최소 차이보다 크게 이겼다.

따라서 다음 단계에서 actual latency를 측정할 수 있는 checkpoint는 계획대로
`dense16k_update_geometry` 하나뿐이다. 이 결과는 실제 추론 효율, 다중 seed, publication claim을
아직 허가하지 않는다.

## 봉인과 실행

- protocol implementation commit: `a2c1058`
- result-blind plan commit: `f4c09b2`
- plan payload SHA-256: `8a75b2be7371a69b6120dbab38c710e81e8c31f45468e672ec13390c587eb4f4`
- summary payload SHA-256: `8d7fc2260a657713dae2fbd6c62dc617a3ace084725dec782d29cee6ead7e85f`
- tracked summary file SHA-256 before commit: `c95d6095a88e1c04ca93df807cfcaedc992f990e07d836eb1f6bd5e51707b611`
- role count: 5
- independent checkpoint replay: 5/5 bitwise-equal

학습 중 role, learning rate, update multiplier, training byte budget, gate 또는 actual candidate를
변경하지 않았다. 다섯 worker가 완료된 뒤에만 공식 summarizer를 실행했다.

## 품질 결과

Primary metric은 356개 공통 calibration document의 raw-byte-normalized BPB다.

| role | vocab | document BPB | optimizer steps | optimizer time | parameters |
|---|---:|---:|---:|---:|---:|
| `dense2k_joint_v2` | 2,048 | 1.408331 | 2,219 | 2,064.98s | 19,667,328 |
| `dense8k_update_geometry_v2` | 8,192 | 1.397882 | 1,682 | 1,639.51s | 25,172,352 |
| `dense16k_standard_joint` | 16,000 | 1.422436 | 1,509 | 1,474.26s | 31,168,896 |
| `dense16k_inplace_two_stage` | 16,000 | 1.439110 | 1,509 | 1,325.73s | 31,168,896 |
| `dense16k_update_geometry` | 16,000 | **1.393474** | 1,509 | 1,475.93s | 31,168,896 |

### Gate A — 두 anchor에 대한 non-inferiority

차이는 `candidate - comparator`다. 음수는 candidate가 더 좋다는 뜻이다.

| comparator | point BPB | paired document bootstrap 95% interval | fixed margin | pass |
|---|---:|---:|---:|---|
| 2K joint | -0.014857 | [-0.016672, -0.013034] | +0.010 | yes |
| 8K update geometry | -0.004408 | [-0.005374, -0.003477] | +0.010 | yes |

두 비교 모두 non-inferiority를 넘어서 관측상 우월했다. 사후에 더 약한 anchor 하나만 고른 결과가
아니다.

### Gate B — 동일 16K control에 대한 method advantage

| control | point BPB | paired document bootstrap 95% interval | fixed minimum | pass |
|---|---:|---:|---:|---|
| standard joint | -0.028961 | [-0.030303, -0.027701] | -0.002 | yes |
| in-place two-stage | -0.045636 | [-0.047258, -0.044074] | -0.002 | yes |

Update geometry의 차이는 단순 16K tokenizer 효과로 설명되지 않는다. 같은 tokenizer, graph,
initial state, data order, raw-byte budget 및 optimizer family에서 update rule만 다른 두 control을
모두 큰 폭으로 앞섰다. 반대로 이 한 실험만으로 multiplier가 보편적 최적값이거나 인과 효과가
다른 모델 규모에도 유지된다고 주장할 수는 없다.

### Cross-data 8K replication

8K update geometry도 새 split에서 2K보다 `-0.010449 BPB`, 95% interval
`[-0.011582, -0.009278]`로 좋았다. Fresh-v1에서 얻은 recipe가 fresh-v2에서도 재현됐기 때문에
cross-vocabulary geometry claim의 개발 단계 조건은 통과했다. 다만 두 stream은 같은 HPLT shard의
서로 다른 문서이므로 새로운 domain replication은 아니다.

## Systems accounting

16K candidate는 2K 대비:

- full token sequence 수 31.98% 감소
- optimizer step 수 32.00% 감소
- 측정 optimizer time 28.53% 감소
- parameter 수 58.48% 증가

8K 대비로는:

- full token sequence 수 10.25% 감소
- optimizer step 수 10.29% 감소
- 측정 optimizer time 9.98% 감소
- parameter 수 23.82% 증가

학습시간 감소는 이 Mac의 이 캠페인에 대한 training systems result다. 이것을 inference speed로
대체하지 않는다. 특히 16K output projection과 resident weight 증가는 decode step cost와 memory를
악화시킬 수 있다.

## 독립 검증

Official summarizer는 각 checkpoint를 새로 load하고 tokenizer별 contiguous 및 공통 document
calibration forward를 전부 재실행했다. 다섯 역할 모두 다음이 worker artifact와 bitwise 동일했다.

- contiguous per-sequence NLL
- contiguous raw target-byte denominator
- document-chunk NLL
- document raw-byte denominator

Actual candidate checkpoint는 다음 identity로 고정됐다.

- artifact SHA-256: `87ac6ca118c0fc60f685523b41185793f38b4daf51592cdd5d2ffff8642731c6`
- state SHA-256: `7efa2ead8a6cc60e81b513c724c9b1b554d6ca0d674dbd8ef1ceff1cc6a05902`

## 연구 방향 판단

현 시점에는 16K 계획을 수정할 근거가 없다. 품질에서 가장 어려운 조건을 모두 통과했고,
16K에서 update-geometry 효과가 두 control과 두 anchor에 대해 동시에 나타났다. 다음 질문은 이제
명확히 systems question이다.

> 더 큰 output head와 weight footprint까지 포함했을 때, trained 16K candidate가 quality-qualified
> 2K baseline보다 controlled replay와 strict-valid free generation의 실제 E2E latency를 각각 10%
> 이상 줄이는가?

Actual preflight는 2K를 primary expansion baseline으로 사용하고, quality가 더 강한 8K
update-geometry checkpoint를 secondary frontier comparator로 함께 측정해야 한다. Primary gate는
16K-vs-2K의 controlled/free joint actual result에만 사전 고정한다. 16K-vs-8K는 8K에서 16K로
확장한 incremental systems effect와 Pareto 해석을 위한 의무 보고 항목이며, 결과에 따라 primary
comparator를 바꾸는 fallback이 아니다.

다음 단계에서도 다음을 유지한다.

1. analytical token/step 감소는 성공이 아니다.
2. cache/full equivalence, greedy byte equality, UTF-8 validity와 repetition determinism을 먼저 검증한다.
3. controlled와 free E2E가 각각 point `>=10%`이고 고정 uncertainty/stability gate를 통과해야만
   multi-seed를 허가한다.
4. 16K의 parameter·checkpoint·role-isolated memory 증가를 속도와 함께 공개한다.
5. one-seed actual이 실패하면 threshold를 낮추거나 8K 결과와 합쳐 성공으로 바꾸지 않는다.

## 주장 경계

현재 허가되는 주장은 다음뿐이다.

- 한 development seed와 같은-source disjoint Korean split에서 fixed 16K update geometry가 두 anchor와
  두 16K control을 모두 통과했다.
- 16K trained actual-inference preflight를 수행할 근거가 생겼다.

다음 주장은 아직 금지된다.

- 실제 추론 효율 개선
- memory 개선
- 다중 seed 재현성
- 한국어 전반 또는 다른 hardware/model 규모 일반화
- publication-ready positive efficiency claim
