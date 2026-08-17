# Conditional-local frozen sensitivity result and representation pivot

> 작성일: 2026-08-14
>
> protocol commit: `aeda6d3bbf613cb454325eb6c5db7f86cb0c0ae3`
>
> protocol: `jamoflow-conditional-local-frozen-sensitivity-v1`
>
> authoritative aggregate:
> `results/conditional-local-frozen-sensitivity-v1/summary.json`

## 1. 판정

고정한 네 operator/component pair는 generic UTF-8와 Hangul route 모두에서 frozen
quality-risk gate를 실패했다. 따라서 어떤 pair도 actual-runtime prototype을 허가하지
않는다.

| component / operator | UTF-8 difference BPB | Hangul difference BPB | Hangul one-sided upper | 판정 |
|---|---:|---:|---:|---|
| encoder+decoder / second-layer K/V | +1.208773 | +1.191123 | +1.198158 | fail |
| decoder / second-layer K/V | +0.272624 | +0.269704 | +0.271623 | fail |
| encoder+decoder / second MLP | +0.865335 | +0.853793 | +0.859208 | fail |
| decoder / second MLP | **+0.200484** | **+0.198832** | **+0.199967** | fail |

Baseline W72 BPB는 1.637935였다. 가장 보수적인 `decoder / second_mlp / hangul_prefix`도
고정 +0.020 risk margin의 약 9.94배를 악화시켰다. 모든 row에서 route rate와 document
coverage는 통과했지만 mean 및 document upper quality gate가 실패했다.

Summary file SHA-256은
`5f48ab269d44de01eef1205636784daaee88d381e81b7854b3fe8735e4c552fb`이고, 내부 canonical
summary SHA-256은
`85148e4288e7d1b8bc65ce2b1d33ed64f5e93d6b39f9a71f8d89554a099d46d6`이다. 여덟 NLL
artifact의 file hash, exact float32 shape `(15625,)`, finite/nonnegative 값과 array hash를
별도로 재검증했다.

## 2. 식별된 사실

첫째, UTF-8 continuation position을 `easy`와 동의어로 볼 수 없다. `utf8_incomplete`는
전체 position의 58.3055%, `hangul_prefix`는 57.5361%를 선택했다. 이 위치에서 다음 byte의
유효 범위는 좁아지지만, 어느 Unicode scalar 또는 Hangul syllable인지 결정하는 정보는
남아 있다. Frozen W72는 그 정보를 두 번째 local layer의 attention과 MLP에 실제로
의존한다.

둘째, encoder를 함께 줄인 처치가 decoder-only보다 훨씬 나빴다. 이는 local encoder
representation을 단순 overhead로 간주하면 안 된다는 정적 geometry 결과와 같은 방향이다.
다만 frozen intervention 하나만으로 encoder와 decoder의 독립 인과 효과나 retraining 후
회복 가능성을 추정하지 않는다.

셋째, Hangul route가 모든 pair에서 generic route보다 약간 덜 나빴지만 이를 한국어 고유
효과로 해석할 수 없다. Hangul mask는 generic mask의 98.6804%이고 0.7694 percentage point
적은 위치만 건너뛴다. 거의 같은 처치에서 작은 차이가 난 것이므로 specificity evidence가
아니다.

넷째, 이 결과는 trained conditional model의 실패가 아니다. Dense W72 checkpoint의
residual update를 사후에 없앤 perturbation이다. 그러나 사전 protocol이 이 screen의 통과를
runtime prototype 전제조건으로 두었으므로, 동일 candidate를 margin이나 route rate만 바꿔
계속 진행하지 않는다.

## 3. Fable 5 검토에 대한 결과 기반 보완

Fable 5의 가장 중요한 원칙 두 가지가 다시 지지됐다.

1. analytical 구조만으로 실제 효율을 선언하면 안 된다.
2. rate와 placement 또는 route identity를 분리하지 않으면 언어학적 기여를 주장할 수 없다.

반면 `규칙으로 가능한 후보를 줄이면 쉬운 위치의 neural compute도 생략할 수 있다`는 초기
직관은 이 형태로는 지지되지 않았다. Output validity constraint와 representation capacity는
다른 문제다. UTF-8 DFA가 불가능한 byte를 제거해도, 가능한 continuation들 사이의 의미
정보를 처리할 local capacity까지 불필요해지는 것은 아니다.

## 4. 연구 방향 수정

다음 항목은 종료한다.

- 현 네 operator/component pair의 incremental runtime 구현
- 같은 8MB calibration에서 skip 비율, margin 또는 route를 사후 조정하는 실험
- frozen Hangul row가 generic row보다 조금 나았다는 이유로 여는 한국어 고유 주장
- conditional skip 결과를 정적 geometry 실패의 구제책으로 해석하는 것

다음 저비용 질문은 **정보를 버리는 skip이 아니라 UTF-8의 여러 sequential encoding
step을 하나의 reversible semantic unit으로 바꿀 수 있는가**다. 구체적으로는 raw byte
fallback을 유지하면서 Unicode scalar를 한 autoregressive step으로 처리하고, precomposed
Hangul syllable의 output/input parameters를 초성·중성·종성 factorization으로 줄이는 hybrid
scalar representation을 검토한다.

이 방향은 곧바로 새 모델을 승인하지 않는다. 선행연구의 Korean three-hot character model,
MYTE, alternative Unicode encodings와 겹치는 부분이 크므로 먼저 아래 opportunity/novelty
audit을 통과해야 한다.

1. byte, generic Unicode-scalar, Hangul-factorized scalar, BPE의 sequential-step 수와
   parameter/output-head cost를 같은 Korean stream에서 계산한다.
2. generic scalar control과 비교해 Hangul factorization이 제공하는 것은 step reduction이
   아니라 parameterization 또는 quality prior임을 분리한다.
3. empirical choseong/jungseong/jongseong dependence를 계수해 독립 three-head가 감수할
   irreducible modeling loss를 추정한다.
4. 기존 EACL 2023 three-hot 결과와 다른 기여가 matched-quality decoder-only generation
   wall time, raw-byte fallback, BLT hierarchy의 결합에서 실제로 남는지 확인한다.
5. 이 audit에서 BPE 또는 generic scalar보다 유리할 실현 가능한 cost frontier가 없으면
   새 학습을 열지 않고 negative/diagnostic paper 정리로 전환한다.

## 5. Claim 경계

현재 추가로 허가되는 결론은 다음뿐이다.

> A frozen Korean W72 BLT was highly sensitive to removing second-layer local updates at
> UTF-8-incomplete or Hangul-prefix positions. Even the least damaging intervention increased
> calibration loss by about 0.199 BPB, so orthographic validity alone did not identify
> computation-free local positions.

이는 conditional computation 일반, retrained routing, Korean scalar representation 또는 다른
hardware의 실패를 뜻하지 않는다. 현재 positive paper 기준인 matched-quality actual
inference improvement도 아직 달성되지 않았다.
