# Korean BPE systems frontier 결과와 quality-frontier 전환

> 작성일: 2026-08-14
>
> 상태: sealed calibration-development 결과; matched-quality 또는 publication speed claim 아님

## 1. 결론

이 Mac의 약 19.6M-parameter, batch-1, float32 MPS 조건에서는 ordinary byte-BPE의
random-weight systems frontier가 **32K vocabulary, 8 layers** 부근에 있었다. Vocabulary를
64K로 늘리면 calibration token 수는 32K보다 8.02% 줄었지만 full-vocabulary head와 더 좁아진
model core 비용 때문에 E2E는 오히려 2.40% 느렸다. 반대로 16K는 head가 작아도 continuation
step이 늘어 32K보다 12.65% 느렸다.

따라서 다음 연구에서 16K 하나만 BPE baseline으로 쓰거나, token count가 가장 작은 64K를
속도 baseline으로 쓰면 안 된다. **32K×8L와 64K×8L를 모두 품질 학습에 남기고**, 더 작은
vocabulary가 같은 parameter budget에서 얻는 model-core capacity가 품질에 필요한지도 함께
검증해야 한다.

## 2. 봉인된 tokenizer 결과

동일한 Korean train split 5,791 documents에서 각 tokenizer를 두 번 학습했고 compact JSON
bytes가 정확히 같았다. 7,999,999 complete UTF-8 calibration bytes의 결과다.

| vocabulary | tokens | bytes/token | CPU encode median | MB/s |
|---:|---:|---:|---:|---:|
| 2,048 | 2,263,476 | 3.534 | 1,841.07 ms | 4.35 |
| 4,096 | 1,947,021 | 4.109 | 1,713.19 ms | 4.67 |
| 8,192 | 1,711,797 | 4.673 | 1,640.52 ms | 4.88 |
| 16,000 | 1,533,938 | 5.215 | 1,628.38 ms | 4.91 |
| 32,000 | 1,388,745 | 5.761 | 1,650.15 ms | 4.85 |
| 64,000 | 1,277,330 | 6.263 | 1,635.60 ms | 4.89 |

16K와 32K는 plan에 공개한 기존 결과를 byte-for-byte 같은 tokenizer와 token count로
재현했다. 64K는 32K보다 token이 8.02%, 16K보다 16.73% 적었다. CPU encode throughput은
vocabulary에 따라 단조적으로 좋아지지 않았고, 2K를 제외하면 4.67--4.91 MB/s 범위였다.
이 비용은 model timer 밖의 진단값이며 아래 E2E에 합산하지 않았다.

## 3. 실제 MPS graph 결과

모든 graph는 target 19,596,096 parameters의 ±0.5% 안에 있었고, 6개 correctness case의
full/no-cache, sequential cache, parallel-prefill cache 비교에서 argmax가 전부 같았다. 가장 큰
normalized tolerance ratio는 0.0202로 고정 상한 1보다 충분히 작았다.

각 vocabulary에서 가장 빠른 depth만 표시하면 다음과 같다.

| vocabulary | fastest depth | parameters | TTFT | decode | E2E | continuation steps |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 8 | 19,667,328 | 4.92 ms | 83.42 ms | 88.28 ms | 35.0 |
| 4,096 | 12 | 19,577,040 | 7.42 ms | 116.16 ms | 123.35 ms | 30.0 |
| 8,192 | 8 | 19,560,288 | 8.03 ms | 105.51 ms | 114.60 ms | 27.0 |
| 16,000 | 8 | 19,605,344 | 7.09 ms | 71.40 ms | 77.87 ms | 24.0 |
| 32,000 | 8 | 19,616,544 | 6.39 ms | 63.37 ms | **69.13 ms** | 22.5 |
| 64,000 | 8 | 19,558,112 | 5.94 ms | 65.38 ms | **70.79 ms** | 21.0 |

`32K×8L` 대비 paired document 결과는 다음과 같다.

- `64K×8L`: median reduction −2.40%, 95% bootstrap interval
  [−8.61%, +5.33%], 36 documents 중 17개에서 더 빠름. 차이를 확정하지 못했다.
- `16K×8L`: median reduction −12.65%, interval [−33.15%, −4.48%], 6/36에서만 더
  빨랐다.
- `8K×8L`: −65.77%, interval [−76.02%, −50.60%].
- `2K×8L`: −27.69%, interval [−34.83%, −15.89%].

4K의 12-layer point가 8-layer보다 0.4ms 낮은 예외가 있었지만, 4K 자체가 global frontier보다
매우 느리므로 그 작은 차이를 architecture 원리로 해석하지 않는다. 전체 compression--latency
Pareto set은 `32K×8L`과 `64K×8L` 두 개다.

## 4. 무엇을 알게 되었는가

1. **token 감소와 wall-clock은 같은 값이 아니다.** 64K는 step을 줄였지만 32K를 이기지
   못했다. Korean tokenizer 연구도 token fertility만 보고 성공이라 할 수 없다.
2. **vocabulary와 model geometry를 함께 비교해야 한다.** Fixed total parameters에서 큰
   vocabulary는 embedding/head에 더 많은 parameter를 써 hidden/FFN을 줄인다. 이 변화는
   latency뿐 아니라 품질에도 영향을 준다.
3. **기존 BPE16K는 약한 속도 comparator였다.** 같은 8-layer family에서도 32K가 16K보다
   12.7% 빨랐다. 이후 candidate가 16K만 이긴 결과는 논문 가치가 없다.
4. **64K를 버릴 근거도 아직 없다.** 32K와 latency 차이가 불확실하고, learned output의
   free-running bytes/token은 controlled replay와 다를 수 있다. 두 role 모두 학습해야 한다.
5. **이 결과 자체는 efficiency paper의 positive evidence가 아니다.** Random weights,
   calibration-development cases, 한 번의 MPS session, fixed route이며 품질과 실제 생성 경로를
   아직 보지 않았다.

## 5. 근거에 따른 다음 계획 수정

기존 문서의 `top-three latency role만 quality 학습`은 strongest quality-qualified BPE를 찾기에는
불충분하다. 작은 vocabulary의 더 큰 Transformer core가 유의미하게 더 좋은 BPB를 얻으면
속도가 느려도 quality frontier의 기준점이 될 수 있기 때문이다. 정확성을 우선해 다음 단계는
각 vocabulary에서 가장 빠른 여섯 graph를 모두 포함한다.

1. `2K×8L`, `4K×12L`, `8K×8L`, `16K×8L`, `32K×8L`, `64K×8L`를 같은 128M Korean raw
   train stream, 같은 seed와 optimizer로 one-pass 학습한다.
2. Token 수가 아니라 source raw bytes를 고정하고, 각 target token이 표현하는 raw byte 수로
   calibration BPB를 계산한다.
3. Training wall-clock, predicted raw bytes/s, token/s를 함께 기록해 vocabulary가 학습 비용에
   미치는 영향도 숨기지 않는다.
4. 가장 낮은 calibration BPB를 quality anchor로 두고 `+0.010 BPB` 안의 graph만
   quality-qualified로 남긴다. 그 집합에서 실제 E2E가 가장 빠른 graph가 BPE systems
   comparator다.
5. 이 one-seed 결과로 comparator를 고정하지 않는다. 다음 Korean-aware/generic tokenizer
   opportunity와 one-seed 비교를 설계하는 development evidence로만 사용한다.
6. 최종 candidate가 생긴 뒤에는 새 sealed final split, 3--5 model seeds, strict-valid
   free-running 생성, fresh timing sessions에서 같은 quality gate를 다시 적용한다.

여섯 모델 전부를 128M bytes로 학습하는 비용은 먼저 실제 한-step train/eval preflight로
측정한다. 메모리나 총 예상 시간이 이 Mac의 합리적 범위를 벗어나면 결과를 본 뒤 임의로
vocabulary를 빼지 않고, 사전 고정한 작은 raw-byte checkpoint를 모든 여섯 role에 동일하게
적용해 learning-curve screen을 만든다.

## 6. Artifact

- plan: `data/manifests/korean-bpe-systems-frontier-v1.json`
- tracked summary: `results/korean-bpe-systems-frontier-v1/summary.json`
- ignored tokenizer/runtime evidence:
  `artifacts/korean-bpe-systems-frontier-v1/`

이 결과가 허용하는 유일한 결론은 **32K 부근이 현재 hardware의 systems optimum 후보이고,
quality 학습으로 이를 검증해야 한다**는 것이다. Korean-aware method의 우월성은 아직 0이다.
