# Dual-BPE sealed-test correction

> 작성일: 2026-08-11
> 상태: **publication-scale 학습·test·downstream·timing 전 고정**
> 교정 대상: [online-tokenization amendment](./60-online-tokenization-and-korean-efficiency-amendment.md) §7.2, [publication comparator protocol](./48-publication-comparator-and-downstream-protocol.md), [Mac feasibility addendum](./47-publication-scale-feasibility-addendum.md)
> graph 후속 교정: 16K total-parameter match는 [BPE body-match correction](./63-bpe-body-match-correction.md)으로 대체됨
> 우선순위: 정확도·인과적 무결성·완성도 > 추가 학습 시간

## 1. 발견한 순서 모순

기존 보강안은 다음 두 조건을 동시에 요구했다.

1. candidate가 32K BPE의 publication actual gate를 통과할 때만 16K stress control을 추가한다.
2. 16K와 32K 중 최종 BPE comparator는 test를 열기 전에 calibration으로 고정한다.

첫 조건의 actual gate는 sealed held-out BPB, downstream, controlled replay와 free-running timing을 이미 사용한다. 따라서 그 결과로 16K 실행 여부를 정한 뒤에는 두 번째 조건의 `test 전 선택`이 성립하지 않는다. Candidate와 32K의 test 값을 알고 새 comparator를 학습하는 것은 규칙을 미리 써 두었더라도 완전한 sealed comparison이 아니다.

## 2. 교정: 선택하지 않고 둘 다 이겨야 한다

Publication-scale campaign에 진입하면 16K와 32K byte-BPE를 처음부터 모두 required comparator로 둔다. 어느 하나를 calibration latency로 선택해 버리지 않는다.

- 32K는 ordinary standard byte-BPE baseline이다.
- 16K는 작은 output embedding/head에서 얻을 수 있는 이점을 직접 통제하는 stress baseline이다.
- 두 tokenizer와 parameter-matched graph는 quality, downstream 또는 timing을 열기 전에 고정한다.
- candidate의 broad Korean inference-efficiency gate는 raw-byte reference, 16K BPE, 32K BPE를 **각각** 통과해야 한다.
- 둘 중 하나만 이기면 `BPE vocabulary-specific result`일 뿐 vocabulary-size confound를 제거한 broad result가 아니다.

이는 여러 baseline 중 유리한 하나를 고르는 union test가 아니다. 모든 null을 기각해야 하는 intersection–union 구조이므로, “두 BPE 모두보다 빠르고 품질을 유지한다”는 주장은 각 comparator gate가 모두 통과할 때만 성립한다.

## 3. 실행 순서

1. Compact Final Value Gate가 publication-scale 진입을 허용한다.
2. Pinned benchmark rows와 HPLT source/file hash를 검증하고 contamination reference-equivalence 후 full scan을 완료한다.
3. 모든 family가 공유할 clean train/calibration/test document order와 stream hash를 봉인한다.
4. 오직 filtered train split으로 16K/32K tokenizer를 학습하고 round-trip, token-byte transition, JSON hash를 봉인한다.
5. Compact selection의 concrete raw-reference descriptor와 candidate/raw/BPE graph를 instantiate한 뒤 실제 common stream·tokenizer·patch/auxiliary path로 quality를 사용하지 않는 four-family Mac feasibility를 실행해 scale을 고정한다.
6. Source, tokenizer, graph, raw descriptor, optimization/workload config를 **pre-training campaign-input lock**으로 commit한다.
7. 같은 clean Korean train stream과 paired seeds `1729/2718/31415`로 candidate, raw reference, 16K BPE, 32K BPE를 학습한다. E/EC raw이면 각 seed router train→full-stream score/cache→calibration threshold→main training 순서를 따른다.
8. 각 BPE에서 data-matched와 compute-matched checkpoint를 만든다.
9. 실제 main/router checkpoint와 calibration bundle hash를 **post-training evidence lock**에 추가한다. Calibration은 checkpoint 선택과 downstream task별 strongest-reference 선택에만 사용하며 BPE vocabulary를 탈락시키지 않는다.
10. 그 뒤 held-out test BPB, sealed downstream split과 actual-inference timing을 한 번만 연다.
11. Candidate는 calibration에서 고정한 strongest raw/16K/32K task reference에 대한 downstream noninferiority를 먼저 통과한다. 이어 두 BPE 각각에 대해 BPB noninferiority, controlled-replay decode, free-running end-to-end, UTF-8/replacement validity gate를 통과해야 한다.

Publication test를 본 뒤 16K를 추가하거나, 16K 결과가 불리하다는 이유로 32K만 남기는 경로는 허용하지 않는다.

## 4. Result-blind graph 고정

두 vocabulary 모두 tied input/output embedding, 12-layer full-MHA Llama-style graph를 쓴다. 32K graph는 hidden width 384–768, FFN 2.5–4.0배, 정수 head divisor의 result-blind grid에서 candidate exact parameter와 가장 가까운 graph를 고른다. 16K stress는 output-head intervention을 분리하기 위해 같은 target의 32K hidden/FFN/layer/head geometry를 그대로 쓰고 vocabulary rows만 줄인다. Quality와 timing은 어느 선택에도 들어가지 않는다.

| Vocabulary | Target | width / heads / layers / FFN | exact params | candidate 대비 관계 |
|---:|---:|---:|---:|---:|
| 16K body-matched | 50M | 448 / 7 / 12 / 1,600 | 42,617,792 | 14.462% smaller |
| 16K body-matched | 75M | 608 / 8 / 12 / 1,792 | 66,710,368 | 12.788% smaller |
| 16K body-matched | 100M | 704 / 11 / 12 / 2,048 | 86,975,680 | 11.613% smaller |
| 32K | 50M | 448 / 7 / 12 / 1,600 | 49,785,792 | 0.076% |
| 32K | 75M | 608 / 8 / 12 / 1,792 | 76,438,368 | 0.071% |
| 32K | 100M | 704 / 11 / 12 / 2,048 | 98,239,680 | 0.166% |

`src/jamoflow/publication_bpe.py`가 analytical parameter count, 32K 전수 grid 선택과 16K body identity를 서로 대조한다. 32K만 candidate parameter의 1% 이내를 요구한다. 16K는 더 작은 model이라는 불리한 조건까지 candidate가 이기게 하는 stress control이다.

## 5. Downstream과 data-adequacy 영향

Task별 downstream reference 후보는 strongest raw reference, compute-matched 16K BPE, compute-matched 32K BPE다. Sealed task split을 열기 전에 calibration/selection score가 가장 높은 model을 고정한다. 32K가 최고값과 0.5 percentage points 이내면 deployment-default tie-break로 32K를 사용한다. Candidate 자신은 reference 후보가 아니다.

Learning-curve adequacy도 candidate–raw, candidate–16K data-matched, candidate–32K data-matched 세 pair 모두에서 교정된 last-two-budget noninferiority를 요구한다. 순위 부호나 gap 크기를 요구하지 않고, 두 최근 budget의 `+0.010 BPB` margin 유지와 양쪽 model의 계속된 학습을 검사한다. 한 BPE만 안정적인 결과는 충분히 학습된 broad comparison으로 인정하지 않는다.

## 6. Feasibility 영향

Core campaign은 기존 `candidate/reference/BPE × 3 = 9` runs에서 다음 12 runs로 바뀐다.

```text
(candidate + raw reference + BPE-16K + BPE-32K) × 3 seeds = 12 runs
```

120시간 상한은 유지한다. 따라서 model당 safety-adjusted 시간이 더 엄격해지고 선택 scale이 낮아질 수 있다. 더 큰 graph 하나보다 output-head confound가 제거된 완결된 비교를 우선한다. 어느 scale도 12-run 조건을 통과하지 못하면 16K를 생략하지 않고 외부 compute가 필요하다고 판정한다.

## 7. 현재 Phase 3와의 관계

이 교정은 진행 중인 compact F/C/W/S/E/EC policy, seed, checkpoint 또는 gate를 바꾸지 않는다. Publication-scale 결과가 아직 없기 때문에 결과를 본 뒤의 protocol 변경도 아니다. 현재 family는 원래 순서대로 봉인해 완주하고, 이 문서는 compact gate를 통과해 publication-scale로 확장할 때만 적용한다.

## 8. Claim rule

| 결과 | 허용되는 해석 |
|---|---|
| raw만 통과 | byte-latent family 내부 결과 |
| raw와 BPE 하나만 통과 | vocabulary-specific BPE comparison; broad claim 금지 |
| raw·16K·32K 모두 통과, data adequacy 실패 | Mac mechanism-scale 결과 |
| raw·16K·32K·data adequacy 모두 통과 | broad Korean inference-efficiency paper 후보 |

논문 제목이나 초록의 `faster Korean inference`는 마지막 행에서만 허용한다.
