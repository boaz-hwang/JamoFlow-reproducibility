# BPE one-seed quality frontier 결과와 연구 방향 수정

> 작성일: 2026-08-14
>
> 상태: sealed one-seed development result; publication comparator 아님

## 결론

동일한 128M Korean raw bytes와 약 19.6M total parameters에서 여섯 BPE system을 학습한
결과, **2K×8L만 `+0.010 BPB` quality gate를 통과했다.** 2K가 연속-stream과 공통-document
평가 모두에서 가장 낮은 BPB였고, 다음으로 좋은 4K×12L도 약 `+0.062 BPB` 뒤졌다.

따라서 random-weight systems frontier의 point-fastest였던 32K×8L를 다음 baseline으로 쓰면
안 된다. 32K는 2K보다 사전 E2E가 21.7% 빨랐지만 문서 BPB가 `+0.195` 나빠 matched-quality
baseline이 아니다. 이 scale에서 fastest quality-qualified development comparator는
**2K×8L, 88.28 ms**다.

이 결과는 다음 계획을 필요한 만큼 수정한다.

1. 다음 tokenizer 후보는 32K/64K가 아니라 exact 2K×8L graph를 기준으로 먼저 비교한다.
2. Vocabulary 크기를 다시 늘려 step을 줄이기 전에, 같은 2K vocabulary budget 안에서
   segmentation/vocabulary content를 바꿔 quality와 core capacity를 보존하는 방법을 시험한다.
3. Generic long-token control과 Korean-aware constrained variant를 같은 vocabulary, 같은 graph,
   같은 total parameters에서 비교한다.
4. `2K가 한국어에 보편적으로 최적`이라고 주장하지 않는다. 이 결과에는 작은 19.6M
   parameter budget과 one-pass training의 강한 scale dependence가 있다.

## 봉인된 결과

| role | embed / total params | train steps | train time | contiguous BPB | document BPB | Δ document BPB | presealed E2E | 적격 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2K×8L | 4.0% | 2,213 | 36.76 min | **1.4372** | **1.4369** | 0 | 88.28 ms | ✅ |
| 4K×12L | 7.0% | 1,904 | 38.57 min | 1.4995 | 1.4991 | +0.0622 | 123.35 ms | ❌ |
| 8K×8L | 14.7% | 1,675 | 29.39 min | 1.5320 | 1.5313 | +0.0944 | 114.60 ms | ❌ |
| 16K×8L | 28.7% | 1,502 | 26.91 min | 1.5967 | 1.5961 | +0.1592 | 77.87 ms | ❌ |
| 32K×8L | 47.0% | 1,358 | 22.75 min | 1.6316 | 1.6318 | +0.1950 | **69.13 ms** | ❌ |
| 64K×8L | 73.3% | 1,244 | 25.59 min | 1.6778 | 1.6787 | +0.2418 | 70.79 ms | ❌ |

문서 bootstrap 95% upper는 4K부터 각각 `+0.0643`, `+0.0972`, `+0.1635`, `+0.1997`,
`+0.2471 BPB`였다. 즉 `+0.010` 경계 근처에서 우연히 한 role만 탈락한 결과가 아니다.
연속-stream과 document-paired 평가의 순서와 차이도 거의 같아 tail/chunk 구성에 의한 역전
징후가 없다.

여섯 training core의 합은 2.9995시간이었다. Resource-only preflight의 train-only projection은
2.8649시간이어서 실제 training은 약 4.7% 길었다. Preflight의 train+contiguous-eval 합계는
2.9228시간이었고, 실제 worker wall-clock에는 tokenization, document evaluation, checkpoint
serialization이 추가된다. 따라서 3-step preflight는 core training을 약간 낙관적으로
예측했지만 24-hour feasibility 판단에는 충분한 여유가 있었고, 128M을 줄일 이유는 없었다.

## 왜 큰 vocabulary가 크게 뒤졌는가

이 결과를 tokenizer compression의 단독 인과효과로 읽으면 안 된다. 적어도 세 메커니즘이
동시에 움직였다.

### 1. 고정 total-parameter에서의 embedding tax

Tied embedding/output matrix만 계산해도 2K는 total parameter의 4.0%를 쓰지만 32K는 47.0%,
64K는 73.3%를 쓴다. 따라서 64K의 Transformer core parameter는 약 5.22M뿐이고, 2K는
약 18.88M이다. 큰 vocabulary가 decode step을 줄이는 대신 sequence model capacity를 크게
잃었다. 이번 BPB 차이의 가장 강한 설명이다.

이는 결함이 아니라 19.6M total-parameter deployment budget에서 실제로 발생하는 tradeoff다.
그러나 model scale이 커지면 embedding fraction이 줄어 optimal vocabulary가 오른쪽으로 이동할
수 있으므로, 이 결과를 100M·1B model에 외삽하지 않는다.

### 2. 한 raw pass에서의 update/data efficiency

2K는 같은 raw stream을 70,798 token sequences와 2,213 optimizer steps로 보았고, 64K는
39,801 sequences와 1,244 steps로 보았다. 모든 model이 같은 raw bytes를 한 번 보았지만 큰
vocabulary는 class당 관측과 update가 적다. 반대로 2K는 더 많은 학습 compute를 받았다.

이 실험의 목적은 tokenizer vocabulary의 순수 효과 추정이 아니라 실제 deployable system
frontier 선택이므로 이 차이를 허용하고 전부 공개했다. 이후 mechanism ablation에서는
same-vocabulary/same-graph 비교로 이 confound를 제거한다. 추가로 compute-matched curve를
만들 수 있으나, raw exposure가 달라지는 별도 질문으로 분리한다.

### 3. 작은 corpus에 대한 large-vocabulary sparsity

128M bytes에서 32K/64K token inventory는 긴 저빈도 item을 많이 가진다. Output class가 커지고
각 class의 학습 사례가 희소해지는 동시에 core가 좁아진다. 64K의 step 수 감소만 보고
효율적이라고 판단할 수 없는 이유다.

## 독립 검증

Summary 단계에서 여섯 checkpoint를 CPU에서 exact state hash로 재구성한 뒤 MPS로 다시
load했다. 각 tokenizer의 8M contiguous calibration과 385-document evaluation을 전부 새로
실행했고, 네 배열을 저장 evidence와 bitwise 비교했다.

- contiguous per-sequence NLL
- contiguous raw target bytes
- document per-record NLL
- document raw bytes

여섯 role 모두 통과했다. 최종 선택은 이 재검증된 배열에서만 계산됐다. Plan hash는
`ee51ccecef2c19caabe38b6776673fa5b1d28db6bb9b51e9b192728350079294`, result hash는
`c9f18d45f22cc7c4a1dc68dec431e4709b8eed9a5c7ac632cb118a7d87cccd0e`다.

## 수정된 다음 실험

### 단계 B1 — same-2K tokenizer opportunity

같은 2,048 vocabulary와 exact 2K×8L graph에서 다음 세 역할을 먼저 만든다.

1. 봉인된 exact ByteLevel BPE
2. generic length-maximizing vocabulary/논문 정의의 longest-match control과 segmentation ablation
3. Korean orthography-constrained length-maximizing variant

결과를 보기 전 다음을 측정·봉인한다.

- full train/calibration byte-exact roundtrip와 byte fallback
- token count, bytes/token, vocabulary item frequency/length distribution
- tokenizer encode throughput
- 동일 128-byte prompt/continuation에서 actual prefill+cached decode steps와 E2E
- Korean syllable/eojeol/morpheme boundary crossing diagnostics

Generic control이 BPE 대비 10% token-step 기회를 만들지 못하면 Korean constraint의 speed ceiling도
낮으므로 candidate 구조를 재검토한다. Korean variant는 generic과 같은 token count만 내는 것으로
충분하지 않다. 같은 또는 더 좋은 step reduction에 더 나은 boundary validity/quality prior를
보이거나, generic보다 명확한 additional latency opportunity가 있어야 한다.

### 단계 B2 — capacity-scale diagnostic

Same-2K 실험과 병렬로 해석하면 안 되지만, 다음 full candidate scale을 정하기 전에 50M 또는
100M에서 2K/16K/32K의 parameter decomposition과 MPS feasibility를 점검한다. 목적은 large-vocab
실패가 19.6M embedding tax에 국한되는지 확인하는 것이다. 품질 결과를 보기 전에 scale과
최소 role set을 resource-only rule로 고정한다.

### 단계 C 이후

Token-only opportunity를 통과한 최소 역할만 one-seed training으로 간다. 최종 positive claim은
여전히 다음 조건을 모두 요구한다.

- fastest quality-qualified BPE 대비 raw-byte quality noninferiority
- generic long-token control을 넘는 Korean-specific contribution
- trained-model batch-1 end-to-end generation latency 최소 10% 개선 가능성
- 이후 multi-seed, 새 sealed final data, fresh timing session에서 재현

이번 결과는 이 목표를 낮추지 않는다. 오히려 32K/64K라는 약한 quality baseline을 제거해
향후 positive result가 실제로 더 강한 증거가 되도록 한다.

## Artifact

- plan: `data/manifests/bpe-quality-frontier-one-seed-v1.json`
- tracked result: `results/bpe-quality-frontier-one-seed-v1/summary.json`
- ignored checkpoint/NLL/worker evidence: `artifacts/bpe-quality-frontier-one-seed-v1/`
