# Fresh-v2 16K vocabulary quality fail-fast protocol

> 작성일: 2026-08-15
>
> 상태: result-blind implementation protocol; plan·training result 미생성

## 연구 결정

Fresh-v1 8K candidate는 calibration quality를 통과했고 controlled actual E2E는 `20.13%`
개선했지만 free-running은 `8.84%`, 95% bootstrap 하한 `-0.17%`로 실패했다. 실패 원인은
token 감소가 부족해서만이 아니다. Candidate의 free-running per-token latency가 오히려 약간
증가했고, 출력 길이 변동이 E2E 효과를 지배했다.

따라서 “더 큰 vocab이면 무조건 빠르다”를 가정하지 않는다. 이미 결과를 보기 전에 수행된
same-body random-weight systems frontier에서 16,000 vocab이 8,192보다 낮은 per-step/E2E
cost를 보였고 32,000은 더 짧아도 head cost로 악화됐다. 이 사전 근거 때문에 16,000을 단 한
번의 다음 크기로 사용한다. 12K/24K/32K를 함께 돌려 결과를 고르지 않는다.

이 단계의 질문은 하나다.

> Fresh-v1에서 고정된 new-row AdamW update geometry를 재조정 없이 16,000-entry canonical
> Korean BPE 확장에 적용하면, 강한 16K adaptation controls와 2K/8K anchors를 모두 이기는
> matched-quality checkpoint를 얻을 수 있는가?

Vocabulary expansion 자체, compositional row initialization, two-stage CPT는 선행 연구가 있는
기법이다. 이 실험의 novelty 후보는 이들을 새 방법처럼 부르는 데 있지 않다. 고정된
optimizer-update geometry와 실제 trained-model E2E 효과의 연결이 재현되는지를 검증한다.

## 결과를 보기 전에 고정하는 다섯 역할

`16K`는 이 repository의 tokenizer frontier와 일치하는 정확히 16,000 entries를 뜻한다.

| role | vocab | 목적 | initialization | training |
|---|---:|---|---|---|
| `dense2k_joint_v2` | 2,048 | cross-data quality anchor | exact source checkpoint | all-parameter joint |
| `dense8k_update_geometry_v2` | 8,192 | fixed recipe cross-data replication/anchor | uniform input + byte-weighted output, untied | fixed update geometry |
| `dense16k_standard_joint` | 16,000 | ordinary CPT control | same 16K initialization | all-parameter joint |
| `dense16k_inplace_two_stage` | 16,000 | literature-aligned strong control | same 16K initialization | new rows 60%, then all 40% |
| `dense16k_update_geometry` | 16,000 | 유일한 actual candidate | same 16K initialization | fixed update geometry |

16K controls는 candidate와 tokenizer, graph, initial state, raw data, order, optimizer family,
effective batch 및 total raw budget이 같다. 차이는 schedule/update rule뿐이다. Candidate 외 역할을
calibration 결과에 따라 actual 후보로 승격하는 fallback은 없다.

## Data와 tokenizer identity

- train: fresh-v2 exact 128,000,000 raw bytes
- calibration: fresh-v2 exact 8,000,000 raw bytes
- document unit: fresh-v2 calibration의 ordered original document
- stable final test: 이 단계에서 열지 않음
- tokenizer set: pre-existing canonical byte-BPE 2,048 / 8,192 / 16,000

Fresh-v2는 Phase-3, sealed final, fresh-v1의 총 14,469문서를 exact 및 고정 normalized
identity로 제외한다. 세 역할군은 같은 ordered raw stream을 각자의 sealed tokenizer로 encode한다.
Tokenizer는 새 corpus에 맞춰 재학습하지 않는다.

## Initialization

Source는 기존 `dense_v2048` checkpoint다. Target tokenizer merge tree가 source 2K tokenizer의
exact extension인지 검증하고, 각 target token을 source subpieces로 byte-exact 분해한다.

- 기존 2,048 rows: input/output 모두 source checkpoint에서 exact copy
- 새 input rows: constituent source embeddings의 uniform mean 후 source mean row norm에 맞춤
- 새 output rows: constituent source output rows의 source-piece byte-length weighted mean
- input/output: untied
- Transformer body: exact source checkpoint copy

새 generic initializer는 8,192에서 prior fresh-v1 builder와 전체 state가 bitwise 동일해야 plan을
봉인할 수 있다. 이 invariant가 16,000 일반화 과정에서 기존 recipe가 바뀌지 않았음을 보장한다.

Parameter count는 다음과 같이 사전 공개한다.

| vocab | parameters | vs 2K |
|---|---:|---:|
| 2,048 tied | 19,667,328 | baseline |
| 8,192 untied | 25,172,352 | +27.99% |
| 16,000 untied | 31,168,896 | +58.48% |

더 큰 checkpoint/weight memory는 숨기지 않으며, actual 단계에서도 별도 보고한다.

## Training contract

- one model seed; development fail-fast
- exact raw order, permutation 없음
- effective batch: 32 token sequences
- microbatch: 2K=32, 8K=8, 16K=4
- evaluation batch: 2K=64, 8K=16, 16K=8
- AdamW betas `(0.9, 0.95)`, epsilon `1e-8`
- body LR `3e-5`
- head peak/min LR `3e-4 / 3e-5`
- raw-target-byte progress 5% warmup + cosine decay
- matrix weight decay `0.1`, vector decay `0`, gradient clip `1.0`

Microbatch 4와 evaluation batch 8은 이 Mac에서 이미 수행된 16K BPE training feasibility
contract와 동일하다. Plan 생성은 exact token inventory, optimizer step 수, raw-target-byte schedule,
document denominator를 다시 계산해 봉인한다.

Plan 봉인 전 `scripts/preflight_fresh_vocabulary_16k_training.py`가 candidate graph의 첫 complete
effective batch를 MPS에서 한 번 실행해 finite 여부와 microbatch memory feasibility만 확인한다.
Loss scalar, calibration score, checkpoint 또는 update vector는 출력·저장하지 않으며 이 smoke는
quality/efficiency evidence가 아니다. 실패하면 plan을 열지 않고 resource contract를 새 버전으로
고쳐야 한다.

Two-stage control은 total target raw bytes의 60%를 처음 넘는 complete effective batch까지 lexical
rows만 학습한다. Source rows는 gradient zeroing 후 매 step exact restore하고 body는 고정한다.
Stage 2 시작 시 optimizer를 재초기화해 전 parameter를 학습한다.

Update-geometry roles는 매 AdamW step 뒤 새 input/output rows의 update vector에 다음 multiplier를
곱한다.

- input: `1.485414522979104`
- output: `2.170601418278963`

이 값은 fresh-v2 calibration이나 16K loss로 조정하지 않는다.

## Calibration quality gate

모든 판단은 per-document NLL과 raw-byte denominator를 사용한다. 10,000회 paired document
bootstrap, seed `20260841`을 고정한다.

### Gate A — 두 anchor 모두에 non-inferior

`dense16k_update_geometry - anchor`의 BPB 차이에 대해 다음을 2K와 fixed 8K 각각 요구한다.

```text
point <= +0.010 BPB
upper 95% <= +0.010 BPB
```

관측상 더 좋은 anchor 하나만 사후 선택하지 않는다. 두 비교 모두 통과해야 한다.

### Gate B — 두 16K control 모두에 method advantage

Candidate가 standard joint와 two-stage 각각에 대해 다음을 만족해야 한다.

```text
point <= -0.002 BPB
upper 95% <= 0
```

Gate A와 B가 모두 통과할 때만 candidate checkpoint 하나가 actual preflight authorization을 얻는다.
8K-vs-2K non-inferiority도 같은 margin으로 별도 계산한다. 이것이 실패하면 cross-vocabulary
geometry generalization claim은 금지하지만, 16K candidate의 직접 Gate A/B 판정을 바꾸지는 않는다.

## Independent replay와 stop rule

각 worker는 checkpoint와 contiguous/document NLL을 저장한다. Summarizer는 다섯 checkpoint를
새로 load하고 calibration forward 전체를 독립 재실행해 float arrays의 bitwise equality를 요구한다.
저장 scalar BPB나 worker pass boolean만 신뢰하지 않는다.

- quality fail: actual timing, multi-seed, final test 모두 실행하지 않음
- quality pass: fixed 16K candidate만 trained actual controlled/free preflight로 이동
- actual pass: 두 mode 각각 point `>=10%` 및 고정 uncertainty/stability gate 통과 시에만 multi-seed
- actual fail: “token-count proxy improvement”로 성공 판정을 대체하지 않음

이 one-seed 결과는 publication claim이 아니다. 같은 raw shard의 disjoint split이므로 source-domain
replication도 아니다.
