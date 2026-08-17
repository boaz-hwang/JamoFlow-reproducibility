# Phase 1 Neural Patching Protocol

> 작성일: 2026-08-10  
> 상태: **결과 확인 전 사전 고정**  
> 선행 단계: [Stage 2 공개 corpus 결과](./07-stage2-public-corpus-results.md)  
> 목적: 작은 BLT에서 boundary placement와 auxiliary patcher cost를 분리해 검증

## 0. 이번 단계에서 바뀐 연구 질문

Phase 0의 `score_evaluations_per_byte`는 learned router의 실제 FLOPs가 아니다. 공식 BLT entropy patcher는 기본적으로 별도 causal Transformer가 모든 byte 위치의 logits를 계산한 뒤 entropy를 구한다. 후보 위치를 사후 mask해도 Transformer backbone은 이미 전체 위치에서 실행되므로, codepoint 후보 수가 60% 줄었다고 곧바로 router FLOPs가 60% 줄지는 않는다.

[BLT 논문](https://aclanthology.org/2025.acl-long.453/)의 기본 entropy model은 100M parameters, 14 layers, 512-byte sliding window다. 반면 논문 부록의 `BLT-FLOPs` 식은 local encoder, global Transformer, local decoder, cross-attention만 더하며 이 auxiliary entropy model을 포함하지 않는다. 공식 코드의 realtime patching도 별도 entropy Transformer를 로드해 전체 byte logits를 계산한다.

이 omission만 지적하는 것으로는 충분한 신규성이 없다. [H-Net](https://openreview.net/forum?id=ZbfLR9NbNF)은 boundary를 end-to-end로 학습하고, [Scratchpad Patching](https://arxiv.org/abs/2605.09630)은 encoder 위의 작은 auxiliary predictor를 비용에 포함한 shared-backbone 실험을 이미 제시한다. 따라서 Phase 1의 질문은 다음 두 개로 제한한다.

> **RQ1 — Boundary necessity:** 동일한 global patch 수에서 UTF-8 codepoint 내부에 entropy boundary를 허용하는 것이 한국어·중국어·영어의 held-out BPB를 실제로 개선하는가?

> **RQ2 — Patcher tax:** BLT backbone뿐 아니라 boundary predictor까지 포함하면 entropy patching의 training/inference cost는 fixed 또는 codepoint-aligned rule보다 얼마나 큰가?

이번 단계는 “한글 규칙이 BLT를 이긴다”를 검증하지 않는다. generic UTF-8 효과와 Korean-specific 효과를 분리하고, 다음 큰 실험을 할 가치가 있는지 판정한다.

## 1. 주장 수준

Phase 1은 약 1.25M-parameter compact BLT와 Wikipedia sentence corpus를 사용하는 controlled mechanism study다. 성공하더라도 다음은 주장하지 않는다.

- 1B 이상 BLT에서 같은 효과가 유지된다는 scaling claim
- downstream task 또는 장문 생성 품질 개선
- production CUDA kernel에서 같은 wall-clock 비율
- Hangul/Jamo 전용 architecture의 우월성
- candidate restriction만으로 dense Transformer router FLOPs가 줄어든다는 주장

Phase 1이 직접 판정하는 것은 **같은 작은 backbone·같은 patch budget에서 경계 위치가 BPB에 미치는 영향**과 **측정 대상 hardware에서 auxiliary router가 더하는 비용**이다.

## 2. 구현 기준선과 고정 버전

### 2.1 Architecture fidelity

주 모델은 Hugging Face Transformers의 `BltForCausalLM`을 사용한다. 이 구현은 BLT의 세 핵심 경로를 유지한다.

1. byte-resolution local encoder
2. patch-resolution global Transformer
3. global patch state를 byte-resolution local decoder로 되돌리는 cross-attention

`patch_in_forward=False`로 두고 policy별 `patch_lengths`를 외부에서 공급한다. 이 방법은 backbone을 바꾸지 않은 채 segmentation만 통제한다.

- `transformers==5.14.1`
- `torch==2.13.0`
- Python 3.13
- 공식 Meta BLT 구현 대조 commit: `9774ed4fcc78313f9f218295f3d7e4decdadf2ae`
- Hugging Face source 대조 commit: `fd12552d770f745fdbe41031ff4daa688f5ed57e`

공식 Meta 구현은 Python 3.12, CUDA nightly PyTorch, xFormers에 고정되어 있어 현재 Apple Silicon에서 그대로 학습하는 기준선으로 부적합하다. 모델 의미를 유지하면서 CPU/MPS에서 실행되는 Hugging Face 구현을 사용한다.

### 2.2 Compact BLT configuration

| Component | Configuration |
|---|---|
| Vocabulary | 256 raw byte values |
| Context | 256 bytes |
| Local encoder | 1 layer, width 64, 4 heads, FFN 192 |
| Global Transformer | 4 layers, width 128, 4 heads, FFN 384 |
| Local decoder | 2 layers, width 64, 4 heads, FFN 192 |
| Cross-attention | `k=2` |
| Hash byte group | size 3, vocabulary 2,048, one function |
| Dropout | 0 |
| Parameters | 약 1.25M, patcher 제외 |

이는 full-scale BLT를 축소한 모델이다. 공식 BLT의 3–8 byte hash groups와 500K buckets를 그대로 쓰지 않으므로 결과는 architecture mechanism pilot으로만 해석한다.

### 2.3 Learned entropy router

동일 library의 `BltPatcher`를 별도 next-byte LM으로 학습한다.

| Item | Configuration |
|---|---|
| Layers | 2 |
| Width | 64 |
| Heads | 4 |
| FFN | 192 |
| Context | 256 bytes |
| Objective | causal next-byte cross entropy |
| Training data | 주 모델과 같은 balanced train bytes |

각 main-model seed에 대응하는 router seed를 하나씩 학습한다. 같은 seed의 `entropy_full`과 `entropy_codepoint`는 **동일 router logits**를 사용한다. 이를 통해 router 품질 차이가 아니라 candidate restriction만 비교한다.

## 3. 데이터

Phase 0에서 고정한 Leipzig Wikipedia 100K sentence corpus를 그대로 사용한다.

- Korean 2021
- Chinese 2018
- English 2016

archive와 processed JSONL의 SHA-256은 [manifest](../data/manifests/leipzig-wikipedia-100k.json)에 기록되어 있다. raw text와 record identifier는 Git에 넣지 않는다.

### 3.1 Split과 byte budget

Phase 0의 normalized-text hash deduplication과 content-hash split을 재사용한다.

- train: hash split의 train partition에서 언어별 최대 6,000,000 bytes
- validation/calibration: calibration partition에서 언어별 500,000 bytes
- test: test partition에서 언어별 1,000,000 bytes

세 언어는 train byte 수를 같게 맞춘다. 언어별 record를 newline byte로 연결한 뒤 256-byte sequence로 자른다. sequence가 UTF-8 codepoint 중간에서 시작할 수 있으므로 codepoint state는 split 전체 stream에서 먼저 계산하고 sequence에 투영한다. 모델의 context는 sequence마다 reset되며 이 artifact는 모든 policy에 동일하다. 시작·끝에서 잘린 codepoint 비율을 결과에 별도로 보고한다.

데이터는 문장 단위 corpus이므로 newline을 document boundary로 해석하지 않는다. 장문 문맥 성능은 이번 단계의 평가 대상이 아니다.

## 4. Primary boundary policies

모든 조건은 256 input bytes마다 실제 data patch 43개를 만든다. 따라서 평균은 정확히 `256 / 43 = 5.953` bytes/patch다. HF BLT가 decoder shift에 사용하는 initial dummy patch 하나를 앞에 추가하므로 전달되는 `patch_lengths` 열 수는 44개지만, compute-rate 통계에서는 dummy를 제외한다.

### P0 — `fixed_byte`

6 bytes마다 boundary를 두고 마지막 patch만 4 bytes로 둔다. router가 없는 값싼 control이다. UTF-8 codepoint 내부 경계를 허용한다.

### P1 — `fixed_codepoint`

현재 sequence에 들어 있는 causal UTF-8 codepoint boundary 후보 중 42개를 목표 위치 `j × 256 / 43`에 가장 가깝게 선택한다. 후보가 겹치지 않도록 순서를 보존하고 뒤의 후보 수를 남긴다. router가 없는 structural control이다.

이 equal-count selection은 전체 256-byte evaluation window를 알고 수행하므로 online generation policy가 아니다. 목적은 P0와 global compute를 정확히 맞춘 상태에서 codepoint 내부 경계를 금지하는 효과만 분리하는 것이다.

### P2 — `entropy_full`

별도 router가 prefix에서 예측한 next-byte distribution의 entropy를 모든 가능한 위치에서 계산한다. 시작점을 제외한 255개 위치 중 entropy가 가장 높은 42개를 boundary로 선택한다.

### P3 — `entropy_codepoint`

P2와 동일한 router entropy를 사용하되, 이미 소비한 prefix가 완전한 UTF-8 codepoint로 끝나는 위치만 후보로 허용한다. 그 후보 중 entropy가 가장 높은 42개를 선택한다.

### 4.1 중요한 causality 구분

“현재 prefix가 codepoint boundary state인가”는 causal하다. 그러나 sequence 안의 top-42를 고르는 것은 미래 위치들의 score와 비교하므로 online causal policy가 아니다. Primary experiment는 official BLT의 global top-k patch allocation과 같은 **offline matched-rate boundary-quality experiment**다.

Phase 2에서 threshold를 calibration split에 고정한 streaming policy를 별도로 평가한다. Primary 결과를 generation latency claim으로 사용하지 않는다.

## 5. Training protocol

### 5.1 Paired design

Seed는 다음 다섯 개로 고정한다.

```text
1729, 2718, 31415, 57721, 65537
```

각 seed에서 네 policy는 다음을 공유한다.

- 같은 initial model state
- 같은 train sequence 순서
- 같은 optimizer schedule
- `entropy_full`과 `entropy_codepoint`의 같은 router checkpoint와 score

따라서 policy 차이를 paired comparison으로 분석한다.

### 5.2 Optimization

- one pass over 18M balanced train bytes
- batch size 32 sequences = 8,192 bytes/step
- AdamW: `beta1=0.9`, `beta2=0.95`, `eps=1e-8`
- learning rate `3e-4`
- 100-step linear warmup 후 cosine decay to `3e-5`
- weight decay `0.1`
- global gradient norm clip `1.0`
- float32 MPS training
- dropout 0
- early stopping 및 test-based selection 없음

Router도 같은 train stream을 한 번 통과한다. Router의 learning rate와 optimizer는 주 모델과 같고 batch size는 feasibility smoke test 뒤 고정된 64를 사용한다.

## 6. Primary metrics and inference

### 6.1 Quality

- overall validation/test BPB
- Korean, Chinese, English test BPB
- byte-normalized NLL
- seed별 paired BPB difference
- language × policy interaction

Primary contrast는 다음 세 개다.

1. `entropy_codepoint - entropy_full`
2. `fixed_codepoint - fixed_byte`
3. `fixed_codepoint - entropy_full`

평균 차이, 다섯 paired seed 값, paired t interval을 모두 보고한다. n=5의 interval만으로 안정성을 과장하지 않도록, test sequence를 언어 안에서 resample하고 seed를 상위 단위로 resample하는 hierarchical bootstrap도 함께 보고한다.

### 6.2 Boundary diagnostics

- exact bytes/patch와 patch count
- mean, median, p95, maximum patch length
- boundary-inside-codepoint rate
- boundary-inside-Hangul-syllable rate
- boundary-inside-CJK-ideograph rate
- router entropy at selected boundaries
- P2/P3 boundary overlap과 boundary displacement

### 6.3 Total cost

비용 표는 다음 세 숫자를 구분한다.

1. **BLT-only:** local/global/decoder/cross-attention만 포함
2. **router-only:** entropy predictor와 entropy/output head
3. **end-to-end:** BLT + router + deterministic boundary construction

각각 다음을 보고한다.

- trainable/frozen parameter count
- official-paper-style analytical forward FLOPs/byte
- teacher-forced bytes/second
- batch-1 median latency
- peak allocated MPS memory가 API에서 안정적으로 제공될 경우 그 값
- training에서 router를 사전 계산하는 데 든 별도 wall-clock과 bytes/second

MPS timing은 warmup 후 synchronization을 넣고 최소 30회 반복해 median과 p10/p90을 기록한다. 이것은 Apple M4 Pro의 local measurement이며 CUDA serving 성능으로 일반화하지 않는다.

`entropy_codepoint`도 dense router 전체를 실행하므로 primary cost accounting에서는 P2와 같은 router FLOPs를 가진다. 후보 수에 비례한 가상 sparse-router FLOPs는 **counterfactual upper bound**로만 별도 표기한다.

## 7. Decision criteria

### Gate A — codepoint restriction

다음을 모두 만족하면 codepoint-internal entropy boundaries가 compact BLT에 필수적이지 않다는 후속 가설을 유지한다.

- `entropy_codepoint - entropy_full <= 0.015 BPB`가 세 언어 각각에서 성립
- 다섯 seed 중 최소 네 개에서 차이의 부호와 크기가 같은 결론을 지지
- 특정 언어에서 `>0.03 BPB`의 명백한 손상이 없음

이 margin은 scaling-independent equivalence를 뜻하지 않는다. Phase 2 확장 여부를 정하기 위한 pilot margin이다.

### Gate B — parameter-free boundary value

`fixed_codepoint`가 `entropy_full`의 언어별 BPB에서 0.02 이내이고 router 포함 end-to-end 비용이 10% 이상 낮으면 cheap structural policy를 더 큰 모델에서 검증한다.

그렇지 않으면 rule-only 방향은 종료하고, integrated learned routing 또는 scratchpad 계열을 기준선으로 전환한다.

### Gate C — patcher tax

별도 router가 end-to-end analytical FLOPs 또는 measured batch-1 latency의 10% 이상이면 auxiliary cost accounting을 논문의 중심 empirical finding으로 유지한다. 10% 미만이면 tax 자체를 핵심 기여로 내세우지 않고 boundary/Unicode 분석의 보조 결과로 내린다.

## 8. Planned Phase 2 only after Phase 1

Gate를 통과한 조건만 다음으로 확장한다.

1. calibration threshold를 고정한 causal streaming patching
2. patch-rate 4/6/8 sensitivity
3. router width/layer ablation과 lookup/n-gram control
4. NFC/NFD, compatibility jamo, emoji, code-mixing robustness
5. UTF-8 validity-constrained generation
6. 모델 폭 또는 data budget 한 단계 확대

Fast BLT식 multi-byte generation, Jamo decomposition, 형태소 FST는 Phase 1의 결과로 정당화되지 않는 한 합치지 않는다.

## 9. Reproducibility outputs

추적할 산출물은 다음과 같다.

- data manifest와 split/cap 통계
- exact config JSON
- seed별 train/eval scalar log
- aggregate report와 machine-readable JSON/CSV
- environment/version capture
- source와 tests

corpus text, private vault content, model checkpoints, per-record text는 Git에 넣지 않는다. checkpoints와 derived boundary cache는 ignored `artifacts/` 또는 `data/cache/`에 둔다.

## 10. Pre-experiment deviations policy

OOM, library bug, 잘못된 tensor alignment처럼 실험 자체를 실행 불가능하게 만드는 문제는 고칠 수 있다. 변경 전후를 deviation log에 기록한다. 결과를 본 뒤 margin, seed, primary contrast, data budget을 바꾸지 않는다. 새로운 설정은 exploratory로만 추가한다.
