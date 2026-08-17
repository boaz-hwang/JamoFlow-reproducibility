# Actual-inference and compute-conversion protocol

> 작성일: 2026-08-11  
> 상태: **incremental latency 및 reduced-rate 결과 생성 전 고정**  
> 상위 protocol: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md)  
> 비용 screen: [cost provenance/stability](./43-phase3-cost-provenance-and-stability-addendum.md)  
> 사후 무결성 교정: [document clustering](./52-document-cluster-inference-integrity-addendum.md), [selection/time-to-output](./53-selection-and-time-to-output-correction.md)
> publication BPE 교정: [dual-BPE sealed-test correction](./61-dual-bpe-sealed-test-correction.md)
> valid-output 교정: [valid-output actual-inference correction](./65-valid-output-actual-inference-correction.md)
> 목적: “실제 모델 추론 효율이 개선된 결과만 연구 가치가 있다”는 최종 판정 기준을 구현·통계·scale gate로 명시함

## 1. 최종 가치 기준의 변경이 아니라 명문화

기존 Gate K는 teacher-forced 512-byte window에서 router-inclusive cost를 검사한다. 이는 method 후보를 빠르게 걸러내는 데는 유용하지만 실제 autoregressive inference 개선을 뜻하지 않는다. 특히 W86은 C86/F86과 global patch 수가 같으므로, boundary 위치만 바뀌어 품질이 좋아져도 자동으로 decode가 빨라지지 않는다.

이 연구에서 positive efficiency paper로 인정할 최종 명제는 다음뿐이다.

> **동일한 Korean quality 기준을 만족하는 가장 강한 비교 방법보다, prefix-causal patching method가 실제 incremental autoregressive 실행에서 end-to-end batch-1 latency를 유의미하게 줄이는가? 그리고 이 효과가 Mac에서 실행 가능한 가장 큰 publication-scale model에서도 유지되는가?**

BPB 개선만 있거나 analytical FLOPs만 줄거나 teacher-forced throughput만 좋아지는 결과는 사용자의 기준에서 연구 성공으로 판정하지 않는다.

## 2. 수정 시점 공개

이 protocol을 고정할 때 initial seed 1,729와 2,718의 F/C/W artifact는 존재했고 seed 31,415의 F가 학습 중이었다. 완결된 3-seed primary summary, OOD, S/E/EC, reduced-rate model, incremental latency 결과는 존재하지 않았다. Reduced-rate 후보 72와 64는 각각 기존 86 대비 global data-patch 수를 약 16.3%와 25.6% 줄이는 architecture-derived 두 단계로 고정하며 partial BPB에 맞춰 선택하지 않았다.

## 3. HF BLT cache 감사와 독립 runtime

Pinned Transformers `5.14.1`의 `BltModel.forward`는 cache 인자를 받지만 publication-grade incremental decoder로 바로 사용할 수 없다.

- local encoder와 local decoder self-attention에는 cache가 전달된다.
- global Transformer에는 `past_key_values=None`이 고정되어 매 호출 전체 global input을 다시 계산한다.
- 한 byte만 다시 넣으면 rolling hash embedding의 앞선 byte group이 사라진다.
- `BltPatcher.forward`는 `DynamicCache`를 만들지만 transformer layer 호출에 cache를 전달하지 않는다.
- open/closed patch와 HF dummy-patch lag를 generation state로 관리하지 않는다.

따라서 `src/jamoflow/incremental_blt.py`는 학습 weight를 바꾸지 않고 batch 1에서 다음 상태를 명시적으로 유지한다.

1. rolling-hash에 필요한 최근 raw bytes
2. local encoder KV cache
3. 현재 encoder patch의 local states와 max reduction
4. 닫힌 global patch KV cache
5. 현재 decoder가 참조하는 lagged global state
6. local decoder KV cache
7. prefix-causal structural boundary state

같은 runtime은 F/C/W의 UTF-8·whitespace parser와 target state를 유지해 prefix 전체를 매번 다시 읽지 않고 byte당 constant-time으로 boundary를 갱신한다. E/EC용 router layer에는 실제 KV cache를 전달하고, 이전 byte의 next-byte entropy를 다음 boundary position에 정렬한다. Structural 및 learned policy 모두 prompt 전체를 한 번에 처리하면서 main/router cache와 닫힌 global patch만 구축하는 parallel prefill 경로와, byte-by-byte correctness reference를 별도로 둔다. 이 구현 사실은 latency 결과가 아니며 아래 equivalence를 통과한 실행만 측정에 사용한다.

HF 정렬에서 decoder boundary byte `b_j`를 관측하면 encoder patch `j`가 닫히고 global token `j`를 한 번 계산한다. 그 global state는 다음 boundary 전까지 decoder bytes가 공유한다. 이는 매 byte global trunk를 다시 실행하지 않으면서 full-prefix graph의 one-byte lag를 보존한다.

### 3.1 equivalence prerequisite

어떤 latency도 다음 검사를 먼저 통과하지 않으면 증거로 사용하지 않는다.

- F/C/W 각각에서 모든 prompt byte와 boundary 전후 byte 비교
- full-prefix `use_cache=False` logit과 incremental logit의 `rtol=2e-5`, `atol=2e-5` allclose
- next-byte argmax 100% 일치
- local encoder/decoder cached byte 수가 observed byte 수와 일치
- global cached token 수가 emitted boundary 수와 일치
- checkpoint, source, prompt-selection, runtime commit hash 기록

Unit test는 작은 무작위 초기화 graph의 모든 prefix에서 이 조건을 고정한다. Evidentiary run은 실제 Phase 3 checkpoint와 held-out prompt에서 별도로 재검증한다. Built-in HF `generate()`와의 속도 비교는 하지 않는다. 잘못 정렬된 cache 경로를 느린 baseline으로 삼으면 허위 speedup이 되기 때문이다.

## 4. 품질 여유를 실제 compute 감소로 바꾸는 조건부 실험

W86의 same-rate 신호가 Gate I를 통과할 때만 reduced-rate study를 실행한다. Gate I가 실패하면 이 실험으로 W를 사후 구제하지 않는다.

### 4.1 고정 후보

동일한 19,596,096-parameter graph와 128M train bytes에서 다음 네 model을 initial seeds 1,729/2,718/31,415로 처음부터 학습한다.

| Policy | Data patches / 512 bytes | 기존 86 대비 감소 |
|---|---:|---:|
| C72 | 72 | 16.28% |
| W72 | 72 | 16.28% |
| C64 | 64 | 25.58% |
| W64 | 64 | 25.58% |

각 rate에서 C와 W는 같은 model initialization, training order, bytes, optimizer step을 공유한다. Global maximum position capacity는 모든 rate에서 기존 Phase 3와 같게 유지해 parameter count와 graph capacity를 바꾸지 않는다.

### 4.2 calibration-only 선택

Test BPB를 보기 전에 initial 3-seed calibration BPB로 하나의 W rate를 선택한다.

1. W64의 mean calibration BPB가 C86보다 `+0.010 BPB` 이내이고 세 seed 중 최소 두 seed에서 그 margin 안이면 W64를 선택한다.
2. 아니면 같은 조건을 만족하는 W72를 선택한다.
3. 둘 다 실패하면 compute-conversion branch는 실패다.

낮은 patch count를 우선하는 rule과 margin은 고정이다. 선택 뒤 initial test에서 다음을 모두 검사한다.

- selected W-rate minus C86 mean test BPB `<= +0.010`
- selected W-rate minus same-rate C mean test BPB `<= -0.002`
- 최소 2/3 seed에서 selected W-rate minus C86가 `<= +0.010`
- 최소 2/3 seed에서 selected W-rate minus same-rate C가 negative

통과하면 selected W-rate와 same-rate C만 confirmation seeds 57,721/65,537에 추가한다. Final five-seed quality noninferiority는 selected W-rate minus 사전 선택된 primary inference comparator의 paired-seed 95% upper bound와 document-cluster 95% upper bound가 모두 `+0.010 BPB`보다 작고, 최소 4/5 seed가 margin 안이어야 한다. Same-rate attribution은 W−C mean `<= -0.003`, 최소 4/5 negative, 기존 Holm/paired-seed 및 document-cluster inference 규칙을 따른다.

## 5. 실제 autoregressive benchmark

### 5.1 비교 방법 선택

Initial latency와 test noninferiority를 보기 전에 initial 3-seed **mean calibration BPB**가 가장 낮은 policy를 primary inference comparator로 고정한다. 후보는 실행 가능하고 lineage가 완전한 F86, C86, W86, S, E, EC, selected C-rate다. Exact tie는 이 candidate order로 푼다. 선택된 comparator가 아직 confirmation seed에 학습되지 않은 S/E/EC 또는 selected C-rate이면 같은 두 confirmation seed를 추가한다. Selected W-rate가 고정 comparator보다 `+0.010 BPB` 이내라는 five-seed test noninferiority를 통과하지 못하면 speed와 무관하게 실패한다. Comparator를 test BPB나 latency가 유리한 방법으로 사후 교체하지 않는다.

Publication scale에서는 16K와 32K standard Korean byte-BPE Transformer, 그리고 실행 가능한 learned dynamic-boundary baseline을 같은 raw training source와 별도 표로 반드시 포함한다. Raw byte model 안에서만 빠르다는 결과와 두 tokenized control까지 포함해 빠르다는 결과를 구분한다.

### 5.2 held-out inputs

- public HPLT3 test split만 사용
- 128-byte strict-UTF-8, Hangul-heavy prompt 64개
- prompt selection은 content hash ordering으로 고정하고 원문은 result에 저장하지 않음
- controlled replay continuation 128 bytes는 같은 held-out row의 실제 다음 bytes
- free-running shared-DFA greedy는 최소 128 valid UTF-8 bytes를 완성한 첫 accept state까지 생성(128--131 byte)
- prompt와 continuation을 포함한 512-byte source window 전체가 한 원문 문서 안에 있음
- warmup 8개와 measured 64개를 합친 72개 case는 서로 다른 72개 원문 문서에서 하나씩 선택

Private Markdown은 latency primary에 사용하지 않는다.

### 5.3 측정 두 축

**Controlled incremental replay**는 모든 method에 동일한 prompt와 continuation bytes를 사용해 N개 conditional output을 계산한다. Prompt prefill final logit이 첫 output을 이미 예측하므로 N개 output의 정확한 time-to-output에는 N−1 incremental forwards만 필요하다. 이는 output divergence 없이 cache update, selector, router, local/global compute를 포함한 decode runtime을 비교하는 primary systems estimand다.

**Free-running shared-DFA greedy generation**은 prefill logit에서 strict RFC 3629 transition mask를 적용한 첫 argmax byte를 얻는다. Horizon closure로 마지막 byte를 ASCII에 맞추지 않고, 최소 128 bytes 이후 첫 UTF-8 accept state에서 멈춘다. 따라서 128--131 output bytes와 127--130 feedback forwards가 가능하다. Static DFA mask compilation만 timing 밖이며 mask 적용, byte 선택, device synchronization, state/stop 검사, selector/router와 cache update는 timing 안이다. 이는 실제 valid-output generation 확인 실험이다. 서로 다른 continuation 때문에 생기는 input 경로 차이를 숨기지 않고 secondary로 보고한다.

각 prompt에서 다음을 별도로 측정한다.

- TTFT: fresh runtime 생성부터 128-byte prompt prefill 완료까지
- decode: prefilled state에서 128 bytes 처리/생성
- end-to-end: runtime 생성, prefill, decode 전체
- milliseconds/byte와 bytes/second
- emitted global patches와 bytes/global-patch
- emitted output bytes, overshoot, decode forward steps, runtime-observed bytes
- process RSS 및 가능한 MPS allocator snapshot

Byte-by-byte `prefill()`은 cache state 검증용이다. Final TTFT와 end-to-end 수치에는 prompt의 local states, closed global patches, decoder cache를 병렬로 구축하는 `prefill_parallel()`만 사용한다. 이 경로도 final logit, cache length, 이어지는 16 decode-byte logits가 sequential reference와 같은 equivalence prerequisite를 통과해야 한다. Sequential-prefill 시간은 diagnostic으로만 남기며 Final Value Gate에 넣지 않는다.

### 5.4 timing discipline

- batch 1 primary
- policy 순서를 prompt와 repetition마다 seeded randomization
- 8 warmup prompts, 64 measured prompts
- prompt당 5 independent repetitions
- 전체 session 시작과 각 seed 측정 직전·직후에 AC 전원, 기본 power mode(`0`), thermal/performance warning 부재를 확인
- 모든 device 구간 전후 synchronize
- model loading, source loading, checkpoint hashing은 timing 밖
- selector와 learned router는 timing 안
- raw prompt·continuation·generation text는 tracked output에 저장하지 않음

전원·power mode·thermal 조건을 만족하지 않은 seed artifact는 저장하거나 재사용하지 않는다. 실행기와 요약기는 `src/jamoflow/actual_inference_protocol.py`의 protocol version, 반복 수, time-to-output horizon, 환경 판정을 함께 사용한다. Manifest, seed report와 timing-array shape를 독립 재구성하는 요약 단계가 이 값 가운데 하나라도 다르면 evidence로 승격하지 않는다.

Prompt `p`와 policy `A`의 다섯 repetition median을 `m_{A,p}`라 한다. Primary reduction은 `1 - median_p(m_{W,p}) / median_p(m_{L,p})`로 고정한다. 각 prompt가 서로 다른 원문 문서에서 오므로 model seed와 source-document/prompt index를 교차하되 prompt는 policy 간 paired 상태로 10,000회 복원추출해 같은 ratio statistic의 percentile 95% interval을 계산하고, seed별 reduction과 `mean_p(1 - m_{W,p}/m_{L,p})`도 stability diagnostic으로 보고한다. Runtime repetition만 독립 표본처럼 세지 않는다.

## 6. Final Value Gate

Positive inference-efficiency result는 다음을 모두 만족해야 한다.

1. 실제 checkpoint의 incremental/full-prefix equivalence prerequisite 통과
2. five-seed paired-seed 및 document-cluster BPB noninferiority upper bound가 모두 `< +0.010 BPB`, coverage `>= 95%`, 최소 4/5 seed가 margin 안
3. controlled replay batch-1 decode median latency reduction `>= 10%`
4. 같은 reduction의 paired-prompt bootstrap 95% lower bound `> 0`
5. free-running valid-output greedy end-to-end latency reduction `>= 10%`
6. free-running paired-prompt bootstrap 95% lower bound `> 0`
7. candidate와 comparator의 free-running output이 모두 strict-valid이며, replacement-character-free mean regression이 2pp 이내이고 최소 4/5 seed가 같은 margin 안
8. speedup이 selector/router, synchronization, cache update를 포함함
9. held-out BPB 외에 publication-scale Korean downstream score가 결과 확인 전 고정한 task별 noninferiority margin을 통과함
10. 아래 publication-scale replication에서도 1–9 유지

하나라도 실패하면 teacher-forced 또는 FLOP 이득만으로 inference paper 성공을 선언하지 않는다. 결과는 진단 자료로 남기되 사용자가 정의한 “연구 가치 있는 효율 개선”에는 실패다.

## 7. Mac publication scale

현재 장비는 Apple M4 Pro, unified memory 48 GB다. Gate I/J와 reduced-rate quality, compact Final Value Gate를 통과한 뒤 다음 blind feasibility 순서로 model scale을 고른다.

1. 50M, 75M, 100M 후보 graph의 단일 train/eval/incremental step memory와 wall time을 quality 결과와 무관하게 측정한다.
2. optimizer, activation, checkpoint와 benchmark 동시 상주 여유를 포함해 48 GB 안에서 안전하게 실행되는 가장 큰 후보를 선택한다.
3. 최소 256M public Korean train bytes를 사용한다.
4. selected W-rate, primary inference comparator, 16K와 32K standard byte-BPE Transformer를 최소 비교한다.
5. main positive comparison은 최소 3 seeds 또는 compact variance 기반 사전 power justification을 사용한다.

Mac MPS 결과를 CUDA 결과로 표현하지 않는다. 외부 CUDA replication을 확보하면 별도 hardware table로 추가하지만, 확보하지 못했다는 이유로 Mac에서 실제 측정을 teacher-forced proxy로 대체하지 않는다.

## 8. 실패 시 연구 전환

Gate I 또는 compute conversion 또는 Final Value Gate가 실패하면 현재 W를 punctuation, morphology, Jamo feature로 사후 수정해 살리지 않는다. 사용자의 성공 기준을 계속 추구할 다음 독립 연구는 새 protocol로 preregister한다.

우선순위는 sequential step 수를 직접 줄이는 **orthography-constrained multi-byte proposal + verification**이다. UTF-8/Hangul DFA는 semantic byte를 대신 생성하는 엔진이 아니라 learned proposer의 invalid branch를 제거하는 verifier로만 쓴다. Baseline과 같은 model/quality budget에서 accepted bytes per expensive forward, rejection, actual batch-1 latency를 측정한다. 이 pivot 역시 실제 end-to-end speedup이 없으면 positive paper로 판정하지 않는다.

## 9. 허용되는 최종 주장

| Evidence | 허용 주장 |
|---|---|
| Gate J만 통과 | same-rate boundary-quality effect |
| Gate K만 추가 통과 | teacher-forced/router-inclusive Pareto screen |
| compact Final Value Gate 통과 | 19.6M MPS setting의 actual incremental speedup |
| publication scale까지 통과 | Korean inference-efficiency method paper 후보 |
| 두 byte-BPE 중 하나보다 느림 | vocabulary-specific 또는 byte-latent family 내부 개선으로 한정 |
| raw·16K BPE·32K BPE 모두 포함한 frontier 개선 | broader Korean LM inference claim 검토 가능 |

논문 제목과 초록의 `efficient inference`, `faster generation`은 publication-scale Final Value Gate 뒤에만 사용한다.
