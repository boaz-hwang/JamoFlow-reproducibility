# 8K compositional vocabulary one-seed quality protocol

> 작성일: 2026-08-14
>
> 상태: resource probe·학습 loss·calibration BPB 관측 전 고정

## 목적

Random-weight systems preflight에서 8K Hangul code는 2K dense와 같은 19,667,328 trainable
parameters로 continuation step을 22.98%, controlled E2E를 19.56% 줄였다. 그러나 같은 8K의
dense·low-rank·generic code도 거의 같은 속도였다. 따라서 다음 단계의 연구 질문은 속도가 아니라
다음 두 가지다.

1. 동일 2K head budget의 8K factorization이 BPE-2K의 raw-byte quality를 보존하는가?
2. Hangul onset/vowel/coda assignment가 compute-identical generic 및 shuffled assignment보다
   일관되게 더 나은가?

두 질문을 분리하기 위해 한 model seed의 여섯 역할을 동일한 Korean raw-byte stream에서 학습한다.
이 결과는 다음 actual-inference 또는 multi-seed 연구로 갈지를 결정하는 development opportunity
gate이며 publication quality claim은 아니다.

## 고정 역할

| role | vocabulary | head | trainable params | 목적 |
|---|---:|---|---:|---|
| dense 2K | 2,048 | tied dense | 19,667,328 | matched-total strongest BPE baseline |
| dense 8K | 8,192 | tied dense | 22,026,624 | same-body quality ceiling |
| low-rank 8K | 8,192 | rank 92 tied factorization | 19,669,888 | 표준 factorization control |
| generic code 8K | 8,192 | 16×128 codebook | 19,667,328 | non-Korean compositional control |
| shuffled-Hangul 8K | 8,192 | 16×128 codebook | 19,667,328 | 분포·graph-matched linguistic null |
| Hangul code 8K | 8,192 | 16×128 codebook | 19,667,328 | primary candidate |

모든 role의 Transformer body는 hidden 384, FFN 1,536, 8 layers, 6 attention/KV heads로
동일하다. Model seed는 20,260,824이고 plan은 각 전체 initial state hash와 vocabulary를 제외한
Transformer body state hash를 봉인한다. 세 codebook role은 초기 trainable code rows가 정확히
같고 token-to-code assignment만 다르다.

Shuffled-Hangul은 token byte length strata 안에서 onset/vowel/coda auxiliary 6 slots를 한 token
단위로 permutation한다. 첫/마지막 Unicode scalar, byte length, token identity slots, slot별 분포,
parameter 수, output graph는 true Hangul과 같다. 따라서 Hangul 대 shuffled 차이는 단순 slot
빈도나 runtime graph가 아니라 linguistic alignment의 opportunity를 검증한다.

## 학습 계약

- source: 기존 clean HPLT3 Korean train split의 처음 128,000,000 raw bytes
- tokenizer: 봉인된 exact byte-BPE 2K 및 8K JSON
- context: 512 tokens
- raw exposure: 정확히 한 stream pass
- effective batch: 32 sequences
- train microbatch: 2K는 32, 모든 8K role은 8
- optimizer: AdamW, peak LR `3e-4`, minimum `3e-5`, betas `(0.9,0.95)`, eps `1e-8`
- warmup: optimizer step의 5%, 이후 cosine decay
- matrix weight decay 0.1, vector weight decay 0, gradient clipping 1.0
- model seed 20,260,824, order seed 20,260,825

2K와 8K는 같은 raw bytes를 보지만 token sequence 수가 다르므로 2K는 70,798 sequences/2,213
steps, 8K는 53,590 sequences/1,675 steps를 사용한다. 이는 이전 BPE frontier와 같은 deployable
raw-data contract이며 2K에 더 많은 optimization compute를 주는 보수적 baseline이다. 8K 내부
다섯 역할은 token IDs, order, step 수, body, microbatch가 정확히 같으므로 head mechanism contrast는
직접적이다.

## 학습 전 resource-only gate

Full training 전에 같은 plan commit에서 여섯 fresh random-weight model을 각각 별도 MPS process로
실행한다.

- 1 warmup + 3 measured effective training steps
- 1 warmup + 3 measured contiguous-evaluation batches
- loss 값은 저장하거나 selection에 사용하지 않고 finite 여부만 확인
- MPS current/driver allocation과 process RSS를 sampled high-water diagnostic으로 기록
- resettable native peak가 아니므로 memory improvement claim에는 사용하지 않음

각 role의 median step/batch time으로 full training+contiguous evaluation core time을 외삽하고 합계에
1.25 safety factor를 곱한다. 다음을 모두 만족해야 full campaign을 자동 허용한다.

1. safety-adjusted projected core time `<= 12 hours`
2. 모든 role의 sampled driver/recommended memory fraction `<= 0.75`
3. 모든 role의 process RSS/physical memory fraction `<= 0.75`

Document evaluation, tokenization, checkpoint serialization은 projection 밖이므로 12-hour gate는
wall-clock guarantee가 아니다. 24-hour operational ceiling보다 보수적인 core budget을 두기 위한
resource gate다. 봉인 전에는 Hangul 8K microbatch 8의 one-step forward/backward가 finite이고 MPS에서
실행된다는 engineering smoke만 확인했다. Loss 값은 기록하지 않았고 role/gate를 바꾸지 않았다.

## 공통 raw-byte 평가

이전 BPE frontier와 동일한 8,000,000-byte calibration stream과 385개 공통 document를 사용한다.

1. **contiguous BPB**: full 512-token sequences에서 첫 token을 context로 제외하고 실제 target raw
   bytes를 분모로 계산한다.
2. **document BPB**: 각 document를 exact one-token NUL context 뒤에서 독립 평가하고 긴 문서는
   context token 하나를 겹쳐 모든 raw bytes를 정확히 한 번 예측한다.
3. **paired document bootstrap**: 같은 385 document NLL 차이를 10,000회 resample한다. 이는
   document 이질성 interval이지 model-seed uncertainty가 아니다.

Worker는 checkpoint, per-sequence/document NLL, denominator, report를 no-clobber로 저장한다. Summary는
각 checkpoint를 새 model에 strict load하고 calibration forward 전체를 다시 실행해 저장 배열과
bitwise 비교한다. 최종 gate는 이 independent replay 배열에서만 계산한다.

## 결과 전에 고정한 contrast gate

모든 차이는 `candidate - reference` BPB라서 음수가 candidate 우위다.

### A. 2K 품질 비열등

Hangul 8K 대 dense 2K에서 다음 세 값이 모두 `<= +0.010 BPB`여야 한다.

- contiguous aggregate difference
- document aggregate difference
- paired document bootstrap 95% upper

### B. Generic 대비 Korean advantage

Hangul 8K 대 generic code 8K에서 contiguous/document point difference가 모두 `<= -0.002 BPB`이고
bootstrap 95% upper가 `<= 0`이어야 한다.

### C. Shuffled 대비 Korean advantage

Hangul 8K 대 shuffled-Hangul 8K에 B와 같은 기준을 적용한다. 이 contrast가 linguistic alignment의
primary ablation이다.

### D. Low-rank 대비 비열등

Hangul 8K 대 low-rank 8K에서 contiguous/document/upper가 모두 `<= +0.002 BPB`여야 한다. 더 복잡한
codebook이 단순 low-rank보다 명확히 나쁘면 한국어 assignment만으로 다음 단계 비용을 정당화할 수
없다.

A--D를 모두 통과해야 `korean_compositional_quality_opportunity_pass`이며 one-seed trained actual
inference를 허용한다. Candidate는 Hangul로 고정하며 결과를 보고 generic, shuffled, low-rank 또는
dense 8K로 바꾸지 않는다.

Generic 8K만 dense 2K 대비 +0.010 비열등을 통과하면
`generic_factorization_only_requires_novelty_reassessment`로 기록한다. 이는 자동 fallback이나
Korean claim이 아니며, 기존 compositional embedding/output 선행연구와 차별성이 충분한지 새
문헌·설계 검토 없이는 scaling하지 않는다. Generic도 실패하면 branch를 종료한다.

## 해석 경계

- Dense 8K는 +2.36M parameter를 쓰는 same-body quality ceiling이며 matched-total 후보가 아니다.
- One model seed와 calibration documents만으로 일반 품질 우월성이나 유의한 model-seed effect를
  주장하지 않는다.
- 이번 단계는 generation latency를 다시 측정하지 않는다. Random-weight speed를 trained speed로
  간주하지 않는다.
- Hangul이 generic만 이기고 shuffled를 이기지 못하면 linguistic prior 증거가 아니다.
- Quality pass 뒤에도 먼저 trained checkpoint의 free-running actual E2E 10% opportunity를 확인하고,
  그 다음에만 multi-seed와 새 sealed final test로 확장한다.

## Commit-separated DAG

1. 이 구현·문서·tests를 commit
2. plan을 별도 commit
3. 같은 plan commit에서 resource probe; pass일 때만 여섯 training worker 실행
4. checkpoint/NLL 전체의 independent replay 뒤 tracked summary 생성·commit
5. 결과가 A--D를 모두 통과할 때만 별도 trained actual-inference protocol을 선고정

Checkpoint와 raw NLL은 개발용 local artifact이고 summary가 그 SHA-256/state/array hash를 봉인한다.
이는 reproducible local evidence이지 원격 append-only one-shot 증명은 아니다.
