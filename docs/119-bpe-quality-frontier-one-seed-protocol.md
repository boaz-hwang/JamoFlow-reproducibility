# BPE one-seed quality frontier protocol

> 작성일: 2026-08-14
>
> 상태: 여섯 모델의 학습 loss·calibration BPB를 보기 전 고정

## 목적

Random-weight systems frontier에서는 32K×8L가 point estimate상 가장 빨랐지만 64K×8L와의
불확실성 구간이 겹쳤다. 더 중요한 문제는 실제로 학습한 모델에서 vocabulary와 graph가
quality를 바꾼다는 점이다. Token count나 random-weight latency만으로 BPE comparator를 고르면
느리지만 더 강한 baseline을 빠뜨리거나, 빠르지만 품질이 떨어지는 baseline을 채택할 수 있다.

이 단계는 vocabulary별 fastest graph 여섯 개를 동일한 Korean raw-byte budget으로 한 번씩
학습해 systems와 quality의 공동 프런티어를 찾는다. 결과는 candidate 설계용 개발 증거이며,
publication comparator 또는 multi-seed matched-quality 증거가 아니다.

## 고정 역할과 모델

| vocabulary | graph | parameters | train token sequences | optimizer steps |
|---:|---:|---:|---:|---:|
| 2,048 | 8L | 약 19.6M | 70,798 | 2,213 |
| 4,096 | 12L | 약 19.6M | 60,916 | 1,904 |
| 8,192 | 8L | 약 19.6M | 53,590 | 1,675 |
| 16,000 | 8L | 약 19.6M | 48,040 | 1,502 |
| 32,000 | 8L | 약 19.6M | 43,436 | 1,358 |
| 64,000 | 8L | 약 19.6M | 39,801 | 1,244 |

각 graph는 systems frontier에서 해당 vocabulary의 실제 end-to-end median이 가장 짧았던
parameter-matched tied-embedding Llama graph다. Model seed는 하나로 고정하고, role별 초기
state hash를 plan에 기록한다.

## 학습 계약

- train source: 동일한 clean HPLT3 Korean train stream 128,000,000 raw bytes
- tokenizer: systems frontier에서 봉인한 exact ByteLevel BPE 2K/4K/8K/16K/32K/64K
- context: 512 tokens
- order: role마다 full token-sequence index를 동일 seed로 독립 permutation
- effective batch: 32 token sequences
- vocabulary별 microbatch: 32/16/8/4/2/1, gradient accumulation으로 effective batch 고정
- optimizer: AdamW, peak learning rate `3e-4`, minimum `3e-5`, betas `(0.9, 0.95)`,
  eps `1e-8`
- warmup: 전체 optimizer step의 5%, 이후 cosine decay
- matrix parameter weight decay 0.1, vector parameter 0, gradient clipping 1.0
- 정확히 한 raw-stream pass, model seed 20,260,817, order seed 20,260,818

여기서 계산량은 role 사이에 같지 않다. 같은 raw bytes가 작은 vocabulary에서는 더 많은
token sequence와 optimizer step을 만든다. 이는 작은 vocabulary에 더 많은 학습 compute를 주는
보수적 baseline 강화이며, vocabulary만의 인과 효과를 추정하는 실험이 아니다. 이번 목적은
candidate가 이겨야 할 가장 강하고 빠른 실제 BPE 시스템을 놓치지 않는 것이다. 학습 시간과
optimizer step을 모두 결과에 공개한다.

## 공통 raw-byte 품질 평가

Token 평균 loss는 vocabulary 사이에서 비교할 수 없으므로 모든 핵심 지표를 raw byte당 bit,
즉 BPB로 변환한다.

### 1. 연속 스트림 BPB

동일한 8,000,000-byte calibration stream을 각 tokenizer로 encode하고 full 512-token sequence를
평가한다. 각 sequence의 첫 token은 문맥으로만 쓰고, 나머지 511 target token이 정확히 나타내는
원시 byte 수를 분모로 사용한다. Tokenizer별 마지막 불완전 token tail 차이 때문에 실제 예측
byte 수는 plan에 봉인된 inventory와 일치해야 한다.

### 2. 문서별 paired BPB

공통 calibration 문서를 입력 순서대로, 다음 문서를 잘라야 하는 지점 직전까지 선택한다.
각 문서는 exact one-token NUL context 뒤에서 독립 평가하며, 512 tokens보다 긴 문서는 마지막
context token을 겹쳐 모든 문서 byte를 정확히 한 번씩 예측한다. 이 방식은 여섯 tokenizer에서
동일한 385개 문서와 7,977,011 raw bytes를 사용한다.

문서별 NLL 차이를 같은 document index로 pair하고 10,000회 document bootstrap을 수행한다.
이는 한 model seed 안에서 문서 이질성을 측정하는 개발용 interval이며 model-seed 불확실성을
대체하지 않는다.

## 결과 전에 고정한 선택 규칙

1. 연속 스트림 aggregate BPB가 가장 낮은 role을 quality anchor로 고정한다. Exact tie는
   `2K, 4K, 8K, 16K, 32K, 64K` 순서로 푼다.
2. 다음 세 조건을 모두 만족한 role만 quality-qualified로 둔다.
   - 연속 스트림 BPB가 anchor보다 `+0.010 BPB` 이내
   - 문서 aggregate BPB가 anchor보다 `+0.010 BPB` 이내
   - paired document bootstrap 차이의 95% upper가 `+0.010 BPB` 이하
3. 적격 role 중 systems frontier에서 사전에 봉인한 end-to-end median이 가장 짧은 role을
   development BPE comparator로 선택한다. Exact tie는 위 role 순서로 푼다.

어떤 model이 선택될지 보고 margin, bootstrap seed, role pool 또는 systems latency를 바꾸지 않는다.
Anchor 자신은 수학적으로 항상 적격이므로 빈 적격 집합 fallback은 없다.

## 무결성과 재개

- 코드·문서·tokenizer·source·feasibility 결과·systems timing·초기 state·학습 order를 plan에
  hash로 봉인한 다음 plan만 별도 commit한다.
- 각 role은 fresh subprocess에서 단독 MPS lock을 잡아 학습한다.
- checkpoint, per-sequence/per-document NLL, raw-byte denominator, report를 모두 no-clobber로
  저장한다.
- 완성된 세 artifact가 모두 있고 현재 plan/commit/state/array hash와 일치할 때만 해당 role을
  resume-skip한다. 부분 artifact는 자동 삭제하거나 덮어쓰지 않고 forensic failure로 멈춘다.
- 모든 role 완료 뒤 repository HEAD와 tracked status가 시작과 같을 때 campaign report를 쓴다.
- summary는 checkpoint를 다시 load해 calibration forward 전체를 독립 재실행하고 저장 NLL과
  bitwise 비교한 뒤, 재검증된 배열로만 BPB와 선택을 재구성한다.

## Claim 경계와 다음 판단

이 단계가 증명할 수 있는 것은 같은 data budget과 약 19.6M main-model parameter scale에서의
one-seed Korean BPE development frontier다. 다음은 아직 증명하지 않는다.

- multi-seed matched-quality 동등성
- trained-model actual generation latency
- 일반 언어·다른 hardware·다른 model scale로의 일반화
- Korean-aware tokenizer 또는 long-token method의 우월성

선택된 BPE는 다음 개발 실험에서 exact BPE baseline이 된다. 그 뒤 generic long-token control과
Korean-aware variant를 같은 품질·시스템 기준으로 비교한다. 실제 결과가 `+0.010 BPB` margin이나
한-seed instability의 부적절함을 드러내면, 이 결과를 사후 재선택하지 않고 다음 단계의
multi-seed protocol을 새 version으로 고친다.
