# Fresh Korean vocabulary-adaptation one-seed protocol

> 작성일: 2026-08-15
>
> 상태: 결과 비공개 사전 계약; plan seal 전
>
> 성공 기준: quality-qualified trained dense-8K의 controlled/free actual E2E가 모두 10% 이상

## 연구 질문

Historical B1에서 관측된 new-row update 증폭이 새 한국어 원자료에서도 dense-2K 품질을 회복하는가,
그리고 품질을 맞춘 ordinary dense-8K가 실제 batch-1 전체 생성 경로를 10% 이상 줄일 수 있는가를
검증한다. 이번 단계는 앞 질문의 one-seed fail-fast screen이다. Actual inference와 multi-seed
confirmation은 이 단계의 양성 결과가 있을 때만 별도 봉인한다.

Fable 5 검토에서 수용한 원칙은 그대로 유지한다.

- token 수, analytical FLOPs, random-weight latency를 실제 효율 성공으로 부르지 않는다.
- 같은 raw stream을 사용하되 tokenizer별 optimizer step 수와 실제 학습시간을 모두 공개한다.
- 새 optimizer recipe는 강한 ordinary joint 및 최신 two-stage 대조군을 모두 이겨야 한다.
- 추가 parameter와 memory를 숨기지 않고 latency--memory Pareto로 보고한다.

## Fresh data와 공통 시작점

`hplt3-korean-vocab-adaptation-v1`의 train 128,000,000 bytes와 calibration 8,000,000
bytes만 사용한다. 이 stream은 historical Phase-3 6,911문서와 sealed final-test 1,482문서를 exact
및 고정 normalized identity로 제외했고, prepare와 독립 full-rescan이 같은 seal을 냈다.

모든 역할은 동일한 historical trained dense-2K checkpoint에서 출발한다.

- 2K 역할은 checkpoint를 그대로 이어 학습한다.
- 8K 역할은 source BPE의 기존 2,048 rows를 그대로 복사한다.
- 새 input row는 constituent source-token uniform mean 뒤 source mean-L2에 맞춘다.
- 새 output row는 constituent byte-length-weighted mean을 사용하고 output norm은 강제하지 않는다.
- input/output matrix는 untied다.

이는 기존 baseline closure에서 고른
`untied_uniform_in_byte_weighted_out`이며, [Beyond Initialization Loss](https://arxiv.org/abs/2608.03494)의
input/output asymmetric composition 방향과 맞닿아 있다. 정확한 대형 논문 설정 재현이라고 주장하지
않는다.

## 네 역할

| role | vocab | 학습 계약 |
|---|---:|---|
| `dense2k_joint` | 2,048 | source checkpoint, all-parameter AdamW |
| `dense8k_standard_joint` | 8,192 | strongest initializer, ordinary joint AdamW |
| `dense8k_inplace_two_stage` | 8,192 | 처음 60% raw-target bytes에서 새 input/output rows만, 이후 40% full CPT |
| `dense8k_update_geometry` | 8,192 | ordinary joint AdamW 뒤 새 input/output row update에 고정 배수 적용 |

Geometry 배수는 historical validation BPB나 이번 fresh metric으로 고르지 않는다. 봉인된 첫 AdamW
update audit의 projection 값만 사용한다.

- input: `1.485414522979104`
- output: `2.170601418278963`

`dense8k_inplace_two_stage`는
[In-Place Tokenizer Expansion](https://arxiv.org/abs/2607.15232)의 핵심 구조를 compact하게
근사한다. 새 row 이외의 body와 copied rows를 동결하고, stage 1에서 LR을 높은 peak까지 warm-up한
뒤 유지한다. 60% 경계를 처음 넘는 complete effective batch까지 stage 1로 두고, optimizer를
재초기화한 뒤 남은 stream을 full-model CPT한다. 원 논문의 600B+400B 비율을 차용하지만 tied
8B-MoE/다국어 mixture를 재현하는 실험은 아니다. 우리 대조군은 untied input/output 새 row를 둘 다
학습한다.

## 동일 노출과 schedule

두 tokenizer는 동일한 ordered 128MB raw stream을 정확히 한 번 encode한다. Tokenizer별 complete
512-token sequence를 sealed rank order 그대로 처리하고 마지막 partial effective batch도 버리지
않는다. 따라서 raw source history는 같지만 tokenizer 경계 때문에 실제 predicted target bytes는
수 KB 다를 수 있다. 두 값을 모두 봉인한다.

이 설계에서는 2K가 8K보다 optimizer update를 더 많이 수행한다. 이것은 제거할 nuisance가 아니라
큰 vocabulary가 동일 원문을 더 적은 autoregressive token으로 처리하는 system effect다. 대신 다음을
모두 분리 보고한다.

- raw stream bytes
- predicted target bytes
- optimizer steps
- optimizer wall time
- parameter/checkpoint bytes
- 이후 trained actual inference

Batch는 effective 32 sequences다. 2K microbatch는 32, 8K는 8이다. Body LR은 `3e-5`로 고정하고,
lexical head는 raw-target-byte 진행률의 첫 5%에서 `3e-4`까지 warm-up한 뒤 cosine으로 `3e-5`까지
감쇠한다. 이로써 서로 다른 update 수를 가진 tokenizer가 같은 원문 진행률에서 같은 schedule phase를
본다. Two-stage 역할은 stage 1에서 warm-up 후 peak를 유지하고 stage 2에서 raw progress를 다시
0부터 시작해 warm-up/cosine을 적용한다.

AdamW는 beta `(0.9, 0.95)`, epsilon `1e-8`, matrix weight decay `0.1`, vector decay `0`, gradient
clip `1.0`이다. Stage 1 copied input/output rows는 gradient를 0으로 만드는 것에 그치지 않고 매
step 뒤 exact 복원해 weight-decay drift도 막는다.

## Calibration과 사전 고정 gate

Calibration은 contiguous stream BPB와 full-document cluster BPB를 모두 계산한다. 선택과 gate는
8MB quota 안에 완전히 들어오는 383개 sealed calibration document의 per-document NLL을 primary로
사용하고 10,000회 document
bootstrap을 적용한다. Sealed final test, historical test NLL, downstream score, latency는 입력이 아니다.

### Quality noninferiority

각 8K 역할을 `dense2k_joint`와 비교한다.

- point difference `<= +0.010 BPB`
- document-bootstrap 95% upper `<= +0.010 BPB`

둘 다 만족한 역할만 trained actual preflight 후보가 된다. 여러 역할이 통과하면 document BPB가 가장
낮은 역할을 고른다. Exact tie는 더 단순한 대조군을 우선한다.

```text
standard_joint -> inplace_two_stage -> update_geometry
```

이 선택은 one-seed development selection이다. Publication 결과로 승격하지 않으며, 선택된 recipe는
이후 새 model seeds에서 고정해 확인해야 한다.

### Optimizer-method gate

`dense8k_update_geometry`가 독립 방법 후보로 남으려면 standard joint와 in-place two-stage 각각에
대해 다음을 모두 만족해야 한다.

- geometry − control point `<= -0.002 BPB`
- bootstrap 95% upper `<= 0`

두 비교 중 하나라도 실패하면 optimizer novelty 주장은 종료한다. 다만 사전 고정된 다른 8K control이
dense-2K noninferiority를 통과하면, tokenizer-adaptation deployment opportunity의 actual preflight는
진행할 수 있다. 이는 geometry의 fallback success가 아니라 별도 system 질문이다.

## 증거와 중단 규칙

Plan은 fresh-data seal/output, source checkpoint, 2K/8K tokenizer, update-audit result, 모든 코드와
환경 hash를 학습 전에 고정한다. 각 worker는 완료 전 checkpoint/NLL/report를 공개하지 않고, 완료된
역할만 exact resume한다. Summary는 네 checkpoint를 다시 load해 contiguous/document NLL을 전부
재계산하고 저장 배열과 bitwise equality를 요구한다.

- 8K noninferiority 역할이 없으면 branch 종료
- geometry method gate 실패면 geometry method branch 종료
- quality-qualified 8K가 있으면 한 seed trained actual controlled/free preflight를 별도 봉인
- 두 actual mode point가 각각 10% 미만이면 scale-up 금지
- 모두 통과한 뒤에만 multi-seed/multi-session, larger Mac-feasible model, Korean downstream,
  CUDA replication과 Hugging Face 공개를 연다

## 주장 경계

이 one-seed 실험은 개발 단계이며 논문 claim을 승인하지 않는다. Fresh stream도 동일 HPLT3 shard의
disjoint sample이지 새로운 domain이 아니다. In-place 대조군은 대규모 recipe의 축소 analogue다.
Dense-8K는 dense-2K보다 약 28% 더 많은 parameters를 가지므로, latency가 개선돼도 memory 개선이라고
부르지 않는다. 최종 성공은 trained model의 실제 end-to-end inference에서만 판정한다.

## 결과 비노출 model-free inventory 확인

Plan sealer와 동일한 재구성을 학습 전에 실행해 다음 좌표를 확인했다. 이는 loss나 checkpoint 결과가
아니며 이후 plan payload에 그대로 봉인한다.

| vocab | full token sequences | optimizer steps | predicted target bytes |
|---|---:|---:|---:|
| 2K | 70,786 | 2,213 | 127,749,593 |
| 8K | 53,640 | 1,677 | 127,747,857 |

8K token step은 2K보다 24.22% 적다. In-place stage 경계는 first-crossing 규칙에 따라 step 1,008,
76,668,094 target bytes(60.0152%)이며 stage 2는 669 steps다. Parameter count는 2K
19,667,328, untied 8K 25,172,352다. 이 수치는 opportunity와 cost contract일 뿐 quality 또는 actual
speed 결과가 아니다.
