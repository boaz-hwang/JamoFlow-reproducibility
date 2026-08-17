# Foldable Jamo residual development protocol

> 작성일: 2026-08-15
>
> 상태: 첫 B1 loss 전에 봉인된 historical protocol; 결과는 `docs/141`에서 Korean-specific gate 실패로 종료

## 연구 질문

강한 generic 2K→8K vocabulary transfer 위에, **학습 때만 존재하고 배포 전에 dense row로 완전히
접히는 한국어 orthographic sharing**을 더하면 같은 배포 graph에서 adaptation quality를 추가로
개선할 수 있는가?

이 단계는 actual inference speed를 측정하지 않는다. 다만 최종 checkpoint에는 별도 Jamo module,
lookup, branch 또는 parameter가 남지 않아야 한다. 이 development screen을 통과한 역할만 fresh-data
equal-history CPT와 trained batch-1 actual inference gate로 갈 수 있다.

> 후속 결과: 두 architecture의 true Jamo 역할은 base와 matched shuffle보다 좋아졌지만 같은 비용의
> generic multi-hash 역할보다 나빴고, 사전 `0.002 BPB` minimum도 충족하지 못했다. 이 protocol
> 아래의 fresh-data 승격은 열리지 않았다. 상세 수치와 별도 generic 가설은 `docs/141`을 따른다.

## B0가 고정한 출발점

`docs/139`의 strong-baseline closure 결과를 그대로 사용한다.

- untied frontier: `untied_eeve_uniform_in_first_out`, step-512 `1.454530 BPB`
- tied deployment frontier: `tied_uniform_no_norm_all`, step-512 `1.495260 BPB`
- dense BPE-2K anchor: `1.4296615772 BPB`
- compact `307+205` two-stage와 Hangul-specific median norm은 사용하지 않음

Untied base는 quality gate를 통과한 strongest generic baseline이다. Tied base는 anchor gap
`+0.065598 BPB`로 gate를 실패했으므로 qualified baseline이 아니라 lower-parameter deployment
frontier다. 두 architecture 결과를 하나의 initializer 순위로 합치지 않는다.

## Training-only reparameterization

새 6,144 target row에만 다음 effective weight를 사용한다.

`W_eff[token] = W_dense[token] + M_new[token] * sum_s R[s, code(token,s)] / sqrt(13)`

- `W_dense`: ordinary dense 8K input/output rows
- `R`: `13 × 128 × 384` shared residual table
- `M_new`: source 2,048 rows에는 0, 새 6,144 rows에는 1인 고정 mask
- `R`의 모든 값은 exact zero로 초기화
- tied graph는 input/output residual 하나를 공유
- untied graph는 같은 shape의 input/output residual을 따로 둠
- dense body, dense rows와 residual을 step 0부터 함께 학습

Zero residual이므로 각 역할의 step-0 folded state와 full calibration NLL은 architecture-matched B0
base와 bitwise 같아야 한다. 마지막에는 `W_eff`를 ordinary `nn.Embedding`과 `nn.Linear` weight로
materialize한다. Residual module과 code assignment를 제거한 deployed checkpoint의 parameter count는
base dense graph와 정확히 같아야 하며, fold 전후 contiguous/document NLL은 bitwise 같아야 한다.

이 설계는 SCRIPT의 online dual-channel fusion이나 이전 pure codebook head와 다르다. Dense token row를
제약하지 않고 공유 residual만 추가하므로 token identity capacity를 유지하며, 추론 때 orthographic
경로가 남지 않는다.

## 동일 비용 feature assignment

모든 역할은 token당 정확히 13개 code lookup과 같은 table shape를 사용한다. 기존 compositional
assignment의 surface slots 0--12를 재사용하되 token-identity slots 13--15는 제외한다.

- slots 0--2: first Unicode scalar/pseudo-byte의 base-127 digits
- slots 3--5: last Unicode scalar/pseudo-byte의 base-127 digits
- slots 6--8: first surface의 auxiliary 또는 초성·중성·종성
- slots 9--11: last surface의 auxiliary 또는 초성·중성·종성
- slot 12: raw token byte length

세 assignment는 다음과 같다.

1. `generic_surface`: slots 6--11도 domain-separated surface hash로 구성
2. `jamo`: first/last가 완성형 한글 음절이면 slots 6--11을 초성·중성·종성으로 구성하고,
   그 밖에는 generic fallback 사용
3. `shuffled_jamo`: `jamo`의 slots 6--11 block을 새 rows 사이에서 고정 permutation

Shuffle stratum은 `(exact raw token byte length, exact actual scheduled token count)`다.
Actual exposure는 512 steps × effective batch 32 × sequence 512의 정확한 고정 training-order prefix에서
센다. 전체 8K inventory 빈도가 아니라 실제 소비할 token을 사용한다. 각 non-singleton stratum을
seeded random order로 정렬한 뒤 한 칸 cyclic rotation하여 fixed point 없는 derangement를 만든다.
6-slot block 전체를 함께 옮기므로 slot별 분포와 component covariance가 보존되고, exact exposure가
같은 token끼리만 바뀌므로 각 slot/code의 exposure-weighted marginal도 exact 보존된다. Seed는
`20,260,829`다.

Loss를 보기 전 model-free audit에서 예정 token은 8,388,608개, 관측된 새 row는 6,141/6,144였다.
Exact length×exposure strata는 3,181개이고 singleton은 1,913개다. 나머지 4,231/6,144
(`68.86%`) 새 row, 실제 새-token 노출의 `58.06%`에 해당하는 auxiliary 연결은 모두 true Jamo와
달라지며, six slot의 exposure-weighted marginal은 exact 동일하다. Singleton에 남은 true 연결은
false positive보다 contrast를 보수적으로 만드는 한계로 공개한다. Plan sealer가 이 수치와 전체
assignment/exposure hash를 다시 계산해 봉인한다.

## 여섯 역할

| architecture | generic control | alignment-negative control | candidate |
|---|---|---|---|
| untied EEVE base | `untied_generic_surface` | `untied_shuffled_jamo` | `untied_jamo` |
| tied uniform base | `tied_generic_surface` | `tied_shuffled_jamo` | `tied_jamo` |

B0의 두 no-residual final checkpoint는 외부 control로 exact artifact/state hash에 결속한다. 새 역할과
동일한 calibration documents에서 independently replay하되, 이미 봉인된 B0 학습을 다시 돌리지 않는다.

## 학습·평가 계약

- source checkpoint/tokenizers/corpus/train order: B0와 동일
- train: repeated development prefix 128,000,000 raw bytes
- calibration contiguous: 8,000,000 raw bytes
- optimizer steps: 512
- effective batch: 32×512 target tokens, microbatch 8
- checkpoints: `0`, `32`, `128`, `512`
- body LR: `3e-5`
- dense head/residual LR: 기존 head cosine/warmup `3e-4 → 3e-5`
- AdamW beta `0.9/0.95`, epsilon `1e-8`, matrix decay `0.1`, gradient clip `1.0`
- model/order seeds: `20,260,824` / `20,260,827`
- document bootstrap: 10,000 repetitions, seed domain beginning `20,260,830`

Worker는 각 unfolded checkpoint의 full contiguous NLL을 저장하고, step 512에는 full document NLL도
저장한다. Final fold 뒤 두 배열을 다시 계산해 bitwise equality를 요구한다. 독립 summarizer는 모든
unfolded checkpoint를 새 모델에 strict-load해 다시 forward하고, final residual을 별도로 materialize해
folded state와 두 NLL을 다시 검증한다. B0 두 base checkpoint도 독립 replay한다.

Training-only residual parameter와 optimizer-step seconds, session elapsed, sampled post-checkpoint MPS/RSS
진단을 공개한다. Native resettable MPS peak가 아니므로 memory improvement를 주장하지 않으며,
B0 시간은 별도 session의 descriptive comparator일 뿐 paired speed gate로 쓰지 않는다.

## 사전 고정 gate

Architecture별 `jamo` 역할은 step-512에서 다음을 모두 만족해야 한다.

1. no-residual architecture base보다 contiguous와 document BPB가 모두 낮다.
2. `generic_surface`보다 contiguous/document BPB가 각각 최소 `0.002` 낮다.
3. `shuffled_jamo`보다 contiguous/document BPB가 각각 최소 `0.002` 낮다.
4. 세 document contrast의 document-cluster bootstrap 95% upper bound가 모두 `<=0`이다.
5. dense BPE-2K anchor gap이 `<=+0.050 BPB`다.
6. step-0 base equivalence, copied-old-row residual mask, final fold state/NLL equivalence가 모두 pass다.

`0.002 BPB`는 B0의 generic decoded-character output ablation 크기 `0.002515 BPB`보다 작은 효과를
Korean-specific contribution으로 과장하지 않기 위한 development minimum이다. 이 선택은 B1 loss
전에 고정한다. Threshold 완화, step-32/128 fallback, assignment/seed/stratum 추가는 금지한다.

한 architecture만 통과하면 그 role만 fresh-data stage로 보존한다. 둘 다 실패하면 Jamo residual을
튜닝하지 않고 Korean branch를 종료한다. Generic transfer만 별도 engineering result로 남을 수는
있지만 Korean-specific 논문 성공으로 간주하지 않는다.

## Claim boundary와 다음 단계

이 실험은 one model seed, 이미 알려진 calibration, source가 이미 본 repeated train prefix의
development causal screen이다. Document bootstrap은 이 고정 corpus의 문서 변동성만 나타내며 model-
seed uncertainty가 아니다. Positive여도 publication quality나 actual inference efficiency를 주장하지
않는다.

통과한 경우에만 다음을 연다.

1. source가 보지 않은 disjoint continuation documents에서 dense-2K continuation, direct dense-8K,
   architecture-matched random transfer, generic EEVE transfer, qualified Jamo transfer에 같은 새 raw-byte
   history를 부여
2. quality-qualified exact folded dense checkpoint만 dense-2K와 batch-1 controlled same-output 및
   strict-valid free-running으로 비교
3. 두 co-primary mode의 E2E point reduction `>=10%`와 uncertainty/stability gate 통과 뒤에만
   multi-seed/larger model/Korean downstream/CUDA/Hugging Face/paper positive claim으로 확장

Fable 5 검토가 지적했고 W72 결과가 확인한 대로 analytical/token-count 이득은 actual E2E의 대체가
아니다. 이 연구의 최종 성공 기준은 여전히 trained-model measured inference efficiency다.
