# Strong vocabulary-transfer baseline-closure protocol

> 작성일: 2026-08-15
>
> 상태: 역할·초기 state·schedule·gate를 loss 관측 전에 구현하고 봉인하기 위한 protocol

> 후속 결과: protocol은 변경 없이 완료됐다. 네 untied 역할이 통과했고 최선은 EEVE initializer
> analogue `1.454530 BPB`였다. Tied 역할은 모두 anchor-recovery gate를 실패했다. 전체 해석과
> 다음 단계 결정은 `docs/139-strong-vocabulary-transfer-baseline-result-and-foldable-jamo-decision.md`를
> 따른다.

## 목적

이 단계는 Korean method를 시험하지 않는다. `docs/137`의 generic 2K→8K transfer가 positive였지만,
그 initializer는 최신 최강 baseline의 정확한 정의와 달랐다. 약한 generic control 위에서 Jamo
효과를 과장하지 않도록 다음 세 축을 같은 compact graph와 512-step 예산에서 닫는다.

1. [Beyond Initialization Loss](https://arxiv.org/abs/2608.03494)의 input/output 비대칭과
   script-specific median norm, decoded-character output weighting
2. [In-Place Tokenizer Expansion](https://arxiv.org/abs/2607.15232)의 continued-BPE mean
   initialization과 new-row-only→full-model two-stage 구조
3. [EEVE](https://arxiv.org/abs/2402.14714)의 uniform-input/first-subword-output initializer

이 실험은 문헌의 대규모 token 수나 EEVE seven-stage convergence schedule을 재현하지 않는다.
BIL initializer 정의는 method-exact하게 구현하고, In-Place는 600B:400B를 307:205 updates로
줄인 ratio analogue, EEVE는 initializer-only analogue로 명명한다.

## 고정 입력

- source model/checkpoint: compositional-quality run의 trained dense BPE-2K
- target: continued ByteLevel BPE-8K, dense hidden 384 / FFN 1,536 / 8 layers
- source/target token relation: target merge tree를 exact source vocabulary frontier에서 절단
- shared rows/body: source checkpoint에서 exact copy
- train: 기존 development train stream first 128,000,000 raw bytes
- calibration: 기존 contiguous 8,000,000 raw bytes 전체
- model/order seeds: 20,260,824 / 20,260,827
- effective batch: 32×512 target tokens; microbatch 8
- checkpoint steps: `0`, `32`, `50`, `128`, `307`, `512`
- dense-2K anchor: `1.4296615772178647 contiguous BPB`

Source model이 이미 같은 128M train prefix를 보았으므로 이 단계는 recovery-method development
evidence다. Fresh continuation document에서의 equal-history full CPT는 별도 단계이며 현재 NLL을
publication quality evidence로 승격하지 않는다.

## Exact BIL metadata contract

각 source token ID를 `tokenizer.decode([id], skip_special_tokens=False)`로 독립 decode한다.
Character length는 Python `len`으로 센 Unicode code-point 수이며 `max(length,1)`을 적용하고 추가
Unicode normalization을 하지 않는다. Incomplete Byte-BPE piece가 U+FFFD로 decode되면 한 code
point로 센다. 원 논문의 정의를 그대로 따른 결과지만 byte tokenizer에서 의미상 완성 문자의 길이와
다를 수 있음을 limitation으로 남긴다.

Hangul source-token subset은 decoded string에 아래 범위가 하나라도 포함되면 true다.

- Hangul Jamo U+1100--11FF
- Compatibility Jamo U+3130--318F
- Jamo Extended-A U+A960--A97F
- precomposed syllables U+AC00--D7A3
- Jamo Extended-B U+D7B0--D7FF

Input target norm은 선택 subset row L2의 conventional midpoint median이다. 짝수 개면 중앙 두
값의 평균을 사용한다. Global ablation은 source 2,048 rows 전체를 쓴다. Norm calibration은 input
new row에만 적용하며 copied row와 output matrix에는 적용하지 않는다.

## 아홉 역할

| role | graph | input new row | output new row | schedule | 해석 |
|---|---|---|---|---|---|
| `untied_random_hangul_median_input_native_output` | untied | native random direction + Hangul median norm | native random | all | architecture-matched random |
| `untied_bil_hangul_median_char_out` | untied | constituent uniform + Hangul median norm | decoded-character weighted | all | BIL method-exact |
| `untied_bil_global_median_char_out` | untied | uniform + global median norm | decoded-character weighted | all | norm-scope ablation |
| `untied_bil_hangul_median_uniform_out` | untied | uniform + Hangul median norm | uniform | all | output-weight ablation |
| `untied_eeve_uniform_in_first_out` | untied | uniform, no norm | first source subword | all | EEVE initializer analogue |
| `tied_random_native_all` | tied | native random | shared | all | tied random control |
| `tied_uniform_no_norm_all` | tied | constituent uniform | shared | all | continued-BPE mean |
| `tied_random_native_two_stage` | tied | native random | shared | 307+205 | staged random control |
| `tied_uniform_no_norm_two_stage` | tied | constituent uniform | shared | 307+205 | In-Place ratio analogue |

Untied graph는 25,172,352 parameters, tied graph는 22,026,624다. 두 graph의 attention와 output
vocabulary geometry는 같지만 input/output sharing과 resident weights가 다르다. 따라서 모든
initializer는 같은 graph 안 random control과 비교하고, tied와 untied의 winner를 하나로
평탄화하지 않는다.

## Two-stage 실행 계약

두 staged role은 총 512 updates와 token order를 다른 역할과 공유한다.

- Stage 1, updates 0--306: tied lexical matrix의 새 6,144 rows만 유효하게 갱신
- boundary checkpoint: step 307
- Stage 2, updates 307--511: optimizer를 새로 만들고 모든 parameters 갱신

Stage 1에서는 body `requires_grad=False`, old row gradient를 0으로 mask하고 AdamW step 뒤 copied
2,048 rows를 exact source value로 복원한다. 매 step equality를 검사한다. Step 307 checkpoint에서
body와 copied rows가 initial state와 bitwise 같고 new rows는 달라야 한다. Stage 2에서는 body와
new/old rows 모두 열리고 최종 body가 변해야 한다.

Learning-rate curve는 staged/non-staged 비교에서 global 512-step 좌표를 공유한다. Stage 2
optimizer state만 reset한다. 이는 original In-Place recipe의 stage별 warmup/constant LR과 대규모
mixture를 재현한 것이 아니며, freezing/order 구조의 compact causal screen이다.

## 판정 계약

Step 0과 BIL-style step 50 rank는 descriptive다. 역할 선택과 progression은 오직 step 512
contiguous raw-byte BPB로 한다. 각 composed role은 다음 두 gate를 동시에 통과해야 한다.

1. `architecture/schedule-matched random BPB − composed BPB >= 0.010`
2. `composed BPB − dense2K anchor <= 0.050`

Candidate/control mapping은 다음과 같다.

- 네 untied composed roles → single untied random control
- tied all-parameter mean → tied all-parameter random
- tied two-stage mean → tied two-stage random

한 역할 이상 통과하면 Korean stage를 열되, qualified tied 중 최저 BPB와 qualified untied 중 최저
BPB를 각각 Pareto role로 보존한다. 한 architecture의 낮은 BPB를 다른 architecture의 추가
parameters와 섞어 단일 initializer 효과로 해석하지 않는다. 아무 역할도 통과하지 않으면 8K
vocabulary-transfer/Korean residual 분기를 중단한다. Threshold 완화, step-50 fallback, 역할 추가는
금지한다.

## 무결성

- Source/tokenizer/corpus/parent result, 아홉 initial states, decoded lengths, Hangul mask, norms,
  decomposition, train order, parameter counts와 implementation hashes를 첫 loss 전에 plan에 넣는다.
- 각 worker는 fresh process와 publication MPS lock에서 실행한다.
- 여섯 checkpoint/NLL/receipt를 전부 메모리에서 완성한 뒤에만 no-clobber publish한다.
- Partial worker namespace는 자동 overwrite하지 않고 forensic failure로 중단한다.
- Worker 완료 시 exact HEAD와 clean worktree를 요구한다.
- Summary는 9×6=54 checkpoints를 새 graph에 strict-load하고 full calibration NLL을 다시
  forward해 저장 float32 arrays와 bitwise 비교한다. 재계산본만 decision에 사용한다.
- 결과 summary 전에는 role별 scalar를 공개하거나 읽어 selection하지 않는다.

## 다음 단계의 범위

이 단계가 positive여도 generic vocabulary transfer, Korean tokenizer 또는 inference-speed 신규성을
주장하지 않는다. 다음 Korean method는 training-time Jamo residual을 ordinary dense 8K rows로
fold해 deployed graph를 strong generic control과 같게 만들고, matched shuffled assignment를 함께
시험해야 한다. 그 뒤 fresh-data equal-history full CPT의 품질 noninferiority와 실제 same-output
batch-1 E2E 10% gate를 차례로 통과해야만 publication positive route가 열린다.
