# Strong vocabulary-transfer baseline 결과와 foldable-Jamo 결정

> 작성일: 2026-08-15
>
> 상태: one-seed development baseline closure 통과; 후속 foldable-Jamo gate는 실패했으며 `docs/141`이 최신 방향을 지배

## 결론

봉인된 9-role strong-baseline 실험은 **untied generic vocabulary transfer가 충분히 강한
development baseline**임을 확인했다. 최선은 EEVE의 input/output initializer를 옮긴
`untied_eeve_uniform_in_first_out`이며 step 512에서 `1.454530 BPB`였다. 같은 untied random
control보다 `0.095090 BPB` 좋고 dense BPE-2K anchor와의 격차는 `+0.024868 BPB`여서 사전 고정한
두 gate를 모두 통과했다.

반면 tied continued-BPE와 compact two-stage schedule은 anchor-recovery gate를 통과하지 못했다.
따라서 다음 단계는 아래처럼 수정한다.

1. **Untied EEVE initializer analogue**를 strongest generic quality baseline으로 고정한다.
2. **Tied uniform/all-parameter**를 동일 배포 parameter frontier의 대조군으로 보존하되 qualified
   quality baseline이라고 부르지 않는다.
3. Hangul-specific norm 선택과 `307+205` two-stage는 폐기한다.
4. 다음 Korean 기법은 추론 때 별도 Jamo module을 남기지 않고 dense 8K rows로 완전히 접히는
   training-only residual이어야 한다.
5. true Jamo assignment가 동일 feature budget의 generic-surface 및 shuffled-Jamo control을
   이기지 못하면 Korean branch를 종료한다.

이 수정은 결과에 맞춘 threshold 변경이 아니다. B0가 미리 허용한 B1을 열되, B0에서 드러난
약한 설계 선택을 제거하고 더 강한 generic comparator를 채택하는 것이다. 최종 성공 기준인
**matched-quality batch-1 end-to-end latency 10% 이상 개선**은 바꾸지 않는다.

## 결과

Dense BPE-2K anchor는 `1.4296615772 BPB`다. 모든 역할은 같은 source checkpoint, continued
BPE-8K tokenizer, 128M-byte repeated development prefix, 512 updates와 calibration stream을 썼다.

| role | graph/schedule | step 50 | step 512 | random advantage | anchor gap | joint gate |
|---|---|---:|---:|---:|---:|---|
| untied random, Hangul-median input norm | untied/all | 1.915829 | 1.549620 | — | +0.119958 | control |
| BIL, Hangul median + character-weighted output | untied/all | 1.674687 | 1.464947 | 0.084673 | +0.035285 | pass |
| BIL, global median + character-weighted output | untied/all | 1.674772 | 1.464968 | 0.084651 | +0.035307 | pass |
| BIL, Hangul median + uniform output | untied/all | 1.685987 | 1.467462 | 0.082158 | +0.037800 | pass |
| **EEVE uniform input + first-subword output** | **untied/all** | **1.634975** | **1.454530** | **0.095090** | **+0.024868** | **pass** |
| tied random | tied/all | 1.932217 | 1.591153 | — | +0.161492 | control |
| tied uniform | tied/all | 1.704656 | 1.495260 | 0.095893 | +0.065598 | fail |
| tied random | tied/307+205 | 2.061213 | 1.648405 | — | +0.218743 | control |
| tied uniform | tied/307+205 | 1.868346 | 1.539426 | 0.108979 | +0.109765 | fail |

`random advantage >=0.010 BPB`와 `anchor gap <=+0.050 BPB`를 동시에 요구했다. 네 untied
composition 역할만 통과했고 tied Pareto role은 없었다. Step 0과 step 50은 기술 통계일 뿐
선택에는 사용하지 않았다.

## 독립 재계산과 artifact

Summarizer는 worker scalar를 신뢰하지 않았다. 아홉 역할의 여섯 checkpoint, 총 54개 state를
strict-load하고 calibration 전 구간을 다시 forward했다. 재계산한 float32 per-sequence NLL은
54개 저장 배열과 모두 bitwise identical이었다.

- plan commit: `ef6ba45`
- result commit: `ff24e4f`
- tracked summary: `results/vocabulary-transfer-baseline-closure-v1/summary.json`
- summary file SHA-256: `ba07726005ef87aacca67771267632981a06e55e15f340ea78d69cc4ac4e8bb9`
- canonical summary payload SHA-256: `67ae0a35dac1575ba40664a5b602d30b91072d8b8e2cb22ba6fd166df8699b64`
- independent replay: 54 checkpoints, bitwise float32 equality, `1779.361 s`

## 무엇을 배웠는가

### 1. 현재 compact setting의 strongest generic baseline은 EEVE initializer analogue다

EEVE analogue는 method-exact BIL Hangul role보다 `0.010417 BPB` 좋았다. 원 논문의 seven-stage
freezing schedule은 재현하지 않았으므로 EEVE 전체 recipe의 우월성으로 일반화하지 않는다.
다만 후속 Korean method가 이 initializer보다 약하면 비교 대상이 약했다는 비판을 피할 수 없다.

### 2. Hangul-specific median norm 효과는 사실상 관찰되지 않았다

BIL의 Hangul-token median과 global median 차이는 `0.0000215 BPB`에 불과했다. 이 규모·tokenizer·
budget에서 Hangul subset을 고르는 행위가 의미 있는 개선을 주었다고 볼 근거가 없다. 따라서 이
선택을 다음 Jamo 실험에 유지하면 불필요한 자유도만 늘어난다.

### 3. Character-weighted output은 작지만 방향성 있는 generic 효과다

Hangul-median input을 고정했을 때 decoded-character-weighted output은 uniform output보다
`0.002515 BPB` 좋았다. 그러나 이는 Korean-specific 효과가 아니라 generic Unicode token
decomposition 효과이며, EEVE first-subword output보다 여전히 약했다.

### 4. Compact two-stage freezing은 이 setting에서 해롭다

Tied uniform two-stage는 all-parameter 학습보다 `0.044166 BPB`, random two-stage는 random
all-parameter보다 `0.057252 BPB` 나빴다. 600B:400B token recipe를 307:205 steps로 축소한
analogue가 장기 학습 효과를 보존하지 못했다. 결과를 보고 stage ratio를 다시 맞추지 않으며,
B1에서는 모든 parameter를 step 0부터 학습한다.

### 5. Untied 우위에는 추가 lexical capacity가 포함된다

Untied graph는 `25,172,352`, tied graph는 `22,026,624` parameters다. 차이는 `3,145,728`, tied
대비 `14.281%`다. 따라서 `1.454530`과 `1.495260`의 차이를 initializer만의 효과라고 해석하지
않는다. 다음 단계는 architecture 안에서만 true/generic/shuffled residual을 비교하고, 최종
배포 판단에서는 untied quality frontier와 tied parameter frontier를 따로 보고한다.

## Fable 5 중간 검토를 이 결과에 반영한 판단

`fable5-연구-중간-검토.md`의 핵심 경고는 이미 실제 결과로 확인됐다. Global patch event 감소는
W72의 compact actual E2E를 약 `2.5%`만 개선했고, 10% gate에는 실패했다. 그러므로 이번
vocabulary-transfer 결과도 BPB 또는 token fertility만으로 효율 논문 성공이라고 부를 수 없다.

수용해야 할 지점은 세 가지다.

1. quality와 analytical compute 이후에도 **trained exact checkpoint의 whole-path latency**를
   별도로 검증한다.
2. 같은 rate/graph 대조로 기전 효과를 분리한다. B1에서는 이를 true Jamo 대 shuffled/generic
   assignment로 구현한다.
3. 작은 compact positive를 곧바로 top-tier scaling claim으로 키우지 않는다. fresh-data,
   multi-seed, larger Mac-feasible model과 별도 hardware replication은 10% gate 이후에만 연다.

수용하지 않는 지점도 명확하다. 이전 W72 실패가 dense BPE-8K vocabulary shortening에도 같은
상한을 준다고 볼 수는 없다. Dense 8K는 이미 random-weight actual system probe에서 2K보다 약
`19.8%` 빠른 별도 경로였고, 현재 병목은 그 graph의 trained quality 회복이다. 그러므로 이번
positive transfer 결과는 foldable Korean residual과 fresh-data equal-history 검증으로 진행할
충분한 근거가 있다.

## B1: training-only foldable orthographic residual

### 핵심 가설

새 6,144 target rows의 dense input/output weight를 다음처럼 학습한다.

`effective_row(token) = base_dense_row(token) + orthographic_residual(features(token))`

Residual table은 정확히 zero로 초기화한다. 따라서 step 0의 effective weight와 logits는 strong
generic base와 bitwise 같아야 한다. 512-step CPT에서는 dense row와 residual을 함께 학습하고,
마지막에 effective rows를 ordinary `nn.Embedding`/`nn.Linear` weight로 materialize한 뒤 residual
module을 삭제한다. Fold 전후 logits와 NLL은 exact equality를 요구한다. 배포 tokenizer,
parameter count, forward graph와 FLOPs는 같은 architecture의 generic dense baseline과 동일하다.

### 비교 역할

두 architecture frontier를 분리한다.

- untied: EEVE initializer analogue base
- tied: uniform/no-norm/all-parameter base

각 frontier에서 다음 세 residual assignment를 같은 table 크기, slot 수, feature lookups와 optimizer
budget으로 비교한다.

1. `generic_surface`: UTF-8/Unicode surface에서 얻은 언어 비특이적 특징
2. `shuffled_jamo`: true Jamo 특징을 exact token byte length와 exact scheduled exposure stratum 안에서 고정
   permutation한 negative control
3. `jamo`: 완성형 음절을 초성·중성·종성으로 분해한 true orthographic assignment

Copied source 2,048 rows에는 residual을 적용하지 않는다. Incomplete UTF-8 piece와 비한글 token은
generic fallback을 사용하고, 세 역할의 feature access 수와 residual cardinality가 달라지지 않게
고정-slot encoding을 사용한다. Assignment, strata, permutation seed, parameter count, initial
state와 fold-equivalence test를 첫 loss 전에 봉인한다.

### 사전 판정

Architecture별 primary Korean-specific contrast는 step 512 contiguous BPB의 paired 차이다.

- `jamo`가 no-residual generic base보다 좋아야 한다.
- `jamo`가 `generic_surface`와 `shuffled_jamo`를 각각 최소 `0.002 BPB` 이겨야 한다.
- document-cluster bootstrap에서 두 contrast의 95% upper bound가 `<=0`이어야 한다.
- 최종 anchor gap은 `<=+0.050 BPB`여야 한다.
- fold 전후 logits/NLL은 exact equality여야 한다.

`0.002 BPB`는 BIL output ablation에서 관찰된 generic 효과(`0.002515`)보다 작은 차이를 Korean
기여로 과장하지 않기 위한 development minimum이다. 이 기준은 B1 loss를 보기 전에 plan에
고정한다. 둘 중 한 architecture만 통과하면 그 frontier만 B2로 보존한다. 둘 다 실패하면 Jamo
residual을 튜닝하지 않고 Korean branch를 종료한다.

Training-only residual의 추가 trainable parameters, step time, peak memory와 총 training energy는
별도 기록한다. 배포 비용이 0이라고 해서 학습 비용도 0이라고 쓰지 않는다.

## 이후 변경 없는 성공 경로

1. **B1 development:** true/generic/shuffled foldable residual의 same-cost causal screen
2. **B2 fresh-data equal-history:** dense-2K continuation, direct dense-8K, random transfer,
   generic EEVE transfer, qualified Jamo transfer를 source가 보지 않은 동일 새 bytes에서 비교
3. **B3 trained actual inference:** quality-qualified exact dense checkpoint를 batch-1 controlled
   same-output와 strict-valid free-running에서 dense-2K와 비교
4. 두 co-primary E2E mode에서 point reduction `>=10%`와 uncertainty/stability gate를 통과할 때만
   multi-seed larger-scale, Korean downstream, CUDA replication, Hugging Face release와 논문
   positive claim으로 확장

B1/B2의 BPB가 좋아도 B3를 실패하면 사용자가 정한 최종 연구 성공이 아니다. 반대로 generic EEVE
transfer만 B2/B3를 통과하고 Jamo residual이 실패하면 유용한 engineering artifact는 남지만,
Korean-specific top-tier contribution으로 주장하지 않는다.

## 후속 결과

봉인 B1은 실제로 Jamo-specific gate를 실패했다. True Jamo는 matched shuffle보다 작게 좋았지만
generic multi-hash residual보다 양 architecture에서 약 `0.0013 BPB` 나빴다. Jamo branch는
threshold 조정 없이 종료한다. Generic residual은 ordinary dense graph로 exact fold되면서 base를
`0.0156–0.0255 BPB` 개선했으므로, 별도 optimizer-confound guard와 fresh-data actual-efficiency
가설로만 다시 연다. `docs/141`이 최신 operational decision이다.
