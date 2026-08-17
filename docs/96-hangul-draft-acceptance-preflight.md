# Hangul draft acceptance preflight

> 작성일: 2026-08-13
>
> 상태: **결과 확인 전 protocol 봉인 예정**
>
> plan: `data/manifests/hangul-draft-acceptance-v1.json`

실행 이력: 최초 봉인 commit `f724b68`의 첫 호출은 model/data를 열기 전에 final
authorization validator의 required `selection_lock` 인자를 넘기지 않아 중단됐다. Cache,
head, metric artifact는 생성되지 않았다. 이 호출계약만 보완하고 implementation hash를
갱신한 뒤 재봉인한다.

두 번째 호출은 frozen hidden 추출 뒤, 어떤 cache/head/metric artifact도 publish하기 전에
prompt feasibility에서 중단됐다. 1 MB calibration에는 Hangul 문자 비중 80% 이상인
128-byte window가 851개뿐이고, 고정 bottom-hash non-overlap 선택으로는 121개만 남아
사전 지정한 128 prompts를 만들 수 없었다. 결과 metric이 아닌 calibration 구조만 조사해
threshold를 가능한 최소 변경인 **79%**로 낮췄다. 같은 고정 선택에서 1,165 eligible / 167
non-overlap windows가 있어 128개를 구성한다. Prompt 수·길이·target horizon·gate는 바꾸지
않았다.

## 1. 왜 이 실험이 필요한가

실제 v5r3에서 W72는 C86보다 patch와 dense matmul을 줄였지만 end-to-end 개선은
2.5%에 그쳤다. Component profile은 127개의 byte-local 순차 step이 decode의 약 84%를
차지하며, patch 4개를 줄여 얻은 약 10 ms가 실제 decode 차이와 거의 같음을 보였다.
8 MB calibration oracle에서는 Hangul scalar를 완벽히 block 처리할 경우 target call을
57.593% 줄일 수 있었지만, 첫 UTF-8 byte나 조합규칙만으로 나머지 두 byte는 결정되지
않았다.

따라서 이 단계의 질문은 다음 하나다.

> Frozen W72의 문맥 hidden으로부터, 같은 작은 parameter budget의 Hangul-aware draft가
> generic dependence-aware byte draft보다 더 많은 미래 byte를 정확히 제안하는가?

이 질문에 답하기 전에는 block verifier나 실제 runtime을 구현하지 않는다.

## 2. 공정한 비교군

모든 head는 같은 192차원 frozen local-decoder hidden과 target이 이미 고른 첫 byte
`EA..ED`를 입력으로 받는다. Head는 뒤의 두 continuation byte만 제안한다. 모든 제안은
그 첫 byte와 일치하는 완성형 한글 11,172자 중 하나여야 한다.

| architecture | trainable parameters | 구조 정보 |
|---|---:|---|
| generic independent UTF-8 | 41,728 | 두 64-way continuation 주변분포 |
| generic joint UTF-8 | 42,733 | 4,096-way continuation pair의 low-rank joint 분포 |
| Hangul parallel components | 42,468 | 초성/중성/종성 parallel score |
| Hangul conditional components | 39,604 | 초성 top-4 → 조건부 중성 → 조건부 종성 |

최대/최소 parameter 비는 1.079다. 그러므로 joint head가 independent head를 이기는 것은
dependence modeling의 효과이고, conditional Hangul head가 joint byte head를 이겨야만
한국어 조합구조의 추가 기여라고 볼 수 있다.

## 3. Data와 frozen target

- target: quality-authorized W72, seed 1729, 19,596,096 parameters
- head train: historical train split 앞 999,936 bytes에서 고정 seed로 뽑은 Hangul
  100,000 contexts
- head selection/evaluation: historical calibration split 앞 999,936 bytes의 별도 Hangul
  100,000 contexts
- free target acceptance: calibration에서 model-free SHA-256 bottom-order로 고른 서로
  겹치지 않는 Hangul 문자 비중 79% 이상 128-byte prompt 128개, prompt당 strict UTF-8
  greedy 380--383 bytes
- head initialization/training seeds: 20260813, 20260817, 20260819

Train과 calibration stream, boundary, W72 patch matrix, target authorization/checkpoint/state는
plan에 SHA-256으로 고정한다. Final test, historical test metric, v5 latency 숫자는 읽지 않는다.
Target weight는 동결하고 head만 8 epoch 학습한다.

Teacher-forced cache에서 쓰는 `BltModel.last_hidden_state → frozen lm_head`가 원래
`BltForCausalLM.logits`와 bitwise equal임은 결과를 보지 않는 1-row mechanical probe로 먼저
확인했다.

## 4. Acceptance 측정

Primary evidence는 teacher-forced corpus accuracy가 아니라 frozen target 자체의 calibration
free-running trace다. Target은 계속 bytewise sequential greedy로 authoritative `b1,b2,b3`를
만든다. `b1 in EA..ED`인 시점의 hidden과 `b1`을 head에 주고, head의 `(b2,b3)` proposal을
target byte와 비교한다.

- first-continuation acceptance: proposed `b2 == target b2`
- complete-pair acceptance: proposed `(b2,b3) == target (b2,b3)`
- mean accepted suffix bytes: `I[b2 match] + I[b2,b3 both match]`
- target scalar가 실제 Hangul인지 여부도 별도 보고
- head parameter 수와 synchronized batch-1 proposal latency 보고

Free-running target이 `EA..ED`로 시작했지만 실제 scalar는 한글이 아닐 수도 있다. 이를
사후 제외하지 않고 실패에 포함한다. 그래야 activation rule의 실제 precision 비용을 숨기지
않는다.

## 5. 사전 gate

각 system의 verifier-prototype feasibility는 모두 다음을 만족해야 한다.

- free attempts >= 10,000
- median complete-pair acceptance >= 40%
- median mean accepted suffix bytes >= 0.90 / 2
- 세 head seed 각각 complete-pair acceptance >= 35%
- synchronized median head proposal latency <= 1.0 ms

한국어-specific prototype은 여기에 더해 conditional Hangul head가 joint UTF-8 control보다
complete-pair acceptance를 절대 1 percentage point 이상 높이고, prompt-paired 10,000회
bootstrap 95% CI lower bound가 0보다 커야 한다.

- Hangul gate pass: exact Hangul block verifier prototype으로 진행
- Hangul gate fail, generic joint feasibility pass: generic verifier는 timing infrastructure
  diagnostic으로만 진행; 한국어-specific claim은 중단
- 둘 다 fail: multi-byte draft branch 중단

이 pass도 speed 증거가 아니다. Exact target verification, rollback/cache correctness, 그리고
draft+verification을 모두 포함한 실제 end-to-end timing은 다음 단계에서 별도로 통과해야 한다.

## 6. 해석 경계

- 단일 frozen target seed의 calibration-only exploratory preflight다.
- Head 학습 품질과 proposal acceptance만 평가하며 target model 품질은 바꾸지 않는다.
- Teacher-forced accuracy는 descriptive이고 gate는 target-generated trace를 사용한다.
- Parameter matching은 했지만 커널 수와 memory traffic은 다르므로 latency도 함께 gate한다.
- Generic joint가 이기면 Fast BLT/MTP/MtPC 계열의 dependence-aware drafting 근거일 뿐,
  JamoFlow의 한국어-specific novelty가 아니다.
- 실제 block target call이 구현되지 않았으므로 end-to-end speedup을 주장하지 않는다.
