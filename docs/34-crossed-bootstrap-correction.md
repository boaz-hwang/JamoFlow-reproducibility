# Phase 1–2 crossed-bootstrap correction

> 교정일: 2026-08-10  
> 영향 범위: Phase 1 및 Phase 2의 보조 percentile-bootstrap 구간  
> 결론 영향: 없음

## 1. 무엇을 교정했는가

Phase 1과 Phase 2의 모든 학습 seed는 서로 다른 모델 초기화와 학습 순서를 갖지만, 같은 held-out sequence 집합을 같은 순서로 평가한다. 데이터 구조는 seed 안에 서로 다른 sequence가 중첩된 구조가 아니라 다음과 같은 **교차 설계**다.

\[
d_{s,i}=\ell_{s,i}^{(A)}-\ell_{s,i}^{(B)},
\qquad
s\in\{1,\ldots,S\},\ i\in\{1,\ldots,N\}.
\]

여기서 동일한 sequence index \(i\)는 모든 seed \(s\)에 걸쳐 같은 원문 window를 가리킨다. 기존 구현은 bootstrap replicate마다 선택된 seed 각각에 독립적인 sequence index를 뽑았다. 이 방식은 seed마다 test set이 따로 표집된 것처럼 취급해, 공통 문장의 난이도 때문에 생기는 seed 간 상관을 보존하지 못했다.

교정 구현은 각 replicate에서 다음 두 축을 각각 한 번씩 복원추출한다.

1. 학습 seed \(S\)개를 복원추출한다.
2. 공통 test sequence index \(N\)개를 한 번 복원추출한다.
3. 선택된 seed와 선택된 공통 sequence의 교차 셀 평균을 계산한다.

따라서 같은 replicate 안에서는 모든 seed가 같은 재표집 sequence 집합을 본다. 입력 배열 길이가 다르면 교차 설계가 성립하지 않으므로 즉시 오류를 낸다. 기계 판독 결과에는 `resampling_design: "crossed seeds x shared test sequences"`를 기록했다.

## 2. 왜 주 추론은 변하지 않는가

이 연구에서 확증 판정의 주 단위는 독립적인 학습 seed이며, 주 구간은 seed별 paired BPB 차이에 대한 Student-t interval이다. Sequence bootstrap은 다음 목적의 보조 진단이었다.

- 공통 평가 문장 구성에 결과가 과도하게 민감한지 확인
- seed 효과와 example-difficulty 효과를 함께 흔들어 방향 안정성 확인
- per-sequence paired loss 산출물의 정합성 확인

그러므로 이번 교정은 사전등록 gate, seed별 효과, 평균 BPB, paired-t interval을 변경하지 않는다. 더 좁은 bootstrap 구간만으로 통계적 확정을 주장하지 않는다는 기존 해석 원칙도 유지한다.

## 3. 수치 영향

### Phase 2 primary

| Contrast | 이전 95% interval | 교정 95% interval | 방향 변화 |
|---|---:|---:|---:|
| C1 − C0 | [−0.00696, −0.00605] | [−0.00709, −0.00595] | 없음 |
| C2 − C1 | [−0.00839, −0.00539] | [−0.00848, −0.00534] | 없음 |
| C4 − C3 | [+0.00793, +0.01440] | [+0.00787, +0.01454] | 없음 |
| C1 − C3 | [−0.00794, +0.00063] | [−0.00799, +0.00069] | 없음 |
| C2 − C3 | [−0.01622, −0.00502] | [−0.01619, −0.00516] | 없음 |

### Phase 2 mechanism controls

| Contrast | 이전 95% interval | 교정 95% interval | 방향 변화 |
|---|---:|---:|---:|
| C2 − delayed | [−0.00839, −0.00590] | [−0.00850, −0.00591] | 없음 |
| C2 − placebo | [−0.01696, −0.01402] | [−0.01704, −0.01396] | 없음 |

Phase 1의 아홉 보조 구간도 모두 다시 계산했다. 경계가 0을 포함하는지 여부는 모든 contrast-language 조합에서 그대로였고, [Phase 1 결과표](./09-phase1-neural-results.md)를 교정 수치로 갱신했다.

## 4. Phase 3에 미친 영향

Phase 3 결과를 보기 전에 별도의 요약 구현은 이미 같은 교차 재표집 설계로 수정했다. 또한 세 seed뿐인 확증 단계에서는 다음을 분리한다.

- 효과 크기와 불확실성: seed별 paired difference와 one-sided paired-seed t test
- 다중 비교: 실제 seed-level p-value에 대한 Holm 보정
- 문장 민감도: crossed seed × shared-sequence bootstrap interval 및 비음수 tail fraction

Bootstrap tail fraction은 p-value로 부르지 않고 진단값으로만 사용한다. 이 교정은 결과 확인 뒤 분석법을 유리하게 고른 것이 아니라, Phase 3의 관련 정책 비교가 완성되기 전에 공통 평가 설계를 코드와 문서에 일치시킨 조치다.

## 5. 재현 경로

- 구현: `src/jamoflow/phase1_analysis.py`
- 단위 검증: `tests/test_phase1_analysis.py`
- 재생성 명령:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_phase1.py
PYTHONPATH=src .venv/bin/python scripts/summarize_phase2.py
PYTHONPATH=src .venv/bin/python scripts/summarize_phase2_controls.py
```

교정 후 summary JSON은 재생성됐으며 raw per-sequence loss와 모델 checkpoint는 변경하지 않았다.
