# Document-cluster inference integrity addendum

> 작성일: 2026-08-11
> 상태: **confirmation·compute-conversion·actual-inference 결과 전 고정**
> 성격: Phase 3 initial F/C/W와 D/P 결과를 본 뒤 발견한 사후 통계 교정
> 적용: HPLT3 primary, mechanism, compute-conversion, compact inference-quality gate

## 1. 발견한 문제

Phase 3 학습·평가 stream은 독립적으로 split한 원문 문서를 newline으로 이어 붙인 뒤 512-byte window로 자른다. 기존 crossed bootstrap은 model seed와 공유 window index를 복원추출했다. 이 방식은 seed pairing은 보존하지만 같은 원문 문서에서 나온 여러 window를 독립 표본처럼 취급한다. 긴 문서 안의 문체·주제·난이도가 loss difference를 함께 움직이면 interval이 지나치게 좁아질 수 있다.

이 문제는 전체 test NLL이나 seed별 BPB point estimate를 바꾸지 않는다. 영향을 받는 것은 test sample 일반화에 대한 uncertainty와 그 interval을 사용한 gate다.

## 2. 교정된 estimand와 resampling

원본 JSONL을 다시 읽어 packed stream을 독립 재구성한다. 저장된 Phase 3 stream과 byte-for-byte 같지 않으면 중단한다. 각 512-byte window가 하나의 원문 문서 안에 완전히 들어갈 때만 해당 문서 index를 부여하고, newline 또는 두 문서 이상을 가로지르는 window는 document-cluster analysis에서 제외한다.

Bootstrap replicate마다 다음 두 축을 교차 복원추출한다.

1. paired model seeds
2. 모든 seed에 공유되는 원문 문서

선택된 문서의 eligible window는 전부 함께 들어간다. 문서는 동일 확률로 뽑고, replicate의 effect는 실제 target byte 수로 가중한다. 따라서 긴 문서가 원래 estimand에 기여한 byte 비중은 보존하면서 같은 문서 안의 window를 쪼개지 않는다.

HPLT3 layout-only audit 결과는 다음과 같다. 이 수치는 policy loss를 열지 않고 계산했다.

| Split | 전체 window | eligible window | 제외 window | eligible 비율 | 문서 수 |
|---|---:|---:|---:|---:|---:|
| calibration | 15,625 | 15,240 | 385 | 97.5360% | 386 |
| test | 31,250 | 30,517 | 733 | 97.6544% | 734 |

사전에 정한 최소 coverage는 95%다. 이보다 낮으면 document-contained subset이 full-stream estimand를 충분히 대표한다고 간주하지 않고 gate를 실패시킨다.

## 3. Gate 교정

기존 full-stream mean과 seed sign 조건은 그대로 유지한다. 여기에 다음을 추가한다.

- Gate I의 W−C document-cluster 95% upper bound `< 0`
- Gate J의 W−C와 W−F 각각 document-cluster 95% upper bound `< 0`
- initial/final Gate M의 W−D와 W−P 각각 같은 조건
- reduced-rate same-rate confirmation의 W−C에 같은 조건
- actual timing 전 five-seed quality noninferiority에서 candidate−reference document-cluster upper bound `< +0.010 BPB`
- noninferiority margin 안인 seed가 최소 4/5
- 모든 경우 eligible coverage `>= 95%`

기존 seed×window bootstrap은 동일 window set에서의 민감도 진단으로 남기되 문서 일반화 근거로 부르지 않는다. Final gate는 paired-seed interval과 document-cluster interval을 모두 요구한다.

## 4. Leipzig OOD의 다른 구조

Leipzig artifact는 평균 길이가 512 bytes보다 짧은 Wikipedia 문장 레코드를 연속 packing한 것이다. layout audit에서 2,818개 window 중 단일 레코드 안에 완전히 들어가는 window는 0개였다. HPLT 방식의 document-contained bootstrap을 여기에 적용하면 표본이 사라진다.

이 OOD endpoint는 처음부터 superiority나 significance test가 아니라 `W−C <= +0.020 BPB`, `W−F <= +0.020 BPB`의 심각한 회귀 방지 guard다. 따라서 gate는 seed-paired full-stream mean만 사용하며, window bootstrap은 진단으로만 표시한다. OOD 결과를 독립 문서에 대한 유의성 또는 한국어 전체로의 일반화 근거로 사용하지 않는다.

## 5. 시간적 투명성

이 교정은 initial F/C/W와 initial D/P 결과를 이미 본 뒤 작성했다. 그러므로 해당 두 contrast에 대해 `preregistered document-cluster analysis`라고 부르지 않는다. 반면 다음 결과는 보지 않은 상태였다.

- S/E/EC initial 9-run family의 완성 결과
- confirmation seed 57,721/65,537
- reduced-rate compute conversion
- actual incremental timing
- publication-scale raw/BPE/downstream 결과

S/E/EC는 family 전체가 끝날 때까지 개별 quality를 열지 않는다. Historical `results/phase3-primary/summary.json`은 당시 authorization 기록으로 보존하고, 교정 결과는 새 경로에 쓴다. 이후 진행 여부는 교정된 gate만 결정한다.

Initial D/P artifact는 historical Gate I로 이미 적법하게 생성되었으므로 corrected Gate I의 pass/fail과 무관하게 document-cluster 통계로 재분석한다. 다만 mechanism confirmation 진행은 corrected Gate I와 corrected initial Gate M이 모두 통과할 때만 허용한다. 이 구분은 [mechanism reanalysis authorization correction](./58-mechanism-reanalysis-authorization-correction.md)에 고정한다.

## 6. 구현과 공개 범위

`src/jamoflow/document_inference.py`가 content-free document assignment, coverage, crossed bootstrap을 구현한다. Tracked output에는 문서 index vector의 hash와 aggregate count만 저장한다. record ID, URL, 원문, per-document loss는 공개하지 않는다.
