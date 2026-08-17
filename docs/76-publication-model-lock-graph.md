# Publication model-lock graph

> 작성일: 2026-08-12
> 상태: **publication 실행 전 결과맹 evidence-identity 교정 구현**
> 주의: 이 문서는 실제 품질 또는 속도 결과가 아니라 cross-evaluation false-pass 방지 계약이다.

## 1. 막아야 했던 오류

Runtime evidence는 이미 seed별 checkpoint와 config를 봉인했지만, BPB는 model family
key와 loss 배열을, downstream은 model family key와 prediction을, data adequacy는 family
key와 learning curve 숫자를 각각 받았다. 따라서 다음처럼 서로 다른 실행을 조합할 수
있었다.

1. BPB는 품질이 좋은 checkpoint A에서 계산한다.
2. Runtime은 속도가 좋은 checkpoint B에서 계산한다.
3. Downstream은 또 다른 tokenizer/config의 checkpoint C에서 계산한다.
4. 세 객체의 `candidate_key == "candidate"`만 같게 만들어 final gate에 넣는다.

Family key는 사람이 붙인 별칭일 뿐 model identity가 아니다. 이 조합을 막지 않으면
“동일 품질에서 실제 추론이 빨라졌다”는 논문의 중심 명제가 성립하지 않는다.

## 2. Model snapshot

`src/jamoflow/publication_model_lock.py`의 `PublicationModelSnapshot`은 한 model role을
다음 값으로 고정한다.

- 고정 seed 순서 `(1729, 2718, 31415)`
- seed별 checkpoint state SHA-256
- seed별 model-config SHA-256
- Raw role이면 compact calibration-only selection에서 재구성한 concrete policy descriptor와 selection/initial-summary SHA-256
- Descriptor가 E/EC이면 seed별 structured entropy-router bundle: checkpoint artifact/state, report, 공통 config, train/calibration/test stream, threshold, maximum patch length, policy definition, threshold cache/diagnostics와 split별 patch matrix SHA-256
- tokenizer SHA-256
- strict UTF-8 transition-table SHA-256
- model-lock protocol version과 snapshot identity SHA-256

세 seed의 checkpoint hash는 모두 달라야 하고 config hash는 모두 같아야 한다. 이
조건은 paired seeds를 같은 checkpoint 세 번으로 위장하거나 seed 사이에서 architecture
설정을 바꾸는 오류를 차단한다. Structural raw descriptor면 auxiliary는 반드시 비고,
E/EC descriptor면 세 router bundle이 모두 있어야 한다. Auxiliary 종류는 worker 입력이
아니라 descriptor policy에서 파생된다. Runtime lineage에도 같은 조건을 적용했다.

## 3. Evaluation별 결속

### BPB

`PublicationBPBNoninferiority`는 candidate와 comparator snapshot을 모두 가진다. 또한
다음 배열의 이름, dtype, shape와 C-order bytes를 하나의 manifest hash로 묶는다.

- 문서별 scored raw-byte 수
- 세 seed의 candidate document NLL
- 세 seed의 comparator document NLL

Raw-context rolling plan의 tokenizer hash와 comparator snapshot의 tokenizer hash가
같아야 한다. 나중에 comparator gate가 이 두 snapshot을 runtime lineage에서 재구성한
snapshot과 정확히 비교한다.

### Downstream

`PublicationDownstreamGate`는 실제 task에서 사용한 candidate/reference snapshot만
고정된 model 순서로 저장한다. Gold label, train-majority label, seed별 candidate와
reference prediction 배열을 하나의 manifest로 해시하고, 별도의 case manifest와 raw
prediction artifact hash도 봉인한다. Task/family pass와 최종 pass는 validator가 다시
계산한다.

### Learning curve와 data adequacy

각 model은 모든 사전등록 budget에 대해 `PublicationLearningCurveModelLock`을 가진다.
Budget 사이에는 main/router config, raw descriptor, tokenizer와 transition table이
같아야 하고, main checkpoint뿐 아니라 router checkpoint/calibration bundle도
budget×seed 전체에서 달라야 한다. Seed index를 회전해 이전 budget artifact를 재사용할
수 없다. Curve 숫자 배열과 private curve artifact도
별도로 해시한다. 마지막 budget snapshot은 downstream과 같아야 data-adequacy 객체를
만들 수 있다.

### Runtime과 comparator gate

Comparator gate는 runtime lineage에서 canonical candidate/comparator snapshot을 다시
만든다. BPB의 두 snapshot 및 downstream candidate snapshot과 완전히 같지 않으면
latency pass 계산 전에 실패한다. Runtime, BPB, downstream의 내부 validator를 다시
호출하고, latency seed count, valid-output rate, encoding pass와 overall boolean도 원시
nested evidence에서 재계산한다.

## 4. Final graph

Final Value Gate는 다음 네 model을 정확히 요구한다.

1. candidate
2. compact selection descriptor가 고정한 concrete raw-byte reference
3. body-matched byte-BPE 16K
4. byte-BPE 32K

세 comparator runtime pair, 세 BPB pair, downstream에서 실제 사용한 model, 네 learning-
curve final snapshot을 하나의 `PublicationModelLockGraph`로 합친다. 모든 candidate
snapshot은 하나여야 하고, comparator별 runtime·BPB·final-curve snapshot도 같아야
한다. Graph는 다음 evidence identity도 함께 봉인한다.

- comparator별 runtime evidence SHA-256 3개
- comparator별 BPB evidence SHA-256 3개
- downstream evidence SHA-256
- learning-curve/data-adequacy evidence SHA-256

Final gate 자체도 raw/BPE comparator gate identity와 data-adequacy identity를 저장하고,
claim level과 status를 nested pass에서 다시 계산한 뒤 최종 identity를 만든다.

## 5. 음성 검증

Protocol v3의 reference/model-lock/runtime/scale/downstream/data-adequacy/final-value
focused test 66개와 전체 회귀 test 371개가 통과했다. 새 검증은 다음 조작을 직접
거부한다.

- 세 seed가 같은 checkpoint hash를 재사용
- seed마다 model-config hash가 달라짐
- learning-curve budget 사이에 checkpoint를 재사용
- learning-curve budget×seed 사이에 router checkpoint/calibration/cache를 회전 재사용
- comparator 하나의 BPB candidate checkpoint만 다른 실행으로 교체
- runtime/BPB/curve 사이에서 candidate checkpoint family를 교체
- downstream prediction-array manifest를 사후 교체
- learning-curve array manifest를 사후 교체
- comparator 또는 final gate의 `overall_pass`만 사후 교체
- BPE-16K와 BPE-32K comparator role을 바꿈
- downstream 또는 data-adequacy 객체의 family key만 바꿈
- E/EC policy를 structural raw로 위장하거나 router checkpoint/config/threshold bundle을 교체

전체 test suite는 단일 MPS training family가 완전히 끝난 뒤 실행했다.

## 6. 아직 증명하지 않은 것

현재 테스트는 content-free 합성 checkpoint hash와 합성 numeric array를 사용한다.
다음 사항은 아직 publication evidence가 아니다.

- 실제 checkpoint file의 state hash 생성
- 실제 model config와 tokenizer/transition artifact hash 생성
- BPB NLL, downstream prediction과 learning curve raw artifact 저장
- 실제 runtime trial artifact와 동일 final checkpoint 결속
- 네 model의 실제 graph를 사용한 Final Value Gate 실행

따라서 이 교정은 양성 결과를 만들지 않는다. 실제 runner가 파일을 읽어 snapshot과
array manifest를 만들고, 그 결과가 동일 graph를 통과해야만 논문 표에 사용할 수
있다.
