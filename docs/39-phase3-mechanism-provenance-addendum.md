# Phase 3 mechanism-control provenance addendum

> 작성일: 2026-08-10
> 상태: **mechanism D/P 결과 생성 전 고정**
> 수정 시점의 정보: primary seed 1,729 F/C/W 완료, 나머지 primary seeds와 Gate I/OOD 미완료
> 영향: D/P 정의·seed·threshold·Gate M 불변; cache와 evidence lineage 검증 강화

## 1. 왜 결과 전에 보강하는가

Gate I가 통과할 경우 delayed-grid D와 rate-matched causal-placebo P를 새로 학습해 W의 관측 이득이 단순 grid phase 또는 임의 event 빈도로 설명되는지 검사한다. 정적 감사에서 기존 구현의 계산 자체보다 실행 재개와 최종 증거 연결에 세 가지 약점이 확인됐다.

1. `mechanism-patches.npz`는 함께 저장된 diagnostics의 matrix hash와만 대조했다. Cache와 diagnostics가 함께 다른 run에서 복사되면 현재 HPLT3 input/boundary/whitespace matrix에서 만들어졌는지 알 수 없었다.
2. 완료된 control을 건너뛸 때 seed/policy, checkpoint state와 loss-file hash만 일부 검사했다. 현재 stream, matrix, seeded order, model/optimization spec, example count와 absolute BPB까지 묶지 않았다.
3. Mechanism summarizer는 D/P/W report와 loss를 읽었지만 실제 checkpoint를 직접 hash하거나 D/P matrix를 현재 HPLT3 stream에서 독립 재구성하지 않았다.

이 문제는 정상적인 한 번의 실행에서는 잘못된 수치를 만들 가능성이 낮지만, 수 시간짜리 학습을 중단·재개하거나 initial 3 seeds와 confirmation 2 seeds를 나눠 실행할 때 stale artifact를 뒤늦게 발견할 수 있다.

## 2. Cache provenance 교정

Mechanism cache diagnostics에 다음 exact provenance를 추가했다.

- schema와 cache kind
- full Phase 3 model spec과 D/P policy names
- split별 input byte matrix SHA-256
- split별 UTF-8 boundary matrix SHA-256
- split별 observed-whitespace event matrix SHA-256

모두 일치할 때만 D/P cache를 재사용한다. 하나라도 다르거나 legacy diagnostics에 provenance가 없으면 cache를 무시하고 현재 입력에서 다시 만든다. Matrix 자체의 shape, 512-byte coverage와 diagnostics hash 검사는 그대로 유지한다.

## 3. 완료 결과의 exact-match 조건

Runner는 기존 D/P report를 다음 현재 실행 정보와 모두 대조한 뒤에만 `already complete`로 처리한다.

- seed, policy, 19,596,096 parameters
- model spec과 optimization spec
- seed에서 재구성한 shuffle-order hash
- split별 selected-stream hash
- split별 현재 D/P matrix hash
- calibration/test example와 predicted-byte counts
- test loss shape·유한성·비음수성, loss에서 재구성한 BPB
- checkpoint artifact SHA-256와 직접 계산한 state-dict SHA-256

새 control report에는 stream hash mapping과 checkpoint artifact hash도 기록한다. `--no-checkpoints`는 smoke 또는 비증거 실행에는 남겨두지만 full summarizer는 checkpoint를 저장한 evidentiary invocation만 허용한다.

## 4. Summarizer의 독립 재구성

Full mechanism summary는 current filesystem에서 다음 chain을 다시 만든다.

```text
HPLT3 processed artifact
  -> deterministic train/calibration/test streams
  -> input / UTF-8-boundary / whitespace matrices
  -> W reference and D/P matrices + placebo calibration
  -> current primary W and control D/P reports/losses/checkpoints
  -> W-D, W-P contrasts and Gate M
```

검증 항목은 다음과 같다.

1. Control manifest의 full limits, model/optimization spec, stream metadata/hash가 현재 재구성과 일치한다.
2. 각 requested seed에 적절한 gate가 기록된 checkpoint-preserving invocation이 존재한다. Initial seed는 Gate I, confirmation seed는 Gate J가 필요하다.
3. Cache diagnostics 전체가 독립 재구성한 diagnostics와 정확히 일치한다.
4. D/P matrix hash와 primary W matrix hash가 현재 재구성과 일치한다.
5. Seeded shuffle order, train/calibration/test count, exact 86-patch row 조건이 맞는다.
6. 모든 W/D/P checkpoint를 직접 읽어 state hash를 report와 대조한다. D/P는 serialized artifact hash도 대조한다.
7. Current primary summary의 W checkpoint/loss mapping과 실제로 불러온 W evidence가 일치한다.
8. Per-sequence loss에서 absolute BPB와 paired contrast를 재구성한다.

Tracked mechanism summary에는 append-only run manifest, 재구성 diagnostics, loss/checkpoint hashes와 composite integrity 결과를 남긴다. Corpus text, matrix, checkpoint와 per-sequence loss는 계속 ignored artifact 경계 밖으로 내보내지 않는다.

## 5. 분석 선택에 미치는 영향

이 수정 시점에는 primary seed 1,729의 F/C/W 결과가 존재했다. 그러나 Gate I에 필요한 나머지 두 seed와 Leipzig OOD가 없었고, D/P 결과는 하나도 생성되지 않았다. 수정은 이미 고정된 D/P algorithm, placebo calibration target, seed, training budget, contrast, threshold 또는 Gate M 판정식을 바꾸지 않는다.

따라서 primary 중간값을 이용한 method rescue가 아니라, Gate I 통과 여부와 관계없이 후속 control의 provenance를 강화한 조치다. Gate I가 실패하면 기존 stopping rule대로 D/P를 실행하지 않는다.

## 6. 검증

다음 회귀 검사를 추가했다.

- input/boundary/whitespace matrix 하나가 바뀌면 mechanism cache provenance가 달라짐
- manifest에 checkpoint-preserving evidentiary invocation이 없으면 summary 승격 거부
- 기존 matrix construction과 W-reference reconstruction 테스트 유지
- 기존 Gate M initial/final threshold 테스트 유지

전체 test suite **158개**가 통과했다.
