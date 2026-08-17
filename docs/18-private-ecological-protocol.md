# Phase 2c protocol: read-only ecological regression check

> 사전 고정일: 2026-08-10
>
> 상태: **neural evaluation 결과 확인 전 고정**
>
> 상위 protocol: [Phase 2 Korean causal protocol](./10-phase2-korean-causal-protocol.md)

## 1. 목적과 증거 수준

이 실험은 Korean Wikipedia에서 학습한 compact byte-BLT의 whitespace-aware 이득이 사용자가 직접 작성한 Markdown 문서에서 큰 회귀로 바뀌지 않는지 검사한다. 이는 Gate E의 ecological no-regression diagnostic이지 primary evidence가 아니다.

- 한 사용자의 convenience sample이므로 한국어 모집단을 대표하지 않는다.
- Phase 0에서 이미 proxy audit에 사용했으므로 독립 external benchmark라고 부르지 않는다.
- 결과가 좋아도 일반적 domain generalization을 주장하지 않는다.

## 2. 읽기 전용 corpus 구성

사용자가 지정한 private vault에 대해 다음 deterministic pipeline을 쓴다.

1. recursive scan을 `.md`에만 제한한다.
2. Markdown 파일 하나를 record 하나로 취급한다.
3. exact-byte SHA-256 ID로 중복을 제거한다.
4. 기존 `split_for_record` 규칙의 content-hash `test` partition만 선택한다.
5. strict UTF-8 valid record만 상대 경로 순서로 newline 하나를 두고 연결한다.
6. 원 byte를 normalization하지 않고, 256-byte row가 되지 못하는 마지막 tail만 버린다.

학습과 threshold calibration에 private 문서를 사용하지 않는다. 실험 script는 파일을 읽기만 하며 vault 안에 아무 파일도 생성·수정·이동하지 않는다.

## 3. 정책과 checkpoint

5개 Phase 2 seed(1,729 / 2,718 / 31,415 / 57,721 / 65,537)의 기존 checkpoint를 그대로 평가한다.

| 라벨 | 학습된 checkpoint | 평가 patch matrix |
|---|---|---|
| C0 | `fixed_byte_6` | exact 43 fixed-byte patches |
| C1 | `causal_codepoint_grid` | exact 43 causal codepoint patches |
| W | `causal_whitespace_grid` | exact 43 causal whitespace-window patches |

모델은 자신을 학습할 때 쓴 정책과 같은 matrix로 평가한다. Policy별 parameter 수와 graph는 동일해야 하고, 모든 row에서 data patch 수 43을 검증한다.

## 4. 지표와 통계

Primary ecological contrast는 `W − C1` test BPB다. 다음을 보고한다.

- seed별 BPB와 paired difference
- 5-seed mean, sample SD, paired-t 95% interval
- sequence-level paired bootstrap 95% interval(10,000 resamples, seed 20260810)
- C1 − C0 diagnostic
- aggregate Korean strata; 50 windows 미만 stratum은 해석하지 않음

Gate E의 external component는 다음을 모두 만족할 때만 통과한다.

1. mean `W − C1 <= +0.02 BPB`
2. 다섯 seed 각각에서 `W − C1 <= +0.02 BPB`
3. W와 C1의 data patch 수가 모든 row에서 43으로 동일

Paired-t/bootstrap upper bound가 +0.02를 넘는지도 보고하지만, 기존 protocol의 표본 회귀 margin을 사후적으로 더 엄격하게 바꾸지 않는다.

## 5. Privacy guardrail

추적하지 않는 것:

- vault 경로, 파일명, record ID
- 원문, prompt, 생성 sample
- 문서별 또는 sequence별 metric
- private content의 hash

로컬 raw run은 `.gitignore`가 적용되는 `results/private/`에 두고, 추적 결과에는 다음 aggregate만 올린다.

- 선택/valid/duplicate record 수
- 사용 byte와 row 수
- 전체/stratum 수준 BPB와 confidence interval
- patch invariant과 gate 판정

## 6. 결과에 따른 행동

- 통과: whitespace-aware policy를 Phase 3 후보로 유지하되 convenience-sample evidence로만 표시한다.
- 실패: Korean whitespace method를 scale-up 중심 기여에서 제외하고 generic codepoint-safe analysis로 축소한다.
- 어느 경우든 SpaceByte 대비를 대체하지 않으며, private corpus 결과를 논문 primary table의 독립 benchmark로 사용하지 않는다.
