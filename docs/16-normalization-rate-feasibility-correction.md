# Protocol correction 2: NFD unit preservation cannot sustain 43 patches

> 작성일: 2026-08-10  
> 상태: **model quality 결과 확인 전 고정**  
> 선행 correction: [NFD causality correction](./15-normalization-protocol-correction.md)

## 0. 실행 전 구조 검증에서 발생한 failure

Normalization runner는 condition data와 patch matrix를 모두 검증한 뒤 checkpoint를 평가하도록 작성했다. `original`과 `NFC` matrix 생성은 성공했지만 NFD oracle matrix 생성에서 다음 invariant가 실패했다.

```text
required data patches: 43
emitted data patches on a row: 41
```

이 시점에는 어떤 normalization model quality 결과도 계산되지 않았다.

## 1. 원인

Primary rate는 256/43 = 5.953 bytes/patch다. Canonical NFD Hangul unit의 UTF-8 길이는 다음과 같다.

- `L+V`: 6 bytes
- `L+V+T`: 9 bytes

Unit 내부 경계를 모두 금지하면 종성 있는 음절이 많은 row에서 available unit boundaries 수가 43보다 작다. 따라서 다음 세 조건은 동시에 만족할 수 없다.

1. 정확히 43 patches
2. 모든 NFD `L+V+optional T` unit 보존
3. 모든 256-byte Korean row 처리

이는 구현 bug가 아니라 rate-feasibility incompatibility다. 원 correction 문서가 causal ambiguity는 찾았지만 candidate-count constraint까지 확인하지 못했다.

## 2. 금지하는 임시 해결

다음 방식은 쓰지 않는다.

- 부족한 row에서 unit 내부 boundary를 몰래 허용
- 실패 row를 test에서 제거
- test quality를 본 뒤 row별 patch rate 선택
- oracle만 patch 수를 줄이고 43-patch C1과 직접 비교

모두 rate 또는 test population confound를 만든다.

## 3. 수정된 paired oracle diagnostic

Deployable 43-patch robustness policies는 그대로 평가한다.

- C0 `fixed_byte_6_rate43`
- C1 `causal_codepoint_grid_rate43`
- W `causal_whitespace_grid_rate43`

Hangul-unit opportunity diagnostic은 별도 exact-rate pair로 바꾼다.

- U0 `causal_codepoint_grid_rate28`
- U1 `oracle_hangul_unit_grid_rate28`

두 policy는 같은 C1 checkpoint와 같은 transformed inputs를 사용하고 patch candidate만 다르다. 모든 row가 정확히 28 data patches여야 한다.

## 4. 왜 28인가

한 canonical Hangul unit의 최대 길이는 9 bytes다. Complete-unit aligned 256-byte row에서 internal unit-end candidates의 이론적 worst case는 27개 이상이다. Initial patch를 포함해 28 data patches는 모든 valid row에서 구성 가능하다.

Implementation은 이론만 믿지 않고 모든 condition의 모든 row에서 다음을 검사한다.

- U0/U1 positive patch count = 28
- positive lengths sum = 256
- U1 unit-internal boundaries = 0
- NFC U0/U1 matrix identity

NFD row 자체가 Jamo unit 중간에서 끊기지 않도록 transformed stream packing도 oracle unit-end candidate에서만 row를 끝낸다. NFD unit은 최대 9 bytes이므로 newline padding 상한은 이 diagnostic에서 0–8 bytes다. 이는 앞 correction의 codepoint-aligned 0–3-byte 규칙을 normalization oracle에 한해 대체한다. 삽입률은 condition별로 별도 보고한다.

하나라도 실패하면 oracle result를 폐기한다.

## 5. 해석 변화

U1−U0은 다음 질문만 답한다.

> 동일한 낮은 global rate에서 non-causal NFD unit oracle이 generic codepoint candidate보다 유용한가?

43-patch production candidate의 직접 성능을 답하지 않는다. 28-patch inference는 training rate 43과 mismatch이므로 절대 BPB를 method claim에 쓰지 않고 paired opportunity diagnostic으로만 쓴다.

## 6. 수정된 opportunity gate

- NFD에서 U1이 U0보다 bits/original-codepoint를 평균 1% 이상 개선
- 5 seeds 중 최소 4개가 개선
- NFC에서 U0/U1 matrix identity
- exact rate 28 invariant

통과해도 causal scale-up method로 포함하지 않는다. 별도 architecture 연구 opportunity만 연다.

## 7. 연구적으로 중요한 negative fact

NFD unit preservation은 “규칙을 적용하면 공짜로 얻는 constraint”가 아니다. 원하는 compute rate가 unit granularity보다 높으면 규칙과 rate가 구조적으로 충돌한다. 이는 논문의 normalization failure taxonomy에 포함할 결과다.
