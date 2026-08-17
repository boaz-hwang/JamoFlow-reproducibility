# Retrieval mechanism v1 invalidation and v2 correction

> 작성일: 2026-08-15
>
> v1 plan SHA-256: `961550a73ceec8a4d79b50b5cdd49db2a3ffe28b62100d6c65ace5d3589975e6`
>
> 판정: **v1 result 미생성; serialization bug로 무효화**

## 1. 실패 지점

V1 replay는 checkpoint에서 target trace를 재생성하고, proposal event를 복원하고, actual timing
counters와 exact 교차검증한 뒤 summary를 메모리에 만들었다. 그러나 result hash를 계산하는
canonical JSON serialization에서 `TypeError`가 발생했다.

Paired-case coverage가 부족한 branch에서는 `np.isfinite(...) and ...` 표현의 첫 항이 false가 되며
Python `bool`이 아니라 `numpy.bool_`가 남을 수 있었다. 또한 unavailable point/interval diagnostic은
NaN이었다. Canonical JSON은 NumPy scalar와 non-finite number를 허용하지 않으므로 result publish
전에 중단됐다.

## 2. 비노출 범위

- v1 result path는 생성되지 않았다.
- Git history에도 v1 result가 없다.
- 실행 stdout에는 status, contrast, cycle count, effect, aggregate가 출력되지 않았다.
- 관찰된 것은 예외 type/stack과 insufficient/non-finite branch가 존재한다는 사실뿐이다.
- actual event aggregates나 gate 결과는 수정 전에 읽지 않았다.

따라서 가설, threshold 또는 feature를 결과에 맞춰 바꿀 근거가 노출되지 않았다. 다만 v1 plan의
implementation hash를 소급 수정하지 않고 역사로 보존한다.

## 3. V2의 유일한 변경

다음 두 값을 명시적으로 Python `bool`로 정규화한다.

- `minimum_effect`
- `bootstrap_lower_positive`

Coverage가 부족해 계산할 수 없는 point/interval은 NaN 대신 JSON `null`로 기록한다.

그 외에는 모두 v1과 같다.

- 같은 closed 64 cases
- 같은 checkpoint/table/target output regeneration
- 같은 free hybrid prompt source
- 같은 `within_hangul_eojeol - after_whitespace` contrast
- 각 stratum 32 cycles, 16 paired cases, 0.25 token/cycle, bootstrap lower >0
- secondary fallback 없음
- latency/efficiency claim 없음

V2는 새 plan/result namespace를 사용하고 v1 plan artifact hash와 `v1_result_published=false`를
correction block에 봉인한다.
