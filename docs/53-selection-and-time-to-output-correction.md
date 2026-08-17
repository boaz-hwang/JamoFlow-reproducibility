# Selection and time-to-output correction

> 작성일: 2026-08-11
> 상태: **S/E/EC family 완성·comparator 선택·actual timing 전 고정**
> 성격: evaluation leakage와 latency off-by-one 교정
> 후속 교정: free-running의 고정 raw-byte 정의는 [valid-output actual-inference correction](./65-valid-output-actual-inference-correction.md)의 protocol v4가 대체함

## 1. Comparator 선택에서 test 재사용 금지

기존 protocol은 initial test BPB가 가장 낮은 F/C/W/S/E/EC/selected-C-rate를 actual-inference reference로 고른 뒤 같은 test loss로 candidate noninferiority를 판정했다. 이는 test가 model selection과 final evaluation에 동시에 쓰이는 구조다.

교정 후 모든 선택은 calibration split에서만 한다.

- E와 EC 중 learned comparator 선택: shared-seed mean calibration BPB
- actual-inference primary raw reference 선택: initial-three-seed mean calibration BPB
- exact tie: 결과 전 고정한 candidate order
- selected W rate: 기존 calibration-only 64-first, then-72 rule

Test BPB는 선택 JSON을 고정한 뒤 quality/noninferiority 평가에만 사용한다. Selection artifact에는 calibration 값과 선택 결과를 기록하고, test 값은 설명용으로만 별도 기록한다. Latency는 선택에 사용하지 않는다.

## 2. 정확한 time-to-N output

Autoregressive prefill의 마지막 logit은 첫 output unit을 이미 예측한다. N개 output byte를 얻는 데 필요한 동작은 다음과 같다.

- prompt parallel prefill 1회
- output byte 1을 prefill logit에서 선택 또는 채점
- output byte 1…N−1을 cache에 넣는 incremental forward N−1회
- 마지막 output 뒤 사용하지 않을 next-logit forward는 실행하지 않음

기존 trial은 continuation N개를 모두 `consume()`해 N+1번째 logit까지 계산했다. 이는 time-to-N이 아니라 불필요한 한 step을 포함한 값이다. 교정된 runtime은 `emitted_output_bytes=N`, `decode_forward_steps=N−1`, `observed_bytes=prompt_bytes+N−1`를 매 trial에서 검증한다. N=1 unit test는 decode forward가 0임을 고정한다.

Controlled replay는 동일 truth continuation의 N개 conditional predictions를 얻는 systems estimand다. 이 문서가 고정했던 free-running의 정확히 N raw bytes 정의는 후속 v4에서 최소 N valid bytes의 첫 UTF-8 accept state로 강화됐다. 따라서 N--N+3 argmax와 N−1--N+2 feedback forward가 가능하며 실제 수를 trial별로 기록한다.

## 3. Prompt 독립성과 timing 안정성

- HPLT test의 단일 원문 문서 안에 512-byte window 전체가 들어가는 경우만 prompt 후보로 사용
- warmup+measured 72개는 원문 문서당 최대 하나만 선택해 prompt bootstrap 단위를 source document와 일치시킴
- strict UTF-8 scalar boundary, 128-byte prompt, 128-byte continuation, Hangul-heavy 조건 유지
- raw prompt·continuation·generation은 tracked artifact에 저장하지 않음
- prompt마다 independent repetition을 3회에서 5회로 상향
- repetition median을 먼저 구하고, seed×prompt crossed bootstrap에서 prompt를 paired resample
- runtime repetition을 독립 통계 표본으로 세지 않음
- policy order는 seed·mode·prompt·repetition마다 고정 seed로 randomize
- selector/router, cache update, argmax, synchronization은 해당 estimand의 timing 안에 포함

전체 실측 session 시작과 각 seed의 측정 직전·직후에 AC 전원, AC 기본 power mode(`0`), thermal/performance warning 부재를 확인한다. 상태를 읽지 못하거나 조건을 만족하지 않은 seed artifact는 증거로 저장·재사용하지 않는다. 결과가 불안정하면 repetition을 결과를 본 뒤 선택적으로 늘리지 않고 protocol 전체를 새 버전으로 다시 실행한다.

실행기와 요약기의 drift를 막기 위해 protocol version, seeds, modes, roles, 5회 repetition, time-to-output forward 수와 runtime-observed byte 수는 `src/jamoflow/actual_inference_protocol.py` 한 곳에서만 정의한다. Protocol v4의 case selector는 72개 고유 source document를 강제한다. 요약기는 controlled 128-byte output의 127 feedback forwards/255 runtime-observed bytes와 free-running 128--131 valid bytes의 대응 관계를 각각 독립 검증한다.

## 4. Artifact 전환

Historical summary를 덮어쓰지 않는다.

- 기존 F/C/W authorization: `results/phase3-primary/summary.json`
- 교정 F/C/W: `results/phase3-primary-clustered/summary.json`
- initial six-policy: `results/phase3-all-initial/summary.json`
- 교정 mechanism: `results/phase3-mechanism-clustered/summary.json`

Confirmation, compute-conversion, comparator selection과 timing은 교정 artifact hash에만 연결한다.
