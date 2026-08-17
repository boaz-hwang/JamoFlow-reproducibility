# EXAONE resource calibration V1 invalidation and V2 correction

> 작성일: 2026-08-15
>
> 상태: **V1 무효화, V2 재봉인 전 교정 기록**

## V1에서 실제로 일어난 일

V1 plan은 commit `906088292885e4219a320b97893660daba40c326`에서 정상 봉인됐다. 첫 실행은
모델 snapshot hash와 MLX model load를 마친 직후, 첫 baseline trial을 만들기 전에 중단됐다.

원인은 새 공통 runtime의 config projection이 EXAONE의 이미 검증된 MLX schema인 `num_layers` 대신
Hugging Face 계열 alias `num_hidden_layers`를 요구한 것이었다. 실제 pinned config와 기존 V4
compatibility runner는 모두 `num_layers=32`를 사용한다.

중단 위치는 trial list comprehension보다 앞이다. 따라서 V1에서 다음 값은 관측되지 않았다.

- baseline latency 및 baseline output
- retrieval table load
- candidate latency, output, acceptance, target-call reduction

V1 `.active` marker는 삭제하지 않고 ignored forensic artifact로 남겼다. Tracked invalidation record에는
그 exact payload와 hash를 함께 넣어 public clone에서도 V1 plan, runner commit과 실패 단계를 재검증할
수 있게 했다.

## V2의 최소 교정

V2는 model config를 임의로 번역하지 않는다. V4 compatibility에서 이미 통과한 것과 같은 방식으로
`PRIMARY_MODEL.config_projection`의 exact key/value projection을 검사한다. 이 교정 외에 workload,
baseline-only result boundary, 75% memory gate, 8시간 schedule rule은 V1에서 바꾸지 않는다.

V2 plan은 다음을 추가 dependency로 봉인한다.

- V1 plan artifact
- V1 tracked invalidation record
- V1 active marker의 tracked exact payload와 hash

따라서 실패를 지우고 같은 namespace에서 다시 실행하지 않는다. V2는 새 plan, artifact, result
namespace만 사용한다.

## 해석 경계

이 수정은 실험 결과를 보고 protocol을 최적화한 변경이 아니다. 첫 baseline latency와 candidate 결과가
모두 관측되기 전에 발견된 loader schema 오류를 고친 것이다. V2 결과 역시 baseline resource
feasibility만 결정하며 retrieval 효율 근거가 아니다.
