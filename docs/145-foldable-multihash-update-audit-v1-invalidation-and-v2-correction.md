# Foldable multi-hash update audit v1 무효화와 v2 교정

> 작성일: 2026-08-15
>
> 상태: v1은 model forward·gradient·update 관측 전에 중단; v2 교정 뒤 별도 shape bug로 다시
> pre-forward 중단됐으며 `docs/146`의 v3가 지배

## 실패 내용

V1 plan은 commit `fcbce1c`에 정상 봉인됐다. 그러나 runner는 MPS model을 만들기 전에 과거 B1
plan의 current-file validator에서 중단됐다.

오류는 다음이었다.

`ValueError: foldable residual implementation or protocol differs`

과거 B1 plan은 실행 당시의 implementation hash에 `docs/139`와 `docs/140`도 포함했다. B1 결과를
개봉한 뒤 두 문서에는 결과와 branch 종료 해석이 추가됐지만, training/runtime source는 바뀌지
않았다. Current-file validator는 이 정당한 historical documentation drift도 model-code drift와
같이 거부했다.

V1에서는 다음이 일어나지 않았다.

- train stream encoding
- checkpoint load
- model forward/backward
- gradient 또는 update 관측
- result artifact 생성

따라서 v1은 결과를 본 protocol이 아니라 pre-observation provenance failure다.

## V2 교정

V2는 과거 parent plan을 다음 두 시간축으로 검증한다.

1. parent plan의 `git_commit_before_plan`에서 모든 implementation file blob의 SHA-256이 봉인값과
   정확히 같아야 한다.
2. 현재 runtime에서는 parent implementation 중 executable source와 dependency가 당시 봉인값과
   같아야 한다. Current drift는 결과 해석이 추가된 정확히 `docs/139`, `docs/140` 두 파일만
   허용한다.

허용 drift set이 하나라도 더 많거나 적으면 실패한다. 즉 문서 변경을 이유로 source drift를
묵인하지 않고, historical blob과 현재 executable identity를 분리한다.

Protocol ID, plan/result path와 schema를 v2로 올린다. V1 plan은 삭제·덮어쓰기하지 않고 failed
historical artifact로 보존한다. Metric, first batch, optimizer, projection multiplier와 safety range는
v1에서 바꾸지 않는다.

## Claim boundary

이 교정은 결과나 model quality를 사용하지 않았다. Audit의 역할은 여전히 첫 AdamW update geometry를
관찰해 다음 dense control의 단일 배율을 고정하는 것뿐이다.
