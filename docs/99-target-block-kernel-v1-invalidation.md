# Target block-kernel v1 측정 무효화

> 작성일: 2026-08-13
>
> 상태: **v1 inference 수치 사용 금지; v2 재측정 필요**

## 1. 발견

봉인된 v1 runner는 정상 종료했고 표면상 다음 수치를 냈다.

- weighted micro target reduction: 57.592%
- perfect-Hangul whole-path reduction: 42.613%
- fixed-head projection: 46.681%
- logits argmax 5,409회와 cache diagnostics 704회 일치

그러나 실행 초기에 PyTorch가 비교 tensor에 gradient가 붙어 있다는 경고를 냈다. 소스를
재검사한 결과 `load_actual_model` 뒤의 measurement가 `torch.inference_mode()` 안에 있지
않았다. 실제로 v1의 sequential micro cost는 byte당 6.697 ms로, 같은 checkpoint 계열을
inference mode에서 잰 component profile의 약 2.36 ms와 크게 어긋났다.

## 2. 판정

Autograd graph 생성은 실제 추론 workload가 아니며 sequential/block 경로를 똑같이
왜곡한다고 보장할 수 없다. 따라서 v1 summary의 `full_speculative_runtime_authorized`는
무효다. Correctness 비교는 참고할 수 있지만 speed gate와 다음 구현 authorization에는
사용하지 않는다.

원 summary를 삭제하거나 수치를 바꾸지 않는다. 대신
`results/target-block-kernel-v1/invalidation.json`이 해당 artifact hash를 정확히 가리키고
모든 positive authorization을 철회한다. 이는 결과가 유리하다는 이유로 알려진 실행 결함을
무시하는 것을 막기 위한 forensic record다.

## 3. v2 요구사항

새 namespace와 manifest를 사용한다.

1. model load 이후 micro, whole timing, correctness oracle 전체를
   `torch.inference_mode()`로 감싼다.
2. 각 measurement 함수가 inference mode 활성화를 직접 assert한다.
3. case, checkpoint, repetitions, bootstrap seed와 gate는 v1에서 바꾸지 않는다.
4. v2 implementation hash를 새로 봉인하고 기존 v1 output을 덮어쓰지 않는다.
5. v2가 같은 gate를 통과할 때만 exact rollback prototype으로 진행한다.

이 수정은 결과를 보고 threshold를 바꾸는 것이 아니라 추론이 아닌 autograd workload를
측정한 구현 오류를 고치는 것이다.
