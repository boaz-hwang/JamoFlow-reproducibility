# Mechanism reanalysis and progression authorization correction

> 작성일: 2026-08-11
> 상태: **S/E/EC family 봉인 중, corrected Gate I/M 생성 전 고정**

## 1. 문제

Initial D/P mechanism controls는 historical three-seed Gate I가 당시 protocol을 통과한 뒤 적법하게 실행됐다. 이후 packed-window dependence를 발견해 source-document clustered inference를 추가했으므로 같은 기존 loss artifact를 corrected statistics로 다시 분석해야 한다.

기존 mechanism summarizer는 corrected current Gate I가 통과해야 initial reanalysis를 시작할 수 있었다. 이 조건은 두 결정을 혼동한다.

- 이미 적법하게 생성된 artifact를 더 정확한 통계로 **재분석할 수 있는가**
- corrected primary result가 다음 two-seed mechanism **confirmation training을 허용하는가**

Current Gate I가 실패할 때 첫 질문의 답은 yes이고 두 번째는 no다. 재분석까지 막으면 이미 존재하는 control evidence의 교정 결과를 투명하게 보고할 수 없다.

## 2. 교정된 규칙

Initial three-seed mechanism summary는 다음을 모두 요구한다.

1. Historical authorization summary의 Gate I와 integrity가 실제 initial-control manifest hash에 일치
2. Corrected primary summary의 integrity와 W checkpoint/loss hash가 현재 loaded evidence에 일치
3. Corrected Gate I가 OOD까지 포함해 `pass` 또는 `fail`로 최종화되어 있음
4. D/P manifest, checkpoint, loss, patch matrix와 source-document map이 독립 재구성됨

Corrected Gate I의 pass 여부와 무관하게 1–4를 만족하면 initial Gate M contrast를 계산한다. 다만 `progression_authorized`는 corrected Gate I와 Gate M이 모두 통과할 때만 true다.

Final five-seed mechanism summary는 예외가 아니다. 새로운 confirmation seeds를 실행하려면 current Gate J가 먼저 통과해야 하고, manifest invocation이 바로 그 corrected summary hash에 연결되어야 한다.

## 3. 해석

Corrected Gate I가 실패하고 corrected Gate M만 통과하는 경우 허용되는 결론은 “historically executed controls 안에서 W가 D/P보다 낮은 loss를 보였다”뿐이다. Primary W−C/W−F method evidence나 confirmation 진행을 주장하지 않는다. 반대로 Gate I와 initial Gate M이 모두 통과해야 mechanism confirmation을 실행할 수 있다.

이 교정은 어떤 quality 값도 열기 전에 이루어졌으며 threshold, seed, contrast 또는 bootstrap 결과를 바꾸지 않는다. 분석 가능성과 후속 실행 권한을 분리해 실패 결과도 완전하게 보고하기 위한 무결성 수정이다.
