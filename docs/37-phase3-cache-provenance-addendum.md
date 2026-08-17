# Phase 3 patch-cache provenance addendum

> 작성일: 2026-08-10  
> 상태: **F/C/W primary process 실행 중, W seed 1,729 결과 생성 전 발견**  
> 영향: patch policy·matrix·model·seed·gate 불변; 재실행과 후속 E/EC cache 안전성 강화

## 1. 발견한 문제

Phase 3 runner는 수억 byte에 대한 patch matrix 계산을 반복하지 않기 위해 다음 ignored cache를 사용한다.

- structural F/C/W/S matrix
- seed별 learned-router E/EC threshold matrix

기존 cache loader는 matrix shape, policy 이름, row coverage만 확인했다. Run manifest는 data stream hash 변경을 거부하고 final summarizer는 current stream에서 F/C/W matrix를 독립 재구성하므로 잘못된 cache가 evidence로 승격될 가능성은 낮았다. 그러나 cache 파일과 run directory가 비정상적으로 섞이면 다음 문제가 남았다.

1. manifest를 쓰기 전에 stale structural cache를 읽어 불필요한 학습을 시작할 수 있다.
2. E/EC cache가 다른 router checkpoint에서 만들어졌는지 load 시점에 직접 확인하지 않았다.
3. 올바르지 않은 cache는 final summary에서 뒤늦게 발견돼 계산 시간을 낭비한다.

## 2. 교정

Cache diagnostics에 `_provenance`를 추가하고 exact match일 때만 재사용한다.

Structural cache provenance:

- model spec
- split별 codepoint-boundary matrix SHA-256
- split별 whitespace-event matrix SHA-256
- split별 SpaceByte-event matrix SHA-256

Threshold cache provenance:

- seed
- trained router state-dict SHA-256
- model spec과 maximum patch length
- split별 input matrix SHA-256
- split별 codepoint-boundary matrix SHA-256

Provenance가 없거나 하나라도 다르면 cache를 stale로 표시하고 재구성한다. E/EC cache를 재사용할 때도 모든 matrix를 다시 `validate_padded_patch_matrix`로 검사한다.

## 3. 현재 primary run에 대한 영향

실행 중인 F/C/W process는 수정 전 commit에서 시작했으므로 새 loader 코드를 동적으로 사용하지 않는다. 그러나 이 process의 structural cache는 같은 invocation이 읽은 HPLT3 stream으로 바로 생성됐고 다음 독립 근거가 이미 있다.

- manifest의 train/calibration/test stream SHA-256
- seed별 report의 split별 patch-matrix SHA-256
- F/C/W matrix의 seed-independent hash
- final summarizer의 current stream 재로딩 및 F/C/W matrix 독립 재구성
- 모든 row의 exact 86-patch·512-byte coverage 검사

따라서 현재 primary 결과를 폐기하거나 재학습할 이유는 없다. 다음 `run_phase3.py` invocation부터 legacy structural cache는 한 번 재구성되며, 이후 S/E/EC와 confirmation runs는 provenance가 맞는 cache만 사용한다.

## 4. 검증

두 회귀 테스트를 추가했다.

1. source event matrix의 byte 하나가 바뀌면 structural provenance가 달라짐
2. diagnostics provenance가 expected mapping과 exact match할 때만 cache가 유효함

전체 test suite 150개가 통과했다. Cache와 원문·checkpoint는 Git에 넣지 않으며, 이 문서는 evidence path의 변경 이유만 기록한다.

> 후속 감사: 당시 문서가 예고한 final summarizer의 full matrix 재구성은 구현에 빠져 있었다. 결과 판정 전에 [42 primary provenance addendum](./42-phase3-primary-provenance-addendum.md)에서 실제 구현과 검증을 추가했다.
