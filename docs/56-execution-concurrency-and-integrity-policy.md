# Execution concurrency and integrity policy

> 작성일: 2026-08-11
> 우선순위: 정확도·연구 깊이·완성도 > wall-clock speed

## 1. 기본 원칙

서로의 model state, thermal state, random schedule, sealed result 또는 artifact lineage에 영향을 줄 수 있는 작업은 순차 실행한다. 병렬화는 논리적으로 독립이며 실패해도 실행 중 실험의 결과를 바꾸지 않는 작업에만 허용한다.

## 2. 반드시 순차 실행하는 작업

- MPS model training, router training/scoring, checkpoint evaluation
- actual latency와 memory benchmark
- feasibility worker처럼 thermal·allocator 상태를 공유하는 실행
- calibration selection → sealed test evaluation → timing의 의존 사슬
- 한 family의 결과를 사용해 다음 family를 여는 gate
- 같은 run/artifact/summary path를 쓰는 모든 작업

Apple MPS evidentiary process는 항상 하나만 둔다. 다른 MPS process와 겹친 latency는 폐기한다.

## 3. 허용되는 병렬 작업

하나의 MPS training이 고정 commit에서 실행 중일 때 다음은 별도 CPU 작업으로 진행할 수 있다.

- 실행 프로세스가 import하지 않은 evaluation/statistics 코드 작성과 unit test
- 문헌 원문 확인과 related-work 정리
- 결과 값을 읽지 않는 corpus layout, schema, hash, license audit
- 문서 작성, reference 정리, static type/syntax/diff 검사

실행 중 process가 import한 training/model/data/policy module은 수정하지 않는다. 분석 코드 변경은 실행 결과 artifact를 바꾸지 않아야 하며, 새 분석 commit에서 생성한 summary는 training-start commit과 analysis commit을 둘 다 기록한다.

## 4. Partial-result 봉인

초기 S/E/EC처럼 한 family를 함께 해석해야 하는 경우 모든 seed×policy run이 끝나기 전 개별 quality를 열지 않는다. Process 생존, elapsed time, 존재하는 artifact filename 같은 content-free 상태만 확인한다. 중간값에 반응해 순서·threshold·실행 수를 바꾸지 않는다.

## 5. Commit 경계

다음은 각각 별도 중요 commit으로 남긴다.

1. analysis/integrity protocol hardening
2. 완성된 family의 sealed summary와 gate 판정
3. confirmation/compute-conversion authorization
4. actual-inference runtime과 결과
5. publication-scale/downstream/BPE 결과
6. paper와 release artifact

Git commit은 재현 경계다. 외부 push, Hugging Face upload, 논문 제출은 별도 명시적 공개 단계에서만 수행한다.
