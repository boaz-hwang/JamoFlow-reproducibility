# Five-seed summary path correction

> 작성일: 2026-08-12
> 상태: **confirmation OOD 및 five-seed summary 생성 전 고정**

`results/phase3-primary-clustered/summary.json`은 세 초기 seed의 corrected Gate I를
봉인했고, confirmation main/OOD 실행의 authorization source로 이미 사용됐다. 같은
경로에 다섯-seed summary를 덮어쓰면 authorization이 가리키는 원본은 Git history에만
남고 작업 트리에서는 사라진다. 또한 재개 시 “현재 authorization file”과 “실행 당시
authorization blob”의 의미가 달라진다.

따라서 초기 summary는 그대로 보존하고 다섯-seed F/C/W+OOD 결과는 다음 고정 경로에
새로 생성한다.

```text
results/phase3-primary-five-seed/summary.json
results/phase3-primary-five-seed/observations.csv
```

Selection-v2 plan과 선택된 S/E/EC 비교군 승인은 새 경로의 정확한 `HEAD` blob을
요구한다. 이 summary는 exact seed 순서 `1729,2718,31415,57721,65537`, F/C/W 정책,
corrected Gate-I authorization, Gate J pass, OOD guard pass와 완전한 artifact integrity를
모두 만족해야 한다.

기존 세-seed summary는 이후에도 다음 용도로만 유지한다.

- 이미 시작된 F/C/W confirmation 및 OOD confirmation의 immutable authorization source
- development/screening 이력 재현

두 summary 모두 기존에 노출된 historical test를 사용했으므로 최종 논문 품질이나
actual timing을 직접 승인하지 않는다. 최종 승인은 calibration-only selection lock과
새 sealed final test를 사용하는 v2 evidence에서만 나온다.
