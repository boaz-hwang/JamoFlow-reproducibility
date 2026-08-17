# Locked comparator artifact-lineage correction

> 작성일: 2026-08-11
> 상태: **comparator selection·confirmation·actual timing 전 고정**

## 1. Threat model

Calibration-only comparator selection은 initial Phase 3와 compute-conversion summary의 hash를 selection JSON에 고정한다. 그러나 five-seed summary가 같은 initial three-seed checkpoint/loss를 실제로 재사용했는지 확인하지 않으면 다음 불일치가 가능하다.

1. Selection은 locked initial summary의 calibration BPB로 policy A를 고른다.
2. 그 뒤 initial checkpoint, loss, router 또는 training report가 교체된다.
3. Five-seed summary는 교체된 artifact와 confirmation seeds를 묶어 유효한 새 summary를 만든다.
4. Selection JSON 자체는 바뀌지 않아 기존 검증만 통과한다.

이 경우 “고를 때 평가한 A”와 “quality/timing에 쓴 A”가 같다는 보장이 없다.

## 2. Corrected lineage requirement

Five-seed inference-quality summarizer는 locked initial summary 파일 hash를 확인한 뒤, initial seeds `1729/2718/31415`에 대해 다음을 final summary와 직접 비교한다.

- F/C/W: training report, test-loss artifact, checkpoint artifact, checkpoint state hash
- 선택된 Phase 3 reference: 같은 네 hash
- reference가 E/EC이면 **각 seed별** entropy router와 router report, threshold cache 및 diagnostics hash. Router 공유는 같은 seed 안에서 E와 EC가 같은 learned router state를 사용한다는 뜻이며 seed 사이 checkpoint 공유를 뜻하지 않는다.
- selected-rate whitespace candidate와 same-rate codepoint control: conversion training report, loss, checkpoint artifact/state hash
- Phase 3 source/integrity artifact, model/optimization spec, byte limits와 모든 stream metadata
- compute-conversion source context

하나라도 다르면 five-seed noninferiority와 actual timing을 시작하지 않는다. Nonselected S/E/EC의 final checkpoint까지 보존할 필요는 없지만, selection artifact 안의 historical 값은 immutable initial-summary hash로 남는다.

## 3. Scope

이 교정은 policy 선택 기준이나 margin을 바꾸지 않는다. Calibration-only 선택과 sealed test evaluation 사이의 model identity를 보장하는 provenance 강화다. Actual benchmark는 이 quality summary hash와 다시 실제 checkpoint/router hash에 연결되므로 selection → quality → runtime의 세 단계가 닫힌다.
