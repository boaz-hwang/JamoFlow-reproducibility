# Phase 3 addendum: public Korean OOD guard

> 작성일: 2026-08-10  
> 상태: **Phase 3 primary 결과 생성 전 고정**  
> 상위 protocol: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md)
> 결과 전 provenance 보강: [OOD provenance addendum](./38-phase3-ood-provenance-addendum.md)

## 1. 목적

Gate I는 HPLT3 in-domain quality만으로 통과하지 않는다. 기존에 pin한 Leipzig Korean Wikipedia 2021 corpus의 hash-test split 전체를 public domain-transfer guard로 사용한다.

이 데이터는 Phase 2 compact model의 평가에 사용됐지만 Phase 3 HPLT3 model의 학습·calibration에는 사용되지 않는다. HPLT web crawl과 내용상 overlap이 없다고 보장할 수 없으므로 contamination-free benchmark라고 부르지 않는다.

## 2. 고정 데이터

- processed source: `data/processed/leipzig-wikipedia-100k-controls/ko.jsonl`
- split: 기존 `stable_record_id(text_bytes)`의 test partition
- sequence length: 512 raw UTF-8 bytes
- available stream: 1,442,916 bytes
- usable stream: **1,442,816 bytes**
- sequences: **2,818**
- valid selected records: 10,138
- discarded tail: 100 bytes
- starts inside codepoint: 1,675 sequences

Source manifest와 processed hash는 기존 tracked Leipzig manifest를 따른다. HPLT3 test와 합치지 않고 별도 결과로 보고한다.

## 3. 정책과 평가

Gate I에는 F/C/W 세 policy를 평가한다.

- Phase 3에서 학습된 checkpoint를 그대로 load
- OOD fine-tuning, threshold calibration, early stopping 없음
- Phase 3의 동일 structural algorithm으로 OOD patch matrix 생성
- seed 1,729 / 2,718 / 31,415
- per-sequence NLL에서 BPB 계산
- seed-level paired t interval과 hierarchical paired bootstrap 보고

Checkpoint state hash는 training report의 `trained_state_sha256`와 일치해야 한다. 공개 source와 선택 stream, 재구성 patch matrix, training-report artifact, checkpoint artifact/state, per-sequence loss의 전체 provenance chain도 독립 재검증한다. F/C/W는 OOD에서도 모든 row에 정확히 86 patches여야 한다.

## 4. Gate I OOD 판정

다음 두 mean contrast를 검사한다.

- `W − C <= +0.020 BPB`
- `W − F <= +0.020 BPB`

둘 다 만족할 때만 OOD guard를 통과한다. Interval이 0을 포함하는 것은 stop condition이 아니다. 이 guard는 superiority test가 아니라 심각한 regression 방지 기준이다.

Gate I 최종 통과는 다음의 conjunction이다.

1. HPLT3 mean `W − C <= −0.002 BPB`
2. HPLT3 3 seeds 중 W−C가 최소 2개 negative
3. 위 Leipzig OOD 두 regression margin 통과
4. initialization/order/rate/checkpoint integrity 통과

Private Markdown 결과는 추가 ecology diagnostic으로만 사용하며 이 public gate를 대체하지 않는다.
