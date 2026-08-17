# Phase 3 normalization-stress provenance addendum

> 작성일: 2026-08-10
> 상태: **NFC/NFD 평가 결과 생성 전 고정**
> 수정 시점의 정보: primary seed 1,729 F/C/W 완료, normalization 결과 0개
> 영향: normalization condition·metric·해석 불변; seed 선택과 evidence reconstruction 강화

## 1. 정적 감사에서 확인한 문제

기존 NFC/NFD runner는 source와 변환 geometry를 자세히 기록했지만, 최종 summarizer는 대부분을 manifest에서 다시 읽었다. 다음 문제가 남아 있었다.

1. Summarizer가 등록된 seed 중 임의의 2개 이상을 허용해, initial 3 또는 final 5라는 분석 단위를 강제하지 않았다.
2. Current HPLT3 artifact에서 source stream, strict-decodable prefix, NFC/NFD padded stream, target mask와 F/C/W matrix를 독립 재구성하지 않았다.
3. Report 안의 checkpoint hash 두 값이 같은지만 확인했으며, 현재 primary report/checkpoint artifact와 직접 대조하지 않았다.
4. Per-sequence loss에서 BPB만 재구성하고, 공통 source denominator를 쓰는 세 scaled metric과 `total_nll_nats`를 다시 계산하지 않았다.
5. 기존 결과 파일이 존재하면 current stream, matrix, checkpoint와 같은 결과인지 확인하지 않고 완료된 것으로 간주했다.

Normalization은 natural-text gate가 아닌 stress diagnostic이지만, 논문에서 Unicode robustness의 근거로 제시하려면 동일한 provenance 기준이 필요하다.

## 2. Seed와 invocation 규칙

Full summary가 허용하는 seed set을 다음 둘로 제한했다.

- initial: 1,729 / 2,718 / 31,415
- final: 위 세 seed + 57,721 / 65,537

각 seed/policy pair는 append-only manifest 안에서 `prepare_only == false`인 실제 evaluation invocation에 포함돼야 한다. `--prepare-only`로 만든 geometry manifest만으로 결과를 승격할 수 없다. Runner는 중단 후 특정 seed/policy만 재개할 수 있지만, summarizer는 사전등록된 완전한 분석 단위만 받는다.

## 3. Runner lineage와 safe resume

Normalization manifest source에 processed `ko.jsonl`의 file size와 SHA-256을 추가했다. Model의 global-position limit도 invariant로 고정했다.

각 normalization report는 다음 primary lineage를 추가로 기록한다.

- primary training-report artifact SHA-256
- primary checkpoint artifact SHA-256
- checkpoint state-dict SHA-256
- training report의 trained-state SHA-256
- model spec과 global-position limit

기존 report/loss를 건너뛰기 전에는 위 lineage뿐 아니라 condition stream, target mask, patch matrix/diagnostics, target-count vector, loss shape·유한성·비음수성, example/predicted-byte counts와 모든 scalar metric을 재계산한다. 하나라도 다르면 stale result로 중단하고 의도적 재평가에만 `--force`를 요구한다.

## 4. Summarizer의 독립 재구성

Summarizer는 current filesystem에서 다음을 다시 계산한다.

```text
processed HPLT3 ko.jsonl
  -> primary 16M-byte test stream
  -> strict-decodable common source text
  -> NFC and NFD transforms
  -> terminal-padded 512-byte rows + target masks
  -> F/C/W patch matrices and Hangul-unit diagnostics
  -> primary checkpoints + normalization losses
  -> BPB and source-normalized stress metrics
```

검증 범위는 다음과 같다.

1. Processed source artifact의 byte 수와 SHA-256
2. Primary test manifest의 model/optimization spec, byte limit와 selected-stream hash
3. Source metadata, strict UTF-8 prefix hash, discarded terminal bytes와 공통 denominators
4. NFC/NFD actual/padded stream hash, target-mask hash와 geometry
5. Current stream에서 재구성한 F/C/W matrix와 전체 patch diagnostics
6. 실제 primary checkpoint의 serialized artifact hash와 state-dict hash
7. Primary training-report artifact와 trained-state hash
8. Per-row target counts가 current target mask와 정확히 같은지
9. Per-sequence NLL에서 재구성한 total NLL, BPB와 다음 세 metric
   - scored bits/source UTF-8 byte
   - scored bits/source Unicode codepoint
   - scored bits/source precomposed Hangul syllable

Tracked summary에는 append-only run manifest, primary context hash, checkpoint lineage, condition geometry와 aggregate만 남긴다. 원문이나 normalized text, per-sequence loss와 checkpoint는 tracked artifact로 승격하지 않는다.

## 5. 분석과 주장에 미치는 영향

NFC/NFD 정의, row-leading target omission, terminal-padding mask, F/C/W policy, metric, denominator와 “별도 decision gate 없음”은 바뀌지 않았다. 수정 시점에 normalization 결과는 하나도 없었다. Primary seed 1,729 중간값은 알려져 있었지만 이 보강은 normalization effect의 방향이나 크기를 사용할 수 없는 상태에서 이루어졌다.

NFD는 계속 synthetic canonical-equivalence stress이며, 결과가 좋아도 Jamo-aware architecture 증거로 해석하지 않는다. 결과가 나빠도 natural-text Gate I/J/K를 변경하지 않는다.

## 6. 회귀 검증

추가한 검사는 다음과 같다.

- 임의의 2-seed subset summary 거부
- evaluation invocation이 없는 prepare-only manifest 거부
- 완료 report의 checkpoint lineage 변경 감지
- target-count와 total-NLL 기반 네 metric 산술 재구성
- 기존 manifest invariant, strict terminal UTF-8 처리, paired-effect 계산 유지

전체 test suite **161개**가 통과했다.
