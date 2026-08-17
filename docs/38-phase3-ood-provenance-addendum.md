# Phase 3 OOD provenance and append-only execution addendum

> 작성일: 2026-08-10  
> 상태: **Phase 3 primary F/C/W 실행 중, OOD 결과 생성 전 고정**  
> 영향: OOD corpus·policy·seed·gate 불변; 실행 및 요약의 provenance 검증 강화

## 1. 발견한 문제

Leipzig Korean OOD 평가는 initial 3 seeds와, Gate I 통과 시 추가되는 confirmation 2 seeds를 서로 다른 invocation에서 실행한다. 기존 runner와 summarizer에는 세 가지 취약점이 있었다.

1. OOD runner가 매 invocation마다 `manifest.json`을 덮어써 confirmation 실행 후 initial invocation 기록을 잃을 수 있었다.
2. 이미 report와 loss 파일이 존재하면 현재 corpus, patch matrix, primary report, checkpoint와 같은 결과인지 확인하지 않고 완료된 것으로 간주했다.
3. OOD summarizer는 report 내부의 두 state-hash field가 서로 같은지만 확인했다. 현재 공개 원본, 재구성된 byte stream, 재구성된 F/C/W matrix, 실제 primary checkpoint에 독립적으로 연결하지 않았다.

Primary summarizer도 세 개의 개별 boolean만 신뢰했으며 OOD 결과가 현재 요약 중인 primary checkpoint에서 나온 것인지 다시 대조하지 않았다. 정상 실행에서는 같은 파일을 사용하므로 값이 틀릴 가능성은 낮지만, 중단·재개·directory 혼합 뒤에도 증거 계보를 보장하려면 충분하지 않다.

## 2. 결과 전 교정

OOD 실행 manifest를 append-only 구조로 바꿨다.

- top-level seed와 policy는 invocation들의 순서 보존 union
- 각 invocation의 시각, commit, device, platform, dependency version, seed, policy, `--force` 보존
- processed source의 byte 수와 SHA-256 고정
- 선택된 test byte stream과 codepoint-boundary stream SHA-256 고정
- requested byte limit, model spec, global position limit 고정
- 위 invariant가 기존 manifest와 다르면 새 결과를 쓰기 전에 중단

기존 결과 skip도 exact provenance 검증으로 바꿨다. 다음이 모두 일치할 때만 평가를 건너뛴다.

- seed, policy, model spec, parameter count
- source와 selected stream hash
- 현재 재구성한 policy patch-matrix hash
- primary training-report artifact hash
- primary checkpoint artifact hash와 state-dict hash
- OOD example/target count와 per-sequence NLL에서 재구성한 absolute BPB

하나라도 다르거나 report/loss 중 하나만 존재하면 명시적으로 실패한다. 사용자가 의도적으로 재평가할 때만 `--force`로 덮어쓸 수 있다.

## 3. 독립 OOD 요약 검증

OOD summarizer는 이제 runner의 report를 그대로 신뢰하지 않고 다음 chain을 현재 filesystem에서 다시 계산한다.

```text
public ko.jsonl artifact
  -> deterministic hash-test partition and 512-byte packing
  -> selected byte/boundary streams
  -> independently rebuilt F/C/W matrices
  -> current primary training reports and checkpoints
  -> OOD reports and per-sequence NLL
  -> BPB contrasts and OOD guard
```

구체적으로 다음을 요구한다.

1. 요청 seed가 사전등록 initial 3 또는 final 5와 정확히 일치한다.
2. 각 seed/policy pair가 실제 manifest invocation 하나에 포함돼 있다.
3. 공개 processed artifact의 크기와 SHA-256이 manifest와 같다.
4. 동일 코드로 다시 만든 stream metadata, data hash, boundary hash가 같다.
5. 동일 stream에서 다시 만든 F/C/W matrix hash가 모든 seed report와 같다.
6. 실제 primary checkpoint를 직접 읽어 계산한 state hash가 primary report와 OOD report 모두에 연결된다.
7. primary report/checkpoint의 artifact hash도 OOD report에 기록된 값과 같다.
8. 모든 F/C/W row가 정확히 86 patches, 512 bytes를 덮고 padding slot이 없다.
9. loss vector shape·유한성·비음수성·predicted-byte count와 absolute BPB 재구성이 맞다.

최종 primary summarizer는 OOD summary의 composite integrity flag뿐 아니라 seed별 F/C/W checkpoint state hash mapping을 현재 primary evidence와 다시 대조한다. 따라서 다른 학습 run에서 만든 OOD summary를 현재 Gate I/J에 잘못 결합할 수 없다.

## 4. 연구 설계와 해석에 미치는 영향

이 교정은 hypothesis, margin, corpus, patch policy, model, seed, bootstrap 또는 gate를 바꾸지 않는다. OOD 결과가 하나도 생성되기 전에 적용됐으며 active primary training에도 영향을 주지 않는다. 목적은 긍정적 결과를 만들기 위한 분석 변경이 아니라, 나중에 나온 결과가 실제로 사전 고정된 입력과 checkpoint의 결과임을 재현 가능하게 증명하는 것이다.

Leipzig 평가는 여전히 domain-transfer regression guard이지 contamination-free benchmark나 한국어 고유성 검정이 아니다. `W − C <= +0.020 BPB`와 `W − F <= +0.020 BPB`라는 기존 margin도 그대로 유지한다.

## 5. 검증

다음 회귀 검사를 추가했다.

- initial 3-seed manifest와 confirmation 2-seed manifest의 append-only union
- source/stream invariant 변경 거부
- seed/policy pair별 invocation coverage 요구
- 기존 OOD report의 stream provenance 변경 감지
- OOD summary와 현재 primary checkpoint mapping의 exact match 요구
- composite integrity flag가 없거나 false인 OOD summary 거부

전체 test suite **155개**가 통과했다. Corpus text, checkpoint, per-sequence loss와 patch matrix는 계속 Git에 넣지 않는다.
