# 8K compositional head 품질 결과와 vocabulary-transfer 전환

> 작성일: 2026-08-14
>
> 상태: sealed one-seed development rejection; pure codebook branch 종료

## 결론

8K Hangul compositional codebook은 random-weight systems 단계에서 BPE-2K보다
end-to-end generation을 19.56% 줄였지만, 실제 128M-byte 학습 뒤에는 사전 고정한 품질
gate를 크게 실패했다. Dense BPE-2K 대비 document BPB 차이는 `+0.20854`, bootstrap 95%
upper는 `+0.21157`로, 허용한 `+0.010` margin의 약 21배다. 따라서 trained-model latency를
측정하거나 generic·low-rank 역할로 사후 교체하지 않는다.

다만 음성 결과 안에도 한 가지 재현된 언어학적 신호가 있다. True Hangul assignment는
byte-length-stratified shuffled assignment보다 document BPB가 `0.01149` 낮았고 bootstrap
upper도 `-0.01068`이었다. 즉 onset/vowel/coda alignment는 완전히 무의미하지 않았다. 그러나
generic code보다 `0.01181` 나빴고 low-rank보다 `0.09606` 나빴으므로, 이 신호만으로 실제
효율 연구를 계속할 근거는 없다.

연구 계획은 다음처럼 필요한 범위에서만 바꾼다.

1. 16×128 pure codebook과 그 Hangul/shuffled/generic assignment 분기는 종료한다.
2. 8K에서 가장 먼저 해결해야 할 문제를 head 압축이 아니라 large-vocabulary
   optimization/data sparsity로 재정의한다.
3. 기존 trained BPE-2K checkpoint를 8K tokenizer로 옮기는 generic subword-composition
   initialization과 짧은 continued-pretraining curve를 먼저 검증한다.
4. Generic transfer가 고정된 품질 회복 기준을 통과할 때만 Jamo-aware initializer 또는
   residual을 연다. 그때도 SCRIPT를 직접 기준선으로 다루고, Jamo injection 자체의 최초성을
   주장하지 않는다.
5. 실제 trained batch-1 E2E가 10% 이상 개선되는 후보가 생긴 경우에만 multi-seed와 큰
   scale로 확장한다.

## 봉인된 품질 결과

모든 역할은 같은 Korean train stream 128,000,000 raw bytes, 같은 model seed와 body
initialization을 사용했다. 8K 내부 다섯 역할은 같은 tokenizer, sequence order와 1,675
optimizer steps를 사용했다. Dense 2K는 tokenizer 차이로 2,213 steps를 사용했다.

| role | trainable params | train time | contiguous BPB | document BPB | dense 2K 대비 |
|---|---:|---:|---:|---:|---:|
| dense 2K | 19,667,328 | 30.08 min | **1.42966** | **1.42891** | 기준 |
| dense 8K | 22,026,624 | 24.52 min | 1.51963 | 1.51869 | +0.08978 |
| low-rank 8K | 19,669,888 | 23.44 min | 1.54196 | 1.54139 | +0.11247 |
| generic code 8K | 19,667,328 | 48.13 min | 1.62610 | 1.62564 | +0.19673 |
| **Hangul code 8K** | **19,667,328** | **49.61 min** | **1.63792** | **1.63745** | **+0.20854** |
| shuffled-Hangul code 8K | 19,667,328 | 49.27 min | 1.64965 | 1.64894 | +0.22003 |

사전 고정한 다섯 contrast는 다음과 같다. 모든 차이는 왼쪽 역할 minus 오른쪽 역할 BPB다.

| contrast | contiguous | document | bootstrap 95% upper | 판정 |
|---|---:|---:|---:|---|
| Hangul − dense 2K | +0.20825 | +0.20854 | +0.21157 | baseline noninferiority 실패 |
| generic − dense 2K | +0.19644 | +0.19673 | +0.19965 | generic fallback도 실패 |
| Hangul − generic | +0.01181 | +0.01181 | +0.01264 | generic advantage gate 실패 |
| Hangul − low-rank | +0.09595 | +0.09606 | +0.09828 | low-rank noninferiority 실패 |
| Hangul − shuffled | **−0.01174** | **−0.01149** | **−0.01068** | linguistic alignment gate 통과 |

Summary는 여섯 checkpoint를 strict-load한 뒤 8M contiguous stream과 385개 document의 NLL을
모두 독립 재계산했다. 저장한 모든 float32 배열과 bitwise 동일했고, 이 재계산 배열만으로
위 수치와 decision을 만들었다.

## 실패 원인의 분해

### 1. 첫 병목은 8K vocabulary 자체의 one-pass optimization이다

Dense 8K는 같은 body와 2.36M개의 추가 head parameter를 가졌는데도 dense 2K보다 document
BPB가 `+0.08978` 나빴다. 이전 parameter-matched BPE frontier의 8K 차이 `+0.09441`과 거의
같다. 즉 Transformer core를 조금 넓히거나 dense head capacity를 더 주는 것만으로는 격차가
닫히지 않았다.

같은 128M raw bytes에서 2K는 2,213 updates, 8K는 1,675 updates를 받는다. 8K는 output class가
4배이고 긴 저빈도 token의 관측도 희소하다. 현재 실험은 tokenizer의 순수 인과 효과가 아니라
deployable one-pass system을 비교했으므로 이 차이를 허용했지만, 다음 구조를 고르기 전에
large-vocabulary convergence가 회복 가능한지를 따로 물어야 한다.

### 2. Low-rank tax는 작지만 무시할 수 없다

Dense 8K에서 rank-92 tied factorization으로 바꾸면 document BPB가 추가로 `+0.02269`
나빠졌다. 이는 pure codebook보다 훨씬 작으며, token-specific degree of freedom을 보존하는
저차원 factorization이 적절한 표준 control이라는 뜻이다. 그러나 low-rank도 dense 2K보다
`+0.11247` 뒤져서 현재 data regime에서는 단독 후보가 아니다.

### 3. Pure codebook constraint가 가장 큰 추가 손실을 만들었다

Generic codebook은 dense 8K보다 document BPB가 `+0.10695`, Hangul codebook은 `+0.11876`
나빴다. Collision-free 16-slot tuple이어도 각 token row가 2,048 shared code vector의 additive
sum으로 제한되어 강한 간섭이 생긴다. Parameter 수가 같다는 사실은 표현 자유도가 같다는
뜻이 아니다.

더구나 codebook training은 low-rank보다 약 2.1배 오래 걸렸다. Random-weight generation에서는
sequence shortening이 gather-add overhead를 상쇄했지만, 학습 효율과 품질에서는 열세였다.
따라서 slot 수·hash·shuffle seed를 다시 조정하는 것은 근본 병목을 피하지 못한다.

### 4. Hangul signal은 존재하지만 구조를 구제하지 못했다

True Hangul assignment가 shuffled control을 안정적으로 이긴 것은 자모 정렬이 공유 구조에
정보를 준다는 증거다. 그러나 generic surface hash보다 나빴다는 사실은 현재 assignment가
token identity를 표현하는 generic diversity를 희생했음을 시사한다. 이를 근거로 low-rank에
작은 Hangul residual을 더하는 실험을 즉시 열 수는 없다. SCRIPT가 이미 original subword
embedding에 Jamo-derived representation을 residual로 주입하므로, 단순 residual 구조는 신규성도
약하다.

## Fable 5 검토에 대한 사후 판정

`fable5-연구-중간-검토.md`의 가장 중요한 지적은 계속 유효하다. 분석적 patch/step 감소를 실제
효율로 간주하면 안 되고, actual wall-clock이 핵심 성공 기준이어야 한다. W72는 이후 실제
generation gate를 실패했고, 이번 8K codebook도 random-weight speed만 양성이었지만 trained
quality가 실패했다. 두 결과 모두 `docs/89`에서 채택한 보수적 해석이 맞았음을 보여 준다.

반대로 speed가 실패해도 작은 quality/method paper로 종료할 수 있다는 제안은 이 프로젝트의
성공 기준으로 채택하지 않는다. 음성 결과는 논문에 공개할 가치가 있지만, 실제 추론 효율이
개선된 모델을 만들었다는 최종 목표를 대신하지 않는다.

## 최신 선행연구가 바꾸는 다음 질문

[SCRIPT](https://aclanthology.org/2026.findings-acl.104/)는 기존 subword input embedding에
Jamo compositional representation을 더하는 model-agnostic module을 이미 제안했다. 따라서
`low-rank + Jamo residual` 자체는 독립 신규성으로 주장할 수 없다.

더 최근의 [Beyond Initialization Loss](https://arxiv.org/abs/2608.03494)는 언어별 vocabulary
확장에서 constituent-subword composition과 input/output 비대칭 초기화가 강한 baseline이고,
초기 loss보다 짧은 continued-pretraining probe가 수렴 전략을 더 잘 고른다고 보고한다. 이
결과는 현재 실패를 곧바로 더 복잡한 head로 덮기보다, trained 2K body와 lexical knowledge를
8K tokenizer로 옮길 수 있는지 먼저 검증해야 한다는 직접 근거다.

이는 두 논문을 합친 것만으로 신규성을 주장한다는 뜻이 아니다. 다음 단계의 잠정적 가치 있는
질문은 더 좁다.

> Korean BPE vocabulary expansion에서 generic subword-composition transfer가 짧은 8K sequence의
> 품질을 회복하는가? 회복한다면 Jamo-aware transfer가 compute-identical generic initialization보다
> 필요한 continued-pretraining steps를 더 줄이고, 최종 matched-quality batch-1 generation을
> 10% 이상 빠르게 만드는가?

## 수정된 실행 순서

### 단계 T0 — generic vocabulary-transfer probe

기존 dense 2K checkpoint와 8K tokenizer를 사용한다. Transformer body를 exact copy하고, 8K
token의 BPE merge tree를 exact 2K vocabulary frontier에서 잘라 새 embedding/output row를
초기화한다. 임의 minimum byte-cover는 source tokenizer가 학습한 merge genealogy와 다를 수
있으므로 canonical merge frontier를 사용한다. 2026년 선행연구가
untied input/output의 비대칭 초기화를 가장 강한 방법으로 보고했으므로, tied-only 실패를 generic
transfer 실패로 오독하지 않게 두 architecture를 함께 시험한다. 사전 고정할 control은 다음과 같다.

1. tied/untied 각각의 shared-token copy + new-token random/norm-matched control
2. tied constituent uniform/byte-length-weighted composition + symmetric norm calibration
3. tied 마지막 constituent piece copy
4. untied uniform-input + uniform-output, input-only norm calibration
5. untied uniform-input + byte-length-weighted-output, input-only norm calibration

동일한 짧은 continued-pretraining checkpoint grid에서 calibration BPB curve를 비교한다. Probe는
전략 선택용 개발 evidence이며 final claim이 아니다. 초기 BPB 한 점으로 고르지 않고, 동일한
update budget 뒤의 raw-byte BPB와 감소율을 사용한다.

다음 중 하나라도 만족하지 못하면 large-vocabulary transfer 분기를 종료한다.

- 각 architecture 안에서 best composed initializer가 대응 random-new-token control보다 명확히
  낮은 BPB
- fixed maximum CPT budget 안에서 dense 2K anchor의 `+0.010 BPB` 근처까지 실제로 접근하거나,
  보수적인 후속 full-run gate를 정당화할 만큼 격차를 크게 축소
- 8K full-vocabulary head의 random-weight systems 이득을 없애는 별도 runtime 구조를 요구하지 않음

### 단계 T1 — Korean contribution

T0가 통과할 때만 generic best initializer, SCRIPT-like Jamo residual/initializer, distribution-matched
shuffled Jamo control을 같은 graph와 CPT budget에서 비교한다. Korean 역할은 generic보다 품질
회복이 빠르고 최종 BPB가 낮아야 한다. SCRIPT와 구조가 같다면 신규 method로 포장하지 않고
직접 reproduction/baseline으로 명명한다.

Full CPT는 transfer candidate에만 추가 학습을 주지 않는다. Dense-2K continuation, 기존 direct-8K
continuation, 선택된 transfer와 대응 random-row transfer가 동일한 두 번째 raw-byte budget을 받아야
한다. 그래야 최종 품질 회복을 vocabulary curriculum/initialization과 단순 추가 compute로 분리할 수
있다.

### 단계 T2 — actual inference와 confirmation

품질을 통과한 exact trained checkpoint만 dense 2K matched-quality baseline과 controlled/free-running
batch-1 E2E로 비교한다. 10% speed gate를 통과해야만 multi-seed, 더 큰 Mac-feasible model,
새 sealed final split과 Hugging Face 공개로 확장한다. Training compute·추가 parameters·tokenizer
시간·break-even inference volume을 함께 공개한다.

## 실행 중 환경 사건

첫 dense-8K worker는 학습과 evaluation을 모두 끝낸 뒤 end-state environment eligibility check가
한 번 실패했다. Checkpoint/NLL/report는 publish되지 않았고 repository HEAD도 변하지 않았다.
같은 sealed plan을 다시 실행했을 때 모든 역할이 정상 완료됐으며, 별도 5초 간격 monitor는
3시간 26분 동안 anomaly 0건이었다. `pmset` log에도 해당 구간의 AC/sleep/thermal event가
없었다. 따라서 원인을 특정할 수 없는 transient environment-read failure로 기록하며 model
결과나 selection에는 영향을 주지 않는다. 후속 장시간 protocol은 failure 시 raw command output의
aggregate receipt를 남긴다.

## Artifacts

- protocol: `docs/134-compositional-head-quality-one-seed-protocol.md`
- plan: `data/manifests/compositional-head-quality-one-seed-v1.json`
- tracked result: `results/compositional-head-quality-one-seed-v1/summary.json`
- ignored checkpoint/NLL/worker evidence: `artifacts/compositional-head-quality-one-seed-v1/`
- result file SHA-256: `c7be388773d5088faf5d6b8920bf56e631d9c171571902a006faccd56be4b0e9`
- result payload SHA-256: `ce5c3577aceb16ea6fa171b7d1a3dc04f01bbe0a09914007e536be06e53fb381`
