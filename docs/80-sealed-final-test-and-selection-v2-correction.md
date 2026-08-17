# Sealed final test and selection-v2 correction

> 작성일: 2026-08-12
> 상태: **새 final test의 loss를 계산하기 전 protocol/manifest 고정**
> 영향: 기존 Phase 3 `test`를 development screening으로 강등하고, 최종 품질 주장을 위한 새 disjoint test와 calibration-only selection-v2를 도입함

## 1. 결론부터

기존 HPLT3 `test` 16MB는 이미 초기 세 seed의 gate, policy 비교와 후속 실행
결정에 여러 번 사용됐다. 계산식이 calibration-only인 comparator selector를 갖고
있더라도, 사람이 test 결과를 본 뒤 selector를 실행할 수 있었고 selection input인
calibration BPB도 checkpoint에서 독립 재구성되지 않았다. 따라서 이 split은 더
이상 publication의 final held-out test가 아니다.

기존 결과는 삭제하지 않는다. 다음처럼 역할을 명시적으로 낮춘다.

- historical HPLT3 `test`: development/screening evidence
- 현재 진행 중인 F/C/W seed 57,721·65,537와 Leipzig OOD 확인: mechanism
  replication/screening evidence
- 새 `hplt3-korean-final-test-v1`: selection lock과 physical-model authorization 이후
  한 번의 공식 Git-sealed evaluation session에서 여는 final quality evidence

이 교정은 현재 확인 학습 결과나 새 final-test loss를 보지 않은 상태에서 고정한다.
기존 test가 이미 노출됐다는 사실은 소급해서 되돌릴 수 없으므로 새 test를 만드는
것만이 시간적 봉인을 회복하는 방법이다.

## 2. 새 final test의 정확한 정의

소스는 이미 SHA-256로 고정된 HPLT3 Korean shard
`10_1.jsonl.zst`다. 새 builder는 다음 순서를 바꾸지 않는다.

1. source file의 1,862,302,013 bytes와 SHA-256을 확인한다.
2. historical processed JSONL의 SHA-256, 6,911 unique documents, stable split별
   count와 기존 bottom-hash commitment를 재검증한다.
3. raw shard 전체를 다시 scan하여 그 6,911개가 모두 원천에 존재하고 기존
   train/calibration/test bottom-hash 표본이 정확히 재구성되는지 확인한다.
4. exact UTF-8 SHA-256가 historical 6,911개 중 하나인 문서는 split과 무관하게
   모두 제외한다.
5. NFKC, casefold, Unicode whitespace collapse 뒤 SHA-256가 historical 문서와
   같은 format-normalized duplicate도 제외한다. 실제 연구 환경인 Python
   3.13.11의 Unicode database 15.1.0으로 고정하고 다른 UCD에서는 실행을 거부한다.
6. 남은 문서 중 stable hash split이 `test`인 문서만 허용한다. train 또는
   calibration fallback은 없다.
7. 사람이 고를 수 있는 salt 대신 source SHA, historical-output SHA, 32MB quota와
   protocol version에서 rank key를 유일하게 도출한다.
8. 남은 candidate끼리 같은 format-normalized digest를 가지면 derived rank가 가장
   낮은 한 문서만 남긴다.
9. `(derived rank, document digest)` 오름차순 full-document prefix가 처음으로
   quota를 넘는 지점까지 선택한다.
10. 선택 문서를 LF 한 byte로 연결한 stream의 앞 32,000,000 bytes만 평가한다.
   이는 512-byte window 62,500개이며 tail drop은 0이다.

Rank key와 rank는 domain-separated SHA-256으로 계산한다.

```text
K = SHA256(
  "JamoFlow/final-test-key/v1\0"
  || source_sha256
  || historical_output_sha256
  || u64be(32_000_000)
  || u64be(1)
)

rank(d) = SHA256("JamoFlow/final-test-rank/v1\0" || K || d)
```

Tracked seal에는 source/manifest hash, aggregate scan counts, exclusion-set commitment,
selected-set commitment, ordered-selection commitment, zero-intersection audit,
ignored JSONL hash와 exact 32MB evaluation-stream hash만 남긴다. 원문, 개별 record
ID/digest/rank, model metric은 tracked artifact에 쓰지 않는다. Output JSONL은
ignored local data에 남고 verifier가 pinned raw shard부터 완전히 재계산한다.

Builder는 model, checkpoint, loss, BPB, latency 또는 result artifact를 입력으로
받지 않는다. 고정 경로만 사용하고 clean Git commit에서 시작하며, output과 seal을
덮어쓰지 않는다. Seal의 `preparation_git_commit`은 사용한 정확한 코드와 manifest를
가리킨다. Seal 생성 commit이 model-selection lock보다 먼저 존재해야 final
evaluation runner가 실행될 수 있다.

## 3. 증명하는 것과 증명하지 않는 것

이 protocol이 증명하는 범위는 다음과 같다.

- pinned raw shard와 historical processed sample을 입력으로 선택을 재현 가능
- historical 6,911 documents와 새 final documents 사이 exact-byte overlap 0
- 위 고정 정규화에서 format-normalized exact overlap 0
- 새 문서가 모두 기존 stable `test` bucket에 속함
- 평가 stream의 순서와 정확한 32MB bytes가 고정됨
- 선택 과정에 model metric을 받는 코드 경로가 없음

이는 암호학적 zero-knowledge proof가 아니다. 제3자는 pinned raw를 받아 동일
commitment를 재계산해야 한다. NFKC/casefold/whitespace가 같은 문서는 추가로
차단하지만, 부분 복제, approximate near duplicate 또는 semantic contamination
부재는 증명하지 않는다. 그런 주장은 별도 사전 고정 audit 없이 하지 않는다.
32MB byte stream의 마지막은 UTF-8 codepoint나 문서 중간일 수 있으며 이는
raw-byte LM 평가 정의와 일치한다.

## 4. Selection-v2: 새 sealed final test를 열기 전에 하나의 결정을 고정

기존 selector의 `select_reference()` 계산은 calibration-only였지만 절차와
evidence가 충분하지 않았다.

- selector가 test 값이 포함된 summary를 읽은 뒤 selection JSON을 만들었다.
- conversion calibration BPB가 per-sequence NLL이 아니라 report scalar였다.
- final consumer가 locked calibration evidence에서 rate/reference를 canonical하게
  다시 계산하지 않았다.
- 임의 output path로 대체 selection artifact를 만들 여지가 있었다.

따라서 rate와 reference 선택을 하나의 canonical selection-v2 builder로 합친다.
고정된 plan은 exact initial seed/policy/rate 순서, calibration stream과 model
identity, tie rule, 새 final-test seal SHA를 포함한다. Builder는 각 checkpoint에서
calibration per-sequence NLL을 deterministic하게 다시 계산하고 다음을 receipt에
봉인한다.

- checkpoint artifact/state와 model/config hash
- calibration stream/input hash와 sequence count
- policy별 patch-matrix hash
- E/EC이면 seed별 router checkpoint/state, threshold, max-patch와 cache lineage
- float32 per-sequence calibration NLL artifact hash와 재구성 BPB

Rate는 기존 규칙대로 64를 먼저 검사하고 실패할 때만 72를 검사한다. Reference는
`F/C/W/S/E/EC/selected-rate C`의 initial 3-seed mean calibration BPB 최저로 고정
순서의 exact tie만 해소한다. Rate와 reference를 별도 API로 고르지 않는다. 둘 다
실패하면 terminal `no-rate`이며 다른 후보나 margin으로 바꾸지 않는다. Historical
screening test scalar/NLL, 새 final test와 latency 값은 decision 함수와 calibration
replay의 선택 입력 schema에 존재할 수 없다. 다만 물리 모델 provenance를 봉인하는
identity projection은 metric이 함께 들어 있는 historical report의 비-metric 필드와
그 전체 artifact hash를 읽는다. 이는 artifact를 읽지 않았다는 주장이 아니라,
그 안의 historical metric을 rate/reference criterion으로 사용하지 않았다는 계약이다.

Strongest raw reference는 broad claim용으로 별도 calibration futility screen을
거친다. Candidate−reference mean calibration BPB가 `+0.010` 이내이고 최소 2/3
seed가 margin 안일 때만 S/E/EC 추가 confirmation과 broad final comparison을
허가한다. 실패해도 C86 대비 within-family efficiency 질문은 막지 않지만,
`best raw-byte model replacement` 주장은 사전에 포기한다. 이 규칙은 strongest
reference를 약화하거나 바꾸는 fallback이 아니다.

`src/jamoflow/inference_selection_v2.py`는 이 rate/reference 결정을 하나의 순수
함수와 canonical 재구성 validator로 구현한다. Strongest reference가 selected-rate
codepoint 모델이면 같은 C/W conversion confirmation에 이미 포함되므로 별도 Phase 3
confirmation을 발급하지 않는다. Broad futility screen을 통과하면서 S/E/EC 중
하나일 때만 정확히 그 policy 하나에 대한 typed reference-confirmation authorization을
추가한다. Futility 실패 시 `phase3_reference=null`이고 candidate/C86/same-rate-C의
narrow path는 계속된다.

Conversion runner는 initial model부터 calibration per-sequence float32 NLL을 test NLL과
별도 artifact로 보존한다. Initial report는 아직 존재하지 않는 selection lock이 아니라
selection plan hash, exact 3×4 seed/policy set, null selection과 clean run commit에
결속된다. Confirmation report만 exact selection-lock hash, selected 2×2 set과
confirmation run commit에 결속된다. Selection lock이 rate/policy를 유일하게
authorization한다. 다만 compute-conversion C/W confirmation runner는 이미 알려진
historical five-seed primary summary의 Gate I/J/OOD를 development-stage progression
조건으로도 확인하고 그 artifact hash를 report에 결속한다. 이 historical gate는
rate/reference를 고르거나 바꾸는 입력은 아니지만 confirmation 실행 가능성에는
영향을 준다. 조건부 S/E/EC runner는 이 historical summary를 직접 authorization
입력으로 사용하지 않는다.
이 lock은 canonical decision을 selection plan, calibration-evidence manifest와 아직
평가하지 않은 final-test seal의 SHA-256에 결속한다.

`seal_inference_selection_plan_v2.py`는 final-test seal이 먼저 commit됐고 conversion
artifact가 아직 없을 때만 고정 plan을 만든다. Initial 학습이 끝난 뒤에는
`seal_inference_initial_model_identity_v2.py`가 exact 3×10 report/checkpoint/state,
source/stream/matrix, E/EC router/cache와 initial conversion binding을 하나의 물리
identity trust root로 봉인한다. 이때 conversion 실행 commit의 exact implementation
blob, selection/confirmation pipeline의 full implementation manifest, Apple MPS
hardware/software 환경을 기록한다. 또한 plan 시점의 rate/reference decision 함수
AST와 핵심 의존 파일을 현재 코드와 비교해 선택 수학이 바뀌지 않았음을 확인한다.

그 identity output을 별도 commit한 뒤에만
`reconstruct_inference_calibration_v2.py`가 initial 3 seeds × exact 10 policies의
checkpoint를 MPS float32로 다시 열어 30개 calibration NLL을 causal forward로
생성한다. Existing ignored receipt/NLL이 있어도 forward를 생략하지 않고 bitwise
float32 일치를 요구한다. Evidence manifest를 별도 commit한 다음
`seal_inference_selection_lock_v2.py`가 같은 30개 checkpoint·router·matrix를 두 번째로
독립 실행하며, 이 두 번째 replay BPB만으로 decision을 만든다. Local NLL 파일은
selection lock의 trust dependency가 아니며 receipt의 committed hash와 두 번째 replay가
직접 비교된다. 이 decision/replay 경로에는 historical test NLL, 새 final-test
text/loss 또는 latency가 입력되지 않는다. Sealer와 downstream authorization은
`identity artifact ≤ first evaluator < calibration-evidence artifact ≤ second verifier < selection-lock artifact`
Git 순서와 두 evaluator commit의 exact implementation blob을 모두 검증한다.

Confirmation에 진입하기 전 이미 봉인된 implementation manifest에는 C/W runner,
조건부 S/E/EC runner, confirmation calibration evaluator와 post-confirmation sealer의
실제 의존 closure가 포함된다. 두 runner와 evaluator는 현재 HEAD blob이 그 manifest와
정확히 같은지 확인한다. Post-selection C/W와 조건부 S/E/EC 학습은 clean start/end,
정확한 checkpoint/report/router/cache hash를 담은 fixed-path training-completion
receipt를 각각 한 번 publish하고, 모든 required receipt를 evaluator에 선행하는 tracked
commit 하나 이상에 고정한 뒤에만 calibration
evaluator가 실행된다. Evaluator와 post-authorization은 receipt와 실제 artifact를
다시 대조하며
`selection-lock artifact commit ≤ run commit < completion-receipt commit ≤ evaluator commit < confirmation-evidence commit`
순서를 검증한다. C86 및 다른 F/C/W confirmation seeds는 이 prospective chain에서
새로 학습한 모델이 아니라 plan에 이미 봉인된 historical five-seed summary의 물리
checkpoint/report/state hash에 직접 결속한다.

## 5. 비교 역할을 분리한다

초기 결과는 S가 W보다 약 0.088 test BPB 좋지만 window당 global patch가 약
153 대 86임을 이미 보였다. 여기서 서로 다른 질문을 하나의 comparator에 넣으면
연구 주장이 모호해진다.

1. **Matched-efficiency baseline C86**: Korean whitespace-informed W64/72가 causal
   codepoint C86의 품질을 유지하면서 실제 latency를 줄이는가?
2. **Strongest raw-byte quality reference**: initial calibration에서 가장 좋은
   S/E/EC/F/C/W/selected-C 중 하나와 비교했을 때 candidate가 어느 Pareto 위치에
   있는가?
3. **Tokenized deployment baselines**: BPE16K/32K와 품질, latency, memory를 함께
   비교할 때 실용적인 이득이 남는가?

기존의 “strongest raw reference에 +0.010 BPB noninferiority가 아니면 모든
efficiency 가치가 없다”는 넓은 주장은 secondary broad gate로 보존한다. 이를
실패하면 “best raw-byte model을 대체한다”는 주장은 폐기한다. 다만 새 final test를
보기 전에 별도로 고정한 W-rate 대 C86 noninferiority와 실제 speed gate를 모두
통과하면, 더 좁고 정확한 **within-family Korean structural efficiency** 주장은
성립할 수 있다. 이는 결과를 본 뒤 comparator를 약화하는 것이 아니라 서로 다른
scientific estimand를 분리하는 교정이다. 초기 S 결과를 이미 보았다는 사실과 이
교정 시점을 논문에 공개한다.

어떤 positive claim도 실제 batch-1 incremental inference 개선 없이는 허용하지
않는다. 최소 조건은 다음과 같다.

- 새 final test에서 W-rate − C86 paired quality upper bound `< +0.010 BPB`
- router/selector, cache, mask, synchronization을 포함한 실제 runtime에서
  prospectively Git-sealed latency improvement gate 통과
- strict-valid free-running output과 checkpoint/cache equivalence 통과
- parameter, auxiliary training/scoring, peak memory를 숨기지 않음
- BPE baselines와 strongest raw reference를 같은 표에 보고

## 6. 이후 fail-closed 실행 DAG

1. 이 protocol, manifest, golden/negative tests를 먼저 commit한다.
2. raw source부터 final test를 생성하고 별도 verifier로 재계산한 aggregate seal을
   commit한다. 이때 model loss는 계산하지 않는다.
3. selection/confirmation hardening implementation을 commit한다.
4. 모든 initial checkpoint·router·source·run binding의
   `initial-model-identity-lock.json`을 만들고 단독 commit한다.
5. 첫 30-unit causal calibration replay를 실행하고 evidence manifest를 단독
   commit한다.
6. 두 번째 독립 30-unit replay로 fixed-path selection-v2 lock을 만들고 단독
   commit한다. `terminal_no_rate`면 여기서 종료한다.
7. selected-rate C/W를 exact typed authorization으로 확인하고, broad-reference
   calibration screen까지 통과한 경우에만 selected S/E/EC의 두 confirmation
   seed를 추가 학습한다.
8. Post-selection C/W run과 조건부 S/E/EC run이 끝날 때 각각 하나의 tracked
   training-completion receipt를 만들고, 모든 required receipt를 evaluator보다 앞선
   tracked commit 하나 이상에 고정한다. 동일 active attempt의 exact
   resume만 허용하며, completion 뒤 재학습이나 deleted receipt 재발행은 거부한다.
   Compute-conversion manifest의 모든 confirmation-stage invocation은 exact authorized
   set이어야 하며 conflict는 0이어야 한다.
9. 모든 required role의 confirmation calibration evidence를 checkpoint forward로
   재구성하고 단독 commit한다. C86/F/C/W historical confirmation seeds는 plan-sealed
   five-seed summary에, prospective C/W와 조건부 S/E/EC는 위 completion receipt에
   각각 물리적으로 결속한다.
10. exact five-seed physical model bundle을 담은
   `post-confirmation-authorization.json`을 만들고 단독 commit한다.
11. 새 final test를 candidate, C86, same-rate C control에 대해 한 번에 평가한다.
   Broad screen이 통과했다면 strongest reference도 같은 one-shot evaluation에
   포함한다. 중간 세 seed를 보고 중단·교체하지 않는다.
12. Candidate−C86 final noninferiority를 통과하면 정확히 그 matched-quality pair의
   primary actual timing에 진입한다. Candidate−same-rate-C mechanism gate는 원인
   귀속 주장을 별도로 허가하며 primary matched-quality timing을 억제하지 않는다.
   Strongest-reference timing은 별도의 broad noninferiority를 통과했을 때만 허용한다.
   이 역할 분리는 `docs/86-publication-actual-inference-v5-protocol.md`가 구체화한다.
13. compact actual speedup이 확인될 때만 publication-scale family feasibility와
   larger-model campaign을 연다.

Historical screening 결과는 provenance와 실패를 포함해 그대로 보존하지만 final
confidence interval이나 p-value에 섞지 않는다. 새 final evaluation 결과를 본 뒤
salt, split, quota, policy, rate, comparator 또는 noninferiority margin을 바꾸면
이 protocol의 positive claim은 무효다.

여기서 `one-shot`은 암호학적 불변성을 뜻하지 않는다. 공식 경로는 하나의
사전 고정 session/unit order, unit별 no-clobber artifact, tracked evidence의 단일
publication commit, 그리고 quality-lock sealer의 별도 deterministic checkpoint
forward replay를 요구한다. Committed evidence나 lock을 삭제한 뒤 재발행하는 Git
history도 거부한다. 그러나 로컬 삭제 권한자가 첫 commit 전에 전체 ignored session을
지우거나 외부에서 결과를 열람한 사실까지 증명하지는 못한다. 논문에서는 이를
“one prospectively Git-sealed analytic evaluation plus one deterministic verification
replay”로만 기술한다.

같은 한계는 confirmation 학습의 completion 이전 구간에도 적용된다. Active marker와
학습 산출물은 첫 tracked completion commit 전까지 로컬 ignored 상태이므로, 이를 모두
삭제한 뒤 버린 실행이 없었다는 사실을 Git만으로 증명할 수 없다. 공식 receipt가 고정한
checkpoint는 calibration evaluator와 post-authorization의 독립 causal replay로 다시
검증하지만, 이를 cryptographic single-attempt training이라고 부르지 않는다. 논문에는
recorded prospective run의 재현 가능한 identity와 이 local-deletion 한계를 함께 공개한다.
