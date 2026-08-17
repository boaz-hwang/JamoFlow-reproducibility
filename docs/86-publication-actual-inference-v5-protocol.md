# Publication actual-inference v5r3 protocol

> 작성일: 2026-08-12
>
> 상태: **속도 결과를 보기 전에 v5/v5r1 correctness와 v5r2 device preflight failure를 봉인하고 v5r3 실행 고정**
>
> 결과: **아직 모름. 이 문서는 속도 향상 결과를 주장하지 않는다.**

## 1. 연구 판단 기준

JamoFlow의 핵심 논문 가치는 새 final test에서 품질을 유지한 정확한 모델 pair가
실제 batch-1 incremental generation에서 더 빠른지로 판정한다. Analytical FLOPs,
patch count, tokenizer 크기 또는 microbenchmark만 좋아서는 positive result가 아니다.

Primary pair는 final-quality lock이 물리 checkpoint bundle 수준에서 허가한 다음 두
역할이다.

- candidate: calibration-only selection이 고정한 Korean whitespace-informed W64 또는
  W72
- matched-efficiency reference: Phase 3 causal codepoint C86

Candidate−C86의 새 sealed-final BPB noninferiority만 이 pair의 timing을 허가한다.
Candidate−same-rate-C 비교는 whitespace structure가 이득의 원인인지 묻는 mechanism
estimand이며, 그 gate가 실패해도 이미 품질이 맞은 candidate−C86 timing을 막지 않는다.
대신 mechanism attribution 주장은 허가하지 않는다. Strongest broad reference도 그
pair의 별도 final noninferiority가 통과할 때만 timing할 수 있다.

이는 comparator를 결과 뒤 약화한 것이 아니다. `docs/80`에서 분리한 세 estimand 중
사용자가 정한 “matched quality에서 실제 추론 효율이 좋아졌는가”를 primary로 삼고,
mechanism 및 broad replacement를 서로 다른 claim으로 봉인한 것이다.

## 2. 시간적 주장 범위

v5의 outcome-sensitive timing, case-selection, 통계 protocol은 새 final-quality loss를
계산하기 전에 Git에 고정한다.
Final-quality lock이 나중에 exact pair를 허가하면, outcome을 읽지 않는 deterministic
case selector가 plan을 물리 bundle과 stream에 인스턴스화한다. Plan sealer는 아래
correctness revision에 명시된 파일을 제외한 모든 implementation file의 마지막 변경
commit이 final-quality session의 `evaluator_git_commit`의 조상 또는 동일 commit인지
검사한다. 이 commit은 첫 final loss 전에 고정된 evaluator identity이므로 quality
결과를 본 뒤 timing 규칙이나 case algorithm을 바꾼 실행은 거부한다.
Final-quality lock을 검증하는 과정에서는 Python tuple이 JSON 저장 뒤 list가 되는
직렬화 차이 때문에, 수치와 canonical SHA-256은 같아도 in-memory 구조 비교가 실패하는
오류가 timing 시작 전에 발견됐다. `InferenceQualityNoninferiority.to_dict()`가 두 tuple을
명시적으로 list로 내보내도록 고쳤으며, 품질 값·role set·quality lock SHA-256·case
selection·timing protocol·효율 gate는 바뀌지 않았다. 예외 파일의 evaluator-era blob,
현재 blob, 전체 diff SHA-256은
`data/manifests/phase3-inference-actual-v5-serialization-erratum.json`에 고정했다.

그 뒤 첫 v5 session은 timing/output/receipt를 공개하기 전에 free-path correctness gate에서
중단됐다. Seed 2718 C86의 한 위치에서 parallel과 sequential logits의 최대 차이는
`2.86102294921875e-06`, 사전 허용오차 대비 `0.059353...`였으나 top-2 margin이
`9.5367431640625e-07` 이하라 argmax 순서가 뒤집혔다. 이는 cache 의미 불일치가 아니라
사전 allclose 안에서 순위가 결정되지 않는 numerical tie다. 실패 plan과 session receipt는
`results/phase3-inference-actual-v5/failures/session-01.json`으로 봉인했고 latency metric은
열지 않았다.

v5r1은 기존 allclose를 그대로 유지했다. Argmax가 다르면 두 선택의 tolerance interval이
겹치는 경우만 tie-ambiguous comparison으로 별도 계수하며, exact+tie 수가 전체 위치 수와
정확히 같아야 한다. Interval이 겹치지 않거나 allclose를 벗어나면 계속 hard failure다.
Timed parallel path가 생성한 masked greedy byte와 저장 output은 여전히 exact match해야 한다.
Model pair, case selection, workload, repetition, 통계, 10% efficiency gate는 전혀 바꾸지
않았다. 이 변경의 전체 evaluator-era/current blob 및 diff는
`data/manifests/phase3-inference-actual-v5r1-correctness-revision.json`에 고정한다. 나머지
implementation file은 계속 final evaluator commit보다 늦은 변경을 거부한다.

AC 전원 복구 후 v5r1은 timing loop 이전의 controlled correctness에서 다시 중단됐다.
Seed 65537 C86 case 66의 한 저확률 logit이 기존 tolerance를 5.1% 넘었지만 argmax는
같았고, 해당 row의 probability total variation은 `1.18e-7`이었다. 같은 checkpoint/case의
CPU 재생과 이어진 10개 bundle CPU 전수 감사는 원래 `2e-5/2e-5` 계약을 모두 통과하고
모든 argmax가 같았다. Timing/output/session receipt는 한 번도 공개되지 않았다.

이 증거에 따라 v5r2는 tolerance 하나를 단순히 넓히지 않고 두 gate의 교집합을 사용한다.

- CPU semantic oracle: 매 session의 5 seeds × 2 roles × 72 controlled cases에서 원래
  `atol=2e-5, rtol=2e-5`를 그대로 통과
- MPS backend gate: `atol=1e-4, rtol=2e-5`, 모든 row의 softmax probability total
  variation `<=1e-5`, exact 또는 interval-overlap argmax, exact timed greedy byte
- CPU/MPS boundary trace exact equality
- 기존 MPS tolerance ratio와 초과 원소 수는 diagnostic으로 모두 보존
- 이 계약이 실패하면 세 번째 tolerance 완화 없이 actual-efficiency gate 실패

Bounds는 correctness-only 전체 MPS audit 전에 commit `ca34a31`에 고정했다. 이후
10/10 bundle이 통과했고 유일한 기존 tolerance 초과는 seed 65537 C86의 한 원소였다.
전체 safety ratio는 0.213 이하, TV는 `4.57e-6` 이하였다. 상세 실패와 전수 감사는
`docs/87-actual-v5-free-path-correctness-debug.md` 및
`data/manifests/phase3-inference-actual-v5r2-backend-correctness-revision.json`에 고정한다.

V5r2 plan 뒤 한 case dry run은 MPS model parameter device가 `mps:0`로 정규화되는 반면
새 contract guard가 literal `mps`만 허용해 비교 전에 중단됐다. Latency trial과
timing/output/receipt는 생성되지 않았다. V5r3은 device family 판정을
`startswith("mps")`로 고치고 `mps:0`에서도 실제 `torch.mps.synchronize()`를 호출하는
실행-only erratum이다. V5r2의 CPU/MPS correctness contract, model pair, cases, workload,
statistics, 10% gate는 그대로다. 실패 plan/receipt와 exact cumulative diff는
`data/manifests/phase3-inference-actual-v5r3-device-identity-erratum.json`에 고정한다.

따라서 정확한 표현은 다음과 같다.

> pre-final outcome-sensitive timing, case-selection, and statistical logic with
> a tracked post-final pre-timing backend-aware correctness revision intersecting
> the original CPU semantic oracle with MPS distribution and greedy invariants;
> timing workload, pair, cases, statistics, and the efficiency gate are unchanged

다음 표현은 사용하지 않는다.

- untouched one-shot confirmatory experiment
- cryptographic one-shot evidence

Case selector는 sealed final stream의 문서 경계, UTF-8 scalar boundary, Hangul 비율과
content hash를 사용하지만 NLL, model output 또는 과거 latency는 읽지 않는다. Plan
sealer 자체는 committed NLL과 quality lock을 재검증하고 허가된 pair를 읽으므로
outcome-gated이지 outcome-free가 아니다. Case set의 구체적 hash는 final-quality 뒤에
생기므로 이 두 단계의 chronology는 논문에 분리해 공개한다. 이는 public registry가
아니라 local Git ancestry evidence다.

## 3. 고정 workload와 모델 실행 의미

각 session은 동일한 Apple MPS 환경에서 정확히 다섯 model seed를 모두 측정한다.

- 독립 fresh-process session: 5개
- warmup case: 8개
- measured case: 서로 다른 문서의 64개 Hangul-heavy case
- prompt: 128 raw bytes
- controlled replay: 고정 continuation 128 bytes
- free running: strict UTF-8 greedy, 최소 128 bytes, 최초 scalar boundary에서 정지,
  최대 131 bytes
- cell당 timing repetition: 5회; 독립 표본으로 세지 않고 median으로 먼저 축약
- mode: controlled replay와 free-running UTF-8 greedy 둘 다 co-primary
- batch size: 1

Session별 seed 순서와 candidate/reference 선행 순서는 서로 다른 고정 seed로
균형화한다. 각 session은 다섯 seed, 두 mode, 64 prompt, 두 role 전체를 수행한다.

모델의 두 길이 개념을 혼동하지 않는다.

- Transformer global/rotary position capacity: 1,032
- 학습된 Phase 3 patch schedule horizon: 512

Prompt와 생성 전체를 수용하려고 model position capacity는 1,032를 사용하지만,
C/W structural schedule과 entropy boundary oracle은 학습 때와 같은 512 horizon을
사용한다. 1,032를 patching horizon으로 쓰면 C/W 간격이 길어져 다른 모델 runtime을
측정하는 것이므로 hard failure다.

## 4. end-to-end timer 경계

Timer 안에는 실제 배포 경로에서 발생하는 다음 연산을 모두 포함한다.

- parallel prefill
- structural selector 또는 entropy router
- local encoder/decoder와 global Transformer forward
- KV/cache append와 boundary state update
- strict UTF-8 mask 적용
- greedy argmax
- tensor에서 host scalar로 읽는 implicit device-host synchronization
- UTF-8 DFA transition과 stop check
- 시작·종료의 명시적 device synchronization

UTF-8 mask table의 사전 compile만 trial timer 밖이다. Report는 “포함했다”는 boolean을
신뢰하지 않고 selector bytes, router calls/scored bytes, main consume, argmax, mask,
DFA, stop, device-host readback, explicit synchronization counter를 trial별 배열로
남긴다. Summary가 mode·role·생성 길이에서 기대되는 exact identity를 다시 계산한다.

TTFT와 decode는 secondary breakdown이고, primary statistic은 전체 end-to-end다.

## 5. 실제 checkpoint equivalence와 출력 증거

모든 session의 모든 seed×role에서 controlled equivalence는 timing 전에 검증한다.
Free-path equivalence는 timed free bytes가 생긴 직후 같은 session에서 검증하며,
둘 다 summary가 결과를 수용하기 전에 완료돼야 한다.

1. report/checkpoint artifact hash, loaded state hash와 parameter count를 post-final
   authorization의 exact model bundle과 비교한다.
2. entropy model이면 router checkpoint/state/config, threshold, candidate-mask,
   max-patch를 비교한다.
3. 8 warmup+64 measured controlled case의 255 observed byte 전 위치에서 full causal
   main logits와 sequential incremental logits를 비교한다.
4. 같은 case의 parallel prefill+127 consume logits를 sequential path와 비교한다.
5. entropy model이면 full no-cache router와 incremental cached router의 logits,
   entropy와 resulting boundary trace를 전 위치 비교한다.
6. free-running의 실제 생성 bytes에 대해 full/sequential/parallel 경로를 다시
   replay하고 strict mask 뒤 greedy argmax byte, DFA 상태, stop 결정과 cache를 확인한다.

CPU semantic 비교는 원래 `atol=2e-5`, `rtol=2e-5`를 사용한다. MPS 비교는
`atol=1e-4`, `rtol=2e-5`를 쓰되 모든 row의 softmax probability total variation을
`1e-5` 이하로 제한한다. Evidence는 active backend envelope와 원래 nominal envelope의
normalized worst-error, nominal 초과 원소 수, 최대 TV를 모두 기록한다.

```text
max_i |actual_i - reference_i| /
      (atol + rtol * |reference_i|)
```

Summary는 active-envelope 값이 finite이고 `<=1`, TV가 `<=1e-5`인지 직접 확인한다.
CPU receipt는 nominal ratio도 `<=1`이고 nominal 초과 수가 0이어야 한다. Full causal,
parallel, router의 exact+tie argmax count가 position count와 같아야 하며, 같은
seed×role의 CPU/MPS controlled boundary trace와 다섯 session의 controlled/free trace가
모두 같아야 한다. 따라서 `pass=true`와 임의의 큰 오차를 함께 넣은 receipt는 거부된다.

Free-running의 모든 repetition raw bytes와 길이는 ignored binary artifact로 보존한다.
Tracked session receipt가 그 artifact hash를 묶고, summary는 모든 bytes를 strict DFA로
다시 읽는다. UTF-8로 decode되는 것만으로는 충분하지 않으며 masked greedy argmax와
최초 허용 boundary stop까지 같아야 한다.

## 6. 통계 단위와 사전 고정 gate

각 seed×prompt×session cell에서 다섯 repetition의 median을 먼저 구한다. Repetition을
독립 표본처럼 bootstrap하지 않는다. Candidate/reference는 같은 session, seed,
prompt index로 paired하며, session×model-seed×prompt의 세 축을 독립 resample하는
10,000회 crossed bootstrap을 사용한다.

Controlled와 free-running end-to-end 각각 다음 조건을 모두 만족해야 한다.

- 전체 cell median 기준 latency reduction `>= 10%`
- 95% percentile bootstrap lower bound `> 0`
- 5/5 session의 point reduction `> 0`
- 최소 3/5 session의 point reduction `>= 10%`
- 최소 4/5 model seed의 point reduction `> 0`
- model-seed reduction의 median `>= 10%`

최종 positive actual-efficiency gate는 final matched-quality authorization과 두
co-primary mode의 교집합이다. 이 설계가 주장하는 것은 “관측 median 개선이 10%
이상이고 95% interval이 0을 제외한다”이다. 개선의 confidence lower bound가 10%
이상이라고 주장하지 않는다.

Session 수가 5로 제한되고 hardware가 한 대이므로 결과가 통과해도 seal에 고정한 exact
Mac model, Apple chip, RAM, OS build, Python/package 환경과 workload를 넘어 일반
hardware 또는 모든 LLM에 일반화하지 않는다. Session별, seed별 효과와
candidate-first/reference-first 효과, within-cell MAD/IQR를 함께 공개한다.

## 7. 메모리와 auxiliary cost

Entropy policy가 선택되면 router parameter/state와 실제 timed router calls를 candidate
총비용에 포함한다. Plan은 role별 main, auxiliary, total parameter count와 float32
parameter bytes를 봉인한다.

메모리는 timing process와 분리한 fresh process에서 role×seed 10개 unit을 측정한다.
MPS를 먼저 초기화·동기화한 뒤 baseline을 잡고, 64 prompt free generation 뒤의
MPS current/driver와 macOS process high-water 증가량을 기록한다. 현재 backend는
resettable native peak를 제공하지 않으므로 메모리는 descriptive evidence일 뿐
publication pass gate가 아니며 “메모리 개선”을 주장하지 않는다.

## 8. provenance와 fail-closed 실행

Plan은 final authorization, independently replay-verified final-quality lock, case
artifact, 전체 `src/jamoflow/*.py`, v5r3 scripts, `pyproject.toml`, Python/platform와 주요
package version을 exact hash로 묶는다. 과거 final authorization의 implementation
목록을 v5 파일 때문에 소급 변경하지 않는다.

각 timing session과 memory unit은 한 fresh process만 machine-global MPS lock을 잡고
실행한다. `ps` inventory가 실패하거나 현재 PID를 parse하지 못하거나 목록에 고정한
다른 JamoFlow neural/MPS entrypoint가 있거나 thermal/power 조건이 맞지 않으면 fail
closed한다. 임의의 제3자 GPU 프로그램을 완전 탐지한다고 주장하지 않는다. Heavy
timing/output arrays는 ignored 영역에 두되 각 session report와 memory receipt는 tracked
canonical path에 둔다.

정상 실행 순서는 다음과 같다.

1. v5r3 code, v5/v5r1/v5r2 실패 receipt와 두 revision, 이 protocol을 commit한다.
2. final-quality lock이 primary pair를 허가한 뒤 plan/case를 seal하고 commit한다.
3. session 하나를 fresh process로 실행한다.
4. tracked session receipt를 별도 commit한 뒤에만 다음 session을 연다.
5. 다섯 timing session 후 memory unit도 하나씩 실행·commit한다.
6. summary는 모든 receipt가 exact HEAD blob이고 각 canonical path가 Git history에서
   한 번만 추가됐는지, plan→measurement→receipt→summary ancestry가 맞는지 확인한다.
7. raw output, counters, model state, environment, correctness와 통계를 재구성한 뒤에만
   summary를 쓴다. 다음 invocation 전 이를 정확히 한 번 commit해야 하며, committed
   summary는 exact HEAD blob·canonical hash·single-touch ancestry로 verify-only다.
   삭제 이력이 있는데 파일이 없으면 재봉인을 거부한다.

Repo-local 및 machine-global file lock은 동시 실행과 accidental resume laundering을
막는다. 그러나 로컬 파일 삭제 권한을 가진 연구자의 선택적 미보고까지 암호학적으로
방지하지는 못한다. 따라서 논문은 “one prospectively Git-sealed analytic evaluation
plus deterministic verification replay”라고 기술하고 public preregistration 또는
cryptographic one-shot이라고 부르지 않는다.

## 9. 이 시점의 판정

Final-quality lock은 W72가 C86 대비 +0.003682 BPB로 noninferiority를 통과했고, W72가
same-rate C72보다 −0.010781 BPB 좋아 primary와 mechanism timing을 모두 허가했다. 다만
v5와 v5r1 timing은 correctness gate에서 결과 공개 없이 중단됐고 v5r2는 device dry run에서
timing 전에 중단됐다. V5r3 timing 결과는 아직 없으므로
연구 성공은 미정이다. Test 실행과 독립 감사 상태는 protocol 불변값이 아니므로 이
문서에 결과 수치로 봉인하지 않고 해당 milestone commit의 검증 기록으로 남긴다.

다음 의사결정은 결과에 종속된다.

- matched quality와 actual v5r3 speed gate를 모두 통과: publication-scale 확장,
  BPE16K/32K 및 broad reference 비교, 논문 핵심 기여로 승격
- 품질 통과·속도 실패: 현재 architecture를 positive efficiency result로 발표하지
  않고 bottleneck profile을 바탕으로 다음 구조를 새 protocol에서 연구
- 품질 실패: W64/72의 matched-quality claim 폐기; test를 본 뒤 margin/rate/comparator
  변경 금지

즉, 문서와 코드가 만드는 것은 성공 주장이 아니라 성공과 실패를 같은 규칙으로
판정할 수 있는 연구 장치다.
