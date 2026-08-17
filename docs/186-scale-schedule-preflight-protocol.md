# Publication-scale W72/C86 schedule sensitivity preflight

> 작성일: 2026-08-16
>
> 상태: **첫 50M/75M/100M schedule timing 전 고정할 protocol**
>
> 상위 결정: [EXAONE retrieval actual 결과와 core-scale 결정](./185-exaone-retrieval-actual-result-and-core-scale-decision.md)

## 1. 목적

Compact 19.6M byte-latent model에서 W72는 matched-quality C86보다 controlled 2.628%,
free-running 2.531% 빨랐다. 효과는 다섯 timing session과 다섯 model seed에서 모두 양수였지만
사전 10% gate를 실패했다. 이후 generic retrieval은 7.8B EXAONE에서 14.938% 느려져
scale-transfer 대안이 되지 못했다.

이 상태에서 50M/75M/100M model을 바로 학습하는 것은 정당하지 않다. 먼저 이미 고정한
publication family geometry에서 global trunk의 상대 비용이 커질 때 W72가 줄이는 patch event가
실제 wall time 10% margin으로 확대되는지 확인한다.

Primary 질문은 하나다.

> 같은 100M random-weight BLT graph, 같은 Korean controlled byte sequence와 같은 Apple MPS
> 환경에서 W72 schedule은 C86 schedule보다 cached incremental end-to-end wall time을 10% 이상
> 줄이는가?

Random weights는 quality evidence가 아니다. 이 실험은 큰 model을 학습할 systems headroom이
있는지 판정하는 fail-fast preflight다.

## 2. 고정 model family

`src/jamoflow/publication_scale.py`의 기존 geometry를 바꾸지 않는다.

| target | parameters | local/global width | global layers |
|---:|---:|---:|---:|
| 50M | 49,823,488 | 256 / 512 | 12 |
| 75M | 76,492,480 | 320 / 640 | 12 |
| 100M | 98,403,360 | 352 / 704 | 13 |

- target order: 50M → 75M → 100M
- model seed: `20260816`
- target×session마다 model object 하나만 만들고 두 schedule이 그 exact object를 공유
- fresh-process session: `session-0`, `session-1`, `session-2`
- global position capacity: 1,032
- dtype/device: repository-pinned float32 / Apple MPS
- parameter count와 deterministic CPU state SHA-256를 plan에서 미리 봉인

Schedule만 다음처럼 다르다.

| role | policy | patch count |
|---|---|---:|
| reference | causal codepoint grid | 86 |
| candidate | causal whitespace grid | 72 |

Schedule별 checkpoint를 따로 만들거나 더 유리한 random seed를 고르지 않는다.

## 3. cases와 timing

새 model-output 기반 case 선택을 하지 않는다. `artifacts/hangul-draft-acceptance-v1/free-target.npz`에
EXAONE actual 결과 전부터 존재한 고정 prompt order를 사용한다. 이 pool 자체는 compact W72 결과 뒤
만들어졌으므로 이 실험을 untouched confirmatory evidence로 부르지 않는다. 초기 초안의 단순 첫 20개에는
measured 두 개의 255-byte observed window가 107 bytes 겹쳤다. 결과를 보기 전 감사에서 이를 발견해,
기존 pool 순서를 유지하면서 **255-byte window 전체가 한 문서 안에 있고 서로 다른 문서인 첫 20개**를
고르는 deterministic filter로 교정했다. 이 subset 교정은 EXAONE 결과 뒤 이루어졌지만 model output,
acceptance, latency 또는 아직 존재하지 않는 scale timing을 입력으로 사용하지 않는다.

- warmup: filtered order 0--3의 4 prompts
- measured: filtered order 4--19의 16 prompts
- warmup과 measured 전체: 20 distinct source documents, observed-window overlap 0
- prompt: 128 bytes
- controlled continuation: calibration stream의 같은 source offset에서 뒤따르는 128 bytes
- timed observed path: 128-byte parallel prefill + continuation 앞 127 bytes consume
- inner repetitions: 3
- role order: `(target_index + session_index + prompt_index + repetition) mod 2`
- timer: fresh runtime construction, parallel prefill, 모든 consume, final MPS synchronize 포함
- model build, source load, correctness oracle는 timing 밖

각 target을 세 개의 별도 fresh subprocess session에서 실행한다. Shared publication MPS flock, AC power,
thermal/process eligibility와 exact hardware/software contract를 시작·끝에서 확인한다.

## 4. correctness와 memory

각 target×schedule에서 첫 4 measured prompts 전체를 sequential runtime과 parallel-prefill runtime으로
재생한다.

- comparisons: `4 × 128 = 512`
- 모든 position argmax exact
- normalized logit error `<=1` under fixed `atol=2e-5`, `rtol=1e-4`
- boundary trace exact
- 모든 512개 prefix에서 독립 offline structural boundary oracle과 exact
- complete cache diagnostics exact
- timed path의 observed/local-cache byte count와 global patch count invariant exact

Target마다 model resident 직후와 각 trial의 마지막 MPS 동기화 직후, runtime/cache가 아직 살아 있는
시점의 driver allocated memory를 기록한다. 그 최대 관측값이 positive이고 recommended maximum의 75%
이하여야 한다. MPS가 제공하지 않는 resettable native peak로 과장하지 않으며, 이 값은 synchronized
post-trial safety snapshot이다. 환경 start/end identity도 byte-for-byte 같아야 한다. 하나라도 실패하면
성능값으로 training을 승인하지 않는다.

## 5. 통계와 고정 gate

Repetition은 독립 표본으로 세지 않는다. Session×prompt×role의 세 반복을 median으로 접은 뒤
3 fresh-process session과 16 prompt를 같은 paired index로 교차 재표집하는 10,000회 percentile
bootstrap을 사용한다. Seed는 `20260816 + target_millions`로 고정한다. Session cluster가 세 개뿐인
한계는 남으므로 이 interval은 training go/no-go용 보수적 sensitivity이지 최종 논문 CI가 아니다.

50M과 75M은 scale curve diagnostic이다. **100M만 primary**이며 다음을 모두 요구한다.

1. 모든 50M/75M/100M correctness·parameter·memory·environment evidence pass
2. 100M median E2E reduction `>=10%`
3. 100M prompt-bootstrap 95% lower bound `>=8%`
4. 100M faster prompts `>=15/16`
5. 100M 세 session 모두 양수
6. 100M session 중 `>=2/3`이 각각 `>=10%`

모두 통과하면 100M W72/C86 one-seed matched-quality training feasibility 하나만 허가한다.
50M이나 75M이 더 좋아도 그 target으로 fallback하거나 target을 고르지 않는다. 100M이 실패하면
publication-scale W72/C86 training, multi-seed, BPE family와 CUDA replication을 종료한다.

## 6. 증거 state machine

1. 이 protocol, core/runner/sealer/tests와 transitive runtime files를 clean commit한다.
2. CPU에서 세 deterministic model state, case arrays, environment와 implementation hash를 계산해
   plan을 no-clobber publish한다.
3. Plan을 별도 commit한다.
4. Clean plan commit에서 orchestrator가 `.active` sentinel을 만든다.
5. 50M/75M/100M × 세 session, 총 아홉 worker를 고정 순서의 fresh subprocess로 실행한다.
6. Worker report와 timing NPZ를 ignored artifact namespace에 no-clobber publish한다.
7. 세 target 전부 검증한 뒤에만 summary를 tracked result namespace에 publish하고 active를 지운다.
8. Summary를 commit한다.
9. Read-only verifier가 raw timing hash와 통계를 재구성하고 세 deterministic checkpoint에서
   correctness/oracle replay를 다시 실행한다.

중간 실패, partial target pair, stale active, deleted/reissued summary는 자동 구조하지 않는다.
성능 결과를 본 뒤 code, gate, prompts, repetition 또는 target을 바꾸려면 새 protocol namespace로
명시적으로 시작해야 하며 현재 결과는 그대로 보존한다.

## 7. Claim boundary

통과해도 다음만 말할 수 있다.

> Fixed publication-family geometry에서 W72 schedule의 larger-graph controlled runtime headroom이
> 100M one-seed quality experiment를 수행할 만큼 크다.

말할 수 없는 것:

- random weights가 Korean BPB quality를 보존한다.
- 100M trained W72가 실제로 C86보다 10% 빠르다.
- W72가 strongest raw/BPE reference보다 낫다.
- CUDA나 다른 hardware에 일반화된다.
- 7.8B retrieval 실패가 구제됐다.
- untouched final 또는 public preregistration 결과다.

실패하면 compact W72의 2.5% 양성 actual result는 유지되지만, model scale만으로 논문 가치가 큰
10% 효율 기법이 된다는 경로는 닫는다.
