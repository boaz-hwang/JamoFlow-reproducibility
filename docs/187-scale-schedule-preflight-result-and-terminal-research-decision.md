# Publication-scale schedule preflight result and terminal research decision

> 작성일: 2026-08-16
>
> 상태: **independently verified primary fail; publication-scale training stopped**
>
> Protocol: [Publication-scale W72/C86 schedule sensitivity preflight](./186-scale-schedule-preflight-protocol.md)
>
> 상위 결정: [EXAONE retrieval actual 결과와 core-scale 결정](./185-exaone-retrieval-actual-result-and-core-scale-decision.md)

## 1. 결론

같은 random-weight BLT graph에서 W72는 C86보다 모든 target, 모든 fresh-process
session, 모든 measured prompt에서 빨랐다. 그러나 100M primary reduction은
**4.460%**였고 고정 10% gate를 통과하지 못했다. 따라서 100M one-seed
matched-quality training을 포함한 publication-scale W72/C86 campaign은 실행하지 않는다.

이 결과는 두 사실을 동시에 보여 준다.

1. W72의 compact actual latency 방향은 우연이나 특정 trained checkpoint의 유리한
   weight 때문만은 아니다. 더 큰 동일-weight graph에서도 schedule만 바꾸어 3.6--4.5%
   실제 E2E 차이가 재현됐다.
2. Model scale이 커지면 그 차이가 다소 커지지만, 100M에서도 patch cadence만으로는
   연구 성공 기준인 10%에 절반도 도달하지 못한다. W72를 그대로 크게 학습하는 것은
   비용 대비 정보가 부족하다.

이는 quality-matched trained 100M 결과가 아니다. Random weights를 사용한 systems
preflight이므로 quality, downstream, broad raw/BPE 우위 또는 production speedup을
주장하지 않는다.

## 2. 봉인된 증거

- implementation/protocol commit: `6bc9ceb`
- plan commit: `e57fcf2`
- plan payload SHA-256:
  `77d8a7ba6b7397124a388dc4ced74a8ac2c6ba6a097dd69f7e3915c65388b83d`
- result commit: `1fbea35`
- summary payload SHA-256:
  `d77b990d1542a59bdc405e474556a9be3bd07f9c50287356f620c526684b9600`
- model targets: 49,823,488 / 76,492,480 / 98,403,360 parameters
- evidence: target 3개 × fresh subprocess session 3개 = 9 workers
- workload: 4 warmup + 16 measured cases, 128-byte prefill + 127 controlled
  consume, cell당 3 repetitions
- statistic: within-cell repetition median 뒤 fresh-session×prompt crossed bootstrap
- correctness: target×schedule×session마다 512 sequential/parallel logit 및 prefix
  boundary comparisons

초기 case 초안은 기존 pool의 첫 20개를 그대로 사용했는데, 감사 중 measured 두
255-byte observed window가 107 bytes 겹친다는 사실을 timing 전에 발견했다. Pool
순서는 그대로 두고 observed window 전체가 하나의 source document 안에 있으며 서로
다른 document인 첫 20개를 고르는 deterministic filter로 고쳤다. 최종 case는 20개
모두 서로 다른 document이고 window overlap은 0이며, 최소 selected offset 간격은
9,077 bytes다. 이 교정은 model output이나 scale timing을 입력으로 쓰지 않았지만,
historical pool과 post-EXAONE subset 교정을 사용했으므로 confirmatory/final로 부르지
않는다.

## 3. 실제 결과

| target | C86 median | W72 median | E2E reduction | crossed 95% interval | positive prompts | session reductions | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50M | 411.772 ms | 397.064 ms | **3.572%** | [2.771%, 4.502%] | 16 / 16 | 3.735%, 3.380%, 3.514% | diagnostic fail |
| 75M | 435.577 ms | 419.208 ms | **3.758%** | [3.302%, 4.253%] | 16 / 16 | 4.039%, 3.844%, 3.816% | diagnostic fail |
| 100M | 461.917 ms | 441.315 ms | **4.460%** | [3.846%, 4.893%] | 16 / 16 | 4.334%, 4.300%, 4.645% | **primary fail** |

모든 target에서 evidence-validity, parameter/state identity, sequential/parallel
correctness, independent offline boundary, cache invariant, environment identity와
memory-safety gate가 통과했다. 100M에서 통과한 performance clause는 세 session 모두
양수와 16/16 prompts 양수뿐이다. 다음은 실패했다.

- aggregate reduction `>=10%`: 4.460%, fail
- crossed-bootstrap lower `>=8%`: 3.846%, fail
- session `>=10%`가 최소 2/3: 0/3, fail

Observed 255 bytes에서 C86은 case마다 43 patches, W72는 36 patches를 만들었다.
Measured 16 cases의 합은 688 대 576으로 patch events가 정확히 **16.279%** 줄었다.
그런데 E2E 감소는 100M에서도 4.460%였다. 이는 patch-event 감소율을 wall-clock
감소율로 읽으면 안 된다는 compact profiler의 결론을 더 큰 graph에서 재확인한다.
Global event cost가 model size와 함께 커져 절대 차이는 50M의 14.707ms에서 100M의
20.602ms로 늘었지만, 양쪽에 공통인 127회 byte-local path가 여전히 지배적이다.

세 target만으로 asymptotic scaling law를 적합하거나 1B/8B 효과를 외삽하지 않는다.
특히 random-weight graph의 cache/runtime 비용은 측정했지만 trained activation이나
quality behavior를 대표하지 않는다.

## 4. 실행 후 postcondition과 forensic 처리

아홉 worker, raw artifact 검증, aggregate와 summary의 no-clobber publication까지는 모두
완료됐다. 그 뒤 runner의 마지막 workspace check가 실패했다. 기본
`git status --porcelain`은 새 result directory를 `?? results/.../`로 접어 표시하지만,
runner는 `?? results/.../summary.json` 한 줄을 기대했기 때문이다. 이 오류는 timing,
통계, gate 또는 artifact publication 이전이 아니라 **모두 끝난 뒤의 표시 형식 검사**다.

결과를 삭제하거나 재실행하지 않았다. 대신 다음 순서로 처리했다.

1. `.active`가 제거되고 9 report + 9 timing NPZ와 summary가 존재함을 확인했다.
2. Committed plan에서 모든 worker identity, raw array hash, role order, patch/boundary
   arrays를 다시 읽었다.
3. Raw timing으로 세 target 통계와 canonical summary를 다시 만들었다.
4. 재구성한 summary가 published summary와 byte-for-byte exact임을 확인했다.
5. Summary를 commit한 뒤 read-only verifier가 raw 통계 재구성과 세 deterministic
   checkpoint의 full MPS correctness replay를 완료했다.

Verifier 결과는 `scale_schedule_full_correctness_verification=pass`다. 따라서 마지막
Git display postcondition failure는 결과를 무효화하지 않지만, exact implementation
commit의 운영상 결함으로 공개한다. 같은 commit에서 재현하려면 Git status가 individual
untracked files를 표시하도록 설정하거나 summary publication 후 verifier를 사용해야 한다.

## 5. 앞선 결과와 종합

| stage | model/workload | actual result | fixed decision |
|---|---|---:|---|
| matched-quality W72 | trained 19.6M BLT | controlled +2.628%, free +2.531% | 10% co-primary fail |
| schedule scale sensitivity | random-weight 50/75/100M BLT | +3.572% / +3.758% / +4.460% controlled | 100M 10% fail |
| generic retrieval transfer | public EXAONE 3.5 7.8B 4-bit | **-14.938%** free E2E | retrieval branch stop |

이 세 결과를 합치면, 현재 연구에서 실제로 확인된 효율은 W72의 작고 안정적인
within-family 감소뿐이다. 단순 scale-up은 그 효과를 논문 success threshold까지 키우지
못했고, byte-sequential work를 건너뛰려던 generic retrieval은 큰 공개 모델에서 오히려
느려졌다. 앞서 검토한 Hangul/Jamo draft, conditional skipping, vocabulary expansion과
retrieval morphology branch도 각각 고정 gate를 통과하지 못했다.

따라서 실패한 mechanism을 다른 threshold, target, case 또는 model로 다시 찾지 않는다.
현재 evidence에서 100M pretraining, 3-seed expansion, BPE16K/32K scale family, CUDA
replication 또는 Hugging Face model upload를 허가할 근거가 없다. 공개 가치가 있는 것은
code, immutable aggregates, failure corrections와 연구 audit trail이지, 효율이 입증되지
않은 checkpoint가 아니다.

## 6. 최종 연구·논문 방향

현 증거로 강한 positive inference-efficiency paper를 주장하는 것은 타당하지 않다.
대신 다음의 좁고 정직한 논문은 연구 가치가 있다.

> Korean whitespace-informed patch relocation improves same-rate modeling quality and
> produces a small, reproducible end-to-end latency reduction, but patch-event savings
> do not translate into a large inference gain because byte-local autoregression remains
> dominant.

논문의 핵심은 새 “빠른 한국어 LLM”이 아니라 다음 세 가지다.

1. **Causal identification:** exact-rate C72/W72가 boundary placement의 품질 효과를
   분리하고, sealed final five-seed evidence로 이를 재현한다.
2. **Matched-quality actual systems evidence:** trained 19.6M physical bundles에서
   controlled/free 양쪽의 작은 양성 효과와 고정 10% 실패를 함께 보고한다.
3. **Scale-boundary evidence:** 같은-weight 50/75/100M graph에서 patch-event 감소가
   실제 wall time으로 변환되는 정도를 측정하고, unchanged schedule scale-up이 충분하지
   않음을 보인다.

이는 negative/diagnostic systems paper 또는 findings/workshop paper의 범위다. Top-tier
positive efficiency claim, Korean morphology superiority, public-model speedup, hardware
generality와 production serving claim은 제거한다. 추후 완전히 새로운 mechanism이 생기면
이 결과와 분리된 새 연구로 시작해야 하며, 현재 JamoFlow campaign의 성공 결론을 사후
변경하지 않는다.

