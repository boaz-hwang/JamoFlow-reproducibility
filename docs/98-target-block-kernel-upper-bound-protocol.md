# Perfect-draft target block-kernel upper-bound protocol

> 봉인일: 2026-08-13
>
> 상태: **target timing 전 protocol seal**
>
> manifest: `data/manifests/target-block-kernel-v1.json`

## 1. 왜 이 측정이 필요한가

Learned Hangul draft preflight는 실패했다. 가장 강한 architecture는 한글 조합 head가
아니라 generic independent UTF-8 head였고, complete-pair acceptance도 24.379%에
그쳤다. 이 결과를 보고 head, threshold, rank 또는 training budget을 다시 튜닝하지 않는다.

다만 기존 stop rule은 speculative mismatch 때 target correction byte가 함께 확정되고,
완전 수락 때 verifier bonus byte가 생긴다는 사실을 비용식에서 빠뜨렸다. 현재 target이
이미 낸 첫 byte와 두 draft byte를 한 번에 검증하면 fixed head seed 20260813의 기대 확정량은

`2 + 0.4237276383 + 0.2437942033 = 2.6675218416 bytes/verification`

이다. 이 값은 speedup이 아니라 target block 호출을 실제로 얼마나 싸게 만들면 되는지를
결정하는 work opportunity다. 따라서 full speculative runtime을 만들기 전에, draft가 항상
맞는다는 가장 유리한 가정에서도 target-side kernel이 충분히 빠른지를 측정한다.

## 2. 고정된 대상과 데이터

- model: quality-authorized W72, seed 1729, weights frozen
- policy horizon: 512 bytes; 1,032-position rotary capacity와 혼동하지 않는다
- data: 기존 HPLT calibration stream 첫 1,000,000-byte quota
- realized stream: 999,936 bytes, SHA-256
  `69f6aa9347f7e265d6df5097e4219c944b4da7cf6d8522a831a26c670a4c39ec`
- final test, final NLL, v5r3 timing case는 case 선택에 사용하지 않는다
- device: Apple MPS, AC power, shared publication MPS exclusion lock

Case는 model output과 무관한 domain-separated SHA-256 rank로 선택한다. 모든 prompt는
128 bytes이고 UTF-8 scalar boundary에서 끝난다.

## 3. 두 측정

### Micro target call

Complete precomposed Hangul scalar의 세 bytes를 다음 두 방식으로 전진시킨다.

1. 기존 `consume` 세 번
2. 새 local encoder/decoder block kernel 한 번

새 W72 patch boundary가 block 안에 없는 32 case와 정확히 하나 있는 32 case를 분리한다.
Prefill은 timer 밖이며, 세 target bytes와 마지막 MPS sync만 잰다. 각 case를 5회 반복하고
두 mode의 선후 순서를 parity로 정확히 균형화한다. 이후 한 follow byte까지 전진시켜
cache propagation도 비교한다.

### Perfect-Hangul whole path

79% 이상 Hangul-heavy prompt 16개에서 255--258-byte continuation을 고정한다. 완벽한
oracle이 모든 precomposed Hangul scalar를 세-byte block으로 주고, 다른 UTF-8 bytes는
하나씩 준다고 가정한다. Fresh runtime 생성, prompt prefill, continuation 전진, 마지막
MPS sync 전체를 timer 안에 넣고 3회 반복한다. Head와 rollback은 실행하지 않으므로 이는
실제 speculative runtime보다 유리한 upper bound다.

별도 untimed oracle에서 continuation의 **모든 위치 logits**, argmax, patch/global/local
cache diagnostics를 기존 sequential runtime과 비교한다. MPS tolerance는 기존 v5r3와 같은
`rtol=2e-5`, `atol=1e-4`이고 argmax와 boundary/cache trace는 exact여야 한다.

## 4. 통계 단위와 gate

Repetition은 독립 표본으로 세지 않는다. 먼저 case 안에서 median으로 접고, micro 두
stratum과 whole cases를 독립 resampling하는 10,000회 case bootstrap을 사용한다. Seed는
20260829로 고정한다.

Full speculative rollback prototype은 아래를 모두 만족할 때만 허가한다.

| gate | point | 95% bootstrap lower |
|---|---:|---:|
| empirical-boundary-weighted target block reduction | ≥30% | ≥20% |
| perfect-Hangul whole-path reduction | ≥20% | ≥10% |
| fixed independent head cost를 더한 projected reduction | ≥20% | ≥10% |

마지막 projection은 micro target cost, fixed head latency 1.0049585 ms, 기대 확정량
2.66752184를 사용한다. Rollback/cache crop, mismatch branch, output mask와 stop logic은
아직 빠져 있으므로 20% point threshold는 최종 목표 10%를 위한 최소 engineering margin이다.

하나라도 실패하면 multi-byte branch를 종료하고 acceptance threshold나 head를 사후
조정하지 않는다. 통과하더라도 다음 단계는 calibration-only exact rollback prototype일
뿐이며, generic all-byte MTP와 W72 AR을 이기는 실제 E2E 결과 전에는 positive efficiency
claim을 허가하지 않는다.

## 5. Claim 경계

- 이 실험은 acceptance 결과를 본 뒤 설계한 exploratory upper-bound다.
- Perfect draft는 실제 모델이 낼 수 있는 quality/acceptance 증거가 아니다.
- Whole-path timing에는 proposal head가 없고 projection에만 기존 isolated head cost가 있다.
- 통과해도 한국어-specific novelty는 복원되지 않는다. 앞선 결과에서는 generic head가
  모든 Hangul-specific head보다 강했다.
- Final test를 재사용하지 않으며 publication result가 아니라 다음 구현에 대한 stop/go다.
