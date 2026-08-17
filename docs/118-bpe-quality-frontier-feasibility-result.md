# BPE quality-frontier feasibility 결과

> 작성일: 2026-08-14
>
> 상태: resource-only sealed result; 128M raw-byte one-seed quality campaign 승인

## 결론

여섯 vocabulary별 fastest graph를 동일한 128M Korean raw-byte stream으로 학습하고 8M
calibration 전체를 평가하는 campaign은 이 Mac에서 실행 가능하다. 고정 projection은 총
**2.923 hours**로 24-hour limit의 12.2%이며, 모든 isolated worker가 memory gate를 통과했다.
따라서 role이나 data budget을 줄이지 않고 128M one-pass quality frontier를 진행한다.

이 판단에는 loss 또는 BPB 값이 들어가지 않았다. Worker는 loss가 finite인지 확인했지만 수치를
stdout/report에 기록하지 않았다.

## Role별 실측과 128M projection

| role | effective train step | eval batch | optimizer steps | projected hours | driver/recommended | RSS/physical |
|---|---:|---:|---:|---:|---:|---:|
| 2K×8L | 0.941 s | 0.501 s | 2,213 | 0.588 | 27.5% | 4.1% |
| 4K×12L | 1.152 s | 0.401 s | 1,904 | 0.623 | 23.2% | 4.1% |
| 8K×8L | 0.997 s | 0.174 s | 1,675 | 0.474 | 14.4% | 4.1% |
| 16K×8L | 1.033 s | 0.090 s | 1,502 | 0.440 | 11.4% | 4.1% |
| 32K×8L | 0.963 s | 0.036 s | 1,358 | 0.370 | 8.6% | 4.1% |
| 64K×8L | 1.213 s | 0.025 s | 1,244 | 0.428 | 6.3% | 3.8% |

Evaluation batch는 vocab에 반비례해 64/32/16/8/4/2였기 때문에 driver allocation이 작은
vocab에서 더 높다. Native resettable MPS peak는 지원되지 않아 fresh process의 sampled
current/driver allocation과 `ru_maxrss`를 사용했다. 이는 정확한 instantaneous peak가 아니라
conservative isolated-process diagnostic이다.

## 해석

1. 64K는 32K보다 optimizer step이 8.4% 적지만 effective step이 26.1% 더 느려 총 projected
   role time도 15.7% 길다. Training에서도 token count만으로 비용을 예측하면 틀린다.
2. 2K는 step 자체가 빠르지만 token sequence가 많아 가장 많은 optimizer step을 필요로 한다.
3. 4K×12L가 role별 시간이 가장 긴 것은 vocabulary뿐 아니라 systems frontier가 선택한
   depth 차이도 반영한다. 이를 vocabulary의 순수 효과로 해석하지 않는다.
4. 128M 전체가 3시간 미만으로 projection돼 64M/32M으로 줄일 과학적 또는 운영상 이유가
   없다. 결과를 더 빨리 얻기 위한 budget 축소는 하지 않는다.

## 다음 단계

Quality campaign은 다음을 결과 전에 봉인한다.

- exact six roles와 tokenizer/model specs
- seed와 initialization state hash
- 128M token stream, deterministic sequence order, optimizer/schedule
- raw target bytes를 denominator로 한 calibration per-sequence NLL/BPB
- checkpoint, training report, loss-array artifact identities
- strongest calibration BPB anchor와 `+0.010 BPB` quality-qualified set
- quality-qualified set 안에서 systems-frontier E2E가 가장 빠른 BPE comparator를 선택하는 규칙

One seed와 calibration-development 결과이므로 그 comparator는 Korean-aware method의 최종
publication comparator가 아니다. 이후 candidate와 함께 새 sealed final data, multiple seeds,
free-running actual timing에서 다시 확인한다.

## Artifact

- plan: `data/manifests/bpe-quality-frontier-feasibility-v1.json`
- tracked summary: `results/bpe-quality-frontier-feasibility-v1/summary.json`
- ignored worker/report evidence: `artifacts/bpe-quality-frontier-feasibility-v1/`
