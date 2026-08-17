# Normalization results: boundary rules do not repair representation shift

> 작성일: 2026-08-10
>
> Corrections: [causality](./15-normalization-protocol-correction.md), [rate feasibility](./16-normalization-rate-feasibility-correction.md)
>
> 기계 판독 결과: [`results/phase2-normalization/summary.json`](../results/phase2-normalization/summary.json)
>
> 상태: **Gate G opportunity 실패**

## 1. 결론

NFC-trained compact byte-BLT는 같은 한국어를 NFD 또는 compatibility jamo로 바꾸면 심각하게 붕괴했다. Non-causal Hangul-unit oracle은 이 붕괴를 복구하지 못했고 generic codepoint grid보다 평균적으로 조금 더 나빴다.

- NFD rate-28 oracle − codepoint: **+0.1430 bits/source-codepoint**
- oracle 상대 개선률: **−0.236%**; negative는 oracle 악화
- 1% 이상 개선 seed: 0/5
- Gate G opportunity: **실패**

핵심 결론은 “Hangul unit rule이 무용하다”가 아니다. 더 정확한 결론은 다음이다.

> 학습에서 보지 못한 byte/Jamo representation의 distribution shift는 inference patch boundary만 바꿔서 해결되지 않는다.

## 2. 먼저 드러난 구조적 negative result

Quality 평가 전에 두 protocol 문제가 발견돼 공개적으로 수정했다.

1. `L+V` 뒤 optional T 존재 여부는 미래 codepoint 없이는 알 수 없어 full unit boundary는 prefix-causal하지 않다.
2. NFD unit은 6–9 bytes라 256 bytes에 exact 43 unit-preserving patches를 항상 만들 수 없다.

따라서 deployable rate-43 정책과 non-causal rate-28 oracle pair를 분리했다.

- Rate 43: fixed-byte / codepoint / whitespace
- Rate 28 opportunity pair: codepoint / non-causal Hangul-unit oracle

모든 condition에서 exact rate invariant를 확인했고 NFC에서 rate-28 두 matrix는 완전히 같았다.

## 3. Data와 packing

Primary Korean test stream의 strict-decodable prefix 999,935 bytes를 동일 source text로 사용했다.

| Condition | Sequences | Newline padding | Represented source codepoints |
|---|---:|---:|---:|
| Original | 3,919 | 0.351% | 410,489 |
| NFC | 3,919 | 0.351% | 410,489 |
| NFD | 8,996 | 1.217% | 410,558 |
| Compatibility jamo | 8,920 | 0.377% | 410,553 |

NFD packing은 Jamo unit 중간에서 row를 끊지 않기 위해 최대 8-byte newline padding을 허용했다. 따라서 primary arbitrary-packing BPB와 직접 합치지 않는다.

Original source는 NFC와 동일했다. Original/NFC의 inputs와 결과도 동일했다.

## 4. 평균 quality

### Rate-43 deployable diagnostics

| Condition | Fixed-byte BPB | Codepoint BPB | Whitespace BPB |
|---|---:|---:|---:|
| NFC | 2.4173 | 2.4106 | **2.4034** |
| NFD | 10.7963 | 10.8685 | **10.7469** |
| Compatibility jamo | 8.4664 | 8.4287 | **8.3890** |

NFC에서 whitespace − codepoint는 −0.01776 bits/source-codepoint, 95% CI [−0.02258, −0.01294]였다. NFD에서도 −0.6794 [−1.1226, −0.2362]로 whitespace policy의 상대 순위는 유지됐다. 하지만 절대 degradation이 너무 커 이를 robustness 해결이라고 부를 수 없다.

BPB는 8을 넘을 수 있다. Cross-entropy는 uniform 256-byte distribution의 8 bits보다 나쁜, 잘못 확신한 prediction에 대해 상한이 없다.

### Rate-28 oracle pair

| Condition | Codepoint bits/source-CP | Oracle bits/source-CP | Oracle − codepoint |
|---|---:|---:|---:|
| NFC | 5.9518 | 5.9518 | 0.0000 |
| NFD | 60.7969 | 60.9399 | **+0.1430** |
| Compatibility jamo | 46.7289 | 46.7289 | 0.0000 |

NFD seed별 oracle relative improvement는:

```text
−0.418%, −0.632%, −0.038%, −0.119%, +0.027%
```

다섯 seed 중 네 개에서 oracle이 나빴다. Paired difference의 95% CI [−0.0695, +0.3555] bits/source-codepoint는 0을 포함한다. 적어도 이 checkpoint와 rate에서는 unit-preserving lookahead가 유용하다는 증거가 없다.

## 5. Representation degradation

NFC 대비 bits/source-codepoint 상대 증가:

| Policy | NFD 증가 | Compatibility-jamo 증가 |
|---|---:|---:|
| fixed-byte rate43 | +925% | +697% |
| codepoint rate43 | +935% | +696% |
| whitespace rate43 | +927% | +694% |
| codepoint rate28 | +922% | +685% |
| oracle rate28 | +924% | +685% |

예를 들어 codepoint rate43은 NFC 5.869에서 NFD 60.727 bits/source-codepoint로 늘었다. 이는 “9.35% 증가”가 아니라 **+935%, 약 10.35배 수준**이다.

Compatibility jamo는 canonical equivalence가 아닌 stress transform이다. NFD는 canonical equivalence인데도 byte sequence가 크게 달라져 유사하게 붕괴했다. Byte-level model이라고 normalization invariant가 자동으로 생기지 않는다는 직접 증거다.

## 6. Boundary diagnostics

NFD에서 oracle-Hangul-unit 내부에 놓인 noninitial boundaries:

- fixed-byte rate43: 81.09%
- codepoint rate43: 56.08%
- whitespace rate43: 46.68%
- codepoint rate28: 56.57%
- oracle rate28: 0%

Oracle은 설계대로 unit을 완전히 보존했지만 quality가 좋아지지 않았다. 즉 boundary alignment metric 자체를 목적 함수처럼 최적화하면 안 된다. Model이 NFC bytes로만 학습된 상태에서는 NFD Jamo sequence를 표현하고 예측하는 능력이 먼저 부족하다.

## 7. Gate G와 연구 방향

Opportunity gate 조건:

- NFD mean relative improvement ≥1%: 실패
- 최소 4 seed에서 ≥1%: 0/5, 실패
- NFC matrix identity: 통과
- exact rate 28: 통과
- oracle unit-internal boundary 0%: 통과

Gate G는 실패했다. NFD full-unit causal architecture를 이번 scale-up method에 넣지 않는다.

대신 normalization robustness를 얻으려면 다음 중 하나가 필요하다.

1. NFC/NFD augmentation으로 학습
2. local encoder에서 canonical-equivalent representation 공유
3. normalization-aware byte/jamo embedding objective
4. 입력 canonicalization baseline

이들은 boundary-only method와 다른 연구 질문이다. 이번 논문의 main method 범위를 늘리지 않고 failure analysis와 future work로 둔다.

## 8. 논문에 남길 메시지

Positive method 결과와 함께 다음 negative result를 남길 가치가 있다.

> Unicode-safe 또는 Hangul-unit-safe patch boundaries는 well-formed segmentation을 보장할 수 있지만, unseen normalization form에 대한 language-model robustness를 보장하지 않는다. Encoding validity, boundary alignment, predictive quality는 서로 다른 축이다.
