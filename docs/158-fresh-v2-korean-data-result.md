# Fresh-v2 Korean data result

> 작성일: 2026-08-15
>
> 상태: model-free prepare + independent full-rescan 검증 완료
>
> protocol commit: `05ded9c`
>
> seal payload SHA-256:
> `5db335c51d6511052404c6d8fa25f09db83e03ea4e29cde388115c47e8c67b80`

## 결론

16K vocabulary-expansion fail-fast에 사용할 두 번째 한국어 train/calibration corpus를 봉인했다.
동일 pinned HPLT3 Korean shard에서 historical Phase-3 6,911개, sealed final 1,482개,
fresh-v1 6,076개를 exact SHA-256과 고정 normalized digest로 모두 제외했다.

Prepare와 독립 verifier가 각각 1.7GB archive를 처음부터 전수 스캔해 같은 ignored JSONL과
aggregate seal을 재구성했다. 이 결과는 model quality나 actual inference 효율의 양성 증거가
아니다. 16K 실험이 이미 학습에 사용한 문서를 재사용하지 않는다는 data-integrity gate다.

## 봉인된 split

| split | exact stream bytes | sequences | selected docs | full selected stream | overshoot |
|---|---:|---:|---:|---:|---:|
| train | 128,000,000 | 250,000 | 5,637 | 128,008,526 | 8,526 |
| calibration | 8,000,000 | 15,625 | 357 | 8,032,088 | 32,088 |

- train stream SHA-256:
  `63ded32e267597bcfc4d07d88445f06af11474680f3fa299eb848fc081f69523`
- calibration stream SHA-256:
  `79a4a4de910fa6e5bbfc267ad40c5ffaf26a2959c419dd7b9e6b775320336e46`
- ignored full JSONL: 136,411,447 bytes,
  SHA-256 `736b897a2eedeceab6e23e93bc1d4c36d849160f7acbe49029e81ce434967169`
- tracked seal file SHA-256:
  `c7ceeb3290db5e1d0b905494d15b54874f22f53da3281f155a6e2e11437bbe9e`
- manifest SHA-256:
  `175dfb7b4661e1c3a68783a49562b8ce30e309b3576f1fe2c63a3cd150dbc5ad`

두 split 모두 full-document rank prefix가 quota에 처음 도달한 지점까지 선택한다. 모델 입력은
마지막 문서를 포함한 joined stream의 정확한 quota prefix만 사용하며, 두 quota는 512로
나누어떨어져 partial sequence가 없다.

## Exclusion과 전체 scan accounting

| 항목 | records |
|---|---:|
| source lines / parsed | 273,839 |
| too long | 477 |
| eligible unique records | 273,362 |
| Phase-3 + final + fresh-v1 exact exclusions | 14,469 |
| stable-test records ignored | 25,222 |
| post-exclusion train candidates | 207,123 |
| post-exclusion calibration candidates | 26,548 |
| normalized-only exclusions | 0 |
| normalized source duplicates | 0 |

Accounting은 다음 등식으로 닫힌다.

```text
273,362 = 14,469 + 25,222 + 207,123 + 26,548
```

14,469개 exact exclusion을 raw shard에서 모두 다시 찾았다. Exact와 normalized exclusion set은
각각 14,469개이고 세 dependency 집합 사이 overlap은 없다.

Fresh-v1 scan과 비교하면 candidate train은 정확히 5,692개, calibration은 정확히 384개
감소했다. 이는 fresh-v1이 선택했던 split별 문서 수와 일치한다. 이번 shard에서는 별도의
normalization-only 중복이나 source normalized duplicate가 없었지만 guard와 aggregate
commitment는 그대로 유지한다.

## 독립 검증

다음 두 실행이 byte-identical output과 seal을 만들었다.

1. `scripts/prepare_hplt3_fresh_adaptation_v2.py`
2. `scripts/verify_hplt3_fresh_adaptation_v2.py`

Verifier는 첫 scan의 selected document 목록을 입력으로 받지 않는다. Pinned raw source와 세
dependency 집합에서 exclusion, stable split, domain-separated rank, quota prefix, JSONL 및 seal을
전부 재계산한다.

Protocol commit 전에 수행한 전체 repository regression은 816 tests와 86 subtests를 통과했다.
V2 전용 synthetic suite는 v1 identity·quota·rank rotation, exact/normalized exclusion,
stable-test representative ordering, seal accounting, model/result import/path 접근을 부정
테스트한다. 이 테스트 통과 수는 data/model 결과가 아니며, 향후 code coverage 수로 재사용하지
않는다.

## 해석 경계

- Fresh-v2는 동일 raw shard의 deterministic disjoint sample이다. 새로운 source domain 또는
  시간적으로 더 최신인 corpus가 아니다.
- 보장 범위는 byte-exact 및 고정 `NFKC→casefold→whitespace collapse` equality다. 일반
  near-duplicate나 semantic contamination 부재를 주장하지 않는다.
- Fresh-v1 actual 결과를 본 뒤 16K 방향과 v2 생성을 선택했다. 이 adaptivity는 명시하며,
  v2 안에서 문서·quota·rank를 결과에 맞춰 바꾸지는 않았다.
- Sealed final-test model loss와 v2 calibration loss는 이 데이터 단계에서 계산하지 않았다.

## 다음 gate

다음 commit에서 2K anchor, fixed 8K cross-data replication, 16K standard joint,
literature-aligned two-stage expansion, fixed update-geometry 16K candidate의 정확한 tokenizer,
initialization, optimizer, budget, quality selection과 independent replay 계약을 봉인한다.

16K candidate가 quality gate를 통과하지 못하면 actual timing을 실행하지 않는다. 통과해도
controlled와 free-running actual end-to-end 각각 `>=10%`라는 원래 성공 기준은 낮추지 않는다.
