# Fresh vocabulary-adaptation Korean data result

> 작성일: 2026-08-15
>
> 상태: model-free prepare + independent full-rescan 검증 완료
>
> protocol commit: `f492488`
>
> seal payload SHA-256:
> `351a1ae05e35198e53f7576eace5e3426fe88150392e1f8f661e388e8599657a`

## 결론

Known B1 corpus에서 발견한 ordinary dense new-row optimization을 fresh하게 검증할 train/calibration
stream을 만들었다. 동일 pinned HPLT3 Korean shard를 사용하지만 historical Phase-3 6,911문서와
sealed final-test 1,482문서를 exact SHA-256 및 고정 normalized digest로 모두 제외했다.

Prepare와 별도 verifier가 1.7GB raw shard를 각각 처음부터 읽어 동일한 ignored JSONL과 aggregate
seal을 재구성했다. 이 결과는 모델 quality나 efficiency의 양성 증거가 아니라 다음 one-seed
matched-quality 실험을 열 수 있는 데이터 무결성 gate다.

## 봉인된 split

| split | exact stream bytes | sequences | selected docs | full selected stream | overshoot |
|---|---:|---:|---:|---:|---:|
| train | 128,000,000 | 250,000 | 5,692 | 128,064,198 | 64,198 |
| calibration | 8,000,000 | 15,625 | 384 | 8,035,141 | 35,141 |

- train stream SHA-256:
  `f1d3cba694a5929e3c46891dd0e70139e12731482555f82534b8043de6e7857b`
- calibration stream SHA-256:
  `8272123a3ee9bc5bba218d43248aaf600434a35659a8c36ed47b04809bb9e630`
- ignored full JSONL: 136,470,910 bytes,
  SHA-256 `7817d3be0d67099735e6a26c741314ba27fe12b9cb7dd7d3b7022af40ea3b2c5`
- tracked seal file SHA-256:
  `2a1457b0b1cd1ffcaabde7997056b0283d5adebe925a7b228a046d0cdbe6f916`

두 stream은 full-document rank prefix가 quota에 처음 도달한 지점까지 선택하고, 학습 입력은 마지막
문서를 포함한 joined stream의 정확한 quota prefix만 사용한다. 두 quota 모두 512로 나누어떨어져
partial training sequence는 없다.

## 전체 scan accounting

| 항목 | records |
|---|---:|
| source lines / parsed | 273,839 |
| too long | 477 |
| eligible unique records | 273,362 |
| historical + final exact exclusions | 8,393 |
| stable-test records ignored | 25,222 |
| post-exclusion train candidates | 212,815 |
| post-exclusion calibration candidates | 26,932 |
| normalized-only exclusions | 0 |
| normalized source duplicates | 0 |

Accounting은 다음 등식으로 닫힌다.

```text
273,362 = 8,393 + 25,222 + 212,815 + 26,932
```

Exact exclusion 8,393개가 raw shard에서 모두 다시 발견됐다. Exact set과 normalized set은 각각
8,393개였고 predecessor와 final-test 사이에도 normalized overlap이 없었다. 이번 source에서는
exact하지 않은 추가 normalized duplicate가 0건이었지만, 해당 guard와 commitment는 protocol에
그대로 유지한다.

## 독립 검증

다음 두 실행은 같은 결과를 냈다.

1. `scripts/prepare_hplt3_fresh_adaptation.py`
2. `scripts/verify_hplt3_fresh_adaptation.py`

Verifier는 기존 selected 목록을 입력으로 받지 않는다. Pinned raw source, predecessor artifacts,
final-test artifacts와 manifest에서 exclusion, stable split, rank order, quota prefix, JSONL 및 seal을
전부 다시 계산하고 byte equality를 요구한다.

Tracked seal에는 원문·개별 document digest·checkpoint·loss·BPB·latency가 없다. Preparation code의
import/path audit도 NumPy, PyTorch, tokenizer/model stacks와 model-result artifacts 접근을 거부한다.

## 해석 경계

- 동일 raw shard에서 결정적으로 disjoint한 새 표본이지, 새로운 source domain은 아니다.
- 보장 범위는 exact byte identity와 `NFKC→casefold→whitespace collapse` equality다. 일반적인
  near-duplicate나 semantic overlap 부재를 주장하지 않는다.
- Sealed final test의 model loss는 이 단계에서 계산하지 않았다.
- 데이터가 fresh하다는 사실만으로 post-hoc 발견 recipe가 새 방법이 되지는 않는다.

## 다음 gate

다음 commit에서 결과를 보기 전에 dense-2K continuation, standard joint dense-8K,
first-batch-derived asymmetric new-row amplification, literature-aligned new-row-only/high-LR control의
정확한 학습·선택 계약을 고정한다. 모든 역할은 이 train stream의 같은 128M raw bytes를 사용한다.

Calibration-only one-seed screen에서 dense-2K matched-quality와 강한 dense-8K controls를 통과한
checkpoint만 actual controlled/free inference로 보낸다. 두 actual mode의 `>=10%` point gate는
변경하지 않는다.
