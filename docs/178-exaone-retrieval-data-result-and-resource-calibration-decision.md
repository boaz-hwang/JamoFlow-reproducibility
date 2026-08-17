# EXAONE retrieval data result and resource-calibration decision

> 작성일: 2026-08-15
>
> 상태: **metric-free data/table/case build 및 독립 재구성 통과; actual efficiency 미측정**

## 봉인 lineage

- implementation commit: `cbaa7b9a307e8ddbdbb4dfd8e01233e7a545bb43`
- data-plan commit: `056763884178360505cf34a013fb5ebbe11a5f6b`
- data plan SHA-256: `b2412dd3b40ce04d4ecf809b1d3e9f8f622c2d2bba31d8dbf3797a216c3a5575`
- data seal commit: `5a281116af75c05bf5111d4e34bda2bd9878e99a`
- data seal SHA-256: `ab1c7967a4f3404587fc9f1a4223e9fc3dae8ae376d25ffdb73f359cfcd06b6d`
- independent verification commit: `139a6848f6b9b6c64be878d77e9352af668b0a20`
- verification SHA-256: `35cef91d182273ba28eec359e30e8bae6dff58d4f0f547957dcc6399f8066a26`

V4 compatibility result를 공식 validator로 다시 검증했고, 현재 cached snapshot의 tokenizer 및 remote-code
8개 파일을 V4 file manifest와 size/SHA-256 수준에서 대조했다. Target model forward, candidate path,
acceptance, latency, throughput은 이 단계에서 실행하거나 읽지 않았다.

## 실제 build 결과

| 항목 | 결과 |
|---|---:|
| available stable-train documents | 5,637 |
| 128MB full-document prefix | 5,636 documents / 127,999,301 bytes |
| EXAONE train tokens | 28,389,609 |
| selected-train vs evaluation exact intersection | 0 |
| selected-train vs evaluation normalized intersection | 0 |
| compact table entries | 200,000 |
| order-1 entries | 1,706 |
| order-2 entries | 63,136 |
| order-3 entries | 135,158 |
| evaluation documents | 1,482 |
| fixed eligibility를 통과한 documents | 1,379 |
| selected cases | 8 warmup + 64 measured |

Normalized intersection은 `NFKC → casefold → whitespace collapse → SHA-256` 규칙으로 selected train
prefix와 evaluation documents를 직접 다시 계산했다. Table context는 `uint64`, next token/count는
`uint32`로 분리했고 cast 전 범위를 검사했다.

## 독립 재구성

Seal commit 뒤 별도 clean process에서 다음을 모두 다시 수행했다.

1. train/evaluation source seal 공식 검증
2. exact/normalized disjointness 재계산
3. EXAONE tokenizer snapshot file identity 재검증
4. 28,389,609-token stream 재생성
5. orders 1/2/3 table 전체 재구성
6. 1,482 evaluation documents에서 case eligibility와 rank 재계산
7. table/case 모든 array의 bitwise equality 비교

결과는 `pass_independent_source_tokenizer_table_case_reconstruction`이다.

## 해석과 다음 결정

이 단계는 데이터 무결성과 deterministic retrieval proposal source만 확립했다. 200,000-entry table이
실제로 target call 수나 wall time을 줄인다는 증거는 아직 없다. 또한 72개 case는 과거 품질평가에 쓰인
document pool에서 새로 선택했기 때문에 untouched final이 아니다. 이 workload는 8B raw-completion
systems development/replication으로만 해석한다.

다음은 candidate를 전혀 실행하지 않는 baseline-only resource calibration이다. 여기서 고정할 것은 다음뿐이다.

- 7.8B ordinary greedy의 prompt prefill/decode wall time와 peak memory
- 72 cases 전체를 한 fresh process에서 실행할 수 있는지
- thermal cooldown 및 fresh-process session 비용
- inner repetition 수와 독립 session 수의 실행 가능 범위
- controlled/free horizon의 안전한 상한

Resource calibration 결과로 candidate의 table size, proposal cap, lookup order, case set을 바꾸지 않는다.
그 뒤 actual plan을 봉인하고 ordinary AR와 exact retrieval candidate를 balanced fresh-process sessions에서
처음 비교한다. Generic scale-transfer가 통과해도 논문 확증 전에는 별도의 미사용 Korean raw-completion과
chat-template workload를 추가해야 한다.
