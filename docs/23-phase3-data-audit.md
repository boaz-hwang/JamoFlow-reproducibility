# Phase 3 data audit: HPLT3 Korean full-shard sample

> 작성일: 2026-08-10  
> 사전등록: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md)  
> 기계 판독 결과: [`results/phase3-data/summary.json`](../results/phase3-data/summary.json)  
> 상태: **데이터 무결성 통과**

## 1. 결론

HPLT3 `sorted/kor_Hang/10_1` 전체 shard를 scan해 Phase 3의 공개 Korean sample을 준비했다.

- compressed source: **1,862,302,013 bytes**
- source SHA-256: `de4dfa43fd9f6c62cc81781e09c1f401cc77e7a956e07ecc80ac13477e699ca4`
- source records: **273,839**
- eligible records: **273,362**
- eligible UTF-8 text: **6,107,283,275 bytes**
- selected records: **6,911**
- selected neural stream: train **128M**, calibration **8M**, test **16M bytes**
- JSON/UTF-8 parse failure: **0**
- exact text duplicate: **0**
- split 간 selected digest-set summary: distinct
- raw/processed text promotion: **없음**

세 split 모두 사전 고정한 quota를 정확히 512-byte sequence로 재구성했다. 데이터 gate는 통과한다.

## 2. source와 selection

Source는 다음 metadata로 고정했다.

| Item | Value |
|---|---|
| dataset | HPLT 3.0 `kor_Hang` sorted release |
| shard | `10_1.jsonl.zst` |
| ETag | `"6f00793d-63be01a95c540"` |
| Last-Modified | `Fri, 08 Aug 2025 20:06:05 GMT` |
| compressed SHA-256 | `de4dfa43fd9f6c62cc81781e09c1f401cc77e7a956e07ecc80ac13477e699ca4` |

Source 순서의 prefix를 자르지 않았다. 273,839개 line을 끝까지 읽고 다음 순서로 처리했다.

1. strict JSON/UTF-8 parse
2. 256–262,144 UTF-8 bytes 문서만 유지
3. exact text SHA-256 deduplication
4. 기존 text-hash 80/10/10 split
5. split별 salted bottom-hash selection
6. 필요한 stream byte quota가 찰 때까지 hash-rank 순서로 선택

262,144 bytes를 넘은 477개 문서, 전체의 약 0.174%만 제외됐다. 256 bytes 미만 문서는 이 shard에 없었다.

Bottom-hash는 문서를 균등하게 뽑고 선택된 문서의 전체 text를 byte stream에 넣는다. 따라서 **document selection은 hash-uniform이지만 training contribution은 문서 길이에 비례**한다. 이는 byte-LM 학습 목적에는 맞지만 문서 단위 corpus prevalence 추정 표본으로 사용하면 안 된다.

## 3. split 무결성

| Split | Records | Available stream bytes | Selected bytes | 512-byte sequences | Exact quota |
|---|---:|---:|---:|---:|---:|
| train | 5,791 | 128,006,987 | 128,000,000 | 250,000 | yes |
| calibration | 386 | 8,004,309 | 8,000,000 | 15,625 | yes |
| test | 734 | 16,006,020 | 16,000,000 | 31,250 | yes |

Processed JSONL을 다시 읽고 `stable_record_id(text_bytes)`와 `split_for_record`를 재적용한 값이다. Preparation 단계가 임의 split label을 주입하지 않는다.

각 split의 ordered selected-digest set을 한 번 더 SHA-256으로 집계했다. 세 aggregate digest는 서로 다르고 loader count도 preparation count와 일치한다. Per-document ID와 hash 목록은 저장소에 승격하지 않았다.

## 4. 실제 Korean/mixed composition

Codepoint와 문서 기준 aggregate는 다음과 같다.

| Split | Hangul syllable / CP | ASCII / CP | Whitespace / CP | Any Latin mixed docs | NFC docs |
|---|---:|---:|---:|---:|---:|
| train | 69.03% | 30.06% | 21.81% | 95.94% | 99.983% |
| calibration | 69.17% | 29.85% | 21.60% | 97.15% | 100% |
| test | 69.19% | 29.88% | 21.99% | 94.28% | 100% |

이 비율들은 상호배타적 분류가 아니다. 예를 들어 whitespace는 대부분 ASCII에도 포함된다. `Any Latin mixed docs`는 영문자 하나만 있어도 true이므로 code-mixing의 강도를 뜻하지 않는다. URL, 표기, 고유명사 때문에 높은 값이 나올 수 있다.

그럼에도 이 표본이 “순수 Hangul-only”가 아니라는 점은 분명하다. 이는 Phase 3에 유리한 문서만 고른 것이 아니라 실제 고품질 Korean web text의 mixed-script 조건을 포함한다는 장점이 있다. 반대로 한국어 구어, 뉴스, 문학 전체를 대표한다고 주장할 수는 없다.

## 5. arbitrary byte packing 진단

512-byte sequence 시작이 UTF-8 codepoint 내부인 비율은 다음과 같다.

- train: **58.332%**
- calibration: **59.046%**
- test: **57.949%**

이는 오류가 아니라 continuous raw-byte stream을 512 bytes마다 자른 결과다. NFC 한글 음절이 3 bytes이고 window size가 codepoint 경계와 정렬되지 않으므로 예상 가능한 현상이다.

다만 다음 해석상 중요하다.

1. F는 내부 시작을 자연스럽게 허용한다.
2. C/W/EC는 position 0 이후 boundary만 codepoint-safe하게 만든다. 이미 내부에서 시작한 첫 incomplete codepoint를 복구해 주지는 않는다.
3. Phase 3의 pre-registered `sequence start inside codepoint` stratum을 반드시 보고해야 한다.
4. 향후 document/codepoint-aligned packing은 별도 ablation이어야 하며 primary 결과를 본 뒤 교체할 수 없다.

## 6. normalization

Calibration/test의 selected 문서는 모두 Python NFC-normalized 판정을 통과했다. Train 5,791개 중 한 문서만 NFC가 아니었다.

따라서 main test는 사실상 NFC Korean 결과다. NFD robustness를 main 평균에 포함하지 않고 paired stress test로 분리한다는 사전등록 판단이 맞다. 이 데이터만으로 Jamo/NFD generality를 주장할 수 없다.

## 7. 제한과 contamination

1. HPLT3의 bucket 10 한 shard만 사용한다. 다른 WDS bucket, 시간대, Korean domain 전체를 대표하지 않는다.
2. HPLT upstream filtering·language identification·dedup 오류를 독립 재현하지 않았다.
3. Exact duplicate 0은 이 selected source 내 byte-identical duplicate에 관한 값이다. near-duplicate나 paraphrase를 배제하지 않는다.
4. Leipzig Wikipedia가 HPLT web crawl과 의미상 겹치지 않는다고 보장할 수 없다. 이를 contamination-free external test라고 부르지 않는다.
5. HPLT dataset package의 CC0 표시는 underlying web content의 모든 권리를 보증하지 않는다. 원문을 재배포하지 않는다.
6. 전체 eligible 6.11GB 중 152M neural bytes만 사용한다. 19.6M model의 publication-scale 충분성은 별도 gate 문제다.

## 8. 재현 및 privacy 판정

Tracked artifact에는 다음만 있다.

- source URL/filename/HTTP metadata/SHA-256
- filter 및 scan counts
- selection salt와 quota
- processed-file aggregate hash/size
- split별 aggregate counts, ratios, digest-set hash
- neural stream count와 codepoint-start diagnostic

다음은 ignored local directory에만 있다.

- 1.86GB compressed archive
- 152MB processed JSONL
- generated local integrity file
- 모든 source/selected document text

따라서 Phase 3 training은 이 표본으로 진행할 수 있다. 다음 구현 단계에서 policy matrix를 만들 때도 patch hash와 aggregate diagnostics만 승격한다.
