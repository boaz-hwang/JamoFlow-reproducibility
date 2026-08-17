# Fresh vocabulary-adaptation Korean data protocol

> 작성일: 2026-08-15
>
> 상태: model-free precommit protocol; train/calibration output 미생성

## 목적

`docs/150`에서 발견한 ordinary dense new-row optimization 후보를 이미 알려진 B1 corpus에서 다시
평가하지 않는다. 같은 pinned HPLT3 Korean shard에서 기존 Phase-3 6,911 documents와 sealed final
test 1,482 documents를 전부 제외한 새 train/calibration stream을 먼저 만든다.

이 단계는 checkpoint, loss, BPB, latency 또는 역할 선택을 읽지 않는다. Tracked artifact에는 원문,
개별 digest, model metric을 넣지 않고 aggregate commitment만 남긴다.

## 고정 source와 quota

- raw source: `10_1.jsonl.zst`, 1,862,302,013 bytes
- raw SHA-256: `de4dfa43...99ca4`
- train: stable `train` bucket에서 정확히 128,000,000 stream bytes
- calibration: stable `calibration` bucket에서 정확히 8,000,000 stream bytes
- sequence length: 512; 각각 250,000 / 15,625 complete sequences
- document eligibility: strict UTF-8, 256--262,144 raw bytes

Stable `test` bucket은 후보가 될 수 없으므로 final-test document와 exact split이 다르다. 추가로
기존 predecessor와 final-test 원문을 model-free code가 메모리에서 digest해 exact SHA-256 및
`NFKC→casefold→whitespace collapse` digest를 모두 제외한다. 따라서 byte-exact뿐 아니라 이 고정
normalization 아래의 중복도 train/calibration에 들어오지 않는다.

Stable `test` 레코드는 source-level normalized representative를 고르기 전에 버린다. 따라서 새
final-test와 같은 bucket의 문서가 나중에 등장하는 train/calibration normalized variant를 선점할 수
없다. 반면 train과 calibration 사이에는 pinned raw-source order에서 처음 등장한 normalized
representative 하나만 남겨 두 split 간 normalized overlap을 막는다.

## 비재량 선택

Split별 rank key는 다음 sealed input으로 유일하게 도출한다.

1. raw source SHA-256
2. historical predecessor output SHA-256
3. sealed final-test full JSONL SHA-256
4. split name과 exact quota
5. protocol version

각 eligible document의 rank는 domain-separated SHA-256이고 `(rank,digest)`가 작은 global prefix를
quota에 처음 도달할 때까지 선택한다. 별도 salt, seed, fallback shard, quota 변경은 없다. Raw shard
내 normalized duplicate는 pinned source order에서 첫 eligible occurrence만 남긴다.

## 출력과 검증

- ignored text: `data/processed/hplt3-korean-vocab-adaptation-v1/ko.jsonl`
- tracked aggregate seal: `data/seals/hplt3-korean-vocab-adaptation-v1.json`
- prepare: `scripts/prepare_hplt3_fresh_adaptation.py`
- independent full rescan: `scripts/verify_hplt3_fresh_adaptation.py`

Seal은 source/predecessor/final identities, exclusion commitments, scan accounting, split별 selected 및
normalized commitments, exact stream hashes/counts와 ignored JSONL hash를 기록한다. Prepare와 verify가
1.7GB shard를 각각 독립 full scan해 byte-identical output과 seal을 요구한다.

Fresh manifest는 predecessor manifest·summary·integrity·processed output과 final-test
manifest·seal·processed output의 파일 SHA-256을 직접 고정한다. Final-test seal을 통한 간접 결속에만
의존하지 않는다.

## 다음 단계 gate

이 seal을 별도 commit한 뒤에만 optimizer-aware vocabulary-adaptation plan을 봉인한다. Fresh train은
candidate와 모든 controls가 같은 raw history로 사용하고 calibration은 one-seed fail-fast 선택에만
사용한다. 기존 historical test와 sealed final test는 이 단계에서 model loss를 계산하지 않는다.

데이터 seal 자체는 method나 actual inference의 양성 증거가 아니다.

## Fable 5 검토의 현재 반영

`fable5-연구-중간-검토.md`의 핵심 경고인 proxy/actual-latency 분리와 rate/placement 인과분리는 후속
W72 실측 및 mechanism controls에서 지지됐다. 그 원칙 때문에 known B1에서 발견한
`update_matched_dense`를 곧바로 방법 결과로 승격하지 않고 이 fresh stream을 만든다.

반대로 S rate 분해, CUDA 확대, speed-negative 소논문 종료를 현재 최우선으로 바꾸지는 않는다.
이들은 과학적으로 가능한 보조 과제지만 실제 효율이라는 성공 기준을 직접 해결하지 않는다. 현재
필요한 최소 수정은 fresh matched-quality 검증이며, 그 결과가 실패하면 이 ordinary-dense optimizer
분기도 종료한다. 즉 Fable 검토는 연구 규율에는 수용하지만 불필요하게 현재 계획을 되돌리는 근거로
사용하지 않는다.
