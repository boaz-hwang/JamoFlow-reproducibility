# Fresh-v2 Korean data protocol for the 16K vocabulary test

> 작성일: 2026-08-15
>
> 상태: model-free precommit protocol; output·seal 미생성

## 결정

8K `dense8k_update_geometry`는 fresh-v1 calibration BPB에서 2K anchor와 두 강한 8K
대조군을 이겼지만, 실제 free-running end-to-end 지연 개선은 `8.84%`이고 bootstrap
하한은 `-0.17%`였다. 따라서 같은 데이터에서 vocabulary 크기만 다시 바꾸거나,
controlled 결과만 근거로 16K를 실행하지 않는다.

16K 실험은 fresh-v1 자체를 완전히 제외한 두 번째 한국어 train/calibration corpus에서
수행한다. 이 단계는 checkpoint, tokenizer, loss, BPB, output text, latency 또는 role 선택을
읽지 않는다. 데이터가 만들어지기 전에 exact source, exclusions, quota, rank-key 도출 및
검증 코드를 고정한다.

## 고정 source와 배제 집합

Raw source는 이전 단계와 동일한 pinned HPLT3 Korean shard다.

- archive: `10_1.jsonl.zst`
- bytes: `1,862,302,013`
- SHA-256: `de4dfa43...99ca4`
- eligibility: strict UTF-8, document raw bytes `256..262,144`
- stable split: SHA-256 prefix modulo 10,000의 기존 train/calibration/test 구획

다음 세 집합을 메모리에서 원문으로 다시 해시해 전부 제외한다.

| source | exact documents |
|---|---:|
| historical Phase-3 corpus | 6,911 |
| sealed final test | 1,482 |
| fresh-v1 train + calibration | 6,076 |
| total | 14,469 |

각 문서는 byte-exact SHA-256뿐 아니라
`NFKC → casefold → whitespace collapse → SHA-256` identity로도 제외한다. 세 집합 사이
exact 또는 normalized overlap이 있거나, raw shard에서 14,469개 exact identity를 전부 다시
찾지 못하면 중단한다. Fresh-v1은 manifest, protocol source, seal, payload, ignored JSONL의
파일 해시를 직접 고정하고, 5,692 train + 384 calibration 문서의 split order와 128MB/8MB
stream hash까지 재검증한다.

## 비재량 selection

Quota와 sequence geometry는 fresh-v1과 동일하다.

- train: 정확히 `128,000,000` bytes = 250,000 sequences
- calibration: 정확히 `8,000,000` bytes = 15,625 sequences
- sequence length: 512 bytes

각 split의 rank key는 다음 identity에서 유일하게 도출한다.

1. raw archive SHA-256
2. Phase-3 processed corpus SHA-256
3. sealed final-test JSONL SHA-256
4. fresh-v1 JSONL SHA-256
5. split name, exact quota, protocol version 2

고정된 domain-separated rank는 다음과 같다.

- train: `eff3b43937f982d12e6bcc304a742ed53dd2f227fb27e75ec6e3990baaa8d418`
- calibration: `a09677ea7f9c271b014a3246285188dba389a0e01082e34da0ce41613b3c9d65`

각 stable train/calibration 후보를 `(rank_digest, document_digest)` 순으로 정렬하고 joined
stream이 quota에 처음 도달하는 최소 full-document prefix를 선택한다. Stable test 문서는
normalized representative를 정하기 전에 버린다. 별도 salt, 임의 seed, fallback source,
quota 축소 또는 결과 기반 재선택은 없다.

## Artifact와 fail-closed 검증

- manifest: `data/manifests/hplt3-korean-vocab-adaptation-v2.json`
- ignored text: `data/processed/hplt3-korean-vocab-adaptation-v2/ko.jsonl`
- tracked aggregate seal: `data/seals/hplt3-korean-vocab-adaptation-v2.json`
- prepare: `scripts/prepare_hplt3_fresh_adaptation_v2.py`
- independent verifier: `scripts/verify_hplt3_fresh_adaptation_v2.py`

Prepare는 clean committed worktree만 허용하고, 과거 Git history에 v2 seal이 있으면 재발행을
거부한다. Seal에는 source/dependency identity, exclusion aggregate commitments, 전체 scan
accounting, split별 selected·normalized commitments, exact stream hashes/counts, ignored JSONL
hash만 기록한다. 원문과 개별 digest, model metric은 tracked artifact에 넣지 않는다.

Verifier는 기존 selected list를 신뢰하지 않고 1.7GB raw shard를 다시 처음부터 읽어 output과
seal의 byte equality를 요구한다. Preparation modules는 NumPy/PyTorch/tokenizer/model stack과
model-result artifact를 import하거나 읽을 수 없다.

## 보장하지 않는 것

- 동일 raw shard의 disjoint sample이지 새로운 source domain이 아니다.
- Exact와 고정 normalization 중복만 배제한다. 일반 near-duplicate, semantic overlap 또는
  웹 원천 오염 부재를 주장하지 않는다.
- 새로운 data seal은 16K quality나 actual efficiency의 양성 증거가 아니다.
- Fresh-v1 결과를 보고 16K 방향을 선택했다는 adaptivity는 숨기지 않는다. 다만 v2 문서와
  16K 결과는 분리하고, v2 안에서는 fresh-v1 문서를 다시 쓰지 않는다.

## 다음 gate

V2 seal을 독립 full rescan으로 확인하고 별도 commit한 뒤에만 16K 학습 계약을 봉인한다.
한 seed에서 다음 다섯 역할이 같은 v2 stream을 사용한다.

1. `dense2k_joint_v2`: cross-data 2K anchor
2. `dense8k_update_geometry_v2`: previously fixed 8K recipe의 cross-data replication
3. `dense16k_standard_joint`: ordinary joint-training control
4. `dense16k_inplace_two_stage`: literature-aligned expansion control
5. `dense16k_update_geometry`: fixed 8K multipliers를 무조정 재사용한 candidate

16K candidate는 min(2K, 8K) anchor에 대한 `+0.010 BPB` non-inferiority와 두 16K control에
대한 최소 `0.002 BPB` 우위를 모두 통과해야 한다. 이 quality gate가 실패하면 actual timing은
열지 않는다. 통과하더라도 controlled와 free-running 실제 end-to-end가 각각 `>=10%` 개선되고
불확실성 gate를 통과해야만 multi-seed 확증으로 진행한다.
