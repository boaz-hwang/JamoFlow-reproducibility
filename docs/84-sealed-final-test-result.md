# Sealed Korean final-test construction result

> 작성일: 2026-08-12
> 상태: **model loss 비노출 상태에서 생성·전수 재검증 완료**

Pinned HPLT3 Korean raw shard를 두 번 전수 스캔해 새 final-test stream과 aggregate
seal을 만들었다. 준비와 검증 중 checkpoint, loss, BPB, latency는 읽지 않았다.

## 결과

| 항목 | 값 |
|---|---:|
| raw source bytes | 1,862,302,013 |
| eligible raw records | 273,362 |
| historical excluded documents | 6,911 / 6,911 found |
| post-exclusion stable-test candidates | 26,704 |
| selected full documents | 1,482 |
| exact evaluation stream | 32,000,000 bytes |
| 512-byte sequences | 62,500 |
| final full-document overshoot | 14,689 bytes |
| exact historical intersection | 0 |
| NFKC/casefold/whitespace-normalized intersection | 0 |

Source SHA-256는
`de4dfa43fd9f6c62cc81781e09c1f401cc77e7a956e07ecc80ac13477e699ca4`,
evaluation-stream SHA-256는
`562fd60c2abc85e2139feb4ed2f248a4556ace925686f9b94bbeff056ae73f99`다.

Tracked aggregate seal:

- path: `data/seals/hplt3-korean-final-test-v1.json`
- file SHA-256:
  `ce42e8a0b2d8161cc59e0b30d5d121b547e22d28709fe48284aa777df4a2290b`
- canonical payload SHA-256:
  `97cf90d1e6e7191e7f8336647f278ae6c0e82d70540bf0f5c43f9cb426e75dc8`
- preparation code commit: `5928bbcc8660db0f4ed85762f88808974b32fae2`

Ignored local JSONL의 SHA-256는
`098ae8b833a1498689dae1d60341aa870fce51c7f9dde6d961c867f751ee3dc2`다.
원문, 개별 document digest/rank와 model metric은 tracked seal에 없다.

## 해석 경계

Zero intersection은 exact byte 및 고정한 format-normalization 기준이다. 부분 복제,
near duplicate 또는 semantic contamination 부재를 뜻하지 않는다. Final stream은
raw-byte LM 정의에 따라 마지막 UTF-8 codepoint나 문서 중간에서 끝날 수 있다.

이 seal이 commit되기 전에는 selection plan이나 final loss 계산을 허용하지 않는다.
다음 단계는 이 commit을 ancestor로 요구하는 calibration-only selection plan을 먼저
봉인하는 것이다.
