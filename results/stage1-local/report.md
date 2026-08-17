# JamoFlow Phase 0 Audit

> 생성 시각: 2026-08-10T01:20:12.639519+00:00
> 성격: reference tooling smoke test; neural LM 결과가 아님
> 코퍼스: JamoFlow repository documents

## 실행 정보

- Python: `3.14.6`
- Platform: `macOS-26.5.2-arm64-arm-64bit-Mach-O`
- Input files: docs/00-topic-selection.md, docs/01-verification-report.md, docs/02-critical-research-direction-review.md, docs/03-citation-verification.md, docs/04-phase0-research-protocol.md
- Resolved files: 5 (suffixes: default text suffixes; plain record unit: line)
- Records: 961 (train 762, calibration 78, test 121)
- Byte n-gram: order 4, alpha 0.1
- Entropy scoring: 1673.210 ns/byte

## Unicode audit

- Raw bytes: 95,587
- Unicode codepoints: 63,407
- Invalid records: 0
- NFC exact records: 961/961
- NFD exact records: 188/961
- Mixed Hangul/CJK/Latin records: 622

| Character category | Count |
|---|---:|
| ascii_latin | 26,932 |
| hangul_syllable | 15,579 |
| whitespace | 11,336 |
| punctuation | 5,581 |
| digit | 2,966 |
| symbol | 974 |
| hangul_compatibility_jamo | 18 |
| other | 12 |
| combining_mark | 7 |
| nonascii_latin | 2 |

## Matched-rate boundary results

| Group | Role | Bytes/patch | Boundary H | Oracle capture | Top-budget overlap | Top-decile recall | Mean lag | UTF-8 split | Hangul split | Score eval/byte | Policy ns/byte |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_byte_6 | rule | 5.828 | 6.361 | 0.808 | 0.156 | 0.157 | 2.503 | 0.328 | 0.314 | 0.000 | 10.8 |
| fixed_byte_6 | entropy_matched | 5.310 | 7.861 | 1.000 | 1.000 | 1.000 | 0.000 | 0.332 | 0.310 | 0.988 | 33.0 |
| fixed_byte_6 | candidate_entropy_matched | 5.425 | 7.798 | 0.992 | 0.668 | 0.677 | 2.766 | 0.000 | 0.000 | 0.665 | 381.1 |
| codepoint_stride_6 | rule | 6.172 | 6.339 | 0.805 | 0.128 | 0.130 | 2.821 | 0.000 | 0.000 | 0.000 | 354.4 |
| codepoint_stride_6 | entropy_matched | 5.310 | 7.861 | 1.000 | 1.000 | 1.000 | 0.000 | 0.332 | 0.310 | 0.988 | 33.4 |
| codepoint_stride_6 | candidate_entropy_matched | 5.425 | 7.798 | 0.992 | 0.668 | 0.677 | 2.766 | 0.000 | 0.000 | 0.665 | 394.9 |
| spacebyte_compatible | rule | 4.084 | 6.081 | 0.778 | 0.182 | 0.172 | 1.966 | 0.447 | 0.437 | 0.000 | 188.6 |
| spacebyte_compatible | entropy_matched | 3.688 | 7.797 | 1.000 | 1.000 | 1.000 | 0.000 | 0.335 | 0.317 | 0.988 | 40.5 |
| spacebyte_compatible | candidate_entropy_matched | 4.182 | 7.722 | 0.987 | 0.670 | 0.677 | 2.052 | 0.000 | 0.000 | 0.665 | 386.2 |
| hangul_syllable | rule | 5.967 | 6.162 | 0.783 | 0.098 | 0.100 | 14.376 | 0.000 | 0.000 | 0.000 | 359.6 |
| hangul_syllable | entropy_matched | 5.310 | 7.861 | 1.000 | 1.000 | 1.000 | 0.000 | 0.332 | 0.310 | 0.988 | 33.4 |
| hangul_syllable | candidate_entropy_matched | 5.425 | 7.798 | 0.992 | 0.668 | 0.677 | 2.766 | 0.000 | 0.000 | 0.665 | 390.1 |
| eojeol_cap_24 | rule | 5.545 | 6.144 | 0.781 | 0.162 | 0.164 | 3.200 | 0.000 | 0.000 | 0.000 | 412.2 |
| eojeol_cap_24 | entropy_matched | 5.310 | 7.861 | 1.000 | 1.000 | 1.000 | 0.000 | 0.332 | 0.310 | 0.988 | 32.6 |
| eojeol_cap_24 | candidate_entropy_matched | 5.425 | 7.798 | 0.992 | 0.668 | 0.677 | 2.766 | 0.000 | 0.000 | 0.665 | 402.7 |

## Calibration

| Group | Rule bytes/patch | Full entropy threshold | Full realized | Candidate threshold | Candidate realized |
|---|---:|---:|---:|---:|---:|
| fixed_byte_6 | 5.854 | 7.774 | 5.347 | 7.623 | 5.650 |
| codepoint_stride_6 | 6.245 | 7.774 | 5.347 | 7.623 | 5.650 |
| spacebyte_compatible | 4.221 | 7.623 | 3.863 | 7.356 | 4.221 |
| hangul_syllable | 6.048 | 7.774 | 5.347 | 7.623 | 5.650 |
| eojeol_cap_24 | 5.754 | 7.774 | 5.347 | 7.623 | 5.650 |

## 해석 제한

- n-gram predictive entropy는 BLT entropy model의 대체물이 아니라 Phase 0 proxy다.
- `entropy_matched`의 oracle 지표는 같은 n-gram entropy score로 boundary와 oracle을 정의하므로 구성상 1에 가깝다. 독립적인 성능 증거가 아니다.
- policy runtime은 Python reference implementation 값이며 GPU kernel latency가 아니다.
- UTF-8/Hangul 내부 경계 비율은 표현 경계 진단값이며, 그 자체가 모델 품질 저하를 입증하지 않는다.
- Corpus-specific: Repository research notes are not a representative Korean corpus.
- matched threshold는 calibration split에서만 정하고 test split에서 고정했다.
