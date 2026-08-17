# JamoFlow Phase 0 Audit

> 생성 시각: 2026-08-10T01:52:26.156437+00:00
> 성격: Phase 0 reference boundary audit; neural LM 결과가 아님
> 코퍼스: Leipzig English Wikipedia 2016 100K

## 실행 정보

- Python: `3.14.6`
- Platform: `macOS-26.5.2-arm64-arm-64bit-Mach-O`
- Input files: data/processed/leipzig-wikipedia-100k-controls/en.jsonl
- Resolved files: 1 (suffixes: default text suffixes; plain record unit: line)
- Records: 100000 (train 79881, calibration 9960, test 10159)
- Byte n-gram: order 4, alpha 0.1
- Entropy scoring: 2954.449 ns/byte

## Unicode audit

- Raw bytes: 12,743,071
- Unicode codepoints: 12,723,553
- Bytes/codepoint: 1.002
- Invalid records: 0
- NFC exact records: 100,000/100,000
- NFD exact records: 97,403/100,000
- Mixed Hangul/CJK/Latin records: 96

| Character category | Count |
|---|---:|
| ascii_latin | 10,224,818 |
| whitespace | 1,981,004 |
| punctuation | 342,597 |
| digit | 167,176 |
| nonascii_latin | 4,015 |
| symbol | 1,825 |
| other | 1,638 |
| cjk_ideograph | 351 |
| combining_mark | 84 |
| hangul_syllable | 45 |

## Matched-rate boundary results

| Group | Role | Bytes/patch | Boundary H | Oracle capture | Top-budget overlap | Top-decile recall | Mean lag | UTF-8 split | Hangul split | CJK split | Score eval/byte | Policy ns/byte |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_byte_6 | rule | 5.884 | 2.898 | 0.492 | 0.164 | 0.162 | 2.534 | 0.001 | 0.000 | 0.000 | 0.000 | 11.3 |
| fixed_byte_6 | entropy_matched | 5.797 | 5.871 | 1.000 | 1.000 | 1.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.992 | 36.6 |
| fixed_byte_6 | candidate_entropy_matched | 5.844 | 5.861 | 0.997 | 0.991 | 0.988 | 0.049 | 0.000 | 0.000 | 0.000 | 0.991 | 378.1 |
| codepoint_stride_6 | rule | 5.886 | 2.896 | 0.492 | 0.164 | 0.162 | 2.543 | 0.000 | 0.000 | 0.000 | 0.000 | 323.3 |
| codepoint_stride_6 | entropy_matched | 5.797 | 5.871 | 1.000 | 1.000 | 1.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.992 | 36.4 |
| codepoint_stride_6 | candidate_entropy_matched | 5.844 | 5.861 | 0.997 | 0.991 | 0.988 | 0.049 | 0.000 | 0.000 | 0.000 | 0.991 | 376.2 |
| spacebyte_compatible | rule | 5.974 | 4.576 | 0.774 | 0.383 | 0.282 | 2.241 | 0.005 | 0.000 | 0.000 | 0.000 | 163.2 |
| spacebyte_compatible | entropy_matched | 5.992 | 5.915 | 1.000 | 1.000 | 1.000 | 0.000 | 0.009 | 0.000 | 0.000 | 0.992 | 35.7 |
| spacebyte_compatible | candidate_entropy_matched | 5.988 | 5.894 | 0.996 | 0.991 | 0.988 | 0.050 | 0.000 | 0.000 | 0.000 | 0.991 | 378.0 |
| eojeol_cap_24 | rule | 5.486 | 4.648 | 0.802 | 0.483 | 0.350 | 2.082 | 0.000 | 0.000 | 0.000 | 0.000 | 410.2 |
| eojeol_cap_24 | entropy_matched | 5.500 | 5.801 | 1.000 | 1.000 | 1.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.992 | 37.5 |
| eojeol_cap_24 | candidate_entropy_matched | 5.497 | 5.781 | 0.997 | 0.992 | 0.988 | 0.045 | 0.000 | 0.000 | 0.000 | 0.991 | 377.8 |

## Record-bootstrap 95% intervals

> Repeats: 500; seed: 1729

| Group | Role | Bytes/patch | Boundary H | Top-decile recall | Mean lag | UTF-8 split | Score eval/byte |
|---|---|---|---|---|---|---|---|
| fixed_byte_6 | rule | [5.882, 5.886] | [2.888, 2.909] | [0.161, 0.164] | [2.528, 2.542] | [0.001, 0.002] | [0.000, 0.000] |
| fixed_byte_6 | entropy_matched | [5.741, 5.852] | [5.861, 5.881] | [1.000, 1.000] | [0.000, 0.000] | [0.008, 0.009] | [0.992, 0.992] |
| fixed_byte_6 | candidate_entropy_matched | [5.788, 5.900] | [5.852, 5.871] | [0.987, 0.989] | [0.043, 0.055] | [0.000, 0.000] | [0.990, 0.991] |
| codepoint_stride_6 | rule | [5.884, 5.887] | [2.887, 2.907] | [0.160, 0.163] | [2.536, 2.550] | [0.000, 0.000] | [0.000, 0.000] |
| codepoint_stride_6 | entropy_matched | [5.741, 5.852] | [5.861, 5.881] | [1.000, 1.000] | [0.000, 0.000] | [0.008, 0.009] | [0.992, 0.992] |
| codepoint_stride_6 | candidate_entropy_matched | [5.788, 5.900] | [5.852, 5.871] | [0.987, 0.989] | [0.043, 0.055] | [0.000, 0.000] | [0.990, 0.991] |
| spacebyte_compatible | rule | [5.963, 5.986] | [4.570, 4.582] | [0.279, 0.284] | [2.221, 2.259] | [0.004, 0.005] | [0.000, 0.000] |
| spacebyte_compatible | entropy_matched | [5.930, 6.053] | [5.906, 5.926] | [1.000, 1.000] | [0.000, 0.000] | [0.008, 0.010] | [0.992, 0.992] |
| spacebyte_compatible | candidate_entropy_matched | [5.928, 6.047] | [5.884, 5.904] | [0.987, 0.989] | [0.045, 0.057] | [0.000, 0.000] | [0.990, 0.991] |
| eojeol_cap_24 | rule | [5.473, 5.500] | [4.642, 4.656] | [0.348, 0.353] | [2.062, 2.102] | [0.000, 0.000] | [0.000, 0.000] |
| eojeol_cap_24 | entropy_matched | [5.449, 5.549] | [5.792, 5.812] | [1.000, 1.000] | [0.000, 0.000] | [0.007, 0.009] | [0.992, 0.992] |
| eojeol_cap_24 | candidate_entropy_matched | [5.448, 5.546] | [5.772, 5.791] | [0.987, 0.989] | [0.040, 0.050] | [0.000, 0.000] | [0.990, 0.991] |

## Calibration

| Group | Rule bytes/patch | Full threshold | Full realized | Codepoint threshold | Codepoint realized | Orthographic threshold | Orthographic realized |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_byte_6 | 5.885 | 4.603 | 5.769 | 4.603 | 5.815 | — | — |
| codepoint_stride_6 | 5.887 | 4.603 | 5.769 | 4.603 | 5.815 | — | — |
| spacebyte_compatible | 5.956 | 4.609 | 5.956 | 4.604 | 5.954 | — | — |
| eojeol_cap_24 | 5.478 | 4.539 | 5.478 | 4.530 | 5.476 | — | — |

## 해석 제한

- n-gram predictive entropy는 BLT entropy model의 대체물이 아니라 Phase 0 proxy다.
- `entropy_matched`의 oracle capture와 top-budget overlap은 같은 n-gram entropy score로 boundary와 oracle을 정의하므로 구성상 1이다. 독립적인 성능 증거가 아니다.
- policy runtime은 Python reference implementation 값이며 GPU kernel latency가 아니다.
- UTF-8/Hangul 내부 경계 비율은 표현 경계 진단값이며, 그 자체가 모델 품질 저하를 입증하지 않는다.
- Corpus-specific: Randomized sentence corpus; snapshot year differs from language controls and n-gram entropy remains a proxy.
- matched threshold는 calibration split에서만 정하고 test split에서 고정했다.
