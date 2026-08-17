# JamoFlow Phase 0 Audit

> 생성 시각: 2026-08-10T01:53:08.456856+00:00
> 성격: Phase 0 reference boundary audit; neural LM 결과가 아님
> 코퍼스: Leipzig Korean Wikipedia 2021 100K

## 실행 정보

- Python: `3.14.6`
- Platform: `macOS-26.5.2-arm64-arm-64bit-Mach-O`
- Input files: data/processed/leipzig-wikipedia-100k-controls/ko.jsonl
- Resolved files: 1 (suffixes: default text suffixes; plain record unit: line)
- Records: 100000 (train 79746, calibration 10116, test 10138)
- Byte n-gram: order 4, alpha 0.1
- Entropy scoring: 2955.701 ns/byte

## Unicode audit

- Raw bytes: 14,272,643
- Unicode codepoints: 5,947,059
- Bytes/codepoint: 2.400
- Invalid records: 0
- NFC exact records: 100,000/100,000
- NFD exact records: 0/100,000
- Mixed Hangul/CJK/Latin records: 21,244

| Character category | Count |
|---|---:|
| hangul_syllable | 4,118,218 |
| whitespace | 1,268,665 |
| punctuation | 252,575 |
| ascii_latin | 150,473 |
| digit | 124,710 |
| cjk_ideograph | 26,101 |
| symbol | 3,299 |
| other | 2,197 |
| nonascii_latin | 527 |
| hangul_compatibility_jamo | 244 |
| combining_mark | 34 |
| control_or_unassigned | 13 |
| hangul_jamo | 3 |

## Matched-rate boundary results

| Group | Role | Bytes/patch | Boundary H | Oracle capture | Top-budget overlap | Top-decile recall | Mean lag | UTF-8 split | Hangul split | CJK split | Score eval/byte | Policy ns/byte |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_byte_6 | rule | 5.896 | 2.730 | 0.485 | 0.162 | 0.164 | 2.481 | 0.576 | 0.570 | 0.004 | 0.000 | 10.3 |
| fixed_byte_6 | entropy_matched | 5.907 | 5.633 | 1.000 | 1.000 | 1.000 | 0.000 | 0.701 | 0.670 | 0.020 | 0.993 | 36.1 |
| fixed_byte_6 | candidate_entropy_matched | 5.910 | 3.899 | 0.692 | 0.300 | 0.418 | 1.494 | 0.000 | 0.000 | 0.000 | 0.409 | 412.0 |
| fixed_byte_6 | orthographic_candidate_entropy_matched | 5.788 | 3.335 | 0.595 | 0.190 | 0.251 | 2.165 | 0.000 | 0.000 | 0.000 | 0.388 | 816.9 |
| codepoint_stride_6 | rule | 6.438 | 2.298 | 0.399 | 0.097 | 0.124 | 3.017 | 0.000 | 0.000 | 0.000 | 0.000 | 380.5 |
| codepoint_stride_6 | entropy_matched | 6.450 | 5.760 | 1.000 | 1.000 | 1.000 | 0.000 | 0.685 | 0.652 | 0.022 | 0.993 | 35.3 |
| codepoint_stride_6 | candidate_entropy_matched | 6.453 | 4.048 | 0.703 | 0.315 | 0.418 | 1.731 | 0.000 | 0.000 | 0.000 | 0.409 | 411.1 |
| codepoint_stride_6 | orthographic_candidate_entropy_matched | 6.384 | 3.455 | 0.601 | 0.196 | 0.251 | 2.331 | 0.000 | 0.000 | 0.000 | 0.388 | 813.1 |
| spacebyte_compatible | rule | 3.277 | 2.901 | 0.608 | 0.362 | 0.160 | 1.485 | 0.691 | 0.684 | 0.004 | 0.000 | 214.0 |
| spacebyte_compatible | entropy_matched | 3.284 | 4.778 | 1.000 | 1.000 | 1.000 | 0.000 | 0.761 | 0.744 | 0.011 | 0.993 | 45.2 |
| spacebyte_compatible | candidate_entropy_matched | 3.282 | 3.026 | 0.634 | 0.239 | 0.418 | 0.949 | 0.000 | 0.000 | 0.000 | 0.409 | 413.2 |
| spacebyte_compatible | orthographic_candidate_entropy_matched | 3.280 | 2.690 | 0.563 | 0.172 | 0.251 | 1.725 | 0.000 | 0.000 | 0.000 | 0.388 | 810.2 |
| hangul_syllable | rule | 3.380 | 2.136 | 0.443 | 0.127 | 0.162 | 3.910 | 0.000 | 0.000 | 0.000 | 0.000 | 394.1 |
| hangul_syllable | entropy_matched | 3.390 | 4.826 | 1.000 | 1.000 | 1.000 | 0.000 | 0.758 | 0.740 | 0.012 | 0.993 | 45.0 |
| hangul_syllable | candidate_entropy_matched | 3.389 | 3.070 | 0.636 | 0.242 | 0.418 | 0.961 | 0.000 | 0.000 | 0.000 | 0.409 | 411.5 |
| hangul_syllable | orthographic_candidate_entropy_matched | 3.382 | 2.726 | 0.565 | 0.173 | 0.251 | 1.735 | 0.000 | 0.000 | 0.000 | 0.388 | 802.2 |
| eojeol_cap_24 | rule | 9.375 | 2.674 | 0.422 | 0.089 | 0.089 | 4.097 | 0.000 | 0.000 | 0.000 | 0.000 | 416.8 |
| eojeol_cap_24 | entropy_matched | 9.400 | 6.335 | 1.000 | 1.000 | 1.000 | 0.000 | 0.582 | 0.533 | 0.033 | 0.993 | 29.9 |
| eojeol_cap_24 | candidate_entropy_matched | 9.349 | 4.776 | 0.755 | 0.416 | 0.418 | 3.286 | 0.000 | 0.000 | 0.000 | 0.409 | 395.2 |
| eojeol_cap_24 | orthographic_candidate_entropy_matched | 8.754 | 3.835 | 0.617 | 0.239 | 0.251 | 3.284 | 0.000 | 0.000 | 0.000 | 0.388 | 791.1 |

## Record-bootstrap 95% intervals

> Repeats: 500; seed: 1729

| Group | Role | Bytes/patch | Boundary H | Top-decile recall | Mean lag | UTF-8 split | Score eval/byte |
|---|---|---|---|---|---|---|---|
| fixed_byte_6 | rule | [5.894, 5.897] | [2.721, 2.741] | [0.163, 0.166] | [2.474, 2.488] | [0.574, 0.577] | [0.000, 0.000] |
| fixed_byte_6 | entropy_matched | [5.852, 5.958] | [5.620, 5.648] | [1.000, 1.000] | [0.000, 0.000] | [0.696, 0.705] | [0.993, 0.993] |
| fixed_byte_6 | candidate_entropy_matched | [5.868, 5.950] | [3.879, 3.922] | [0.413, 0.424] | [1.471, 1.514] | [0.000, 0.000] | [0.409, 0.410] |
| fixed_byte_6 | orthographic_candidate_entropy_matched | [5.766, 5.811] | [3.325, 3.346] | [0.248, 0.253] | [2.130, 2.203] | [0.000, 0.000] | [0.388, 0.388] |
| codepoint_stride_6 | rule | [6.435, 6.441] | [2.289, 2.308] | [0.122, 0.125] | [3.007, 3.028] | [0.000, 0.000] | [0.000, 0.000] |
| codepoint_stride_6 | entropy_matched | [6.386, 6.510] | [5.747, 5.775] | [1.000, 1.000] | [0.000, 0.000] | [0.680, 0.689] | [0.993, 0.993] |
| codepoint_stride_6 | candidate_entropy_matched | [6.403, 6.500] | [4.027, 4.072] | [0.413, 0.424] | [1.703, 1.758] | [0.000, 0.000] | [0.409, 0.410] |
| codepoint_stride_6 | orthographic_candidate_entropy_matched | [6.359, 6.411] | [3.444, 3.466] | [0.248, 0.253] | [2.293, 2.370] | [0.000, 0.000] | [0.388, 0.388] |
| spacebyte_compatible | rule | [3.275, 3.279] | [2.894, 2.908] | [0.158, 0.163] | [1.467, 1.506] | [0.690, 0.691] | [0.000, 0.000] |
| spacebyte_compatible | entropy_matched | [3.266, 3.301] | [4.766, 4.790] | [1.000, 1.000] | [0.000, 0.000] | [0.758, 0.764] | [0.993, 0.993] |
| spacebyte_compatible | candidate_entropy_matched | [3.271, 3.292] | [3.012, 3.043] | [0.413, 0.424] | [0.938, 0.958] | [0.000, 0.000] | [0.409, 0.410] |
| spacebyte_compatible | orthographic_candidate_entropy_matched | [3.275, 3.285] | [2.683, 2.698] | [0.248, 0.253] | [1.690, 1.762] | [0.000, 0.000] | [0.388, 0.388] |
| hangul_syllable | rule | [3.375, 3.385] | [2.131, 2.142] | [0.159, 0.164] | [3.709, 4.201] | [0.000, 0.000] | [0.000, 0.000] |
| hangul_syllable | entropy_matched | [3.371, 3.408] | [4.814, 4.838] | [1.000, 1.000] | [0.000, 0.000] | [0.755, 0.761] | [0.993, 0.993] |
| hangul_syllable | candidate_entropy_matched | [3.377, 3.399] | [3.055, 3.087] | [0.413, 0.424] | [0.950, 0.970] | [0.000, 0.000] | [0.409, 0.410] |
| hangul_syllable | orthographic_candidate_entropy_matched | [3.375, 3.387] | [2.719, 2.734] | [0.248, 0.253] | [1.699, 1.771] | [0.000, 0.000] | [0.388, 0.388] |
| eojeol_cap_24 | rule | [9.348, 9.404] | [2.660, 2.689] | [0.087, 0.091] | [4.059, 4.135] | [0.000, 0.000] | [0.000, 0.000] |
| eojeol_cap_24 | entropy_matched | [9.261, 9.522] | [6.323, 6.348] | [1.000, 1.000] | [0.000, 0.000] | [0.576, 0.587] | [0.993, 0.993] |
| eojeol_cap_24 | candidate_entropy_matched | [9.235, 9.454] | [4.753, 4.802] | [0.413, 0.424] | [3.221, 3.346] | [0.000, 0.000] | [0.409, 0.410] |
| eojeol_cap_24 | orthographic_candidate_entropy_matched | [8.705, 8.804] | [3.822, 3.848] | [0.249, 0.253] | [3.242, 3.324] | [0.000, 0.000] | [0.388, 0.388] |

## Calibration

| Group | Rule bytes/patch | Full threshold | Full realized | Codepoint threshold | Codepoint realized | Orthographic threshold | Orthographic realized |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_byte_6 | 5.897 | 4.265 | 5.856 | 2.305 | 5.896 | 2.165 | 5.791 |
| codepoint_stride_6 | 6.445 | 4.391 | 6.384 | 2.409 | 6.434 | 2.248 | 6.383 |
| spacebyte_compatible | 3.280 | 3.304 | 3.280 | 1.686 | 3.277 | 1.523 | 3.280 |
| hangul_syllable | 3.384 | 3.360 | 3.383 | 1.727 | 3.384 | 1.581 | 3.382 |
| eojeol_cap_24 | 9.339 | 4.774 | 9.263 | 2.722 | 9.330 | 2.527 | 8.766 |

## 해석 제한

- n-gram predictive entropy는 BLT entropy model의 대체물이 아니라 Phase 0 proxy다.
- `entropy_matched`의 oracle capture와 top-budget overlap은 같은 n-gram entropy score로 boundary와 oracle을 정의하므로 구성상 1이다. 독립적인 성능 증거가 아니다.
- policy runtime은 Python reference implementation 값이며 GPU kernel latency가 아니다.
- UTF-8/Hangul 내부 경계 비율은 표현 경계 진단값이며, 그 자체가 모델 품질 저하를 입증하지 않는다.
- Corpus-specific: Randomized sentence corpus; snapshot year differs from language controls and n-gram entropy remains a proxy.
- matched threshold는 calibration split에서만 정하고 test split에서 고정했다.
