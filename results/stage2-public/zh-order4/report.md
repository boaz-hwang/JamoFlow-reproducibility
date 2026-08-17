# JamoFlow Phase 0 Audit

> 생성 시각: 2026-08-10T01:52:47.441076+00:00
> 성격: Phase 0 reference boundary audit; neural LM 결과가 아님
> 코퍼스: Leipzig Chinese Wikipedia 2018 100K

## 실행 정보

- Python: `3.14.6`
- Platform: `macOS-26.5.2-arm64-arm-64bit-Mach-O`
- Input files: data/processed/leipzig-wikipedia-100k-controls/zh.jsonl
- Resolved files: 1 (suffixes: default text suffixes; plain record unit: line)
- Records: 100000 (train 80106, calibration 9897, test 9997)
- Byte n-gram: order 4, alpha 0.1
- Entropy scoring: 3114.796 ns/byte

## Unicode audit

- Raw bytes: 10,576,852
- Unicode codepoints: 3,824,271
- Bytes/codepoint: 2.766
- Invalid records: 0
- NFC exact records: 100,000/100,000
- NFD exact records: 99,486/100,000
- Mixed Hangul/CJK/Latin records: 17,653

| Character category | Count |
|---|---:|
| cjk_ideograph | 2,984,734 |
| punctuation | 410,174 |
| ascii_latin | 231,804 |
| digit | 147,844 |
| whitespace | 44,338 |
| other | 2,866 |
| symbol | 1,561 |
| nonascii_latin | 760 |
| hangul_syllable | 104 |
| combining_mark | 49 |
| control_or_unassigned | 35 |
| hangul_compatibility_jamo | 2 |

## Matched-rate boundary results

| Group | Role | Bytes/patch | Boundary H | Oracle capture | Top-budget overlap | Top-decile recall | Mean lag | UTF-8 split | Hangul split | CJK split | Score eval/byte | Policy ns/byte |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_byte_6 | rule | 5.888 | 4.770 | 0.641 | 0.192 | 0.188 | 2.554 | 0.198 | 0.000 | 0.169 | 0.000 | 11.8 |
| fixed_byte_6 | entropy_matched | 5.829 | 7.433 | 1.000 | 1.000 | 1.000 | 0.000 | 0.525 | 0.000 | 0.482 | 0.990 | 38.1 |
| fixed_byte_6 | candidate_entropy_matched | 5.828 | 6.574 | 0.884 | 0.475 | 0.478 | 2.084 | 0.000 | 0.000 | 0.000 | 0.353 | 412.4 |
| fixed_byte_6 | orthographic_candidate_entropy_matched | 5.844 | 6.191 | 0.833 | 0.366 | 0.348 | 2.531 | 0.000 | 0.000 | 0.000 | 0.315 | 808.5 |
| codepoint_stride_6 | rule | 5.965 | 4.885 | 0.656 | 0.193 | 0.186 | 2.671 | 0.000 | 0.000 | 0.000 | 0.000 | 380.1 |
| codepoint_stride_6 | entropy_matched | 5.923 | 7.445 | 1.000 | 1.000 | 1.000 | 0.000 | 0.523 | 0.000 | 0.481 | 0.990 | 37.0 |
| codepoint_stride_6 | candidate_entropy_matched | 5.903 | 6.596 | 0.886 | 0.477 | 0.478 | 2.118 | 0.000 | 0.000 | 0.000 | 0.353 | 411.2 |
| codepoint_stride_6 | orthographic_candidate_entropy_matched | 5.919 | 6.214 | 0.835 | 0.367 | 0.348 | 2.559 | 0.000 | 0.000 | 0.000 | 0.315 | 816.8 |
| spacebyte_compatible | rule | 3.013 | 4.233 | 0.633 | 0.207 | 0.113 | 1.619 | 0.983 | 0.000 | 0.870 | 0.000 | 209.7 |
| spacebyte_compatible | entropy_matched | 2.998 | 6.681 | 1.000 | 1.000 | 1.000 | 0.000 | 0.560 | 0.000 | 0.520 | 0.990 | 47.5 |
| spacebyte_compatible | candidate_entropy_matched | 2.994 | 5.151 | 0.771 | 0.440 | 0.478 | 0.977 | 0.000 | 0.000 | 0.000 | 0.353 | 413.5 |
| spacebyte_compatible | orthographic_candidate_entropy_matched | 3.078 | 4.829 | 0.719 | 0.369 | 0.348 | 1.506 | 0.000 | 0.000 | 0.000 | 0.315 | 819.7 |
| cjk_ideograph | rule | 3.437 | 4.918 | 0.716 | 0.353 | 0.318 | 3.693 | 0.000 | 0.000 | 0.000 | 0.000 | 396.6 |
| cjk_ideograph | entropy_matched | 3.416 | 6.865 | 1.000 | 1.000 | 1.000 | 0.000 | 0.539 | 0.000 | 0.497 | 0.990 | 44.9 |
| cjk_ideograph | candidate_entropy_matched | 3.406 | 5.456 | 0.795 | 0.461 | 0.478 | 1.070 | 0.000 | 0.000 | 0.000 | 0.353 | 409.3 |
| cjk_ideograph | orthographic_candidate_entropy_matched | 3.424 | 5.082 | 0.740 | 0.377 | 0.348 | 1.576 | 0.000 | 0.000 | 0.000 | 0.315 | 807.7 |
| eojeol_cap_24 | rule | 15.296 | 4.449 | 0.568 | 0.055 | 0.055 | 9.755 | 0.000 | 0.000 | 0.000 | 0.000 | 423.5 |
| eojeol_cap_24 | entropy_matched | 15.286 | 7.831 | 1.000 | 1.000 | 0.562 | 7.145 | 0.526 | 0.001 | 0.484 | 0.990 | 28.1 |
| eojeol_cap_24 | candidate_entropy_matched | 15.048 | 7.609 | 0.972 | 0.476 | 0.478 | 7.329 | 0.000 | 0.000 | 0.000 | 0.353 | 396.5 |
| eojeol_cap_24 | orthographic_candidate_entropy_matched | 12.497 | 6.766 | 0.869 | 0.334 | 0.350 | 5.018 | 0.000 | 0.000 | 0.000 | 0.315 | 796.5 |

## Record-bootstrap 95% intervals

> Repeats: 500; seed: 1729

| Group | Role | Bytes/patch | Boundary H | Top-decile recall | Mean lag | UTF-8 split | Score eval/byte |
|---|---|---|---|---|---|---|---|
| fixed_byte_6 | rule | [5.886, 5.890] | [4.759, 4.782] | [0.185, 0.190] | [2.545, 2.564] | [0.189, 0.205] | [0.000, 0.000] |
| fixed_byte_6 | entropy_matched | [5.775, 5.885] | [7.431, 7.435] | [1.000, 1.000] | [0.000, 0.000] | [0.522, 0.529] | [0.990, 0.991] |
| fixed_byte_6 | candidate_entropy_matched | [5.780, 5.879] | [6.567, 6.581] | [0.474, 0.482] | [2.052, 2.115] | [0.000, 0.000] | [0.352, 0.354] |
| fixed_byte_6 | orthographic_candidate_entropy_matched | [5.818, 5.871] | [6.184, 6.198] | [0.345, 0.352] | [2.493, 2.567] | [0.000, 0.000] | [0.315, 0.316] |
| codepoint_stride_6 | rule | [5.962, 5.967] | [4.874, 4.895] | [0.184, 0.188] | [2.662, 2.681] | [0.000, 0.000] | [0.000, 0.000] |
| codepoint_stride_6 | entropy_matched | [5.868, 5.981] | [7.443, 7.447] | [1.000, 1.000] | [0.000, 0.000] | [0.519, 0.527] | [0.990, 0.991] |
| codepoint_stride_6 | candidate_entropy_matched | [5.852, 5.955] | [6.589, 6.603] | [0.474, 0.482] | [2.086, 2.150] | [0.000, 0.000] | [0.352, 0.354] |
| codepoint_stride_6 | orthographic_candidate_entropy_matched | [5.891, 5.946] | [6.207, 6.221] | [0.345, 0.352] | [2.522, 2.596] | [0.000, 0.000] | [0.315, 0.316] |
| spacebyte_compatible | rule | [3.009, 3.016] | [4.225, 4.241] | [0.111, 0.115] | [1.599, 1.638] | [0.982, 0.984] | [0.000, 0.000] |
| spacebyte_compatible | entropy_matched | [2.978, 3.018] | [6.677, 6.686] | [1.000, 1.000] | [0.000, 0.000] | [0.557, 0.562] | [0.990, 0.991] |
| spacebyte_compatible | candidate_entropy_matched | [2.984, 3.005] | [5.139, 5.162] | [0.474, 0.482] | [0.967, 0.986] | [0.000, 0.000] | [0.352, 0.354] |
| spacebyte_compatible | orthographic_candidate_entropy_matched | [3.074, 3.082] | [4.820, 4.838] | [0.345, 0.352] | [1.480, 1.531] | [0.000, 0.000] | [0.315, 0.316] |
| cjk_ideograph | rule | [3.428, 3.445] | [4.908, 4.928] | [0.314, 0.322] | [3.263, 4.179] | [0.000, 0.000] | [0.000, 0.000] |
| cjk_ideograph | entropy_matched | [3.393, 3.440] | [6.861, 6.869] | [1.000, 1.000] | [0.000, 0.000] | [0.536, 0.542] | [0.990, 0.991] |
| cjk_ideograph | candidate_entropy_matched | [3.391, 3.423] | [5.445, 5.467] | [0.474, 0.482] | [1.058, 1.081] | [0.000, 0.000] | [0.352, 0.354] |
| cjk_ideograph | orthographic_candidate_entropy_matched | [3.417, 3.429] | [5.073, 5.091] | [0.345, 0.352] | [1.550, 1.602] | [0.000, 0.000] | [0.315, 0.316] |
| eojeol_cap_24 | rule | [15.205, 15.392] | [4.427, 4.468] | [0.053, 0.056] | [9.679, 9.829] | [0.000, 0.000] | [0.000, 0.000] |
| eojeol_cap_24 | entropy_matched | [15.094, 15.494] | [7.831, 7.832] | [0.558, 0.565] | [7.000, 7.303] | [0.520, 0.532] | [0.990, 0.991] |
| eojeol_cap_24 | candidate_entropy_matched | [14.829, 15.277] | [7.606, 7.611] | [0.474, 0.482] | [7.185, 7.483] | [0.000, 0.000] | [0.352, 0.354] |
| eojeol_cap_24 | orthographic_candidate_entropy_matched | [12.428, 12.574] | [6.753, 6.780] | [0.346, 0.353] | [4.959, 5.075] | [0.000, 0.000] | [0.315, 0.316] |

## Calibration

| Group | Rule bytes/patch | Full threshold | Full realized | Codepoint threshold | Codepoint realized | Orthographic threshold | Orthographic realized |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_byte_6 | 5.890 | 6.723 | 5.868 | 4.928 | 5.890 | 4.449 | 5.844 |
| codepoint_stride_6 | 5.964 | 6.741 | 5.963 | 4.966 | 5.964 | 4.499 | 5.914 |
| spacebyte_compatible | 3.009 | 5.305 | 3.009 | 2.811 | 3.009 | −∞ | 3.075 |
| cjk_ideograph | 3.427 | 5.545 | 3.427 | 3.195 | 3.427 | 2.897 | 3.422 |
| eojeol_cap_24 | 15.404 | 7.695 | 15.392 | 7.144 | 15.342 | 6.820 | 12.505 |

## 해석 제한

- n-gram predictive entropy는 BLT entropy model의 대체물이 아니라 Phase 0 proxy다.
- `entropy_matched`의 oracle capture와 top-budget overlap은 같은 n-gram entropy score로 boundary와 oracle을 정의하므로 구성상 1이다. 독립적인 성능 증거가 아니다.
- policy runtime은 Python reference implementation 값이며 GPU kernel latency가 아니다.
- UTF-8/Hangul 내부 경계 비율은 표현 경계 진단값이며, 그 자체가 모델 품질 저하를 입증하지 않는다.
- Corpus-specific: Randomized sentence corpus; snapshot year differs from language controls and n-gram entropy remains a proxy.
- matched threshold는 calibration split에서만 정하고 test split에서 고정했다.
