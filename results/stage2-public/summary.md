# Stage 2 Aggregate Summary

> Generated: 2026-08-10T02:06:04.670748+00:00
> Aggregate only; no corpus text or record identifiers are included.

## Corpora

| Language | Records | Raw bytes | Codepoints | Bytes/codepoint |
|---|---:|---:|---:|---:|
| ko | 100,000 | 14,272,643 | 5,947,059 | 2.400 |
| zh | 100,000 | 10,576,852 | 3,824,271 | 2.766 |
| en | 100,000 | 12,743,071 | 12,723,553 | 1.002 |

## Fixed-rate candidate comparison

| Language | Candidate | Bytes/patch | Oracle capture | Top-budget overlap | Top-decile recall | Score eval/byte | Eval reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| ko | codepoint | 5.910 | 0.692 | 0.300 | 0.418 | 0.409 | 0.591 |
| ko | script+delimiter | 5.788 | 0.595 | 0.190 | 0.251 | 0.388 | 0.612 |
| zh | codepoint | 5.828 | 0.884 | 0.475 | 0.478 | 0.353 | 0.647 |
| zh | script+delimiter | 5.844 | 0.833 | 0.366 | 0.348 | 0.315 | 0.685 |
| en | codepoint | 5.844 | 0.997 | 0.991 | 0.988 | 0.991 | 0.009 |

## SpaceByte-compatible structural diagnostic

| Language | Bytes/patch | UTF-8 internal | Hangul internal | CJK internal |
|---|---:|---:|---:|---:|
| ko | 3.277 | 0.691 | 0.684 | 0.004 |
| zh | 3.013 | 0.983 | 0.000 | 0.870 |
| en | 5.974 | 0.005 | 0.000 | 0.000 |

## N-gram sensitivity: fixed-rate codepoint candidate

| Language | Setting | Oracle capture | Top-budget overlap | Top-decile recall |
|---|---|---:|---:|---:|
| ko | order2_alpha0.1 | 0.607 | 0.110 | 0.133 |
| ko | order4_alpha0.01 | 0.637 | 0.155 | 0.186 |
| ko | order4_alpha0.1 | 0.692 | 0.300 | 0.418 |
| ko | order4_alpha1.0 | 0.803 | 0.420 | 0.455 |
| zh | order2_alpha0.1 | 0.712 | 0.091 | 0.133 |
| zh | order4_alpha0.01 | 0.772 | 0.255 | 0.308 |
| zh | order4_alpha0.1 | 0.884 | 0.475 | 0.478 |
| zh | order4_alpha1.0 | 0.987 | 0.479 | 0.480 |
| en | order2_alpha0.1 | 0.999 | 0.997 | 0.995 |
| en | order4_alpha0.01 | 0.999 | 0.996 | 0.994 |
| en | order4_alpha0.1 | 0.997 | 0.991 | 0.988 |
| en | order4_alpha1.0 | 0.999 | 0.991 | 0.986 |

## Patch-rate sensitivity: codepoint candidate

| Language | Setting | Bytes/patch | Oracle capture | Top-budget overlap | Top-decile recall |
|---|---|---:|---:|---:|---:|
| ko | stride4 | 3.957 | 0.650 | 0.251 | 0.418 |
| ko | stride6 | 5.910 | 0.692 | 0.300 | 0.418 |
| ko | stride8 | 7.824 | 0.729 | 0.359 | 0.418 |
| zh | stride4 | 3.913 | 0.820 | 0.470 | 0.478 |
| zh | stride6 | 5.828 | 0.884 | 0.475 | 0.478 |
| zh | stride8 | 7.661 | 0.920 | 0.476 | 0.478 |
| en | stride4 | 3.960 | 0.997 | 0.994 | 0.988 |
| en | stride6 | 5.844 | 0.997 | 0.991 | 0.988 |
| en | stride8 | 7.814 | 0.996 | 0.989 | 0.988 |
