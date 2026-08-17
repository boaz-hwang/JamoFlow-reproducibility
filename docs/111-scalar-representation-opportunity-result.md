# Scalar representation opportunity result and BPE constraint

> 작성일: 2026-08-14
>
> protocol commit: `155c4ad1e5576358faa26fcdb051cc584fda275a`
>
> authoritative aggregate:
> `results/scalar-representation-opportunity-v1/summary.json`

## 1. 판정

고정한 model-free gate는 모두 통과했다. 따라서 다음 **parameter-matched random-weight
construction 및 actual-runtime feasibility**를 열 수 있다. 이는 학습, quality,
matched-quality speed 또는 novelty 통과가 아니다.

| representation | 8MB sequential units | raw byte 대비 감소 |
|---|---:|---:|
| raw byte | 8,000,000 | — |
| generic Unicode scalar | 3,330,977 | 58.363% |
| Hangul scalar / otherwise byte hybrid | 3,392,568 | 57.593% |
| train-only ByteLevel BPE 16K | 1,533,938 | 80.826% |
| train-only ByteLevel BPE 32K | 1,388,745 | 82.641% |

Summary file SHA-256은
`5bd1fce05e209842189580e32328ec383c0741d15cbc22519fb023268ccb8c0a`이고,
내부 canonical summary SHA-256은
`7c60212d640d1ea8521183eac5a8daea8a03584bf64c1b3ad71e480619070bcd`다.

## 2. 가장 중요한 새 제약: BPE가 훨씬 짧다

BPE32K는 calibration complete-scalar prefix에서 1,388,745 token, 평균 5.761
bytes/token이었다. Generic scalar는 3,330,977 step, hybrid는 3,392,568 step이므로 각각
BPE32K의 약 2.40배와 2.44배다. BPE16K도 1,533,938 token으로 두 scalar 후보보다 훨씬
짧다.

따라서 연구 방향을 다음처럼 보완한다.

1. scalar/hybrid를 `BPE보다 짧은 sequence`라고 주장하지 않는다.
2. BPE를 publication-scale 마지막 comparator로 미루지 않고 바로 다음 random-weight
   runtime feasibility에 포함한다.
3. scalar candidate가 성립하려면 작은 conditional head와 BLT local/global hierarchy가
   BPE의 더 짧은 sequence를 wall time에서 상쇄해야 한다.
4. compact random-weight 조건에서도 BPE frontier와 경쟁할 가능성이 없으면 scalar branch를
   학습하지 않는다.

이는 계획을 불필요하게 갈아엎는 변경이 아니다. 원래 protocol도 BPE를 필수 comparator로
고정했지만, observed 2.4× unit gap 때문에 그 비교 시점을 앞으로 당긴 것이다.

## 3. Generic scalar와 Hangul hybrid의 관계

Hybrid는 generic scalar보다 61,591 step, 1.849% 길다. 반면 resident conditional-output
row 합은 generic 448, hybrid 324로 hybrid가 27.68% 작다. Local width 192의 단일 projection
parameter로는 86,016 대 62,208이다.

고정 W72의 5.640B dense-matmul FLOPs/512 raw bytes와 비교한 opportunity estimate는 다음과
같다.

- generic scalar: 3.575B, 36.622% 감소
- Hangul hybrid: 3.595B, 36.252% 감소

두 차이는 작고 식에는 conditional kernel/dispatch/cache가 없다. 따라서 현재 데이터는 어느
쪽이 실제로 빠를지 말해 주지 않는다. Hybrid의 한국어-specific 가치는 sequence reduction이
아니라 다음 조합에 있다.

- 대부분의 Korean scalar savings를 유지
- 더 작은 conditional head
- arbitrary/non-Hangul byte fallback 유지
- `L→V→T` compositional prior가 matched quality를 더 적은 capacity로 보존할 가능성

이 중 마지막은 아직 가설이다. EACL 2023의 conditional three-hot이 이미 방법적 선례이므로,
hybrid가 generic scalar를 실제 speed/quality에서 이기지 못하면 Hangul-specific contribution은
없다.

## 4. Vocabulary와 OOV

Train documents에는 newline separator를 포함해 53,386,833 scalar occurrence와 7,006 unique
scalar가 있었다. 그중 precomposed Hangul은 2,298 types, non-Hangul은 4,708 types다.

Calibration prefix의 3,330,976 complete scalar 중 train vocabulary에 없던 것은 149회,
0.004473%였고, unseen Hangul은 38회, Hangul occurrence의 0.001650%였다. 이는 작은 scalar
vocabulary가 Korean in-domain corpus를 거의 덮는다는 진단이지만, OOD·Unicode robustness를
증명하지 않는다. Raw fallback은 여전히 필수다.

Flat train-scalar + raw-fallback head는 `7,006+256=7,262` rows이고 local width 192에서 약
1.394M projection parameters다. Factorized hybrid는 324 rows, 62,208 parameters다. 다만
flat scalar도 fallback alias를 막는 canonical transducer가 필요하며 이 표는 embedding tying,
conditional dependencies 또는 total graph parameter를 포함하지 않는다.

## 5. Conditional factorization의 필요성

Train Hangul 36,849,780 observations에서 empirical joint entropy는 8.0882 bits였다.

- `H(L) = 3.5030`
- `H(V|L) = 3.0177`
- `H(T|L,V) = 1.5676`
- 합 = joint 8.0882 bits

독립 marginal entropy의 합은 joint보다 1.2602 bits 컸다. Calibration에서도 1.2635 bits로
거의 같았다. 이는 `L/V/T`가 독립이라는 가정이 distribution 수준에서 틀렸음을 확인한다.
반면 conditional chain은 chain rule로 joint를 정확히 표현할 수 있다.

이 수치는 contextual neural model의 BPB 이득이 아니다. 다만 독립 three-head를 새 후보로
열 이유가 없고, EACL 2023과 동일하게 conditional chain이 최소한의 올바른 control임을
보인다.

## 6. BPE artifact 검증

16K와 32K tokenizer는 train split만으로 각각 두 번 독립 학습했다.

- 두 replicate의 compact tokenizer JSON bytes exact 일치
- `tokenizers==0.22.2`
- normalizer 없음
- full 256-byte alphabet, no added/special token
- calibration decode exact roundtrip
- token raw-byte concatenation이 calibration bytes와 exact 일치

Ignored tokenizer SHA-256은 다음과 같다.

- BPE16K: `98b626d6b268773d4fe599ac1ae8f2869d18edfb33566ba1d04bd84bdce3c263`
- BPE32K: `a6181de532b4134ceb40d9a11a71fc1140d7b47533cd00d1d241af26069c9851`

Tokenizer JSON은 corpus-derived string을 포함할 수 있으므로 tracked Git에 올리지 않는다.

## 7. 다음 단계의 정확한 질문

다음은 representation별 모델 품질을 보지 않는 construction/runtime preflight다. 같은
raw-byte case와 hardware에서 최소 다음을 구현한다.

1. 기존 byte W72 actual runtime
2. generic conditional UTF-8 scalar
3. Hangul conditional L/V/T hybrid
4. reversible ByteLevel BPE16K/32K token Transformer

비교는 단순 같은 hidden width가 아니라 total resident parameter와 raw-byte output horizon을
함께 맞춘다. Random-weight 단계에서는 의미 있는 free-running Korean generation을 만들 수
없으므로, exact 128 raw-byte continuation의 route/길이를 고정하되 각 conditional head의
device-side argmax dependency를 실제 실행한다. 모든 incremental cache는 full-prefix oracle과
일치해야 한다. Tokenization/unit encoding은 timing 밖이며 runtime/cache construction, prefill,
output head, incremental decode와 synchronization은 timing 안이다.

Random weights는 quality evidence가 아니므로 다음 질문만 답한다.

- 실제 MPS에서 conditional micro-head가 sequential local Transformer 감소를 먹어 치우는가?
- hybrid가 generic scalar보다 작은 head로 runtime frontier를 개선하는가?
- 어느 scalar 경로라도 BPE의 더 짧은 token sequence와 경쟁 가능한가?

유망한 구조만 동일 train/calibration budget의 한 seed 학습을 연다. 이후 noninferiority와
actual E2E를 모두 통과해야 나머지 seeds 또는 큰 모델로 확장한다.

정확한 parameter geometry, case filter와 결과 전 decision rule은
`docs/112-scalar-runtime-preflight-protocol.md`에 고정한다.

## 8. Claim 경계

현재 안전한 결론은 다음뿐이다.

> On an 8 MB Korean calibration stream, reversible Unicode-scalar and
> Hangul-hybrid representations would remove about 58% of byte-level main
> steps, and a transparent BLT dense-matmul opportunity model estimated about
> 36% lower cost than W72. However, train-only ByteLevel BPE used only 1.39–1.53
> million units versus 3.33–3.39 million scalar units, so any scalar-BLT
> efficiency claim must beat a much shorter BPE sequence in measured,
> parameter-matched inference.

Actual latency, matched quality, memory, OOD robustness, Korean-specific superiority 및
publication-grade efficiency는 아직 증명되지 않았다.
