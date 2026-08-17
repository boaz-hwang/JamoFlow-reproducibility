# Byte-LengthGain-2K 결과와 상수 비용 조합 헤드 전환

> 작성일: 2026-08-14
>
> 상태: sealed calibration-development result; same-2K tokenizer branch 종료

## 결론

Train split의 실제 비중첩 token saving을 224 rounds 동안 직접 최적화한
Byte-LengthGain-2K도 사전 고정한 10% opportunity gate를 통과하지 못했다.

- 배포 가능한 left-most-longest는 BPE-2K보다 calibration token이 **1.141% 더 많았다.**
- 같은 vocabulary에서 가능한 minimum-token DP는 calibration token을 **4.153%**, 사전 고정한
  36 continuation token을 **5.668%** 줄였다.
- 두 수치는 10% 기준의 절반 안팎이며, exact byte roundtrip만 통과했다.

따라서 batch size, score, train prefix, maximum piece length 또는 segmentation을 결과에 맞춰
다시 고르지 않는다. Korean-complete eligibility variant도 열지 않고 **same-2K tokenizer
construction branch를 종료한다.** 다음 실험은 기존 BPE-8K/16K/32K가 이미 제공한 짧은
sequence를 사용하되, 큰 dense embedding/output matrix가 작은 모델의 Transformer capacity와
매-step latency를 잠식하지 않도록 head를 factorize하는 방향이다.

이 전환은 positive 결과를 만들기 위한 사후 완화가 아니다. 10% gate와 실패 시
factorized-large-vocabulary로 이동한다는 규칙은 calibration 결과를 보기 전에
docs/130에 고정되어 있었다.

## 봉인된 결과

공통 calibration complete UTF-8 prefix는 7,999,999 bytes이고, incomplete suffix 1 byte는
모든 역할에서 동일하게 제외했다. Continuation 수치는 6 warmup 뒤 36 measured document
cases의 합이다.

| role | calibration tokens | BPE 대비 감소 | measured continuation tokens | BPE 대비 감소 | encode MB/s |
|---|---:|---:|---:|---:|---:|
| Byte-BPE-2K | 2,263,476 | 기준 | 1,288 | 기준 | 6.386 |
| LengthGain left-most-longest | 2,289,304 | **-1.141%** | 1,279 | 0.699% | **8.112** |
| LengthGain minimum-token DP | 2,169,463 | **4.153%** | 1,215 | **5.668%** | 5.702 |

Minimum-token DP의 measured prompt와 joint 합은 각각 1,245와 2,446으로, BPE의 1,293과
2,564보다 짧았다. 방향은 일관되지만 효과 크기가 부족하다. Left-most-longest의 encode
throughput이 가장 빠르다는 사실도 model의 autoregressive step 수 증가를 상쇄하지 못하므로
selection에 사용하지 않았다.

Train-only constructor는 raw byte 8,000,000개에서 시작해 224 rounds 뒤 2,161,188개 token을
만들었다. Worker와 verifier는 전체 construction을 각각 독립 실행했고 ordered pieces,
round trace, final token IDs를 exact 비교했다. Verifier replay와 evaluation은 2,556.885초였다.

- plan payload SHA-256: `6cefffdbcdb0df7d1ce42d9823c8fdccdb2534bfa2aff6ee4e9a582cdadd4592`
- ordered pieces SHA-256: `8932cf344cc94bc8581482985fe9cdd4458451de189e26752251882ebf90a073`
- train token IDs SHA-256: `b5bf2d0e9cde25c2ccac551d9ed791b63f36b13a465266c1d1b376556b2ab6ba`
- result payload SHA-256: `01ad167fb340fba466e206b67c5f3519f2f41152b8b9e824c2660290c3c5c5c0`
- result file SHA-256: `1facd71efdcd70ea96c14feaf7291314d5ab5fa0ae295bb953fd7c4ef9406c24`

## 결과가 말해 주는 것

### 1. Same-2K에서 남은 token-count 여유가 10%보다 작다

Minimum-token DP는 고정 vocabulary에 대한 token-count 최적 분할이다. 이 역할조차 calibration
4.153%, continuation 5.668%에 그쳤다. 현재 vocabulary에서 다른 deterministic segmentation을
사용해 10%를 만드는 것은 불가능하다. Left-most-longest와의 5.294%p calibration 차이는
segmentation도 중요하다는 뜻이지만, 이상적인 DP조차 gate를 넘지 못한다.

### 2. Train saving objective의 일반화 격차가 크다

Constructor는 현재 train segmentation을 매 round 다시 계산하고 exact non-overlapping saving을
직접 최적화했다. 그런데 frozen calibration에서 이득이 작아졌다. 이는 후보가 train prefix의
반복 substring에 과도하게 맞고, BPE의 hierarchical merge가 제공하는 더 넓은 조합 재사용을
2,048 direct pieces가 대체하지 못했음을 시사한다. 이 해석은 인과적으로 확정된 것은 아니지만,
동일 objective를 다른 이름의 trainer로 반복할 근거는 약해졌다.

### 3. 구조적으로 그럴듯한 vocabulary 통계는 systems opportunity가 아니다

LengthGain vocabulary는 BPE보다 strict-UTF-8 multibyte piece가 1,642 대 1,520,
Hangul-containing piece가 1,541 대 1,433으로 많고, 최대 사용 piece도 31 대 13 bytes였다.
공백을 넘는 multibyte piece도 124개였다. 그런데 실제 token count는 left-most-longest에서
악화됐다. UTF-8 완결성, Hangul 포함 여부, 긴 piece 수, cross-eojeol 수를 독립 목적함수로
삼아서는 안 된다는 앞선 부정 결과가 다시 확인됐다.

### 4. 한국어 제약은 generic 실패의 구제책이 될 수 없다

Generic constructor가 10%의 speed ceiling을 만들지 못한 뒤 Korean constraint를 추가하면 후보
집합은 더 좁아진다. 품질 prior는 생길 수 있지만 token-count ceiling을 사후에 높인다고 기대할
근거가 없다. 따라서 docs/130의 조건대로 Korean-complete tokenizer를 실행하지 않는다.

## 선행연구를 반영한 다음 가설

큰 vocabulary, factorized embedding, compositional output 어느 하나도 독립적인 신규성은 아니다.

- ALBERT는 vocabulary embedding과 hidden dimension을 factorize했다.
- Adaptive Input/Softmax와 DeFINE은 큰 vocabulary의 입력·출력 비용을 빈도 또는 저차원 구조로
  줄였다.
- Deep Compositional Code Learning과 DPQ는 여러 codebook으로 token embedding을 압축했다.
- Grounded Compositional Outputs는 lexical structure에서 output embedding을 합성했다.
- Korean three-hot은 한 음절을 초성·중성·종성으로 factorize해 syllable step을 유지하면서
  embedding parameters를 99.6% 줄였다.
- SCRIPT는 subword embedding에 Jamo compositional representation을 주입했다.
- zip2zip은 dynamic hypertoken embedding/unembedding과 15--40% sequence reduction, 최대 40%
  end-to-end latency 개선을 이미 보고했다.
- 2026 vector-index output embedding은 compact CPU LM의 dense output projection을 approximate
  MIPS로 대체해 큰 batch-1 이득을 보고했다.

따라서 다음의 잠정적 연구 가설은 더 좁다.

> **동일한 2K dense head parameter budget과 동일한 Transformer body에서, 큰 Korean BPE
> vocabulary의 token embedding과 exact full-vocabulary logits를 고정 크기 codebook으로
> 합성하면, 8K 이상 vocabulary의 짧은 sequence를 유지하면서 raw-byte quality와 Apple-MPS
> batch-1 generation latency를 동시에 개선할 수 있다. 그중 Hangul onset/vowel/coda assignment가
> compute-identical generic 및 shuffled assignments보다 quality를 보존한다면 한국어 특화 기여가
> 성립한다.**

핵심은 `compositional embedding이 새롭다`가 아니라 다음 네 조건의 교집합이다.

1. 2K와 같은 Transformer body 및 같은 trainable head parameter 수
2. 큰 Korean byte-BPE vocabulary가 제공하는 실제 autoregressive step 감소
3. dense `V × d` unembedding을 하지 않는 exact full-vocabulary head
4. 일반 factorization control을 넘는 Hangul-specific quality contribution과 실제 E2E 10% 개선

## 제안 아키텍처: Hangul-compositional codebook head

개발 기준인 BPE-2K×8L은 hidden 384, FFN 1,536, 8 layers이고 tied head가
`2,048 × 384 = 786,432` parameters다. 이 matrix를 `16 × 128 × 384` codebook으로 바꾸면
trainable head parameters가 정확히 같다. Transformer body를 포함한 총계도 19,667,328로
동일하다.

각 큰-vocabulary token은 16개 code index를 갖고, dense-equivalent tied embedding을 다음처럼
정의한다.

```text
E(token) = (1 / sqrt(16)) * sum_m Codebook[m, code(token, m)]
logit(token | h) = dot(h, E(token))
```

출력은 먼저 hidden과 2,048 code vectors의 logits를 계산한 뒤 token별 16개 값을 gather-add한다.
즉 exact full-vocabulary logit과 standard cross-entropy를 유지하면서 dense `V × 384` matmul을
피한다. Approximate retrieval이나 invalid-token pruning을 primary mechanism으로 쓰지 않는다.
Assignment buffer와 gather-add 비용은 parameter count 밖으로 숨기지 않고 실제 timer와 memory에
포함한다.

### Compute-identical assignment controls

같은 tokenizer, model graph, parameter 수, initialization, train order를 사용한다.

1. **generic Unicode-surface code**: token의 첫·마지막 Unicode scalar와 residual token identity를
   result-blind deterministic code로 표현한다.
2. **shuffled-Hangul control**: Hangul assignment의 slot별 분포와 byte-length strata를 보존하되
   token 간 surface assignment를 deterministic하게 permutation한다.
3. **Hangul code**: 첫·마지막 완성형 Hangul syllable을 onset/vowel/coda로 분해하고, non-Hangul과
   incomplete UTF-8 pieces는 generic fallback을 쓴다.
4. **tied low-rank projection**: ALBERT/adaptive-input 계열과 가까운 표준 factorization control이다.
5. **same-body dense large vocabulary**: parameter budget은 더 크지만 factorization의 quality ceiling과
   head latency tax를 보여 주는 mechanism control이다.

Residual identity code가 전체 token tuple의 uniqueness를 보장한다. 따라서 Hangul code는 token을
동일 embedding으로 collapse하지 않으며, linguistic slots가 주는 공유 구조만 바뀐다.

## 다음 gate

### Gate A — random-weight actual systems preflight

기존 sealed Korean BPE tokenizers의 8K, 16K, 32K만 사용한다. 64K는 32K보다 step 감소가 작고
기존 random-weight E2E도 더 느렸으므로 결과 근거로 제외한다. 모든 역할은 BPE-2K body와 같은
384/1,536/8 geometry를 사용한다.

- BPE-2K dense baseline
- same-body dense 8K/16K/32K controls
- tied low-rank 8K/16K/32K controls
- generic/Hangul codebook 8K/16K/32K

같은 42 document cases, parallel prefill, cached incremental continuation, full-vocabulary argmax를
사용한다. Tokenizer는 기존 protocol처럼 model timer 밖에 두되 encode cost를 별도 보고한다.
Cached logits는 full forward의 dense-equivalent logits와 tolerance 및 exact argmax로 검증한다.

Codebook candidate는 BPE-2K 대비 measured continuation step이 10% 이상 적고, random-weight
end-to-end median도 10% 이상 빨라야 한다. 통과 vocabulary 중 가장 작은 것을 먼저 선택해
quality risk를 줄인다. 어느 codebook도 통과하지 못하면 model training 없이 이 branch를
종료한다.

### Gate B — one-seed quality와 trained-model timing

Gate A를 통과한 최소 vocabulary에서 다음을 동일 128M raw bytes로 학습한다.

- existing BPE-2K baseline
- same-body dense upper control
- tied low-rank control
- generic Unicode code
- shuffled-Hangul code
- Hangul code

Hangul candidate는 BPE-2K 대비 contiguous/document raw-byte BPB와 document bootstrap upper가
모두 `+0.010 BPB` 이하여야 한다. 또한 generic과 shuffled control보다 paired document BPB가
낮아야 하며, low-rank control에 latency--quality 양쪽으로 지배당하면 안 된다. 이 조건을
통과한 뒤에만 trained-model controlled/free-running actual E2E 10% gate를 연다.

### Gate C — scale과 publication confirmation

Compact actual gate가 양수일 때만 50M/100M family-aware feasibility를 수행하고, 이 Mac에서
통과하는 가장 큰 scale을 선택한다. 이후 3--5 model seeds, 새 sealed final Korean split,
독립 fresh-process timing sessions, tokenizer/API cost, memory, Korean downstream을 실행한다.

## 반증 조건과 claim 경계

다음 중 하나면 positive Korean-efficiency paper 경로를 중단하거나 다시 근본적으로 검토한다.

- codebook head가 random-weight actual E2E 10%를 만들지 못함
- Hangul code가 BPE-2K raw-byte quality를 맞추지 못함
- generic/shuffled code와 차이가 없어 Korean prior의 기여가 없음
- low-rank control이 같은 quality에서 더 빠름
- trained compact model의 실제 E2E가 10% 미만
- 큰 scale 또는 다중 seed에서 효과가 재현되지 않음

현재 결과는 compositional head의 성공 증거가 아니다. 또한 static codebook, low-rank projection,
Jamo embedding 자체의 최초성을 주장하지 않는다. 앞으로 논문 가치가 생기는 경우는
**Korean orthographic assignment가 generic factorization보다 품질을 보존하고, 그 보존 덕분에
큰-vocabulary step reduction이 실제 matched-quality latency 개선으로 전환되는 경우뿐**이다.

## Artifacts

- plan: `data/manifests/length-gain-opportunity-v1.json`
- result: `results/length-gain-opportunity-v1/summary.json`
- ignored worker/pieces: `artifacts/length-gain-opportunity-v1/`
