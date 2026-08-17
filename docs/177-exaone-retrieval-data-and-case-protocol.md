# EXAONE retrieval data and case protocol

> 작성일: 2026-08-15
>
> 상태: **tokenization, table construction, case selection 전에 고정할 metric-free protocol**

## 목적

EXAONE 8B actual timing에 들어갈 generic retrieval table과 Korean prompt cases를 model output이나
latency를 보지 않고 만든다. 이 단계는 효율 실험이 아니며 target model forward를 실행하지 않는다.

## 고정 입력

- target/tokenizer: V4 compatibility를 통과한
  `mlx-community/EXAONE-3.5-7.8B-Instruct-4bit`, revision
  `6f8fba5756a6e2987aecacd8d7e8bb9410ef1a53`
- table source: `hplt3-korean-vocab-adaptation-v2`의 stable train split
- table byte budget: full-document prefix가 128,000,000 bytes를 넘기기 직전까지
- evaluation source: `hplt3-korean-final-test-v1`의 1,482 sealed stable-test documents
- fresh-v2 source seal이 final 1,482 documents를 exact/normalized exclusion했고 final-test seal도
  predecessor intersection 0을 증명해야 한다.

이 evaluation document pool은 이전 품질 실험에 이미 사용되었다. 따라서 여기서 새로 고르는 72개
timing case는 latency/model output에 대해서는 outcome-blind이지만, untouched final 또는 독립
confirmatory set은 아니다. 이 단계는 8B systems development/replication workload이고, 최종 논문용
확증은 별도의 미사용 Korean raw-completion 및 chat workload에서 반복한다.

사용자 `vault/` 문서는 table, case, threshold 선택에 쓰지 않는다.

## tokenizer와 table

각 train document를 independently `add_special_tokens=False`로 encode하고 decode cleanup은 끈다.
Document 사이에는 독립적으로
encode한 newline token sequence를 넣는다. Context order는 1, 2, 3이고 next-token proposal을 세 번까지
재귀적으로 만든다.

- maximum entries: 200,000
- minimum context count: 5
- minimum winning-next count: 5
- minimum winning-next probability: 0.8
- rank: best count 내림차순, confidence 내림차순, order 내림차순, context/next 오름차순
- 한 context에서 winning-next count가 동률이면 가장 작은 token ID를 선택
- lookup: longest context first
- hybrid precedence: corpus n-gram first, 없으면 prompt/self-output longest suffix match
- prompt match: maximum 4 tokens, equal length면 earliest prior occurrence

102,400 vocabulary에서 `V^4`는 `uint64` 범위를 넘는다. 따라서 3-token context를 `uint64`에 pack하되
next token은 `uint32`로 분리하고 `(context,next)`를 lexicographic sort해 count한다. Context와 next를
한 `uint64`로 합치거나 hash collision을 허용하지 않는다.

## cases

각 final-test document를 EXAONE tokenizer로 encode한다. 다음 조건을 모두 만족하는 문서만 eligible이다.

- 최소 256 tokens
- 첫 256 tokens에 special token 없음
- 전체 256-token prefix, 128-token prompt, 128-token continuation 각각의 decode 후 re-encode token
  IDs가 exact
- `decode(prompt) + decode(continuation) == decode(full 256 tokens)`
- 첫 128-token prompt에 precomposed Hangul codepoint가 최소 32개
- alphabetic characters 중 precomposed Hangul fraction ≥0.8
- visible non-whitespace characters 중 precomposed Hangul fraction ≥0.5

Evaluation seal artifact hash, V4 compatibility result artifact/payload hash, exact case contract에서
유일하게 도출한 rank key와 document exact-byte SHA-256으로 domain-separated rank를 만들고
`(rank,digest)` 오름차순 첫 72개를 선택한다. 임의 salt를 고르지 않는다. 처음 8개는 baseline-only
resource calibration/warmup, 나머지 64개는 actual measured cases다.
한 document는 한 case만 만든다.

- prompt: first 128 tokens
- controlled continuation: next 128 tokens
- token arrays와 document/rank digests는 ignored NPZ에 저장
- tracked seal에는 aggregate commitment, artifact/array hashes, counts만 기록
- text나 individual token IDs는 tracked JSON에 넣지 않는다.

## fail-closed 순서

1. implementation/tests/본 문서를 commit
2. clean tree에서 data plan exclusive create
3. plan을 별도 commit
4. exact source/seal/model compatibility dependency 재검증
5. V4 result를 공식 validator로 재검증하고 tokenizer/remote-code 8개 파일의 current snapshot
   size/SHA-256을 V4 result manifest와 exact 대조
6. tokenizer-only table/case build; model forward 금지
7. selected train/evaluation exact 및 NFKC+casefold+whitespace-collapse normalized set intersection을
   다시 계산하고 둘 다 0으로 봉인
8. artifact arrays 재로딩 및 table/case invariant 검증
9. tracked aggregate seal exclusive create 및 별도 commit
10. source/tokenizer에서 table과 cases를 독립적으로 한 번 더 전체 재구성하여 bitwise equality를
    검증하고 tracked verification receipt를 별도 commit
11. verification commit 뒤에만 baseline-only resource calibration

Plan/seal history가 있거나 namespace가 partial이면 자동 overwrite하지 않는다.

## claim boundary

이 seal은 다음만 증명한다.

> The EXAONE retrieval table uses a sealed train-only Korean source, while 72 deterministic Korean cases
> come from an exact- and normalized-disjoint, previously used sealed evaluation pool without model-output
> or latency input to case selection.

Table hit rate, acceptance, target-call reduction, latency, quality, Korean-specific advantage는 아직
증명하지 않는다. 200,000 entries는 small-model positive에서 가져온 고정 generic baseline이지 새로운
method contribution이 아니다. 이 workload는 raw continuation이며 chat template/deployment generality,
final blindness, memory improvement도 주장하지 않는다.
