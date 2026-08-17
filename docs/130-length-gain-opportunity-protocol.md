# Byte-LengthGain-2K opportunity protocol

> 작성일: 2026-08-14
>
> 상태: train-only configuration preflight 이후, candidate calibration token count를
> 보기 전에 고정할 protocol
>
> 결과: minimum-token DP도 calibration 4.153%, measured continuation 5.668% 감소에
> 그쳐 10% gate를 실패했다. `docs/131-length-gain-result-and-compositional-head-pivot.md` 참조.

## 결론과 목적

Same-2K Byte-Unigram은 BPE보다 11.7--15.7% 긴 sequence를 만들어 실패했다. 실패 원인은
segmentation보다 vocabulary였다. 이 protocol은 2,048개라는 동일한 row/graph budget에서
현재 segmentation의 **실제 비중첩 token 절감량**을 직접 최적화하면 BPE보다 최소 10%
짧아질 수 있는지를 한 번 더 검증한다.

이것은 모델 학습, 품질, actual latency 또는 논문 성과가 아니다. Minimum-token DP까지
10% opportunity를 만들지 못하면 같은 vocabulary의 어떤 deterministic segmentation도 현재
어휘에서 더 적은 token을 만들 수 없으므로, same-2K tokenizer 분기를 종료한다.

## 결과 전에 고정한 constructor

- train split의 첫 8,000,000 raw bytes만 사용
- mandatory identity byte 256개 + learned direct byte pieces 1,792개
- final vocabulary 정확히 2,048
- maximum piece length 48 bytes
- current minimum-token segmentation에서 token n-gram arity `2..8`
- score: exact left-to-right non-overlapping occurrence count × `(arity - 1)`
- 라운드마다 8개를 추가하고 전체 train prefix를 minimum-token DP로 다시 segmentation
- 224 rounds, 결과를 보고 조기 중단하지 않음
- newline을 포함하는 candidate는 제외해 line/document separator를 넘는 piece를 만들지 않음
- score, occurrence, arity, raw bytes의 고정 tie order
- 두 개의 uint64 rolling hash는 indexing에만 사용하고, 검토한 모든 group의 exact token tuple을
  확인한다. 충돌은 hard failure다.

한 라운드에서 8개를 함께 고르는 것은 exact one-at-a-time Length-MAX 재현이 아니다. 그래서
방법명을 `Byte-LengthGain`으로 제한한다. 같은 이유로 공개 논문의 수치를 재현했다고 주장하지
않는다. 이 실험이 묻는 것은 더 좁다. 실제 token saving을 반복 최적화하는 실현 가능한 2K
constructor가 현재 Korean stream에서 10% systems margin을 만드는가다.

## 이미 본 train-only engineering anchors

2MB train prefix에서 double-hash overlap-frequency prototype을 실행했다. Calibration은 보지
않았다.

| score / batch | BPE-2K 대비 train token 감소 |
|---|---:|
| immediate saving / 256 | -4.270% |
| immediate saving / 32 | +5.017% |
| immediate saving / 8 | **+6.416%** |
| current-token length / 32 | +4.407% |

Batch 32→8의 추가 이득은 1.399%p였고 current-token-length 점수는 더 나빴다. 따라서
batch 8과 immediate-saving을 고정한다. 새 구현의 exact non-overlap 첫 라운드는 2MB에서
6.557초였다. 8MB×224 rounds와 독립 replay는 이 Mac에서 실행 가능하지만 publication-scale
모델 학습보다 훨씬 싸다.

이 표는 configuration을 고르기 위한 train-only 기록이며 gate evidence가 아니다. 특히
2MB 자기 train 절감 6.416%를 calibration 성능으로 외삽하지 않는다.

## 독립 replay와 calibration 개봉 순서

1. 이 문서, constructor, worker, verifier, tests를 clean commit한다.
2. plan을 seal하고 별도 commit한다.
3. worker는 train 8MB만 읽어 2K vocabulary와 224-round trace를 ignored artifact로 만든다.
4. verifier는 checkpoint처럼 저장 vocabulary를 신뢰하지 않고 train construction 전체를 다시
   실행해 ordered pieces, 매 round count, final token-ID hash를 exact 비교한다.
5. 두 construction이 같을 때만 기존 8MB calibration과 사전 고정 42 document cases를 연다.
6. BPE-2K, frozen left-most-longest, 같은 vocabulary의 minimum-token DP를 평가한다.
7. tracked aggregate를 한 번 publish하고 result history가 있으면 재봉인하지 않는다.

Tokenizer JSON/pieces와 corpus substring은 `artifacts/`에만 둔다. Tracked result에는 aggregate,
hash, count, timing만 기록한다.

## 판정

Primary order는 paper-inspired left-most-longest를 먼저, minimum-token DP를 두 번째로 둔다.
각 역할은 모두 다음을 통과해야 한다.

- calibration total token reduction vs exact BPE-2K `>=10%`
- 36 measured continuation token 합 reduction `>=10%`
- exact byte roundtrip

첫 통과 역할만 Korean-complete paired control 설계를 연다. 둘 다 실패하면:

- Byte-LengthGain model을 학습하지 않는다.
- batch, score, train prefix 또는 gate를 calibration 결과에 맞춰 다시 고르지 않는다.
- “긴 token이 부족했다”는 동일 가설을 다른 trainer 이름으로 반복하지 않는다.
- 다음 연구는 docs/114가 남긴 **factorized large-vocabulary head / capacity-scale** novelty audit로
  이동한다. 이 방향은 32K-like step 감소와 2K-like Transformer core capacity를 분리할 수
  있는지가 핵심이며, adaptive/hierarchical softmax 및 factorized embedding 선행과 먼저
  구별해야 한다.

Generic이 통과한 경우에만 동일 constructor budget에 UTF-8 codepoint, Hangul syllable,
eojeol-boundary completeness를 eligibility constraint로 넣은 Korean variant를 봉인한다. 규칙을
generic 실패의 사후 구제책으로 쓰지 않는다.

## Claim 경계

- calibration-development only
- model-free token opportunity only
- tokenizer encode throughput은 diagnostic이며 model timer와 분리
- same 19.6M model quality나 wall time을 아직 증명하지 않음
- batch-8 generalized n-gram constructor이며 Length-MAX paper-faithful reproduction이 아님
- Korean-aware method는 이 단계에서 아직 평가하지 않음

최종 성공 기준은 바뀌지 않는다. 통과 vocabulary를 학습하더라도 fastest quality-qualified
BPE 대비 raw-byte quality noninferiority와 trained-model batch-1 E2E 10% 개선을 별도로
통과해야 한다.
