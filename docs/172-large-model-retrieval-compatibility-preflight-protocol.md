# Large-model retrieval compatibility preflight protocol

> 작성일: 2026-08-15
>
> 상태: **historical base protocol; V4 operational contract는 docs/175가 지배**
>
> Primary: `mlx-community/EXAONE-3.5-7.8B-Instruct-4bit` at revision
> `6f8fba5756a6e2987aecacd8d7e8bb9410ef1a53`
>
> 운영 정정: V1 loader omission은 `docs/173`, V2 cache-oracle correction은 `docs/174`, V3의
> all-logit numeric gate correction은 `docs/175`가 지배한다.

## 목적

이 단계는 speedup을 측정하지 않는다. 공개 7.8B 한국어 중심 model이 현재 Mac/MLX에서 exact
retrieval speculative transaction을 구현할 수 있는지 다음 항목만 판정한다.

- pinned revision/weight integrity
- built-in MLX `exaone` loader
- tokenizer와 chat template determinism
- full-prefix와 cached incremental의 finite-logit 및 greedy argmax equivalence
- cache trim 뒤 fresh-cache의 finite-logit 및 greedy argmax equivalence
- repeated greedy token sequence identity
- forced full-accept, immediate-reject, partial-accept transaction이 ordinary greedy token sequence와 exact
- model load와 짧은 correctness replay의 memory safety

처리량, latency ratio, acceptance rate, corpus/prompt lookup hit rate, 생성 품질은 계산하거나 출력하지
않는다. 따라서 이 결과는 model/runtime 선택의 compatibility evidence일 뿐 efficiency evidence가
아니다.

## 고정 model 선택

Primary는 EXAONE 3.5 7.8B 4-bit다. 다음 technical failure 중 하나가 재현될 때만 고정 Qwen3-8B
fallback을 열 수 있다.

1. built-in loader failure
2. tokenizer/chat-template failure
3. full/cache greedy-decision equivalence failure
4. cache trim/rollback failure
5. deterministic greedy failure
6. forced speculative exactness failure
7. memory safety failure
8. runtime crash

Acceptance, tokens/s, output quality, candidate-vs-baseline 결과는 fallback 사유가 아니다. Primary가
compatibility pass하면 EXAONE을 이후 actual protocol의 target으로 고정한다.

## 고정 environment와 artifact

- Python 3.13 environment의 exact package versions를 plan에 기록
- `mlx==0.31.2`
- `mlx-lm==0.31.3`
- 별도 `requirements/apple-retrieval-v1.txt`를 기존 research environment 위에 설치하며, 과거 봉인된
  `pyproject.toml`은 변경하지 않는다.
- Hugging Face revision과 모든 downloaded file의 SHA-256/bytes 기록
- `model.safetensors`: 4,398,345,620 bytes,
  SHA-256 `d9796bd9c23f506751f618fc08780b197106c50adbf317e4fa518a3c8a40974c`
- generated text/token IDs는 공개하지 않고 domain-separated ordered token hash만 기록

Hugging Face cache는 ignored local storage이며 authority는 repo revision + content hashes다. Model
artifact를 다운로드할 때 alternate revision, mirror, conversion을 선택할 수 없다.

## V4 equivalence contract

Full/cache와 rollback 비교는 reference-side diagnostic normalization

`0.05 + 0.01 * abs(reference_logit)`

을 모든 vocabulary logit에 적용해 maximum absolute/normalized error와 기존 bound 통과 여부를
그대로 기록한다. 다만 이는 102,400-way 전체 분포의 수치 진단이지 greedy transaction의 의미론적
판정이 아니다. Hard gate는 모든 비교 position의 logit이 finite이고 argmax가 exact한지 여부다.
NaN/Inf 또는 단 하나의 argmax 차이도 즉시 실패한다. Ordinary greedy 두 회와 세 forced path의 전체
token sequence도 exact해야 한다. 이 변경의 관측 근거와 사후 claim boundary는 `docs/175`에 적는다.

## forced transaction

Ordinary greedy로 고정 16-token target sequence를 한 번 만든다. 같은 model/cache에서 다음 세
proposal provider를 별도로 실행한다.

- `full_accept`: 다음 target token 세 개를 proposal
- `immediate_reject`: 첫 proposal부터 다른 valid token ID
- `partial_accept`: 첫 token은 target과 같고 두 번째부터 다르게 proposal

세 path 모두 ordinary sequence와 16/16 token exact해야 하고 각각의 intended counter가 양수여야
한다. 이 과정은 lookup acceptance를 평가하는 실험이 아니라 rollback state machine을 검증하기 위한
forced oracle다.

## memory gate

Model load 전에 MLX peak memory를 reset한다. 모든 correctness replay 후 peak allocated bytes가
`max_recommended_working_set_size`의 75% 이하이어야 한다. 이 수치는 actual generation peak 개선이나
energy 효율을 뜻하지 않는다.

## fail-closed 순서

1. implementation과 본 문서를 commit
2. clean worktree에서 plan을 exclusive create
3. plan을 별도 commit
4. pinned snapshot download 및 weight hash 검증
5. clean committed plan/implementation 재검증
6. timing API 없이 compatibility forward 실행
7. result exclusive create 및 commit

Plan/result history가 있거나 current HEAD blob과 다르면 재봉인하지 않는다. Primary pass 뒤에만
public Korean table/case와 actual E2E plan을 설계한다.

## claim boundary

Pass가 허용하는 문장:

> The fixed 7.8B EXAONE MLX target supports deterministic cached greedy decisions and exact
> forced-proposal verification/rollback within the prospectively sealed memory safety limit.

Pass가 허용하지 않는 문장:

- retrieval이 빠르다.
- 한국어에서 speculative decoding이 유리하다.
- EXAONE이 Qwen보다 적합하다.
- 형태론이나 한글 구조가 acceptance를 높인다.
- publication-ready efficiency evidence가 생겼다.
