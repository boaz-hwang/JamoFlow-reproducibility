# Large-model preflight V1 invalidation and V2 correction

> 작성일: 2026-08-15
>
> V1 plan SHA-256: `a1964a1b623a7e8c7904ec2cfce811e122802aa6d8df3c99733dfaaf3cac325d`
>
> V1 plan artifact SHA-256:
> `2c64dd73afc67f0cac8a29e409fe43cf457f69ffa7a9683c25f974cda24ce4aa`
>
> 판정: **V1 runner invalid; no compatibility or efficiency result**

## 실패 지점

Pinned EXAONE snapshot의 weight와 10개 file download/hash 검증은 완료됐다. V1 runner는 MLX built-in
`exaone` model을 구성한 뒤 tokenizer를 load하는 과정에서 중단됐다. Repository의 `config.json`은
custom `ExaoneConfig`를 가리키지만 runner가 MLX-LM의 `tokenizer_config`에
`trust_remote_code=True`를 전달하지 않았기 때문이다.

`mlx_lm.load()`가 반환되기 전에 예외가 발생했으므로 다음은 실행되지 않았다.

- tokenizer round trip
- model forward
- full/cache logit 비교
- greedy generation 또는 generated token 관측
- forced speculative transaction
- candidate-vs-baseline 실행
- timing/throughput/acceptance 계산

V1 result file은 생성되지 않았다. 따라서 실패는 EXAONE compatibility failure나 Qwen fallback
authorization이 아니라 runner configuration omission이다.

## V2의 유일한 의미 변경

V2는 다음만 바꾼다.

1. `mlx_lm.load(..., tokenizer_config={"trust_remote_code": True})`
2. custom code가 model revision으로 고정되고 downloaded `.py` file SHA-256가 result에 포함됨을 plan에
   명시
3. deprecated MLX memory API를 동일 의미의 current API로 교체
4. V1 plan identity와 no-forward/no-result 사실을 V2 plan에 결속

Model, revision, weight, prompt texts, tolerance, generated-token count, three forced paths, memory gate,
fallback rule, claim boundary는 바꾸지 않는다.

## 보안·재현성 경계

`trust_remote_code=True`는 이동하는 branch code를 신뢰한다는 뜻이 아니다. Exact Hugging Face
revision은 이미 고정되어 있고, V2 result는 `configuration_exaone.py`와 `modeling_exaone.py`를 포함한
모든 downloaded file의 content hash를 기록한다. MLX model forward는 여전히 installed built-in
`mlx_lm.models.exaone` implementation을 사용한다. Remote configuration code는 tokenizer/config
resolution에만 허용한다.

## V2 실행 규칙

V2 implementation과 본 문서를 먼저 commit하고 새 `large-model-retrieval-preflight-v2.json`을
exclusive create한다. V1 namespace를 삭제·덮어쓰기하지 않는다. V2 pass 전에는 actual retrieval
table/case plan을 만들지 않는다.

