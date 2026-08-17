# Publication comparator-role lock

> 작성일: 2026-08-12
> 상태: **publication-scale reference selection과 test artifact 생성 전 결과맹 고정**
> 선행 교정: [publication evidence-identity correction](./68-publication-evidence-identity-correction.md)

## 1. 남아 있던 우회

Evidence identity를 연결한 뒤에도 identity 문자열의 허용 집합은 열려 있었다.
특히 `choose_validation_reference`는 32K BPE key만 지정하면 raw와 16K 중 하나가
빠진 score mapping도 받을 수 있었다. 문서상 선택 후보는 raw, 16K, 32K 세
reference로 봉인되어 있으므로, 두 후보만 비교하거나 16K/32K 이름을 바꾸는
실행은 protocol과 다르다.

또한 Final Value Gate의 mapping key가 `16_000`이어도 내부 gate가 실제 32K
family를 가리키는 경우를 명시적으로 막아야 한다. 단순히 두 BPE gate가 있다는
사실만으로 vocabulary-size stress control이 성립하지 않는다.

## 2. 봉인된 역할

`src/jamoflow/publication_protocol.py`를 단일 source of truth로 삼아 다음 stable
model-family key를 고정했다.

| 역할 | key |
|---|---|
| candidate | `candidate` |
| strongest locked raw-byte reference | `raw_byte_reference` |
| body-matched 16K stress BPE | `byte_bpe_16000_body_matched` |
| ordinary 32K BPE | `byte_bpe_32000` |

Candidate의 구체적 patch policy와 각 seed checkpoint는 향후 lock artifact의
config/hash로 식별한다. Raw role도 alias만 저장하지 않는다. Compact calibration-only
selection의 `policy/runtime_policy/model_family/patch_count` descriptor와 selection hash를
승계하며, E/EC이면 descriptor에서 `entropy_router`를 파생해 seed별 router/calibration
bundle을 추가한다. 위 key는 결과를 보고 고르는 checkpoint 이름이 아니라 실험 전부터
고정된 역할이다.

## 3. 강제 조건

- Validation reference selection은 매 task에서 raw·16K·32K의 세-seed score를
  정확히 모두 요구한다. 일부 누락이나 extra/post-hoc reference는 거부한다.
- 32K만 0.5 pp near-tie deployment default가 될 수 있다.
- Downstream task comparison의 candidate와 reference key는 봉인 집합에 있어야
  한다.
- Data-adequacy curve도 candidate, raw, 16K, 32K의 exact role set을 요구한다.
- BPB와 actual-inference comparator는 declared family와 role key가 일치해야 한다.
- Final Value Gate의 `16_000`/`32_000` mapping을 서로 바꾸면 실패한다.

이 교정은 모델 score, latency 또는 선택 결과를 사용하지 않았다. 진행 중 compact
S/E/EC family의 데이터·학습·평가에는 영향을 주지 않는다.

## 4. 검증 경계

집중 테스트는 exact three-reference selection, candidate/reference role,
data-adequacy alias와 swapped BPE vocabulary mapping을 포함한다. 이 role lock만으로
checkpoint provenance가 완성되는 것은 아니다. 최종 runner는 family key와 함께
checkpoint state hash, tokenizer hash, raw-byte budget, parameter graph와 Git commit을
독립 재구성해야 한다. Raw role은 concrete selection descriptor와 auxiliary bundle까지
같아야 하며 stable alias만 같은 것으로는 충분하지 않다.
