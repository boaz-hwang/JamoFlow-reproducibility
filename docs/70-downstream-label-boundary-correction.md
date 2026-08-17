# Downstream label-boundary correction

> 작성일: 2026-08-12
> 상태: **publication tokenizer·fine-tuning·prediction artifact 생성 전 결과맹 교정**
> 적용 범위: raw-byte candidate/reference와 16K/32K byte-BPE downstream 평가

## 1. 문제

Primary downstream interface는 Korean prompt 뒤의 ASCII digit 한 개를 예측한다.
Byte model에서는 prompt bytes와 digit byte의 경계가 자명하지만, BPE에서
`encode(prompt + digit)`을 호출하면 마지막 prompt bytes와 digit이 하나의 token으로
merge될 수 있다.

이 joint tokenization을 그대로 학습·평가하면 다음 문제가 생긴다.

- 정답 digit 정보가 prompt와 합쳐진 token identity에 들어간다.
- 이미 prefill한 prompt KV cache에서 next-unit label을 예측한다는 조건이 깨진다.
- Candidate의 한 byte와 BPE의 경계-crossing token이 서로 다른 conditional event가
  된다.
- tokenizer merge graph에 따라 task score가 달라지는 불필요한 confound가 생긴다.

Controlled replay에서 prompt와 continuation을 별도 encode한 이유와 같은 문제다.

## 2. 고정한 conditional event

모든 architecture에서 primary fine-tuning과 prediction은 다음 순서를 사용한다.

1. raw Korean prompt를 독립 encode한다.
2. 허용 label string `"0"` … `"6"`을 각각 독립 encode한다.
3. 각 label이 정확히 하나의 서로 다른 model unit인지 확인한다.
4. Prompt units를 causal prefill하고 그 final logit에서 허용 label unit만 비교한다.
5. 학습 시 prompt unit loss는 전부 mask하고 독립 encode한 gold label unit 한 개의
   full-vocabulary cross entropy만 계산한다.

`encode(prompt + label)`은 boundary-merge 발생률을 기록하는 diagnostic에만 쓰고
primary input이나 loss를 만드는 데 쓰지 않는다. Byte/BPE 모두
`prompt_units + separately_encoded_label_unit`이라는 동일한 conditioning semantics를
갖는다.

## 3. 실행 가능한 계약

`src/jamoflow/downstream_data.py`의 `encode_downstream_conditioning`은 다음을
검증하고 runner가 사용할 ephemeral units를 반환한다.

- prompt renderer version과 task identity
- 모든 task label의 single-unit·distinct-unit 조건
- prompt plus answer가 512 model units 이내인지
- separate encoding contract string
- joint encoding과 separate encoding의 일치 여부 diagnostic

Prompt token IDs와 gold token ID는 benchmark text를 복원할 수 있으므로 tracked
aggregate metadata에서 제외한다. 공개 metadata에는 unit count, label count,
boundary merge 발생 여부와 contract version만 남긴다.

## 4. 해석

이 교정은 BPE를 불리하게 만들기 위한 tokenization 변경이 아니다. 배포 시 이미
주어진 prompt에서 다음 answer unit을 예측하는 conditional likelihood를 정확히
정의한다. Joint encoding sensitivity는 별도 표로 공개할 수 있지만 primary Korean
downstream noninferiority를 대체하지 않는다.

