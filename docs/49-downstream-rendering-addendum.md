# Korean downstream rendering and sealed-split addendum

> 작성일: 2026-08-11
> 상태: **downstream row·score 확인 전 prompt/split 고정**
> 상위 protocol: [Publication comparator and downstream protocol](./48-publication-comparator-and-downstream-protocol.md)
> label-boundary 후속 교정: [downstream label-boundary correction](./70-downstream-label-boundary-correction.md)
> 실행 조건: compact Final Value Gate 통과 뒤 pinned artifact audit

## 1. 확인한 pinned schema

[KLUE pinned card](https://huggingface.co/datasets/klue/klue/blob/349481ec73fff722f88e0453ca05c77a447d967c/README.md)는 YNAT의 `guid, title, label`과 NLI의 `guid, premise, hypothesis, label`을 정의한다. [KoBEST pinned card](https://huggingface.co/datasets/skt/kobest_v1/blob/a5ea15e3ac77ed694b79f6204eb31889a2ba989f/README.md)와 pinned JSON 구조에서 primary field는 다음과 같다.

- BoolQ: `paragraph, question, label`
- COPA: `premise, question, alternative_1, alternative_2, label`
- WiC: `word, context_1, context_2, label`
- SentiNeg: `sentence, label`

KoBEST card의 Data Fields 일부는 WiC를 `target_word`라고 쓰지만 example, 이전 pinned loader와 JSON은 `word`다. 이를 alias로 조용히 허용하지 않고 publication loader는 `word` exact schema를 요구한다. Pinned artifact가 다르면 schema drift로 중단한다.

## 2. Prompt version

Version은 `publication-v1-20260811`이다. 모든 template은 마지막이 `[정답]\n`이고 정답은 바로 뒤의 ASCII digit 한 개다. Prompt loss는 mask하며 digit loss만 학습한다.

Token model도 prompt와 digit을 별도 encode한다. Joint `prompt + digit` encoding이
경계를 가로질러 merge되더라도 primary sequence는 `encode(prompt) +
encode(digit)`으로 만들고, joint 결과는 sensitivity diagnostic으로만 남긴다.

### BoolQ

```text
[문단]
{paragraph}
[질문]
{question}
[선택지]
0: 아니오
1: 예
[정답]
```

### COPA

```text
[전제]
{premise}
[질문 유형]
{question}
[선택지]
0: {alternative_1}
1: {alternative_2}
[정답]
```

### WiC

```text
[대상어]
{word}
[문장 1]
{context_1}
[문장 2]
{context_2}
[질문]
두 문장에서 대상어의 의미가 같은가?
[선택지]
0: 다른 의미
1: 같은 의미
[정답]
```

### SentiNeg

```text
[문장]
{sentence}
[감성]
0: 부정
1: 긍정
[정답]
```

### YNAT

```text
[뉴스 제목]
{title}
[주제]
0: IT과학
1: 경제
2: 사회
3: 생활문화
4: 세계
5: 스포츠
6: 정치
[정답]
```

### NLI

```text
[전제]
{premise}
[가설]
{hypothesis}
[관계]
0: 함의
1: 중립
2: 모순
[정답]
```

## 3. Unicode와 truncation

모든 field를 NFC로 변환하고 변경 field 수를 split별로 공개한다. Candidate와 BPE에 동일하게 적용한다. Prompt는 최대 511 UTF-8 bytes이며 answer digit을 붙인 전체 sequence가 512 bytes 이하다.

Options, question, target word와 instruction은 보존한다. 초과하면 다음 context tail만 제거한다.

- BoolQ `paragraph`, COPA `premise`, SentiNeg `sentence`, YNAT `title`
- WiC `context_1/context_2`, NLI `premise/hypothesis`는 현재 UTF-8 byte 길이가 긴 field부터 번갈아 제거

각 field 최소 scalar 수는 code의 `_TRUNCATABLE_FIELDS`로 고정한다. Fixed/preserved 부분만으로 cap을 넘으면 row를 임의로 손상하지 않고 task audit를 실패시킨다. Sealed split truncation rate가 10%를 넘으면 상위 protocol에 따라 해당 task는 primary에서 탈락하고 전체 gate가 block된다.

## 4. KLUE sealed split

Public test label이 없으므로 official train을 fit/internal selection으로 나누고 official validation을 sealed evaluation으로 둔다. YNAT와 NLI에서 label별로 다음을 실행한다.

1. `SHA256(dataset revision ␟ task key ␟ guid ␟ NFC input fields)`를 계산한다.
2. digest 오름차순으로 정렬한다.
3. 각 label의 `ceil(n × 0.10)`개를 selection, 나머지를 fit으로 둔다.
4. duplicate guid 또는 digest collision, label당 2개 미만이면 실패한다.
5. row order와 무관한 assignment SHA-256과 label별 count만 tracked artifact에 기록한다.

KoBEST는 official train/validation/test를 그대로 쓴다. `test_originated`는 SentiNeg 학습이나 checkpoint 선택에 넣지 않으며 perturbation provenance 설명에만 사용한다.

## 5. Contamination input

오염 detector에는 label, prompt instruction, label-name list를 넣지 않는다. 위 schema의 자연어 input field만 순서대로 newline 결합한다. 따라서 YNAT의 category text나 NLI의 `함의/중립/모순`이 HPLT filtering signal로 들어가지 않는다.

`src/jamoflow/downstream_data.py`가 template, NFC, byte cap, label mapping과 KLUE split의 단일 source of truth다. Prompt snapshot hash와 synthetic split fixture가 drift를 막는다.
