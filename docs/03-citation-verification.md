# 02 재검토 문서의 인용 전수 검증 결과

> 작성일: 2026-08-08
> 검증 대상: [02-critical-research-direction-review.md](./02-critical-research-direction-review.md)가 근거로 인용한 논문 9건 + 세부 메커니즘 주장 10건
> 검증 방식: 독립 에이전트 3팀 — ① 2023 선행연구 + AU-Net 경계 방식, ② 컷오프 이후 2026년 인용, ③ Scratchpad·BLT·SpaceByte 메커니즘 세부 서술. 전부 1차 출처(ACL Anthology·arXiv raw HTML) 직접 대조. 특히 ③팀은 arXiv HTML을 curl로 받아 원문 문장을 grep 대조.

---

## 최종 판정

**02 문서는 사실검증을 통과했다. 허위 인용 0건, 핵심 주장 왜곡 0건.**
따라서 02의 §9 수정 권고를 00·01 문서에 반영하고, 00의 "주제 확정"을 "후보 가설"로 되돌린다.

유보 2건: 02가 논문의 실제 발견을 넘어 덧붙인 해석적 확장 2곳(아래 표시)은 "논문의 직접 주장"이 아니라 "02 저자의 추론"으로 구분해 취급한다.

---

## 1. 인용 논문 실존·정확성 (9건)

| 인용 | 판정 | 확인 내용 |
|---|---|---|
| Learn Your Tokens (EMNLP 2023 Findings) | ✅ | Thawani·Ghanekar·Zhu·Pujara. abstract 원문: "utilizes the word boundary to pool bytes/characters into word representations... before again decoding individual characters/bytes per word **in parallel**" — 02의 요약과 정확히 일치 |
| Dynamic Token Pooling (ACL 2023) | ✅ | Nawrot·Chorowski·Łańcucki·Ponti. entropy spike·subword tokenizer·언어학적 경계 + end-to-end 학습(02가 언급 안 한 4번째)을 직접 비교. **01 보고서의 신규성 조사가 놓친 직계 선행 확정** |
| AU-Net (arXiv:2506.14761) | ✅ | 원문: "splits on spaces using **different regular expressions** at each stage" — word→2단어→4단어 계층, **규칙 기반 경계 확정**. 01의 "선행은 전부 학습 신호" 서술은 오류였음. 단 저자들이 "구분자 없는 언어 확장은 future work" 명시 — 한국어 확장 여지가 원문에 남아 있음 |
| FLEXITOKENS (Findings of ACL 2026) | ✅ | Owodunni·Ahia·Kumar, pp.17170–17190. arXiv:2507.12720. learnable boundary predictor + one-sided margin 손실로 인스턴스 적응 압축, 다국어 평가, BPE 대비 최대 10%p. URL 정확 |
| From Bytes to Subwords (Findings of ACL 2026) | ✅ | van der Goot 단독. UTF-8이 fairness·efficiency에 비최적일 수 있다는 주장 실존. ⚠️ "representation과 patching 효과 분리" 프레이밍은 02의 해석적 확장 |
| Beyond Perplexity: UTF-8 Validity (arXiv:2606.14122) | ✅ | Moon·Oba·Ma·Hiraoka·Okazaki. 355M/80B tokens(영·일·한·중). **perplexity는 2.1B 토큰에 안정, UTF-8 validity는 4.2B 토큰 필요(약 2배 지연)** — 핵심 수치 확인. ⚠️ "hard constraint는 correctness에서 먼저 기여" 결론은 02의 외삽 |
| Scratchpad Patching (arXiv:2605.09630) | ✅ | 아래 §2 참조 |
| BLT entropy 모델 세부 (arXiv:2412.09871) | ✅ | 아래 §2 참조 |
| SpaceByte spacelike 규칙 (arXiv:2404.14408) | ✅ (caveat 2건) | 아래 §2 참조 |

주목: Beyond Perplexity 저자진(Moon·Okazaki)은 **EACL 2023 3-hot(Cognetta et al.)과 같은 연구 라인** — 이 그룹이 이미 한국어 포함 byte-level validity를 다루고 있어 경쟁/인용 지형의 핵심 노드.

## 2. 메커니즘 세부 주장 (10건)

### Scratchpad Patching — 4/4 CONFIRMED

| 02의 서술 | 원문 근거 |
|---|---|
| patch 내부 transient scratchpad로 patch lag 감소, 최종 KV cache에 안 남음 | "inserts transient scratchpads inside each patch..." / "excluded from the persistent KV cache at inference" (Abstract, §3) |
| SP 추가 시 단순 방식이 entropy/H-Net 계열과 가까워짐 | §4.2 실측(NLU 평균): Fixed 48.0→54.2, **SpaceByte 54.5→56.2**, Entropy 53.2→55.3, **H-Net 55.4→55.5**. SP 적용 후 SpaceByte가 H-Net·entropy를 **역전** |
| boundary rule보다 compute allocation이 더 중요할 수 있음 | "compute allocation may matter more than the choice of patchification" (§4.3) / "primary bottleneck may be insufficient compute rather than suboptimal boundary placement" (§1) |
| entropy predictor를 별도 100M LM이 아닌 encoder 위 2-layer aux head로 구현 | Appendix B.3: "two additional Transformer layers placed on top of the encoder... adding more layers does not yield further improvements" |

### BLT — 3/3 CONFIRMED

- "1M~100M entropy model 실험" — §7 원문 확인
- "50M 이상 수확체감" — Fig.8 캡션 + §7: "diminishing returns when we scale beyond 50m parameters"
- "짧은 receptive field면 lookup table 구현 가능" — §4.2: "can be encoded in an efficient lookup table"
- (기본 실험용 entropy 모델은 100M/14-layer/512 sliding window로 별도 명시 — 01의 31% 추정이 이 기본값 기준이었음은 사실이나, 논문 스스로 더 싼 대안을 제시하고 있음)

### SpaceByte — 메커니즘 CONFIRMED + caveat 2건

- spacelike 정의: "the byte does not encode a letter, number, or UTF-8 continuation byte" — **multi-byte 문자의 leading byte가 spacelike로 분류되어 문자당 1회 global block 트리거.** 02의 "CJK에도 global cadence 제공" 서술 정확.
- **Caveat 1 (원문 자체 오기)**: 본문 한 문장이 "We define continuation bytes to be spacelike..."라고 각주와 자구 모순 — 저자의 오기로 추정. 인용 시 원문 문장을 그대로 옮기면 안 됨.
- **Caveat 2 (실험 결과)**: SpaceByte 스스로 Limitations에서 "our preliminary experiments suggest that SpaceByte **performs worse than subword transformers on Chinese text**"라고 보고. 즉 **규칙 cadence의 메커니즘은 있으나 spaceless/CJK에서의 실험적 성공은 미확립** — 이 지점은 오히려 우리 연구 질문(규칙 경계가 CJK에서 유효한가)이 진짜 열려 있음을 보여줌.

---

## 3. 검증이 확정한 것들 (00·01 수정의 근거)

1. **"31% 절감"은 낡은 구현 공격이 맞다.** BLT 원문이 50M 수확체감·lookup table을 명시하고, Scratchpad가 patcher를 2-layer 통합 head로 대체 — "별도 100M 모델 제거"를 핵심 기여로 삼을 수 없음. 가정 명시된 이론 상한으로 강등.
2. **"선행은 전부 학습 신호"는 오류.** AU-Net(regex), SpaceByte(공백 규칙)가 반례. Dynamic Token Pooling은 경계 종류 비교를 이미 수행. 신규성은 "규칙 사용 여부"가 아니라 **"detector 비용까지 포함한 총비용 Pareto 통제 실험의 부재"**로 좁혀야 함.
3. **boundary 품질보다 compute allocation이 중요할 수 있다는 Scratchpad의 결과**는 "어느 경계가 좋은가" 단독 질문의 가치를 낮추고, 02가 권고한 비용-제약 hybrid 프레이밍을 지지.
4. **UTF-8 validity의 늦은 수렴(2.1B vs 4.2B)**은 hard orthographic constraint의 1차 기여처가 속도가 아니라 correctness라는 방향(02 §7의 2차 연구)에 실증적 힌트를 제공. 단 이 연결은 02의 외삽이므로 가설로 취급.
5. **SpaceByte의 중국어 열위 자백**은 CJK에서 rule cadence의 유효성이 미해결임을 보여줌 — 중국어 control 실험의 가치를 오히려 높임.

## 4. 조치

- [00-topic-selection.md](./00-topic-selection.md): "주제 확정" → "후보 가설"로 격하, 중심 질문 교체, zero-cost→parameter-free, 31% 강등, Phase 0을 matched patch-rate audit로 교체 (02 §9 반영)
- [01-verification-report.md](./01-verification-report.md): 정정(errata) 블록 추가 — A∧¬A 완화, free lunch 격하, 31% caveat, 누락 선행 3건 추가, 0xAC00/UTF-8 구분, Δ 게이트 appendix 이동
