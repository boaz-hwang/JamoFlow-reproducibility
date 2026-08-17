# ByteFlow causal and systems literature amendment

> 확인일: 2026-08-11
> 방법: 사용자 지정 `aside-browser`로 arXiv v1 원문과 서지 metadata 확인
> 시점: initial F/C/W·D/P 결과 관측 후, S/E/EC family 완성 결과와 모든 actual-inference 결과 전
> 영향: novelty 축소, learned baseline 범위 보강, Phase 3 policy/gate 불변

## 1. 누락

[Deng et al., *ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer*](https://arxiv.org/abs/2603.03583)은 ICLR 2026 논문으로, coding-rate 기반 importance와 Top-K selection을 사용해 raw bytes를 adaptive latent units로 압축한다. Static computation graph를 유지하면서 BPE Transformer와 기존 byte architecture보다 높은 reported quality를 보인 직접 관련 선행이다.

따라서 다음 주장은 금지한다.

- adaptive learned byte compression을 처음 연구했다
- static graph에서 learned variable boundaries를 처음 구현했다
- E/EC 하나와의 비교로 최신 learned byte patcher 전반을 이겼다

## 2. Prefix causality의 차이

ByteFlow §3.2의 teacher-forced compression은 full sequence의 importance profile에서 Top-K positions를 고른다. 각 position score의 feature가 causal mask를 쓰더라도, 어떤 position이 Top-K에 드는지는 suffix positions의 score와 경쟁한 뒤 결정된다. 따라서 논문에 기술된 training/evaluation selector는 동일 prefix에 대한 boundary가 unseen suffix와 무관하다는 JamoFlow의 prefix-invariance 조건을 자동으로 만족하지 않는다.

이 지적은 ByteFlow quality 결과가 잘못됐다는 뜻이 아니다. ByteFlow는 strong full-sequence adaptive-compression baseline이다. 다만 prefix-causal incremental decoder의 selector cost와 state update를 실제로 검증한 baseline으로 취급할 수 없다는 범위 구분이다.

## 3. Systems evidence의 차이

ByteFlow가 보고한 efficiency table은 8×A100 환경의 training words/second와 analytical FLOPs 중심이다. 논문에서 cached autoregressive generation의 TTFT, time-to-N output, batch-1 decode latency, selector 포함 end-to-end latency를 찾지 못했다.

따라서 ByteFlow의 reported WPS를 JamoFlow actual generation latency와 같은 지표로 합치지 않는다. 공개 code/checkpoint가 재현 가능하면 다음 두 표에 분리한다.

- quality/analytical/training-throughput related-work table
- prefix-causal incremental-runtime compatibility audit

Incremental compatibility가 확인되지 않으면 actual-inference Final Value Gate의 직접 comparator가 아니라 limitation으로 남긴다.

## 4. JamoFlow에 남는 질문

ByteFlow 이후 허용 가능한 좁은 질문은 다음이다.

> In a fixed-rate, prefix-causal Korean BLT setting, can an observed-whitespace relocation rule preserve quality while reducing measured cached autoregressive inference cost relative to quality-qualified raw-byte and tokenized references?

신규성은 `learned adaptive compression`이 아니라 Korean same-rate causal identification, detector-inclusive actual runtime, document-clustered quality, standard BPE deployment comparison의 결합에 있다. 이 결합도 결과가 모든 gate를 통과할 때만 method contribution이 된다.
