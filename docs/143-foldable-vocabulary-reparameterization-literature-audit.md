# Foldable vocabulary reparameterization 문헌·신규성 감사

> 작성일: 2026-08-15
>
> 상태: foldable generic residual의 후속 mechanism protocol을 봉인하기 전 직계 선행 재검증;
> `docs/150`의 음성 결과로 multi-hash branch 종료

> **후속 결과:** generic, stratified-shuffle, balanced-random multi-hash가 모두
> `update_matched_dense`보다 나빴고 surface minimum도 실패했다. 아래 novelty 후보는 실증적으로
> 승격되지 않았다. 최신 operational decision은 `docs/150`이다.

## 결론

현재 관측한 generic residual의 구성요소는 각각 이미 알려져 있다.

- target-language vocabulary expansion으로 token step과 inference cost를 줄이는 것
- 새 embedding row를 강하게 초기화하고 continued pretraining하는 것
- 여러 hash codebook을 합쳐 token representation을 만드는 것
- 학습 중에만 overparameterize하고 추론 전 dense graph로 접는 것

따라서 `vocabulary expansion`, `hash embedding`, `structural reparameterization` 자체는 신규성이
아니다. 논문 기여가 될 수 있는 좁은 조합은 다음뿐이다.

> Strong-initialized expanded vocabulary의 새 input/output row에만 zero-initialized additive
> multi-hash branches를 붙여 full-model continued pretraining하고, 이를 ordinary dense embedding과
> LM head로 정확히 접어 배포 graph를 전혀 바꾸지 않으면서, optimizer-equivalent control보다 좋은
> matched-quality 회복과 실제 token-generation speed를 보이는가?

이 질문도 mechanism control과 fresh actual inference가 없으면 engineering ablation에 머문다.

## 가장 가까운 선행

### Vocabulary adaptation과 inference

- [Yamaguchi et al., Computational Linguistics 2026](https://aclanthology.org/2026.cl-1.9/)은
  target-language token expansion을 더 빠른 inference를 위한 표준적 접근으로 다루고, 적은
  target text에서도 embedding initialization과 continual-pretraining 전략을 비교한다.
- [TokAlign, ACL 2025](https://aclanthology.org/2025.acl-long.207/)은 token co-occurrence alignment로
  source와 target vocabulary를 정렬하고 embedding/head를 재배치한 뒤 progressive fine-tuning한다.
- [TokAlign++, 2026](https://arxiv.org/abs/2605.13429)은 alignment lexicon과 빠른 vocabulary
  recovery를 더 강화한다.
- [Teaching Old Tokenizers New Words, Findings EACL 2026](https://aclanthology.org/2026.findings-eacl.341/)
  은 continued-BPE extension과 leaf pruning을 다룬다.
- [VocADT, ICLR 2025](https://openreview.net/forum?id=KxQRHOre9D)는 기존 embedding의 학습 가능한
  선형조합으로 새 vocabulary embedding을 만들고, vocabulary adapter를 학습한 뒤 새 embedding으로
  사용할 수 있게 한다. 현재 가설과 가장 가까운 vocabulary-adaptation comparator다.
- [An Empirical Comparison of Vocabulary Expansion and Initialization, CoNLL 2024](https://aclanthology.org/2024.conll-1.8/)
  은 새 embedding 초기화가 convex-hull 안에 있어야 한다는 분석과 강한 initialization baseline을
  제공한다.

결론: `한국어 vocabulary를 키워 token 수를 줄인다`와 `좋은 initializer로 품질을 회복한다`는
기여가 될 수 없다. EEVE analogue뿐 아니라 VocADT/TokAlign 계열의 strong adaptation control이
publication-scale 비교에 필요하다.

### Hash representation

- [Hash Embeddings, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/file/f0f6ba4b5e0000340312d33c212c3ae8-Paper.pdf)
  은 여러 hash function이 고른 shared vectors의 조합으로 큰 vocabulary를 표현하고 collision을
  관리한다.
- [MultiHashFormer, 2026](https://arxiv.org/abs/2606.28057)는 causal LM에서 token을 unique
  multi-hash signature로 표현하고 Hash Encoder/Decoder를 통해 multilingual vocabulary expansion을
  constant parameter footprint로 수행한다.

현재 방식과 중요한 차이는 배포 graph다. MultiHashFormer와 고전 hash embedding은 hash
representation/decoder가 inference graph에 남는다. JamoFlow 후보는 hash branch를 training에서만
사용하고 최종 checkpoint는 ordinary dense input/output matrices여야 한다. 이 차이가 실제 systems
가치가 되려면 fold 후 동일 dense checkpoint의 quality와 E2E를 측정해야 한다.

### Training-time overparameterization과 exact folding

- [ExpandNets, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/0e1ebad68af7f0ae4830b7ac92bc3c6f-Abstract.html)
  은 compact network의 linear layer를 학습 중 확장하고 inference 전에 대수적으로 축약한다.
- [Small PLMs Can be Fine-tuned as Large Models via Over-Parameterization, ACL 2023](https://aclanthology.org/2023.acl-long.212/)
  은 fine-tuning 동안만 parameter matrix를 확장해 inference latency 증가 없이 성능을 높인다.
- [RepSpec, ICLR 2026](https://iclr.cc/virtual/2026/poster/10008568)은 draft model 학습에 redundant
  linear branches를 넣고 inference 전에 backbone으로 merge한다.

결론: `training-only extra parameters, zero deployment overhead`는 신규성이 아니다. 현재 후보는
새 vocabulary row와 input/output head에 국한된 multi-hash additive parameterization, strong
vocabulary-transfer setting, exact fold 및 actual token-step speed를 하나의 causal experiment로
묶어야 차별화된다.

## 현재 B1 결과가 아직 말하지 못하는 것

Generic residual은 base보다 untied 0.01556, tied 0.02547 BPB 좋았지만 다음을 식별하지 못했다.

1. multi-hash collision coupling이 유용했는가
2. zero-initialized redundant branches의 AdamW update amplification만 유용했는가
3. extra optimizer state가 유용했는가
4. byte/Unicode-derived hash가 arbitrary balanced hash보다 유용했는가
5. 알려진 development prefix 밖에서 quality recovery가 유지되는가
6. folded dense-8K의 짧은 token sequence가 trained E2E 10%로 이어지는가

특히 residual scale을 `1/sqrt(13)`로 두었다고 effective optimizer update가 plain SGD의 2배로
끝나는 것은 아니다. AdamW의 첫 update는 coordinate-wise moment normalization을 하므로, 이상적인
collision-free branch의 effective residual step은 `sqrt(13)`배에 가까워질 수 있다. 실제 bucket은
여러 token gradient를 합치며 global clipping, epsilon, weight decay와 later moments도 개입한다.
따라서 learning-rate sweep보다 실제 update geometry audit과 predeclared control이 먼저다.

## 최소 mechanism experiment

Untied frontier만 먼저 사용한다. 이유는 현재 strongest initializer와 anchor recovery가 untied에서만
통과했고, tied 결과는 품질 frontier가 아니다.

### 재사용 가능한 고정 역할

- `dense_base`: EEVE analogue, 1.454530 BPB
- `multihash_surface`: current 13-way foldable role, 1.438968 BPB

이 두 수치는 known development evidence이며 새 선택 데이터가 아니다.

### 새 역할

1. `update_matched_dense_or_diagonal`
   - 첫 fixed batch에서 관측한 new-row effective update의 norm/projection을 사전 정의한 방식으로
     맞춘다.
   - hash sharing 없이 self-update/extra-state 이득을 흡수한다.
   - continuous learning-rate sweep은 하지 않는다.
2. `balanced_random_multihash`
   - current role과 slot 수, codebook size, lookup 수, residual parameter 수와 per-slot occupancy를
     맞춘다.
   - token bytes/Unicode 의미와 무관한 presealed seed를 사용한다.

### 판정

- current multi-hash가 update-matched control보다 document/contiguous BPB에서 의미 있게 좋아야
  hash-coupled parameterization을 방법 기여로 남긴다.
- current와 random multi-hash 차이가 작으면 surface semantic claim은 삭제한다. 둘 중 stronger인
  generic hash recipe만 fresh stage 후보가 될 수 있다.
- update control이 같거나 더 좋으면 multi-hash branch를 종료하거나 `row-wise optimization recipe`
  로 범위를 낮춘다.
- 어떤 역할도 현재 development dense-2K anchor와의 gap을 악화시키는 방향이면 fresh stage를 열지
  않는다.

이 screen은 post-result exploratory mechanism evidence다. publication confirmatory quality로
승격하지 않는다.

## Fresh quality와 systems contract

Mechanism guard를 통과한 뒤에만 source가 보지 않은 Korean stream을 만들고 최소 3 model seeds로
다음을 같은 raw-byte history에서 비교한다.

1. continued dense-2K
2. ordinary dense-8K strongest initializer
3. foldable dense-8K candidate
4. strongest optimizer/VocADT-like control

Selection은 calibration-only로 고정하고 final loss는 역할과 checkpoint를 봉인한 뒤 한 번 연다.
Quality-qualified folded checkpoint의 timed graph에는 residual, hash lookup, adapter 또는 auxiliary
parameter가 없어야 한다.

최종 primary systems 비교는 dense-2K 대 exact folded dense-8K다.

- batch 1
- controlled same-output와 strict-valid free-running co-primary
- whole-path tokenizer/embedding/head/cache/argmax/synchronization 포함
- E2E point reduction `>=10%`
- bootstrap uncertainty와 model-seed/session stability
- output validity와 deterministic correctness

이 gate 전에는 token count, fertility, random-weight latency, BPB 또는 analytical FLOPs를 실제 효율
개선이라고 부르지 않는다.

## 논문 claim 경계

성공하더라도 첫 논문의 안전한 claim은 다음 범위다.

> Compact Korean vocabulary expansion에서, training-only additive multi-hash reparameterization이
> strong initialization 및 optimizer-equivalent controls보다 quality recovery를 개선했고, exact fold
> 뒤 ordinary dense graph가 matched-quality small-vocabulary baseline보다 measured batch-1 generation을
> 줄였다.

Korean-specific orthography, general multilinguality, large-scale LLM, CUDA serving과 downstream
improvement는 별도 증거 없이는 주장하지 않는다. 현재 true-Jamo branch는 실패했으므로 이름과
초록에서 Jamo/composition을 positive method처럼 전면에 두지 않는다.
