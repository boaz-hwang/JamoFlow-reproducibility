# 최신 선행연구 재검증과 연구 주장 축소

> 작성일: 2026-08-12
> 상태: **Phase 3 마지막 family 실행 중 결과맹 문헌 감사**
> 범위: 2026-08-12 현재 공개된 1차 출처를 다시 대조했다. 이 문서는 실험 결과가 아니다.

## 1. 왜 다시 검증했는가

최초 공유 대화는 `SpaceByte + BLT/Fast BLT + 한글 규칙`의 결합을 넓은 연구 공백으로
해석했다. `01-verification-report.md`와 `02-critical-research-direction-review.md`는 그
공백을 상당히 좁혔지만, 연구가 길어지는 동안 2026년 논문이 추가되었고 원고에는
직계 선행인 AU-Net이 인용되지 않았다. 특히 다음 네 문장을 서로 구별해야 한다.

1. 규칙으로 경계를 정하는가.
2. 높은 계층의 neural compute를 경계에서만 실행하는가.
3. 같은 압축률에서 한국어 경계가 더 나은 위치인가.
4. 동일 품질의 실제 생성에서 wall-clock 시간이 줄어드는가.

앞의 두 질문은 이미 선행연구가 상당 부분 답했다. JamoFlow의 채택 여부는 세 번째를
통제 실험으로 확인하고, 최종적으로 네 번째를 실제 측정으로 통과하는지에 달려 있다.

## 2. 1차 출처 재검증

| 연구 | 1차 출처에서 확인한 범위 | JamoFlow에 미치는 영향 |
|---|---|---|
| [AU-Net](https://arxiv.org/html/2506.14761v1) | raw byte를 regex word boundary에서 모으고, 다시 2단어와 4단어 단위로 pooling한다. 생성 중 깊은 단계는 더 드물게 활성화된다. delimiter가 없는 언어는 future work로 남긴다. | “규칙 계층”이나 “경계에서만 큰 compute” 자체의 신규성은 성립하지 않는다. 한국어 exact-rate 비교와 실제 latency만 남는다. |
| [Fast BLT](https://arxiv.org/abs/2605.08044) | BLT-D, BLT-S, BLT-DV가 여러 byte를 병렬 또는 speculative하게 생성한다. 주요 절감 수치는 실제 wall-clock이 아니라 **estimated memory-bandwidth cost**다. | byte별 출력 step이 병목이라는 판단은 강화되지만, 발표 수치를 JamoFlow의 실제 속도 근거로 사용할 수 없다. |
| [MtPC](https://arxiv.org/abs/2511.11346) | Probabilistic circuit으로 future-byte dependence를 모델링하고 byte-level verifier의 greedy output을 speculative verification으로 보존한다. EvaByte/Llama Byte에서 실제 GPU throughput을 보고하지만 English만 평가한다. | generic independent 또는 joint multi-byte head는 novelty가 아니다. Hangul-aware candidate는 same-cost generic byte-MTP보다 acceptance와 wall time을 추가 개선해야 한다. |
| [ByteFlow](https://arxiv.org/abs/2603.03583) | coding-rate 기반 Top-K byte 선택으로 static graph와 품질·학습 효율을 보고한다. | 각 score가 causal이어도 전체 Top-K membership은 뒤 suffix score에 따라 달라질 수 있다. 보고된 비용을 prefix-causal cached generation 비용으로 간주할 수 없다. |
| [Disentangling Language Modeling and Boundaries](https://arxiv.org/abs/2608.03599) | 언어모델 분포와 boundary 분포를 분리·전이하자는 실험을 제안한다. | 중요한 동시대 가설이지만 제안된 전이 실험의 완료 결과는 아니다. 현재의 separate-training 비교와도 질문이 다르다. |
| [Writing-System-Level Tokenizer Adaptation](https://arxiv.org/abs/2608.00582) | 우크라이나어에서 고정 vocabulary 크기의 tokenizer만 바꾸어 token 수를 줄인다. | sequence-length 절감은 유의미하지만 한국어 품질 또는 동일 출력 실제 latency의 증거는 아니다. |
| [Korean vocabulary pruning](https://arxiv.org/abs/2604.16235) | 한국어 token vocabulary를 줄여 메모리를 아끼지만 latency 개선은 작다고 스스로 제한한다. | vocabulary/parameter 절감과 실제 decode 효율을 분리해야 한다는 직접 근거다. |
| [UTF-8 validity acquisition](https://arxiv.org/abs/2606.14122) | byte LM의 perplexity 안정화보다 유효 UTF-8 생성 습득이 늦으며, 보고 설정에서 대략 두 배의 token이 필요하다. | 작은 모델이 유효 byte열을 자연히 배웠다고 가정할 수 없다. 양쪽 comparator에 동일한 strict UTF-8 DFA 계약이 필요하다. |
| [Morpheme Matters](https://aclanthology.org/2026.eacl-short.22/) | 형태소 기반 한국어 tokenization이 더 짧은 열과 task 품질을 보고한다. | 강한 한국어 관련 baseline이지만 token count를 실제 latency로 치환할 수 없다. |

## 3. 기존 정리에서 유지할 것과 버릴 것

### 유지할 판단

- output-vocabulary mask만으로는 attention, FFN, KV-cache와 memory movement를 줄이지
  못하므로 핵심 기여가 될 수 없다.
- boundary 품질과 압축률을 분리하려면 exact matched patch rate가 필요하다.
- learned entropy router의 detector 비용도 전체 추론 비용에 포함해야 한다.
- teacher-forced FLOPs나 global-position 수는 생성 latency의 대리변수일 뿐이다.
- 한국어 NFC, spacing, 혼합 script, Unicode corruption을 분리한 진단이 필요하다.

### 철회하거나 더 좁혀야 할 판단

- **철회:** 규칙 기반 boundary hierarchy가 새롭다. AU-Net, SpaceByte, Dynamic Token
  Pooling이 반례다.
- **철회:** 여러 byte를 한 번에 생성하는 축이 새롭다. Fast BLT와 Learn Your Tokens,
  zip2zip이 직접 선행이다.
- **축소:** 한글 codepoint 경계는 독점적인 언어 규칙이다. UTF-8 lead/continuation
  상태만으로 모든 script에 동일하게 얻을 수 있으며 SpaceByte도 이를 활용한다.
- **축소:** token 수 또는 global call 감소가 곧 효율 개선이다. 최종 판정은 같은 raw
  output을 생성하는 batch-1 cached wall-clock과 품질 non-inferiority다.
- **유보:** learned router보다 규칙 router가 싸다는 사실만으로 우월하다. 경계 위치가
  나빠진 품질을 회복하기 위한 추가 compute까지 포함한 Pareto 비교가 필요하다.

## 4. 현재 방어 가능한 연구 질문

현재 Phase 3가 직접 답할 수 있는 질문은 다음 하나다.

> 동일한 byte-latent backbone과 정확히 같은 global-position rate에서, 이미 관측된
> 한국어 공백 쪽으로 causal patch start를 이동하는 parameter-free 정책이 generic
> UTF-8 codepoint cadence보다 BPB를 개선하는가? 그리고 realized-rate, detector 비용,
> strict-valid output과 동일 raw-byte output을 모두 통제했을 때 SpaceByte, entropy
> routing 및 BPE 대비 실제 batch-1 생성 Pareto frontier에 남는가?

이 질문의 첫 문장은 boundary-placement 메커니즘 연구이고, 둘째 문장만 사용자의
최종 가치 기준인 **실제 추론 효율 개선**을 판정한다. 첫 문장만 양성이면 분석 결과는
될 수 있지만 효율 기법의 성공으로 발표하지 않는다.

현재 기여 주장은 다음처럼 제한한다.

1. 한국어 공백과 generic codepoint event를 동일 rate·동일 graph에서 분리한 통제 실험
2. learned detector, padding, selector, parameter memory를 포함한 total-cost 분석
3. 동일 raw prompt와 동일 최소 raw-byte output에 대한 cached generation wall-clock
4. NFC·NFD·혼합 script·spacing noise 및 공개 Korean OOD에서의 실패 범위

새 architecture, 최초 linguistic boundary, 최초 hierarchical compute라는 주장은 하지
않는다.

## 5. 결과에 따른 연구 경로

### A. W가 quality와 실제 latency를 모두 통과

compact 결과를 독립 scale/domain 실험에서 재현한다. BPE-16K와 BPE-32K 각각에 대해
동일 raw context, 동일 최소 raw output, strict UTF-8 validity를 유지하고 end-to-end와
decode latency의 paired confidence interval을 모두 제시한다. 이때만 Korean inference
efficiency의 positive paper로 진행한다.

### B. W가 BPB/경계 품질은 개선하지만 BPE 실제 latency를 이기지 못함

positive efficiency 결론은 기각한다. 원인은 byte-by-byte output step인지, global
trunk인지 profiler로 분해한다. byte step이 지배적이면 별도의 사전등록된 후속 연구로
Fast-BLT식 verified block generation 또는 안전한 speculative proposal을 시험할 수
있다. 기존 실험에 사후로 붙여 최초 가설을 구제하지 않는다.

### C. W가 same-rate codepoint 또는 SpaceByte를 이기지 못함

한국어 spacing relocation을 주 방법으로 중단한다. entropy/router 비교는 negative
boundary result와 total-cost benchmark로만 정리한다. morphology나 Jamo를 근거 없이
추가해 가설을 이동하지 않는다.

### D. compact에서만 양성이고 scale/OOD에서 실패

일반적인 효율 기법 주장을 철회하고 scale/domain failure를 명시한다. 더 큰 모델에서
다시 이길 것이라는 외삽으로 결론을 대신하지 않는다.

## 6. 병렬 실행 원칙

정확도와 연구 깊이를 지키기 위해 “파일이 다르다”가 아니라 “증거와 자원이
독립적이다”를 병렬화 조건으로 사용한다.

병행 가능한 작업:

- 단일 MPS 학습 중 정적 문헌 검증과 원고 정정
- 모델을 import하지 않는 schema/hash unit test 작성
- 결과 숫자를 읽지 않는 artifact 존재·완결성 확인
- private 원문을 출력하지 않는 corpus integrity 감사

순차 실행해야 하는 작업:

1. 동일 장치의 학습·추론·전체 test suite
2. 9개 initial run 완료 전의 개별 결과 열람
3. initial blind summary → all-policy summary → mechanism analysis
4. compact gate → publication-scale 실행
5. runtime equivalence → warm-up → latency timing → final value gate

특히 publication timing 중에는 문서 빌드나 테스트도 병행하지 않는다. 시스템 부하를
추가하지 않는 것이 wall-clock 결과의 신뢰성에 더 중요하다.

## 7. 현재 결론

최초 공유 대화가 제안한 넓은 “Jamo rule-guided adaptive LLM”은 그대로는 새롭지도,
실제 효율을 보장하지도 않는다. `01`의 비판과 `00`의 exact-rate 방향 전환은 대체로
옳았지만, 최종 논문 가치는 여전히 미확정이다. 남은 연구 공백은 매우 좁다.

> 한국어에서 싼 observed boundary가 같은 압축률의 generic boundary보다 더 좋은
> 위치인지, 그리고 그 차이가 detector-inclusive·matched-output **실제 생성 시간**의
> 유의미한 개선으로 이어지는지를 처음부터 끝까지 같은 evidence lineage로 검증한다.

실측 개선이 없으면 주제의 효율 기법 가치는 “없음”으로 판정한다. 그 경우에도 결과를
숨기지 않고 병목을 규명한 뒤, multi-unit verified generation을 새로운 독립 가설로
넘기는 것이 올바른 다음 연구 방향이다.
