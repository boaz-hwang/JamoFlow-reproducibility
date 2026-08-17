# Contamination indexed-retrieval correction

> 작성일: 2026-08-12
> 상태: **publication benchmark download와 full-corpus scan 전 결과맹 구현**
> 적용 범위: Korean downstream contamination candidate retrieval

## 1. 기존 공백

`src/jamoflow/contamination.py`의 detector는 exact/near-match 판정의 투명한
correctness reference였지만, 모든 benchmark example과 모든 HPLT document pair를
직접 비교해야 했다. 이 방식은 synthetic test에는 정확해도 publication corpus
전체를 처리할 수 없다. 문서에서 “향후 index가 가속할 수 있다”고 적은 것만으로는
실행 가능한 오염 통제가 아니다.

## 2. Reference-complete retrieval

`IndexedContaminationDetector`는 benchmark 원문을 메모리에만 유지하고 두 index를
만든다.

1. **Exact anchor index:** exact-eligible benchmark마다 전체 benchmark 집합에서 가장
   드문 5-scalar shingle 하나를 결정적으로 선택한다. Exact containment가 있으면 그
   anchor도 반드시 document에 있으므로 exact candidate를 놓치지 않는다.
2. **Near shingle index:** benchmark의 unique 5-scalar shingles를 inverted index에
   넣고 document와 shared unique shingle이 10개 이상인 benchmark만 near candidate로
   올린다. Reference near-match 자체가 최소 10개를 요구하므로 이 pruning에도 false
   negative가 없다.

두 candidate set의 합집합은 최종 판정이 아니다. 모든 candidate를 기존
`compare_document_to_benchmark`로 다시 검사하고 reference가 반환한 match만
인정한다. 따라서 index는 속도만 바꾸며 exact/near threshold, local-span search와
canonicalization semantics를 바꾸지 않는다.

## 3. 공개 안전성

Tracked metadata에는 다음만 남긴다.

- detector version
- benchmark/eligible/index entry count
- benchmark ID, canonical hash와 scalar count로 만든 manifest SHA-256

Benchmark text, shingle, token ID, matched document text와 local span text는 넣지
않는다. 향후 corpus filter report도 aggregate removed-document/match count만 공개하며
개별 HPLT text를 저장소에 넣지 않는다.

## 4. 검증과 남은 실행

고정 exact, one-edit near, repetitive exact와 unrelated fixture에서 indexed 결과가
full reference와 정확히 같았다. 고정 seed로 만든 12개 benchmark의 exact/one-edit
문서와 12개 unrelated 문서에서도 match tuple 전체가 일치했다. 관련 unit test
10개가 통과했다.

아직 public benchmark pinned rows를 내려받아 full publication HPLT source를 scan하지
않았다. 진행 중 MPS training과 CPU-heavy scan을 겹치지 않는 실행 정책 때문이다.
실제 publication stream을 만들 때는 pinned dataset/file hash, reference-equivalence
fixture, input/output stream hash와 aggregate removal count를 모두 검증한 뒤에만
candidate/raw/BPE 공통 train split을 생성한다.

