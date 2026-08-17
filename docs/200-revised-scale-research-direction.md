# 규모 확장 결과를 반영한 연구 방향 수정

> 작성일: 2026-08-17  
> 상태: trained 19.6M 및 188.6M 실제 추론 완료 후 확정  
> 목적: “모델이 커지면 2.5%가 커지는가?”라는 질문을 결과에 맞게 다시 정의한다.

## 결론부터

모델 크기만 키우면 whitespace patching의 실제 개선율도 커진다는 가설은 지지되지
않았다. 대신 다음 두 사실이 동시에 성립한다.

1. **시스템 상한은 규모와 함께 커질 수 있다.** 같은 random weights에서 W72와 C86의
   controlled 차이는 49.8M의 3.572%에서 1.618B의 10.217%까지 커졌다.
2. **품질을 보존한 trained effect는 자동으로 커지지 않았다.** 19.6M W72는
   2.628% controlled / 2.531% free였고, 188.6M에서는 동일 W72가 품질을 잃었다.
   패치 밀도를 W80으로 완화해 품질을 회복한 뒤에도 2.887% / 2.475%에 머물렀다.

따라서 올바른 연구 질문은 “크기가 커지면 빨라지는가?”가 아니라 다음이다.

> 모델 규모가 키운 global-event 절감 상한 중에서, 품질 제약과 byte-local 순차
> 경로를 통과해 실제 end-to-end 개선으로 전환되는 비율은 무엇이 결정하는가?

## 기존 가설에서 수정한 부분

### 폐기하는 단순 가설

`parameter count 증가 → saved global event의 절대 비용 증가 → E2E 개선율 증가`

첫 번째 화살표는 random-weight systems curve에서 관찰됐지만, 두 번째 화살표는 trained
quality constraint 때문에 성립하지 않았다. 188.6M에서 W72가 품질을 잃은 사실은
모델이 커질수록 같은 압축률을 그대로 유지할 수 있다고 가정하면 안 된다는 직접적인
반례다.

### 새 작업 가설

실제 개선율은 다음 세 항의 결합으로 본다.

`realized E2E gain ≈ removable global-event fraction × event-cost share × quality-feasible density`

- `removable global-event fraction`: candidate가 reference보다 실제로 줄인 patch/global
  update 비율
- `event-cost share`: 전체 생성 시간에서 global event가 차지하는 비중
- `quality-feasible density`: 고정 품질 margin 안에서 실제로 제거할 수 있는 event 비율

여기에 두 schedule이 공통으로 수행하는 byte-local 1-step 경로, selector, host sync,
LM head 같은 고정 비용이 Amdahl 상한을 만든다. Random weights는 주로 두 번째 항을
측정하지만, trained model은 세 번째 항까지 만족해야 한다.

## 현재 데이터가 허용하는 주장

### 허용

- Korean HPLT 기반 동일 BLT graph에서 whitespace-informed relocation은 same-rate
  codepoint placement보다 품질이 좋았다.
- C86 대비 품질을 보존한 실제 latency 감소가 19.6M과 188.6M 두 trained scale에서
  약 2.5--2.9%로 재현됐다.
- random same-weight graph에서는 규모가 커질수록 더 큰 schedule headroom이 나타날 수
  있다.
- random systems headroom과 trained quality-qualified speedup은 서로 다른 estimand다.
- 현재 구현에서는 quality-feasible patch density와 공통 byte-local 경로가 증폭을
  막았다.

### 금지

- 모델 크기가 커질수록 trained speedup이 증가한다는 scaling law
- 188.6M 결과를 순수 scale contrast로 해석하는 것(W72와 W80가 다름)
- 한 seed의 severely undertrained 188.6M 결과를 충분히 학습된 large-model 일반화로
  부르는 것
- random-weight 1.618B의 10.217%를 실제 trained inference speedup으로 인용하는 것
- Apple MPS 결과를 CUDA, server batching 또는 production serving으로 일반화하는 것

## 논문에서의 위치

이번 논문의 중심 주장은 새로운 10% 효율 기법이 아니다. 중심 기여는 다음 네 가지다.

1. 같은 patch rate에서 whitespace와 codepoint boundary placement를 분리한 인과 비교
2. 품질, router/selector 비용, 실제 cached generation을 분리하지 않는 회계
3. random systems headroom과 trained quality-qualified effect의 실증적 분리
4. scale amplification이 실패한 이유를 품질 제약과 Amdahl 병목으로 좁힌 음성 결과

상세 감사 초안은 모든 탐색 경로와 실패를 보존한다. ARR 제출본은 위 네 기여를 8쪽
본문에 집중시키고, 역사적 screen, 전체 provenance state machine, 추가 negative branch,
EXAONE retrieval stress test는 부록 및 재현 패키지로 이동한다.

## 출판 뒤의 확장 연구

현재 결과를 먼저 공개한 뒤 다음 연구는 별도 사전 고정 protocol로 수행한다.
구체적인 estimand, training-adequacy 축, staged gate와 중단 기준은
[품질 제약 규모 frontier 후속 연구 프로그램](./202-quality-constrained-scale-frontier-program.md)에
분리했다. 이 확장은 현재 paper의 결과나 threshold를 바꾸지 않는다.

### A. 순수 규모 효과 분리

- 최소 두 trained scale에서 **동일 W72를 유지하는 fixed-policy estimand**와
  **scale별 calibration-only quality-feasible rate를 쓰는 frontier estimand**를 따로 보고
- scale마다 3개 이상 독립 training seed
- parameter당 학습 byte와 learning-curve checkpoint를 맞추고, 이를 compute-optimal이라고
  과장하지 않음
- 동일 prompt, 동일 output semantics, 동일 hardware에서 actual timing

이 설계 없이는 scale, density와 188.6M screen의 severe undertraining을 분리할 수 없다.

### B. 병목을 직접 줄이는 구조

- 매 byte 공통 local path를 줄이는 block/local speculative mechanism
- whitespace 전용 candidate뿐 아니라 generic UTF-8/scalar 및 generic MTP를 같은 비용으로
  비교
- accepted bytes가 아니라 quality-matched end-to-end time을 primary endpoint로 사용

기존 frozen Hangul draft, local thinning, vocabulary-transfer, retrieval branch가 실패했으므로
같은 candidate를 threshold만 바꿔 재탐색하지 않는다. 새 구조가 필요하다.

### C. 외부 타당성

- CUDA에서 kernel/profile을 포함한 독립 replication
- raw completion과 chat-template workload 분리
- 한국어 외 whitespace-rich language와 delimiter-poor language control
- BPE16K/32K 및 강한 raw-byte comparator를 포함한 품질--비용 frontier

## 즉시 실행 결정

추가로 큰 모델을 더 학습해 “언젠가 증폭될 것”을 찾지 않는다. 현재 두 trained-scale
결과와 공개된 실패를 논문으로 먼저 고정한다. 공개 대상은 코드, aggregate evidence,
프로토콜/감사 trail, 재생성 가능한 그림과 논문이다. 품질을 보존한 새 효율 모델이
없으므로 현 단계에서 Hugging Face에 positive efficient-model weights를 올리지 않는다.
후속 구조가 실제 개선 gate를 통과한 경우에만 별도 모델 릴리스를 연다.

이 결정은 scale 질문을 영구 폐기한다는 뜻이 아니다. **현재 원고 제출 전에 결과를 보고
더 큰 candidate를 추가하지 않는다**는 뜻이다. 후속 연구는 새 untouched evaluation stream,
고정 W-rate와 adaptive frontier의 분리, matched bytes/parameter, 다중 seed와 실제 E2E를
갖춘 독립 연구로 다시 연다.
