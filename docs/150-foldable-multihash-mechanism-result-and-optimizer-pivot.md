# Foldable multi-hash mechanism-control 결과와 optimizer-only 전환

> 작성일: 2026-08-15
>
> 상태: sealed one-seed development result; multi-hash/Jamo branch 종료
>
> plan commit: `c28c1b6`
>
> result commit: `f41f5e8`
>
> canonical summary SHA-256:
> `874b020505cb431942ac97ed474a98c4c8d52dede7ceabb4e026968f9c9971ad`

## 결론

Foldable multi-hash의 기존 양성 BPB는 shared-hash 또는 Unicode surface assignment의 독립 이점으로
남지 않았다. Historical `untied_generic_surface`는 ordinary dense base보다 좋았지만, 첫 AdamW
update audit에서 고정한 input/output 배수만 적용한 `update_matched_dense`보다 contiguous
`+0.007990 BPB`, document `+0.007953 BPB` 나빴다. Document bootstrap 95% 구간도
`[+0.007596, +0.008315]`로 0에서 충분히 떨어져 있다.

따라서 다음을 확정한다.

1. foldable multi-hash를 fresh Korean multi-seed 또는 actual-inference 단계로 승격하지 않는다.
2. Jamo, Unicode surface, collision-coupled hash representation 기여를 주장하지 않는다.
3. random hash seed나 threshold를 더 탐색하지 않는다.
4. 가장 강했던 ordinary dense control은 별도 **new-row optimization** 가설의 관측 근거로만
   보존한다. 기존 plan의 candidate fallback으로 취급하지 않는다.

## 최종 BPB

모든 역할은 같은 known B1 train/calibration stream, 같은 initial physical checkpoint, model seed,
training order와 512 updates를 사용했다. 새 세 역할은 summary 단계에서 checkpoint를 다시 load해
contiguous와 document NLL을 독립 재계산했고 저장 배열과 bitwise equality를 통과했다.

| role | contiguous BPB | document BPB | 해석 |
|---|---:|---:|---|
| `update_matched_dense` | **1.430979** | **1.430227** | ordinary dense 8K; 최강 역할 |
| `balanced_random_multihash` | 1.438852 | 1.438030 | surface-independent shared hash |
| `untied_generic_surface` | 1.438968 | 1.438181 | historical candidate |
| `stratified_generic_shuffle` | 1.440509 | 1.439716 | exposure/byte-length matched shuffle |
| `untied_base` | 1.454530 | 1.453841 | ordinary dense historical base |

`update_matched_dense`는 base보다 contiguous `0.023551 BPB`, document `0.023613 BPB` 좋았다.
Dense-2K anchor `1.429662/1.428914`와의 격차도 각각 약 `+0.001317/+0.001314 BPB`까지 줄었다.
이는 이번 compact development setting에서 large-vocabulary quality-recovery 문제가 거의 닫혔다는
강한 관측이다. 그러나 한 seed와 이미 알려진 corpus에서 얻은 post-hoc-discovered recipe이므로
fresh noninferiority나 논문 결과는 아니다.

## 사전 고정 gate 판정

### Primary scale control

`generic_surface - update_matched_dense`는 좋아야 하는 방향과 반대로 양수였다.

- contiguous: `+0.007990 BPB`
- document: `+0.007953 BPB`
- bootstrap 95%: `[+0.007596, +0.008315]`
- required: 두 point `<=-0.002`, upper `<=0`
- 판정: **실패**

Historical generic role은 base를 `0.015562/0.015660 BPB` 이기고 anchor-recovery도 통과했지만,
더 강한 causal control을 이기지 못했다. 약한 base 대비 양성 결과를 보존해 primary 실패를 뒤집지
않는다.

### Surface assignment

- generic − balanced random: contiguous `+0.000117`, document `+0.000150`, bootstrap upper
  `+0.000326`
- generic − stratified shuffle: contiguous `-0.001540`, document `-0.001536`, bootstrap upper
  `-0.001412`

Generic surface는 balanced random과 실질적으로 동률이고 shuffle보다 조금 좋았지만, 사전 minimum
`0.002 BPB`를 넘지 못했다. 따라서 `surface_assignment_supported=false`다. 이 결과로 byte/Unicode
surface가 유용한 semantic prior였다고 주장할 수 없다.

### Shared-hash opportunity

Balanced random과 stratified shuffle은 `update_matched_dense`보다 각각 document
`+0.007803`, `+0.009489 BPB` 나빴다. 둘 다 별도 hash 가설을 열기 위한 minimum을 반대 방향으로
실패했고 `random_opportunities_requiring_new_protocol=[]`다.

## 인과 해석

첫-update audit은 multi-hash update가 dense update의 단순 scalar multiple이 아니며 큰 orthogonal
성분을 가진다는 사실을 보였다. 그러나 전체 512-step quality screen에서는 그 복잡한 collision
diffusion이 필요하지 않았다. Dense new-row update의 정렬된 성분만 고정 배수로 증폭한 더 단순한
control이 모든 hash 역할을 명확히 이겼다.

따라서 현재 증거가 지지하는 최소 설명은 다음이다.

> Short-budget vocabulary expansion에서는 초기화 방법만큼 신규 input/output row가 joint CPT에서
> 이동하는 유효 update scale이 중요하다. Foldable multi-hash의 이전 이득은 새로운 표현 구조보다
> 새 row의 optimization dynamics를 간접 변경한 결과로 설명된다.

이것은 multi-hash와 optimizer-equivalence를 증명한 것이 아니다. `update_matched_dense`는
multi-hash의 moment, cross-token direction, bucket collision을 재현하지 않는다. 오히려 그런 잔여
기전이 필요하지 않았다는 empirical rejection이다.

## 최신 선행과 novelty 경계

이 관측만으로 새로운 optimization method를 주장할 수 없다.

- [In-Place Tokenizer Expansion for Pre-trained LLMs](https://arxiv.org/abs/2607.15232)는 신규
  embedding row만 높은 learning rate로 먼저 학습한 뒤 full-model CPT를 수행한다.
- [Beyond Initialization Loss](https://arxiv.org/abs/2608.03494)는 vocabulary expansion에서
  input/output asymmetric initialization과 짧은 CPT probe의 중요성을 체계적으로 보인다.
- FOCUS, OFA, ReTok 및 최근 vocabulary-adaptation 연구도 새 row 초기화·동결·별도 adaptation
  schedule을 이미 다룬다.

따라서 `새 token을 더 크게 학습한다`는 문장 자체에는 충분한 신규성이 없다. 남을 수 있는 질문은
validation loss나 sweep 없이 첫 train batch의 물리적 update geometry에서 input/output 보정을
고정하는 간단한 rule이, 문헌의 embedding-only/high-LR baseline보다 짧은 Korean CPT에서 더
안정적으로 quality를 회복하고 실제 on-device latency 이득을 여는가이다.

## 연구 계획 수정

### 종료하는 분기

- foldable Jamo residual
- generic/surface/random multi-hash reparameterization
- fresh multi-hash multi-seed campaign
- hash branch를 근거로 한 Hugging Face method release

### 별도 새 가설로만 보존하는 분기

`update_matched_dense`의 관측은 기존 mechanism plan의 fallback이 아니다. 다음 protocol은 결과를
보기 전에 별도로 봉인하며 최소한 다음을 포함해야 한다.

1. source가 보지 않은 Korean continuation stream과 별도 calibration stream
2. equal raw-byte history의 dense-2K continuation
3. ordinary dense-8K strongest initialization + standard joint AdamW
4. 고정 asymmetric new-row update amplification
5. 최신 문헌에 맞춘 new-row-only/high-LR 또는 two-stage control
6. input/output multiplier를 validation BPB로 고르지 않는 rule
7. dense-2K 대비 raw-byte BPB noninferiority와 full memory/parameter accounting

현재 8K tokenizer는 같은 128-byte continuation에서 2K보다 token step을 `22.98%` 줄였고,
random-weight dense graph는 E2E를 `19.81%` 줄였다. 이 systems headroom 때문에 위 fresh screen은
실제효율 목표와 직접 연결된다. 반면 candidate가 fresh quality를 못 맞추거나 강한 staged/high-LR
control과 구별되지 않으면 방법 논문 분기는 즉시 닫는다.

Quality를 통과한 trained checkpoint만 다음 actual gate로 간다.

- batch 1 controlled same-output와 strict-valid free-running co-primary
- fastest quality-qualified dense-2K baseline
- point reduction 각 `>=10%`
- prompt/document uncertainty 및 seed/session stability
- tokenizer, cache, sampling, synchronization을 포함한 whole-path timing
- candidate의 추가 resident parameter와 memory를 숨기지 않는 latency--memory Pareto

한 seed fresh screen과 trained actual preflight가 모두 통과한 뒤에만 multi-seed, 더 큰
Mac-feasible scale, CUDA replication, Korean downstream 및 Hugging Face 공개를 연다.

## 논문 방향에 미치는 영향

이 결과는 foldable vocabulary-adaptation positive paper 후보를 현재 형태로 종료한다. W72
boundary-placement manuscript와 섞어 multi-hash를 성공 사례로 넣지 않는다. 대신 두 가지 가능성만
남긴다.

1. fixed new-row optimization이 fresh matched quality와 `>=10%` trained actual latency를 모두
   통과하면, contribution은 **optimizer-aware compact vocabulary expansion for Korean**으로 새로
   정의한다. Multi-hash는 복잡한 실패 대조군/발견 경로로 supplement에 둔다.
2. 통과하지 못하면 positive efficiency claim은 만들지 않고, W72의 안정적 2.5% 효과와 10% gate
   실패, local-byte bottleneck, tokenizer/vocabulary quality--latency tradeoff를 묶은 정직한
   systems diagnostic paper만 완성한다.

어느 경우에도 BPB, token count, random-weight latency를 실제 추론 효율 성공으로 대체하지 않는다.
