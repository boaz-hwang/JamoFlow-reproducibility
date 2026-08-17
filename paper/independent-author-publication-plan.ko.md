# 독립 저자 출판 실행 계획

작성 기준일: 2026-08-17

## 1. 결론

대학, 대학원, 연구소 소속이나 학위가 없어도 이 논문을 제출하고 출판할
수 있다. 논문의 저자는 자연인인 황경찬이며, `Priming Water`는 마중물의
영문 사업자등록증에 적힌 현재 소속이다. 개인사업자 자체를 저자로
기재하지 않는다.

현재 연구에 가장 적합한 경로는 다음과 같다.

1. OpenReview에 황경찬 개인 프로필을 만든다.
2. 현재 소속을 `Priming Water`로, 직책을 `Founder` 또는
   `Founder and Independent Researcher`로 사실에 맞게 등록한다.
3. 익명 장문 논문을 2026-10-12 ACL Rolling Review(ARR)에 제출한다.
4. 메타리뷰를 받은 뒤 NAACL 2027 또는 COLING 2027 중 한 곳에
   커밋한다.
5. 채택되면 실명·소속을 넣은 camera-ready 원고를 제출하고 ACL
   Anthology에 출판한다.

ARR는 출판사가 아니라 심사 플랫폼이다. ARR에 PDF를 올린 것만으로는
학회 논문이 되지 않는다. 완성된 리뷰와 메타리뷰를 출판 학회에 커밋하고
채택되어야 정식 논문 출판이 완료된다.

## 2. 권장 저자·소속 표기

영문 저자명은 여권 로마자 표기 `GYEONGCHAN`을 기준으로
`Gyeongchan Hwang`으로 확정했다. 영문 사업자등록증의
`Name of Business`는 `Priming water`이며 논문에서는 일반적인 제목
대문자화에 따라 `Priming Water`로 표기한다.

```text
Gyeongchan Hwang
Priming Water
Seongnam, Gyeonggi-do, Republic of Korea
support@boaz.page
```

권장 원칙은 다음과 같다.

- 영문 저자명은 앞으로 논문, ORCID, OpenReview에서 계속 사용할 한
  가지 표기로 고정한다.
- 사업자등록증의 영문명 `Priming Water`를 일관되게 사용한다. 개인
  사업자이므로 법인으로 오해되는 `Inc.` 또는 `Co., Ltd.`를 붙이지
  않는다.
- 논문 소속에는 직책을 넣지 않아도 된다. OpenReview 경력에는
  `Founder`, `Representative`, `Independent Researcher` 중 실제 역할을
  가장 정확히 설명하는 표현을 쓴다.
- 공개 논문 주소에는 집이나 사업장 도로명 주소를 노출할 필요가 없다.
  등록증의 상세 주소 대신 `Seongnam, Gyeonggi-do, Republic of Korea`를
  사용한다.
- 사업자등록번호는 논문, ORCID, OpenReview에 입력하지 않는다.
- Priming Water를 연구 소속으로 쓰고 싶지 않다면
  `Independent Researcher, Republic of Korea`도 유효한 대안이다. 다만
  현재 사업 활동과 연구가 연결되어 있다면 `Priming Water` 표기가 더
  일관되고 검증하기 쉽다.

심사용 PDF에는 저자명과 소속이 들어가지 않는다. 실명 저자 정보는
OpenReview 제출 폼에만 입력되고, 논문에는 채택 후 camera-ready 단계에서
들어간다.

## 3. 현재 논문으로 주장할 내용

현재 제출 원고의 중심 결과는 다음과 같다.

- 19.6M 한국어 byte-latent 모델에서 W72는 C86 대비 최종 품질을
  비열등하게 유지했다.
- 같은 모델에서 실제 end-to-end 생성 지연은 controlled 2.628%,
  free-running 2.531% 감소했다.
- 188.6M 학습 모델의 사전 봉인된 W80 구조에서도 controlled 2.887%,
  free-running 2.475% 감소가 재현되었다.
- random-weight 그래프에서는 큰 모델에서 10% 이상의 이론적 systems
  headroom이 관찰됐지만, 학습 모델에서는 그 확대가 재현되지 않았다.

따라서 이 논문은 `10% 속도 향상 기법`으로 제출하지 않는다. 한국어
byte-latent 모델에서 causal whitespace boundary가 동일 patch rate의 품질을
개선하고, 품질을 맞춘 실제 추론에서 작지만 재현 가능한 속도 개선을
보이며, random-weight scaling headroom이 학습 모델의 개선으로 자동
전환되지 않는다는 분석 논문으로 제출한다.

현재 권장 분류는 다음과 같다.

- paper type: `Long paper`
- primary area: `Efficient Methods for NLP`
- contribution types:
  - `NLP engineering experiment`
  - `Model analysis & interpretability`
  - `Data analysis`

## 4. 계정과 신원 준비

### 4.1 ORCID

ORCID는 소속이나 학위 없이 무료로 만들 수 있다. 권장 작업은 다음과
같다.

1. <https://orcid.org/register>에서 본인 계정을 만든다.
2. given name은 `Gyeongchan`, family name은 `Hwang`, published name은
   `Gyeongchan Hwang`으로 설정한다.
3. `Also known as`에는 `Boaz Hwang`과 `황경찬`을 추가한다. 필요하면
   여권 순서인 `Hwang Gyeongchan`도 추가할 수 있다.
4. 현재 소속을 `Priming Water`로, 역할을
   `Founder and Independent Researcher`, 시작일을 2020-05로 입력한다.
5. ORCID iD를 OpenReview 프로필에 연결한다.

ORCID는 제출 자격을 주는 증명서가 아니라 동명이인을 구분하고 향후
논문을 한 사람에게 연결하는 영구 식별자다.

### 4.2 공개 전문 프로필 페이지

회사 도메인이 없다면 OpenReview 프로필 승인에 최대 약 2주가 걸릴 수
있다. 다음 정보가 보이는 간단한 공개 페이지를 먼저 준비하는 것이 좋다.

- 영문 이름
- `Founder and Independent Researcher at Priming Water`라는 현재 역할
- OpenReview에 등록할 동일 이메일 주소
- GitHub 프로필 또는 JamoFlow 프로젝트 링크
- 짧은 연구 관심사: Korean NLP, efficient language-model inference,
  byte-level modeling

개인 홈페이지, GitHub 프로필 또는 GitHub Pages로 충분하다. 회사 전체
소개만 있는 일반 랜딩 페이지보다 저자 개인의 이름·소속·이메일이 함께
보이는 페이지가 좋다. 현재 홈페이지 후보는 <https://ax.boaz.page>이고,
공개 이메일 `support@boaz.page` 및 창업자 황경찬을 표시한다. OpenReview
검토를 더 원활하게 하려면 추후 영문 이름과 `Priming Water` 소속도 같은
페이지에 명시하는 것이 좋다.

### 4.3 OpenReview

OpenReview에는 대학 소속이 없어도 가입할 수 있다. 현재 회사나 사업체가
있으면 그 도메인과 직책을 사용할 수 있고, 회사 도메인이 없으면 개인
이메일을 사용할 수 있다.

입력 예시는 다음과 같다.

```text
Preferred publication name: Gyeongchan Hwang
Current position: Founder and Independent Researcher
Institution: Priming Water
Country: Republic of Korea
Institution domain: boaz.page
Start date: 2020-05
Homepage: https://ax.boaz.page
```

공용 이메일을 쓰더라도 허위 대학 소속이나 과거 소속을 현재 소속으로
입력해서는 안 된다. 프로필 정보는 가능한 한 완전하게 채우고 하나의
OpenReview 계정만 사용한다.

## 5. 익명성·코드·preprint 권고

첫 제출의 기본 권고는 다음과 같다.

```text
preprint policy: arr_anonymous_only_until_meta_review
software archive: none_for_anonymous_review
public code release: after_review
```

이유는 현재 GitHub 프로젝트가 실명 계정 및 고유 프로젝트명과 연결되어
있기 때문이다. 해당 저장소를 익명 supplementary software로 첨부하면
저자 신원을 쉽게 추정할 수 있다. 심사용 PDF는 자체 완결적이므로 첫
제출에는 소프트웨어를 첨부하지 않는 편이 안전하다.

ARR 규정은 일반적으로 심사 중 실명 preprint를 금지하지 않는다. 그러나
제출 폼에서 `no non-anonymous preprint`를 선택하면 메타리뷰가 나올 때까지
그 선택을 지켜야 한다. 이 연구에는 익명 ARR preprint만 허용하고,
메타리뷰 후 실명 arXiv와 코드 공개를 결정하는 보수적 선택을 권장한다.

arXiv는 무료 공개 저장소이며 동료심사를 제공하지 않는다. 첫 논문이거나
새 분류에 처음 제출할 때 endorsement가 요구될 수 있다. 그 경우 먼저
제출을 시작해 받은 endorsement 링크를, 이 분야를 알고 자격이 있는
arXiv 저자 한 명에게 정중하게 보낸다. 다수에게 한꺼번에 요청하지 않는다.

## 6. 2026년 제출 일정

현재 공식 일정에서 다음 ARR 마감은 2026-10-12다. 이 주기는 NAACL 2027
및 COLING 2027의 마지막 ARR 제출 기회로 표시되어 있고, 커밋 마감은
2026-12-20이다. 실제 OpenReview 마감 시각은 제출 사이트가 열리면 다시
확인한다.

권장 내부 일정은 다음과 같다.

| 시점 | 행동 |
|---|---|
| 즉시 | 영문 이름, 마중물 영문명, 공개 이메일, 활동 시작일 확정 |
| 2026-08-24 이전 | ORCID와 OpenReview 계정 신청, 공개 개인 프로필 준비 |
| 2026-09 초 | 저자·소속·funding/conflict/AI 사용 공개 문구 확정 |
| 2026-09 중 | 외부 독자 1~2명에게 익명 원고의 논리와 영어 표현 검토 요청 |
| 2026-09 말 | 논문·체크리스트·익명 PDF 최종 승인 및 해시 고정 |
| 2026-10 초 | OpenReview 폼을 미리 작성하고 PDF 렌더링 재확인 |
| 2026-10-11 KST까지 | 마지막 날 위험을 피하여 ARR 제출 완료 권고 |
| 제출 마감 후 48시간 이내 | 저자 reviewer-registration form 완료 |
| 리뷰 공개 후 | 짧고 증거 중심의 author response 작성 |
| 메타리뷰 후 | 수정 재제출 또는 NAACL 2027/COLING 2027 커밋 결정 |
| 채택 후 | 실명 camera-ready, 저자 소속, 코드·arXiv 공개 실행 |

ARR 제출 저자는 모두 제출 후 reviewer registration을 완료해야 한다.
다만 실제 리뷰 배정에는 공식 자격 기준이 있다. 논문 경력이 없는 첫
저자는 경력을 부풀리지 말고 등록 폼에 사실대로 답한다. 자격이 없어
배정되지 않을 수 있지만, 등록 자체를 생략하면 안 된다. 배정을 받으면
기한 내 수행하거나 즉시 공식 절차로 문제를 알려야 한다.

## 7. 실제 제출 순서

1. 저자 선택값을 `paper/private/arr-submission-decisions.json`에 입력한다.
2. 자금 지원이 없으면 `funding: []`를 명시적으로 승인한다.
3. 이해상충이 없으면 `conflicts_of_interest: []`를 명시적으로 승인한다.
4. 인간 저자는 실질 기여한 사람만 넣는다. 현재 단독 연구라면 황경찬
   한 명만 저자로 확정한다. 생성형 AI는 저자가 될 수 없으며 사용 범위를
   Responsible NLP Checklist와 acknowledgments에 공개한다.
5. 다음 감사 명령을 실행한다.

```bash
.venv/bin/python scripts/audit_arr_submission_readiness.py
```

6. 감사가 모두 통과하면 제출용 비공개 묶음을 만든다.

```bash
.venv/bin/python scripts/audit_arr_submission_readiness.py --write-handoff
```

7. `build/arr/main.pdf`와 생성된 OpenReview 폼을 사람이 마지막으로 읽는다.
8. OpenReview에 long paper로 제출한다. 공개 GitHub 링크나 실명 코드
   archive는 익명 첨부물로 넣지 않는다.
9. 제출 후 확인 이메일, forum URL, 제출 PDF 해시를 비공개 기록으로
   보존한다.
10. 리뷰와 메타리뷰가 끝나기 전에는 다른 학술대회나 저널에 동일 논문을
    동시에 제출하지 않는다.

## 8. 제출 전에 저자가 확정해야 할 값

다음 값은 저장소나 Git 기록에서 추측하지 않는다.

- 확정 영문 이름: `Gyeongchan Hwang`
- ORCID name variants: `Boaz Hwang`, `황경찬`
- 소속: `Priming Water`
- 소속 지역: `Seongnam, Gyeonggi-do, Republic of Korea`
- 이메일: `support@boaz.page`
- 홈페이지: <https://ax.boaz.page>
- 대표/연구 활동 시작일: 2020-05-27
- ORCID iD: `0009-0007-5840-3274`
  (<https://orcid.org/0009-0007-5840-3274>, 2026-08-17 이메일 인증 및 공개
  프로필 검증 완료)
- OpenReview profile ID: `~Gyeongchan_Hwang1` (2026-08-17 이메일 확인과
  등록 완료, 운영진 moderation 승인 대기 중; ORCID URL 입력 완료)
- 저자 승인 익명 PDF SHA-256:
  `d9aef7a80cd041f0a645578ba2c54971d58a1a61ef5b84340e221fd36a8fdb42`
- 저자 승인 Responsible NLP 체크리스트 SHA-256:
  `f8e33df1b3bd96f7e27eb81647ed998c93e456acf3f7c4d51ca38cb36d6596fc`
- 저자: `Gyeongchan Hwang` 단독 저자
- 외부 funding 및 이해상충: 없음
- acknowledgments: AI 보조 범위 공개, 별도 인간·기관 감사 대상 없음
- 우선 희망 학회: `NAACL 2027`
- preprint: 메타리뷰 전 ARR 익명 preprint만 허용
- 익명 심사 소프트웨어 첨부: 없음
- 코드 공개: 심사 후, 라이선스는 공개 전에 별도 확정
- 익명 연구 메타데이터 공유: 동의하지 않음

이 값이 모두 확정되기 전에는 실명 PDF, arXiv 원고 또는 외부 제출물을
만들지 않는다.

## 9. 비용과 제도적 제약

- 논문을 쓰거나 OpenReview 프로필을 만드는 데 대학의 승인이 필요하지
  않다.
- arXiv 제출은 무료다.
- 학회에 채택된 뒤에는 최소 한 명의 저자 등록비가 발생할 수 있다.
  2027년 학회의 정확한 등록비와 원격 발표 정책은 아직 각 학회 공지를
  확인해야 한다.
- 현재 연구처럼 사람을 모집하거나 민감한 인간대상 실험을 하지 않은
  모델·시스템 연구는 대학 소속이 없다는 이유만으로 IRB 단계가 생기지
  않는다. 다만 데이터 사용 조건, 개인정보 위험, 계산량, AI 보조 사용은
  ARR Responsible NLP Checklist에 사실대로 답해야 한다.
- 사업 관련 비용 처리와 저작권 귀속은 연구 심사 자격과 별개의 문제다.
  필요하면 한국 세무사 또는 법률 전문가에게 별도로 확인한다.

## 10. 공식 확인 자료

- OpenReview 독립 연구자 가입:
  <https://docs.openreview.net/getting-started/frequently-asked-questions/i-am-an-independent-researcher-how-do-i-sign-up>
- OpenReview 계정 생성과 검토 기간:
  <https://docs.openreview.net/getting-started/creating-an-openreview-profile/signing-up-for-openreview>
- OpenReview 소속 정보 입력:
  <https://docs.openreview.net/getting-started/creating-an-openreview-profile/entering-institutional-data>
- ARR 저자 절차:
  <https://aclrollingreview.org/authors>
- ARR CFP와 익명성·저자·심사 의무:
  <https://aclrollingreview.org/cfp>
- ARR 제출 폼:
  <https://aclrollingreview.org/submissionform>
- ARR 일정과 참여 학회:
  <https://aclrollingreview.org/dates>
- ORCID 가입 자격:
  <https://support.orcid.org/hc/en-us/articles/360006897334-What-is-an-ORCID-iD-and-how-do-I-use-it>
- arXiv 제출과 endorsement:
  <https://info.arxiv.org/help/submit/index.html>
  <https://info.arxiv.org/help/endorsement.html>
