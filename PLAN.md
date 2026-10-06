# 미니 프로젝트 계획서 (v2)

**Who Gets Heard? Per-Species Performance Disparity in Birdsong Classification and Audio-Language Models**
(새소리 분류 모델과 오디오-언어 모델에서 평균 성능 뒤에 가려진 데이터 희소 종의 성능 격차 분석)

> v1 대비 주요 변경: ① 12종 → 포함 규칙을 만족하는 조류 전체(또는 층화 추출 50~60종), ② 단일 분할 → author 그룹 5-fold 교차검증, ③ "희귀종" → "데이터 희소 종(under-recorded species)"으로 용어 정정, ④ 텍스트 노출도 → 사전학습 빈도 근사(Xeno-canto 전체 녹음 수) 중심으로 변경하고 누수 통제 부분집합 추가, ⑤ site 시뮬레이션 → 해석적 탐지 확률 분석으로 축소, ⑥ RF+SHAP → 혼동 분석으로 대체, ⑦ 분석 계획 사전 등록, ⑧ 일정에 버퍼 추가. 변경 사유는 부록 A 참고.

## 이 프로젝트를 하는 목적

**최종 목표: AICOSS-UTAS 글로벌 학생 프로그램(Birdsong Big Data: AI & Machine Learning Hackathon) 최종 선발**

1. **오디오 AI 경험의 공백을 메운다.**
   NLP와 HCI 중심으로 연구해 왔기 때문에 오디오 데이터를 다뤄 본 경험이 없다. 프로그램 커리큘럼(오디오 처리, 오디오 딥러닝)을 축소판으로 직접 수행해, 면접에서 "audio 경험이 있는가"라는 질문에 실제 결과물로 답한다.

2. **나의 연구 관심이 이 프로그램과 일관되게 연결됨을 보여 준다.**
   나는 중립적으로 보이는 AI가 특정 대상을 체계적으로 불리하게 다루는 문제에 관심이 있다. 새소리 분류에서 데이터가 적은 종이 평균 성능 뒤에 가려지는 문제는 같은 질문의 다른 모달리티 버전이다. 특히 오디오-언어 모델에서 이 종들은 언어 모델의 **롱테일 지식(long-tail knowledge)**, NLP의 **저자원 언어**와 같은 위치에 있다. 도메인은 프로그램에 맞추고, 질문과 방법론은 NLP·멀티모달 연구에서 가져와 두 영역을 잇는다.

3. **다른 지원자와 구별되는 강점(엄밀한 평가 역량)을 드러낸다.**
   모델을 학습시키는 데서 멈추지 않고, 분석 계획 사전 등록, 녹음자 단위 교차검증, 사전학습 누수 통제, 부트스트랩 신뢰구간으로 결과가 의미 있는지 검증한다. 이는 4주차 산업 연계 해커톤에서 팀에 기여할 수 있는 역할을 보여 준다.

4. **현지 프로그램을 끝까지 성실히 완수할 사람임을 보여 준다.**
   지원 전에 스스로 계획을 세우고 기한 안에 결과물을 공개하는 과정 자체가 준비성과 완주 능력의 증거가 된다.

5. **서류와 면접에서 쓸 구체적 소재를 만든다.**
   학업계획서에 넣을 핵심 그림과 결과 요약, GitHub 저장소, 영어 1분 설명을 확보한다. UTAS Birdsong Project와 태즈메이니아 현지 연구가 다룬 문제(데이터 불균형, 소음으로 인한 오탐, 숲 건강도 평가)와 연결한다.

### 프로젝트의 세 층 구조

| 층 | 내용 | 무엇을 보여 주는가 |
|---|---|---|
| 도메인 | 열대 지역 새소리 데이터 (BirdCLEF+ 2025) | 프로그램 적합성 |
| 질문 | 평균 성능이 데이터 희소 종의 소외를 가리는가 | 연구 관심의 일관성 |
| 방법론 접점 | 오디오-언어 모델의 제로샷 성능과 사전학습 빈도의 관계 | NLP·멀티모달 역량과의 연결 |

---

| 항목 | 내용 |
|---|---|
| 기간 | 2026.10.07 (수) ~ 10.12 (월), 6일 (10.13 ~ 14는 서류 작성) |
| 환경 | Python, PyTorch, torchaudio/librosa, scikit-learn, Hugging Face transformers / Kaggle Notebook (GPU, 인터넷 사용 설정) |
| 최종 산출물 | GitHub 저장소 (영문 README), 핵심 그림 1장(2패널), 학업계획서용 요약, 영어 1분 발표 |

---

## 1. 배경과 문제의식

### 1-1. 생태 모니터링과 데이터 불균형

생물음향 모니터링은 녹음기로 어떤 종이 사는지 파악해 생태계 상태를 평가하는 방법이다. UTAS Birdsong Project(2018)는 새소리로 숲의 건강도를 저비용으로 평가하는 것을 목표로 했고, 2025년 태즈메이니아 멸종위기종(가면올빼미, 칼새앵무) 음향 모니터링 연구는 **대상 종의 극심한 데이터 불균형**과 **소음 및 비대상 소리로 인한 오탐**을 핵심 과제로 꼽았다.

**용어 주의**: 이 프로젝트가 측정하는 것은 **학습 데이터 속 녹음 수가 적은 종(data-poor / under-recorded species)**이다. 녹음 수는 녹음자의 접근성과 관심을 반영하며 **생태적 희귀성(개체군 크기, 위협 등급)과 같지 않다.** 두 개념이 얼마나 겹치는지는 선택 분석(5-6)에서 확인한다.

### 1-2. 오디오-언어 모델과 롱테일 편향

**BioLingual**은 동물 소리와 텍스트 설명을 대조학습(CLAP 방식)한 모델로, 종 이름을 텍스트로 넣어 제로샷 분류를 할 수 있다. 비전-언어 모델에서는 사전학습 데이터 속 개념 빈도가 제로샷 성능을 로그선형적으로 결정한다는 결과가 있고(Udandarao et al., 2024; Parashar et al., 2024), 언어 모델도 사전학습에 드물게 등장하는 지식에 약하다(Kandpal et al., 2023). 같은 현상이 생물음향 오디오-언어 모델에서도 나타나는지는 아직 잘 검증되지 않았고, 생태 모니터링의 신뢰성과 직결된다.

### 1-3. 핵심 질문

> 분류 모델의 평균 성능이 높더라도 데이터가 적은 종을 체계적으로 놓친다면, 그 모델로 한 모니터링은 얼마나 믿을 수 있는가? 그리고 이 격차는 지도학습 없이 언어로 종을 지정하는 오디오-언어 모델에서 더 커지는가?

---

## 2. 연구 질문과 사전 등록 가설

| 번호 | 질문 | 가설 (H) | 대응 분석 |
|---|---|---|---|
| RQ1 | 학습 녹음 수가 적은 종일수록 지도학습 모델의 성능이 낮은가? 녹음자 수, 녹음 길이, 품질을 통제해도 그런가? | 종별 F1은 log(녹음 수)와 양의 관계를 보이며, 공변량 통제 후에도 유지된다 | 실험 A |
| RQ2 | 오디오-언어 모델의 제로샷 성능은 사전학습 데이터 속 종별 빈도와 관계가 있는가? 학습 데이터에 없던 녹음에서도 그런가? | 종별 제로샷 recall은 log(AnimalSpeak 캡션 수)와 양의 관계를 보이며, unseen 녹음에서도 유지된다 | 실험 E |
| RQ3 | 데이터 양에 따른 격차는 제로샷 모델에서 지도학습 모델보다 더 큰가? | 기울기는 M3(제로샷) > M4(같은 임베딩 + 지도학습)이다 | 실험 E |
| RQ4 | 이 격차는 모니터링에서 종 탐지 확률을 얼마나 낮추는가? | 하위 3분위 종은 같은 조사 노력에서 탐지 확률이 뚜렷이 낮다 | 실험 C |

- 이 표와 5장의 분석 방법은 **결과를 보기 전(10.07)에 README에 커밋**하고, 커밋 해시를 기록한다. 사후에 추가한 분석은 "exploratory"로 명시한다.
- 가설이 기각되어도 그대로 보고한다.

---

## 3. 데이터

### 3-1. 주 데이터: BirdCLEF+ 2025 (Kaggle)

- 콜롬비아 Middle Magdalena 지역 대상 대회로, 공고의 "열대우림 새소리" 주제와 일치
- `train_audio/`: 종별 초점 녹음(Xeno-canto, iNaturalist 등), 32 kHz `.ogg`
- 규모: 녹음 28,564개, 206종 (조류 외 양서류, 포유류, 곤충, 파충류 포함. 곤충은 대부분 종이 아닌 sonotype 단위)
- `train.csv`: `primary_label`, `secondary_labels`, `rating`, `author`, `collection`(출처), `url`, 위경도 등
- `taxonomy.csv`: 학명, 일반명, iNaturalist taxon ID, `class_name`(Aves / Amphibia / Mammalia / Insecta / Reptilia)
- **이용 조건**: 대회 규칙상 데이터 재배포 불가. 저장소에는 오디오와 특징 캐시를 올리지 않고 다운로드 안내만 둔다.

### 3-2. 종 선정: 규칙 기반 포함

1. `taxonomy.csv`에서 조류(Aves)만 남긴다.
2. 다음 조건을 모두 만족하는 종을 포함한다. 기준값은 EDA 후 **결과를 보기 전에** 확정하고 사전 등록에 포함한다.

| 조건 | 기본값 | 이유 |
|---|---|---|
| 녹음 수 | 8개 이상 | 종별 지표의 최소 표본 |
| 서로 다른 녹음자 수 | 3명 이상 | author 그룹 분할 시 어느 fold에서든 학습 데이터에 그 종이 남도록 하기 위함 |

3. 제외된 종의 수와 그 녹음 수 분포를 보고한다. ("가장 희소한 종은 평가조차 불가능하다"는 사실도 결과의 일부다.)
4. **대체 경로**: 첫날 측정한 처리 속도로 볼 때 적격 종 전체가 어려우면, log(녹음 수) 구간별 층화 무작위 추출(시드 고정)로 50~60종을 고른다. 이 경우에도 BioLingual 계열 실험(M3, M4)은 적격 종 전체에서 수행한다.

### 3-3. EDA에서 반드시 확인할 것

- 종별 녹음 수, 녹음자 수, 녹음 길이 분포
- **`rating` 분포의 출처별 차이**: iNaturalist 녹음이 rating 0(평가 없음)으로 되어 있다면, 품질 필터가 iNat 녹음 전체를 지우게 된다. 이 경우 rating으로 필터링하지 않고, 공변량으로만 사용하거나 Xeno-canto 녹음에 한정해 사용한다.
- `secondary_labels`가 있는 녹음의 비율

### 3-4. 종별 노출도 지표

| 지표 | 수집 방법 | 무엇의 근사인가 |
|---|---|---|
| 학습 녹음 수 | `train.csv` | 지도학습 데이터 양 (RQ1) |
| **AnimalSpeak 내 종별 캡션 수** | BioLingual 학습 데이터 공개본(`AnimalSpeak_limited.csv`, 89만 행, CC BY-NC-SA)에서 `species_scientific` 기준 집계 | **BioLingual 사전학습 빈도의 직접 측정** (RQ2, 주 지표) |
| Xeno-canto 전체 녹음 수 (선택) | Xeno-canto API v3 (`https://xeno-canto.org/api/3/recordings`, 인증된 회원의 API 키 필요, 응답의 `numRecordings`) | 공개 아카이브 전체의 녹음 규모 (보조) |
| 영문 위키피디아 월간 조회수 (선택) | Wikimedia Pageviews API (키 불필요) | 일반 텍스트 노출 (보조) |

- **AnimalSpeak 공개본 확인 결과 (10.06)**: 컬럼은 `caption, caption2, url, source, recordist, species_common, species_scientific`. 출처는 Xeno-canto 60.7만, iNaturalist 25.7만, Animal Sound Archive 1.6만, Watkins 1.5만 행이고, 학명 기준 26,368종이다. 이 덕분에 근사 지표를 쓸 필요 없이 사전학습 빈도를 종별로 직접 셀 수 있다.
- 주의: 파일 이름이 "limited"이므로 논문이 학습에 쓴 전체 데이터보다 작을 수 있다. 따라서 이 수치는 상대적 빈도로 사용하고, 그 점을 한계로 적는다. BirdCLEF와 AnimalSpeak의 학명 표기가 다를 수 있으므로(분류 체계 개정), 매칭되지 않은 종은 수동으로 확인한다.
- 지표가 둘 이상이면 서로의 상관을 먼저 보고한다.
- 집계 스크립트와 결과(`data/exposure.csv`)는 저장소에 포함하고, 원본 CSV는 포함하지 않는다(`data/external/`, gitignore).

### 3-5. 보조 데이터

- **ESC-50** (CC BY-NC): 선택 실험인 소음 강건성 분석(실험 B)에만 사용

### 3-6. 데이터 분할: author 그룹 5-fold 교차검증

- `StratifiedGroupKFold(n_splits=5, group=author, stratify=primary_label)`
- 같은 녹음의 조각도, 같은 녹음자의 녹음도 train과 test에 동시에 들어가지 않는다.
- 각 fold의 학습 데이터에 모든 포함 종이 남아 있는지 assert로 확인한다.
- 모든 녹음이 정확히 한 번씩 test 예측(out-of-fold, OOF)을 받으므로, **녹음이 적은 종도 전체 녹음으로 평가**할 수 있다.
- M2, M3, M4 모두 **같은 fold 정의**를 사용한다. M3는 학습이 없으므로 fold와 무관하게 전체 녹음에서 평가한다.

---

## 4. 방법

### 4-1. 전처리

| 단계 | 설정 |
|---|---|
| 샘플링 레이트 | CNN은 32 kHz 유지, BioLingual은 **48 kHz**로 리샘플링 (`preprocessor_config.json` 확인: `ClapFeatureExtractor`, 최대 10초, 짧으면 `repeatpad`, 길면 `rand_trunc`) |
| BioLingual 입력 | 같은 5초 조각을 그대로 넣는다. 10초 이하라 무작위 자르기(`rand_trunc`)가 일어나지 않아 결과가 결정적이다. |
| 조각화 | 5초 조각, 짧으면 패딩 |
| 조각 수 상한 | 녹음당 에너지 상위 **최대 5개 조각** (학습과 평가에 동일 적용). 종별 녹음 수에는 상한을 두지 않는다 (불균형이 독립 변수이므로). |
| 특징 | log-mel spectrogram, 128 mel bins, float16 memmap으로 1회 캐시 |
| 녹음 단위 예측 | 조각별 확률의 평균 |

### 4-2. 모델

| 모델 | 설명 | 역할 |
|---|---|---|
| M2. CNN | ImageNet 사전학습 EfficientNet-B0를 log-mel로 미세조정 | 지도학습 주 모델 (RQ1) |
| M3. BioLingual 제로샷 | 오디오 임베딩과 종 이름 텍스트 임베딩의 코사인 유사도로 분류 (학습 없음) | RQ2, RQ3 |
| M4. BioLingual 선형 탐침 | 같은 오디오 임베딩 + 로지스틱 회귀 (fold별 학습) | RQ3의 비교 기준. CNN 실패 시 지도학습 주 모델로 대체 |

**M3와 M4 비교의 의미**: 두 모델은 같은 오디오 임베딩을 쓰고, 종을 지정하는 방법만 다르다(텍스트 vs. 지도학습 라벨). 따라서 M3의 격차가 M4보다 크다면, 그 추가분은 텍스트 쪽 표상이나 사전학습 빈도와 관련된다고 해석할 수 있다. 그 원인이 언어 표상 때문인지, 기억 때문인지까지는 단정하지 않는다.

v1의 M1(MFCC + Random Forest + SHAP)은 제외한다. MFCC 계수의 중요도는 생태적으로 해석하기 어렵다. 그 대신 혼동 분석(실험 F)으로 대체한다.

### 4-3. 학습 설정 (M2)

- 손실: cross-entropy (기본). class-weighted CE는 선택 실험 D.
- 증강: 시간 이동, SpecAugment
- **에폭 수 결정**: fold 0의 학습 데이터 안에서 author 그룹으로 검증 세트를 떼어 최적 에폭을 정하고, 모든 fold에 같은 값을 고정한다. test fold로 조기 종료하지 않는다.
- 혼합 정밀도(AMP), 시드 1개 (시간이 남으면 시드를 추가)

### 4-4. 제로샷 프롬프트 (M3)

포함 종 전체를 후보로 하는 닫힌 집합 분류

| 조건 | 프롬프트 예시 |
|---|---|
| P1. 일반명 | "the sound of a {common name}" |
| P2. 학명 | "the sound of {scientific name}" |
| P3. 일반명 + 학명 (선택) | "the sound of a {common name}, {scientific name}" |

- 질문: 노출이 적은 종일수록 프롬프트 선택에 더 민감한가 (종별 |recall(P1) - recall(P2)| vs. log 노출도)

### 4-5. 사전학습 누수 통제 (M3, M4)

BioLingual 학습 데이터(AnimalSpeak)는 Xeno-canto와 iNaturalist에서 수집되었으므로 BirdCLEF 녹음과 겹칠 수 있다. AnimalSpeak 공개본에 원본 URL이 있으므로, 시점으로 추정하지 않고 **녹음 단위로 직접 매칭**한다.

- AnimalSpeak URL에서 ID를 뽑는다. Xeno-canto는 `/XC{id}-`, iNaturalist는 `/sounds/{id}.`. BirdCLEF `train.csv`의 `url`에서도 같은 ID를 뽑아 맞춘다. BirdCLEF 쪽 URL 형식은 EDA에서 확인한다.
- 각 BirdCLEF 녹음을 세 그룹으로 나눈다.

| 그룹 | 기준 | 의미 |
|---|---|---|
| seen | ID가 AnimalSpeak에 있음 | 학습 데이터에 포함됨 (확실) |
| unseen | Xeno-canto ID > 798,785 또는 iNaturalist sound ID > 697,758 (AnimalSpeak의 출처별 최대 ID) | 업로드 시점상 포함될 수 없음 |
| unknown | 그 외 | 판단 불가 |

- unknown이 생기는 이유는 두 가지다. AnimalSpeak의 Xeno-canto 행 중 약 8만 행(13%)은 URL에 XC ID가 없다. 또 공개본이 학습 데이터 전체가 아닐 수 있다. 그래서 매칭이 안 되었다는 것만으로는 unseen으로 보지 않는다.
- 실험 E를 **seen과 unseen에서 각각** 수행해 기울기를 비교한다. unseen에서도 빈도와의 관계가 유지되면 "기억"이 아니라 일반화의 문제라고 말할 수 있다.
- unseen 녹음이 너무 적으면 표본 크기와 함께 보고하고, 해석을 그에 맞게 제한한다.

---

## 5. 실험 및 평가

### 5-1. 지표

- 녹음 단위 top-1 예측 기준: **종별 recall과 F1**, macro F1, accuracy. 제로샷은 top-5 accuracy도 함께 보고한다.
- 모든 지표는 OOF 예측으로 계산한다.
- **신뢰구간**
  - 종별 지표: 해당 종의 녹음을 재표집하는 부트스트랩 1,000회
  - 회귀 기울기와 상관: 종을 재표집하는 부트스트랩 1,000회
- `secondary_labels`가 있는 녹음을 제외한 민감도 분석을 함께 보고한다.

### 5-2. 실험 A: 데이터 양과 성능 격차 (RQ1)

- 종별 F1 vs. log(학습 녹음 수) 산점도와 Spearman ρ
- 회귀: `F1 ~ log(n_rec) + log(n_author) + 평균 녹음 길이 (+ 평균 rating)`, 기울기의 부트스트랩 95% CI
- accuracy와 macro F1의 차이도 보고하되, 불균형 데이터에서 예상되는 현상이므로 단독 결과로 내세우지 않는다.
- 그림: 녹음 수 3분위별로 색을 구분한 산점도 (**핵심 그림 왼쪽 패널**)

### 5-3. 실험 E: 오디오-언어 모델의 롱테일 편향 (RQ2, RQ3)

- M3 종별 recall vs. log(AnimalSpeak 캡션 수), Spearman ρ와 기울기 CI (**핵심 그림 오른쪽 패널**)
- 같은 축에서 M3와 M4의 기울기를 비교하고, 차이에 대한 부트스트랩 CI를 구한다 (RQ3)
- seen vs. unseen 녹음 비교 (4-5)
- 프롬프트 민감도 분석 (4-4)
- 보조 (선택): Xeno-canto 전체 녹음 수, 위키피디아 조회수로 같은 분석

### 5-4. 실험 F: 혼동 분석

- 데이터 희소 종의 오분류가 어디로 가는지 본다: 같은 속(genus)인가, 녹음 수가 많은 종으로 쏠리는가
- "희소 종의 오분류 중 상위 3분위 종으로 간 비율"을 M2와 M3에 대해 각각 비교한다 (다수 클래스로 흡수되는 현상)
- 격차가 데이터 양이 아니라 음향적 유사성 때문일 가능성에 대한 점검이기도 하다

### 5-5. 실험 C: 모니터링 영향 (RQ4, 축소판)

- 종 s의 OOF recall r_s로부터, 그 종이 녹음된 k개 녹음 중 한 번 이상 탐지될 확률 1 - (1 - r_s)^k를 계산하고, k = 1~20에 대해 녹음 수 3분위별 곡선을 그린다.
- 같은 방식으로 precision에서 종별 기대 오탐 수를 계산한다.
- **한계 명시**: 초점 녹음 기반이므로 실제 사운드스케이프 성능보다 낙관적인 상한이다 (Bioacoustics 2024 논문 인용). 녹음 간 독립도 가정한다.

### 5-6. 선택 실험 (시간이 남을 때만)

| 실험 | 내용 |
|---|---|
| D | class-weighted CE 또는 희소 종 오버샘플링 적용 시 실험 A의 기울기 변화 |
| B | 데이터 희소 종이 소음에 더 취약한지: ESC-50 혼합, SNR 20 / 10 / 5 / 0 dB, 3분위별 ΔF1 |
| 희소성 겹침 | IUCN 등급을 붙여 "녹음 수가 적은 종"과 "위협 등급이 높은 종"이 얼마나 겹치는지 표 1개 |
| 시드 반복 | M2 시드 2개 추가 |

---

## 6. 일정

| 날짜 | 작업 | 완료 기준 |
|---|---|---|
| 10.07 (수) | Kaggle 대회 규칙 확인. EDA(3-3), 포함 기준 확정, 5-fold 분할, **사전 등록 커밋**, AnimalSpeak 종별 집계와 녹음 매칭(seen / unseen / unknown), BioLingual 로딩 동작 확인 | 적격 종 표, fold 표, 사전 등록 커밋 해시, `exposure.csv`, 매칭 결과 표 |
| 10.08 (목) | 오디오 디코딩 → log-mel 캐시, BioLingual 임베딩 추출, **M3 제로샷**, M4 선형 탐침 | M3·M4 OOF 예측 파일 |
| 10.09 (금) | M2 에폭 결정(fold 0) → 5-fold 학습 | M2 OOF 예측 파일 |
| 10.10 (토) | 실험 A, 실험 E | **핵심 그림 (2패널)** |
| 10.11 (일) | 오전에 실험 F와 실험 C. **오후는 버퍼** (밀린 작업, 남으면 선택 실험) | 그림 3, 4 |
| 10.12 (월) | README 작성, 학업계획서용 그림과 요약 정리, 저장소 공개 | 저장소 공개 |
| 10.13 ~ 10.14 | 지원 서류 작성 및 제출 (프로젝트는 버그 수정만) | |
| 10.15 ~ 10.18 | 영어 1분 발표 연습, 예상 질문 대비 | |

**처리 속도 판단 시점**: 10.08 오전에 디코딩 속도를 측정한다. 적격 종 전체를 하루 안에 캐시할 수 없다면, 그 즉시 CNN은 층화 추출 50~60종으로 전환한다 (3-2의 4번).

**우선순위 (시간이 부족할 때 줄이는 순서)**: 선택 실험 → P3 → 실험 F → seen/unseen 분석 → M2 (M4를 지도학습 주 모델로 사용). **실험 A, 실험 E(M3 vs. M4 포함), 실험 C는 끝까지 유지**한다.

---

## 7. 저장소 구조

```
who-gets-heard/
├── README.md                # 영문 요약, 핵심 그림, 실행 방법, 한계
├── PREREGISTRATION.md       # 10.07 커밋: 가설, 포함 기준, 분석 방법
├── configs/
│   ├── inclusion.yaml       # 포함 기준값, 시드
│   └── prompts.yaml         # 제로샷 프롬프트 템플릿
├── data/
│   ├── species.csv          # 포함 종, 학명, 일반명, 녹음 수, 녹음자 수, 3분위
│   ├── folds.csv            # 녹음별 fold 배정
│   └── exposure.csv         # Xeno-canto 전체 녹음 수, 위키피디아 조회수
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cache_features.ipynb
│   ├── 03_biolingual.ipynb  # M3, M4
│   ├── 04_cnn.ipynb         # M2
│   └── 05_analysis.ipynb    # 실험 A, E, F, C
├── src/
│   ├── data.py              # 포함 기준, 분할, 조각화
│   ├── features.py          # log-mel 캐시
│   ├── exposure.py          # API 수집
│   ├── train.py
│   ├── zeroshot.py
│   └── evaluate.py          # 부트스트랩, 회귀, 탐지 확률
├── outputs/                 # OOF 예측 (오디오 없음)
└── figures/
```

README의 Limitations에는 다음을 반드시 적는다.
- 초점 녹음 기반 평가가 갖는 낙관성
- 녹음 수와 생태적 희귀성은 다르다는 점
- 사전학습 누수 가능성과 통제 방법
- 노출도 지표들 사이의 상관
- 포함 기준에서 제외된 종

---

## 8. 예상 결과물과 활용처

| 결과물 | 활용처 |
|---|---|
| 핵심 그림 (왼쪽: 지도학습 F1 vs. 학습 녹음 수 / 오른쪽: 제로샷 recall vs. AnimalSpeak 캡션 수, M4 기울기 겹쳐 표시) | 학업계획서, 면접 |
| 3~4줄 결과 요약 | 학업계획서 |
| GitHub 저장소 링크 | 학업계획서, 지원서 프로그래밍 능력 근거 |
| 영어 1분 설명 | 면접 "audio 경험이 있는가" 질문 대비 |
| 확장 아이디어 | 현지 Week 3·4 주제 후보 (사운드스케이프 기반 재평가, NatureLM-audio 등 대형 오디오-언어 모델로 확장, 태즈메이니아 고유종 적용) |

### 학업계획서용 요약 문장 틀

결과에 맞는 쪽을 쓴다. 수치는 모두 OOF 결과와 95% CI로 적는다.

**(가설이 지지된 경우)**
> 열대우림 조류 음향 데이터(BirdCLEF+ 2025)에서 녹음자 단위 교차검증으로 __종 분류 파이프라인을 직접 구축했다. CNN의 전체 정확도는 __%였지만, 종별 F1은 학습 녹음 수와 함께 증가했다(ρ = __, 95% CI __). 녹음 수 하위 3분위 종의 평균 F1은 __로 상위 3분위(__)보다 낮았다. 오디오-언어 모델(BioLingual)의 제로샷 분류에서는 이 격차가 더 가팔라(기울기 __ vs. 지도학습 __), 사전학습 데이터에 드문 개념에 약한 언어·비전 모델의 롱테일 문제가 생물음향에서도 나타남을 확인했다.

**(가설이 지지되지 않은 경우)**
> … 종별 F1은 학습 녹음 수와 뚜렷한 관계가 없었고(ρ = __, 95% CI __), 대신 __(녹음자 수 / 음향적으로 유사한 근연종의 존재)가 성능 차이를 더 잘 설명했다. 이는 데이터를 더 모으는 것만으로는 격차가 줄지 않을 수 있음을 시사한다. …

### 영어 1분 설명 틀

> "Before applying, I built a birdsong classifier on BirdCLEF data from a tropical region in Colombia, evaluating it with author-grouped cross-validation so the model never saw the same recordist in training and testing. Overall accuracy looked good, about __ percent, but per-species results told a different story: species with few training recordings were recognized much less reliably. Because my background is in NLP, I also tested BioLingual, an audio-language model, in a zero-shot setting. Its accuracy tracked how often each species appeared in the model's own training data — the same long-tail pattern we see in language and vision-language models — and the gap was [larger / not larger] than for a supervised model on the same embeddings. In monitoring terms, that means under-recorded species need many more survey recordings to be detected at all. So in this program, I want to learn how audio AI can be evaluated and improved so that it doesn't just work well on average."

(결과가 다르면 해당 문장을 실제 결과에 맞게 고친다.)

---

## 9. 위험 요소와 대응

| 위험 | 대응 |
|---|---|
| 디코딩·캐시가 오래 걸림 | 녹음당 조각 수 상한, 멀티프로세싱, 1회 캐시. 10.08 오전 측정 결과에 따라 층화 추출로 전환 |
| Kaggle GPU 주간 할당량 (약 30시간) | CNN은 시드 1개와 AMP 사용. BioLingual 임베딩은 1회만 추출 |
| 포함 기준을 만족하는 종이 예상보다 적음 | 녹음자 수 기준을 2명으로 완화하고, 그 영향을 보고 |
| CNN 학습이 잘 안 됨 | M4를 지도학습 주 모델로 사용. 프로젝트의 핵심은 평가 |
| 제로샷 성능이 거의 우연 수준 | 그 자체로 결과. top-5와 M4 대비로 보고 |
| unseen 녹음이 너무 적음 | 표본 크기와 함께 보고하고, 누수 통제는 시도했으나 결론을 내릴 수 없었다고 명시 |
| 노출도 지표들이 서로 강하게 상관됨 | 분리 불가를 한계로 명시하고, "노출이 많을수록 유리하다"는 관찰 수준으로만 서술 |
| 결과가 가설과 다름 | 사전 등록대로 보고. 실험 F(혼동 분석)로 대안 설명 탐색 |
| 서류 작성 시간 부족 | 10.11 오후 버퍼, 10.12에 프로젝트 마감 |

---

## 10. 참고 자료

- [The Birdsong Project: monitoring forest health using machine learning (UTAS, 2018)](https://www.utas.edu.au/about/news-and-stories/articles/2018/778-the-birdsong-project-monitoring-forest-health-using-machine-learning)
- [Passive acoustic monitoring of Tasmanian masked owls and swift parrots (bioRxiv, 2025)](https://www.biorxiv.org/content/10.1101/2025.08.07.669245v1)
- [BirdCLEF+ 2025 (ImageCLEF)](https://www.imageclef.org/BirdCLEF2025)
- [Overview of BirdCLEF+ 2025](https://www.researchgate.net/publication/396180283_Overview_of_BirdCLEF_2025_Multi-Taxonomic_Sound_Identification_in_the_Middle_Magdalena_Colombia)
- [Transferable Models for Bioacoustics with Human Language Supervision (BioLingual, arXiv 2308.04978)](https://arxiv.org/abs/2308.04978) · [모델](https://huggingface.co/davidrrobinson/BioLingual) · [코드](https://github.com/david-rx/biolingual) · [AnimalSpeak 데이터](https://huggingface.co/datasets/davidrrobinson/AnimalSpeak)
- [Multi-modal language models in bioacoustics with zero-shot transfer (Scientific Reports, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11871236)
- [An inadequate sampling of the soundscape leads to over-optimistic estimates of recogniser performance (Bioacoustics, 2024)](https://www.tandfonline.com/doi/abs/10.1080/09524622.2024.2353647)
- Udandarao et al., No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance (NeurIPS 2024, arXiv 2404.04125)
- Parashar et al., The Neglected Tails in Vision-Language Models (CVPR 2024, arXiv 2401.12425)
- Kandpal et al., Large Language Models Struggle to Learn Long-Tail Knowledge (ICML 2023, arXiv 2211.08411)
- [Xeno-canto API v3](https://xeno-canto.org/explore/api) · [Wikimedia Pageviews API](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html)
- Piczak, ESC-50: Dataset for Environmental Sound Classification (2015)

---

## 부록 A. v1 → v2 변경 사유

| v1 | 문제 | v2 |
|---|---|---|
| 12종 (4/4/4) | 순열이 C(8,4)=70가지뿐이고 n=12 상관의 검정력이 낮음. 희소 종의 test 녹음은 3~5개 | 규칙 기반 포함(조류 전체 또는 층화 추출 50~60종), 연속 변수 분석 |
| `GroupShuffleSplit` 단일 분할 | 녹음자가 적은 종이 test에서 빠지거나, 학습에서 아예 보이지 않음 | 녹음자 3명 이상 기준 + 5-fold `StratifiedGroupKFold`, OOF 평가 |
| "희귀종" | 녹음 수와 생태적 희귀성을 혼동 | "데이터 희소 종"으로 정정, 겹침 분석은 선택 |
| 위키피디아 노출도, 저자원 언어 비유 | BioLingual의 학습 분포와 관련이 적고, 녹음 수와 분리 불가 | AnimalSpeak 종별 캡션 수(사전학습 빈도 직접 측정)를 주 지표로, 롱테일·개념 빈도 문헌과 연결 |
| 누수는 한계로만 기술 | 통제 가능한데 하지 않음 | AnimalSpeak URL과 녹음 단위로 매칭해 seen / unseen 비교 |
| "데이터 양 vs 언어 표상" 구분 | 설계상 답할 수 없음 | 같은 임베딩을 쓰는 M3 vs. M4 기울기 비교로 범위를 좁혀 질문 |
| 가상 site 200개 시뮬레이션 | 같은 test 세트의 recall을 다시 계산하는 순환 논리, site 구성에 결과가 좌우됨 | 해석적 탐지 확률 곡선 + 낙관적 상한이라는 한계 명시 |
| RF + SHAP | MFCC 중요도는 생태적으로 해석하기 어려움 | 혼동 분석 |
| test 기준 조기 종료 가능성 | test 정보 누출 | fold 0 내부 검증으로 에폭 고정 |
| 결과를 미리 써 둔 요약 틀 | 확증 편향 | 사전 등록 + 결과별 두 가지 틀 |
| 일정 버퍼 없음 | 하루만 밀려도 서류 기간 침범 | 10.11 오후 버퍼, 10.08 오전 전환 판단 시점 |
| `rating` 필터 | iNat 녹음 전체가 빠질 위험 | EDA에서 출처별 분포 확인 후 결정 |
