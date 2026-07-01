# RS-18 Track B: 오염 샘플 탐지 파이프라인

## 최종 확정 버전: v9 (Public Score: 0.38955) — 최고 기록

---

## 핵심 전략: 세 축의 분리 탐지

데이터 오염을 세 가지 독립적인 축으로 분리해서 각각 다른 방법론으로 점수화한다.
세 축이 서로 거의 독립적(Spearman 상관계수 모두 0.1 미만)이어서, 각 축이 서로
다른 오염 유형을 담당하는 구조가 실측으로 확인됨.

---

## 1. dup_score (근접 중복 탐지)

### 핵심 아이디어
같은 패널을 다른 각도나 시점에서 찍은 거의 동일한 이미지쌍을 탐지한다.
두 가지 신호를 결합:

- **pHash (Perceptual Hash, 가중치 0.6)**: 이미지를 저주파 성분으로 압축한
  256비트 해시 간 해밍 거리. 거의 동일한 이미지에 민감하고 연산이 빠름.
  두 이미지의 해시 거리가 작을수록 유사도가 높음.
- **ResNet18 임베딩 코사인 유사도 (가중치 0.4)**: 512차원 임베딩 공간에서의
  방향 유사도. 약간의 크롭/색보정/리사이즈가 있어도 견고하게 작동함.

각 이미지마다 "가장 유사한 다른 이미지"와의 유사도를 dup_score로 사용(1-NN).

### 핵심 효과
- 실측 검증(train_00965/train_01161 등)에서 진짜 중복쌍을 정확히 상위로 탐지 확인
- pHash는 색상 변화에 강하고, 임베딩은 구도 변화에 강해 두 신호가 상호 보완
- 이중봉 분포(두 쌍이 매우 높고, 나머지는 낮음)를 보여 임계값 설정이 용이

### 실측에서 확인된 한계
- 절대 스케일이 신뢰하기 어려워(최솟값이 0에서 한참 떨어진 곳에서 시작)
  percentile 기반 임계값(상위 5%) 사용이 절대값 임계값보다 안정적
- 정확한 dup 그룹핑(transitive closure)은 미구현,
  "가장 유사한 1개 이웃"만 보는 근사 방식 사용

---

## 2. ood_score (Off-Topic 이미지 탐지)

### 핵심 아이디어
"태양광 패널 사진"이 아닌 이미지를 탐지한다. 실제 데이터에서 발견된 off-topic 유형:
화성 탐사선 사진, 광고/마케팅 배너, 제품 카탈로그 인포그래픽, DIY 콜라주,
50:50 혼합(청소 전/후 비교) 이미지, 작업자 인물 사진 등.

여러 신호를 계층적으로 결합:

**① Hand-crafted 특징 (5종)**
- `grid_regularity_score`: Hough 직선 검출로 격자 패턴 강도 측정
  (패널 특유의 수직/수평 격자가 뚜렷할수록 off-topic 아님)
- `dark_blue_gray_ratio`: 패널 특유의 짙은 청/회색 톤 비율
- `skin_color_ratio`: 피부색(HSV 기반) 비율 — 작업자 인물 사진 탐지
- `green_vegetation_ratio`: 녹색 식물 비율 — 자연 배경 과도한 이미지 탐지
- `watermark_text_ratio`: MSER 기반 고대비 소형 영역 비율 — 워터마크 근사

**② OCR 텍스트 길이**
- pytesseract로 실제 텍스트를 추출해 길이를 측정
- 워터마크/광고배너/인포그래픽의 텍스트를 직접 탐지(MSER 근사보다 훨씬 정확)
- 텍스트가 검출된 이미지(128/1366, 9.4%)의 ood_score 평균이 전체보다 현저히 높음(0.95 vs 0.50)
- 결합 단계에서의 희석을 막기 위해 np.maximum으로 최솟값 보장

**③ CLIP zero-shot 분류 (핵심 신호, 가중치 0.7)**
- "a photo of a solar panel" vs "a random unrelated image" 등의 텍스트 프롬프트로
  이미지의 의미적 off-topic 여부를 직접 질의
- 화성 탐사선(InSight/Perseverance)처럼 시각적으로는 패널과 유사한 원형 장치도
  의미적으로 정확히 구분(실측 확인) — hand-crafted로는 절대 불가능한 케이스
- Hand-crafted 결과(가중치 0.3)와 percentile rank 기반으로 결합

**④ 패치 분산 (혼합 이미지 탐지, 신규)**
- 이미지를 3×3 그리드로 나눠 각 패치의 dusty_prob를 측정하고
  패치 간 표준편차를 계산
- 한 이미지 안에 Clean/Dusty가 혼재(50:50 콜라주, 청소 중 사진)하는 케이스 탐지
- Track A 분류기(model.pt)가 필수 — 밝기 분산 폴백은 역광/노을 등을 오탐함

**⑤ Mask R-CNN 사람 검출 (신규)**
- COCO 사전학습 Mask R-CNN으로 person 클래스 마스크 면적 비율 측정
- skin_ratio(HSV 피부색 근사)보다 훨씬 정확
- 작업자 설치/청소 사진처럼 사람이 화면을 크게 차지하는 이미지를 직접 탐지

### 핵심 효과
- CLIP 도입(v4→v9)으로 +0.0167 점수 향상 — 세 번의 개선 중 가장 큰 단일 기여
- 의미적 off-topic(화성 탐사선, 광고 배너, 제품 카탈로그) 탐지: hand-crafted로는
  불가능한 케이스를 CLIP이 정확히 잡아냄(실측 직접 검증)
- 세 신호가 모두 독립적(상관계수 ~0): dup/mislabel에 영향 주지 않고 ood만 담당

### 실측에서 확인된 한계
- CLIP을 뺀 v14가 가장 낮은 점수(0.358) — CLIP이 핵심임을 역으로 증명
- OCR 신호는 정확하지만 파이프라인 전체 점수를 오히려 낮춤(v13, 0.385)
  → 부분 검증의 정확성이 전체 점수 개선을 보장하지 않음
- 패치 분산의 밝기 폴백: 역광/노을/나무 그림자를 혼합 이미지로 오탐.
  Track A model.pt 없이는 사용 금지

---

## 3. mislabel_score (라벨 오류 탐지)

### 핵심 아이디어
라벨이 의심되는 샘플을 모델의 "예측과 라벨의 불일치"로 탐지한다.
사람 눈으로 1,366장을 전수 검토하지 않고도 라벨 오류를 찾는 방법.

**① ResNet18 임베딩 기반 5-fold Logistic Regression**
- 전체 데이터를 5개 fold로 나눠, 각 샘플이 "한 번도 학습에 쓰이지 않은"
  out-of-fold(OOF) 상태에서 예측됨 → 과적합 없이 신뢰할 수 있는 예측 확률 획득
- 분류기는 단순한 LogisticRegression(임베딩 → Clean/Dusty)
- `score = |P(dusty) - label|`: 예측과 라벨이 크게 불일치할수록 높은 score

**② 강한 정규화 (C=0.1)**
- 512차원 임베딩에 대해 기본값(C=1.0)은 과확신(예측이 0/1 근처로 쏠림)을 유발
- C=0.1로 낮춰 더 보수적인 예측 → mislabel_score 분포가 더 고르고 신뢰도 향상
- C=0.1 도입(v1→v4)으로 +0.0167 점수 향상

**③ Self-training (2회 반복)**
- 1회차 mislabel_score 상위 5%를 2회차 학습에서 제외하고 재학습
- 라벨 노이즈가 제거된 상태에서 더 깨끗한 결정 경계를 학습해
  나머지 샘플들의 mislabel_score 추정 정확도 향상

**④ ood_score 기반 soft exclusion (sample_weight)**
- 학습 시 각 샘플의 가중치 = `1 - ood_score`
- off-topic일수록 학습 기여도를 줄여서 ood성 노이즈가 분류기를 오염시키는 것을 방지
- Hard exclusion(완전 제외)은 OOF 예측 단계에서 해당 샘플을 다른 fold 모델로
  대신 예측하므로, ood 상위 5% + dup 상위 5%에만 적용

**⑤ Dup-label 충돌 보강**
- dup_score가 높은 쌍 중 라벨이 다른 경우(Clean/Dusty가 뒤바뀐 쌍)를 탐지해
  mislabel_score를 추가로 높이는 보강 신호

### 핵심 효과
- 직접 검증(상위 10개 중 6개): 육안으로 확인 시 60%가 명확하거나 그럴듯한 mislabel
- C=0.1 정규화로 라벨 분포 균형 개선 (Dusty:Clean 비율이 더 균형 있게 분포)
- Self-training으로 "라벨이 잘못된 샘플이 분류기를 학습할 때 나쁜 영향을 주는" 악순환 차단

### 실측에서 확인된 한계
- `exclude_mask`는 분류기 **학습**에서만 제외, OOF **예측** 대상에서는 제외 안 됨
  → ood성이 매우 강한 샘플(작업자 인물샷 train_00763: oof_pred=0.001, label=1)이
  mislabel_score 상위에 반복 등장하는 구조적 문제
- 이를 해결하려 한 v10(ood_score 기반 mislabel_score damping)은
  3D 렌더링 이미지가 오히려 상위로 올라오는 역설 발생 → 폐기

---

## 세 축의 관계 구조

```
run_pipeline.py 실행 순서:

[1] embed.py
    └── ResNet18(ImageNet) → 512차원 임베딩

[2] dup_score.py
    └── pHash + 임베딩 코사인 유사도 → dup_score
          ↓
    percentile 상위 5% → exclude_mask (mislabel 학습에서 하드 제외)

[3] ood_score.py
    └── hand-crafted(5종) + OCR + CLIP → ood_score
        + patch_variance(혼합 이미지) + person_ratio(사람 검출) → np.maximum 결합
          ↓
    percentile 상위 5% → exclude_mask 추가
    (1 - ood_score) → sample_weight (mislabel 학습의 소프트 가중치)

[4] mislabel_score.py
    └── exclude_mask 적용 후 5-fold LR(C=0.1) → OOF 예측
        → |P(dusty) - label| → mislabel_score
        → self-training 2회 반복
        + dup-label 충돌 보강
```

---

## 전체 실험 기록 (제출 점수 순)

| 버전 | 구성 | Public Score |
|---|---|---|
| **v9** | CLIP(0.7)+hand-crafted(0.3) rank결합, C=0.1, percentile threshold | **0.38955** |
| v13 | v9 + OCR max-protected 결합 | 0.38502 |
| v4 | C=0.1 정규화 + percentile threshold (CLIP 없음) | 0.37285 |
| v14 | centroid+knn+ocr 단순 rank 평균 (레퍼런스 재현, CLIP 없음) | 0.35790 |
| v1 | pHash+ResNet18+CV (베이스라인) | 0.35624 |
| v10 | ood damping 적용 (미제출, 부작용 확인) | - |
| v11 | 텍스처 특징(contrast_std, high_freq) 임베딩 추가 (미제출, 역효과) | - |

---

## 핵심 교훈 (실험을 통해 확인)

1. **CLIP이 ood_score의 핵심**: CLIP을 뺀 v14가 가장 낮은 점수(0.358).
   hand-crafted 격자 패턴만으로는 의미적 off-topic(화성 탐사선, 광고 이미지)을 못 잡음.

2. **부분 검증 ≠ 전체 성능**: OCR(v13), ood damping(v10), 텍스처 특징(v11)은
   모두 직접 검증에서는 "맞는 방향"으로 보였지만 실제 점수는 하락.
   세 축이 exclude_mask/sample_weight를 통해 간접적으로 얽혀있어
   한 축의 개선이 다른 축에 의도치 않은 영향을 주기 때문.

3. **단순함의 힘**: 레퍼런스 노트북(centroid+knn+ocr, 0.422)은 더 단순한 구조로
   더 높은 점수를 냈지만, 그 구조를 dup_score/mislabel_score가 다른 파이프라인에
   이식하면 0.358로 오히려 낮아짐. 알고리즘이 같아도 조합이 달라지면 결과가 달라짐.

4. **밝기 폴백의 위험**: patch_variance의 밝기 분산 폴백은 역광/노을/나무 그림자를
   혼합 이미지로 오탐. Track A model.pt 없이 사용 금지.

---

## 분포 그래프 확인 과정과 인사이트

### 왜 분포 그래프를 확인했는가

Track B는 ground truth 라벨이 없는 비지도 학습 구조라, 점수의 절대값이 맞는지
직접 확인할 방법이 없다. 유일하게 가능한 자체 검증 방법은 두 가지였다.
1. **상위 샘플 이미지를 직접 열어서 눈으로 확인** (정성적 검증)
2. **점수 분포의 형태를 보고 신호가 제대로 작동하는지 간접 판단** (정량적 진단)

분포 그래프는 두 번째 방법이었다. 제출해서 점수를 받아보기 전에,
"이 파이프라인이 합리적인 출력을 내고 있는가"를 확인하기 위한 사전 진단 수단이었다.

세 종류의 그래프를 확인했다.

---

### 1. dup_score 분포: 이중봉(bimodal) 확인

**확인 이유**: dup_score가 실제로 중복과 비중복을 구분하고 있는지 보려 했다.
중복이 잘 탐지된다면 "진짜 중복"과 "그냥 비슷한 이미지"가 점수 공간에서
명확히 구분되어야 하고, 이는 이중봉 분포로 나타난다.

**확인 결과**: 0.68~0.72 구간에 거대한 봉우리(일반 샘플)와
0.95~1.0 구간에 분리된 작은 봉우리(진짜 중복 후보)가 뚜렷하게 나타났다.
두 봉우리 사이(0.85~0.95)는 거의 비어 있었다.

**인사이트**:
- 이 빈 구간이 "중복이다/아니다"의 경계가 임베딩 공간에서 깔끔하게 갈린다는 뜻이었다
- percentile 95(threshold=0.99)가 정확히 두 번째 봉우리를 잡아내는 위치였다
- 절대 임계값이 아닌 percentile 기반 임계값이 안정적으로 작동한다는 근거가 됐다
- 직접 검증(train_00965/train_01161 쌍)에서 실제로 동일한 이미지임을 확인해
  이중봉 분포와 실제 중복 여부가 일치함을 검증했다

---

### 2. OOF 예측 확률 분포: 과확신 여부 점검

`diagnose_oof.py`를 만들어서 mislabel 분류기의 out-of-fold 예측 확률 분포를
라벨별로 시각화했다.

**확인 이유**: mislabel_score가 `|P(dusty) - label|`로 계산되기 때문에,
OOF 예측 확률 자체가 어떻게 분포하는지가 mislabel_score의 품질을 결정한다.
만약 분류기가 과확신(거의 항상 0 또는 1만 예측)하면, mislabel_score도
0 아니면 1로 양극화되어 미세한 신뢰도 차이를 포착할 수 없게 된다.

그래프에서 두 가지를 함께 봤다.
- **왼쪽**: 라벨별(Clean/Dusty) OOF 예측 확률 분포를 겹쳐서 비교
  → 두 분포가 잘 분리되면 분류기가 실제로 구분력이 있다는 뜻
- **오른쪽**: 전체 OOF 예측 확률 분포 + "중간 구간(0.3~0.7) 비율" 수치
  → 중간 구간 비율이 너무 낮으면 과확신(양극화), 너무 높으면 분류기가 쓸모없는 것

**확인 결과 (v9 기준)**: 과확신(>0.9 또는 <0.1) 비율이 46.6%, 중간 구간 비율이 21.6%.
과확신했는데 라벨과 반대인 경우(강한 mislabel 후보)가 111건(8.1%).

**인사이트**:
- 분류기가 상당히 자신감 있게(46.6%) 예측하고 있었다.
  이건 임베딩 공간에서 Clean/Dusty가 어느 정도 분리되어 있다는 뜻으로 긍정적인 신호
- 동시에 "과확신했는데 틀린 경우(8.1%)"가 mislabel 탐지의 실질적인 신호가 됨
- C=0.1 정규화 도입 이전(C=1.0)에는 이 과확신 비율이 훨씬 높아서 분포가 양끝에
  지나치게 몰려 있었다. C=0.1로 낮추자 분포가 약간 평탄해지면서 중간 구간이 늘었고,
  실제 점수도 +0.0167 향상됐다. 분포 모양의 변화가 점수 향상의 메커니즘을 설명해줬다

---

### 3. ood_score 특징별 분포: 각 신호의 기여도 진단

`diagnose_ood_features.py`로 hand-crafted 특징(grid_regularity, dark_blue_gray,
skin_ratio, vegetation_ratio, watermark_ratio) 각각의 히스토그램과
최종 ood_score와의 Spearman 상관계수를 확인했다.

**확인 이유**: 5개 특징 중 실제로 최종 ood_score에 기여하는 게 무엇인지,
어떤 특징이 노이즈에 가까운지 구분하기 위해서였다.

**확인 결과**:
- `dark_blue_gray`: Spearman -0.314 — 가장 강한 신호. 패널 특유의 짙은 청/회색이
  없을수록(낮을수록) ood_score가 높아지는 관계가 명확했다
- `skin_ratio`: Spearman +0.247 — 두 번째로 강한 신호. 피부색 비율이 높을수록
  ood_score가 높아지는 직관적인 관계
- `watermark_ratio`: Spearman -0.078 — 거의 기여 없음. MSER 기반 워터마크 근사가
  실제로는 노이즈에 가까워서 신뢰할 수 없는 신호임을 수치로 확인했다
  → 이게 OCR 직접 탐지로 교체한 동기가 됐다

**OCR 추가 후 (v12~v13)**: ocr_text_length의 Spearman이 0.502로 다른 모든 특징을
압도하는 가장 강한 신호로 등장했다. OCR 검출된 128장의 ood_score 평균이
전체 평균(0.50) 대비 0.95까지 올라가며 효과가 명확히 드러났다. 다만 실제 제출
점수는 v9보다 낮아져, "지표상의 개선"과 "최종 점수"가 반드시 일치하지 않음을 확인했다.

---

### 종합: 분포 확인이 실제로 도움이 된 것들

| 확인한 것 | 발견한 인사이트 | 실제로 이어진 변화 |
|---|---|---|
| dup_score 이중봉 | 임계값 설정이 percentile 기반으로 충분함 | percentile 95 임계값 채택 |
| OOF 분포 과확신 | C=1.0이 과확신을 유발 | C=0.1로 변경 → +0.017 |
| watermark Spearman 낮음 | MSER 근사가 노이즈 | OCR 직접 탐지로 교체 |
| OCR Spearman 0.502 | OCR 신호가 강력 | 그러나 실제 점수 하락 확인 |

분포 확인은 "이 신호가 작동하는가"를 점수 없이 판단하는 유일한 도구였지만,
"지표가 좋다고 최종 점수도 좋다"는 보장이 없었다는 점이 핵심 교훈이었다.

---



```bash
# 기본 실행 (CLIP 포함, patch variance는 밝기 폴백)
python3 run_pipeline.py --competition rs-18-track-b --out submission_b.csv

# Track A 분류기 + Mask R-CNN 사람 검출 모두 사용 (권장)
python3 run_pipeline.py --competition rs-18-track-b --out submission_b.csv \
    --track_a_model ../track-a/model_v3.pt \
    --track_a_arch efficientnet_b0 \
    --use_person_detection
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `embed.py` | ResNet18 512차원 임베딩 추출 |
| `dup_score.py` | pHash + 임베딩 코사인 유사도로 근접 중복 탐지 |
| `ood_score.py` | hand-crafted + OCR + CLIP + patch_variance + person_ratio |
| `mislabel_score.py` | 5-fold OOF LR(C=0.1) + self-training + dup-label 보강 |
| `run_pipeline.py` | 전체 파이프라인 실행, submission.csv 생성 |
| `diagnose_oof.py` | OOF 예측 확률 분포 시각화 (과확신 여부 진단) |
| `diagnose_ood_features.py` | ood 특징별 분포 및 상관관계 진단 |
| `diagnose_damping.py` | mislabel damping 전후 순위 변화 비교 |
| `visualize_scores.py` | 세 축 점수 분포 시각화 |
| `prepare_dataset_for_upload.py` | 이미지를 리사이즈해서 zip으로 압축 (검증용) |
| `submit.py` | Kaggle 제출 |