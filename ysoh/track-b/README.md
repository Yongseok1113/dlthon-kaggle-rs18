# RS-18 Track B: 오염 샘플 탐지 파이프라인

## 최종 확정 버전: v9 (Public Score: 0.38955) — 최고 기록

## 전체 실험 기록 (제출 점수 순)

| 버전 | 구성 | Public Score |
|---|---|---|
| **v9** | dup: pHash+ResNet18 코사인 / ood: CLIP(0.7)+hand-crafted(0.3), rank 결합 / mislabel: ResNet18+LR(C=0.1)+self-training | **0.38955** |
| v13 | v9 + OCR 텍스트 길이를 ood에 max-protected 결합 | 0.38502 |
| v4 | mislabel만 C=0.1 정규화 (ood는 z-score+watermark, CLIP 없음) | 0.37285 |
| v1 | 베이스라인 (pHash+ResNet18, C=1.0 기본 정규화) | 0.35624 |
| v14 | ood를 레퍼런스 노트북 방식으로 전면 교체(centroid+knn+ocr만, CLIP/hand-crafted 제외) | 0.35790 |
| v10 | mislabel에 ood 기반 damping 적용 (미제출, 실측 검증에서 부작용 확인되어 폐기) | - |

## 핵심 교훈

1. **CLIP이 핵심 자산**: v14(CLIP 제외)가 가장 낮은 점수를 받음 — ResNet18 임베딩
   거리만으로는 화성 탐사선 사진 같은 의미적 off-topic을 못 잡음. CLIP의 zero-shot
   판단력이 ood_score 품질에 결정적으로 기여.
2. **OCR 텍스트 신호는 양날의 검**: 직접 검증(워터마크/광고 이미지)에서는 명백히
   정확했지만, 전체 파이프라인에 추가하니 오히려 점수가 하락(v9→v13). 신호 자체가
   맞아도 결합 방식이나 다른 축과의 상호작용에 따라 전체 점수가 나빠질 수 있음 —
   부분 검증의 정확성이 전체 점수 개선을 보장하지 않음.
2-1. **레퍼런스 코드의 부분 재현은 위험**: 동일한 ood_score 알고리즘이라도 다른
   dup_score/mislabel_score 조합 위에서는 다르게 작동함(세 축이 `exclude_mask`,
   `sample_weight`를 통해 간접적으로 얽혀있음). 다른 프로젝트의 한 구성요소만
   떼어 이식할 때는 반드시 실측 검증 필요, "정확한 재현"이 "더 나은 결과"를
   보장하지 않음.
3. **mislabel_score의 구조적 한계**: `exclude_mask`는 분류기 학습에서만 샘플을
   빼고 OOF 예측 대상에서는 빼지 않으므로, ood성이 매우 강한 샘플(예: 작업자
   인물샷)이 mislabel_score 상위에 계속 등장하는 현상이 모든 버전에서 반복 관찰됨
   (`train_00763`: oof_pred_prob=0.001인데 label=1 → mislabel_score=0.999).
   이 구조를 고치려 한 v10(ood damping)은 더 큰 부작용을 일으켜 폐기함.

## 실행 방법

**중요**: 현재 `ood_score.py`의 `compute_ood_score_handcrafted()`는 OCR 텍스트
길이가 포함된 v13 상태입니다(0.38502). 최고 기록인 v9(0.38955, OCR 미포함)를
정확히 재현하려면 `extract_ood_features()`에서 `ocr_text_length(p)` 항목과
`watermark_text_ratio`/`ocr_text` 관련 `np.maximum` 보호 로직을 제거하고
원래의 5특징(grid/dark/skin/veg/watermark) z-score 결합 버전으로 되돌려야
합니다. 다만 v13과 v9의 점수 차이(0.0045)가 크지 않으므로, 새로 작업을
시작한다면 현재 v13 코드 그대로 사용해도 무방합니다.

```bash
pip install --break-system-packages torch torchvision imagehash opencv-python-headless scikit-learn pandas
# (선택, ood_score 강화용) pip install --break-system-packages git+https://github.com/openai/CLIP.git ftfy regex

python3 run_pipeline.py \
    --train_dir /path/to/train \
    --labels_csv /path/to/train_labels.csv \
    --out submission.csv
```

ResNet18(ImageNet 사전학습) 가중치와 CLIP 가중치는 최초 실행 시 인터넷에서
자동 다운로드됩니다. Kaggle 노트북에서 실행 시 Internet 옵션을 켜두세요.

## 파일 구성

- `embed.py` : ResNet18 기반 이미지 임베딩(512차원) 추출
- `dup_score.py` : pHash + 임베딩 코사인 유사도로 근접 중복 탐지
- `ood_score.py` : hand-crafted 특징(격자 패턴, 색상, 피부색, 식물색) +
  임베딩 k-NN 이상치 탐지. CLIP zero-shot 분류 함수도 포함(권장, 더 강력함)
- `mislabel_score.py` : K-fold cross-validation 분류기의 out-of-fold
  예측과 실제 라벨의 불일치도로 라벨 노이즈 탐지. self-training으로 2회 반복
- `run_pipeline.py` : 전체 파이프라인 실행 및 submission.csv 생성

## 검증 내역

- 5장 샘플 이미지로 dup_score, ood_score, mislabel_score 각 모듈의 코드 흐름 검증 완료
- mislabel_score 로직은 합성 데이터(200 샘플, 20개 인위적 라벨 오염)로 검증:
  상위 20개 예측 중 19개가 실제 오염 샘플과 일치 (강건하게 동작 확인)
- **중요 발견**: hand-crafted ood 특징(Hough 직선 검출 기반 격자 규칙성)이
  DIY 제작 콜라주 이미지에서 오작동함 (나무판/창틀의 직선을 패널 격자로 오인,
  오히려 가장 낮은 ood_score를 부여하는 역효과 확인).
  → **반드시 임베딩 기반 신호(특히 CLIP zero-shot)를 주 신호로 사용**하고
    hand-crafted 특징은 보조 신호로만 작은 가중치를 줄 것을 권장.

## 다음 단계 (본인 환경에서 진행)

1. 전체 1,366장으로 `run_pipeline.py` 실행
2. 각 축 상위 20~30개 이미지를 실제로 열어서 시각 검증
   (스크립트 실행 시 콘솔에 상위 10개가 자동 출력됨)
3. hand-crafted ood_score와 CLIP ood_score를 둘 다 계산해서 상관관계 확인,
   CLIP이 더 합리적이면 가중치를 CLIP 쪽으로 더 주는 것을 권장
4. mislabel_score의 self-training 반복 횟수(`n_iterations`)를 2~4 사이로
   조절하며 상위 샘플 목록이 안정적으로 수렴하는지 확인
5. 세 점수의 분포(히스토그램)를 확인해 극단적으로 한쪽에 몰려있지 않은지 점검
   (AP는 순위만 중요하므로 분포 모양 자체보다 "진짜 의심 샘플이 상위에 오는가"가 핵심)