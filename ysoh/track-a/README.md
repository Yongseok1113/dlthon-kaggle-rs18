# RS-18 Track A: 태양광 패널 Clean/Dusty 분류

Track B(rs-18-track-b)의 정제 결과를 활용해 학습 데이터를 정제한 뒤,
ResNet18을 파인튜닝해 분류기를 학습한다.

## 전체 파이프라인

```
[Track A train_labels.csv] + [Track B submission.csv]
        |
        v
  clean_trainset.py  -> cleaned_train.csv (id, label, sample_weight, exclude_reason)
        |
        v
  train_track_a.py   -> model.pt
        |
        v
  predict_track_a.py -> submission_a.csv
        |
        v
  submit.py           (Kaggle 제출)
```

## 1단계: Track B 결과로 train set 정제

Track B에서 만든 submission.csv(id, mislabel_score, dup_score, ood_score)를
그대로 입력으로 사용한다. Track A와 Track B는 같은 1,366장 train 이미지를
공유한다는 전제.

```bash
python3 clean_trainset.py \
    --labels_csv <track-a 경로>/train_labels.csv \
    --track_b_scores submission_b_v9.csv \
    --out cleaned_train.csv \
    --ood_pct 95 \
    --dup_pct 95 \
    --mislabel_pct 90 \
    --mislabel_mode weight
```

정제 규칙:
- **ood_score 상위 5%**: 학습에서 완전히 제외 (패널이 아닌 이미지)
- **dup_score 상위 5%**: 근접 중복 그룹에서 1장만 남기고 제외
  (데이터 누수 방지. 단, submission.csv만으로는 정확한 그룹핑이 어려워
  단순 정렬 기반 근사를 사용함 — 더 정확한 그룹핑이 필요하면 Track B의
  `dup_score.py`의 `nearest_idx` 정보를 직접 활용하도록 확장 가능)
- **mislabel_score 상위 10%**: 기본은 `--mislabel_mode weight`로 학습
  손실 가중치를 감쇠(완전 제외 아님, 라벨이 실제로 맞을 수도 있으므로).
  `--mislabel_mode exclude`로 하드 제외도 가능.

## 2단계: 분류기 학습

```bash
python3 train_track_a.py \
    --train_dir <track-a 경로>/train \
    --cleaned_train cleaned_train.csv \
    --out_model model.pt \
    --epochs 10 \
    --batch_size 32
```

- 기본 아키텍처: **EfficientNet-B0**(ImageNet 사전학습). `--arch resnet18`로 변경 가능.
  EfficientNet-B0는 ResNet18과 연산량이 비슷하면서 파라미터 효율이 더 좋아,
  1천 장 규모의 소규모 데이터셋에서도 과적합 위험을 크게 늘리지 않고 기본값으로 채택.

- ResNet18(ImageNet 사전학습) 2-class 파인튜닝
- `sample_weight`를 `CrossEntropyLoss(reduction="none")`에 곱해 손실에 반영
  (mislabel 의심 샘플의 기여도를 낮춤)
- `exclude_reason`이 있는 샘플(ood/dup)은 애초에 학습 풀에서 제외
- val_acc 기준 베스트 모델만 저장

## 3단계: 추론 및 제출 파일 생성

```bash
python3 predict_track_a.py \
    --test_dir <track-a 경로>/test \
    --sample_submission <track-a 경로>/sample_submission.csv \
    --model model.pt \
    --out submission_a.csv
```

`--arch`를 학습 때 바꿨다면(`resnet18` 등) 추론 시에도 동일하게 지정해야 함
(기본값은 둘 다 `efficientnet_b0`로 일치되어 있음).

## 4단계: 제출

```bash
python3 submit.py --competition rs-18-track-a --file submission_a.csv --message "v1"
```

## 참고: Track B 작업에서 얻은 교훈 반영

- mislabel_score를 하드 제외 대신 가중치 감쇠로 기본 사용: Track B에서
  mislabel 판단이 애매한 케이스(예: 작업자 인물샷처럼 ood성이 섞인 경우)가
  섞여있어 하드 제외 시 과도하게 데이터를 잃을 위험이 있음을 확인했음.
- ood_score 임계값은 percentile 기반(절대값 아님) 사용: dup_score/ood_score
  모두 절대 스케일이 신뢰하기 어렵다는 점을 Track B에서 확인했음.