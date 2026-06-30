"""
Track A(rs-18-track-a) 분류기 학습.
ResNet18(ImageNet 사전학습)을 2-class(Clean/Dusty) 분류로 파인튜닝.

핵심: train set 구성 시 Track B(rs-18-track-b)의 정제 결과를 사용.
  - clean_trainset.py로 만든 cleaned_train.csv(id, label, sample_weight, exclude_reason)를
    그대로 입력받아, exclude_reason이 있는 샘플은 학습에서 제외하고
    나머지는 sample_weight를 손실 가중치로 반영.

사용법:
    # 1) 먼저 Track B 결과로 train set 정제
    python3 clean_trainset.py \
        --labels_csv <track-a>/train_labels.csv \
        --track_b_scores <track-b 제출 파일>.csv \
        --out cleaned_train.csv

    # 2) 정제된 train set으로 학습
    python3 train_track_a.py \
        --train_dir <track-a>/train \
        --cleaned_train cleaned_train.csv \
        --out_model model.pt
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_augmentation(base_transform, rotation_degrees=12, hflip_p=0.5):
    """
    base_transform(EfficientNet/ResNet의 표준 전처리 transform) 앞단에
    학습용 증강을 끼워넣은 transform을 반환.

    - 수평 뒤집기(hflip): 패널이 보통 좌우 대칭적인 구도라 자연스러운 증강
    - 10~15도 범위의 무작위 회전: 약간 기울어진 촬영 각도를 흉내냄
      (너무 큰 회전은 패널 격자 패턴이 비정상적으로 보일 수 있어 범위를 제한)

    base_transform 자체가 Resize+CenterCrop+ToTensor+Normalize를 포함하는
    경우(torchvision의 weights.transforms())가 많아, 증강은 PIL 이미지
    단계에서 먼저 적용한 뒤 base_transform을 그대로 이어붙이는 구조로 작성.
    """
    augment = T.Compose([
        T.RandomHorizontalFlip(p=hflip_p),
        T.RandomRotation(degrees=rotation_degrees),
    ])

    class AugmentedTransform:
        def __call__(self, img):
            img = augment(img)
            return base_transform(img)

    return AugmentedTransform()


class PanelDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        """
        df: id, label, sample_weight 컬럼을 가진 DataFrame
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = int(row["label"])
        weight = float(row.get("sample_weight", 1.0))
        return img, label, weight


def build_model(arch="efficientnet_b0"):
    """
    arch: "resnet18" 또는 "efficientnet_b0" (기본값)
    EfficientNet-B0은 ResNet18과 연산량이 비슷하면서(파라미터 효율이 더 좋음)
    1천 장 규모 데이터에서도 과적합 위험이 크게 늘지 않아 기본값으로 채택.
    """
    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model, weights.transforms()

    elif arch == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet의 classifier는 Sequential(Dropout, Linear) 구조
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model, weights.transforms()

    else:
        raise ValueError(f"지원하지 않는 arch: {arch} (resnet18 또는 efficientnet_b0만 가능)")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, labels, weights in loader:
        imgs, labels, weights = imgs.to(DEVICE), labels.to(DEVICE), weights.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        # 샘플별 가중치를 적용한 손실 (mislabel 의심 샘플의 기여도를 낮춤)
        loss_per_sample = criterion(outputs, labels)
        loss = (loss_per_sample * weights).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    all_labels = []
    all_probs = []
    for imgs, labels, weights in loader:
        imgs, labels, weights = imgs.to(DEVICE), labels.to(DEVICE), weights.to(DEVICE)
        outputs = model(imgs)
        loss_per_sample = criterion(outputs, labels)
        loss = (loss_per_sample * weights).mean()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n += imgs.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1 = Dusty 확률
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # 실제 대회 평가지표가 ROC-AUC이므로, val_acc(0.5 임계값 기준 정확도)보다
    # val_auc를 베스트 모델 선택 기준으로 사용하는 것이 더 정확함.
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # validation fold에 한쪽 클래스만 있는 등 AUC 계산 불가 상황 대비
        auc = float("nan")

    return total_loss / n, correct / n, auc


def main(args):
    cleaned = pd.read_csv(args.cleaned_train)

    # exclude_reason이 있는 샘플(ood/dup/mislabel 하드제외) 제거
    n_before = len(cleaned)
    train_pool = cleaned[cleaned["exclude_reason"].isnull()].copy()
    print(f"정제 전 {n_before}개 -> 학습 사용 가능 {len(train_pool)}개 "
          f"(제외 {n_before - len(train_pool)}개)")

    if "sample_weight" not in train_pool.columns:
        train_pool["sample_weight"] = 1.0

    label_counts = train_pool["label"].value_counts()
    can_stratify = (label_counts.min() >= 2) and (len(train_pool) * args.val_ratio >= len(label_counts))
    if can_stratify:
        train_df, val_df = train_test_split(
            train_pool, test_size=args.val_ratio, stratify=train_pool["label"],
            random_state=args.seed
        )
    else:
        print("[경고] 클래스별 샘플 수가 너무 적어 stratify 분할이 불가능합니다. "
              "일반 무작위 분할로 대체합니다 (소규모 테스트 환경에서 흔히 발생).")
        train_df, val_df = train_test_split(
            train_pool, test_size=args.val_ratio, random_state=args.seed
        )
    print(f"train: {len(train_df)}개, val: {len(val_df)}개")

    model, transform = build_model(arch=args.arch)
    print(f"모델 아키텍처: {args.arch}")
    model = model.to(DEVICE)

    if args.augment:
        train_transform = build_augmentation(
            transform, rotation_degrees=args.rotation_degrees, hflip_p=args.hflip_prob
        )
        print(f"학습 데이터 증강 적용: 수평뒤집기(p={args.hflip_prob}), "
              f"무작위회전(±{args.rotation_degrees}도)")
    else:
        train_transform = transform

    train_ds = PanelDataset(train_df, args.train_dir, train_transform)
    val_ds = PanelDataset(val_df, args.train_dir, transform)  # val은 증강 없이 원본 그대로
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")  # reduction="none" -> 샘플별 가중치 적용 위해

    best_val_auc = -1.0
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)
        print(f"[epoch {epoch + 1}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_auc={val_auc:.4f}")

        # 실제 대회 평가지표(ROC-AUC)에 맞춰 베스트 모델을 선택.
        # val_acc(0.5 임계값 정확도)는 AUC와 다른 지표라 베스트 선택 기준으로
        # 부적절할 수 있음(실측: val_acc는 계속 올랐지만 val_loss는 과적합으로
        # 악화되는 구간에서 베스트가 선택되어, 실제 제출 점수가 하락한 사례 확인).
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.out_model)
            print(f"  -> 베스트 모델 갱신(val_auc 기준), 저장: {args.out_model}")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"\n  {args.patience} epoch 연속 val_auc 개선 없음 -> early stopping")
                break

    print(f"\n학습 완료. 베스트 val_auc={best_val_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Track A의 train 이미지 폴더")
    parser.add_argument("--cleaned_train", required=True,
                         help="clean_trainset.py가 생성한 cleaned_train.csv")
    parser.add_argument("--out_model", default="model.pt")
    parser.add_argument("--arch", default="efficientnet_b0",
                         choices=["resnet18", "efficientnet_b0"],
                         help="사용할 모델 아키텍처 (기본: efficientnet_b0)")
    parser.add_argument("--epochs", type=int, default=20,
                         help="최대 epoch 수 (early stopping으로 보통 더 일찍 끝남)")
    parser.add_argument("--patience", type=int, default=3,
                         help="val_auc가 이 횟수만큼 연속으로 개선되지 않으면 조기 종료. "
                              "0이면 early stopping 비활성화")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", default=True,
                         help="학습 데이터에 수평뒤집기+회전 증강 적용 (기본 활성화)")
    parser.add_argument("--no_augment", dest="augment", action="store_false",
                         help="증강을 끄고 원본 이미지만 사용")
    parser.add_argument("--rotation_degrees", type=float, default=12,
                         help="무작위 회전 범위(±도), 기본 12도 (10~15도 권장 범위 중간값)")
    parser.add_argument("--hflip_prob", type=float, default=0.5,
                         help="수평 뒤집기 적용 확률, 기본 0.5")
    args = parser.parse_args()
    main(args)