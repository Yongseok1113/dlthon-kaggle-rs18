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
from torchvision import models
from sklearn.model_selection import train_test_split
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    for imgs, labels, weights in loader:
        imgs, labels, weights = imgs.to(DEVICE), labels.to(DEVICE), weights.to(DEVICE)
        outputs = model(imgs)
        loss_per_sample = criterion(outputs, labels)
        loss = (loss_per_sample * weights).mean()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


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

    train_ds = PanelDataset(train_df, args.train_dir, transform)
    val_ds = PanelDataset(val_df, args.train_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")  # reduction="none" -> 샘플별 가중치 적용 위해

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"[epoch {epoch + 1}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out_model)
            print(f"  -> 베스트 모델 갱신, 저장: {args.out_model}")

    print(f"\n학습 완료. 베스트 val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Track A의 train 이미지 폴더")
    parser.add_argument("--cleaned_train", required=True,
                         help="clean_trainset.py가 생성한 cleaned_train.csv")
    parser.add_argument("--out_model", default="model.pt")
    parser.add_argument("--arch", default="efficientnet_b0",
                         choices=["resnet18", "efficientnet_b0"],
                         help="사용할 모델 아키텍처 (기본: efficientnet_b0)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)