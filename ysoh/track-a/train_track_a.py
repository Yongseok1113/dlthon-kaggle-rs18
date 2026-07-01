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
from PIL import Image, ImageFilter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_augmentation(base_transform, rotation_degrees=12, hflip_p=0.5,
                        use_lab_channels=True, use_residual_map=True,
                        channel_noise_std=0.05):
    """
    base_transform 앞단에 학습용 증강과 도메인 특화 전처리를 추가.

    기본 증강:
    - 수평 뒤집기(hflip)
    - 무작위 회전(±rotation_degrees도)

    도메인 특화 전처리 (패널 먼지/얼룩 특징 강조):
    A) 채널 노이즈 (channel_noise_std): R/G 채널에 학습 시 무작위 노이즈를 추가해서
       모델이 B 채널(청색, 패널 특유 색상) 패턴에 더 의존하도록 암묵적으로 유도.
       단순 B 채널 강조보다 조명 변화에 강건함(모델이 스스로 적응).

    B) LAB 색공간 잔차 채널 (use_lab_channels): L(밝기)를 제거하고 A(적녹 대비),
       B(청황 대비) 채널만 남긴 이미지를 원본에 추가로 이어붙임(채널 수 5).
       조명 변화 영향 없이 색상 정보(먼지의 황갈색 vs 패널의 청색)만 강조.
       단, 모델 입력 채널이 3→5로 바뀌므로 use_lab_channels=True 시 모델의
       첫 레이어를 수정해야 함(build_model에서 처리).

    C) 고주파 잔차 맵 (use_residual_map): 원본 - Gaussian blur = 먼지/얼룩의
       고주파 패턴. 청색 배경(저주파)은 사라지고 먼지 경계선이 도드라짐.
       원본과 concat해서 4채널로 사용하거나, 증강으로만 활용 가능.

    주의: B(LAB)와 C(잔차)는 채널 수를 늘리므로 use_lab_channels=False,
    use_residual_map=False(기본 off)로 두고 채널 노이즈(A)만 기본 활성화.
    채널 수 변경이 필요한 B/C는 실험적 옵션으로 분리.
    """
    augment = T.Compose([
        T.RandomHorizontalFlip(p=hflip_p),
        T.RandomRotation(degrees=rotation_degrees),
    ])

    class AugmentedTransform:
        def __call__(self, img):
            # 기본 증강
            img = augment(img)

            # A) 채널 노이즈: R/G에 노이즈 추가 -> B 채널 상대적 강조
            if channel_noise_std > 0:
                arr = np.array(img, dtype=np.float32)
                h, w = arr.shape[:2]
                noise_rg = np.random.normal(0, channel_noise_std * 255, (h, w, 2))
                arr[:, :, 0] += noise_rg[:, :, 0]  # R 채널
                arr[:, :, 1] += noise_rg[:, :, 1]  # G 채널
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

            return base_transform(img)

    return AugmentedTransform()


def add_residual_channel(img_pil, blur_radius=2):
    """
    방법 C: 원본 이미지에서 Gaussian blur를 뺀 고주파 잔차 맵 생성.
    먼지/얼룩의 경계선이 도드라지고 청색 배경은 사라짐.
    반환: (H, W, 4) numpy 배열 (원본 RGB + 잔차 L)
    """
    import numpy as np
    arr = np.array(img_pil, dtype=np.float32)
    blurred = np.array(img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
                        dtype=np.float32)
    residual = np.abs(arr - blurred).mean(axis=2, keepdims=True)  # (H,W,1)
    residual = (residual / (residual.max() + 1e-8) * 255).astype(np.uint8)
    return np.concatenate([np.array(img_pil), residual], axis=2)  # (H,W,4)


def add_lab_ab_channels(img_pil):
    """
    방법 B: LAB 색공간의 A, B 채널을 원본 RGB에 추가.
    L(밝기, 조명 변화 영향) 제거, A(적녹)/B(청황) 색차 채널만 보존.
    반환: (H, W, 5) numpy 배열 (원본 RGB + LAB의 A채널 + B채널)
    """
    import numpy as np
    import cv2
    arr = np.array(img_pil)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1:2]  # A 채널
    b_ch = lab[:, :, 2:3]  # B 채널
    return np.concatenate([arr, a_ch, b_ch], axis=2)  # (H,W,5)



class PanelDataset(Dataset):
    def __init__(self, df, img_dir, transform, channel_method="none"):
        """
        df: id, label, sample_weight 컬럼을 가진 DataFrame
        channel_method: "none"(기본 RGB), "residual"(+잔차맵), "lab"(+LAB A/B)
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.channel_method = channel_method

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.channel_method == "residual":
            # 방법 C: 원본 RGB + 잔차맵 -> (H,W,4) -> 4채널 텐서
            arr4 = add_residual_channel(img)  # (H,W,4) numpy
            # base_transform(Resize/CenterCrop/Normalize 등)을 각 채널에 적용하되,
            # 추가 채널은 별도로 처리 후 concat
            img_rgb = Image.fromarray(arr4[:, :, :3])
            img_res = Image.fromarray(arr4[:, :, 3])  # 단채널
            t_rgb = self.transform(img_rgb)  # (3, H, W)
            # 잔차 채널은 단순 리사이즈 + 정규화만 적용
            import torchvision.transforms.functional as TF
            res_tensor = TF.to_tensor(img_res.resize(
                (t_rgb.shape[2], t_rgb.shape[1]), Image.BILINEAR
            ))  # (1, H, W)
            img_tensor = torch.cat([t_rgb, res_tensor], dim=0)  # (4, H, W)

        elif self.channel_method == "lab":
            # 방법 B: 원본 RGB + LAB A,B 채널 -> (H,W,5) -> 5채널 텐서
            arr5 = add_lab_ab_channels(img)  # (H,W,5) numpy
            img_rgb = Image.fromarray(arr5[:, :, :3])
            t_rgb = self.transform(img_rgb)  # (3, H, W)
            import torchvision.transforms.functional as TF
            size = (t_rgb.shape[1], t_rgb.shape[2])
            a_ch = Image.fromarray(arr5[:, :, 3]).resize((size[1], size[0]), Image.BILINEAR)
            b_ch = Image.fromarray(arr5[:, :, 4]).resize((size[1], size[0]), Image.BILINEAR)
            a_tensor = TF.to_tensor(a_ch)  # (1, H, W)
            b_tensor = TF.to_tensor(b_ch)  # (1, H, W)
            img_tensor = torch.cat([t_rgb, a_tensor, b_tensor], dim=0)  # (5, H, W)

        else:
            # 기본: RGB 3채널
            img_tensor = self.transform(img)

        label = int(row["label"])
        weight = float(row.get("sample_weight", 1.0))
        return img_tensor, label, weight


def build_model(arch="efficientnet_b0", in_channels=3):
    """
    arch: "resnet18" 또는 "efficientnet_b0" (기본값)
    in_channels: 입력 채널 수.
        3: 기본 RGB
        4: RGB + 잔차맵(방법 C)
        5: RGB + LAB A/B 채널(방법 B)
        3 이외의 값이면 첫 번째 conv 레이어를 교체해서 채널 수를 맞춤.
        단, 사전학습 가중치를 최대한 살리기 위해 추가 채널은 평균값으로 초기화.
    """
    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        if in_channels != 3:
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                                     kernel_size=old_conv.kernel_size,
                                     stride=old_conv.stride,
                                     padding=old_conv.padding, bias=False)
            with torch.no_grad():
                model.conv1.weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    model.conv1.weight[:, c] = old_conv.weight.mean(dim=1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model, weights.transforms()

    elif arch == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        if in_channels != 3:
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(in_channels, old_conv.out_channels,
                                               kernel_size=old_conv.kernel_size,
                                               stride=old_conv.stride,
                                               padding=old_conv.padding, bias=False)
            with torch.no_grad():
                model.features[0][0].weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    model.features[0][0].weight[:, c] = old_conv.weight.mean(dim=1)
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

    # 채널 방법에 따라 입력 채널 수 결정
    channel_method = args.channel_method
    in_channels = {"none": 3, "residual": 4, "lab": 5}[channel_method]
    if channel_method != "none":
        print(f"채널 방법: {channel_method} (입력 채널 수: {in_channels})")

    model, transform = build_model(arch=args.arch, in_channels=in_channels)
    print(f"모델 아키텍처: {args.arch} (입력 채널: {in_channels})")
    model = model.to(DEVICE)

    if args.augment:
        train_transform = build_augmentation(
            transform, rotation_degrees=args.rotation_degrees, hflip_p=args.hflip_prob,
            channel_noise_std=args.channel_noise_std
        )
        print(f"학습 데이터 증강 적용: 수평뒤집기(p={args.hflip_prob}), "
              f"무작위회전(±{args.rotation_degrees}도), "
              f"채널노이즈(R/G std={args.channel_noise_std})")
    else:
        train_transform = transform

    # channel_method에 따라 Dataset의 이미지 전달 방식을 다르게 적용.
    # residual/lab은 채널을 추가한 numpy 배열을 토대로 별도 처리가 필요하므로,
    # PanelDataset에 channel_method를 전달해서 __getitem__ 내부에서 처리.
    train_ds = PanelDataset(train_df, args.train_dir, train_transform,
                             channel_method=channel_method)
    val_ds = PanelDataset(val_df, args.train_dir, transform,
                           channel_method=channel_method)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # CosineAnnealingLR: epoch마다 lr을 코사인 곡선으로 감소.
    # 초반엔 lr이 높아 빠르게 수렴하고, 후반엔 낮아지며 fine-tuning 효과.
    # T_max=args.epochs로 설정해서 전체 epoch 동안 lr이 eta_min까지 감소.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(reduction="none")  # reduction="none" -> 샘플별 가중치 적용 위해

    best_val_auc = -1.0
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)
        current_lr = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch + 1}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_auc={val_auc:.4f}  lr={current_lr:.2e}")
        scheduler.step()

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
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help="Adam 옵티마이저 L2 정규화 강도 (기본 1e-4). "
                              "과적합 억제 효과. 너무 크면 underfitting 주의.")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                         help="validation 비율 (기본 0.2, 이전 0.15에서 상향). "
                              "샘플 수가 많을수록 val_auc 추정이 더 안정적.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", default=True,
                         help="학습 데이터에 수평뒤집기+회전 증강 적용 (기본 활성화)")
    parser.add_argument("--no_augment", dest="augment", action="store_false",
                         help="증강을 끄고 원본 이미지만 사용")
    parser.add_argument("--rotation_degrees", type=float, default=12,
                         help="무작위 회전 범위(±도), 기본 12도 (10~15도 권장 범위 중간값)")
    parser.add_argument("--hflip_prob", type=float, default=0.5,
                         help="수평 뒤집기 적용 확률, 기본 0.5")
    parser.add_argument("--channel_noise_std", type=float, default=0.05,
                         help="방법 A: R/G 채널 노이즈 강도 (0이면 비활성화, 기본 0.05)")
    parser.add_argument("--channel_method", default="none",
                         choices=["none", "residual", "lab"],
                         help="추가 채널 방법. none(기본, 3채널 RGB), "
                              "residual(방법C: +잔차맵 -> 4채널), "
                              "lab(방법B: +LAB A/B -> 5채널). "
                              "residual/lab 선택 시 모델 첫 레이어가 자동 교체됨.")
    args = parser.parse_args()
    main(args)