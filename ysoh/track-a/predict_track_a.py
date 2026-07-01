"""
Track A 추론: 학습된 모델로 test 이미지의 dusty_prob 예측, submission.csv 생성.

사용법:
    python3 predict_track_a.py \
        --test_dir <track-a>/test \
        --sample_submission <track-a>/sample_submission.csv \
        --model model.pt \
        --out submission_a.csv
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from train_track_a import build_model, add_residual_channel, add_lab_ab_channels, DEVICE


class TestPanelDataset(Dataset):
    def __init__(self, ids, img_dir, transform, channel_method="none"):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform
        self.channel_method = channel_method

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        import torch
        import torchvision.transforms.functional as TF
        id_ = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{id_}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.channel_method == "residual":
            arr4 = add_residual_channel(img)
            img_rgb = Image.fromarray(arr4[:, :, :3])
            img_res = Image.fromarray(arr4[:, :, 3])
            t_rgb = self.transform(img_rgb)
            res_tensor = TF.to_tensor(img_res.resize(
                (t_rgb.shape[2], t_rgb.shape[1]), Image.BILINEAR))
            return torch.cat([t_rgb, res_tensor], dim=0), id_

        elif self.channel_method == "lab":
            arr5 = add_lab_ab_channels(img)
            img_rgb = Image.fromarray(arr5[:, :, :3])
            t_rgb = self.transform(img_rgb)
            size = (t_rgb.shape[1], t_rgb.shape[2])
            a_t = TF.to_tensor(Image.fromarray(arr5[:, :, 3]).resize(
                (size[1], size[0]), Image.BILINEAR))
            b_t = TF.to_tensor(Image.fromarray(arr5[:, :, 4]).resize(
                (size[1], size[0]), Image.BILINEAR))
            return torch.cat([t_rgb, a_t, b_t], dim=0), id_

        else:
            return self.transform(img), id_


def main(args):
    sample_sub = pd.read_csv(args.sample_submission)
    ids = sample_sub["id"].tolist()

    in_channels = {"none": 3, "residual": 4, "lab": 5}[args.channel_method]
    model, transform = build_model(arch=args.arch, in_channels=in_channels)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    ds = TestPanelDataset(ids, args.test_dir, transform,
                           channel_method=args.channel_method)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = {}
    with torch.no_grad():
        for imgs, batch_ids in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)[:, 1]
            for id_, p in zip(batch_ids, probs.cpu().numpy()):
                results[id_] = float(p)
            print(f"  추론 {len(results)}/{len(ids)}")

    submission = pd.DataFrame({
        "id": ids,
        "dusty_prob": [results[id_] for id_ in ids],
    })
    submission.to_csv(args.out, index=False)
    print(f"\n제출 파일 저장 완료: {args.out}")
    print(submission.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--sample_submission", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--arch", default="efficientnet_b0",
                         choices=["resnet18", "efficientnet_b0"],
                         help="학습 시 사용한 아키텍처와 반드시 동일해야 함")
    parser.add_argument("--channel_method", default="none",
                         choices=["none", "residual", "lab"],
                         help="학습 시 사용한 채널 방법과 반드시 동일해야 함")
    parser.add_argument("--out", default="submission_a.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)