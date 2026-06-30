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

from train_track_a import build_model, DEVICE


class TestPanelDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{id_}.jpg")
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), id_


def main(args):
    sample_sub = pd.read_csv(args.sample_submission)
    ids = sample_sub["id"].tolist()

    model, transform = build_model(arch=args.arch)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    ds = TestPanelDataset(ids, args.test_dir, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = {}
    with torch.no_grad():
        for imgs, batch_ids in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # class 1 = Dusty
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
    parser.add_argument("--out", default="submission_a.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)