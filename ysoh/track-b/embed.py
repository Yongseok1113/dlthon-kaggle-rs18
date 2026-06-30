"""
이미지 임베딩 추출 모듈
사전학습 ResNet18(ImageNet)을 특징 추출기로 사용.
GPU 없으면 자동으로 CPU 사용.
"""
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_weights = ResNet18_Weights.IMAGENET1K_V1
_model = resnet18(weights=_weights)
_model.fc = torch.nn.Identity()  # 분류 head 제거, 512차원 특징만 추출
_model.eval().to(DEVICE)

_transform = _weights.transforms()  # 모델에 맞는 전처리(리사이즈/정규화) 자동 적용


@torch.no_grad()
def embed_images(image_paths, batch_size=32, verbose=True):
    """
    이미지 경로 리스트 -> (N, 512) numpy 배열
    """
    embeddings = []
    valid_paths = []
    n = len(image_paths)
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        kept_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(_transform(img))
                kept_paths.append(p)
            except Exception as e:
                print(f"[경고] 이미지 로드 실패: {p} ({e})")
        if not tensors:
            continue
        batch = torch.stack(tensors).to(DEVICE)
        feats = _model(batch).cpu().numpy()
        embeddings.append(feats)
        valid_paths.extend(kept_paths)
        if verbose:
            print(f"  embed {min(i+batch_size, n)}/{n}")
    return np.concatenate(embeddings, axis=0), valid_paths


if __name__ == "__main__":
    # 간단한 동작 테스트
    upload_dir = "/mnt/user-data/uploads"
    paths = [os.path.join(upload_dir, f) for f in sorted(os.listdir(upload_dir)) if f.endswith(".jpg")]
    emb, valid = embed_images(paths)
    print("임베딩 shape:", emb.shape)
    print("사용 디바이스:", DEVICE)