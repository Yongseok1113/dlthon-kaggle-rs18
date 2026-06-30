"""
패널 표면의 먼지/얼룩 같은 미세 텍스처 차이를 포착하는 보조 특징 추출 모듈.

배경: ResNet18(ImageNet 사전학습) 임베딩만으로는 "약간의 얼룩/먼지"처럼
저수준 텍스처 차이를 충분히 구분하지 못해, 매우 깨끗한 패널을 Dusty로
과확신하는 오탐이 다수 관찰됨(실측 검증 결과). ImageNet은 객체/형태 인식에
최적화되어 있어 표면 텍스처의 미세한 불균일성에는 상대적으로 둔감함.

이 모듈은 다음 신호를 추출:
1) local_contrast_std: 국소 영역별 명도 표준편차의 분산 -> 얼룩/먼지가
   있으면 표면이 불균일해지므로 이 값이 커짐
2) high_freq_energy: 고주파 성분의 에너지 -> 깨끗한 패널은 격자선 외에는
   매끈하지만, 먼지가 쌓이면 미세한 고주파 노이즈가 늘어남
3) glare_uniformity: 반사광(glare) 영역의 매끈함 -> 깨끗한 패널은 빛 반사가
   매끈한 그라데이션을, 더러운 패널은 반사가 불균일하게 끊긴 패턴을 보임
"""
import numpy as np
import cv2
from PIL import Image


def _to_gray(path, max_side=384):
    img = Image.open(path).convert("L")  # 텍스처 분석은 그레이스케일로 충분
    img.thumbnail((max_side, max_side))
    return np.array(img)


def local_contrast_std(gray, block_size=16):
    """
    이미지를 block_size 단위 패치로 나눠 각 패치의 표준편차(국소 대비)를 구하고,
    그 표준편차들의 분산을 반환. 얼룩/먼지가 있으면 패치별 대비가 들쭉날쭉해져
    이 값이 커짐. 완전히 균일한 표면(매우 깨끗)이나 완전히 균일하게 더러운
    표면은 오히려 낮게 나올 수 있어, "불균일성"을 직접 포착하는 지표.
    """
    h, w = gray.shape
    contrasts = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            patch = gray[y:y + block_size, x:x + block_size]
            contrasts.append(patch.std())
    if not contrasts:
        return 0.0
    return float(np.std(contrasts))


def high_freq_energy(gray):
    """
    FFT 고주파 성분의 에너지 비율. 먼지/얼룩은 미세한 고주파 노이즈를 추가함.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4  # 중심에서 이 반경 밖을 고주파로 간주

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    high_freq_mask = dist > radius

    total_energy = magnitude.sum() + 1e-8
    high_freq_energy_ratio = magnitude[high_freq_mask].sum() / total_energy
    return float(high_freq_energy_ratio)


def glare_unevenness(gray):
    """
    밝은 영역(반사광 후보, 상위 10% 밝기)의 매끈함을 측정.
    깨끗한 패널의 반사는 부드러운 그라데이션이라 라플라시안 분산이 작고,
    더러운 패널의 반사는 먼지에 의해 끊기고 거칠어 라플라시안 분산이 큼.
    밝은 영역이 거의 없으면(반사광 없는 사진) 0 반환.
    """
    threshold = np.percentile(gray, 90)
    bright_mask = gray >= threshold
    if bright_mask.sum() < 50:  # 밝은 영역이 너무 작으면 신뢰 불가
        return 0.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    bright_laplacian_values = laplacian[bright_mask]
    return float(np.var(bright_laplacian_values))


def extract_texture_features(image_paths, verbose=True, include_glare=False):
    """
    반환: include_glare=False(기본)이면 (N, 2) 배열 [local_contrast_std, high_freq_energy]
          include_glare=True면 (N, 3) 배열 [..., glare_unevenness]

    주의: glare_unevenness는 5장 소규모 실측 검증에서 가정과 반대 방향으로 나타남
    (깨끗한 패널이 강한 반사로 인해 오히려 라플라시안 분산이 커지는 경향 발견).
    추가 검증 전까지 기본 추출에서는 제외함.
    """
    feats = []
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        try:
            gray = _to_gray(p)
            f = [
                local_contrast_std(gray),
                high_freq_energy(gray),
            ]
            if include_glare:
                f.append(glare_unevenness(gray))
        except Exception as e:
            print(f"[경고] 텍스처 특징 추출 실패: {p} ({e})")
            f = [0.0, 0.0, 0.0] if include_glare else [0.0, 0.0]
        feats.append(f)
        if verbose and i % 200 == 0:
            print(f"  texture feature {i}/{n}")
    return np.array(feats)


def augment_embeddings_with_texture(embeddings, texture_feats, texture_weight=3.0):
    """
    ResNet18 임베딩(N, 512)에 텍스처 특징(N, 3)을 표준화 후 이어붙임.
    texture_weight: 텍스처 특징의 상대적 중요도를 키우기 위한 스케일.
    3차원 텍스처 특징을 512차원 임베딩에 그대로 붙이면 분류기가 거의 무시할
    수 있으므로(차원 수 차이로 인한 희석), 표준화 후 weight를 곱해 영향력을 보정.
    """
    mu = texture_feats.mean(axis=0)
    sigma = texture_feats.std(axis=0) + 1e-8
    texture_z = (texture_feats - mu) / sigma * texture_weight
    return np.concatenate([embeddings, texture_z], axis=1)


if __name__ == "__main__":
    import os
    upload_dir = "/mnt/user-data/uploads"
    paths = [os.path.join(upload_dir, f) for f in sorted(os.listdir(upload_dir)) if f.endswith(".jpg")]
    feats = extract_texture_features(paths, include_glare=True)
    for i, p in enumerate(paths):
        print(f"{os.path.basename(p)}: contrast_std={feats[i,0]:.2f}, "
              f"high_freq={feats[i,1]:.4f}, glare_uneven={feats[i,2]:.2f}")