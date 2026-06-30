"""
ood_score 계산 (off-topic, 비-패널 이미지 탐지)

전략:
1) hand-crafted 특징으로 '패널다움'을 정의
   - 직선 격자 패턴 강도 (Hough 변환 또는 FFT 기반 주기성)
   - 색상 분포 (패널은 보통 짙은 청/회색 톤이 우세)
   - 텍스트/로고 영역 비율 (제품 박스샷에 텍스트가 많음 -> 간단히 고대비 작은 영역 비율로 근사)
   - 사람 피부색 비율 (작업자가 화면을 크게 차지하는 경우 탐지)
2) 임베딩 기반 이상치 탐지 (제공되면)
   - 전체 임베딩 분포에서 k-NN 평균거리 또는 Isolation Forest 점수
3) 두 신호를 rank 기반으로 결합
"""
import numpy as np
import cv2
from PIL import Image


def _to_cv2(path, max_side=512):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def grid_regularity_score(bgr):
    """
    패널의 직선 격자 패턴이 강할수록 높은 점수.
    Canny 엣지 + Hough 직선 검출로 근사.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=min(gray.shape) // 4, maxLineGap=10)
    if lines is None:
        return 0.0
    n_lines = len(lines)
    # 직선 개수를 이미지 면적으로 정규화 (너무 큰 이미지에서 직선이 단순히 많아지는 것 방지)
    area = gray.shape[0] * gray.shape[1]
    score = n_lines / (area / 1e5)
    return float(score)


def dark_blue_gray_ratio(bgr):
    """
    패널 특유의 짙은 청/회색 톤 비율.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # 짙은 색(낮은 명도) + 채도 낮거나 청색 계열
    mask = (v < 120) & ((s < 60) | ((h > 90) & (h < 140)))
    return float(mask.mean())


def skin_color_ratio(bgr):
    """
    사람 피부색 비율 (작업자가 화면을 크게 차지하는 경우 탐지용 보조 신호)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float((mask > 0).mean())


def watermark_text_ratio(bgr):
    """
    워터마크/텍스트 영역 비율 추정.
    텍스트는 작은 크기의 고대비 연결요소가 조밀하게 모여있는 패턴을 보임.
    MSER(Maximally Stable Extremal Regions)로 텍스트 후보 영역을 찾고,
    전체 면적 대비 비율을 반환.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create()
        mser.setMinArea(10)
        mser.setMaxArea(800)
        regions, _ = mser.detectRegions(gray)
    except Exception:
        return 0.0

    if len(regions) == 0:
        return 0.0

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        # region: (N, 2) 좌표들의 convex hull로 채우기
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.fillPoly(mask, [hull], 255)

    return float((mask > 0).mean())


def green_vegetation_ratio(bgr):
    """
    풀/나무 등 식물 비율. 너무 높으면 패널이 작게 나온 풍경샷일 가능성.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float((mask > 0).mean())


def ocr_text_length(path, max_side=512):
    """
    이미지에서 OCR로 텍스트를 추출해 길이를 반환.
    워터마크/제품샷/인포그래픽 등 텍스트가 포함된 off-topic 이미지를
    직접적으로 잡아내는 신호. 기존 MSER 기반 워터마크 근사(watermark_text_ratio)보다
    훨씬 정확함(실측: MSER은 노이즈에 취약해 신호가 약했으나, 직접 OCR은
    "shutterstock" 등 실제 텍스트를 명확히 검출함).
    """
    try:
        import pytesseract
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_side, max_side))
        text = pytesseract.image_to_string(img)
        return len(text.strip())
    except Exception as e:
        print(f"[경고] OCR 실패: {path} ({e})")
        return 0


def extract_ood_features(image_paths, verbose=True):
    feats = []
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        try:
            bgr = _to_cv2(p)
            f = [
                grid_regularity_score(bgr),
                dark_blue_gray_ratio(bgr),
                skin_color_ratio(bgr),
                green_vegetation_ratio(bgr),
                watermark_text_ratio(bgr),
                ocr_text_length(p),
            ]
        except Exception as e:
            print(f"[경고] ood 특징 추출 실패: {p} ({e})")
            f = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        feats.append(f)
        if verbose and i % 200 == 0:
            print(f"  ood feature {i}/{n}")
    return np.array(feats)


def compute_ood_score_simple_rank_avg(image_paths, embeddings):
    """
    레퍼런스 노트북(0.422 달성) 방식을 그대로 재현.
    centroid_dist(임베딩 평균에서 거리) + knn_dist(k=5 최근접 평균거리) +
    ocr_text_length 세 가지를 각각 rank로 변환 후 단순 평균(/3).

    기존 compute_ood_score_handcrafted()와의 핵심 차이:
    - hand-crafted 보조 특징(grid/dark/skin/veg/watermark), CLIP을 전혀 쓰지 않음
    - np.maximum 보호 로직 없이 순수 평균만 사용
    - 신호 수를 3개로 최소화 -> 단순함이 오히려 안정적인 결과를 낼 수 있음
      (실측: np.maximum 보호 2단계 + CLIP 결합 버전(v13)이 단순 버전보다
      오히려 public score가 낮게 나옴 0.385 vs 0.390)

    embeddings는 필수 인자(레퍼런스도 ResNet18 임베딩을 항상 사용).
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import rankdata

    centroid = embeddings.mean(axis=0)
    centroid_dist = np.linalg.norm(embeddings - centroid, axis=1)

    k = 5
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    dists, _ = nn.kneighbors(embeddings)
    knn_dist = dists[:, 1:].mean(axis=1)

    ocr_lengths = np.array([ocr_text_length(p) for p in image_paths])

    centroid_rank = rankdata(centroid_dist)
    knn_rank = rankdata(knn_dist)
    text_rank = rankdata(ocr_lengths)

    ood_score_raw = (centroid_rank + knn_rank + text_rank) / 3
    # 0~1로 정규화 (레퍼런스는 정규화 없이 순위값 그대로 제출했으나,
    # 본 대회 제출 형식이 0~1 점수를 요구하므로 min-max 정규화 적용)
    ood_score = (ood_score_raw - ood_score_raw.min()) / (ood_score_raw.max() - ood_score_raw.min() + 1e-8)

    feats = np.column_stack([centroid_dist, knn_dist, ocr_lengths])
    return ood_score, feats


def compute_ood_score_handcrafted(image_paths, embeddings=None):
    """
    hand-crafted 특징을 percentile rank로 변환 후, '전형적 패널'에서 벗어난 정도를 계산.
    전형적 패널 = grid_regularity 높고 dark_blue_gray 높고 skin/vegetation/watermark 낮은 쪽.

    주의: z-score 표준화 대신 percentile rank(0~1)를 사용. z-score는 분산이 큰 특징이
    같은 가중치를 줘도 실제 영향력이 더 커지는 문제가 있었음(예: watermark_ratio가
    분산이 커서 가중치 1.5를 줬음에도 dark_blue_gray(가중치 1.0)보다 실제 영향력이 작았음).
    percentile rank는 모든 특징이 동일한 0~1 스케일을 갖게 되어 가중치가 의도한 대로 반영됨.
    """
    feats = extract_ood_features(image_paths)

    def to_rank(x):
        order = np.argsort(x)
        rank = np.empty_like(order, dtype=float)
        rank[order] = np.arange(len(order))
        return rank / (len(order) - 1 + 1e-8)

    r_grid = to_rank(feats[:, 0])
    r_dark = to_rank(feats[:, 1])
    r_skin = to_rank(feats[:, 2])
    r_veg = to_rank(feats[:, 3])
    r_watermark = to_rank(feats[:, 4])
    r_ocr_text = to_rank(feats[:, 5])

    # 패널다움 점수(0~1 스케일 rank들의 가중합): grid(+, 약하게), dark_blue_gray(+),
    # skin(-), vegetation(-).
    # watermark/ocr_text는 선형결합에 포함시키지 않음 -> 다른 신호들에 묻혀
    # 거의 무영향이 되는 문제가 실측 진단(Spearman corr가 가장 낮은 축)으로 확인됨.
    panel_likeness = 0.5 * r_grid + r_dark - r_skin - r_veg
    order = np.argsort(panel_likeness)
    rank = np.empty_like(order, dtype=float)
    rank[order] = np.arange(len(order))
    base_ood_score = 1 - (rank / (len(order) - 1 + 1e-8))

    # watermark(MSER 근사)와 ocr_text(실제 OCR, 더 신뢰도 높음)는 모두 별도 신호로 두고,
    # base/watermark/ocr_text 중 가장 강하게 의심되는 쪽(max)을 최종 점수로 사용.
    # 텍스트가 명확히 검출되면(워터마크/제품샷/인포그래픽) 다른 신호와 무관하게
    # 강하게 ood로 의심되도록 보장.
    hc_ood_score = np.maximum(np.maximum(base_ood_score, r_watermark), r_ocr_text)

    if embeddings is not None:
        # 임베딩 기반 이상치 점수도 결합 (k-NN 평균거리, 작은 k로 지역밀도 근사)
        from sklearn.neighbors import NearestNeighbors
        k = min(10, len(embeddings) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
        dists, _ = nn.kneighbors(embeddings)
        knn_dist = dists[:, 1:].mean(axis=1)  # 자기 자신(거리0) 제외
        order2 = np.argsort(knn_dist)
        rank2 = np.empty_like(order2, dtype=float)
        rank2[order2] = np.arange(len(order2))
        emb_ood_score = rank2 / (len(order2) - 1 + 1e-8)  # 거리가 클수록(=고립될수록) 높은 점수

        # 주의: 0.5*hc + 0.5*emb로 단순 평균하면 np.maximum으로 보호했던 OCR/watermark
        # 신호가 다시 희석되는 문제가 실측에서 확인됨(OCR 64자 검출됐는데 ood_score 0.40
        # 같은 낮은 값이 나오는 역설 발생). watermark/ocr_text는 임베딩 결합 이후에도
        # 보호되도록, 평균 낸 결과와 watermark/ocr_text 중 다시 max를 취함.
        combined = 0.5 * hc_ood_score + 0.5 * emb_ood_score
        ood_score = np.maximum(np.maximum(combined, r_watermark), r_ocr_text)
    else:
        ood_score = hc_ood_score

    return ood_score, feats


def compute_ood_score_clip_prompts(image_paths, device=None, batch_size=32, verbose=True):
    """
    (선택적, CLIP 설치 시에만 사용 권장: pip install --break-system-packages -q git+https://github.com/openai/CLIP.git ftfy regex)

    CLIP의 zero-shot 능력을 이용해 "이 이미지가 태양광 패널 사진인지"를 직접 텍스트로 질의.
    hand-crafted 격자 패턴보다 의미적으로 훨씬 강건함 (제품박스/사람/풍경/콜라주 등을
    실제로 구분 가능). ResNet18 + hand-crafted 특징이 실패하는 경우를 보완하기 위한
    핵심 신호로 사용을 강력히 권장.

    반환: ood_score (패널이 아닐 확률에 가까울수록 1에 근접)
    """
    import torch
    import clip
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)

    positive_prompts = [
        "a photo of a solar panel",
        "a close-up of a solar panel surface",
        "an array of solar panels outdoors",
    ]
    negative_prompts = [
        "a photo of a product box or package",
        "a collage of multiple unrelated photos",
        "a photo of a person",
        "a photo of nature or landscape without solar panels",
        "a random unrelated object",
    ]
    all_prompts = positive_prompts + negative_prompts
    n_pos = len(positive_prompts)

    with torch.no_grad():
        text_tokens = clip.tokenize(all_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores = []
    n = len(image_paths)
    for start in range(0, n, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"[경고] CLIP 이미지 로드 실패: {p} ({e})")
                imgs.append(None)

        valid_mask = [im is not None for im in imgs]
        if any(valid_mask):
            batch_tensor = torch.stack([im for im in imgs if im is not None]).to(device)
            with torch.no_grad():
                img_feats = model.encode_image(batch_tensor)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                sims = img_feats @ text_features.T  # (B, len(all_prompts))
                probs = sims.softmax(dim=1)
                pos_probs = probs[:, :n_pos].sum(dim=1).cpu().numpy()
        else:
            pos_probs = np.array([])

        it = iter(pos_probs)
        for valid in valid_mask:
            if valid:
                scores.append(1.0 - float(next(it)))
            else:
                scores.append(0.5)  # 로드 실패 시 중립값

        if verbose:
            print(f"  clip ood {min(start + batch_size, n)}/{n}")

    return np.array(scores)


if __name__ == "__main__":
    import os
    upload_dir = "/mnt/user-data/uploads"
    paths = [os.path.join(upload_dir, f) for f in sorted(os.listdir(upload_dir)) if f.endswith(".jpg")]
    ood_score, feats = compute_ood_score_handcrafted(paths)
    for i, p in enumerate(paths):
        print(f"{os.path.basename(p)}: ood_score={ood_score[i]:.3f}, "
              f"grid={feats[i,0]:.2f}, darkblue={feats[i,1]:.2f}, "
              f"skin={feats[i,2]:.2f}, veg={feats[i,3]:.2f}, watermark={feats[i,4]:.3f}")