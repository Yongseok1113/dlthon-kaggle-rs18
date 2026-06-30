"""
dup_score 계산
- pHash 해밍거리: 강건하고 빠름, 거의 동일한 이미지에 민감
- 임베딩 코사인 유사도: 약간의 변형(자르기/색보정)에도 견고
두 신호를 결합해 최종 dup_score 산출.
"""
import numpy as np
import imagehash
from PIL import Image


def compute_phashes(image_paths):
    hashes = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            hashes.append(imagehash.phash(img, hash_size=16))  # 16x16 = 256bit, 해상도 높여 민감도 향상
        except Exception as e:
            print(f"[경고] phash 실패: {p} ({e})")
            hashes.append(None)
    return hashes


def phash_distance_matrix(hashes):
    n = len(hashes)
    dist = np.full((n, n), fill_value=np.inf)
    for i in range(n):
        if hashes[i] is None:
            continue
        for j in range(i + 1, n):
            if hashes[j] is None:
                continue
            d = hashes[i] - hashes[j]  # 해밍 거리
            dist[i, j] = d
            dist[j, i] = d
    return dist


def cosine_sim_matrix(embeddings):
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return norm @ norm.T


def compute_dup_scores(image_paths, embeddings=None, phash_weight=0.6, emb_weight=0.4):
    """
    각 이미지에 대해 '가장 유사한 다른 이미지'와의 유사도를 dup_score로 사용.
    embeddings가 None이면 pHash만 사용.
    """
    n = len(image_paths)
    hashes = compute_phashes(image_paths)
    dist = phash_distance_matrix(hashes)
    max_bits = 16 * 16  # hash_size=16 기준 총 비트수
    phash_sim = 1 - (dist / max_bits)  # 거리를 0~1 유사도로 변환
    np.fill_diagonal(phash_sim, -1)  # 자기 자신 제외

    if embeddings is not None:
        emb_sim = cosine_sim_matrix(embeddings)
        np.fill_diagonal(emb_sim, -1)
        combined = phash_weight * phash_sim + emb_weight * emb_sim
    else:
        combined = phash_sim

    # 각 이미지의 최근접 유사도 = dup_score 원시값
    nearest_sim = combined.max(axis=1)
    nearest_idx = combined.argmax(axis=1)

    # 0~1 정규화 (음수 가능성 있으니 clip)
    dup_score = np.clip(nearest_sim, 0, 1)
    return dup_score, nearest_idx, combined


if __name__ == "__main__":
    import os
    upload_dir = "/mnt/user-data/uploads"
    paths = [os.path.join(upload_dir, f) for f in sorted(os.listdir(upload_dir)) if f.endswith(".jpg")]
    dup_score, nearest_idx, combined = compute_dup_scores(paths)
    for i, p in enumerate(paths):
        print(f"{os.path.basename(p)}: dup_score={dup_score[i]:.3f}, nearest={os.path.basename(paths[nearest_idx[i]])}")