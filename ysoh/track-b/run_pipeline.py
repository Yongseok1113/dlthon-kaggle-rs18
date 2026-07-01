"""
Track B 전체 파이프라인
실행 순서: 임베딩 추출 -> dup_score -> ood_score -> mislabel_score -> 제출 파일 생성

사용법 (본인 로컬/Kaggle 환경에서):
    python3 run_pipeline.py --train_dir /path/to/train --labels_csv /path/to/train_labels.csv --out submission.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

from embed import embed_images
from dup_score import compute_dup_scores
from ood_score import (compute_ood_score_handcrafted, compute_ood_score_clip_prompts,
                        compute_ood_score_simple_rank_avg,
                        compute_patch_variance_score, compute_person_ratio_score)
from mislabel_score import compute_mislabel_scores, combine_with_dup_signal


def resolve_data_dir(competition_slug):
    """
    kagglehub로 competition 데이터를 받아 로컬 캐시 경로를 반환.
    이미 다운로드되어 있으면 재다운로드 없이 캐시 경로만 즉시 반환됨.
    """
    import kagglehub
    path = kagglehub.competition_download(competition_slug)
    print(f"[kagglehub] '{competition_slug}' 데이터 경로: {path}")
    return path


def find_dup_label_conflicts(dup_score, nearest_idx, labels, dup_threshold=0.85):
    """
    dup_score가 임계값을 넘는 강한 중복 쌍 중에서, 라벨이 서로 다른 경우를 찾는다.
    """
    n = len(labels)
    conflict = np.zeros(n, dtype=bool)
    for i in range(n):
        if dup_score[i] >= dup_threshold:
            j = nearest_idx[i]
            if labels[i] != labels[j]:
                conflict[i] = True
                conflict[j] = True
    return conflict


def main(train_dir, labels_csv, out_path, use_clip=True, ood_method="complex",
         track_a_model=None, track_a_arch="efficientnet_b0", use_person_detection=False):
    df = pd.read_csv(labels_csv)
    df = df.sort_values("id").reset_index(drop=True)
    image_paths = [os.path.join(train_dir, f"{i}.jpg") for i in df["id"]]
    labels = df["label"].values

    print(f"[1/4] 임베딩 추출 중... ({len(image_paths)}장)")
    embeddings, valid_paths = embed_images(image_paths)
    assert len(embeddings) == len(image_paths), "일부 이미지 로드 실패 - 경로 확인 필요"

    print("[2/4] dup_score 계산 중...")
    dup_scores, nearest_idx, _ = compute_dup_scores(image_paths, embeddings=embeddings)

    print("[3/4] ood_score 계산 중...")

    if ood_method == "simple":
        print("  단순 rank 평균 방식 (centroid_dist + knn_dist + ocr_text_length) 사용")
        ood_scores, ood_feats_simple = compute_ood_score_simple_rank_avg(image_paths, embeddings)
        ood_feats = np.zeros((len(image_paths), 6))
        ood_feats[:, 5] = ood_feats_simple[:, 2]
        ood_scores_hc = ood_scores
        ood_scores_clip_raw = np.full(len(image_paths), np.nan)
    else:
        ood_scores_hc, ood_feats = compute_ood_score_handcrafted(image_paths, embeddings=embeddings)

        if use_clip:
            print("  CLIP zero-shot ood_score 계산 중... (최초 실행 시 가중치 다운로드)")
            try:
                ood_scores_clip_raw = compute_ood_score_clip_prompts(image_paths)

                def to_rank(x):
                    order = np.argsort(x)
                    rank = np.empty_like(order, dtype=float)
                    rank[order] = np.arange(len(order))
                    return rank / (len(order) - 1 + 1e-8)

                ood_scores_clip_rank = to_rank(ood_scores_clip_raw)
                ood_scores_hc_rank = to_rank(ood_scores_hc)
                combined = 0.7 * ood_scores_clip_rank + 0.3 * ood_scores_hc_rank
                ocr_rank = to_rank(ood_feats[:, 5])
                ood_scores = np.maximum(combined, ocr_rank)
            except Exception as e:
                print(f"  [경고] CLIP 계산 실패, hand-crafted만 사용: {e}")
                ood_scores = ood_scores_hc
                ood_scores_clip_raw = np.full_like(ood_scores_hc, np.nan)
        else:
            ood_scores = ood_scores_hc
            ood_scores_clip_raw = np.full_like(ood_scores_hc, np.nan)

    # --- 추가 신호 공통: rank 변환 함수 ---
    def to_rank_local(x):
        order = np.argsort(x)
        rank = np.empty_like(order, dtype=float)
        rank[order] = np.arange(len(order))
        return rank / (len(order) - 1 + 1e-8)

    # --- patch_variance는 비활성화 ---
    # 실측 검증 결과 model.pt 기반 추론에서도 3×3 소형 패치 단위 추론이 불안정해
    # 정상 패널(역광/나무 배경 등)을 혼합 이미지로 오탐하는 문제가 지속 확인됨.
    # (밝기 폴백: 역광/노을 오탐, model.pt 사용: 패치 크기 불일치로 인한 예측 불안정)
    # ood_score.py의 compute_patch_variance_score() 함수는 향후 재실험을 위해 보존.

    # --- 추가 신호: 사람 검출 (Mask R-CNN) ---
    person_ratio = np.zeros(len(image_paths))  # 기본값: 검출 안 함
    if use_person_detection:
        print("  Mask R-CNN 사람 검출 중... (최초 실행 시 가중치 다운로드)")
        try:
            person_ratio = compute_person_ratio_score(image_paths)
            person_rank = to_rank_local(person_ratio)
            # submission용 ood_score에는 결합 반영 (시각적 확인/진단용)
            ood_scores = np.maximum(ood_scores, person_rank)
            print(f"  사람 점유 비율 > 0.1인 이미지: "
                  f"{(person_ratio > 0.1).sum()}개")
        except Exception as e:
            print(f"  [경고] 사람 검출 실패: {e}")
    else:
        print("  사람 검출 생략 (--use_person_detection 옵션으로 활성화)")

    print("[4/4] mislabel_score 계산 중...")
    dup_threshold = np.percentile(dup_scores, 95)
    # person_ratio를 np.maximum으로 결합 후 percentile 계산하면 person 상위 샘플이
    # 상위 5% 슬롯을 채워 CLIP 신호(화성 탐사선 등)가 threshold 밖으로 밀려나는 문제 확인.
    # 해결: CLIP 기반 ood_score는 percentile 95로, person_ratio는 절대 임계값(0.1)으로
    # 각각 독립적으로 exclude_mask에 OR 결합 → 두 신호가 서로를 희석시키지 않음.
    ood_threshold = np.percentile(ood_scores, 95)
    person_threshold = 0.1  # 사람이 화면의 10% 이상 차지하면 하드 제외
    exclude_mask = (
        (dup_scores > dup_threshold) |
        (ood_scores > ood_threshold) |
        (person_ratio > person_threshold)
    )
    print(f"  dup 임계값(상위5%): {dup_threshold:.4f}, ood 임계값(상위5%): {ood_threshold:.4f}")
    print(f"  person_ratio 임계값: {person_threshold} (> 0이면 55개 포함)")
    print(f"  1차 하드 제외 샘플 수: {exclude_mask.sum()} / {len(exclude_mask)}")

    # ood_score가 높을수록 학습 기여도를 줄이는 soft weighting
    # (워터마크/비전형적 구도 등 약하게 의심되는 경우까지 완만하게 반영)
    sample_weight = 1.0 - ood_scores
    sample_weight = np.clip(sample_weight, 0.05, 1.0)  # 완전히 0이 되어 학습에서 사라지는 것은 방지

    mislabel_scores, oof_pred = compute_mislabel_scores(
        embeddings, labels, exclude_mask=exclude_mask,
        sample_weight=sample_weight, n_iterations=2, return_raw=True
    )

    # dup 그룹 내 라벨 충돌 신호로 mislabel_score 보강
    conflict_mask = find_dup_label_conflicts(dup_scores, nearest_idx, labels, dup_threshold=dup_threshold)
    mislabel_scores = combine_with_dup_signal(mislabel_scores, conflict_mask)
    print(f"  dup-label 충돌로 보강된 샘플 수: {conflict_mask.sum()}")

    submission = pd.DataFrame({
        "id": df["id"],
        "mislabel_score": mislabel_scores,
        "dup_score": dup_scores,
        "ood_score": ood_scores,
    })
    submission.to_csv(out_path, index=False)
    print(f"제출 파일 저장 완료: {out_path}")

    # 진단용: OOF raw 예측확률(P(dusty))과 원본 라벨을 함께 저장 -> 분류기 과확신 여부 점검용
    diag_path = out_path.replace(".csv", "_diag.csv")
    diag_df = pd.DataFrame({
        "id": df["id"],
        "label": labels,
        "oof_pred_prob": oof_pred,
        "mislabel_score": mislabel_scores,
        "dup_score": dup_scores,
        "ood_score": ood_scores,
    })
    diag_df.to_csv(diag_path, index=False)
    print(f"진단 파일 저장 완료: {diag_path}")

    # ood hand-crafted 특징별 원본값 저장 -> 어느 특징이 실제로 변별력 있는지 진단용
    ood_feat_path = out_path.replace(".csv", "_ood_features.csv")
    ood_feat_df = pd.DataFrame({
        "id": df["id"],
        "label": labels,
        "ood_score": ood_scores,
        "ood_score_handcrafted": ood_scores_hc,
        "ood_score_clip_raw": ood_scores_clip_raw,
        "grid_regularity": ood_feats[:, 0],
        "dark_blue_gray": ood_feats[:, 1],
        "skin_ratio": ood_feats[:, 2],
        "vegetation_ratio": ood_feats[:, 3],
        "watermark_ratio": ood_feats[:, 4],
        "ocr_text_length": ood_feats[:, 5],
    })
    ood_feat_df.to_csv(ood_feat_path, index=False)
    print(f"ood 특징 진단 파일 저장 완료: {ood_feat_path}")

    # 각 축 상위 10개 출력 (육안 검증용)
    for col in ["mislabel_score", "dup_score", "ood_score"]:
        top = submission.nlargest(10, col)
        print(f"\n=== {col} 상위 10개 ===")
        print(top[["id", col]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=None,
                         help="kagglehub competition slug (예: rs-18-track-b). "
                              "지정하면 --train_dir/--labels_csv를 자동으로 찾음")
    parser.add_argument("--train_dir", default=None,
                         help="train 이미지 폴더 경로 (직접 지정 시 --competition 무시)")
    parser.add_argument("--labels_csv", default=None,
                         help="train_labels.csv 경로 (직접 지정 시 --competition 무시)")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--no_clip", action="store_true",
                         help="CLIP을 사용하지 않고 hand-crafted ood_score만 사용 (설치/다운로드 불가 시)")
    parser.add_argument("--ood_method", default="complex", choices=["simple", "complex"],
                         help="complex(기본, 최종 채택: CLIP+hand-crafted+max보호, public score 0.390) "
                              "또는 simple(레퍼런스 노트북 재현: centroid+knn+ocr rank 평균, "
                              "실측 결과 public score 0.358로 더 낮아 기본값에서 제외함)")
    parser.add_argument("--track_a_model", default=None,
                         help="(현재 비활성화) patch_variance가 제거됨에 따라 미사용. "
                              "향후 재실험 시 ood_score.py의 compute_patch_variance_score() 참조.")
    parser.add_argument("--track_a_arch", default="efficientnet_b0",
                         choices=["resnet18", "efficientnet_b0"],
                         help="(현재 비활성화) --track_a_model과 함께 사용 예정이었던 옵션.")
    parser.add_argument("--use_person_detection", action="store_true", default=False,
                         help="Mask R-CNN으로 사람 검출 후 ood_score 보강. "
                              "최초 실행 시 가중치 다운로드 필요(약 170MB).")
    args = parser.parse_args()

    if args.train_dir and args.labels_csv:
        train_dir, labels_csv = args.train_dir, args.labels_csv
    elif args.competition:
        data_dir = resolve_data_dir(args.competition)
        train_dir = os.path.join(data_dir, "train")
        labels_csv = os.path.join(data_dir, "train_labels.csv")
    else:
        parser.error("--competition 또는 (--train_dir와 --labels_csv)를 함께 지정해야 합니다.")

    main(train_dir, labels_csv, args.out,
         use_clip=not args.no_clip, ood_method=args.ood_method,
         track_a_model=args.track_a_model, track_a_arch=args.track_a_arch,
         use_person_detection=args.use_person_detection)