"""
ood_score를 구성하는 hand-crafted 특징들(grid/darkblue/skin/veg/watermark)이
실제로 ood_score 산정에 얼마나 기여하는지, 그리고 각 특징 자체의 분포가
변별력 있는 모양인지 진단.

사용법:
    python3 diagnose_ood_features.py --ood_features submission_b_v4_ood_features.csv --out ood_features_dist.png
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_COLS = ["grid_regularity", "dark_blue_gray", "skin_ratio",
                 "vegetation_ratio", "watermark_ratio", "ocr_text_length"]


def diagnose(ood_features_path, out_path, top_n=20):
    df = pd.read_csv(ood_features_path)

    # 1) 각 특징의 분포 히스토그램
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, FEATURE_COLS):
        ax.hist(df[col], bins=40, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    # 다음 칸에 ood_score 자체도 표시
    axes[len(FEATURE_COLS)].hist(df["ood_score"], bins=40, color="salmon", edgecolor="white")
    axes[len(FEATURE_COLS)].set_title("ood_score (final)")
    # 남는 칸은 숨김
    for ax in axes[len(FEATURE_COLS) + 1:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"저장 완료: {out_path}\n")

    # 2) 각 특징과 ood_score 간의 상관관계 (스피어만 순위 상관, AP와 무관하게
    #    "이 특징이 ood_score 순위에 실제로 얼마나 기여했는지"를 보여줌)
    print("=== 각 특징과 최종 ood_score의 상관관계 (Spearman) ===")
    from scipy.stats import spearmanr
    for col in FEATURE_COLS:
        corr, _ = spearmanr(df[col], df["ood_score"])
        print(f"  {col}: {corr:.3f}")

    # 3) ood_score 상위 N개의 워터마크 비율이 전체 평균보다 실제로 높은지 확인
    print(f"\n=== ood_score 상위 {top_n}개 vs 전체 평균 비교 ===")
    top = df.nlargest(top_n, "ood_score")
    for col in FEATURE_COLS:
        top_mean = top[col].mean()
        overall_mean = df[col].mean()
        diff_ratio = (top_mean - overall_mean) / (overall_mean + 1e-8) * 100
        direction = "↑" if diff_ratio > 0 else "↓"
        print(f"  {col}: 상위{top_n} 평균={top_mean:.4f}, 전체 평균={overall_mean:.4f} "
              f"({direction} {abs(diff_ratio):.1f}%)")

    # 4) watermark_ratio 단독으로 상위 정렬했을 때 실제로 의미있는 이미지들이 잡히는지
    #    (직접 눈으로 검증할 수 있도록 id 목록 출력)
    print(f"\n=== watermark_ratio 단독 상위 {top_n}개 id (직접 이미지 확인용) ===")
    top_wm = df.nlargest(top_n, "watermark_ratio")
    print(top_wm[["id", "watermark_ratio", "ood_score"]].to_string(index=False))

    # 5) OCR 텍스트가 검출된 이미지(ocr_text_length > 0)만 따로 확인
    #    -> 대부분 0이라 일반 히스토그램으로는 분포가 안 보이므로 별도 점검
    if "ocr_text_length" in df.columns:
        ocr_positive = df[df["ocr_text_length"] > 0]
        print(f"\n=== OCR 텍스트 검출된 이미지: {len(ocr_positive)} / {len(df)} "
              f"({len(ocr_positive) / len(df) * 100:.1f}%) ===")
        if len(ocr_positive) > 0:
            print(f"  이 중 ood_score 평균: {ocr_positive['ood_score'].mean():.4f} "
                  f"(전체 평균: {df['ood_score'].mean():.4f})")
            print(ocr_positive.nlargest(min(top_n, len(ocr_positive)), "ocr_text_length")
                  [["id", "ocr_text_length", "ood_score"]].to_string(index=False))
        else:
            print("  OCR로 텍스트가 검출된 이미지가 없습니다. "
                  "tesseract 설치/경로를 확인하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood_features", required=True,
                         help="run_pipeline.py가 생성한 *_ood_features.csv 경로")
    parser.add_argument("--out", default="ood_features_dist.png")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()
    diagnose(args.ood_features, args.out, args.top_n)