"""
submission.csv의 세 점수(mislabel/dup/ood) 분포를 히스토그램으로 시각화.

사용법:
    python3 visualize_scores.py --submission submission_b.csv --out scores_dist.png
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경(서버/WSL)에서도 안전하게 파일로 저장
import matplotlib.pyplot as plt


def visualize(submission_path, out_path):
    sub = pd.read_csv(submission_path)
    cols = ["mislabel_score", "dup_score", "ood_score"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, cols):
        ax.hist(sub[col], bins=50, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        mean_v = sub[col].mean()
        p95_v = sub[col].quantile(0.95)
        ax.axvline(mean_v, color="orange", linestyle="--", label=f"mean={mean_v:.3f}")
        ax.axvline(p95_v, color="red", linestyle="--", label=f"p95={p95_v:.3f}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", default="scores_dist.png")
    args = parser.parse_args()
    visualize(args.submission, args.out)