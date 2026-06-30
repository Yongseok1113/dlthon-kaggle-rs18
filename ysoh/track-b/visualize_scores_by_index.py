"""
submission.csv의 세 점수를 id(파일명 내 인덱스) 순서로 scatter plot 시각화.
x축 = id에 포함된 숫자 인덱스 (예: train_00128 -> 128)
y축 = 점수값
상위 N% 지점을 가로선으로 표시하고, 그 이상인 점은 빨간색으로 강조.

사용법:
    python3 visualize_scores_by_index.py --submission submission_b.csv --out scores_by_index.png
"""
import argparse
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경(서버/WSL)에서도 안전하게 파일로 저장
import matplotlib.pyplot as plt


def extract_index(id_str):
    """'train_00128' -> 128, 'test_00045' -> 45 등 id 문자열에서 숫자만 추출."""
    m = re.search(r"(\d+)", id_str)
    return int(m.group(1)) if m else None


def visualize_by_index(submission_path, out_path, highlight_percentile=95):
    sub = pd.read_csv(submission_path)
    sub["__index"] = sub["id"].apply(extract_index)
    sub = sub.sort_values("__index")

    cols = ["mislabel_score", "dup_score", "ood_score"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, col in zip(axes, cols):
        threshold = sub[col].quantile(highlight_percentile / 100)
        is_high = sub[col] >= threshold

        ax.scatter(sub.loc[~is_high, "__index"], sub.loc[~is_high, col],
                   s=8, color="steelblue", alpha=0.5, label="normal")
        ax.scatter(sub.loc[is_high, "__index"], sub.loc[is_high, col],
                   s=14, color="red", alpha=0.8, label=f"top {100 - highlight_percentile:.0f}%")
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.6)

        ax.set_title(col)
        ax.set_ylabel("score")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("id index (e.g. train_00128 -> 128)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"저장 완료: {out_path}")

    # 상위 구간이 특정 id 범위에 몰려있는지 간단히 진단
    for col in cols:
        threshold = sub[col].quantile(highlight_percentile / 100)
        high_idx = sub.loc[sub[col] >= threshold, "__index"]
        if len(high_idx) > 0:
            print(f"{col}: 상위 {100 - highlight_percentile}% 구간 id index 범위 "
                  f"[{high_idx.min()} ~ {high_idx.max()}], 개수={len(high_idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", default="scores_by_index.png")
    parser.add_argument("--highlight_percentile", type=float, default=95,
                         help="강조 표시할 상위 퍼센타일 (기본 95 = 상위 5%)")
    args = parser.parse_args()
    visualize_by_index(args.submission, args.out, args.highlight_percentile)