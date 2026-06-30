"""
mislabel 분류기의 OOF 예측확률(P(dusty)) 분포를 라벨별로 나눠 시각화.
분류기가 과확신(거의 항상 0 또는 1 근처)하는지, 라벨별로 합리적으로
갈리는지 확인하는 용도.

사용법:
    python3 diagnose_oof.py --diag submission_b_v3_diag.csv --out oof_dist.png
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def diagnose_oof(diag_path, out_path):
    df = pd.read_csv(diag_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1) 라벨별 OOF 예측확률 분포 (겹쳐서 비교)
    ax = axes[0]
    for label_val, color, name in [(0, "steelblue", "label=0 (Clean)"),
                                     (1, "orange", "label=1 (Dusty)")]:
        subset = df[df["label"] == label_val]["oof_pred_prob"]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=f"{name} (n={len(subset)})")
    ax.set_xlabel("OOF predicted P(dusty)")
    ax.set_ylabel("count")
    ax.set_title("OOF predicted probability by label")
    ax.legend(fontsize=8)

    # 2) 전체 OOF 예측확률 분포 (과확신 여부 한눈에)
    ax = axes[1]
    ax.hist(df["oof_pred_prob"], bins=50, color="gray", edgecolor="white")
    mid_frac = ((df["oof_pred_prob"] > 0.3) & (df["oof_pred_prob"] < 0.7)).mean()
    ax.set_title(f"Overall OOF probability distribution\n(fraction in 0.3~0.7: {mid_frac*100:.1f}%)")
    ax.set_xlabel("OOF predicted P(dusty)")
    ax.set_ylabel("count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"저장 완료: {out_path}")

    # 정량 진단: 분류기가 자신있게 틀린 비율(과확신 + 라벨 불일치)
    confident = (df["oof_pred_prob"] > 0.9) | (df["oof_pred_prob"] < 0.1)
    print(f"\n과확신(>0.9 또는 <0.1) 비율: {confident.mean()*100:.1f}%")
    wrong_confident = confident & (
        ((df["oof_pred_prob"] > 0.9) & (df["label"] == 0)) |
        ((df["oof_pred_prob"] < 0.1) & (df["label"] == 1))
    )
    print(f"과확신했는데 라벨과 반대인 비율(=강한 mislabel 후보): "
          f"{wrong_confident.sum()}건 ({wrong_confident.mean()*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", required=True, help="run_pipeline.py가 생성한 *_diag.csv 경로")
    parser.add_argument("--out", default="oof_dist.png")
    args = parser.parse_args()
    diagnose_oof(args.diag, args.out)