"""
submission.csv의 각 축(mislabel/dup/ood) 점수 분포를 진단하는 스크립트.
실제 정답(AP)은 알 수 없지만, 분포 모양과 상위/하위 목록을 통해
어느 축이 약할지 가늠하는 데 도움을 줌.

사용법:
    python3 diagnose.py --submission submission_b.csv --train_dir <train 폴더> --labels_csv <train_labels.csv>
"""
import argparse
import pandas as pd
import numpy as np


def diagnose(submission_path, labels_csv=None):
    sub = pd.read_csv(submission_path)
    print(f"행 수: {len(sub)}\n")

    for col in ["mislabel_score", "dup_score", "ood_score"]:
        s = sub[col]
        print(f"=== {col} ===")
        print(f"  min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}, std={s.std():.4f}")
        print(f"  분위수: 25%={s.quantile(.25):.4f}, 50%={s.quantile(.5):.4f}, "
              f"75%={s.quantile(.75):.4f}, 95%={s.quantile(.95):.4f}, 99%={s.quantile(.99):.4f}")
        # 점수가 한쪽으로 쏠려 거의 변별력이 없는 경우 경고
        if s.std() < 1e-3:
            print("  [경고] 분산이 거의 0입니다 — 이 축은 변별력이 없을 가능성이 큽니다.")
        n_top_tied = (s == s.max()).sum()
        if n_top_tied > len(sub) * 0.1:
            print(f"  [경고] 최댓값을 가진 샘플이 {n_top_tied}개({n_top_tied/len(sub)*100:.1f}%)"
                  f"로 너무 많습니다 — 점수가 이산적(binary)이면 AP가 낮게 나올 수 있습니다.")
        print()

    if labels_csv:
        labels = pd.read_csv(labels_csv)
        merged = sub.merge(labels, on="id")
        # mislabel_score 상위와 라벨 분포 교차 확인 (참고용)
        top_mis = merged.nlargest(20, "mislabel_score")
        print("=== mislabel_score 상위 20개의 라벨 분포 ===")
        print(top_mis["label"].value_counts())
        print()

    for col in ["mislabel_score", "dup_score", "ood_score"]:
        print(f"=== {col} 상위 15개 ===")
        print(sub.nlargest(15, col)[["id", col]].to_string(index=False))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--labels_csv", default=None)
    args = parser.parse_args()
    diagnose(args.submission, args.labels_csv)