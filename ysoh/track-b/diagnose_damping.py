"""
mislabel_score 감쇠(ood 기반) 전후를 비교하는 진단 스크립트.

사용법:
    python3 diagnose_damping.py --diag submission_b_v10_diag.csv --top_n 15
"""
import argparse
import pandas as pd


def diagnose(diag_path, top_n):
    df = pd.read_csv(diag_path)

    if "mislabel_score_before_damping" not in df.columns:
        print("이 진단 파일에는 감쇠 전/후 비교 컬럼이 없습니다. "
              "최신 run_pipeline.py로 다시 생성해주세요.")
        return

    print(f"=== 감쇠 전후 평균 비교 ===")
    print(f"감쇠 전 평균: {df['mislabel_score_before_damping'].mean():.4f}")
    print(f"감쇠 후 평균: {df['mislabel_score'].mean():.4f}")
    print()

    # 감쇠로 순위가 가장 많이 떨어진 샘플 (ood성 때문에 걸러진 것들)
    df["rank_before"] = df["mislabel_score_before_damping"].rank(ascending=False)
    df["rank_after"] = df["mislabel_score"].rank(ascending=False)
    df["rank_drop"] = df["rank_after"] - df["rank_before"]

    print(f"=== 감쇠로 순위가 가장 많이 떨어진 상위 {top_n}개 (ood성 때문에 걸러짐) ===")
    dropped = df.nlargest(top_n, "rank_drop")
    print(dropped[["id", "label", "mislabel_score_before_damping",
                    "mislabel_score", "ood_score", "rank_before", "rank_after"]]
          .to_string(index=False))
    print()

    print(f"=== 감쇠로 순위가 가장 많이 오른 상위 {top_n}개 (ood 낮은 진짜 mislabel 후보가 부각됨) ===")
    risen = df.nsmallest(top_n, "rank_drop")
    print(risen[["id", "label", "mislabel_score_before_damping",
                  "mislabel_score", "ood_score", "rank_before", "rank_after"]]
          .to_string(index=False))
    print()

    print(f"=== 감쇠 후 mislabel_score 상위 {top_n}개 (최종 제출에 쓰일 목록) ===")
    final_top = df.nlargest(top_n, "mislabel_score")
    print(final_top[["id", "label", "mislabel_score", "ood_score"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", required=True)
    parser.add_argument("--top_n", type=int, default=15)
    args = parser.parse_args()
    diagnose(args.diag, args.top_n)