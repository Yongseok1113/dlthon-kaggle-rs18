"""
Track B(rs-18-track-b)에서 산출한 mislabel/dup/ood 점수로 Track A의 학습 데이터를 정제.

Track A의 train 이미지/라벨이 Track B와 동일한 출처(같은 1,366장)라는 전제 하에,
Track B submission.csv(id, mislabel_score, dup_score, ood_score)를 그대로 불러와
다음 정제 작업을 수행:

1. ood_score가 높은 샘플(패널이 아닌 이미지) -> 학습에서 완전히 제외
2. dup_score가 높은 샘플(근접 중복) -> 그룹별로 1장만 남기고 나머지 제외
   (중복 이미지가 학습/검증에 동시에 들어가 데이터 누수가 생기는 것을 방지)
3. mislabel_score가 높은 샘플(라벨 오류 의심) -> 완전 제외하거나, 학습 가중치를
   낮추는 두 가지 옵션 제공 (기본은 가중치 감쇠, 라벨 자체가 일부 맞을 수도
   있어 하드 제외보다 안전)
"""
import argparse
import numpy as np
import pandas as pd


def clean_train_set(
    labels_df,
    track_b_scores_df,
    ood_threshold_percentile=95,
    dup_threshold_percentile=95,
    mislabel_threshold_percentile=90,
    mislabel_mode="weight",  # "weight" 또는 "exclude"
    mislabel_weight_floor=0.1,
):
    """
    labels_df: id, label 컬럼을 가진 원본 train_labels.csv
    track_b_scores_df: id, mislabel_score, dup_score, ood_score 컬럼을 가진
                        Track B의 submission.csv

    반환: 정제된 DataFrame (id, label, sample_weight, exclude_reason)
          exclude_reason이 None이 아니면 학습에서 제외 대상.
          sample_weight는 학습 시 손실 가중치로 사용(제외 안 된 샘플은 보통 1.0).
    """
    df = labels_df.merge(track_b_scores_df, on="id", how="left")
    missing = df["mislabel_score"].isnull().sum()
    if missing > 0:
        print(f"[경고] Track B 점수가 없는 샘플 {missing}개 발견 (id 불일치 가능성). "
              f"해당 샘플은 정제 없이 그대로 사용합니다.")
        df[["mislabel_score", "dup_score", "ood_score"]] = df[
            ["mislabel_score", "dup_score", "ood_score"]
        ].fillna(0.0)

    df["sample_weight"] = 1.0
    df["exclude_reason"] = None

    # 1) ood 제외
    ood_threshold = np.percentile(df["ood_score"], ood_threshold_percentile)
    ood_mask = df["ood_score"] > ood_threshold
    df.loc[ood_mask, "exclude_reason"] = "ood"
    print(f"[1/3] ood 제외: {ood_mask.sum()}개 (임계값 상위 {100 - ood_threshold_percentile}%, "
          f"threshold={ood_threshold:.4f})")

    # 2) dup 그룹 중복 제거 (이미 제외 대상이 아닌 샘플 중에서)
    dup_threshold = np.percentile(df["dup_score"], dup_threshold_percentile)
    dup_candidates = df[(df["dup_score"] > dup_threshold) & df["exclude_reason"].isnull()]
    # 근접 중복으로 의심되는 샘플들은, dup_score가 높은 순으로 정렬해 짝수번째(나중에 나오는 것)를
    # 제외하는 단순한 방식 사용(진짜 그룹핑은 Track B 쪽 nearest_idx 정보가 있어야 정확하지만,
    # 여기서는 submission.csv만으로 가능한 보수적인 근사를 적용).
    if len(dup_candidates) > 0:
        dup_excluded_idx = dup_candidates.sort_values("dup_score", ascending=False).index[1::2]
        df.loc[dup_excluded_idx, "exclude_reason"] = "dup"
        print(f"[2/3] dup 제외: {len(dup_excluded_idx)}개 (임계값 상위 {100 - dup_threshold_percentile}%, "
              f"threshold={dup_threshold:.4f})")
    else:
        print(f"[2/3] dup 제외: 0개 (임계값 상위 {100 - dup_threshold_percentile}% 후보 없음)")

    # 3) mislabel 처리 (제외 또는 가중치 감쇠)
    mislabel_threshold = np.percentile(df["mislabel_score"], mislabel_threshold_percentile)
    mislabel_mask = (df["mislabel_score"] > mislabel_threshold) & df["exclude_reason"].isnull()

    if mislabel_mode == "exclude":
        df.loc[mislabel_mask, "exclude_reason"] = "mislabel"
        print(f"[3/3] mislabel 제외: {mislabel_mask.sum()}개 (임계값 상위 "
              f"{100 - mislabel_threshold_percentile}%, threshold={mislabel_threshold:.4f})")
    else:
        # 가중치 감쇠: mislabel_score가 높을수록 학습 기여도를 줄임(완전히 0으로는 안 만듦)
        weight = 1.0 - (1.0 - mislabel_weight_floor) * df["mislabel_score"]
        df.loc[mislabel_mask, "sample_weight"] = weight[mislabel_mask]
        print(f"[3/3] mislabel 가중치 감쇠 적용: {mislabel_mask.sum()}개 "
              f"(임계값 상위 {100 - mislabel_threshold_percentile}%, threshold={mislabel_threshold:.4f}, "
              f"weight_floor={mislabel_weight_floor})")

    n_excluded = df["exclude_reason"].notnull().sum()
    print(f"\n총 제외: {n_excluded} / {len(df)} ({n_excluded / len(df) * 100:.1f}%)")
    print(f"최종 학습 사용 샘플: {len(df) - n_excluded}개")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True, help="Track A의 train_labels.csv")
    parser.add_argument("--track_b_scores", required=True,
                         help="Track B의 submission.csv (id, mislabel_score, dup_score, ood_score)")
    parser.add_argument("--out", default="cleaned_train.csv")
    parser.add_argument("--ood_pct", type=float, default=95)
    parser.add_argument("--dup_pct", type=float, default=95)
    parser.add_argument("--mislabel_pct", type=float, default=90)
    parser.add_argument("--mislabel_mode", choices=["weight", "exclude"], default="weight")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_csv)
    track_b_df = pd.read_csv(args.track_b_scores)

    cleaned = clean_train_set(
        labels_df, track_b_df,
        ood_threshold_percentile=args.ood_pct,
        dup_threshold_percentile=args.dup_pct,
        mislabel_threshold_percentile=args.mislabel_pct,
        mislabel_mode=args.mislabel_mode,
    )
    cleaned.to_csv(args.out, index=False)
    print(f"\n저장 완료: {args.out}")