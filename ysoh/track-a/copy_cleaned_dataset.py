"""
cleaned_train.csv 기준으로 학습에 실제 사용되는 이미지만
별도 폴더로 복사. 직접 이미지셋을 확인할 때 사용.

폴더 구조 결과:
    datasets/
        rs-18-track-a/          (원본, 건드리지 않음)
            train/
            test/
            train_labels.csv
        rs-18-track-a-cleaned/  (정제된 학습 이미지만)
            train/
                clean/          (label=0)
                dusty/          (label=1)
            excluded/           (제외된 이미지, 확인용)
                ood/
                dup/

사용법:
    python3 copy_cleaned_dataset.py \
        --src_dir datasets/rs-18-track-a/train \
        --cleaned_csv track-a/cleaned_train.csv \
        --out_dir datasets/rs-18-track-a-cleaned
"""
import argparse
import os
import shutil
import pandas as pd


def main(args):
    df = pd.read_csv(args.cleaned_csv)
    print(f"전체 샘플: {len(df)}개")

    used = df[df["exclude_reason"].isnull()]
    excluded = df[df["exclude_reason"].notnull()]
    print(f"학습 사용: {len(used)}개, 제외: {len(excluded)}개")

    # 스크립트를 track-a/ 폴더 안에서 실행할 때
    # os.getcwd() = .../ysoh/track-a  ->  ../ = .../ysoh (프로젝트 루트 기준)
    base = os.path.join(os.getcwd(), "..")

    # 출력 폴더 구성
    dirs = {
        "clean":    os.path.join(base, args.out_dir, "train", "clean"),
        "dusty":    os.path.join(base, args.out_dir, "train", "dusty"),
        "ood":      os.path.join(base, args.out_dir, "excluded", "ood"),
        "dup":      os.path.join(base, args.out_dir, "excluded", "dup"),
        "mislabel": os.path.join(base, args.out_dir, "excluded", "mislabel"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    src_base = os.path.join(base, args.src_dir)

    # 학습에 사용되는 이미지를 라벨별 폴더로 복사
    copied = 0
    for _, row in used.iterrows():
        src = os.path.join(src_base, f"{row['id']}.jpg")
        label_dir = "clean" if row["label"] == 0 else "dusty"
        dst = os.path.join(dirs[label_dir], f"{row['id']}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"[경고] 원본 파일 없음: {src}")

    print(f"학습 이미지 복사 완료: {copied}개 "
          f"(clean: {len(used[used['label']==0])}개, dusty: {len(used[used['label']==1])}개)")

    # 제외된 이미지를 사유별 폴더로 복사 (확인용)
    excluded_copied = {"ood": 0, "dup": 0, "mislabel": 0}
    for _, row in excluded.iterrows():
        src = os.path.join(src_base, f"{row['id']}.jpg")
        reason = row["exclude_reason"]
        if reason not in dirs:
            reason = "mislabel"
        dst = os.path.join(dirs[reason], f"{row['id']}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            excluded_copied[reason] = excluded_copied.get(reason, 0) + 1

    print(f"제외 이미지 복사 완료: ood={excluded_copied['ood']}개, "
          f"dup={excluded_copied['dup']}개, mislabel={excluded_copied['mislabel']}개")
    out_abs = os.path.normpath(os.path.join(base, args.out_dir))
    print(f"\n출력 폴더: {out_abs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True,
                         help="원본 train 이미지 폴더 (datasets/rs-18-track-a/train)")
    parser.add_argument("--cleaned_csv", required=True,
                         help="clean_trainset.py가 생성한 cleaned_train.csv")
    parser.add_argument("--out_dir", default="datasets/rs-18-track-a-cleaned",
                         help="정제된 이미지를 복사할 출력 폴더")
    args = parser.parse_args()
    main(args)