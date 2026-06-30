"""
train 이미지 전체(또는 일부)를 리사이즈해서 압축 zip으로 저장.
Claude 대화에 업로드 가능한 크기로 만드는 용도.

사용법:
    # 전체 1,366장을 224px로 리사이즈해서 압축 (가장 흔한 용도)
    python3 prepare_dataset_for_upload.py --train_dir /path/to/train --out train_resized.zip

    # 특정 id 목록만 선택 (예: 진단 스크립트가 뽑아준 상위 N개 검증할 때)
    python3 prepare_dataset_for_upload.py --train_dir /path/to/train \
        --ids train_00004,train_00009,train_00358 --out top_ood.zip

    # submission.csv에서 특정 컬럼 상위 N개를 자동으로 뽑아서 압축
    python3 prepare_dataset_for_upload.py --train_dir /path/to/train \
        --from_submission submission_b_v9.csv --score_col ood_score --top_n 30 \
        --out top30_ood.zip
"""
import argparse
import os
import zipfile
import pandas as pd
from PIL import Image


def resize_and_save(src_path, dst_path, max_side, quality):
    img = Image.open(src_path).convert("RGB")
    img.thumbnail((max_side, max_side))
    img.save(dst_path, "JPEG", quality=quality)


def collect_target_ids(args):
    if args.ids:
        return [x.strip() for x in args.ids.split(",")]
    if args.from_submission:
        df = pd.read_csv(args.from_submission)
        if args.score_col not in df.columns:
            raise ValueError(f"'{args.score_col}' 컬럼이 {args.from_submission}에 없습니다. "
                              f"사용 가능한 컬럼: {list(df.columns)}")
        top = df.nlargest(args.top_n, args.score_col)
        return top["id"].tolist()
    # 둘 다 없으면 train_dir의 모든 jpg 사용
    return None


def main(args):
    target_ids = collect_target_ids(args)

    if target_ids is None:
        filenames = sorted(f for f in os.listdir(args.train_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    else:
        filenames = []
        for id_ in target_ids:
            # 확장자가 이미 포함된 경우와 아닌 경우 모두 처리
            candidates = [id_, f"{id_}.jpg", f"{id_}.jpeg", f"{id_}.png"]
            found = None
            for c in candidates:
                if os.path.exists(os.path.join(args.train_dir, c)):
                    found = c
                    break
            if found is None:
                print(f"[경고] 파일을 찾을 수 없음: {id_}")
                continue
            filenames.append(found)

    print(f"대상 이미지 수: {len(filenames)}")

    tmp_dir = args.out + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    for i, fname in enumerate(filenames):
        src = os.path.join(args.train_dir, fname)
        dst = os.path.join(tmp_dir, fname)
        try:
            resize_and_save(src, dst, args.max_side, args.quality)
        except Exception as e:
            print(f"[경고] 리사이즈 실패: {fname} ({e})")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(filenames)}")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(tmp_dir):
            zf.write(os.path.join(tmp_dir, fname), arcname=fname)

    # 임시 폴더 정리
    for fname in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fname))
    os.rmdir(tmp_dir)

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\n저장 완료: {args.out} ({size_mb:.1f} MB)")
    if size_mb > 25:
        print("[주의] 25MB를 초과합니다. --max_side를 줄이거나(예: 160) "
              "--top_n으로 이미지 수를 줄여서 다시 시도하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="원본 train 이미지 폴더")
    parser.add_argument("--out", required=True, help="저장할 zip 파일 경로")
    parser.add_argument("--max_side", type=int, default=224, help="긴 변 기준 리사이즈 크기 (기본 224)")
    parser.add_argument("--quality", type=int, default=70, help="JPEG 압축 품질 (기본 70)")
    parser.add_argument("--ids", default=None, help="쉼표로 구분된 id 목록 (예: train_00004,train_00009)")
    parser.add_argument("--from_submission", default=None, help="submission csv 경로 (상위 N개 자동 선택용)")
    parser.add_argument("--score_col", default=None, help="--from_submission 사용 시 정렬 기준 컬럼명")
    parser.add_argument("--top_n", type=int, default=30, help="--from_submission 사용 시 상위 몇 개")
    args = parser.parse_args()
    main(args)