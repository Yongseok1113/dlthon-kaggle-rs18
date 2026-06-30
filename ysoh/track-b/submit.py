"""
Kaggle 대회에 submission.csv를 제출하는 스크립트.

사전 준비:
  1) ~/.kaggle/kaggle.json 에 Legacy API Credentials가 설정되어 있어야 함
     (kaggle.com -> Settings -> API -> "Create Legacy API Key")
  2) 대회 규칙에 동의(Join Competition)가 되어 있어야 함
     (웹사이트에서 해당 competition 페이지의 "I Understand and Accept" 클릭 필요)

사용법:
  python3 submit.py --competition rs-18-track-a --file submission_a.csv --message "baseline v1"
  python3 submit.py --competition rs-18-track-b --file submission_b.csv --message "phash+resnet18+cv v1"

  또는 환경변수로 기본값 지정 가능:
  export KAGGLE_COMPETITION=rs-18-track-b
  python3 submit.py --file submission_b.csv --message "v2"
"""
import argparse
import os
import subprocess
import sys
import pandas as pd


REQUIRED_COLUMNS = {
    "rs-18-track-a": {"id", "dusty_prob"},
    "rs-18-track-b": {"id", "mislabel_score", "dup_score", "ood_score"},
}


def validate_submission(file_path, competition):
    """제출 전 로컬에서 형식을 미리 검증 (컬럼명, 값 범위, 행 수)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"제출 파일을 찾을 수 없습니다: {file_path}")

    df = pd.read_csv(file_path)

    expected_cols = REQUIRED_COLUMNS.get(competition)
    if expected_cols is None:
        print(f"[경고] '{competition}'에 대한 컬럼 검증 규칙이 없습니다. 컬럼 검증을 건너뜁니다.")
    else:
        actual_cols = set(df.columns)
        if actual_cols != expected_cols:
            raise ValueError(
                f"컬럼이 일치하지 않습니다.\n"
                f"  기대: {sorted(expected_cols)}\n"
                f"  실제: {sorted(actual_cols)}"
            )

        score_cols = [c for c in df.columns if c != "id"]
        for col in score_cols:
            if df[col].isnull().any():
                raise ValueError(f"'{col}' 컬럼에 결측치가 있습니다.")
            if (df[col] < 0).any() or (df[col] > 1).any():
                raise ValueError(f"'{col}' 컬럼 값이 0~1 범위를 벗어났습니다. "
                                  f"min={df[col].min()}, max={df[col].max()}")

    if df["id"].duplicated().any():
        n_dup = df["id"].duplicated().sum()
        raise ValueError(f"id 컬럼에 중복이 {n_dup}건 있습니다.")

    print(f"[검증 완료] 행 수: {len(df)}, 컬럼: {list(df.columns)}")
    return df


def submit(competition, file_path, message):
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition,
        "-f", file_path,
        "-m", message,
    ]
    print("실행 명령:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("kaggle 제출 명령이 실패했습니다. 위 stderr 메시지를 확인하세요.")

    return result.stdout


def show_submission_status(competition, n=5):
    cmd = ["kaggle", "competitions", "submissions", "-c", competition]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"\n=== {competition} 최근 제출 내역 ===")
    print(result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competition",
        default=os.environ.get("KAGGLE_COMPETITION"),
        choices=["rs-18-track-a", "rs-18-track-b"],
        required=os.environ.get("KAGGLE_COMPETITION") is None,
        help="제출할 대회 slug",
    )
    parser.add_argument("--file", required=True, help="제출할 submission.csv 경로")
    parser.add_argument("--message", required=True, help="제출 메모 (예: 'baseline v1')")
    parser.add_argument("--skip-validate", action="store_true", help="로컬 형식 검증을 건너뜀")
    args = parser.parse_args()

    if not args.skip_validate:
        validate_submission(args.file, args.competition)
    else:
        print("[주의] --skip-validate 옵션으로 로컬 검증을 건너뜁니다.")

    submit(args.competition, args.file, args.message)
    show_submission_status(args.competition)