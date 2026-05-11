"""file_path 기반 라벨 생성/검증 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml_config import OUTPUT_ML_DIR
from ml_utils import (
    detect_label_style,
    format_class_distribution_table,
    load_feature_csv,
    normalize_label_from_path,
    save_dataframe_csv,
    save_text_file,
    validate_label_column,
)


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="file_path 기반 label 생성 및 검증 도구")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 feature CSV 경로 (예: ../outputs/features/features_all.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 labeled CSV 경로 (예: ../outputs/features/features_all_labeled.csv)",
    )
    parser.add_argument(
        "--drop-unknown",
        action="store_true",
        help="unknown 라벨 행을 제거하고 저장할지 여부",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="생성할 라벨 컬럼명",
    )
    return parser.parse_args()


def build_label_check_summary(
    df: pd.DataFrame,
    label_column: str,
    summary_path: Path,
    distribution_path: Path,
) -> None:
    """라벨 검증 텍스트/CSV 결과를 저장한다."""
    validate_label_column(df, label_column)
    class_dist = format_class_distribution_table(df[label_column])
    save_dataframe_csv(class_dist, distribution_path)

    lines: list[str] = []
    lines.append("[라벨 검증 요약]")
    lines.append(f"- 전체 데이터 개수: {len(df)}")
    lines.append(f"- 라벨 컬럼명: {label_column}")
    lines.append(f"- 라벨 스타일 추정: {detect_label_style(df[label_column])}")
    lines.append("")
    lines.append("[라벨별 샘플 수/비율]")
    for _, row in class_dist.iterrows():
        lines.append(f"- {row['label']}: {int(row['count'])}개 ({float(row['ratio']) * 100:.2f}%)")
    lines.append("")
    lines.append("[라벨별 file_path 샘플(최대 5개)]")

    for label_name in class_dist["label"].tolist():
        lines.append(f"\n<{label_name}>")
        sample_paths = (
            df.loc[df[label_column].astype(str) == label_name, "file_path"]
            .astype(str)
            .head(5)
            .tolist()
        )
        if not sample_paths:
            lines.append("- (샘플 없음)")
            continue
        for item in sample_paths:
            inferred = normalize_label_from_path(item)
            matching = "일치" if inferred == label_name else f"불일치(추정:{inferred})"
            lines.append(f"- {item} | 매핑검증: {matching}")

    save_text_file(summary_path, "\n".join(lines))


def main() -> None:
    """라벨 생성 및 검증 파이프라인을 실행한다."""
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = OUTPUT_ML_DIR / "label_check_summary.txt"
    distribution_path = OUTPUT_ML_DIR / "class_distribution.csv"
    OUTPUT_ML_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_csv(input_path)
    if "file_path" not in df.columns:
        raise KeyError("[오류] file_path 컬럼이 없어 라벨 자동 생성이 불가능합니다.")

    # file_path 규칙 기반으로 라벨을 생성한다.
    df[args.label_column] = df["file_path"].astype(str).map(normalize_label_from_path)

    if args.drop_unknown:
        before_rows = len(df)
        df = df.loc[df[args.label_column] != "unknown"].reset_index(drop=True)
        dropped = before_rows - len(df)
        print(f"[안내] unknown 라벨 {dropped}개를 제거했습니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataframe_csv(df, output_path)
    build_label_check_summary(
        df=df,
        label_column=args.label_column,
        summary_path=summary_path,
        distribution_path=distribution_path,
    )

    print("[완료] 라벨 생성/검증 결과 저장")
    print(f"- labeled CSV: {output_path}")
    print(f"- label 요약: {summary_path}")
    print(f"- class 분포: {distribution_path}")


if __name__ == "__main__":
    main()
