"""저장된 분류 모델로 신규 특징 CSV 상태를 예측하는 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml_config import (
    BEST_MODEL_PATH,
    DEFAULT_LABEL_COLUMN,
    LABEL_ENCODER_PATH,
    OUTPUT_ML_DIR,
    PREDICTION_OUTPUT_PATH,
    SCALER_PATH,
)
from ml_utils import (
    clean_feature_matrix,
    ensure_directories,
    get_effective_exclude_columns,
    load_feature_csv,
    load_pickle,
    save_dataframe_csv,
)


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="저장된 모델 기반 설비 상태 예측")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="예측할 신규 특징 CSV 경로",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=DEFAULT_LABEL_COLUMN,
        help="입력 CSV에 라벨 컬럼이 존재할 경우 제외하기 위한 컬럼명",
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        type=str,
        default=str(BEST_MODEL_PATH),
        help="best_model 파일 경로 (예: ../models/best_model_multiclass.pkl)",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=None,
        help="스케일러 경로(미지정 시 모델명 기반 자동 추정)",
    )
    parser.add_argument(
        "--label-encoder-path",
        type=str,
        default=None,
        help="라벨 인코더 경로(미지정 시 모델명 기반 자동 추정)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PREDICTION_OUTPUT_PATH),
        help="예측 결과 저장 CSV 경로",
    )
    return parser.parse_args()


def resolve_auxiliary_paths(
    model_path: Path,
    scaler_path: Path | None,
    label_encoder_path: Path | None,
) -> tuple[Path, Path]:
    """모델 파일명에 맞는 scaler/label_encoder 경로를 자동 보정한다."""
    if scaler_path is not None and label_encoder_path is not None:
        return scaler_path, label_encoder_path

    stem = model_path.stem
    if stem.startswith("best_model_"):
        suffix = stem.replace("best_model_", "", 1)
        inferred_scaler = model_path.parent / f"scaler_{suffix}.pkl"
        inferred_encoder = model_path.parent / f"label_encoder_{suffix}.pkl"
    else:
        inferred_scaler = SCALER_PATH
        inferred_encoder = LABEL_ENCODER_PATH

    final_scaler = scaler_path if scaler_path is not None else inferred_scaler
    final_encoder = label_encoder_path if label_encoder_path is not None else inferred_encoder
    return final_scaler, final_encoder


def apply_feature_schema(
    input_features: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> pd.DataFrame:
    """학습 시 사용된 피처 스키마에 맞춰 입력 피처를 정렬/보정한다."""
    aligned = input_features.copy()

    # 학습에 있었지만 신규 입력에는 없는 컬럼은 기본값으로 채운다.
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = float(fill_values.get(col, 0.0))

    # 신규 입력에만 있는 컬럼은 제거하고 학습 컬럼 순서대로 정렬한다.
    aligned = aligned[feature_columns]
    return aligned


def build_probability_frame(
    model: Any,
    x_input: np.ndarray | pd.DataFrame,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """모델이 확률 출력을 지원하면 클래스별 확률 테이블을 생성한다."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_input)
        class_names = [str(c) for c in label_encoder.classes_]
        columns = [f"proba_{name}" for name in class_names]
        return pd.DataFrame(proba, columns=columns)

    # 확률 미지원 모델인 경우 빈 데이터프레임 반환
    return pd.DataFrame()


def main() -> None:
    """저장된 모델 객체를 사용해 신규 특징 데이터의 상태를 예측한다."""
    args = parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    scaler_arg = Path(args.scaler_path) if args.scaler_path else None
    encoder_arg = Path(args.label_encoder_path) if args.label_encoder_path else None
    scaler_path, label_encoder_path = resolve_auxiliary_paths(
        model_path=model_path,
        scaler_path=scaler_arg,
        label_encoder_path=encoder_arg,
    )
    output_path = Path(args.output)

    ensure_directories([OUTPUT_ML_DIR, output_path.parent])

    # 1) 입력 CSV/모델/전처리 객체 로드
    input_df = load_feature_csv(input_path)
    model_bundle = load_pickle(model_path)
    scaler = load_pickle(scaler_path)
    label_encoder = load_pickle(label_encoder_path)

    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        raise ValueError(
            "[오류] best_model.pkl 형식이 올바르지 않습니다. "
            "train_ml.py로 생성한 파일인지 확인하세요."
        )
    if not isinstance(scaler, StandardScaler):
        raise TypeError("[오류] scaler.pkl이 StandardScaler 객체가 아닙니다.")
    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError("[오류] label_encoder.pkl이 LabelEncoder 객체가 아닙니다.")

    model = model_bundle["model"]
    model_name = str(model_bundle.get("model_name", "best_model"))
    feature_columns = model_bundle.get("feature_columns", [])
    fill_values = model_bundle.get("fill_values", {})

    if not feature_columns:
        raise ValueError("[오류] 모델 메타데이터에 feature_columns 정보가 없습니다.")

    # 2) 예측용 피처 추출 (라벨/경로성 컬럼 제외 + 숫자형만)
    exclude_columns = get_effective_exclude_columns(
        input_df,
        label_column=args.label_column,
    )
    x_candidate = input_df.drop(columns=exclude_columns, errors="ignore")
    x_numeric = x_candidate.select_dtypes(include=[np.number]).copy()
    if x_numeric.empty:
        raise ValueError("[오류] 예측 가능한 숫자형 feature가 없습니다.")

    # 3) 학습 스키마에 맞춰 컬럼 정렬/결측 보정
    x_aligned = apply_feature_schema(
        input_features=x_numeric,
        feature_columns=feature_columns,
        fill_values=fill_values,
    )
    fill_series = pd.Series(fill_values, dtype=float)
    x_clean, _ = clean_feature_matrix(x_aligned, fill_values=fill_series)

    # 4) 스케일링 + 예측
    x_scaled = scaler.transform(x_clean)
    # 학습 시 RandomForest는 비스케일 입력, SVM/MLP는 스케일 입력을 사용했다.
    model_input = x_scaled if model_name in {"svm", "mlp"} else x_clean
    y_pred_encoded = model.predict(model_input)
    y_pred_label = label_encoder.inverse_transform(y_pred_encoded)

    # 5) 확률값(가능한 경우) 계산
    prob_df = build_probability_frame(model=model, x_input=model_input, label_encoder=label_encoder)

    # 6) 결과 구성 및 저장
    result_df = input_df.copy()
    result_df["predicted_label"] = y_pred_label
    if not prob_df.empty:
        result_df = pd.concat([result_df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
        result_df["predicted_confidence"] = prob_df.max(axis=1).round(6)
    else:
        result_df["predicted_confidence"] = np.nan
    if args.label_column in result_df.columns:
        result_df["actual_label"] = result_df[args.label_column].astype(str)
        result_df["correct"] = result_df["actual_label"] == result_df["predicted_label"]

    save_dataframe_csv(result_df, output_path)

    # 7) 콘솔 출력
    print("[예측 완료]")
    print(f"- 사용 모델: {model_name}")
    print(f"- 입력 파일: {input_path}")
    print(f"- 예측 건수: {len(result_df)}")
    print(f"- 저장 파일: {output_path}")

    # 결과 일부 미리보기(보고서/발표 확인 편의)
    preview_cols = ["predicted_label", "predicted_confidence"]
    if "actual_label" in result_df.columns:
        preview_cols.extend(["actual_label", "correct"])
    print("\n[예측 결과 미리보기]")
    print(result_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
