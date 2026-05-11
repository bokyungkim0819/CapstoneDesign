"""머신러닝 모델 평가 및 시각화 유틸리티."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from ml_config import (
    BEST_MODEL_PATH,
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_COLUMN,
    LABEL_ENCODER_PATH,
    OUTPUT_ML_DIR,
    SCALER_PATH,
)
from ml_utils import (
    clean_feature_matrix,
    get_effective_exclude_columns,
    load_feature_csv,
    load_pickle,
    select_numeric_features,
    validate_label_column,
)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """예측 결과의 핵심 분류 성능 지표를 계산한다."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_classification_report_text(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> str:
    """classification_report 문자열을 생성한다."""
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str,
) -> None:
    """혼동행렬 이미지를 저장한다."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_metrics_table(metrics: dict[str, float], model_name: str) -> None:
    """사람이 읽기 쉬운 형태로 성능 지표를 출력한다."""
    print(f"\n[{model_name} 평가 결과]")
    print(f"- Accuracy       : {metrics['accuracy']:.4f}")
    print(f"- PrecisionMacro : {metrics['precision_macro']:.4f}")
    print(f"- RecallMacro    : {metrics['recall_macro']:.4f}")
    print(f"- F1Macro        : {metrics['f1_macro']:.4f}")


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="저장된 머신러닝 모델 평가 도구")
    parser.add_argument(
        "--features",
        type=str,
        default=str(DEFAULT_FEATURE_CSV),
        help="평가할 특징 CSV 경로",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=DEFAULT_LABEL_COLUMN,
        help="정답 라벨 컬럼명",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(BEST_MODEL_PATH),
        help="평가 대상 모델 파일 경로(best_model.pkl)",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=str(SCALER_PATH),
        help="스케일러 파일 경로(scaler.pkl)",
    )
    parser.add_argument(
        "--label-encoder-path",
        type=str,
        default=str(LABEL_ENCODER_PATH),
        help="라벨 인코더 파일 경로(label_encoder.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ML_DIR),
        help="평가 결과 저장 폴더",
    )
    return parser.parse_args()


def main() -> None:
    """저장된 모델을 불러와 데이터셋 기준으로 성능을 평가한다."""
    args = parse_args()

    feature_path = Path(args.features)
    model_path = Path(args.model_path)
    scaler_path = Path(args.scaler_path)
    label_encoder_path = Path(args.label_encoder_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드 및 라벨 검증
    df = load_feature_csv(feature_path)
    validate_label_column(df, args.label_column)

    # 2) 모델/전처리 객체 로드
    model_bundle = load_pickle(model_path)
    scaler = load_pickle(scaler_path)
    label_encoder = load_pickle(label_encoder_path)

    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        raise ValueError(
            "[오류] best_model.pkl 형식이 올바르지 않습니다. "
            "train_ml.py로 저장한 파일인지 확인하세요."
        )
    if not isinstance(scaler, StandardScaler):
        raise TypeError("[오류] scaler.pkl이 StandardScaler 객체가 아닙니다.")

    model = model_bundle["model"]
    model_name = str(model_bundle.get("model_name", "best_model"))
    feature_columns = model_bundle.get("feature_columns", [])
    fill_values_dict = model_bundle.get("fill_values", {})

    # 3) 입력 피처 정리(학습 시 컬럼 구조와 동일하게 정렬)
    exclude_columns = get_effective_exclude_columns(df, args.label_column)
    x_numeric, _ = select_numeric_features(df, exclude_columns=exclude_columns)

    if not feature_columns:
        raise ValueError("[오류] 모델 메타데이터에 feature_columns가 없습니다.")

    # 학습 시 존재했지만 현재 CSV에는 없는 피처를 기본값으로 복원
    for col in feature_columns:
        if col not in x_numeric.columns:
            default_value = float(fill_values_dict.get(col, 0.0))
            x_numeric[col] = default_value

    # 현재 CSV에만 있고 학습에 없던 컬럼은 제거
    x_numeric = x_numeric[feature_columns]
    fill_series = pd.Series(fill_values_dict, dtype=float)
    x_clean, _ = clean_feature_matrix(x_numeric, fill_values=fill_series)

    # 4) 정답 라벨 인코딩 및 예측
    y_true = label_encoder.transform(df[args.label_column].astype(str))
    x_scaled = scaler.transform(x_clean)
    y_pred = model.predict(x_scaled)

    # 5) 정량 지표 + 리포트 + 혼동행렬 저장
    class_names = label_encoder.classes_.tolist()
    metrics = evaluate_predictions(y_true=y_true, y_pred=y_pred)
    report_text = build_classification_report_text(y_true=y_true, y_pred=y_pred, class_names=class_names)

    report_path = output_dir / f"classification_report_{model_name}_evaluate.txt"
    report_path.write_text(report_text, encoding="utf-8")

    cm_path = output_dir / f"confusion_matrix_{model_name}_evaluate.png"
    save_confusion_matrix_figure(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=cm_path,
        title=f"Confusion Matrix ({model_name})",
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                **metrics,
            }
        ]
    )
    metrics_path = output_dir / f"evaluation_metrics_{model_name}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # 6) 콘솔 출력
    print_metrics_table(metrics=metrics, model_name=model_name)
    print("\n[분류 리포트]")
    print(report_text)
    print(f"[저장] 분류 리포트: {report_path}")
    print(f"[저장] 혼동행렬 이미지: {cm_path}")
    print(f"[저장] 정량 지표 CSV: {metrics_path}")


if __name__ == "__main__":
    main()
