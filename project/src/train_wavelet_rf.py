"""Wavelet 특징 CSV로 Random Forest 정상/고장 및 고장유형 모델을 학습한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from config import PROJECT_ROOT
from ml_utils import (
    apply_binary_task,
    clean_feature_matrix,
    infer_equipment_id,
    load_feature_csv,
    save_dataframe_csv,
    save_pickle,
    select_numeric_features,
)


DEFAULT_FEATURES = (
    PROJECT_ROOT / "outputs" / "features" / "features_wavelet_labeled.csv"
)
MODEL_DIR = PROJECT_ROOT / "models" / "wavelet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ml" / "wavelet"
LABEL_EN = {
    "정상": "Normal",
    "고장": "Fault",
    "축정렬불량": "Misalignment",
    "베어링불량": "Bearing fault",
    "회전체불평형": "Unbalance",
    "벨트느슨함": "Belt looseness",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DWT 특징 기반 Random Forest 모델 학습"
    )
    parser.add_argument("--features", type=str, default=str(DEFAULT_FEATURES))
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default="multiclass",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--run-name",
        type=str,
        default="wavelet",
        help="모델/평가 파일명 구분자(예: prototype, full)",
    )
    parser.add_argument(
        "--group-split",
        action="store_true",
        help="설비 ID가 train/test 양쪽에 섞이지 않도록 분할",
    )
    parser.add_argument("--group-level", type=int, default=None)
    return parser.parse_args()


def _split_data(
    features: pd.DataFrame,
    labels: pd.Series,
    source_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("[오류] --test-size는 0과 1 사이여야 합니다.")

    if args.group_split:
        if "file_path" not in source_df:
            raise KeyError("[오류] group split에는 file_path가 필요합니다.")
        groups = source_df["file_path"].astype(str).map(
            lambda value: infer_equipment_id(value, args.group_level)
        )
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_idx, test_idx = next(splitter.split(features, labels, groups))
        return (
            features.iloc[train_idx].copy(),
            features.iloc[test_idx].copy(),
            labels.iloc[train_idx].copy(),
            labels.iloc[test_idx].copy(),
            "equipment_group_split",
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    return x_train, x_test, y_train, y_test, "stratified_random_split"


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder: LabelEncoder,
    output_path: Path,
    title: str,
) -> None:
    labels = np.arange(len(encoder.classes_))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    display_labels = [
        LABEL_EN.get(str(label), str(label)) for label in encoder.classes_
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_feature_importance(
    model: RandomForestClassifier,
    feature_columns: list[str],
    output_path: Path,
    csv_path: Path,
) -> None:
    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    save_dataframe_csv(importance, csv_path)

    top = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"], color="#3569A8")
    ax.set_title("Top Wavelet Feature Importance - Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.run_name).strip("_")
    if not run_name:
        raise ValueError("[오류] --run-name에 사용할 수 있는 문자가 없습니다.")
    feature_path = Path(args.features)
    df = load_feature_csv(feature_path)
    if "label" not in df:
        raise KeyError("[오류] Wavelet 특징 CSV에 label 컬럼이 없습니다.")

    labels = df["label"].astype(str).str.strip()
    valid_mask = labels.ne("unknown") & labels.ne("")
    df = df.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].reset_index(drop=True)
    if args.task == "binary":
        labels = apply_binary_task(labels)

    excluded = [
        "file_path",
        "label",
        "column",
        "fs_hz",
        "samples",
        "wavelet_name",
        "wavelet_level",
    ]
    x_numeric, feature_columns = select_numeric_features(df, excluded)
    x_clean, fill_values = clean_feature_matrix(x_numeric)
    x_train, x_test, y_train_text, y_test_text, split_mode = _split_data(
        x_clean,
        labels,
        df,
        args,
    )

    encoder = LabelEncoder()
    encoder.fit(labels)
    y_train = encoder.transform(y_train_text)
    y_test = encoder.transform(y_test_text)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = {
        "task": args.task,
        "split_mode": split_mode,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "feature_count": len(feature_columns),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{run_name}_{args.task}_{'group' if args.group_split else 'random'}"
    )
    model_path = MODEL_DIR / f"random_forest_{suffix}.pkl"
    metrics_path = OUTPUT_DIR / f"metrics_{suffix}.csv"
    confusion_path = OUTPUT_DIR / f"confusion_matrix_{suffix}.png"
    importance_path = OUTPUT_DIR / f"feature_importance_{suffix}.png"
    importance_csv = OUTPUT_DIR / f"feature_importance_{suffix}.csv"

    bundle = {
        "model_name": "random_forest",
        "analysis": "wavelet_dwt",
        "task": args.task,
        "split_mode": split_mode,
        "model": model,
        "label_encoder": encoder,
        "feature_columns": feature_columns,
        "fill_values": fill_values.to_dict(),
        "excluded_columns": excluded,
    }
    save_pickle(bundle, model_path)
    save_dataframe_csv(pd.DataFrame([metrics]), metrics_path)
    _save_confusion_matrix(
        y_test,
        y_pred,
        encoder,
        confusion_path,
        f"Wavelet Random Forest ({args.task}, {split_mode})",
    )
    _save_feature_importance(
        model,
        feature_columns,
        importance_path,
        importance_csv,
    )

    print("[Wavelet Random Forest 학습 완료]")
    print(f"- split: {split_mode}")
    print(f"- features: {len(feature_columns)}")
    print(f"- accuracy: {metrics['accuracy']:.4f}")
    print(f"- F1-macro: {metrics['f1_macro']:.4f}")
    print(f"- model: {model_path}")
    print(f"- metrics: {metrics_path}")
    print(f"- confusion matrix: {confusion_path}")


if __name__ == "__main__":
    main()
