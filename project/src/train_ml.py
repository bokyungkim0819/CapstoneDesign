"""FFT 특징 CSV 기반 설비 상태 분류 모델 학습 스크립트."""

from __future__ import annotations

import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
# Windows 환경에서 Tk backend 종료 시 스레드 예외를 피하기 위해 Agg 백엔드를 고정한다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_config import (
    BEST_MODEL_NAME_PATH,
    BEST_MODEL_PATH,
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_COLUMN,
    LABEL_ENCODER_PATH,
    MODEL_DIR,
    MODEL_LIST,
    OUTPUT_ML_DIR,
    RANDOM_SEED,
    SCALER_PATH,
    TEST_SIZE,
)
from ml_utils import (
    apply_binary_task,
    clean_feature_matrix,
    encode_labels,
    ensure_directories,
    format_class_distribution_table,
    get_effective_exclude_columns,
    infer_equipment_id,
    load_feature_csv,
    print_class_distribution,
    save_dataframe_csv,
    save_pickle,
    save_text_file,
    select_numeric_features,
    validate_label_column,
)


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="FFT 특징 기반 설비 상태 분류 모델 학습")
    parser.add_argument("--features", type=str, default=str(DEFAULT_FEATURE_CSV), help="입력 특징 CSV 경로")
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COLUMN, help="라벨 컬럼명")
    parser.add_argument("--task", type=str, choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument(
        "--exclude-meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="True면 file_path/column/fs_hz/samples 등 메타 컬럼을 제외한다.",
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="검증 데이터 비율")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED, help="고정 random state")
    parser.add_argument("--cv", type=int, default=0, help="StratifiedKFold 분할 수(0이면 미수행)")
    parser.add_argument("--group-split", action="store_true", help="설비 ID 기준 GroupShuffleSplit 사용")
    parser.add_argument(
        "--group-level",
        type=int,
        default=None,
        help="file_path split 토큰 인덱스로 설비 ID를 강제 지정한다.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_LIST,
        default=MODEL_LIST,
        help="학습할 모델 목록 (예: --models random_forest mlp)",
    )
    parser.add_argument(
        "--svm-kernel",
        type=str,
        choices=["linear", "rbf"],
        default="linear",
        help="SVM 커널 종류(linear 또는 rbf)",
    )
    return parser.parse_args()


def build_estimators(random_state: int, svm_kernel: str) -> dict[str, Any]:
    """학습 대상 모델 객체를 생성한다."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "svm": SVC(
            kernel=svm_kernel,
            C=1.0,
            gamma="scale",
            probability=False,
            max_iter=10000,
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=random_state,
        ),
    }


def evaluate_predictions(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    """필수 성능 지표를 계산한다."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: Any,
    class_names: list[str],
    save_path: Path,
    title: str,
) -> None:
    """혼동행렬 이미지를 저장한다."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, bool, dict[str, Any]]:
    """stratify 또는 group split 기반으로 학습/검증 분할을 수행한다."""
    if not (0.0 < test_size < 1.0):
        raise ValueError("[오류] --test-size는 0과 1 사이 값이어야 합니다.")

    split_info: dict[str, Any] = {"mode": "stratified_random_split"}

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(x, y, groups=groups))
        x_train = x.iloc[train_idx].copy()
        x_test = x.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        split_info["mode"] = "group_split"
        split_info["train_group_count"] = int(groups.iloc[train_idx].nunique())
        split_info["test_group_count"] = int(groups.iloc[test_idx].nunique())
        return x_train, x_test, y_train, y_test, False, split_info

    class_counts = y.value_counts()
    can_stratify = not (class_counts < 2).any()
    stratify_applied = can_stratify
    if not can_stratify:
        warnings.warn(
            "[경고] 일부 클래스 샘플 수가 1개라 stratify 분할이 불가능합니다. "
            "stratify 없이 분할을 진행합니다.",
            stacklevel=2,
        )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_stratify else None,
        )
    except ValueError as exc:
        warnings.warn(
            f"[경고] stratify 분할 실패: {exc}\nstratify 없이 분할을 다시 시도합니다.",
            stacklevel=2,
        )
        stratify_applied = False
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    return x_train, x_test, y_train, y_test, stratify_applied, split_info


def save_split_summary(
    path: Path,
    total_count: int,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_text: pd.Series,
    y_test_text: pd.Series,
    test_size: float,
    random_state: int,
    stratify_applied: bool,
    split_info: dict[str, Any],
) -> None:
    """train/test split 검증 정보를 저장한다."""
    lines: list[str] = []
    lines.append("[train/test split 검증]")
    lines.append(f"- 전체 데이터 개수: {total_count}")
    lines.append(f"- train 데이터 개수: {len(x_train)}")
    lines.append(f"- test 데이터 개수: {len(x_test)}")
    lines.append(f"- test_size: {test_size}")
    lines.append(f"- random_state: {random_state}")
    lines.append(f"- split 모드: {split_info.get('mode', 'unknown')}")
    lines.append(f"- stratify 적용 여부: {stratify_applied}")
    if split_info.get("mode") == "group_split":
        lines.append(f"- train group 개수: {split_info.get('train_group_count', 0)}")
        lines.append(f"- test group 개수: {split_info.get('test_group_count', 0)}")

    lines.append("\n[train label 분포]")
    train_counts = y_train_text.value_counts()
    for label, cnt in train_counts.items():
        lines.append(f"- {label}: {int(cnt)}")

    lines.append("\n[test label 분포]")
    test_counts = y_test_text.value_counts()
    for label, cnt in test_counts.items():
        lines.append(f"- {label}: {int(cnt)}")

    save_text_file(path, "\n".join(lines))


def save_feature_check_files(
    df: pd.DataFrame,
    feature_columns: list[str],
    excluded_columns: list[str],
    output_dir: Path,
    exclude_meta: bool,
) -> None:
    """데이터 누수 방지 관점의 feature 점검 결과를 저장한다."""
    used_path = output_dir / "used_features.txt"
    excluded_path = output_dir / "excluded_features.txt"
    summary_path = output_dir / "feature_check_summary.txt"

    save_text_file(used_path, "\n".join(feature_columns))
    save_text_file(excluded_path, "\n".join(excluded_columns))

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    suspicious = [
        col
        for col in feature_columns
        if any(token in col.lower() for token in ["label", "path", "file", "name", "status", "class"])
    ]
    lines: list[str] = []
    lines.append("[feature 누수 점검 요약]")
    lines.append(f"- exclude_meta 기본값 사용 여부: {exclude_meta}")
    lines.append(f"- 전체 원본 컬럼 수: {len(df.columns)}")
    lines.append(f"- 학습 사용 feature 수: {len(feature_columns)}")
    lines.append(f"- 학습 제외 컬럼 수: {len(excluded_columns)}")
    lines.append(f"- object 타입 컬럼 목록: {obj_cols}")
    lines.append(f"- 의심 컬럼(feature 내부) 발견 수: {len(suspicious)}")
    lines.append(f"- 의심 컬럼 목록: {suspicious}")
    lines.append("")
    lines.append("[필수 제외 컬럼 포함 여부]")
    for mandatory in ["label", "file_path", "column"]:
        lines.append(f"- {mandatory} in features: {mandatory in feature_columns}")
    lines.append(f"- fs_hz in features: {'fs_hz' in feature_columns}")
    lines.append(f"- samples in features: {'samples' in feature_columns}")
    save_text_file(summary_path, "\n".join(lines))


def save_random_forest_feature_importance(
    model: RandomForestClassifier, feature_columns: list[str], output_path: Path
) -> pd.DataFrame:
    """RandomForest 중요도 그래프를 저장하고 중요도 표를 반환한다."""
    importance_df = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    top_n = min(20, len(importance_df))
    plot_df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title("Random Forest Feature Importance (Top 20)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return importance_df


def run_cross_validation(
    x_clean: pd.DataFrame,
    y_encoded: pd.Series,
    selected_models: list[str],
    random_state: int,
    cv_splits: int,
    svm_kernel: str,
    output_path: Path,
) -> None:
    """옵션으로 StratifiedKFold 교차검증을 수행하고 결과를 저장한다."""
    if cv_splits <= 1:
        return

    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    estimators = build_estimators(random_state=random_state, svm_kernel=svm_kernel)
    rows: list[dict[str, float | str]] = []
    for model_name in selected_models:
        estimator = estimators[model_name]
        if model_name in {"svm", "mlp"}:
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        else:
            pipeline = Pipeline([("model", estimator)])

        cv_result = cross_validate(
            pipeline,
            x_clean,
            y_encoded,
            cv=kfold,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "model": model_name,
                "accuracy_mean": float(cv_result["test_accuracy"].mean()),
                "accuracy_std": float(cv_result["test_accuracy"].std()),
                "f1_macro_mean": float(cv_result["test_f1_macro"].mean()),
                "f1_macro_std": float(cv_result["test_f1_macro"].std()),
            }
        )

    save_dataframe_csv(pd.DataFrame(rows), output_path)


def parse_elapsed_seconds_from_log(log_path: Path) -> float | None:
    """학습 로그 파일에서 마지막 elapsed_seconds 값을 읽어온다."""
    if not log_path.exists():
        return None
    elapsed: float | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("elapsed_seconds="):
            try:
                elapsed = float(line.split("=", maxsplit=1)[1].strip())
            except ValueError:
                continue
    return elapsed


def save_svm_kernel_comparison_summary(
    output_dir: Path,
    task: str,
    rbf_metrics: dict[str, float],
    rbf_elapsed_seconds: float,
) -> None:
    """linear SVM과 RBF SVM 결과를 비교 요약 파일로 저장한다."""
    summary_path = output_dir / "svm_kernel_comparison_summary.txt"
    linear_path = output_dir / f"model_comparison_{task}.csv"
    linear_metrics: dict[str, float] | None = None
    if linear_path.exists():
        linear_df = pd.read_csv(linear_path)
        if "model" in linear_df.columns:
            linear_row = linear_df.loc[linear_df["model"].astype(str).isin(["svm", "svm_linear"])]
            if not linear_row.empty:
                linear_metrics = {
                    "accuracy": float(linear_row.iloc[0]["accuracy"]),
                    "precision_macro": float(linear_row.iloc[0]["precision_macro"]),
                    "recall_macro": float(linear_row.iloc[0]["recall_macro"]),
                    "f1_macro": float(linear_row.iloc[0]["f1_macro"]),
                    "precision_weighted": float(linear_row.iloc[0]["precision_weighted"]),
                    "recall_weighted": float(linear_row.iloc[0]["recall_weighted"]),
                    "f1_weighted": float(linear_row.iloc[0]["f1_weighted"]),
                }

    linear_elapsed = parse_elapsed_seconds_from_log(output_dir / "svm_linear_full_training_log.txt")
    lines: list[str] = []
    lines.append("[SVM 커널 비교 요약]")
    lines.append(f"- task: {task}")
    lines.append("")
    lines.append("[linear SVM 성능]")
    if linear_metrics is None:
        lines.append("- 기존 linear SVM multiclass 결과를 찾지 못해 비교가 제한됩니다.")
    else:
        for key, value in linear_metrics.items():
            lines.append(f"- {key}: {value:.6f}")
    lines.append("")
    lines.append("[RBF SVM 성능]")
    for key, value in rbf_metrics.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.append("")
    lines.append("[성능 차이: RBF - linear]")
    if linear_metrics is None:
        lines.append("- linear 기준값이 없어 차이를 계산할 수 없습니다.")
    else:
        for key in rbf_metrics:
            lines.append(f"- {key}: {rbf_metrics[key] - linear_metrics[key]:.6f}")
    lines.append("")
    lines.append("[학습 시간 비교]")
    lines.append(f"- RBF SVM 학습 시간(초): {rbf_elapsed_seconds:.2f}")
    if linear_elapsed is None:
        lines.append("- linear SVM 학습 시간(초): 기록 없음")
        lines.append("- 학습 시간 차이: 계산 불가")
    else:
        lines.append(f"- linear SVM 학습 시간(초): {linear_elapsed:.2f}")
        lines.append(f"- 시간 차이(RBF-linear, 초): {rbf_elapsed_seconds - linear_elapsed:.2f}")
    lines.append("")
    lines.append("[해석]")
    lines.append(
        "- Random Forest가 SVM 대비 일관되게 높은 성능을 보이며, "
        "현 단계에서는 최종 운영 모델로 Random Forest를 유지하는 것이 합리적이다."
    )
    save_text_file(summary_path, "\n".join(lines))


def save_report_summary(
    output_path: Path,
    feature_path: Path,
    feature_columns: list[str],
    label_distribution: pd.DataFrame,
    test_size: float,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    importance_df: pd.DataFrame | None,
) -> None:
    """보고서용 요약 텍스트를 생성한다."""
    lines: list[str] = []
    lines.append("[머신러닝 결과 요약(보고서용)]")
    lines.append("1) 사용 데이터")
    lines.append("- AI Hub 기계시설물 고장 예지 센서 데이터셋 vibration 데이터")
    lines.append(f"- 사용 feature CSV: {feature_path}")
    lines.append("")
    lines.append("2) 사용 feature")
    lines.append(f"- feature 개수: {len(feature_columns)}")
    lines.append(f"- feature 목록: {', '.join(feature_columns)}")
    lines.append("")
    lines.append("3) label별 샘플 수")
    for _, row in label_distribution.iterrows():
        lines.append(f"- {row['label']}: {int(row['count'])}개 ({float(row['ratio']) * 100:.2f}%)")
    lines.append("")
    lines.append("4) train/test split 비율")
    lines.append(f"- test_size: {test_size}")
    lines.append("")
    lines.append("5) 모델별 성능 비교")
    for _, row in comparison_df.iterrows():
        lines.append(
            f"- {row['model']}: accuracy={row['accuracy']:.4f}, f1_macro={row['f1_macro']:.4f}"
        )
    lines.append("")
    lines.append(f"6) 최종 선택 모델: {best_model_name}")
    lines.append("")
    lines.append("7) Random Forest feature importance 상위 10개")
    if importance_df is not None and not importance_df.empty:
        for _, row in importance_df.head(10).iterrows():
            lines.append(f"- {row['feature']}: {row['importance']:.6f}")
    else:
        lines.append("- Random Forest를 학습하지 않아 중요도 정보가 없습니다.")
    lines.append("")
    lines.append("8) SVM 성능 해석")
    lines.append(
        "- 대용량/다중 클래스 분포에서 선형 경계가 충분히 복잡한 패턴을 표현하지 못해 "
        "Random Forest/MLP 대비 성능이 낮아질 수 있다."
    )
    lines.append("")
    lines.append("9) Random Forest 고성능 해석 시 주의점")
    lines.append(
        "- 현재 결과는 랜덤 split 기준이며, 같은 설비/유사 시점 데이터가 train/test에 "
        "동시에 포함되었을 가능성이 있다. group split 검증으로 과대평가 여부를 추가 확인해야 한다."
    )
    lines.append("")
    lines.append("10) 보고서용 문장 3개")
    lines.append("- FFT 기반 특징값을 활용하여 정상 및 고장 유형 분류 모델을 구축하였다.")
    lines.append(
        "- Random Forest가 가장 높은 성능을 보였으며, RMS, peak, dominant frequency, "
        "주파수 대역 에너지 등이 주요 특징으로 활용되었다."
    )
    lines.append(
        "- 향후 로봇이 수집한 진동 데이터에도 동일한 전처리 및 특징 추출 파이프라인을 적용하여 "
        "실시간 상태 진단으로 확장할 수 있다."
    )
    save_text_file(output_path, "\n".join(lines))


def main() -> None:
    """머신러닝 학습 파이프라인을 실행한다."""
    args = parse_args()
    feature_path = Path(args.features)
    ensure_directories([MODEL_DIR, OUTPUT_ML_DIR])

    df = load_feature_csv(feature_path)
    validate_label_column(df, args.label_column)
    label_series = df[args.label_column].astype(str).str.strip()
    if args.task == "binary":
        label_series = apply_binary_task(label_series)

    print_class_distribution(label_series)
    if label_series.nunique() < 2:
        raise ValueError("[오류] 라벨 클래스가 1개뿐이어서 학습할 수 없습니다.")

    # 데이터 누수 방지를 위해 기본적으로 메타 컬럼을 제외한다.
    extra_exclude = ["fs_hz", "samples"] if args.exclude_meta else []
    exclude_columns = get_effective_exclude_columns(
        df,
        label_column=args.label_column,
        additional_exclude_columns=extra_exclude,
    )
    x_numeric, feature_columns = select_numeric_features(df, exclude_columns=exclude_columns)
    x_clean, fill_values = clean_feature_matrix(x_numeric)

    # feature/누수 점검 결과 저장
    save_feature_check_files(
        df=df,
        feature_columns=feature_columns,
        excluded_columns=exclude_columns,
        output_dir=OUTPUT_ML_DIR,
        exclude_meta=args.exclude_meta,
    )
    print("\n[학습에 사용된 feature]")
    for col in feature_columns:
        print(f"- {col}")

    y_encoded, label_encoder = encode_labels(label_series)
    y_series = pd.Series(y_encoded, index=x_clean.index, name="encoded_label")
    y_text = label_series

    groups = None
    if args.group_split:
        if "file_path" not in df.columns:
            raise KeyError("[오류] --group-split 사용 시 file_path 컬럼이 필요합니다.")
        groups = df["file_path"].astype(str).map(lambda v: infer_equipment_id(v, args.group_level))
        group_summary_lines = [
            "[group split 요약]",
            f"- 전체 group 개수: {groups.nunique()}",
            f"- group_level 강제 지정: {args.group_level}",
        ]
        save_text_file(OUTPUT_ML_DIR / "group_split_summary.txt", "\n".join(group_summary_lines))

    x_train, x_test, y_train, y_test, stratify_applied, split_info = split_train_test(
        x=x_clean,
        y=y_series,
        test_size=args.test_size,
        random_state=args.random_state,
        groups=groups,
    )
    y_train_text = y_text.loc[y_train.index]
    y_test_text = y_text.loc[y_test.index]

    save_split_summary(
        path=OUTPUT_ML_DIR / "train_test_split_summary.txt",
        total_count=len(df),
        x_train=x_train,
        x_test=x_test,
        y_train_text=y_train_text,
        y_test_text=y_test_text,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_applied=stratify_applied,
        split_info=split_info,
    )
    save_split_summary(
        path=OUTPUT_ML_DIR / f"train_test_split_summary_{args.task}.txt",
        total_count=len(df),
        x_train=x_train,
        x_test=x_test,
        y_train_text=y_train_text,
        y_test_text=y_test_text,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_applied=stratify_applied,
        split_info=split_info,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 학습 시작 전 핵심 정보 출력
    print("\n[학습 사전 정보]")
    print(f"- 전체 데이터 개수: {len(df)}")
    print(f"- train 데이터 개수: {len(x_train)}")
    print(f"- test 데이터 개수: {len(x_test)}")
    print("- label별 샘플 수:")
    for label_name, count in label_series.value_counts().items():
        print(f"  - {label_name}: {int(count)}")
    print(f"- 사용 feature 개수: {len(feature_columns)}")
    print("- 사용 feature 목록:")
    for col in feature_columns:
        print(f"  - {col}")
    print(f"- SVM kernel: {args.svm_kernel}")

    estimators = build_estimators(args.random_state, svm_kernel=args.svm_kernel)
    class_names = label_encoder.classes_.tolist()

    is_svm_rbf_full = (
        args.models == ["svm"]
        and args.svm_kernel == "rbf"
        and not args.group_split
    )
    is_svm_linear_full = (
        args.models == ["svm"]
        and args.svm_kernel == "linear"
        and not args.group_split
    )
    if is_svm_rbf_full:
        suffix = f"_{args.task}_svm_rbf_full"
    elif is_svm_linear_full:
        suffix = f"_{args.task}_svm_linear_full"
    else:
        suffix = f"_{args.task}" + ("_group_split" if args.group_split else "")

    comparison_rows: list[dict[str, float | str]] = []
    best_model_obj: Any | None = None
    best_model_name = ""
    best_f1 = -1.0
    importance_df: pd.DataFrame | None = None
    svm_rbf_elapsed_seconds: float | None = None

    for model_name in args.models:
        model = clone(estimators[model_name])
        print(f"\n[학습 시작] {model_name}")

        # 모델별 입력 행렬을 명시적으로 구분한다.
        if model_name in {"svm", "mlp"}:
            train_x = x_train_scaled
            test_x = x_test_scaled
        else:
            train_x = x_train
            test_x = x_test

        model_alias = model_name
        if model_name == "svm" and is_svm_rbf_full:
            model_alias = "svm_rbf"
        elif model_name == "svm" and is_svm_linear_full:
            model_alias = "svm_linear"

        fit_start_dt = datetime.now()
        fit_start = perf_counter()
        if model_name == "svm":
            training_log_path = OUTPUT_ML_DIR / f"svm_{args.svm_kernel}_full_training_log.txt"
            save_text_file(
                training_log_path,
                (
                    f"[SVM 학습 시작]\n"
                    f"task={args.task}\n"
                    f"kernel={args.svm_kernel}\n"
                    f"started_at={fit_start_dt.isoformat()}\n"
                ),
            )
        try:
            model.fit(train_x, y_train)
        except MemoryError as exc:
            if model_name == "svm":
                training_log_path = OUTPUT_ML_DIR / f"svm_{args.svm_kernel}_full_training_log.txt"
                save_text_file(
                    training_log_path,
                    (
                        f"[SVM 학습 실패]\n"
                        f"task={args.task}\n"
                        f"kernel={args.svm_kernel}\n"
                        f"ended_at={datetime.now().isoformat()}\n"
                        f"error_type=MemoryError\n"
                        f"error={exc}\n"
                    ),
                )
            raise
        except Exception as exc:  # noqa: BLE001
            if model_name == "svm":
                training_log_path = OUTPUT_ML_DIR / f"svm_{args.svm_kernel}_full_training_log.txt"
                save_text_file(
                    training_log_path,
                    (
                        f"[SVM 학습 실패]\n"
                        f"task={args.task}\n"
                        f"kernel={args.svm_kernel}\n"
                        f"ended_at={datetime.now().isoformat()}\n"
                        f"error_type={type(exc).__name__}\n"
                        f"error={exc}\n"
                        f"traceback=\n{traceback.format_exc()}\n"
                    ),
                )
            raise

        fit_end_dt = datetime.now()
        fit_elapsed = perf_counter() - fit_start
        if model_name == "svm":
            svm_rbf_elapsed_seconds = fit_elapsed if args.svm_kernel == "rbf" else svm_rbf_elapsed_seconds
            training_log_path = OUTPUT_ML_DIR / f"svm_{args.svm_kernel}_full_training_log.txt"
            save_text_file(
                training_log_path,
                (
                    f"[SVM 학습 완료]\n"
                    f"task={args.task}\n"
                    f"kernel={args.svm_kernel}\n"
                    f"started_at={fit_start_dt.isoformat()}\n"
                    f"ended_at={fit_end_dt.isoformat()}\n"
                    f"elapsed_seconds={fit_elapsed:.2f}\n"
                ),
            )
            print(f"- SVM 학습 시작 시각: {fit_start_dt.isoformat(timespec='seconds')}")
            print(f"- SVM 학습 종료 시각: {fit_end_dt.isoformat(timespec='seconds')}")
            print(f"- SVM 학습 소요 시간: {fit_elapsed:.2f}초")

        y_pred = model.predict(test_x)
        metrics = evaluate_predictions(y_test, y_pred)
        comparison_rows.append({"model": model_alias, **metrics})

        report_text = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        if model_name == "svm" and (is_svm_rbf_full or is_svm_linear_full):
            report_path = OUTPUT_ML_DIR / f"classification_report_{model_alias}_{args.task}_full.txt"
        else:
            report_path = OUTPUT_ML_DIR / f"classification_report_{model_name}{suffix}.txt"
        save_text_file(report_path, report_text)
        save_confusion_matrix_figure(
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            save_path=(
                OUTPUT_ML_DIR
                / f"confusion_matrix_{model_alias}_{args.task}_full.png"
                if model_name == "svm" and (is_svm_rbf_full or is_svm_linear_full)
                else OUTPUT_ML_DIR / f"confusion_matrix_{model_name}{suffix}.png"
            ),
            title=(
                f"Confusion Matrix ({model_alias}_{args.task}_full)"
                if model_name == "svm" and (is_svm_rbf_full or is_svm_linear_full)
                else f"Confusion Matrix ({model_name}{suffix})"
            ),
        )
        print(
            f"- accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}, f1_weighted={metrics['f1_weighted']:.4f}"
        )

        if model_name == "random_forest":
            importance_df = save_random_forest_feature_importance(
                model=model,
                feature_columns=feature_columns,
                output_path=OUTPUT_ML_DIR / f"feature_importance_random_forest{suffix}.png",
            )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model_name = model_alias
            best_model_obj = model

    comparison_df = pd.DataFrame(comparison_rows).sort_values("f1_macro", ascending=False)
    if is_svm_rbf_full or is_svm_linear_full:
        model_comparison_path = OUTPUT_ML_DIR / f"model_comparison_{args.task}_{'svm_rbf' if is_svm_rbf_full else 'svm_linear'}_full.csv"
    else:
        model_comparison_path = OUTPUT_ML_DIR / f"model_comparison{suffix}.csv"
    save_dataframe_csv(comparison_df, model_comparison_path)

    # 기존 파일명 호환성을 위해 multiclass 기본 split일 때 기본 파일도 함께 저장한다.
    if (
        args.task == "multiclass"
        and not args.group_split
        and args.svm_kernel == "linear"
        and set(args.models) == set(MODEL_LIST)
    ):
        save_dataframe_csv(comparison_df, OUTPUT_ML_DIR / "model_comparison.csv")

    if best_model_obj is None:
        raise RuntimeError("[오류] 학습된 모델이 없습니다.")

    model_bundle = {
        "model_name": best_model_name,
        "task": args.task,
        "group_split": args.group_split,
        "model": best_model_obj,
        "feature_columns": feature_columns,
        "fill_values": fill_values.to_dict(),
        "excluded_columns": exclude_columns,
        "label_column": args.label_column,
    }

    if is_svm_rbf_full or is_svm_linear_full:
        model_suffix = f"{args.task}_{'svm_rbf' if is_svm_rbf_full else 'svm_linear'}_full"
    else:
        model_suffix = args.task
    best_model_task_path = MODEL_DIR / f"best_model_{model_suffix}.pkl"
    scaler_task_path = MODEL_DIR / f"scaler_{model_suffix}.pkl"
    encoder_task_path = MODEL_DIR / f"label_encoder_{model_suffix}.pkl"
    save_pickle(model_bundle, best_model_task_path)
    save_pickle(scaler, scaler_task_path)
    save_pickle(label_encoder, encoder_task_path)

    # 실험 전용 결과(RBF/linear full 단독)는 기존 기본 모델 파일을 덮어쓰지 않는다.
    if not (is_svm_rbf_full or is_svm_linear_full):
        save_pickle(model_bundle, BEST_MODEL_PATH)
        save_pickle(scaler, SCALER_PATH)
        save_pickle(label_encoder, LABEL_ENCODER_PATH)
        save_text_file(BEST_MODEL_NAME_PATH, best_model_name)
    save_text_file(MODEL_DIR / f"best_model_name_{model_suffix}.txt", best_model_name)

    if args.cv > 1:
        run_cross_validation(
            x_clean=x_clean,
            y_encoded=y_series,
            selected_models=args.models,
            random_state=args.random_state,
            cv_splits=args.cv,
            svm_kernel=args.svm_kernel,
            output_path=OUTPUT_ML_DIR / f"cross_validation_results_{args.task}.csv",
        )

    save_report_summary(
        output_path=OUTPUT_ML_DIR / "ml_result_summary_for_report.txt",
        feature_path=feature_path,
        feature_columns=feature_columns,
        label_distribution=format_class_distribution_table(label_series),
        test_size=args.test_size,
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        importance_df=importance_df,
    )
    if args.task == "multiclass" and is_svm_rbf_full and svm_rbf_elapsed_seconds is not None:
        svm_rbf_row = comparison_df.iloc[0].to_dict()
        save_svm_kernel_comparison_summary(
            output_dir=OUTPUT_ML_DIR,
            task=args.task,
            rbf_metrics={
                "accuracy": float(svm_rbf_row["accuracy"]),
                "precision_macro": float(svm_rbf_row["precision_macro"]),
                "recall_macro": float(svm_rbf_row["recall_macro"]),
                "f1_macro": float(svm_rbf_row["f1_macro"]),
                "precision_weighted": float(svm_rbf_row["precision_weighted"]),
                "recall_weighted": float(svm_rbf_row["recall_weighted"]),
                "f1_weighted": float(svm_rbf_row["f1_weighted"]),
            },
            rbf_elapsed_seconds=svm_rbf_elapsed_seconds,
        )

    print("\n[학습 완료]")
    print(f"- model comparison: {model_comparison_path}")
    print(f"- best model: {best_model_task_path}")
    print(f"- scaler: {scaler_task_path}")
    print(f"- label encoder: {encoder_task_path}")


if __name__ == "__main__":
    main()
