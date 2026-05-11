"""머신러닝 학습/평가/예측 공통 유틸리티 모음."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml_config import DEFAULT_EXCLUDE_COLUMNS


def ensure_directories(paths: Iterable[Path]) -> None:
    """필요한 디렉토리를 미리 생성한다."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_feature_csv(feature_csv_path: Path) -> pd.DataFrame:
    """특징 CSV를 로드하고 기본 유효성을 검사한다."""
    if not feature_csv_path.exists():
        raise FileNotFoundError(f"[오류] 특징 CSV 파일이 존재하지 않습니다: {feature_csv_path}")

    try:
        df = pd.read_csv(feature_csv_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"[오류] 특징 CSV 로드 중 오류가 발생했습니다: {exc}") from exc

    if df.empty:
        raise ValueError(f"[오류] 특징 CSV가 비어 있습니다: {feature_csv_path}")

    return df


def split_path_parts(path_value: str) -> list[str]:
    """파일 경로 문자열을 구분자(/, \\) 기준으로 분해해 유의미한 토큰만 반환한다."""
    parts = [token.strip() for token in re.split(r"[\\/]+", str(path_value)) if token.strip()]
    return [token for token in parts if token not in {".", ".."}]


def normalize_label_from_path(path_value: str) -> str:
    """file_path 문자열에서 상태 라벨을 규칙 기반으로 정규화한다."""
    text = str(path_value)
    compact = text.replace(" ", "")

    if "정상" in compact:
        return "정상"
    if "베어링불량" in compact:
        return "베어링불량"
    if "회전체불평형" in compact:
        return "회전체불평형"
    if "축정렬불량" in compact:
        return "축정렬불량"
    if "벨트느슨함" in compact:
        return "벨트느슨함"
    return "unknown"


def infer_equipment_id(path_value: str, group_level: int | None = None) -> str:
    """file_path에서 설비 ID를 추정한다.

    - group_level이 지정되면 해당 인덱스를 우선 사용한다.
    - 미지정이면 상태 라벨 직전 토큰을 설비 ID로 간주한다.
    """
    parts = split_path_parts(path_value)
    if not parts:
        return "unknown_group"

    if group_level is not None:
        if -len(parts) <= group_level < len(parts):
            return parts[group_level]
        return "unknown_group"

    # 라벨 직전 토큰을 설비 ID로 자동 추정한다.
    for idx, token in enumerate(parts):
        if normalize_label_from_path(token) != "unknown":
            if idx - 1 >= 0:
                return parts[idx - 1]

    # 마지막 fallback: 파일명 바로 상위 디렉토리
    if len(parts) >= 2:
        return parts[-2]
    return "unknown_group"


def detect_label_style(labels: pd.Series) -> str:
    """라벨이 한글 기반인지 영어 기반인지 추정한다."""
    values = labels.astype(str).dropna().tolist()
    if not values:
        return "unknown"
    has_korean = any(re.search(r"[가-힣]", value) for value in values)
    if has_korean:
        return "korean"
    return "english_or_other"


def validate_label_column(df: pd.DataFrame, label_column: str) -> None:
    """라벨 컬럼 존재 여부를 검증한다."""
    if label_column not in df.columns:
        available = ", ".join(df.columns.astype(str).tolist())
        raise KeyError(
            f"[오류] 라벨 컬럼 '{label_column}'을(를) 찾을 수 없습니다. "
            f"현재 CSV 컬럼: [{available}]"
        )


def get_effective_exclude_columns(
    df: pd.DataFrame,
    label_column: str,
    additional_exclude_columns: Iterable[str] | None = None,
) -> list[str]:
    """실제 데이터프레임에 존재하는 제외 대상 컬럼 목록을 계산한다."""
    candidates = list(DEFAULT_EXCLUDE_COLUMNS)
    if additional_exclude_columns:
        candidates.extend(additional_exclude_columns)
    candidates.append(label_column)

    # 순서를 유지하면서 중복 제거
    deduplicated: list[str] = []
    for col in candidates:
        if col not in deduplicated:
            deduplicated.append(col)

    return [col for col in deduplicated if col in df.columns]


def apply_binary_task(labels: pd.Series) -> pd.Series:
    """다중 라벨을 정상/고장 2진 분류 라벨로 변환한다."""
    normalized = labels.astype(str).str.strip()
    normal_tokens = {"정상", "normal"}
    return normalized.map(lambda x: "정상" if x in normal_tokens else "고장")


def select_numeric_features(
    df: pd.DataFrame, exclude_columns: Iterable[str]
) -> tuple[pd.DataFrame, list[str]]:
    """제외 컬럼을 뺀 뒤 숫자형 피처만 선택한다."""
    usable_df = df.drop(columns=[col for col in exclude_columns if col in df.columns], errors="ignore")
    numeric_df = usable_df.select_dtypes(include=[np.number]).copy()

    if numeric_df.empty:
        raise ValueError(
            "[오류] 학습 가능한 숫자형 피처가 없습니다. "
            "features.csv의 컬럼 타입과 제외 컬럼 설정을 확인하세요."
        )

    return numeric_df, numeric_df.columns.tolist()


def clean_feature_matrix(
    features: pd.DataFrame, fill_values: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """결측치/무한대 값을 처리하여 학습 가능한 행렬로 변환한다."""
    clean_df = features.copy()

    # inf/-inf를 NaN으로 통일한 뒤 컬럼 중앙값으로 대체한다.
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)

    if fill_values is None:
        computed_fill_values = clean_df.median(numeric_only=True)
    else:
        computed_fill_values = fill_values.copy()

    # 중앙값조차 NaN인 컬럼(전부 결측) 대응을 위해 0으로 2차 대체
    computed_fill_values = computed_fill_values.fillna(0.0)
    clean_df = clean_df.fillna(computed_fill_values)

    # 혹시 남은 NaN이 있으면 최종 방어적으로 0 처리
    clean_df = clean_df.fillna(0.0)

    return clean_df, computed_fill_values


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """라벨 문자열을 정수 인덱스로 인코딩한다."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.astype(str))
    return encoded, encoder


def save_pickle(obj: object, path: Path) -> None:
    """파이썬 객체를 pickle 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(path: Path) -> object:
    """pickle 파일을 로드한다."""
    if not path.exists():
        raise FileNotFoundError(f"[오류] 파일을 찾을 수 없습니다: {path}")
    with path.open("rb") as fp:
        return pickle.load(fp)


def save_text_file(path: Path, content: str) -> None:
    """텍스트 파일을 UTF-8로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    """데이터프레임을 CSV 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def print_class_distribution(labels: pd.Series) -> None:
    """클래스 분포를 콘솔에 출력한다."""
    counts = labels.astype(str).value_counts(dropna=False)
    print("\n[클래스 분포]")
    for cls, cnt in counts.items():
        print(f"- {cls}: {int(cnt)}개")


def format_class_distribution_table(labels: pd.Series) -> pd.DataFrame:
    """클래스 분포(개수/비율) 표를 생성한다."""
    counts = labels.astype(str).value_counts(dropna=False)
    total = int(counts.sum())
    distribution = pd.DataFrame(
        {
            "label": counts.index.astype(str),
            "count": counts.values.astype(int),
        }
    )
    distribution["ratio"] = distribution["count"] / total
    return distribution

