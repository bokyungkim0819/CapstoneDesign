"""진동 신호 전처리(결측치 제거, DC 제거, 정규화, 길이 제한) 모듈."""

import numpy as np
import pandas as pd


def drop_missing(signal: pd.Series) -> pd.Series:
    """결측치(NaN)를 제거한다."""
    cleaned = signal.dropna()
    if cleaned.empty:
        raise ValueError("[오류] 결측치 제거 후 데이터가 비어 있습니다.")
    return cleaned


def remove_dc_offset(signal: pd.Series) -> pd.Series:
    """평균값(DC offset)을 제거하여 중심을 0으로 맞춘다."""
    mean_val = float(signal.mean())
    return signal - mean_val


def normalize_signal(signal: pd.Series, method: str = "zscore") -> pd.Series:
    """선택적으로 신호를 정규화한다.

    Args:
        signal: 입력 신호
        method: "zscore" 또는 "minmax"
    """
    if method == "zscore":
        std_val = float(signal.std(ddof=0))
        if np.isclose(std_val, 0.0):
            raise ValueError("[오류] 표준편차가 0이라 z-score 정규화를 수행할 수 없습니다.")
        return (signal - float(signal.mean())) / std_val

    if method == "minmax":
        min_val = float(signal.min())
        max_val = float(signal.max())
        if np.isclose(max_val - min_val, 0.0):
            raise ValueError("[오류] 값 범위가 0이라 min-max 정규화를 수행할 수 없습니다.")
        return (signal - min_val) / (max_val - min_val)

    raise ValueError(f"[오류] 지원하지 않는 정규화 방법입니다: {method}")


def trim_signal(signal: pd.Series, max_samples: int | None = None) -> pd.Series:
    """신호 길이를 앞부분 기준으로 제한한다."""
    if max_samples is None:
        return signal
    if max_samples <= 0:
        raise ValueError("[오류] max_samples는 1 이상의 정수여야 합니다.")
    return signal.iloc[:max_samples]


def preprocess_signal(
    signal: pd.Series,
    max_samples: int | None = None,
    apply_normalization: bool = False,
    normalization_method: str = "zscore",
) -> np.ndarray:
    """전처리 파이프라인을 순차 적용해 numpy 배열로 반환한다."""
    processed = drop_missing(signal)
    processed = trim_signal(processed, max_samples=max_samples)
    processed = remove_dc_offset(processed)

    if apply_normalization:
        processed = normalize_signal(processed, method=normalization_method)

    if processed.empty:
        raise ValueError("[오류] 전처리 후 신호가 비어 있습니다.")

    return processed.to_numpy(dtype=float)
