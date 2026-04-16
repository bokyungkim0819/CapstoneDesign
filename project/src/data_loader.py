"""CSV 진동 데이터 로딩과 컬럼 추출을 담당하는 모듈."""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.errors import EmptyDataError, ParserError


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """구분자/인코딩 차이가 있는 CSV를 여러 방식으로 시도해 읽는다."""
    read_options: list[dict[str, object]] = [
        {},
        {"sep": None, "engine": "python"},
        {"encoding": "cp949"},
        {"encoding": "cp949", "sep": None, "engine": "python"},
        {"sep": None, "engine": "python", "on_bad_lines": "skip"},
        {"encoding": "cp949", "sep": None, "engine": "python", "on_bad_lines": "skip"},
    ]

    last_error: Exception | None = None
    for options in read_options:
        try:
            df = pd.read_csv(path, **options)
            if df.shape[1] >= 1:
                return df
        except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
            last_error = exc

    if last_error is None:
        raise ValueError(f"[오류] CSV를 읽을 수 없습니다: {path}")
    raise ValueError(f"[오류] CSV 파싱에 실패했습니다: {path}") from last_error


def read_csv_file(file_path: str | Path) -> pd.DataFrame:
    """CSV 파일을 읽어 DataFrame으로 반환한다.

    Args:
        file_path: 읽을 CSV 파일 경로

    Returns:
        읽어온 DataFrame

    Raises:
        FileNotFoundError: 파일이 없는 경우
        ValueError: 빈 파일, 파싱 실패, 데이터 없음 등의 경우
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"[오류] CSV 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"[오류] CSV 파일만 지원합니다: {path}")

    df = _read_csv_with_fallback(path)

    if df.shape[0] == 0:
        raise ValueError(f"[오류] 데이터 행이 없는 CSV 파일입니다: {path}")

    return df


def extract_numeric_series_from_aihub_raw_csv(file_path: str | Path) -> pd.Series:
    """AI Hub 원시 진동 CSV(메타정보 + time,value 행)에서 진동값을 추출한다."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"[오류] CSV 파일을 찾을 수 없습니다: {path}")

    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue

            try:
                float(parts[0])
                value = float(parts[1])
            except ValueError:
                continue

            values.append(value)

    if not values:
        raise ValueError(
            f"[오류] 파일에서 진동 시계열을 찾지 못했습니다: {path}. "
            "원시 포맷(시간,진동값) 데이터가 있는지 확인하세요."
        )

    return pd.Series(values, name="vibration")


def extract_sample_rate_from_aihub_raw_csv(file_path: str | Path) -> float | None:
    """AI Hub 원시 CSV 헤더의 Sample Rate 값을 추출한다."""
    path = Path(file_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            if not line.lower().startswith("sample rate"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                return float(parts[1])
            except ValueError:
                return None

    return None


def extract_numeric_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """사용자가 지정한 컬럼을 숫자형 시계열로 추출한다.

    Args:
        df: 원본 DataFrame
        column_name: 추출할 컬럼명

    Returns:
        float 타입 Series

    Raises:
        ValueError: 컬럼이 없거나 숫자형으로 해석할 수 없는 경우
    """
    if not column_name:
        raise ValueError("[오류] 컬럼명이 비어 있습니다. --column 옵션을 확인하세요.")
    if column_name not in df.columns:
        raise ValueError(
            f"[오류] 지정한 컬럼이 없습니다: '{column_name}'. "
            f"사용 가능한 컬럼: {df.columns.tolist()}"
        )

    series = df[column_name]

    if is_numeric_dtype(series):
        return series.astype(float)

    converted = pd.to_numeric(series, errors="coerce")
    invalid_count = int(converted.isna().sum() - series.isna().sum())
    if invalid_count > 0:
        raise ValueError(
            f"[오류] 컬럼 '{column_name}'은 숫자형이 아닙니다. "
            f"숫자로 변환할 수 없는 값 {invalid_count}개가 포함되어 있습니다."
        )

    return converted.astype(float)


def extract_vibration_signal(
    df: pd.DataFrame, file_path: str | Path, column_name: str
) -> pd.Series:
    """사용자 지정 컬럼 우선, 실패 시 AI Hub 원시 포맷 추출을 시도한다."""
    try:
        return extract_numeric_column(df, column_name)
    except ValueError as exc:
        msg = str(exc)
        if "지정한 컬럼이 없습니다" not in msg:
            raise
        return extract_numeric_series_from_aihub_raw_csv(file_path)
