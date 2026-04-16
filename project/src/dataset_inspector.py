"""AI Hub 진동 데이터셋 구조를 빠르게 확인하기 위한 유틸리티."""

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


def _read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    """구분자/인코딩이 다른 CSV를 점검 모드에서도 최대한 유연하게 읽는다."""
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
            df = pd.read_csv(file_path, **options)
            if df.shape[1] >= 1:
                return df
        except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
            last_error = exc

    if last_error is None:
        raise ValueError(f"[오류] CSV를 읽을 수 없습니다: {file_path}")
    raise ValueError(f"[오류] CSV 파싱에 실패했습니다: {file_path}") from last_error


def list_csv_files(folder_path: str | Path) -> list[Path]:
    """지정한 폴더 아래의 CSV 파일 목록을 수집한다.

    Args:
        folder_path: 탐색할 대상 폴더 경로

    Returns:
        CSV 파일 Path 리스트

    Raises:
        FileNotFoundError: 폴더가 존재하지 않는 경우
        NotADirectoryError: 폴더 경로가 디렉터리가 아닌 경우
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"[오류] 폴더를 찾을 수 없습니다: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"[오류] 디렉터리가 아닙니다: {folder}")

    csv_files = sorted(folder.rglob("*.csv"))
    return csv_files


def inspect_sample_csv(sample_file: str | Path, head_rows: int = 5) -> dict[str, object]:
    """샘플 CSV 파일의 기본 정보를 읽어 반환한다.

    Args:
        sample_file: 샘플 CSV 파일 경로
        head_rows: 상위 몇 행을 확인할지 지정

    Returns:
        파일 정보(컬럼명, shape, head 텍스트 포함)

    Raises:
        FileNotFoundError: 파일이 없는 경우
        ValueError: 빈 CSV이거나 파싱 실패한 경우
    """
    file_path = Path(sample_file)
    if not file_path.exists():
        raise FileNotFoundError(f"[오류] 샘플 파일을 찾을 수 없습니다: {file_path}")

    df = _read_csv_with_fallback(file_path)

    if df.empty:
        raise ValueError(f"[오류] 데이터 행이 없는 CSV 파일입니다: {file_path}")

    return {
        "path": str(file_path),
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": df.head(head_rows).to_string(index=False),
    }


def inspect_vibration_folder(folder_path: str | Path, head_rows: int = 5) -> list[Path]:
    """진동 데이터 폴더의 구조를 출력하고 샘플 파일 정보를 보여준다.

    Args:
        folder_path: vibration CSV 파일들이 저장된 폴더 경로
        head_rows: 샘플 CSV에서 출력할 상위 행 수

    Returns:
        발견된 CSV 파일 목록
    """
    csv_files = list_csv_files(folder_path)

    print("\n=== 진동(vibration) 폴더 점검 결과 ===")
    print(f"- 대상 폴더: {Path(folder_path).resolve()}")
    print(f"- CSV 파일 개수: {len(csv_files)}")

    if not csv_files:
        print("[안내] CSV 파일이 없습니다. 파일 추가 후 다시 실행하세요.")
        return csv_files

    print("\n[파일 목록 일부 미리보기]")
    preview_count = min(10, len(csv_files))
    for idx, file_path in enumerate(csv_files[:preview_count], start=1):
        print(f"{idx:02d}. {file_path}")
    if len(csv_files) > preview_count:
        print(f"... (총 {len(csv_files)}개 중 {preview_count}개만 표시)")

    sample_info: dict[str, object] | None = None
    sample_error: Exception | None = None
    for sample_file in csv_files:
        try:
            sample_info = inspect_sample_csv(sample_file, head_rows=head_rows)
            break
        except Exception as exc:  # noqa: BLE001
            sample_error = exc
            continue

    if sample_info is None:
        raise ValueError(
            "[오류] 샘플 파일 점검에 실패했습니다. CSV 포맷/인코딩을 확인하세요."
        ) from sample_error

    print("\n[샘플 파일 정보]")
    print(f"- 파일: {sample_info['path']}")
    print(f"- shape: {sample_info['shape']}")
    print(f"- 컬럼명: {sample_info['columns']}")
    print(f"- 상위 {head_rows}행:\n{sample_info['head']}")

    return csv_files
