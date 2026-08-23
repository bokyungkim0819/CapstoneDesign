"""원시 진동 CSV에서 DWT 특징 CSV와 선택적 Wavelet 시각화를 생성한다."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

from config import DEFAULT_FS, FEATURES_DIR, PROJECT_ROOT, WAVELET_PLOTS_DIR
from data_loader import (
    extract_sample_rate_from_aihub_raw_csv,
    extract_vibration_signal,
    read_csv_file,
)
from dataset_inspector import list_csv_files
from ml_utils import normalize_label_from_path
from preprocess import preprocess_signal
from wavelet_analysis import (
    DEFAULT_DWT_LEVEL,
    DEFAULT_DWT_WAVELET,
    extract_wavelet_features,
    save_cwt_scalogram,
    save_wavelet_energy_plot,
)


DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "vibration"
DEFAULT_OUTPUT_CSV = FEATURES_DIR / "features_wavelet_labeled.csv"
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
        description="원시 진동 CSV -> DWT 특징 추출 및 CWT/DWT 시각화"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", type=str, default=None, help="단일 원시 CSV 경로")
    source.add_argument(
        "--folder",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="원시 vibration CSV 폴더",
    )
    parser.add_argument("--column", type=str, default="acc_x", help="진동 컬럼명")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS, help="기본 샘플링 주파수")
    parser.add_argument(
        "--auto-fs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CSV 헤더의 Sample Rate를 우선 사용",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4096,
        help="각 파일에서 사용할 고정 앞부분 길이",
    )
    parser.add_argument("--wavelet", type=str, default=DEFAULT_DWT_WAVELET)
    parser.add_argument("--level", type=int, default=DEFAULT_DWT_LEVEL)
    parser.add_argument("--limit", type=int, default=None, help="처리 파일 수 제한")
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=None,
        help="각 라벨에서 동일하게 추출할 최대 파일 수(빠른 균형 실험용)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="--per-class-limit 파일 선택 시 사용할 고정 시드",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Wavelet 특징 CSV 저장 경로",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 출력 CSV가 있으면 삭제 후 새로 생성",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="CWT scalogram과 DWT 대역 에너지 그림 저장",
    )
    parser.add_argument(
        "--plot-limit",
        type=int,
        default=10,
        help="폴더 처리 시 그림을 저장할 최대 파일 수",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="특징 CSV에 한 번에 기록할 행 수",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="특징 추출 병렬 프로세스 수(--save-plots 사용 시 1로 제한)",
    )
    return parser.parse_args()


def _read_aihub_signal_fast(
    file_path: Path,
    max_samples: int,
) -> tuple[pd.Series, float | None] | None:
    """AI Hub 메타정보+time,value 형식을 필요한 길이만 한 번 읽는다."""
    sample_rate: float | None = None
    is_aihub_format = False
    data_start_line: int | None = None

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith(("data label,", "motor spec,", "data length,")):
                is_aihub_format = True
            if lower.startswith("sample rate,"):
                is_aihub_format = True
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2:
                    try:
                        sample_rate = float(parts[1])
                    except ValueError:
                        sample_rate = None
                continue
            parts = [part.strip() for part in line.split(",")]
            if not is_aihub_format or len(parts) < 2:
                continue
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                continue
            data_start_line = line_index
            break

    if not is_aihub_format or data_start_line is None:
        return None

    values = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=data_start_line,
        max_rows=max_samples,
        usecols=1,
        encoding="utf-8",
    )
    values = np.atleast_1d(values).astype(float, copy=False)
    if values.size == 0:
        return None
    return pd.Series(values, name="vibration"), sample_rate


def extract_wavelet_row(
    file_path: Path,
    column: str,
    fallback_fs: float,
    auto_fs: bool,
    max_samples: int,
    wavelet: str,
    level: int,
) -> tuple[dict[str, object], np.ndarray, float]:
    """원시 CSV 한 건에서 메타데이터와 DWT 특징을 추출한다."""
    fast_result = _read_aihub_signal_fast(file_path, max_samples)
    if fast_result is not None:
        raw_signal, header_fs = fast_result
        detected_fs = header_fs if auto_fs else None
    else:
        df = read_csv_file(file_path)
        raw_signal = extract_vibration_signal(
            df,
            file_path=file_path,
            column_name=column,
        )
        detected_fs = (
            extract_sample_rate_from_aihub_raw_csv(file_path) if auto_fs else None
        )
    fs = float(detected_fs or fallback_fs)
    signal = preprocess_signal(
        raw_signal,
        max_samples=max_samples,
        apply_normalization=False,
    )
    features = extract_wavelet_features(signal, wavelet=wavelet, level=level)
    row: dict[str, object] = {
        "file_path": str(file_path),
        "label": normalize_label_from_path(str(file_path)),
        "column": column,
        "fs_hz": fs,
        "samples": int(signal.size),
        "wavelet_name": wavelet,
        "wavelet_level": level,
    }
    row.update(features)
    return row, signal, fs


def _write_chunk(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )
    rows.clear()


def _safe_plot_stem(file_path: Path, index: int) -> str:
    return f"{index:06d}_{file_path.stem[:80]}"


def _extract_worker(
    payload: tuple[Path, str, float, bool, int, str, int],
) -> tuple[Path, dict[str, object] | None, str | None]:
    """병렬 처리 프로세스에서 파일 한 건을 특징 행으로 변환한다."""
    file_path, column, fs, auto_fs, max_samples, wavelet, level = payload
    try:
        row, _, _ = extract_wavelet_row(
            file_path=file_path,
            column=column,
            fallback_fs=fs,
            auto_fs=auto_fs,
            max_samples=max_samples,
            wavelet=wavelet,
            level=level,
        )
        return file_path, row, None
    except Exception as exc:  # noqa: BLE001
        return file_path, None, str(exc)


def run(args: argparse.Namespace) -> tuple[int, int, Path]:
    if args.fs <= 0:
        raise ValueError("[오류] --fs는 0보다 커야 합니다.")
    if args.max_samples <= 0:
        raise ValueError("[오류] --max-samples는 1 이상이어야 합니다.")
    if args.level <= 0:
        raise ValueError("[오류] --level은 1 이상이어야 합니다.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("[오류] --limit은 1 이상이어야 합니다.")
    if args.per_class_limit is not None and args.per_class_limit <= 0:
        raise ValueError("[오류] --per-class-limit은 1 이상이어야 합니다.")
    if args.plot_limit < 0:
        raise ValueError("[오류] --plot-limit은 0 이상이어야 합니다.")
    if args.chunk_size <= 0:
        raise ValueError("[오류] --chunk-size는 1 이상이어야 합니다.")
    if args.workers <= 0:
        raise ValueError("[오류] --workers는 1 이상이어야 합니다.")

    output_path = Path(args.output)
    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(
                f"[오류] 출력 파일이 이미 있습니다: {output_path}. "
                "--overwrite를 지정하면 새로 생성합니다."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.file:
        files = [Path(args.file)]
    else:
        files = list_csv_files(Path(args.folder))
    if args.per_class_limit is not None:
        randomizer = random.Random(args.sample_seed)
        randomizer.shuffle(files)
        selected: list[Path] = []
        label_counts: dict[str, int] = {}
        for file_path in files:
            label = normalize_label_from_path(str(file_path))
            if label == "unknown":
                continue
            current_count = label_counts.get(label, 0)
            if current_count >= args.per_class_limit:
                continue
            selected.append(file_path)
            label_counts[label] = current_count + 1
        files = selected
        print(f"[라벨 균형 선택] {label_counts}")
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError("[오류] 처리할 원시 vibration CSV가 없습니다.")

    print(
        f"[Wavelet 특징 추출 시작] files={len(files)}, wavelet={args.wavelet}, "
        f"level={args.level}, samples={args.max_samples}"
    )
    rows: list[dict[str, object]] = []
    success_count = 0
    failure_count = 0

    if args.workers > 1 and not args.save_plots:
        payloads = [
            (
                file_path,
                args.column,
                args.fs,
                args.auto_fs,
                args.max_samples,
                args.wavelet,
                args.level,
            )
            for file_path in files
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(_extract_worker, payloads, chunksize=20)
            for index, (file_path, row, error) in enumerate(results, start=1):
                if error is not None or row is None:
                    failure_count += 1
                    print(f"[실패] {file_path}: {error}")
                else:
                    rows.append(row)
                    success_count += 1
                    if len(rows) >= args.chunk_size:
                        _write_chunk(rows, output_path)
                if index == 1 or index % 1000 == 0 or index == len(files):
                    print(
                        f"[{index}/{len(files)}] success={success_count}, "
                        f"failed={failure_count}"
                    )
        _write_chunk(rows, output_path)
        if success_count == 0:
            raise RuntimeError(
                f"[오류] Wavelet 특징을 추출한 파일이 없습니다. 실패={failure_count}"
            )
        print(
            f"[완료] success={success_count}, failed={failure_count}, "
            f"output={output_path}"
        )
        return success_count, failure_count, output_path

    if args.save_plots and args.workers > 1:
        print("[안내] 그림 저장 시 안정성을 위해 단일 프로세스로 실행합니다.")

    for index, file_path in enumerate(files, start=1):
        try:
            row, signal, fs = extract_wavelet_row(
                file_path=file_path,
                column=args.column,
                fallback_fs=args.fs,
                auto_fs=args.auto_fs,
                max_samples=args.max_samples,
                wavelet=args.wavelet,
                level=args.level,
            )
            rows.append(row)
            success_count += 1

            if args.save_plots and success_count <= args.plot_limit:
                stem = _safe_plot_stem(file_path, index)
                save_cwt_scalogram(
                    signal,
                    fs,
                    WAVELET_PLOTS_DIR / f"{stem}_scalogram.png",
                    title=(
                        "Wavelet Scalogram - "
                        f"{LABEL_EN.get(str(row['label']), str(row['label']))}"
                    ),
                )
                save_wavelet_energy_plot(
                    signal,
                    fs,
                    WAVELET_PLOTS_DIR / f"{stem}_dwt_energy.png",
                    wavelet=args.wavelet,
                    level=args.level,
                    title=(
                        "DWT Energy - "
                        f"{LABEL_EN.get(str(row['label']), str(row['label']))}"
                    ),
                )

            if len(rows) >= args.chunk_size:
                _write_chunk(rows, output_path)
            if index == 1 or index % 1000 == 0 or index == len(files):
                print(
                    f"[{index}/{len(files)}] success={success_count}, "
                    f"failed={failure_count}"
                )
        except Exception as exc:  # noqa: BLE001
            failure_count += 1
            print(f"[실패] {file_path}: {exc}")

    _write_chunk(rows, output_path)
    if success_count == 0:
        raise RuntimeError(
            f"[오류] Wavelet 특징을 추출한 파일이 없습니다. 실패={failure_count}"
        )
    print(
        f"[완료] success={success_count}, failed={failure_count}, "
        f"output={output_path}"
    )
    return success_count, failure_count, output_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
