"""진동 데이터셋 점검 및 FFT 분석 실행 진입점."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import (
    DATA_RAW_DIR,
    DEFAULT_BANDS,
    DEFAULT_FS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_TOP_K_FREQUENCIES,
    DEFAULT_VIBRATION_COLUMN,
    FEATURES_DIR,
    FFT_PLOTS_DIR,
    LOG_DIR,
    TIME_PLOTS_DIR,
    ensure_output_directories,
)
from data_loader import (
    extract_sample_rate_from_aihub_raw_csv,
    extract_vibration_signal,
    read_csv_file,
)
from dataset_inspector import inspect_vibration_folder, list_csv_files
from fft_analysis import compute_fft, find_dominant_frequency
from feature_extraction import (
    compute_band_energies,
    compute_peak,
    compute_rms,
    top_k_frequencies,
)
from preprocess import preprocess_signal
from visualize import save_fft_plot, save_time_plot


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="AI Hub 진동(vibration) 데이터 로드/전처리/FFT/시각화 도구"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="데이터셋 구조 확인 모드(파일 개수, 컬럼, 샘플 행 출력)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help=f"대상 폴더 경로(기본 예시: {DATA_RAW_DIR / 'vibration'})",
    )
    parser.add_argument("--file", type=str, default=None, help="단일 CSV 파일 경로")
    parser.add_argument(
        "--column",
        type=str,
        default=DEFAULT_VIBRATION_COLUMN,
        help="분석할 진동 컬럼명(예: acc_x)",
    )
    parser.add_argument("--fs", type=float, default=DEFAULT_FS, help="샘플링 주파수(Hz)")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="신호 앞부분 최대 샘플 수(None이면 전체 사용)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="전처리 시 z-score 정규화 적용",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="폴더 모드에서 처리할 최대 파일 개수",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="배치 처리 속도를 위해 그래프 저장을 생략",
    )
    parser.add_argument(
        "--feature-csv",
        type=str,
        default=str(FEATURES_DIR / "features.csv"),
        help="추출 특징 저장 CSV 경로",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K_FREQUENCIES,
        help="상위 주요 주파수 성분 개수",
    )
    parser.add_argument(
        "--auto-fs",
        action="store_true",
        help="파일 헤더의 Sample Rate를 우선 사용",
    )
    return parser.parse_args()


def is_vibration_path(path_value: str | Path) -> bool:
    """경로에 vibration 디렉터리가 포함되는지 확인한다."""
    path = Path(path_value)
    return "vibration" in {part.lower() for part in path.parts}


def build_output_name(csv_file: Path) -> str:
    """CSV 파일명을 기반으로 출력 이미지의 기본 이름을 만든다."""
    return csv_file.stem


def write_log(message: str) -> None:
    """실행 로그를 outputs/logs에 누적 기록한다."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "analysis_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as fp:
        fp.write(f"[{timestamp}] {message}\n")


def append_feature_row(feature_csv: Path, row: dict[str, object]) -> None:
    """특징값 1건을 CSV 파일에 누적 저장한다."""
    feature_csv.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if not feature_csv.exists():
        row_df.to_csv(feature_csv, index=False, encoding="utf-8-sig")
        return
    row_df.to_csv(feature_csv, mode="a", index=False, header=False, encoding="utf-8-sig")


def analyze_single_file(
    file_path: Path,
    column_name: str,
    fs: float,
    max_samples: int | None,
    normalize: bool,
    save_plots: bool,
    feature_csv: Path,
    top_k: int,
    auto_fs: bool,
) -> dict[str, object]:
    """단일 CSV 파일에 대해 로드->전처리->FFT->플롯 저장까지 수행한다."""
    df = read_csv_file(file_path)
    signal_series = extract_vibration_signal(df, file_path=file_path, column_name=column_name)
    actual_fs = fs
    if auto_fs:
        detected_fs = extract_sample_rate_from_aihub_raw_csv(file_path)
        if detected_fs and detected_fs > 0:
            actual_fs = detected_fs

    signal = preprocess_signal(
        signal_series,
        max_samples=max_samples,
        apply_normalization=normalize,
        normalization_method="zscore",
    )

    freqs, amplitudes = compute_fft(signal, actual_fs)
    dom_freq, dom_amp = find_dominant_frequency(freqs, amplitudes, ignore_dc=True)
    rms = compute_rms(signal)
    peak = compute_peak(signal)
    top_features = top_k_frequencies(freqs, amplitudes, k=top_k, ignore_dc=True)
    band_features = compute_band_energies(freqs, amplitudes, DEFAULT_BANDS)

    output_name = build_output_name(file_path)
    time_output = TIME_PLOTS_DIR / f"{output_name}_time.png"
    fft_output = FFT_PLOTS_DIR / f"{output_name}_fft.png"

    if save_plots:
        save_time_plot(
            signal=signal,
            fs=actual_fs,
            output_path=time_output,
            title=f"Time Domain - {output_name}",
        )
        save_fft_plot(
            freqs=freqs,
            amplitudes=amplitudes,
            output_path=fft_output,
            title=f"FFT Spectrum - {output_name}",
        )

    feature_row: dict[str, object] = {
        "file_path": str(file_path),
        "column": column_name,
        "fs_hz": actual_fs,
        "samples": int(signal.size),
        "rms": rms,
        "peak_abs": peak,
        "dominant_freq_hz": dom_freq,
        "dominant_amp": dom_amp,
    }
    feature_row.update(top_features)
    feature_row.update(band_features)
    append_feature_row(feature_csv, feature_row)

    print(f"\n[분석 완료] {file_path}")
    print(f"- 사용 컬럼: {column_name}")
    print(f"- 샘플링 주파수: {actual_fs} Hz")
    print(f"- 샘플 수: {signal.size}")
    print(f"- RMS: {rms:.6f}")
    print(f"- Peak(|x|): {peak:.6f}")
    print(f"- Dominant Frequency: {dom_freq:.4f} Hz")
    print(f"- Dominant Amplitude: {dom_amp:.6f}")
    if save_plots:
        print(f"- 시간영역 플롯: {time_output}")
        print(f"- FFT 플롯: {fft_output}")
    print(f"- 특징 저장 CSV: {feature_csv}")

    write_log(
        f"file={file_path}, column={column_name}, fs={actual_fs}, samples={signal.size}, "
        f"rms={rms:.6f}, peak={peak:.6f}, dominant_freq={dom_freq:.4f}, dominant_amp={dom_amp:.6f}"
    )

    return feature_row


def apply_limit(files: Iterable[Path], limit: int | None) -> list[Path]:
    """limit 옵션에 따라 처리 대상 파일 개수를 제한한다."""
    file_list = list(files)
    if limit is None:
        return file_list
    if limit <= 0:
        raise ValueError("[오류] --limit은 1 이상의 정수여야 합니다.")
    return file_list[:limit]


def main() -> None:
    """프로그램 메인 실행 함수."""
    args = parse_args()
    ensure_output_directories()

    if args.fs <= 0:
        raise ValueError("[오류] --fs는 0보다 커야 합니다.")
    if args.top_k <= 0:
        raise ValueError("[오류] --top-k는 1 이상의 정수여야 합니다.")

    feature_csv = Path(args.feature_csv)

    if args.inspect:
        target_folder = Path(args.folder) if args.folder else DATA_RAW_DIR / "vibration"
        if not is_vibration_path(target_folder):
            raise ValueError(
                "[오류] 이번 버전은 vibration 데이터만 지원합니다. "
                "폴더 경로에 'vibration'이 포함되어야 합니다."
            )
        inspect_vibration_folder(target_folder)
        return

    if not args.column:
        raise ValueError(
            "[오류] 분석할 컬럼명이 필요합니다. --column 옵션으로 진동 컬럼명을 입력하세요."
        )

    if args.file:
        file_path = Path(args.file)
        if not is_vibration_path(file_path):
            raise ValueError(
                "[오류] 이번 버전은 vibration 데이터만 지원합니다. "
                "파일 경로에 'vibration'이 포함되어야 합니다."
            )
        analyze_single_file(
            file_path=file_path,
            column_name=args.column,
            fs=args.fs,
            max_samples=args.max_samples,
            normalize=args.normalize,
            save_plots=not args.no_plot,
            feature_csv=feature_csv,
            top_k=args.top_k,
            auto_fs=args.auto_fs,
        )
        return

    if args.folder:
        folder_path = Path(args.folder)
        if not is_vibration_path(folder_path):
            raise ValueError(
                "[오류] 이번 버전은 vibration 데이터만 지원합니다. "
                "폴더 경로에 'vibration'이 포함되어야 합니다."
            )

        csv_files = list_csv_files(folder_path)
        csv_files = apply_limit(csv_files, args.limit)

        if not csv_files:
            print("[안내] 분석할 CSV 파일이 없습니다.")
            return

        print(f"[시작] 총 {len(csv_files)}개 파일 분석을 수행합니다.")
        success_count = 0
        fail_count = 0
        for idx, csv_file in enumerate(csv_files, start=1):
            print(f"\n[{idx}/{len(csv_files)}] 처리 중: {csv_file}")
            try:
                analyze_single_file(
                    file_path=csv_file,
                    column_name=args.column,
                    fs=args.fs,
                    max_samples=args.max_samples,
                    normalize=args.normalize,
                    save_plots=not args.no_plot,
                    feature_csv=feature_csv,
                    top_k=args.top_k,
                    auto_fs=args.auto_fs,
                )
                success_count += 1
            except Exception as exc:  # noqa: BLE001
                error_msg = f"[실패] {csv_file}: {exc}"
                print(error_msg)
                write_log(error_msg)
                fail_count += 1
        print(f"\n[완료] 성공: {success_count}개, 실패: {fail_count}개")
        print(f"[완료] 특징 CSV: {feature_csv}")
        return

    raise ValueError(
        "[오류] 실행 모드가 지정되지 않았습니다. --inspect 또는 --file/--folder 옵션을 사용하세요."
    )


if __name__ == "__main__":
    main()
