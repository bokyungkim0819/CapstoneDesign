"""발표용 분포/대표 샘플 시각화를 생성한다."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import read_csv_file, extract_vibration_signal
from fft_analysis import compute_fft
from preprocess import preprocess_signal


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
FEATURE_CSV = BASE_DIR / "outputs" / "features" / "features_all.csv"
OUT_DIR = BASE_DIR / "outputs" / "presentation"
METRIC_COLUMNS = ["rms", "peak_abs", "dominant_freq_hz"]


def extract_label(file_path: str) -> str:
    """파일 경로에서 상태 라벨(예: 정상, 축정렬불량)을 추출한다."""
    match = re.search(r"vibration\\[^\\]+\\[^\\]+\\([^\\]+)\\", file_path)
    if match:
        return match.group(1)
    return "unknown"


def label_to_ascii(label: str) -> str:
    """플롯 가독성을 위해 라벨을 영문으로 변환한다."""
    mapping = {
        "정상": "normal",
        "축정렬불량": "misalignment",
        "회전체불평형": "unbalance",
        "베어링불량": "bearing_fault",
        "벨트느슨함": "belt_looseness",
    }
    return mapping.get(label, label.encode("ascii", errors="ignore").decode() or "unknown")


def choose_representative_rows(df: pd.DataFrame) -> pd.DataFrame:
    """라벨별 중앙값에 가장 가까운 대표 샘플 1건을 선택한다."""
    df = df.copy()
    df["label"] = df["file_path"].map(extract_label)

    labels = set(df["label"].unique())
    preferred_labels = ["정상", "축정렬불량", "베어링불량", "회전체불평형"]
    selected_labels: list[str] = [label for label in preferred_labels if label in labels]

    if len(selected_labels) < 4:
        for label in df["label"].value_counts().index.tolist():
            if label not in selected_labels:
                selected_labels.append(label)
            if len(selected_labels) >= 4:
                break

    selected_df = df[df["label"].isin(selected_labels)].copy()
    reps: list[pd.Series] = []

    for label in selected_labels:
        label_df = selected_df[selected_df["label"] == label].copy()
        if label_df.empty:
            continue

        medians = label_df[METRIC_COLUMNS].median()
        std = label_df[METRIC_COLUMNS].std(ddof=0).replace(0, 1.0)
        z = (label_df[METRIC_COLUMNS] - medians) / std
        label_df["distance"] = np.sqrt((z**2).sum(axis=1))
        reps.append(label_df.sort_values("distance", ascending=True).iloc[0])

    return pd.DataFrame(reps)


def plot_metric_distributions(df: pd.DataFrame) -> None:
    """RMS/Peak/Dominant Frequency 히스토그램과 박스플롯을 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dist_df = df.copy()
    dist_df["label"] = dist_df["file_path"].map(extract_label)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title in zip(
        axes,
        METRIC_COLUMNS,
        ["RMS Distribution", "Peak Distribution", "Dominant Frequency Distribution"],
    ):
        ax.hist(dist_df[col].dropna(), bins=60, alpha=0.85, color="#4C72B0")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_histograms.png", dpi=180)
    plt.close(fig)

    top_labels = dist_df["label"].value_counts().head(4).index.tolist()
    box_df = dist_df[dist_df["label"].isin(top_labels)].copy()
    tick_labels = [label_to_ascii(lbl) for lbl in top_labels]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, col, title in zip(
        axes,
        METRIC_COLUMNS,
        ["RMS by Label", "Peak by Label", "Dominant Frequency by Label"],
    ):
        groups = [box_df[box_df["label"] == lbl][col].dropna().values for lbl in top_labels]
        ax.boxplot(groups, tick_labels=tick_labels, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_boxplots_by_label.png", dpi=180)
    plt.close(fig)


def load_signal_from_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """대표 샘플 행에서 시간영역 신호와 FFT를 계산한다."""
    file_path = (SRC_DIR / str(row["file_path"])).resolve()
    df = read_csv_file(file_path)
    signal_series = extract_vibration_signal(df, file_path=file_path, column_name=str(row["column"]))
    signal = preprocess_signal(
        signal_series,
        max_samples=int(row["samples"]),
        apply_normalization=False,
        normalization_method="zscore",
    )
    freqs, amps = compute_fft(signal, float(row["fs_hz"]))
    return signal, freqs, amps


def plot_representative_comparison(rep_df: pd.DataFrame) -> None:
    """대표 샘플의 시간영역/FFT를 라벨별로 비교 저장한다."""
    if rep_df.empty:
        return

    rows = len(rep_df)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
    if rows == 1:
        axes = np.array([axes])

    for i, (_, row) in enumerate(rep_df.iterrows()):
        signal, freqs, amps = load_signal_from_row(row)
        fs = float(row["fs_hz"])
        t = np.arange(signal.size) / fs

        axes[i, 0].plot(t, signal, lw=0.9)
        label_name = label_to_ascii(str(row["label"]))
        axes[i, 0].set_title(f"{label_name} - Time Domain")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(alpha=0.2)

        axes[i, 1].plot(freqs, amps, lw=0.9)
        axes[i, 1].set_xlim(0, min(1200, float(np.max(freqs))))
        axes[i, 1].set_title(f"{label_name} - FFT Spectrum")
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "representative_time_fft_comparison.png", dpi=200)
    plt.close(fig)

    save_df = rep_df[["label", "file_path", "rms", "peak_abs", "dominant_freq_hz"]].copy()
    save_df["label_ascii"] = save_df["label"].map(label_to_ascii)
    save_df.to_csv(
        OUT_DIR / "representative_samples.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    """발표용 시각화 생성을 실행한다."""
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"features csv not found: {FEATURE_CSV}")

    df = pd.read_csv(FEATURE_CSV)
    plot_metric_distributions(df)
    rep_df = choose_representative_rows(df)
    plot_representative_comparison(rep_df)
    print(f"[완료] 발표용 시각화 생성: {OUT_DIR}")


if __name__ == "__main__":
    main()
