"""정상/불량 대표 진동의 Wavelet 발표용 시각화를 생성한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt

from config import PROJECT_ROOT
from ml_utils import normalize_label_from_path
from sample_demo import find_sample_files
from wavelet_analysis import (
    DEFAULT_CWT_WAVELET,
    DEFAULT_DWT_LEVEL,
    DEFAULT_DWT_WAVELET,
    save_cwt_scalogram,
    save_wavelet_energy_plot,
)
from wavelet_pipeline import extract_wavelet_row


OUT_DIR = PROJECT_ROOT / "outputs" / "presentation" / "wavelet"
LABEL_EN = {
    "정상": "Normal",
    "고장": "Fault",
    "축정렬불량": "Misalignment",
    "베어링불량": "Bearing fault",
    "회전체불평형": "Unbalance",
    "벨트느슨함": "Belt looseness",
}


def to_en(label: str) -> str:
    return LABEL_EN.get(str(label).strip(), str(label))


def save_comparison_scalogram(
    samples: list[dict[str, object]],
    output_path: Path,
    wavelet: str = DEFAULT_CWT_WAVELET,
) -> None:
    """정상/불량 CWT scalogram을 동일한 색 범위로 비교 저장한다."""
    transformed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    global_min = np.inf
    global_max = -np.inf

    for item in samples:
        signal = np.asarray(item["signal"], dtype=float)
        fs = float(item["fs"])
        target_frequencies = np.geomspace(1.0, fs / 2.0, 96)
        scales = pywt.central_frequency(wavelet) * fs / target_frequencies
        coefficients, frequencies = pywt.cwt(
            signal,
            scales,
            wavelet,
            sampling_period=1.0 / fs,
        )
        power_db = 10.0 * np.log10(np.square(np.abs(coefficients)) + 1e-12)
        time_axis = np.arange(signal.size, dtype=float) / fs
        transformed.append((time_axis, frequencies, power_db))
        global_min = min(global_min, float(np.percentile(power_db, 2)))
        global_max = max(global_max, float(np.percentile(power_db, 98)))

    fig, axes = plt.subplots(len(samples), 1, figsize=(12, 4.5 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    mesh = None
    for ax, item, (time_axis, frequencies, power_db) in zip(
        axes, samples, transformed, strict=True
    ):
        mesh = ax.pcolormesh(
            time_axis,
            frequencies,
            power_db,
            shading="auto",
            cmap="viridis",
            vmin=global_min,
            vmax=global_max,
        )
        ax.set_yscale("log")
        ax.set_ylim(1.0, float(item["fs"]) / 2.0)
        ax.set_title(str(item["title"]), fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    if mesh is not None:
        colorbar_axis = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        colorbar = fig.colorbar(mesh, cax=colorbar_axis)
        colorbar.set_label("Power (dB)")
    fig.suptitle(
        "CWT Time-Frequency Comparison: Normal vs Fault",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, right=0.86, hspace=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    normal_file, fault_file = find_sample_files(None, None)
    source_samples = [
        ("normal", "Normal sample", normal_file),
        ("fault", "Fault sample", fault_file),
    ]
    comparison_samples: list[dict[str, object]] = []

    for tag, sample_name, file_path in source_samples:
        row, signal, fs = extract_wavelet_row(
            file_path=file_path,
            column="acc_x",
            fallback_fs=1000.0,
            auto_fs=True,
            max_samples=4096,
            wavelet=DEFAULT_DWT_WAVELET,
            level=DEFAULT_DWT_LEVEL,
        )
        actual = to_en(normalize_label_from_path(str(file_path)))
        title = f"{sample_name} ({actual})"

        scalogram_path = OUT_DIR / f"{tag}_wavelet_scalogram.png"
        energy_path = OUT_DIR / f"{tag}_wavelet_energy.png"
        save_cwt_scalogram(
            signal,
            fs,
            scalogram_path,
            title=f"CWT Scalogram - {title}",
        )
        save_wavelet_energy_plot(
            signal,
            fs,
            energy_path,
            wavelet=DEFAULT_DWT_WAVELET,
            level=DEFAULT_DWT_LEVEL,
            title=f"DWT Sub-band Energy - {title}",
        )
        comparison_samples.append(
            {"signal": signal, "fs": fs, "title": title, "features": row}
        )
        print(f"[{sample_name}]")
        print(f"- Scalogram: {scalogram_path}")
        print(f"- DWT energy: {energy_path}")

    comparison_path = OUT_DIR / "normal_fault_wavelet_comparison.png"
    save_comparison_scalogram(comparison_samples, comparison_path)
    print(f"[완료] Comparison: {comparison_path}")


if __name__ == "__main__":
    main()
