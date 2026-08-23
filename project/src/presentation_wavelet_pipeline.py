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
    compute_dwt_coefficients,
    compute_global_wavelet_spectrum,
    dwt_frequency_ranges,
    reconstruct_dwt_subbands,
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


def save_global_spectrum_comparison(
    samples: list[dict[str, object]],
    output_path: Path,
    wavelet: str = DEFAULT_CWT_WAVELET,
) -> None:
    """FFT spectrum에 대응하는 정상/불량 Global Wavelet Spectrum을 저장한다."""
    spectra: list[tuple[np.ndarray, np.ndarray]] = []
    reference_power = 0.0
    for item in samples:
        frequencies, mean_power = compute_global_wavelet_spectrum(
            np.asarray(item["signal"], dtype=float),
            float(item["fs"]),
            wavelet=wavelet,
        )
        spectra.append((frequencies, mean_power))
        reference_power = max(reference_power, float(np.max(mean_power)))

    reference_power = max(reference_power, np.finfo(float).tiny)
    colors = ["#2E7D32", "#C62828", "#3569A8", "#7B1FA2"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for color, item, (frequencies, mean_power) in zip(
        colors,
        samples,
        spectra,
        strict=False,
    ):
        relative_power_db = 10.0 * np.log10(
            mean_power / reference_power + 1e-12
        )
        ax.semilogx(
            frequencies,
            relative_power_db,
            linewidth=2.0,
            color=color,
            label=str(item["title"]),
        )

    ax.set_title(
        "Global Wavelet Spectrum: Normal vs Fault",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time-averaged CWT power (dB, shared reference)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dwt_energy_comparison(
    samples: list[dict[str, object]],
    output_path: Path,
    wavelet: str = DEFAULT_DWT_WAVELET,
    level: int = DEFAULT_DWT_LEVEL,
) -> None:
    """정상/불량 DWT 대역별 상대 에너지를 그룹 막대그래프로 저장한다."""
    if not samples:
        raise ValueError("[오류] DWT 에너지 비교용 샘플이 없습니다.")

    first_fs = float(samples[0]["fs"])
    ranges = dwt_frequency_ranges(first_fs, level)
    band_names = list(
        compute_dwt_coefficients(
            np.asarray(samples[0]["signal"], dtype=float),
            wavelet=wavelet,
            level=level,
        )
    )
    labels = [
        f"{name}\n{ranges[name][0]:.0f}-{ranges[name][1]:.0f} Hz"
        for name in band_names
    ]
    x_positions = np.arange(len(band_names), dtype=float)
    bar_width = 0.36
    colors = ["#2E7D32", "#C62828", "#3569A8", "#7B1FA2"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for sample_index, (color, item) in enumerate(
        zip(colors, samples, strict=False)
    ):
        coeffs = compute_dwt_coefficients(
            np.asarray(item["signal"], dtype=float),
            wavelet=wavelet,
            level=level,
        )
        energies = np.asarray(
            [np.sum(np.square(coeffs[name])) for name in band_names],
            dtype=float,
        )
        ratios = energies / np.sum(energies) * 100.0
        offset = (sample_index - (len(samples) - 1) / 2.0) * bar_width
        bars = ax.bar(
            x_positions + offset,
            ratios,
            width=bar_width,
            color=color,
            label=str(item["title"]),
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    ax.set_title(
        "DWT Sub-band Energy Comparison: Normal vs Fault",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x_positions, labels)
    ax.set_xlabel("Wavelet sub-band")
    ax.set_ylabel("Relative energy (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dwt_decomposition_plot(
    signal: np.ndarray,
    fs: float,
    output_path: Path,
    title: str,
    wavelet: str = DEFAULT_DWT_WAVELET,
    level: int = DEFAULT_DWT_LEVEL,
) -> None:
    """원신호와 A5/D5~D1 복원 성분을 동일 시간축의 적층 그래프로 저장한다."""
    values = np.asarray(signal, dtype=float)
    components = reconstruct_dwt_subbands(
        values,
        wavelet=wavelet,
        level=level,
    )
    ranges = dwt_frequency_ranges(fs, level)
    time_axis = np.arange(values.size, dtype=float) / fs
    rows = 1 + len(components)
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(12, 1.55 * rows),
        sharex=True,
    )

    axes[0].plot(time_axis, values, color="#333333", linewidth=0.7)
    axes[0].set_ylabel("Signal")
    axes[0].grid(alpha=0.2)
    for ax, (band_name, component) in zip(
        axes[1:],
        components.items(),
        strict=True,
    ):
        low, high = ranges[band_name]
        ax.plot(time_axis, component, linewidth=0.7, color="#3569A8")
        ax.set_ylabel(f"{band_name}\n{low:.0f}-{high:.0f} Hz")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"DWT Reconstructed Sub-band Signals - {title}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
        decomposition_path = OUT_DIR / f"{tag}_dwt_decomposition.png"
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
        save_dwt_decomposition_plot(
            signal,
            fs,
            decomposition_path,
            title=title,
        )
        comparison_samples.append(
            {"signal": signal, "fs": fs, "title": title, "features": row}
        )
        print(f"[{sample_name}]")
        print(f"- Scalogram: {scalogram_path}")
        print(f"- DWT energy: {energy_path}")
        print(f"- DWT decomposition: {decomposition_path}")

    comparison_path = OUT_DIR / "normal_fault_wavelet_comparison.png"
    save_comparison_scalogram(comparison_samples, comparison_path)
    spectrum_path = OUT_DIR / "normal_fault_global_wavelet_spectrum.png"
    save_global_spectrum_comparison(comparison_samples, spectrum_path)
    energy_comparison_path = OUT_DIR / "normal_fault_dwt_energy_comparison.png"
    save_dwt_energy_comparison(comparison_samples, energy_comparison_path)
    print(f"[완료] Scalogram comparison: {comparison_path}")
    print(f"[완료] Global spectrum: {spectrum_path}")
    print(f"[완료] Energy comparison: {energy_comparison_path}")


if __name__ == "__main__":
    main()
