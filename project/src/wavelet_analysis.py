"""진동 신호의 Wavelet 변환, 특징 추출, 발표용 시각화."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt


DEFAULT_DWT_WAVELET = "db4"
DEFAULT_DWT_LEVEL = 5
DEFAULT_CWT_WAVELET = "cmor1.5-1.0"


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    # pandas 2.x가 읽기 전용 NumPy view를 반환할 수 있으므로 PyWavelets에
    # 전달하기 전에 쓰기 가능한 C-contiguous 배열로 복사한다.
    values = np.array(signal, dtype=float, copy=True, order="C").reshape(-1)
    if values.size < 2:
        raise ValueError("[오류] Wavelet 변환에는 2개 이상의 신호 샘플이 필요합니다.")
    if not np.all(np.isfinite(values)):
        raise ValueError("[오류] Wavelet 입력 신호에 NaN 또는 무한대가 포함되어 있습니다.")
    return values


def compute_dwt_coefficients(
    signal: np.ndarray,
    wavelet: str = DEFAULT_DWT_WAVELET,
    level: int = DEFAULT_DWT_LEVEL,
) -> dict[str, np.ndarray]:
    """다단계 DWT 계수를 ``A{level}, D{level} ... D1`` 형태로 반환한다."""
    values = _validate_signal(signal)
    if level <= 0:
        raise ValueError("[오류] DWT level은 1 이상이어야 합니다.")

    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(values.size, wavelet_obj.dec_len)
    if level > max_level:
        raise ValueError(
            f"[오류] 신호 길이 {values.size}에서 {wavelet}의 최대 DWT level은 "
            f"{max_level}입니다. 요청 level={level}"
        )

    raw_coeffs = pywt.wavedec(values, wavelet_obj, level=level, mode="symmetric")
    named_coeffs: dict[str, np.ndarray] = {f"A{level}": raw_coeffs[0]}
    for detail_level, coeff in zip(range(level, 0, -1), raw_coeffs[1:], strict=True):
        named_coeffs[f"D{detail_level}"] = coeff
    return named_coeffs


def _coefficient_statistics(coefficients: np.ndarray) -> dict[str, float]:
    values = np.asarray(coefficients, dtype=float)
    energy = float(np.sum(np.square(values)))
    mean = float(np.mean(values))
    centered = values - mean
    std = float(np.std(values))

    if np.isclose(std, 0.0):
        skewness = 0.0
        kurtosis = 0.0
    else:
        standardized = centered / std
        skewness = float(np.mean(np.power(standardized, 3)))
        kurtosis = float(np.mean(np.power(standardized, 4)))

    squared = np.square(values)
    squared_sum = float(np.sum(squared))
    if np.isclose(squared_sum, 0.0):
        entropy = 0.0
    else:
        probabilities = squared / squared_sum
        nonzero = probabilities[probabilities > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    return {
        "energy": energy,
        "rms": float(np.sqrt(np.mean(squared))),
        "std": std,
        "max_abs": float(np.max(np.abs(values))),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "entropy": entropy,
    }


def extract_wavelet_features(
    signal: np.ndarray,
    wavelet: str = DEFAULT_DWT_WAVELET,
    level: int = DEFAULT_DWT_LEVEL,
) -> dict[str, float]:
    """DWT 각 대역의 에너지·통계·엔트로피 특징을 추출한다."""
    coeffs = compute_dwt_coefficients(signal, wavelet=wavelet, level=level)
    stats_by_band = {
        band_name: _coefficient_statistics(values)
        for band_name, values in coeffs.items()
    }
    total_energy = float(sum(stats["energy"] for stats in stats_by_band.values()))

    features: dict[str, float] = {
        "wavelet_total_energy": total_energy,
    }
    for band_name, stats in stats_by_band.items():
        prefix = f"wavelet_{band_name.lower()}"
        for stat_name, value in stats.items():
            features[f"{prefix}_{stat_name}"] = value
        features[f"{prefix}_energy_ratio"] = (
            stats["energy"] / total_energy if total_energy > 0 else 0.0
        )

    band_energy = np.asarray(
        [stats["energy"] for stats in stats_by_band.values()],
        dtype=float,
    )
    if total_energy > 0:
        band_probabilities = band_energy / total_energy
        nonzero = band_probabilities[band_probabilities > 0]
        features["wavelet_band_entropy"] = float(
            -np.sum(nonzero * np.log2(nonzero))
        )
    else:
        features["wavelet_band_entropy"] = 0.0
    return features


def dwt_frequency_ranges(fs: float, level: int) -> dict[str, tuple[float, float]]:
    """DWT 계수별 근사 주파수 범위를 반환한다."""
    if fs <= 0:
        raise ValueError("[오류] 샘플링 주파수(fs)는 0보다 커야 합니다.")
    ranges = {
        f"A{level}": (0.0, fs / (2 ** (level + 1))),
    }
    for detail_level in range(level, 0, -1):
        ranges[f"D{detail_level}"] = (
            fs / (2 ** (detail_level + 1)),
            fs / (2**detail_level),
        )
    return ranges


def save_wavelet_energy_plot(
    signal: np.ndarray,
    fs: float,
    output_path: str | Path,
    wavelet: str = DEFAULT_DWT_WAVELET,
    level: int = DEFAULT_DWT_LEVEL,
    title: str = "DWT Sub-band Energy",
) -> Path:
    """DWT 대역별 상대 에너지를 막대그래프로 저장한다."""
    coeffs = compute_dwt_coefficients(signal, wavelet=wavelet, level=level)
    ranges = dwt_frequency_ranges(fs, level)
    energies = np.asarray(
        [np.sum(np.square(values)) for values in coeffs.values()],
        dtype=float,
    )
    total = float(np.sum(energies))
    ratios = energies / total if total > 0 else np.zeros_like(energies)
    labels = [
        f"{name}\n{ranges[name][0]:.1f}-{ranges[name][1]:.1f} Hz"
        for name in coeffs
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, ratios * 100.0, color="#3569A8")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title(f"{title} ({wavelet}, level {level})")
    ax.set_xlabel("Wavelet sub-band")
    ax.set_ylabel("Relative energy (%)")
    ax.set_ylim(0, max(100.0, float(np.max(ratios) * 115.0)))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def save_cwt_scalogram(
    signal: np.ndarray,
    fs: float,
    output_path: str | Path,
    title: str = "Wavelet Scalogram",
    wavelet: str = DEFAULT_CWT_WAVELET,
    min_frequency_hz: float = 1.0,
    max_frequency_hz: float | None = None,
    frequency_bins: int = 96,
) -> Path:
    """CWT 시간-주파수 파워 scalogram을 저장한다."""
    values = _validate_signal(signal)
    if fs <= 0:
        raise ValueError("[오류] 샘플링 주파수(fs)는 0보다 커야 합니다.")
    if frequency_bins < 2:
        raise ValueError("[오류] frequency_bins는 2 이상이어야 합니다.")

    nyquist = fs / 2.0
    upper = min(max_frequency_hz or nyquist, nyquist)
    lower = max(float(min_frequency_hz), np.finfo(float).eps)
    if upper <= lower:
        raise ValueError("[오류] CWT 최대 주파수는 최소 주파수보다 커야 합니다.")

    target_frequencies = np.geomspace(lower, upper, frequency_bins)
    center_frequency = pywt.central_frequency(wavelet)
    scales = center_frequency * fs / target_frequencies
    coefficients, frequencies = pywt.cwt(
        values,
        scales,
        wavelet,
        sampling_period=1.0 / fs,
    )
    power_db = 10.0 * np.log10(np.square(np.abs(coefficients)) + 1e-12)
    time_axis = np.arange(values.size, dtype=float) / fs

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    mesh = ax.pcolormesh(
        time_axis,
        frequencies,
        power_db,
        shading="auto",
        cmap="viridis",
    )
    ax.set_yscale("log")
    ax.set_ylim(lower, upper)
    ax.set_title(f"{title} ({wavelet})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Power (dB)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output
