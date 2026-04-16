"""시간영역/주파수영역 시각화 이미지를 저장하는 모듈."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_time_plot(
    signal: np.ndarray, fs: float, output_path: str | Path, title: str = "Time Domain Signal"
) -> Path:
    """시간영역 신호 그래프를 PNG 파일로 저장한다."""
    if signal.size == 0:
        raise ValueError("[오류] 시간영역 플롯 입력 신호가 비어 있습니다.")
    if fs <= 0:
        raise ValueError("[오류] 샘플링 주파수(fs)는 0보다 커야 합니다.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    time_axis = np.arange(signal.size, dtype=float) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, signal, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    return output


def save_fft_plot(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    output_path: str | Path,
    title: str = "FFT Amplitude Spectrum",
) -> Path:
    """FFT 진폭 스펙트럼 그래프를 PNG 파일로 저장한다."""
    if freqs.size == 0 or amplitudes.size == 0:
        raise ValueError("[오류] FFT 플롯 입력 데이터가 비어 있습니다.")
    if freqs.size != amplitudes.size:
        raise ValueError("[오류] 주파수 벡터와 진폭 벡터 길이가 다릅니다.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(freqs, amplitudes, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    return output
