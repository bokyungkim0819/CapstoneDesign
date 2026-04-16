"""진동 신호 FFT 분석 관련 함수 모음."""

import numpy as np


def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """시간영역 신호의 단측(positive) 진폭 스펙트럼을 계산한다.

    Args:
        signal: 전처리 완료된 1차원 신호 배열
        fs: 샘플링 주파수(Hz)

    Returns:
        (frequency_vector, amplitude_spectrum)
    """
    if fs <= 0:
        raise ValueError("[오류] 샘플링 주파수(fs)는 0보다 커야 합니다.")
    if signal.ndim != 1:
        raise ValueError("[오류] FFT 입력 신호는 1차원 배열이어야 합니다.")
    if signal.size == 0:
        raise ValueError("[오류] FFT 입력 신호가 비어 있습니다.")

    n = signal.size
    fft_values = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    amplitudes = (2.0 / n) * np.abs(fft_values)
    amplitudes[0] = amplitudes[0] / 2.0

    if n % 2 == 0 and amplitudes.size > 1:
        amplitudes[-1] = amplitudes[-1] / 2.0

    return freqs, amplitudes


def find_dominant_frequency(
    freqs: np.ndarray, amplitudes: np.ndarray, ignore_dc: bool = True
) -> tuple[float, float]:
    """가장 큰 진폭을 가지는 지배 주파수 성분을 찾는다.

    Args:
        freqs: 주파수 벡터
        amplitudes: 진폭 스펙트럼
        ignore_dc: True이면 0Hz(DC) 성분 제외

    Returns:
        (dominant_frequency_hz, dominant_amplitude)
    """
    if freqs.size == 0 or amplitudes.size == 0:
        raise ValueError("[오류] 주파수 벡터 또는 진폭 스펙트럼이 비어 있습니다.")
    if freqs.size != amplitudes.size:
        raise ValueError("[오류] 주파수 벡터와 진폭 스펙트럼 길이가 다릅니다.")

    start_idx = 1 if ignore_dc and freqs.size > 1 else 0
    target_amplitudes = amplitudes[start_idx:]
    target_freqs = freqs[start_idx:]

    if target_amplitudes.size == 0:
        return 0.0, float(amplitudes[0])

    max_idx = int(np.argmax(target_amplitudes))
    return float(target_freqs[max_idx]), float(target_amplitudes[max_idx])
