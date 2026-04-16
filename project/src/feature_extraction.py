"""FFT 결과로부터 특징값을 추출하는 모듈."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_rms(signal: np.ndarray) -> float:
    """시간영역 신호 RMS를 계산한다."""
    if signal.size == 0:
        raise ValueError("[오류] RMS 계산 입력 신호가 비어 있습니다.")
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_peak(signal: np.ndarray) -> float:
    """시간영역 신호 절대 최대 피크를 계산한다."""
    if signal.size == 0:
        raise ValueError("[오류] 피크 계산 입력 신호가 비어 있습니다.")
    return float(np.max(np.abs(signal)))


def compute_band_energies(
    freqs: np.ndarray, amplitudes: np.ndarray, bands: Iterable[tuple[float, float]]
) -> dict[str, float]:
    """지정한 주파수 대역별 에너지를 계산한다."""
    if freqs.size != amplitudes.size:
        raise ValueError("[오류] 주파수 벡터와 진폭 스펙트럼 길이가 다릅니다.")

    power = np.square(amplitudes)
    features: dict[str, float] = {}
    for low, high in bands:
        if low < 0 or high <= low:
            raise ValueError(f"[오류] 잘못된 대역 범위입니다: ({low}, {high})")
        mask = (freqs >= low) & (freqs < high)
        key = f"energy_{int(low)}_{int(high)}Hz"
        features[key] = float(np.sum(power[mask])) if np.any(mask) else 0.0
    return features


def top_k_frequencies(
    freqs: np.ndarray, amplitudes: np.ndarray, k: int = 3, ignore_dc: bool = True
) -> dict[str, float]:
    """진폭 기준 상위 k개 주파수/진폭을 추출한다."""
    if k <= 0:
        raise ValueError("[오류] top-k 개수는 1 이상이어야 합니다.")
    if freqs.size != amplitudes.size:
        raise ValueError("[오류] 주파수 벡터와 진폭 스펙트럼 길이가 다릅니다.")

    start_idx = 1 if ignore_dc and freqs.size > 1 else 0
    cand_freqs = freqs[start_idx:]
    cand_amps = amplitudes[start_idx:]
    if cand_amps.size == 0:
        return {}

    order = np.argsort(cand_amps)[::-1]
    top_idx = order[:k]

    result: dict[str, float] = {}
    for rank, idx in enumerate(top_idx, start=1):
        result[f"top{rank}_freq_hz"] = float(cand_freqs[idx])
        result[f"top{rank}_amp"] = float(cand_amps[idx])
    return result
