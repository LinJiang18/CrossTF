"""Inverse of FIDE's high-frequency boost (not part of upstream FIDE).

Upstream's preprocessing multiplies the top `percentage_of_freq_enhanced` % of rFFT
frequencies by `c` before training. Generated samples live in that boosted space; this
undoes the boost with the same frequency indices as the forward transform.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq


def invert_fft_enhancement(
    x_time: np.ndarray,
    seq_len: int,
    c: float,
    percentage_of_freq_enhanced: int,
) -> np.ndarray:
    """Undo high-frequency boost applied before training (same indices as forward)."""
    x_time = np.asarray(x_time, dtype=np.float64)
    X = rfft(x_time, axis=1)
    freq = rfftfreq(seq_len)
    top_n = max(1, int((X.shape[1] * percentage_of_freq_enhanced) / 100))
    top_indices = np.argsort(freq)[-top_n:]
    X_un = X.copy()
    X_un[:, top_indices] /= c
    return irfft(X_un, n=seq_len, axis=1).astype(np.float32)
