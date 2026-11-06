"""1/3-octave analysis grid and the synthesis filter bank.

The whole engine works on 1/3-octave bands because that is the resolution at
which the underlying physics is specified (material absorption, transmission
loss, atmospheric absorption, diffraction) and at which building-acoustics
standards are computed.  A *partition of unity* in log-frequency turns the band
description into an exact, artifact-free full-band impulse response: the band
windows sum to 1 at every frequency, so no comb ripple or "banding" is audible.
"""

from __future__ import annotations

import numpy as np

# 1/3-octave nominal centre frequencies, 25 Hz .. 20 kHz (IEC 61260)
BAND_CENTRES = np.array([
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0,
    315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0,
    2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
    16000.0, 20000.0,
])
N_BANDS = len(BAND_CENTRES)


def band_windows(freqs_hz, centres=BAND_CENTRES, width_octaves: float = 1 / 3):
    """Raised-cosine filter bank that sums to exactly 1 at every frequency.

    Each window spans ``width_octaves`` on each side of its centre in log2 space;
    overlapping neighbours are normalised so ``sum_b W_b(f) = 1``.  Any smooth
    overlap works -- the normalisation is what guarantees an artifact-free
    reconstruction of the band gains.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    lf = np.log2(np.maximum(f, 1e-6))
    lc = np.log2(centres)
    half = width_octaves
    w = np.zeros((len(centres), len(f)))
    for i, c in enumerate(lc):
        d = np.abs(lf - c) / half
        w[i] = np.where(d < 1.0, 0.5 * (1.0 + np.cos(np.pi * d)), 0.0)
    total = w.sum(axis=0)
    # outside the covered range keep the nearest band at unity gain
    w[:, total <= 0] = 0.0
    below = lf < lc[0] - half
    above = lf > lc[-1] + half
    w[0, below] = 1.0
    w[-1, above] = 1.0
    total = w.sum(axis=0)
    return w / np.maximum(total, 1e-12)


def bands_up_to(sample_rate: int, centres=BAND_CENTRES):
    """Band centres that fit below the Nyquist frequency for this sample rate."""
    nyq = sample_rate / 2.0
    keep = centres < nyq * 0.999
    return centres[keep]


def band_of(freq_hz, centres=BAND_CENTRES) -> int:
    """Index of the nearest 1/3-octave band."""
    return int(np.argmin(np.abs(np.log2(np.asarray(freq_hz) / centres))))
