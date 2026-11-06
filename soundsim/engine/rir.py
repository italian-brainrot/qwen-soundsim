"""Impulse-response synthesis from a set of acoustic paths.

Method: **1/3-octave band convolution** (Schmitz & Müller 1994), the standard
hybrid auralisation scheme:

1. every path contributes a delayed impulse whose amplitude is that path's
   pressure gain in each 1/3-octave band -> one sparse impulse train per band;
2. each train is filtered by its band window ``W_b(f)`` (a partition of unity,
   see :mod:`soundsim.engine.bands`) and the results are summed;
3. the incoherent diffuse tail is generated per band with the band's own decay
   constant, scaled so the *total* band energy satisfies the room's energy
   balance -- this is what lets a modest image-source order still give the
   correct reverberation time and the correct reverberant-to-direct ratio.

Fractional delays use a 4-tap Lagrange kernel, which is accurate well past
10 kHz and is the usual choice in image-source synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bands import BAND_CENTRES, band_windows
from ..acoustics.rooms import decay_curve_db, t60 as measure_t60


@dataclass
class Path:
    """One propagation path: a delay plus a per-band pressure gain."""
    delay_s: float
    gains: np.ndarray                 # (n_bands,) pressure gain per band
    kind: str = "direct"              # direct|reflection|transmission|diffraction|aperture
    label: str = ""
    order: int = 0                    # reflection order (for reporting)

    @property
    def energy(self) -> np.ndarray:
        return np.abs(self.gains) ** 2


@dataclass
class Tail:
    """Incoherent diffuse tail: per-band energy and decay time."""
    energy: np.ndarray                # (n_bands,) target energy of the tail
    t60: np.ndarray                   # (n_bands,) reverberation time
    onset_s: float = 0.0              # start of the tail


@dataclass
class RenderResult:
    ir: np.ndarray
    sample_rate: int
    band_centres: np.ndarray
    paths: list[Path] = field(default_factory=list)
    tail: Tail | None = None
    metrics: dict = field(default_factory=dict)


def _lagrange_taps(delay_samples: float) -> tuple[np.ndarray, int]:
    """4-tap Lagrange fractional-delay kernel and its first index."""
    d = delay_samples
    n0 = int(np.floor(d)) - 1
    x = np.arange(4) - (d - n0) + 1.0        # offsets relative to the sample point
    # Lagrange basis for the point at position (d - n0) evaluated on x = 0..3
    p = d - n0
    taps = np.ones(4)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            taps[i] *= (p - j) / (i - j)
    return taps, n0


def scatter_paths(paths: list[Path], sample_rate: int, length: int,
                  band_slice: slice) -> np.ndarray:
    """Sparse impulse train for one band: every path deposits its (fractional) delay."""
    train = np.zeros(length, dtype=np.float64)
    for pth in paths:
        g = pth.gains[band_slice]
        if not np.any(g):
            continue
        d = pth.delay_s * sample_rate
        taps, n0 = _lagrange_taps(d)
        idx = n0 + np.arange(4)
        m = (idx >= 0) & (idx < length)
        train[idx[m]] += g * taps[m]
    return train


def synthesize(paths: list[Path], tail: Tail | None, sample_rate: int,
               length: int | None = None, band_centres=BAND_CENTRES,
               seed: int = 12345) -> RenderResult:
    """Build the room impulse response of a scene."""
    band_centres = np.asarray(band_centres, dtype=np.float64)
    n_bands = len(band_centres)
    if length is None:
        longest = max([p.delay_s for p in paths], default=0.0)
        tail_len = 0.0 if tail is None else float(np.max(tail.t60)) * 0.9
        length = int((longest + tail_len + 0.05) * sample_rate)
        length = 1 << int(np.ceil(np.log2(max(length, 1024))))

    freqs = np.fft.rfftfreq(length, 1.0 / sample_rate)
    windows = band_windows(freqs, band_centres)

    ir_spec = np.zeros(len(freqs), dtype=np.complex128)
    rng = np.random.default_rng(seed)

    for b in range(n_bands):
        sl = slice(b, b + 1)
        train = scatter_paths(paths, sample_rate, length, sl)
        if np.any(train):
            ir_spec += np.fft.rfft(train) * windows[b]
        if tail is not None and tail.energy[b] > 0:
            tb = _make_tail(length, sample_rate, tail.t60[b], tail.energy[b],
                            tail.onset_s, rng)
            ir_spec += np.fft.rfft(tb) * windows[b]

    ir = np.fft.irfft(ir_spec, n=length)
    return RenderResult(ir=ir, sample_rate=sample_rate, band_centres=band_centres,
                        paths=paths, tail=tail)


def _make_tail(length: int, sample_rate: int, t60_s: float, energy: float,
               onset_s: float, rng) -> np.ndarray:
    """Decaying noise burst with the band's T60, normalised to a target energy."""
    if t60_s <= 0 or not np.isfinite(t60_s):
        return np.zeros(length)
    t = np.arange(length) / sample_rate
    onset = int(onset_s * sample_rate)
    if onset >= length:
        return np.zeros(length)
    # amplitude falls 60 dB over T60
    env = np.zeros(length)
    tt = t[onset:] - t[onset]
    env[onset:] = np.power(10.0, -3.0 * tt / t60_s)
    noise = rng.standard_normal(length - onset)
    seg = noise * env[onset:]
    e = float(np.sum(seg**2))
    if e <= 0:
        return np.zeros(length)
    out = np.zeros(length)
    out[onset:] = seg * np.sqrt(energy / e)
    return out
