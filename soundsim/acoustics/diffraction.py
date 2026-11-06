"""Diffraction: barriers, screen edges, and the shadow zone.

The reference model is Maekawa's formula (Maekawa 1968), which is what
ISO 9613-2 §8 uses for barrier attenuation:

    A_dB = 10 log10(3 + 20 N),      N = 2 delta / lambda

where ``delta`` is the excess path length of the diffracted ray over the direct
ray.  The ``lambda`` is what makes the physics audible: a barrier that stops
4 kHz gives a couple of dB at 125 Hz, which is exactly the "thump of traffic /
bass from a distant party" signature.

Two edges (a thick barrier) are combined by summing the two edge paths with
their own path differences; multiple screens follow the ISO 9613-2 cascade rule.
"""

from __future__ import annotations

import numpy as np

from .constants import speed_of_sound


def fresnel_number(path_excess_m, freq_hz, temp_c: float = 20.0) -> np.ndarray:
    """Fresnel number ``N = 2 delta / lambda`` (positive in the shadow zone)."""
    c = speed_of_sound(temp_c)
    f = np.asarray(freq_hz, dtype=np.float64)
    lam = c / f
    return 2.0 * np.asarray(path_excess_m, dtype=np.float64) / lam


def maekawa_attenuation_db(path_excess_m, freq_hz, temp_c: float = 20.0) -> np.ndarray:
    """Maekawa screen attenuation, dB (>= 0 in the shadow, 0 in the light zone).

    For ``N < 0`` (the direct ray is unobstructed) the formula still models the
    partial-shadow/interference region down to ``N = -0.2``, below which the
    attenuation is zero.
    """
    n = fresnel_number(path_excess_m, freq_hz, temp_c)
    n = np.maximum(n, 0.0)
    return 10.0 * np.log10(3.0 + 20.0 * n)


def maekawa_transfer(path_excess_m: float, freqs_hz, temp_c: float = 20.0) -> np.ndarray:
    """Complex transfer of a diffracted ray (magnitude of Maekawa + path delay).

    The phase of a diffracted field is nearly linear across the audio band for a
    single edge, so a magnitude filter on top of the geometric delay (which the
    caller applies through ``path_excess_m / c``) is an excellent approximation
    and avoids the well-known instability of Maekawa IIR filters.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    a_db = maekawa_attenuation_db(path_excess_m, f, temp_c)
    c = speed_of_sound(temp_c)
    gain = np.power(10.0, -a_db / 20.0)
    return gain * np.exp(-1j * 2.0 * np.pi * f * path_excess_m / c)


def barrier_path_excess(source, barrier_top, receiver) -> float:
    """Excess path length over the top of a screen, metres.

    ``source``, ``barrier_top``, ``receiver`` are 3-element coordinate tuples.
    Returns a negative value (clipped to 0 by the caller) when the barrier does
    not actually obstruct the direct ray.
    """
    s, b, r = np.asarray(source, float), np.asarray(barrier_top, float), np.asarray(receiver, float)
    direct = np.linalg.norm(r - s)
    bent = np.linalg.norm(b - s) + np.linalg.norm(r - b)
    return bent - direct


def is_shadowed(source, barrier_top, receiver) -> bool:
    """True when the straight source->receiver segment passes below ``barrier_top``."""
    s, b, r = np.asarray(source, float), np.asarray(barrier_top, float), np.asarray(receiver, float)
    d = r - s
    t = np.dot(b - s, d) / max(np.dot(d, d), 1e-12)
    if not 0.0 <= t <= 1.0:
        return False
    closest = s + t * d
    return closest[-1] < b[-1]


def multi_screen_attenuation_db(path_excesses, freq_hz, temp_c: float = 20.0) -> np.ndarray:
    """ISO 9613-2 cascade rule for ``n`` successive screens.

    ``A_total = 10 log10( prod_i (3 + 20 N_i) ) - 10 (n-1) log10 3``
    Each screen contributes its own edge; the constant term is subtracted so that
    a chain of screens with zero path excess costs nothing.
    """
    prod = np.ones_like(np.asarray(freq_hz, dtype=np.float64))
    n = 0
    for ex in path_excesses:
        if ex is None:
            continue
        n += 1
        prod = prod * (3.0 + 20.0 * np.maximum(fresnel_number(ex, freq_hz, temp_c), 0.0))
    if n == 0:
        return np.zeros_like(prod)
    return 10.0 * np.log10(prod) - 10.0 * (n - 1) * np.log10(3.0)
