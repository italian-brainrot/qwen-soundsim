"""Diffraction: barriers, screen edges, and the shadow zone.

The reference model is Maekawa's formula (Maekawa 1968), which is what
ISO 9613-2 section 8 uses for barrier attenuation:

    A_dB = 10 log10(3 + 20 N),      N = 2 delta / lambda

where ``delta`` is the excess path length of the diffracted ray over the direct
ray.  The ``lambda`` is what makes the physics audible: a barrier that stops
4 kHz gives a couple of dB at 125 Hz, which is exactly the "thump of traffic /
bass from a distant party" signature.

Two things this module is *not* asked to decide, and therefore keeps separate:

* **whether a screen is in the way at all** -- :func:`is_shadowed` (one edge) or
  ``Barrier.blocks`` (a finite screen).  ``A_dB`` is an *insertion loss on the
  direct ray*: a screen the ray flies over costs nothing, and
  ``N = 2 delta / lambda`` cannot tell you that, because ``delta`` is a sum of
  two sides of a triangle and so is never negative.
* **how several edges combine.**  Two edges of one screen top are alternative
  routes to the same receiver and are summed in energy (:func:`edge_group_attenuation_db`);
  several screens one behind the other are in series and follow the
  ISO 9613-2 cascade rule (:func:`multi_screen_attenuation_db`).
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
    """Maekawa screen attenuation, dB, for a receiver in the shadow zone.

    ``N`` is clipped at zero, so the lowest value this can return is the
    grazing/limiting one, ``10 log10 3 = 4.77 dB``.  Attenuation in the *light*
    zone -- where the direct ray passes over the edge unobstructed -- is zero,
    and that is decided by geometry (:func:`is_shadowed`), not by this formula:
    the path excess ``delta = |b-s| + |r-b| - |r-s|`` is a sum of two sides of a
    triangle minus the third, hence ``>= 0`` for *every* position of the edge,
    including one the ray flies over.
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
    Always ``>= 0`` (triangle inequality) -- it is the extra *travel*, not a
    measure of obstruction: ask :func:`is_shadowed` whether the screen is in the
    way before applying this.
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


def edge_group_attenuation_db(path_excesses, freq_hz, temp_c: float = 20.0) -> np.ndarray:
    """Attenuation of one screen top that has several diffracting edges.

    The two edges of a flat-topped barrier are two alternative routes from the
    same source to the same receiver, so their contributions add **in energy**,
    each referenced to the free field.  Averaging the transmitted powers (rather
    than adding them) keeps the bookkeeping normalised: a screen whose two edges
    coincide -- zero thickness, the usual case -- comes back out as exactly the
    single-edge Maekawa loss instead of 3 dB better than it.
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    powers = [np.power(10.0, -maekawa_attenuation_db(ex, f, temp_c) / 10.0)
              for ex in path_excesses]
    if not powers:
        return np.zeros_like(f)
    return -10.0 * np.log10(np.mean(powers, axis=0))


def series_attenuation_db(attenuations_db) -> np.ndarray:
    """ISO 9613-2 cascade rule for ``n`` screens one behind the other.

    ``A_total = SUM_i A_i - (n - 1) 10 log10 3``

    Each screen's own insertion loss already contains Maekawa's constant
    ``10 log10 3 = 4.77 dB``, which is the cost of *an* edge rather than the cost
    of *this* edge, so it is charged once: a chain of screens that all sit exactly
    on the line of sight costs 4.77 dB, not n times 4.77 dB.
    """
    a = [np.asarray(x, dtype=np.float64) for x in attenuations_db if x is not None]
    if not a:
        return np.zeros(1, dtype=np.float64)
    return sum(a) - np.log10(3.0) * 10.0 * (len(a) - 1)


def multi_screen_attenuation_db(path_excesses, freq_hz, temp_c: float = 20.0) -> np.ndarray:
    """ISO 9613-2 cascade rule for ``n`` successive screens, from path excesses.

    ``A_total = 10 log10( prod_i (3 + 20 N_i) ) - 10 (n-1) log10 3``
    Each screen contributes its own edge; the constant term is subtracted so that
    a chain of screens with zero path excess costs nothing.

    "Successive" means one behind the other as seen from the source, each one
    taking its own insertion loss: it is *not* the rule for the two edges of a
    single top, which are parallel routes -- see :func:`edge_group_attenuation_db`.
    """
    return series_attenuation_db([maekawa_attenuation_db(ex, freq_hz, temp_c)
                                  for ex in path_excesses if ex is not None])
