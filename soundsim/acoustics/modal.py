"""Exact modal (normal-mode) response of a rectangular room.

Below the Schroeder frequency a room is not a diffuse field: it is a handful of
Rayleigh modes, and *geometrical* methods (image sources, rays) are the wrong
tool -- they get the total energy roughly right but not the ringing, and they
invent a decay slope where there is none.  This module solves the same box
exactly, as a sum over its eigenmodes, and it is used by the renderer for the
low bands of any enclosed space.  That is what produces the correct seat-to-seat
bass variation in a car, the "boom" of a stairwell and the boxy colouration of a
small bedroom.

The Green's function of a rigid-walled box is

    G(r, r_s; k) = SUM_n  phi_n(r) phi_n(r_s) / (k_n^2 - k^2 + j k_n^2 / Q_n)

with the Rayleigh modes ``k_n = pi (n_x/Lx, n_y/Ly, n_z/Lz)`` and the orthonormal
shapes ``phi_n = prod cos(n_i pi x_i / L_i)`` (norm 1/V per axis, 2 for the
non-zero indices).  Since ``(laplacian + k^2) G = -delta``, a source whose
free-field pressure at 1 m is 1 Pa has volume velocity ``q = 4 pi / (rho0 w)``,
and then

    H(k) = j w rho0 q G = j 4 pi G      ->      |H| -> 1/r   in free field

so the modal sum is automatically on the same scale as the 1/r direct sound.

Modal damping uses the mode's own wall absorption: a mode "sees" the walls it
presses hardest, weighted by its direction cosines, giving a per-mode Q and hence
a per-mode decay ``sigma_n = c alpha_n S / (4V)`` which reduces exactly to Sabine
when absorption is uniform.
"""

from __future__ import annotations

import numpy as np

from .constants import speed_of_sound
from .materials import Absorber

_MODAL_CHUNK_BYTES = 24_000_000      # cap the temporaries at ~24 MB per step


def _alpha_of(ab, freqs_hz) -> np.ndarray:
    """Accept either an Absorber or a plain callable freq -> alpha."""
    if hasattr(ab, "alpha_at"):
        return ab.alpha_at(freqs_hz)
    return np.asarray(ab(np.atleast_1d(freqs_hz)), dtype=np.float64)


def modal_frequencies(dims, f_max: float, temp_c: float = 20.0):
    """Rayleigh eigenfrequencies and mode indices up to ``f_max``."""
    c = speed_of_sound(temp_c)
    dims = np.asarray(dims, float)
    ns = [int(np.floor(2.0 * f_max * L / c)) for L in dims]
    grid = np.meshgrid(*[np.arange(n + 1) for n in ns], indexing="ij")
    idx = np.stack([g.ravel() for g in grid], axis=1)
    k = np.pi * np.linalg.norm(idx / dims[None, :], axis=1)
    f = c * k / (2.0 * np.pi)
    keep = f > 1e-6
    return idx[keep], f[keep]


def mode_shapes(idx, dims, src, dst) -> tuple[np.ndarray, np.ndarray]:
    """phi_n(source) and phi_n(receiver) for the orthonormal mode set."""
    dims = np.asarray(dims, float)
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    out = []
    for p in (src, dst):
        # per-axis factor: 1/sqrt(L) for n=0, sqrt(2/L) for n>0
        vals = np.ones(len(idx))
        for axis, L in enumerate(dims):
            n = idx[:, axis]
            coef = np.where(n == 0, 1.0 / np.sqrt(L), np.sqrt(2.0 / L))
            vals *= coef * np.cos(n * np.pi * p[axis] / L)
        out.append(vals)
    return out[0], out[1]


def modal_absorption(idx, dims, alpha_by_axis) -> np.ndarray:
    """Absorption coefficient each mode experiences (Kuttruff, ch. 6).

    A mode loses energy mainly through the walls it has antinodes against: the
    wall pair normal to axis i is loaded in proportion to the mode's direction
    cosine along that axis.
    """
    dims = np.asarray(dims, float)
    nz = np.maximum(idx, 0).astype(float)
    weights = (nz / dims[None, :]) ** 2
    total = weights.sum(axis=1)
    total[total == 0] = 1.0
    frac = weights / total[:, None]
    a = np.zeros(len(idx))
    for axis in range(3):
        a += frac[:, axis] * np.asarray(alpha_by_axis[axis], dtype=np.float64)
    return np.clip(a, 1e-4, 1.0)


def modal_transfer(dims, src, dst, freqs_hz, alpha_by_axis,
                   temp_c: float = 20.0, f_max: float | None = None) -> np.ndarray:
    """Complex transfer function (same 1/r free-field normalisation as the paths).

    Convergence: a plain modal sum is useless when truncated (its terms fall only
    as ``1/k_n^2``), so the free-field Green's function is added and subtracted
    (Krenk / Rayleigh subtraction)::

        G = G_free + SUM phi_n phi_n [1/(k_n^2 - k^2 + j k sigma_n/c) - 1/k_n^2]

    the bracket now falling like ``k^2/k_n^4``, and vanishing identically at
    ``k -> 0`` so the exact DC limit is kept.  ``alpha_by_axis[axis]`` is an
    absorber (or a callable freq -> alpha) for the two walls normal to that axis,
    evaluated at each mode's frequency.
    """
    c = speed_of_sound(temp_c)
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    k = 2.0 * np.pi * f / c
    if f_max is None:
        f_max = 2.2 * float(np.max(f)) if len(f) else 100.0
    idx, f_n = modal_frequencies(dims, f_max, temp_c)
    k_n = 2.0 * np.pi * f_n / c

    r = max(float(np.linalg.norm(np.asarray(dst, float) - np.asarray(src, float))), 1e-9)

    axes_alpha = [_alpha_of(alpha_by_axis[axis], f_n) for axis in range(3)]
    a_n = modal_absorption(idx, dims, axes_alpha)
    v = float(np.prod(dims))
    sigma = c * a_n * float(np.sum(_axis_areas(dims))) / (4.0 * v)   # decay rate, 1/s

    phi_s, phi_r = mode_shapes(idx, dims, src, dst)
    coupling = phi_s * phi_r

    # Accumulate mode-by-mode in bounded chunks.  The naive (n_freq x n_modes)
    # temporary is enormous -- a 20 s response at 48 kHz has ~500k FFT bins and a
    # 9 m tall room has tens of thousands of modes below 900 Hz -- so stream it.
    correction = np.zeros(len(f), dtype=np.complex128)
    chunk = max(1, int(_MODAL_CHUNK_BYTES // (16 * max(len(f), 1))))
    for i0 in range(0, len(f_n), chunk):
        sl = slice(i0, i0 + chunk)
        # causal lossy-oscillator denominator: poles in the lower half plane
        denom = k_n[None, sl] ** 2 - k[:, None] ** 2 + 1j * k[:, None] * sigma[None, sl] / c
        correction += np.sum(coupling[None, sl] * (1.0 / denom - 1.0 / k_n[None, sl] ** 2),
                             axis=1)

    g_free = np.exp(-1j * k * r) / (4.0 * np.pi * r)
    return 1j * 4.0 * np.pi * (g_free + correction)


def _axis_areas(dims) -> np.ndarray:
    """Area of the two walls normal to x, y and z."""
    lx, ly, lz = dims
    return np.array([2.0 * ly * lz, 2.0 * lx * lz, 2.0 * lx * ly])


def modal_decay_times(dims, alpha_by_axis, temp_c: float = 20.0,
                      f_max: float = 400.0) -> np.ndarray:
    """T60 of each mode -- useful for spotting which low modes ring on."""
    c = speed_of_sound(temp_c)
    idx, f_n = modal_frequencies(dims, f_max, temp_c)
    axes_alpha = [_alpha_of(ab, f_n) for ab in alpha_by_axis]
    a_n = modal_absorption(idx, dims, axes_alpha)
    v = float(np.prod(dims))
    s_tot = float(np.sum(_axis_areas(dims)))
    sigma = c * a_n * s_tot / (4.0 * v)
    return f_n, 13.8155 / np.maximum(sigma, 1e-9)
