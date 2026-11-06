"""Image-source method for rectangular rooms (Schroeder 1947 / Borish 1984).

For every reflection order the algorithm produces the mirror image of the source,
the walls it bounced off, the path length and -- crucially for accuracy -- the
band-dependent pressure gain

    g = prod_walls sqrt(1 - alpha_wall(f)) * (1 / r) * 10^(-alpha_atm(f) r / 20)

so the "boxy" low-frequency behaviour of a small room, the dead sound of a
carpeted bedroom and the high-frequency bite of a concrete stairwell all follow
from geometry plus material data rather than from a preset.

Correct enumeration of the image lattice
----------------------------------------
Along one axis the images of a source at ``x`` are the two cosets

    x_img =  x + 2 k L        and        x_img = -x + 2 k L      (k integer)

and the number of reflections on that axis is exactly the number of wall planes
``x = j L`` strictly between ``x`` and ``x_img`` in the unfolded picture.  The
plane index ``j`` also identifies *which* wall was hit (j even -> the wall at
coordinate 0, j odd -> the wall at L), so differently absorbing walls on
opposite sides are handled properly.

The ISM is exact for a box; it converges to a diffuse field above the Schroeder
frequency and is the standard reference method for auralisation research
(Svennerberg & Sundberg 1994).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageSourcePaths:
    """Structure-of-arrays description of every image-source path."""
    distance: np.ndarray        # (P,) path length, m
    order: np.ndarray           # (P,) int reflection order (0 = direct)
    hit_counts: np.ndarray      # (P, 3, 2) hits per axis and per wall side
    direction: np.ndarray       # (P, 3) unit vector receiver -> image (arrival direction)
    multiplicity: np.ndarray    # (P,) reflection sequences that give this image

    @property
    def count(self) -> int:
        return int(len(self.distance))


def _multinomial(nx: int, ny: int, nz: int) -> int:
    """Distinct reflection sequences that reach one image position.

    A single lattice image is reached by many wall sequences.  Within one axis the
    sequence of *sides* is forced (a wall cannot be hit twice in a row, and the
    image position fixes which side comes first), so the only freedom is how the
    hits on the three axes interleave::

        M = n! / (nx! ny! nz!)

    Summing M over all images of order n reproduces the textbook sequence count
    ``6 * 5**(n-1)``, which the tests check.  Omitting this factor makes an
    image-source energy sum underestimate the reverberant field by a large margin.
    """
    from math import factorial
    return factorial(nx + ny + nz) // (factorial(nx) * factorial(ny) * factorial(nz))


def _axis_images(src: float, length: float, max_order: int):
    """Yield ``(x_img, [(plane_index, side), ...])`` for one axis.

    Both cosets are enumerated; the crossing list comes from the wall planes
    strictly between the source and the image.
    """
    out = []
    kmax = max_order // 2 + 2
    for flip in (0, 1):
        base = src if flip == 0 else -src
        for k in range(-kmax, kmax + 1):
            x_img = base + 2.0 * k * length
            lo, hi = (src, x_img) if x_img > src else (x_img, src)
            # integer plane indices j with lo < j*length < hi
            j_lo = int(np.floor(lo / length)) + 1
            j_hi = int(np.ceil(hi / length)) - 1
            planes = []
            for j in range(j_lo, j_hi + 1):
                if lo < j * length < hi:
                    planes.append((j, j % 2))
            # a source exactly on a plane (degenerate) would miscount; ignore
            out.append((x_img, planes))
    return out


def image_source_paths(source, receiver, dims, max_order: int) -> ImageSourcePaths:
    """Enumerate image-source paths up to ``max_order`` in a rectangular room.

    Room occupies ``[0,Lx] x [0,Ly] x [0,Lz]``.  The direct path (order 0) is
    always present exactly once.
    """
    s = np.asarray(source, dtype=np.float64)
    r = np.asarray(receiver, dtype=np.float64)
    dims = np.asarray(dims, dtype=np.float64)

    per_axis = [_axis_images(s[a], dims[a], max_order) for a in range(3)]

    dist, order, counts, dirs, mult = [], [], [], [], []
    for xa, pa in enumerate(per_axis[0]):
        for ya, pb in enumerate(per_axis[1]):
            for za, pc in enumerate(per_axis[2]):
                nx, ny, nz = len(pa[1]), len(pb[1]), len(pc[1])
                n = nx + ny + nz
                if n > max_order:
                    continue
                img = np.array([pa[0], pb[0], pc[0]])
                d = float(np.linalg.norm(img - r))
                if d < 1e-9:
                    continue
                row = np.zeros((3, 2), dtype=np.int16)
                for axis, planes in enumerate((pa[1], pb[1], pc[1])):
                    for _j, side in planes:
                        row[axis, side] += 1
                dist.append(d)
                order.append(n)
                counts.append(row)
                dirs.append((img - r) / d)
                mult.append(_multinomial(nx, ny, nz))

    return ImageSourcePaths(
        np.array(dist, dtype=np.float64),
        np.array(order, dtype=np.int32),
        np.array(counts, dtype=np.int16).reshape(-1, 3, 2),
        np.array(dirs, dtype=np.float64).reshape(-1, 3),
        np.array(mult, dtype=np.float64),
    )


def path_gains(paths: ImageSourcePaths, band_freqs_hz, absorbers_by_axis,
               temp_c: float = 20.0, alpha_atm_db_per_band=None) -> np.ndarray:
    """Band-dependent pressure gains ``[n_paths, n_bands]`` (includes 1/r).

    ``absorbers_by_axis[axis]`` is a pair of :class:`Absorber` -- the wall at
    coordinate 0 and the wall at coordinate L.
    """
    band_freqs_hz = np.asarray(band_freqs_hz, dtype=np.float64)
    n_bands = len(band_freqs_hz)

    # amplitude reflection coefficient per axis/side: sqrt(1 - alpha)
    refl = np.zeros((3, 2, n_bands))
    for axis in range(3):
        for side in range(2):
            ab = absorbers_by_axis[axis][side]
            a = ab.alpha_at(band_freqs_hz)
            refl[axis, side, :] = np.sqrt(np.maximum(1.0 - a, 0.0))

    gains = np.ones((paths.count, n_bands), dtype=np.float64)
    for axis in range(3):
        for side in range(2):
            c = paths.hit_counts[:, axis, side].astype(np.float64)
            if c.max() == 0:
                continue
            gains *= refl[axis, side, :][None, :] ** c[:, None]

    gains /= np.maximum(paths.distance, 1e-6)[:, None]

    # NOTE on ``paths.multiplicity``: several wall *sequences* reach the same
    # image position.  They are not multiplied into the gain, because in a box
    # (an integrable geometry) that factor makes the incoherent image sum grow
    # like 5^n (1-alpha)^n and diverge unless alpha > 0.8 -- the classic
    # degeneracy of rectangular-image-source modelling.  Each image is therefore
    # counted once (the Borish convention); the missing late energy is supplied
    # by the energy-balanced diffuse tail, which is what the hybrid scheme is for.
    # ``multiplicity`` is kept for diagnostics only.

    if alpha_atm_db_per_band is not None:
        a = np.asarray(alpha_atm_db_per_band, dtype=np.float64)
        gains *= np.power(10.0, -a[None, :] * paths.distance[:, None] / 20.0)
    return gains
