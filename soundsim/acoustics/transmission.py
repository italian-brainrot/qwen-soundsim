"""Transmission of sound through partitions: walls, windows, car body panels.

Everything here is computed from the *damped-plate* equations of structural
acoustics (Cremer & Muller, *Structural Acoustics*, ch. 10; Ingeman 1953), so a
"wall" is described by thickness, density, Young's modulus, Poisson ratio and a
structural loss factor -- not by a measured R_w number.  The familiar mass law,
the coincidence dip and the low-frequency stiffness (governed) region all fall
out of the same expression.

Plate surface impedance for a wave whose trace wavenumber along the plate is
``k_t = k sin(theta)``::

    Z_p(theta, w) = j w m' (1 + j eta) [1 - (f0/f)^2]  -  j B k_t^4 / w

* the first term is the (damped) mass reactance, optionally corrected for the
  fundamental bending resonance ``f0`` of a finite panel;
* the second is the bending-stiffness reactance, which cancels the mass term at
  the coincidence angle -> the coincidence dip;

and for a plate in an infinite baffle in the same fluid on both sides the
power transmission coefficient at incidence angle ``theta`` is

    tau(theta) = 4 / | 2 + Z_p cos(theta) / z0 |^2

which reproduces the normal-incidence mass law and tends to unity at grazing
incidence.  The diffuse-field (reverberant room) value used throughout the
engine is the Beranek integral

    tau_diffuse(f) = 2 * int_0^{pi/2} tau(theta) sin(theta) cos(theta) dtheta
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import j1, struve

from .constants import C0, Z0, air_density, speed_of_sound, specific_impedance
from .materials import Panel, panel as get_panel


# ---------------------------------------------------------------------------
# Plate quantities
# ---------------------------------------------------------------------------
def coincidence_frequency_grazing(p: Panel, temp_c: float = 20.0) -> float:
    """Coincidence frequency at grazing incidence, Hz.

    Equating the plate wavenumber to the acoustic wavenumber, ``k_b = k``, with
    ``k_b^4 = w^2 m'/B`` gives ``w_c = c^2 sqrt(m'/B)``, i.e.

        f_c = c^2 / (2 pi) * sqrt(m' / B)
            = c^2 / (2 pi h) * sqrt(12 rho_s (1 - nu^2) / E)
    """
    c = speed_of_sound(temp_c)
    return c * c / (2.0 * np.pi) * np.sqrt(p.surface_density / p.bending_stiffness)


def fundamental_mode_freq(p: Panel, length_m: float, width_m: float) -> float:
    """(1,1) bending resonance of a simply supported rectangular plate, Hz.

    ``f0 = (pi / 2) * sqrt(B / (rho_s h)) * (1/a^2 + 1/b^2)``
    """
    b_over_m = p.bending_stiffness / p.surface_density
    return (np.pi / 2.0) * np.sqrt(b_over_m) * (1.0 / length_m**2 + 1.0 / width_m**2)


def plate_impedance(freqs_hz, theta_rad, p: Panel, f0_hz: float | None = None,
                    temp_c: float = 20.0, z0: float | None = None) -> np.ndarray:
    """Complex specific surface impedance of a damped plate, Pa s/m."""
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    w = 2.0 * np.pi * f
    c = speed_of_sound(temp_c)
    z0 = z0 if z0 is not None else specific_impedance(temp_c)
    k = w / c
    m_eff = p.surface_density
    if f0_hz:
        # SDOF mass-spring analogue: below the panel's own first bending mode the
        # panel is stiffness controlled and its mobility falls (TL rises).
        mass_factor = 1.0 - (f0_hz / f) ** 2
    else:
        mass_factor = 1.0
    mass_term = 1j * w * m_eff * (1.0 + 1j * p.loss_factor) * mass_factor
    k_t = k * np.sin(theta_rad)
    stiff_term = -1j * p.bending_stiffness * k_t**4 / w
    return mass_term + stiff_term


def transmission_coefficient(theta_rad, freqs_hz, p: Panel,
                             f0_hz: float | None = None,
                             temp_c: float = 20.0) -> np.ndarray:
    """Sound power transmission coefficient tau at a single incidence angle."""
    z0 = specific_impedance(temp_c)
    zp = plate_impedance(freqs_hz, theta_rad, p, f0_hz, temp_c, z0)
    denom = 2.0 + zp * np.cos(theta_rad) / z0
    return 4.0 / np.abs(denom) ** 2


def tau_normal(freqs_hz, p: Panel, f0_hz: float | None = None,
               temp_c: float = 20.0) -> np.ndarray:
    """Normal-incidence transmission coefficient."""
    return transmission_coefficient(0.0, freqs_hz, p, f0_hz, temp_c)


def tau_field(freqs_hz, p: Panel, f0_hz: float | None = None,
              temp_c: float = 20.0, n_angles: int = 181) -> np.ndarray:
    """Diffuse-field transmission coefficient (Beranek angle weighting)."""
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    theta = np.linspace(0.0, np.pi / 2.0, n_angles)
    th = theta[1:-1]                                   # endpoints: measure-zero
    # Simpson-friendly: use trapezoid on a dense grid instead
    integ = transmission_coefficient(th[:, None], f[None, :], p, f0_hz, temp_c)
    weight = np.sin(th) * np.cos(th)
    tau = 2.0 * np.trapezoid(integ * weight[:, None], th, axis=0)
    return np.clip(tau, 1e-12, 1.0)


def transmission_loss_db(freqs_hz, p: Panel | str, field_incidence: bool = True,
                         f0_hz: float | None = None,
                         temp_c: float = 20.0) -> np.ndarray:
    """Transmission loss R = -10 log10(tau), dB.  The workhorse function."""
    if isinstance(p, str):
        p = get_panel(p)
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    tau = tau_field(f, p, f0_hz, temp_c) if field_incidence else tau_normal(f, p, f0_hz, temp_c)
    return -10.0 * np.log10(tau)


def mass_law_db(freqs_hz, surface_density: float, field_incidence: bool = True,
                temp_c: float = 20.0) -> np.ndarray:
    """Analytic mass law, dB.

    Normal incidence:  ``20 log10(m' f) - 47.4 + 5.3``  (the -42.4 form);
    field incidence (random incidence, the usual building-physics number):
    ``20 log10(m' f) - 47.4``.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    z0 = specific_impedance(temp_c)
    tl_normal = 20.0 * np.log10(np.maximum(f * surface_density / (2.0 * z0 / (2 * np.pi)), 1e-12))
    return tl_normal - (8.5 if field_incidence else 0.0)


# ---------------------------------------------------------------------------
# Multi-leaf partitions: mass - cavity - mass
# ---------------------------------------------------------------------------
@dataclass
class Leaf:
    panel: Panel
    f0_hz: float | None = None


def cavity_impedance(freqs_hz, depth_m: float, temp_c: float = 20.0,
                     loss: float = 0.05, z0: float | None = None) -> np.ndarray:
    """Specific acoustic impedance of a closed air gap of depth ``d``.

    ``Z_cav = -j z0 cot(k d)`` for a rigid-backed layer; the cavity resonances
    (k d = pi/2, 3pi/2 ...) are the reason double glazing loses its extra
    isolation at high frequency.  ``loss`` adds a real part that represents
    absorption/damping of the cavity (fibre filling, perforation).
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    z0 = z0 if z0 is not None else specific_impedance(temp_c)
    c = speed_of_sound(temp_c)
    k = 2.0 * np.pi * f / c
    kd = np.clip(k * depth_m, 1e-4, None)
    zc = -1j * z0 / np.tan(kd * (1.0 + 1j * loss))
    return zc


def _as_panel(p):
    return get_panel(p) if isinstance(p, str) else p


def double_leaf_tl_db(freqs_hz, panel1: "Panel | str", panel2: "Panel | str",
                      cavity_depth_m: float, field_incidence: bool = True,
                      temp_c: float = 20.0,
                      mechanical_coupling_db: float = 3.0) -> np.ndarray:
    """Transmission loss of two leaves separated by an air cavity.

    The classic limp double-leaf prediction (Beranek; Mather & Hendrick): the
    cavity acts as a spring, so the pair has a resonance

        f0 = 60 sqrt( (m1 + m2) / (m1 m2 d) )

    * below ``f0`` both leaves move together and the pair obeys the mass law of
      the *sum* of the surface densities;
    * above ``f0`` the leaves move out of phase and the loss is the sum of the
      two leaves' own losses plus the cavity bonus ``10 log10(d f) - 33``.

    Each leaf's own loss comes from the damped-plate model above, so coincidence
    and damping are already inside it.  ``field_incidence`` is accepted for
    call-compatibility: the pair rule is a diffuse-field construction, and there
    is no meaningful oblique version of the cavity bonus.  ``mechanical_coupling_db`` accounts for
    the frame and spacer tying real windows together, which is why measured
    double glazing sits a few dB under the limp theory.

    (A full 2-DOF plate/cavity solve is *not* used: with stiff, lightly damped
    plates below their own resonances the lumped impedances blow up and predict
    >100 dB of isolation for a window, which no window delivers.)
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    p1, p2 = _as_panel(panel1), _as_panel(panel2)
    m1, m2 = p1.surface_density, p2.surface_density
    d = max(cavity_depth_m, 1e-3)

    f0 = 60.0 * math.sqrt((m1 + m2) / (m1 * m2 * d))
    # leaves moving together
    tl_low = 20.0 * np.log10(np.maximum((m1 + m2) * f, 1e-6)) - 47.5
    # leaves moving independently, cavity spring active
    tl1 = transmission_loss_db(f, p1, field_incidence=True, temp_c=temp_c)
    tl2 = transmission_loss_db(f, p2, field_incidence=True, temp_c=temp_c)
    bonus = np.minimum(10.0 * np.log10(d * np.maximum(f, 1e-6)) - 33.0, 20.0)
    tl_high = tl1 + tl2 + bonus - mechanical_coupling_db

    tl = np.where(f < f0, tl_low, tl_high)
    # the double-leaf resonance: a dip of a few dB around f0
    dip = 6.0 * np.exp(-0.5 * (np.log2(np.maximum(f, 1e-6) / f0) / 0.35) ** 2)
    tl = tl - dip
    # both leaves' mass laws put a hard ceiling on what any pair can do
    ceiling = tl_low + 20.0 * np.log10(np.maximum(f / f0, 1.0)) + 10.0
    return np.maximum(np.minimum(tl, ceiling), 0.0)


def tl_double_leaf_db(*a, **kw) -> np.ndarray:      # backwards-compatible alias
    return double_leaf_tl_db(*a, **kw)


def combine_transmission(taus, areas) -> np.ndarray:
    """Energy-sum several elements of different area (ISO 12354 / classic rule).

    ``tau_avg = sum(S_i tau_i) / sum(S_i)``
    """
    taus = np.atleast_2d(np.asarray(taus, dtype=np.float64))
    areas = np.asarray(areas, dtype=np.float64)
    areas = areas / areas.sum()
    return np.sum(taus * areas[:, None], axis=0)


def flanking_tau(tau_direct, tl_flanking_db, area_direct: float,
                 area_flanking: float) -> np.ndarray:
    """Add a flanking path, in the spirit of ISO 12354-1.

    The flanking route arrives with its own transmission loss but also with a
    longer, more damped path; only the transmission step is modelled here.
    """
    tau_fl = 10.0 ** (-np.asarray(tl_flanking_db, dtype=np.float64) / 10.0)
    w = np.array([area_direct, area_flanking], dtype=np.float64)
    w /= w.sum()
    return w[0] * np.atleast_1d(tau_direct) + w[1] * np.atleast_1d(tau_fl)


# ---------------------------------------------------------------------------
# Openings: window left open, doorway, vent
# ---------------------------------------------------------------------------
def piston_radiation_impedance(freqs_hz, area_m2: float, temp_c: float = 20.0) -> np.ndarray:
    """Normalised specific radiation impedance ``z = Z_r / (rho c S)`` of a
    baffled piston of area ``S`` (equivalent radius ``a = sqrt(S/pi)``).

    Rayleigh's exact result (with ``x = 2ka``)::

        z = 1 - 2 J1(x)/x  +  2 j H1(x)/x

    where ``J1`` is the Bessel and ``H1`` the Struve function of the first kind.
    Both correction terms decay like ``x^-1/2``, so ``z -> 1``: a large opening
    stops being a load at all.  At low frequency ``Re z ~ (ka)^2/2`` (radiation
    resistance) and ``Im z ~ 8ka/(3 pi)`` (the mass of the air plug, i.e.
    the classic ``8a/(3 pi)`` end correction).
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    c = speed_of_sound(temp_c)
    a = np.sqrt(area_m2 / np.pi)
    ka = 2.0 * np.pi * f * a / c
    x = np.clip(2.0 * ka, 1e-8, None)
    r1 = 1.0 - 2.0 * j1(x) / x
    x1 = 2.0 * struve(1, x) / x
    return r1 + 1j * x1


def aperture_transmission(freqs_hz, area_m2: float, temp_c: float = 20.0) -> np.ndarray:
    """Energy transmission coefficient of an opening in a rigid wall.

    Treat the aperture as a boundary with normalised specific impedance ``z``:
    the fraction of incident intensity that passes is the classic normal-
    incidence absorption of that boundary,

        tau_ap = 4 Re(z) / |z + 1|^2

    -> ``tau_ap ~ 2 (ka)^2`` for a hole small compared with a wavelength (so a
    keyhole barely lets bass through, and the *effective* open area grows as a^4),
    and -> 1 once the opening is large compared with the wavelength, which is the
    everyday "an open window is no insulation at all" limit.
    """
    z = piston_radiation_impedance(freqs_hz, area_m2, temp_c)
    return np.clip(4.0 * np.real(z) / np.abs(z + 1.0) ** 2, 0.0, 1.0)


def radiation_efficiency(freqs_hz, area_m2: float, temp_c: float = 20.0) -> np.ndarray:
    """Fraction of a baffled piston's vibration that actually leaves as sound.

        sigma = 1 - J1(2ka) / (ka),      a = sqrt(S / pi)

    This is the missing half of low-frequency transmission.  A plate model says
    a window or a door isolates only a few dB below its double-leaf resonance --
    true, and why neighbours are all thump -- but the infinite-baffle transmission
    coefficient also assumes the partition can couple to the air.  Once the
    wavelength is long compared with the element (ka < 1) it cannot: the radiation
    resistance collapses as (ka)^2, so most of what gets through the panel stays
    as near-field motion and never propagates.

    Without it, an engine can put more energy at 30 Hz out of a 2 m^2 window than
    a real facade ever does -- inaudible, all of it in the cone, all of it buzz.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    a = math.sqrt(max(float(area_m2), 1e-6) / math.pi)
    ka = 2.0 * np.pi * f * a / speed_of_sound(temp_c)
    small = ka < 1e-3
    sig = np.where(small, 0.5 * np.maximum(ka, 1e-9) ** 2,
                   1.0 - j1(2.0 * np.maximum(ka, 1e-9)) / np.maximum(ka, 1e-9))
    return np.clip(sig, 0.0, 1.2)


def piston_directivity(freqs_hz, area_m2: float, angle_rad: float, temp_c: float = 20.0,
                       n_theta: int = 129) -> np.ndarray:
    """Energy-normalised directivity of a baffled piston, ``D(theta, f)``.

    ``2 J1(ka sin theta)/(ka sin theta)`` normalised so that
    ``int |D|^2 dOmega = 2 pi`` over the radiating hemisphere, which keeps the
    total radiated power correct while still giving the narrowing beam (and the
    on-axis gain) of a large opening.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    c = speed_of_sound(temp_c)
    a = np.sqrt(area_m2 / np.pi)
    k = 2.0 * np.pi * f / c

    def d(th):
        x = np.clip(k * a * np.sin(th), 1e-9, None)
        val = 2.0 * j1(x) / x
        return np.where(k * a * np.sin(th) < 1e-6, 1.0, val)

    # normalise over the hemisphere
    th = np.linspace(0.0, np.pi / 2.0, n_theta)
    power = 2.0 * np.pi * np.trapezoid(np.abs(d(th[:, None])) ** 2 * np.cos(th)[:, None], th, axis=0)
    return d(angle_rad) / np.sqrt(np.maximum(power / (2.0 * np.pi), 1e-12))


def aperture_transfer(freqs_hz, area_m2: float, distance_m: float, temp_c: float = 20.0,
                      angle_rad: float = 0.0) -> np.ndarray:
    """Pressure transfer ``p_far / p_at_wall`` for an opening in a rigid wall.

    Energy route (the same cascade used for a transmitting wall, see
    :func:`panel_transfer`): the interior intensity ``I = p^2/(rho c)`` delivers
    power ``W = I S tau_ap`` through the opening, which is re-radiated into the
    exterior hemisphere::

        p_far^2 = rho c W / (2 pi r^2) * D^2
        |H|     = sqrt(S tau_ap / (2 pi r^2)) * D

    Modelling the *intensity* (not a prescribed plug velocity) is what makes the
    scaling come out right: bigger openings really do transmit more, and small
    ones really do cut the bass, which is the classic "music from the flat next
    door" signature.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    tau = aperture_transmission(f, area_m2, temp_c)
    d = piston_directivity(f, area_m2, angle_rad, temp_c)
    mag = np.sqrt(tau * area_m2 / (2.0 * np.pi * distance_m**2)) * np.abs(d)
    k = 2.0 * np.pi * f / speed_of_sound(temp_c)
    return mag * np.exp(-1j * k * distance_m)


def panel_transfer(freqs_hz, area_m2: float, distance_m: float, tau_panel,
                   temp_c: float = 20.0) -> np.ndarray:
    """Pressure transfer of a *transmitting wall element* to a far receiver.

    Same cascade as the aperture: ``|H| = sqrt(S tau / (2 pi r^2))``, i.e. the
    element becomes a secondary source whose strength is the transmitted power.
    A wall element radiates essentially omnidirectionally at the wavelengths that
    matter for its own dimensions, so no directivity is applied here.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    tau = np.clip(np.asarray(tau_panel, dtype=np.float64), 1e-12, 1.0)
    mag = np.sqrt(tau * area_m2 / (2.0 * np.pi * distance_m**2))
    k = 2.0 * np.pi * f / speed_of_sound(temp_c)
    return mag * np.exp(-1j * k * distance_m)


def aperture_loss_db(freqs_hz, area_m2: float, distance_m: float,
                     temp_c: float = 20.0) -> np.ndarray:
    """Insertion loss of an opening, dB, relative to an unobstructed field."""
    h = np.abs(aperture_transfer(freqs_hz, area_m2, distance_m, temp_c))
    ref = 1.0 / distance_m                      # free-field 1/r reference
    return -20.0 * np.log10(np.maximum(h / ref, 1e-30))
