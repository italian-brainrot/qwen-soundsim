"""Atmospheric (molecular + classical) sound absorption in air.

Implements ISO 9613-1:1993 Annex A (Bass, King, Langevelen & Sutherland, 1984),
which is the basis of the tabulated coefficients of ANSI/ASA S1.26-1978.

The attenuation coefficient is the sum of four contributions::

    alpha = alpha_cr + alpha_vib,O + alpha_vib,N

* ``alpha_cr``  classical (thermal conduction + viscosity) plus rotational
  relaxation of N2/O2 -- merged into one term by the standard because the
  rotational relaxation is essentially complete at audio frequencies.
* ``alpha_vib,O`` vibrational relaxation of oxygen (dominant loss above ~2 kHz).
* ``alpha_vib,N`` vibrational relaxation of nitrogen, catalysed by water vapour,
  which produces the strong humidity dependence around 100 Hz - 4 kHz.

Note on units: the published MATLAB listing that circulates with this annex
passes ``f`` in kHz; the physical equations require **Hz** throughout (the
(f/c) factor of the relaxation lineshape and the f^2 of the classical term).
Using Hz reproduces the ANSI S1.26 tables exactly, which is what the unit tests
assert.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    P_REF_PA,
    T0_K,
    T_TRIPLE_K,
    THETA_N2_K,
    THETA_O2_K,
    X_N2,
    X_O2,
    C0,
    temp_k,
    speed_of_sound,
)


def saturation_pressure_ratio(temp_c: float) -> float:
    """Saturation vapour pressure divided by reference pressure, p_sat/p_r.

    ISO 9613-1 Annex A: ``p_sat/p_r = 10**C`` with
    ``C = -6.8346 (T01/T)^1.261 + 4.6151``.
    """
    t = temp_k(temp_c)
    c = -6.8346 * (T_TRIPLE_K / t) ** 1.261 + 4.6151
    return 10.0 ** c


def molar_humidity(temp_c: float, rh_pct: float,
                   pressure_pa: float = P_REF_PA) -> float:
    """Molar concentration of water vapour, in percent (the standard's ``h``).

    ``h = h_r (p_sat/p_r) / (p_a/p_r)``.  At 20 C / 101.325 kPa the molar
    fraction of saturated air is 2.31 %, i.e. ``h = 2.31`` at RH = 100 %.
    """
    rh = float(np.clip(rh_pct, 0.0, 100.0))
    return rh * saturation_pressure_ratio(temp_c) / (pressure_pa / P_REF_PA)


def relaxation_freq_oxygen(temp_c: float, rh_pct: float,
                           pressure_pa: float = P_REF_PA) -> float:
    """Oxygen vibrational relaxation frequency f_rO, Hz (ISO 9613-1 Table 1)."""
    h = molar_humidity(temp_c, rh_pct, pressure_pa)
    p_rel = pressure_pa / P_REF_PA
    return p_rel * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))


def relaxation_freq_nitrogen(temp_c: float, rh_pct: float,
                             pressure_pa: float = P_REF_PA) -> float:
    """Nitrogen vibrational relaxation frequency f_rN, Hz (ISO 9613-1 Table 1).

    The strong humidity term is why a cold, dry day kills high frequencies over
    hundreds of metres far more than a warm humid one.
    """
    h = molar_humidity(temp_c, rh_pct, pressure_pa)
    p_rel = pressure_pa / P_REF_PA
    t_rel = temp_k(temp_c) / T0_K
    return p_rel * t_rel ** (-1.0 / 3.0) * (
        9.0 + 280.0 * h * math.exp(-4.170 * (t_rel ** (-1.0 / 3.0) - 1.0))
    )


def attenuation_coefficient(freq_hz, temp_c: float = 20.0, rh_pct: float = 50.0,
                            pressure_pa: float = P_REF_PA) -> np.ndarray:
    """Atmospheric absorption coefficient alpha in **dB per metre**.

    Parameters
    ----------
    freq_hz : float or array-like
        Frequency in Hz.  The model is valid 50 Hz .. 10 kHz (the ISO range
        is -20..+50 C and 10..100 % RH; outside that it degrades smoothly).
    temp_c, rh_pct, pressure_pa : ambient conditions.
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    t_k = temp_k(temp_c)
    t_rel = t_k / T0_K
    p_rel = pressure_pa / P_REF_PA
    c = speed_of_sound(temp_c)

    # classical + rotational (ISO 9613-1 eq. A.2). f in Hz, alpha in dB/m.
    alpha_cr = 1.60e-10 * math.sqrt(t_rel) * f**2 / p_rel

    # (alpha*lambda)_max : peak of the relaxation line shape (dimensionless, dB)
    pref = (2.0 * math.pi / 35.0) * (10.0 * math.log10(math.e**2))

    th_o = THETA_O2_K / t_k
    th_n = THETA_N2_K / t_k
    amax_o = pref * X_O2 * th_o**2 * math.exp(-th_o)
    amax_n = pref * X_N2 * th_n**2 * math.exp(-th_n)

    f_ro = relaxation_freq_oxygen(temp_c, rh_pct, pressure_pa)
    f_rn = relaxation_freq_nitrogen(temp_c, rh_pct, pressure_pa)

    # Lorentzian lineshape 2(f/fr) / (1 + (f/fr)^2), alpha = (alpha*lambda)*f/c
    def lineshape(fr: float) -> np.ndarray:
        x = f / fr
        return 2.0 * x / (1.0 + x * x)

    alpha_vib_o = amax_o * (f / c) * lineshape(f_ro)
    alpha_vib_n = amax_n * (f / c) * lineshape(f_rn)

    return alpha_cr + alpha_vib_o + alpha_vib_n


def air_absorption_db(distance_m, freq_hz, temp_c: float = 20.0,
                      rh_pct: float = 50.0,
                      pressure_pa: float = P_REF_PA) -> np.ndarray:
    """Total atmospheric attenuation ``A_atm = alpha(f) * d`` in dB."""
    return attenuation_coefficient(freq_hz, temp_c, rh_pct, pressure_pa) * np.asarray(distance_m)


def air_absorption_response(freqs_hz, distance_m: float, temp_c: float = 20.0,
                            rh_pct: float = 50.0,
                            pressure_pa: float = P_REF_PA) -> np.ndarray:
    """Complex (magnitude-only, zero-phase) transfer function of air absorption.

    Air absorption is a minimum-phase phenomenon; using a zero-phase magnitude
    response over a short impulse response is the accepted engineering
    approximation (the accompanying phase shift is < 1 ms below 10 km).
    Returns linear gain per row of ``freqs_hz``.
    """
    a_db = air_absorption_db(distance_m, np.asarray(freqs_hz, dtype=np.float64),
                             temp_c, rh_pct, pressure_pa)
    return np.power(10.0, -a_db / 20.0)
