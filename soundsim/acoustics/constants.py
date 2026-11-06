"""Physical constants of the medium and reference quantities.

All acoustical quantities in :mod:`soundsim.acoustics` are derived from these
constants so that a change in, e.g., temperature propagates consistently through
spreading losses, wavelengths, air absorption and panel coincidence.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Reference state (dry air at sea level, 20 C)
# --------------------------------------------------------------------------
T0_K = 293.15                # reference temperature, K (20 C)
T_TRIPLE_K = 273.16          # triple-point isotherm, K
P_REF_PA = 101_325.0         # reference ambient pressure, Pa
RHO0 = 1.2041                # dry-air density at 20 C / 101.325 kPa, kg/m^3
C0 = 343.2                   # speed of sound at 20 C, m/s
Z0 = 413.0                   # characteristic specific acoustic impedance, Pa s/m
GAMMA_AIR = 1.4

# Gas composition (dry air, mole fractions) used by the molecular-relaxation
# absorption model of ISO 9613-1 Annex A.
X_O2 = 0.209
X_N2 = 0.781

# Classical rotational/vibrational relaxation temperatures of ISO 9613-1 Annex A
THETA_O2_K = 2239.1
THETA_N2_K = 3352.0

# --------------------------------------------------------------------------
# Perceptual / psychoacoustic reference quantities
# --------------------------------------------------------------------------
P_REF_RMS = 20e-6            # threshold of hearing at 1 kHz, Pa
I_REF = 1e-12                # reference sound intensity, W/m^2
P_REF_DBFS = 1.0             # full-scale sine peak in a float32 sample stream

# A-weighting is defined by these corner frequencies (IEC 61672-1)
A_WT_F1 = 20.598997
A_WT_F2 = 107.6526
A_WT_F3 = 737.862
A_WT_F4 = 12194.0
A_WT_1K = 1.9993328558548801  # value of the unnormalised polynomial at 1 kHz


def temp_k(temp_c: float) -> float:
    """Celsius -> Kelvin."""
    return temp_c + 273.15


def speed_of_sound(temp_c: float = 20.0) -> float:
    """Speed of sound in dry air, m/s.

    ``c = 343.2 * sqrt(T / T0)`` (ISO 9613-1 eq. 3).  The full expression
    ``sqrt(gamma R T / M)`` agrees to within 0.1 % over the modeled range.
    """
    return C0 * math.sqrt(temp_k(temp_c) / T0_K)


def air_density(temp_c: float = 20.0, pressure_pa: float = P_REF_PA) -> float:
    """Ideal-gas density of dry air, kg/m^3."""
    return pressure_pa * 0.0289647 / (8.31446 * temp_k(temp_c))


def specific_impedance(temp_c: float = 20.0, pressure_pa: float = P_REF_PA) -> float:
    """Characteristic specific acoustic impedance rho*c, Pa s/m."""
    return air_density(temp_c, pressure_pa) * speed_of_sound(temp_c)


def wavelength(freq_hz: float, temp_c: float = 20.0) -> float:
    """Wavelength in air, m."""
    return speed_of_sound(temp_c) / freq_hz


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-300))


def power_db_to_lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


def spl_to_rms_pa(spl_db: float) -> float:
    """Sound-pressure level -> rms pressure, Pa."""
    return P_REF_RMS * power_db_to_lin(spl_db)


def rms_pa_to_spl(p_rms: float) -> float:
    return 10.0 * math.log10(max(p_rms, 1e-300) ** 2 / P_REF_RMS**2)


def band_edges(center_hz: float, fraction: float = 1.0) -> tuple[float, float]:
    """Lower/upper band edges for a `fraction`-octave band around `center_hz`."""
    ratio = 2.0 ** (fraction / 2.0)
    return center_hz / ratio, center_hz * ratio


# Standard 1/1-octave band centre frequencies (IEC 61260), 31.5 Hz .. 16 kHz
OCTAVE_CENTRES = [
    31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
]
# 1/3-octave centres, 50 Hz .. 20 kHz
THIRD_OCTAVE_CENTRES = [
    50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
    5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0,
]
