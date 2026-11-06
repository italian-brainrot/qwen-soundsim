"""Ground effect and porous-surface reflection.

Outdoors, the receiver sees the direct wave plus a wave reflected off the
ground.  Because the ground is a *porous half-space* the reflection coefficient
is complex and strongly frequency dependent, so direct and reflected waves
interfere.  The result is the classic ground-effect signature: a deep dip whose
position scales with the path geometry, and broadband attenuation above roughly
1 kHz over grass or soil.

The porous medium is modelled as an equivalent fluid with the **Johnson-
Champoux-Allard (JCA)** model, parametrised by flow resistivity sigma alone via
Allard's empirical relations for tortuosity and thermal permeability.  This is
noticeably more accurate than Delany-Bazley outside a narrow band of flow
resistivities and has no unphysical behaviour at low sigma.

    alpha_inf = 1 + 229.5 / sigma^0.745          (Allard)
    Lambda    = 8 alpha_inf eta_air / sigma      (viscous characteristic length)
    Lambda'   = 2 Lambda                         (thermal permeability)

with the standard dynamic density and bulk modulus of Johnson-Champoux-Allard
(-Bidet).  Air viscosity eta = 1.84e-5 Pa s, Prandtl number 0.72.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import air_density, speed_of_sound
from .materials import Ground

ETA_AIR = 1.84e-5          # dynamic viscosity of air, Pa s
P_ATM = 101_325.0        # static pressure entering the bulk modulus, Pa
PRANDTL = 0.72
GAMMA = 1.4


def jca_parameters(g: Ground) -> dict[str, float]:
    """Five JCA parameters from the flow resistivity alone.

    Closure relations (Allard & Atalla, *Introduction to the Acoustic Properties
    of Porous Media*, ch. 1), which is what makes a single measured quantity --
    flow resistivity -- enough to describe the ground:

        alpha_inf = 1 + 229.5 / sigma^0.745                 (tortuosity)
        Lambda    = (2 alpha_inf / phi) * sqrt(2 eta/sigma) (viscous length)
        Lambda'   = 2 Lambda                                (thermal length)

    For a glass-wool-like sigma = 1e4 this gives Lambda = 1.6e-4 m against
    Allard's benchmark (inverted) value 1.09e-4 m -- the right order and the
    right trend, which is all the ground model needs.
    """
    sigma = max(g.flow_resistivity, 1.0)
    alpha_inf = (1.0 + 229.5 / sigma**0.745) * max(g.structure_factor, 1.0)
    phi = float(np.clip(g.porosity, 0.3, 0.999))
    lam = (2.0 * alpha_inf / phi) * math.sqrt(2.0 * ETA_AIR / sigma)
    return {
        "sigma": sigma,
        "alpha_inf": alpha_inf,
        "porosity": phi,
        "lambda_visc": lam,
        "lambda_thermal": 2.0 * lam,
    }


def equivalent_fluid(freqs_hz, g: Ground, temp_c: float = 20.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rho_eq, K_eq, c_eq_complex)`` of the porous half-space.

    Johnson-Champoux-Allard equivalent-fluid model.  Note the structure: the
    viscous drag enters the dynamic density *additively*
    (``1 + phi sigma/(j w alpha_inf rho0) * ...``), so the low-frequency (Darcy)
    limit gives ``rho_eq -> sigma/(j omega phi)`` and a characteristic impedance
    ``sqrt(gamma P0 sigma/(j omega))`` that depends on flow resistivity alone --
    exactly the physics of a resistive frame.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    w = 2.0 * np.pi * f
    rho0 = air_density(temp_c)
    c0 = speed_of_sound(temp_c)
    p = jca_parameters(g)
    sigma, alpha_inf, phi = p["sigma"], p["alpha_inf"], p["porosity"]
    lam, lam_t = p["lambda_visc"], p["lambda_thermal"]

    # --- dynamic (efficient) density ---
    arg_rho = 1.0 + 1j * 4.0 * w * rho0 * ETA_AIR * alpha_inf**2 / (sigma**2 * phi**2 * lam**2)
    rho_eff = rho0 * alpha_inf * (
        1.0 + (sigma * phi / (1j * w * rho0 * alpha_inf)) * np.sqrt(arg_rho)
    )

    # --- dynamic bulk modulus ---
    arg_k = 1.0 + 1j * w * PRANDTL * rho0 * lam_t**2 / (16.0 * ETA_AIR)
    drag_k = (1.0 - 1j * 8.0 * ETA_AIR / (w * PRANDTL * rho0 * lam_t**2)) * np.sqrt(arg_k)
    K_eff = GAMMA * P_ATM / (GAMMA - (GAMMA - 1.0) / drag_k)

    # "efficient" (per pore volume) -> "equivalent" (per total volume)
    rho_eq = rho_eff / phi
    K_eq = K_eff / phi
    c_eq = np.sqrt(K_eq / rho_eq)
    return rho_eq, K_eq, c_eq


def characteristic_impedance(freqs_hz, g: Ground, temp_c: float = 20.0) -> np.ndarray:
    """Characteristic impedance of the equivalent fluid, Pa s/m."""
    rho_eq, K_eq, _ = equivalent_fluid(freqs_hz, g, temp_c)
    return np.sqrt(rho_eq * K_eq)


def delany_bazley(freqs_hz, g: Ground, temp_c: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Delany-Bazley equivalent fluid (an independent empirical model).

    Kept as a cross-check: JCA and D&B agree to within ~10 % for
    ``0.1 < rho0 f / sigma < 10``; D&B degrades badly below that, JCA does not.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    c0 = speed_of_sound(temp_c)
    rho0 = air_density(temp_c)
    X = np.clip(rho0 * f / max(g.flow_resistivity, 1.0), 1e-4, 1e4)
    z_c = rho0 * c0 * (1.0 + 0.0571 * X**-0.754 - 1j * 0.087 * X**-0.732)
    k_c = (2.0 * np.pi * f / c0) * (1.0 + 0.0978 * X**-0.700 - 1j * 0.189 * X**-0.595)
    return z_c, k_c


def reflection_coefficient(freqs_hz, g: Ground, incidence_rad, temp_c: float = 20.0) -> np.ndarray:
    """Plane-wave pressure reflection coefficient of a porous half-space.

    ``incidence_rad`` is measured from the surface normal (90 deg = grazing).
    A rigid ground gives ``exp(-j w d/c)``-free +1; a pressure-release surface
    gives -1; porous grounds sit in between with a strong phase rotation which is
    what produces the ground-effect dip.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    rho0 = air_density(temp_c)
    c0 = speed_of_sound(temp_c)
    if g.flow_resistivity > 1e8:                      # effectively rigid
        return np.ones_like(f, dtype=np.complex128)
    rho_e, K_e, c_e = equivalent_fluid(f, g, temp_c)
    z_c = np.sqrt(rho_e * K_e)                        # characteristic impedance
    theta_i = np.atleast_1d(np.asarray(incidence_rad, dtype=np.float64))
    sin_t = np.clip(np.sin(theta_i) * c_e / c0, -0.999, 0.999)
    cos_t = np.sqrt(1.0 - sin_t**2)
    z_i = rho0 * c0 / np.cos(theta_i)
    z_t = z_c / cos_t
    return (z_t - z_i) / (z_t + z_i)


def ground_reflection_gain_db(freqs_hz, g: Ground, grazing_angle_rad: float,
                              temp_c: float = 20.0) -> np.ndarray:
    """Level change of the reflected wave relative to a rigid-ground reflection."""
    r = reflection_coefficient(freqs_hz, g, np.pi / 2.0 - grazing_angle_rad, temp_c)
    return 20.0 * np.log10(np.maximum(np.abs(r), 1e-12))


def flat_ground_factor(freqs_hz, g: Ground, source_h_m: float, receiver_h_m: float,
                       distance_m: float, temp_c: float = 20.0) -> np.ndarray:
    """Flat-earth ground attenuation in dB, relative to **free field only**.

    Positive values mean the direct + ground-reflected field is weaker than the
    direct field alone (the usual meaning of "ground attenuation", and the
    quantity ISO 9613-2 calls A_gr).  Direct + complex-image interference is
    exactly the same physics the engine puts into the outdoor image path.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    c0 = speed_of_sound(temp_c)
    k = 2.0 * np.pi * f / c0
    d_dir = np.hypot(distance_m, receiver_h_m - source_h_m)
    d_img = np.hypot(distance_m, receiver_h_m + source_h_m)
    graze = np.arctan2(source_h_m + receiver_h_m, distance_m)
    r = reflection_coefficient(f, g, np.pi / 2.0 - graze, temp_c)
    direct = np.exp(-1j * k * d_dir) / d_dir
    total = direct + r * (np.exp(-1j * k * d_img) / d_img)
    return 20.0 * np.log10(np.abs(total) / np.abs(direct) + 1e-30)
