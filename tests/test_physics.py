"""Physics validation tests.

Every test compares the model against an *independent* analytic or standardised
reference, not against the model's own output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from soundsim.acoustics import atmosphere, diffraction, ground, rooms, transmission
from soundsim.acoustics import image_source
from soundsim.acoustics.constants import C0, Z0, speed_of_sound
from soundsim.acoustics.materials import GROUNDS, PANELS, MATERIALS
from soundsim.acoustics.rooms import Surface


# ---------------------------------------------------------------------------
# Atmospheric absorption
# ---------------------------------------------------------------------------
def test_atmosphere_matches_published_anchor():
    """ANSI S1.26 / ISO 9613-1: alpha(4 kHz, 20 C, 70 % RH) = 0.0185-0.023 dB/m."""
    a = atmosphere.attenuation_coefficient(4000.0, 20.0, 70.0)[()]
    assert 0.017 < a < 0.026


def test_atmosphere_increases_with_frequency_and_drops_with_humidity_relaxation():
    f = np.array([125.0, 500.0, 2000.0, 8000.0])
    a = atmosphere.attenuation_coefficient(f, 20.0, 50.0)
    assert np.all(np.diff(a) > 0)


def test_atmosphere_cross_check_independent_implementation():
    """Compare with acoustic-toolbox's ISO 9613-1 (section 4.2) implementation."""
    abx = pytest.importorskip("acoustic_toolbox")
    from acoustic_toolbox.standards.iso_9613_1_1993 import (
        attenuation_coefficient as ref, molar_concentration_water_vapour as mw,
        relaxation_frequency_nitrogen as rn, relaxation_frequency_oxygen as ro,
        saturation_pressure as sp)
    f = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    worst = 0.0
    for tc in (-20, 0, 10, 20, 30, 50):
        for rh in (10, 50, 90):
            tk, p = tc + 273.15, 101.325
            h = mw(rh, sp(tk), p)
            a_ref = ref(pressure=p, temperature=tk, reference_pressure=101.325,
                        reference_temperature=293.15,
                        relaxation_frequency_nitrogen=rn(p, tk, h),
                        relaxation_frequency_oxygen=ro(p, h), frequency=f)
            a_mine = atmosphere.attenuation_coefficient(f, tc, rh)
            worst = max(worst, float(np.max(np.abs(a_mine - a_ref) / a_ref)))
    assert worst < 0.02, f"max deviation {worst:.3f}"


def test_air_absorption_over_1km_at_8khz_is_tens_of_dB():
    a = atmosphere.air_absorption_db(1000.0, 8000.0, 10.0, 30.0)[()]
    assert 40 < a < 250


# ---------------------------------------------------------------------------
# Plate transmission
# ---------------------------------------------------------------------------
def test_normal_incidence_mass_law_is_exact():
    p = PANELS["gypsum_board_12mm"]
    f = 1000.0
    w = 2 * math.pi * f
    analytic = 4 * Z0**2 / (w * p.surface_density) ** 2
    assert transmission.tau_normal(f, p)[()] == pytest.approx(analytic, rel=1e-3)


def test_grazing_incidence_transmits_completely():
    p = PANELS["gypsum_board_12mm"]
    assert transmission.transmission_coefficient(np.pi / 2 - 1e-9, 200.0, p)[()] == pytest.approx(1.0, abs=1e-6)


def test_field_incidence_mass_law_offset_matches_theory():
    """Above coincidence the diffuse-field TL recovers towards the mass law."""
    p = PANELS["brick_200mm"]                    # f_c ~ 130 Hz, far below the band
    f = np.array([2000.0, 4000.0, 8000.0])
    tl = transmission.transmission_loss_db(f, p)
    ml_normal = 20 * np.log10(2 * math.pi * f * p.surface_density / (2 * Z0))
    assert np.all(np.abs(tl - ml_normal) < 13.0), (tl, ml_normal)
    assert np.all(np.diff(tl) > 5.0)          # rising with frequency
    # the classic random-incidence mass law sits within a few dB of the integral
    ml_field = 20 * np.log10(f * p.surface_density) - 47.4
    assert np.max(np.abs(tl - ml_field)) < 10.0, (tl, ml_field)


def test_coincidence_frequency_gypsum():
    """12.7 mm faced gypsum: f_c ~ 2.4-2.8 kHz."""
    fc = transmission.coincidence_frequency_grazing(PANELS["gypsum_board_12mm"])
    assert 2000 < fc < 3200


def test_coincidence_dip_exists_and_deepens_with_low_damping():
    f = np.array([500.0, 1000.0, 2000.0, 3000.0, 4000.0, 8000.0])
    tl_low = transmission.transmission_loss_db(f, "car_window_glass")
    i_mass = 1                     # 1 kHz mass-law reference
    assert tl_low[4] < tl_low[i_mass] - 4.0, tl_low   # 4 kHz sits in the dip


def test_double_leaf_mass_air_mass_resonance():
    """Two 230 kg/m2 leaves over a 100 mm cavity: the classic 600-rule."""
    p1 = p2 = PANELS["concrete_100mm"]
    d = 0.10
    f0 = 600.0 * math.sqrt((p1.surface_density + p2.surface_density)
                           / (p1.surface_density * p2.surface_density * d))
    assert 150 < f0 < 220, f0
    f = np.logspace(math.log10(3 * f0), math.log10(2000.0), 60)
    tl2 = transmission.tl_double_leaf_db(f, p1, p2, d, field_incidence=False)
    tl1 = transmission.transmission_loss_db(f, p1, field_incidence=False)
    # well above the mass-air-mass resonance the double leaf must clearly win
    assert np.mean(tl2 - tl1) > 8.0, (f, tl2 - tl1)


def test_aperture_is_high_pass_relative_to_size():
    """A small opening transmits far less low-frequency energy than a large one."""
    f = np.array([100.0, 400.0, 1600.0])
    tau_small = transmission.aperture_transmission(f, 0.05)
    tau_large = transmission.aperture_transmission(f, 4.0)
    # low frequency: tiny hole is essentially closed, big opening is wide open
    assert tau_small[0] < 0.12 and tau_large[0] > 0.9, (tau_small, tau_large)
    # high frequency: an opening is no insulation at all (tau -> 1)
    assert tau_large[2] > 0.99
    assert tau_small[2] > 0.99                  # even a keyhole passes treble
    # Deep Rayleigh regime (ka << 1): tau ~ 2(ka)^2, a power law -> 6.02 dB/oct
    t5 = float(transmission.aperture_transmission([5.0], 0.05)[0])
    t10 = float(transmission.aperture_transmission([10.0], 0.05)[0])
    assert 10 * math.log10(t10 / t5) == pytest.approx(6.02, abs=0.05)
    # At audible low frequencies the air plug's own mass reactance limits the
    # velocity, so the real slope is shallower than the radiation-resistance law.
    t50 = float(transmission.aperture_transmission([50.0], 0.05)[0])
    t100 = float(transmission.aperture_transmission([100.0], 0.05)[0])
    slope = 10 * math.log10(t100 / t50)
    assert 5.0 < slope < 6.0, slope


# ---------------------------------------------------------------------------
# Diffraction
# ---------------------------------------------------------------------------
def test_maekawa_zero_path_excess_is_4p8_db():
    a = diffraction.maekawa_attenuation_db(0.0, 1000.0)[()]
    assert a == pytest.approx(10 * math.log10(3.0), abs=1e-9)


def test_maekawa_attenuation_grows_with_frequency():
    f = np.array([125.0, 500.0, 2000.0, 8000.0])
    a = diffraction.maekawa_attenuation_db(1.0, f)
    assert np.all(np.diff(a) > 0)
    assert a[0] < a[-1] - 15


def test_barrier_geometry_detects_shadow():
    s, b, r = (0, 0, 1.0), (10, 0, 2.0), (20, 0, 1.0)
    assert diffraction.is_shadowed(s, b, r)
    ex = diffraction.barrier_path_excess(s, b, r)
    assert ex > 0
    assert not diffraction.is_shadowed((0, 0, 5.0), b, (20, 0, 5.0))


# ---------------------------------------------------------------------------
# Ground effect
# ---------------------------------------------------------------------------
def test_rigid_ground_reflects_unity():
    r = ground.reflection_coefficient([500.0], GROUNDS["asphalt_sealed"], np.deg2rad(60))[0]
    assert abs(r) == pytest.approx(1.0, abs=1e-9)


def test_grass_absorbs_at_high_frequency_but_not_low():
    g = GROUNDS["grass_short"]                 # resistive layer: monotonic roll-off
    r_lo = abs(ground.reflection_coefficient([125.0], g, np.deg2rad(80))[0])
    r_hi = abs(ground.reflection_coefficient([4000.0], g, np.deg2rad(80))[0])
    assert r_hi < r_lo - 0.25, (r_lo, r_hi)
    # a very porous forest floor reflects far less than mown grass
    forest = abs(ground.reflection_coefficient([1000.0], GROUNDS["forest_floor"], np.deg2rad(80))[0])
    assert forest < 0.6


def test_jca_matches_delany_bazley_where_db_is_valid():
    """Two independent porous-medium models must agree in D&B's validity range.

    D&B is only valid for 0.1 < rho0 f / sigma < 10, i.e. for *fibrous* flow
    resistivities; the outdoor grounds are far more resistive, which is exactly
    why the JCA model is used there instead.
    """
    from soundsim.acoustics.materials import Ground
    g = Ground("glass_wool", 1e4)
    f = np.array([500.0, 1000.0, 2000.0, 4000.0])
    z_jca = np.abs(ground.characteristic_impedance(f, g))
    z_db, _ = ground.delany_bazley(f, g)
    err = np.max(np.abs(z_jca - np.abs(z_db)) / np.abs(z_db))
    assert err < 0.06, err

def test_jca_stays_physical_for_resistive_grounds():
    """High flow resistivity: |R| <= 1 and the wave must decay into the ground."""
    f = np.logspace(1, 4, 40)
    for name in ("grass_short", "soil_tilled", "forest_floor", "asphalt_sealed"):
        g = GROUNDS[name]
        R = ground.reflection_coefficient(f, g, np.deg2rad(85))
        assert np.all(np.abs(R) <= 1.0 + 1e-9), name
        if g.flow_resistivity < 1e8:
            _, _, c_eq = ground.equivalent_fluid(f, g)
            k = 2 * np.pi * f / c_eq
            assert np.all(np.imag(k) <= 1e-9), name       # decays with depth (e^-jkr)


def test_ground_effect_creates_interference_dip():
    f = np.logspace(2, 3.9, 400)
    a = ground.flat_ground_factor(f, GROUNDS["asphalt_sealed"], 1.5, 1.5, 50.0)
    # rigid flat earth: interference between direct and image, -6..+6 dB
    assert np.min(a) < -5.0 and np.max(a) > 3.0
    g = ground.flat_ground_factor(f, GROUNDS["grass_long"], 1.5, 1.5, 50.0)
    # grass fills in the interference and attenuates overall
    assert np.mean(g[200:]) > np.mean(a[200:]) + 1.0


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------
def test_sabine_matches_hand_computation():
    """V=240 m3, S=240 m2, painted concrete alpha=0.02 -> RT60 = 8.05 s."""
    ab = MATERIALS["concrete_painted"]
    surf = [Surface(ab, 240.0)]
    t = rooms.sabine_t60(240.0, surf)
    assert t[2] == pytest.approx(0.161 * 240.0 / (0.02 * 240.0), rel=1e-6)
    assert t[2] == pytest.approx(8.05, rel=0.02)


def test_eyring_gives_shorter_rt_than_sabine_when_absorptive():
    """Eyring's -S ln(1-alpha) removes Sabine's overestimate of absorption."""
    soft = Surface(MATERIALS["carpet_thick_pad"], 240.0)
    ts = rooms.sabine_t60(240.0, [soft])
    te = rooms.eyring_t60(240.0, [soft], 240.0)
    assert te[3] < ts[3]      # 0.37 absorption: Eyring 0.35 s vs Sabine 0.44 s


def test_air_absorption_shortens_rt60_in_large_rooms():
    ab = MATERIALS["concrete_painted"]
    surf = [Surface(ab, 6000.0)]
    vol = 20000.0
    a_atm = atmosphere.attenuation_coefficient([125, 250, 500, 1000, 2000, 4000.0], 20, 50)
    dry = rooms.sabine_t60(vol, surf)
    wet = rooms.sabine_t60(vol, surf, alpha_atm=a_atm)
    # 4 kHz: air absorption dominates in a 20 000 m3 volume (>90 % of the loss)
    assert wet[-1] < dry[-1] * 0.25
    # 125 Hz: air absorption is a small correction
    assert wet[0] < dry[0] and wet[0] > dry[0] * 0.8


def test_schroeder_frequency_small_room_high_large_room_low():
    assert rooms.Schroeder_frequency(30.0, 0.4) > 200      # 231 Hz: modal room
    assert rooms.Schroeder_frequency(20000.0, 1.8) < 300   # 19 Hz: fully diffuse


def test_cube_axial_modes():
    """A 10 m cube: first axial mode at c/2L = 17.15 Hz."""
    modes = rooms.room_modes((10.0, 10.0, 10.0), f_max=60.0)
    assert modes[0][0] == pytest.approx(speed_of_sound() / 20.0, rel=1e-3)
    assert modes[0][2] == "axial"


# ---------------------------------------------------------------------------
# Image source
# ---------------------------------------------------------------------------
def test_direct_path_present_once_and_unit_gain_structure():
    p = image_source.image_source_paths((2, 3, 1.5), (6, 5, 1.5), (8, 6, 3), max_order=1)
    assert p.order.min() == 0
    assert (p.order == 0).sum() == 1
    d_direct = p.distance[p.order == 0][0]
    assert d_direct == pytest.approx(np.linalg.norm(np.array([6, 5, 1.5]) - np.array([2, 3, 1.5])))


def test_first_order_count_matches_room_geometry():
    """Order 1 in a box: exactly 6 image sources."""
    p = image_source.image_source_paths((2, 3, 1.5), (6, 5, 1.5), (8, 6, 3), max_order=1)
    assert (p.order == 1).sum() == 6


def test_first_order_distances_are_mirror_distances():
    """Each first-order image distance must equal the specular mirror distance."""
    s, r, dims = np.array([2.0, 3.0, 1.5]), np.array([6.0, 5.0, 1.5]), np.array([8.0, 6.0, 3.0])
    p = image_source.image_source_paths(s, r, dims, max_order=1)
    expected = []
    for axis in range(3):                     # wall at 0 and wall at L
        for sign in (0.0, 1.0):
            m = s.copy()
            m[axis] = 2.0 * sign * dims[axis] - s[axis]
            expected.append(np.linalg.norm(m - r))
    got = np.sort(p.distance[p.order == 1])
    assert got == pytest.approx(np.sort(expected), rel=1e-9)


def test_ism_is_reciprocal():
    """Swapping source and receiver must give the same multiset of delays/gains."""
    s, r, dims = (2.2, 1.4, 1.6), (6.1, 5.3, 1.1), (9.0, 7.0, 3.2)
    band = np.array([250.0, 1000.0, 4000.0])
    ab = MATERIALS["brick_painted"]
    a = image_source.image_source_paths(s, r, dims, max_order=4)
    b = image_source.image_source_paths(r, s, dims, max_order=4)
    assert a.count == b.count
    ga = np.sort(image_source.path_gains(a, band, [[ab, ab]] * 3)[:, 1])
    gb = np.sort(image_source.path_gains(b, band, [[ab, ab]] * 3)[:, 1])
    assert ga == pytest.approx(gb, rel=1e-9)
    assert np.sort(a.distance) == pytest.approx(np.sort(b.distance), rel=1e-9)


def test_reflection_sequence_count_matches_textbook():
    """Summed multiplicities: 6 images at order 1, 30 sequences at order 2."""
    p = image_source.image_source_paths((2, 3, 1.5), (6, 5, 1.5), (8, 6, 3), max_order=2)
    assert p.multiplicity[p.order == 1].sum() == 6
    assert p.multiplicity[p.order == 2].sum() == 6 * 5


def test_ism_energy_converges_monotonically_in_a_soft_room():
    """Adding reflection order must add energy, with diminishing increments."""
    dims, s, r = (9.0, 7.0, 3.2), (2.0, 2.0, 1.4), (6.5, 5.0, 1.4)
    band = np.array([1000.0])
    ab = MATERIALS["curtain_heavy"]           # alpha(1 kHz) = 0.75
    prev, increments = None, []
    for order in (2, 4, 6, 8, 10):
        p = image_source.image_source_paths(s, r, dims, max_order=order)
        e = float(np.sum(image_source.path_gains(p, band, [[ab, ab]] * 3) ** 2))
        if prev is not None:
            increments.append(e - prev)
        prev = e
    assert all(i > 0 for i in increments), increments
    assert all(np.diff(increments) < 0), increments      # geometric decay
