"""Audit pass 1: regression tests for the issues listed in AUDIT.md.

Each test here is written against an *independent* physical requirement -- a
transfer function must be causal, a screen that the ray flies over must not
change the level, a knob that only blurs interference must not change the
energy, a unit called "centiseconds" must be centiseconds -- so that it fails
for the reason described in AUDIT.md and stays green once the cause is fixed.
"""
from __future__ import annotations

import numpy as np

from soundsim.acoustics import modal, rooms
from soundsim.acoustics.constants import speed_of_sound
from soundsim.acoustics.materials import Absorber
from soundsim.engine import presets, renderer
from soundsim.engine.paths import build_paths
from soundsim.engine.scene import Scene


# ---------------------------------------------------------------------------
# 1. Screens: the engine must know whether the screen is in the way at all
#    (AUDIT.md pass 1, finding "a screen that misses the ray still attenuates")
# ---------------------------------------------------------------------------
def _park_with_screen(height: float) -> Scene:
    sc = presets.far_field_park(distance_m=300.0, ground="asphalt_sealed",
                                barrier_height=height)
    return sc


def _received_level(sc: Scene) -> float:
    """Total received energy of the path set, dB re 1 (incoherent sum over
    paths == the energy the synthesised response carries)."""
    paths, _ = build_paths(sc)
    return 10.0 * np.log10(sum(float(np.sum(np.abs(p.gains) ** 2))
                               for p in paths))


def test_a_screen_below_the_line_of_sight_changes_nothing():
    """Source 2 m up, receiver 1.6 m up, 300 m apart: the ray passes the middle
    of the path at 1.8 m.  A 1 m screen is not in the way -- ISO 9613-2 puts the
    receiver in the "light zone" there and gives it zero attenuation."""
    # geometry sanity: the module's own shadow test agrees it is not shadowed
    from soundsim.acoustics import diffraction
    assert not diffraction.is_shadowed((0, 0, 2.0), (150, 0, 1.0), (300, 0, 1.6))

    open_field = _received_level(_park_with_screen(0.0))
    for h in (0.2, 0.5, 1.0, 1.5):
        got = _received_level(_park_with_screen(h))
        assert abs(got - open_field) < 0.25, (
            f"a {h} m screen with clear line of sight moved the level by "
            f"{got - open_field:+.2f} dB")


def test_a_screen_that_shadows_the_ray_makes_the_scene_quieter():
    """Maekawa attenuation is an *insertion loss*: a screen tall enough to block
    the ray must remove energy, never add it."""
    open_field = _received_level(_park_with_screen(0.0))
    shadowed = _received_level(_park_with_screen(8.0))
    assert shadowed < open_field - 5.0, (
        f"an 8 m screen in full shadow gave {shadowed - open_field:+.2f} dB")


# ---------------------------------------------------------------------------
# 2. Ground/turbulence: spreading the reflection over a fan of delays models
#    the loss of coherence, it must not throw energy away
#    (AUDIT.md pass 1, finding "turbulence splits the ground ray and loses 1 dB")
# ---------------------------------------------------------------------------
def _outdoor_level_at(turbulence: float) -> float:
    sc = presets.far_field_park(ground="grass_short")
    sc.weather.turbulence = turbulence
    paths, _ = build_paths(sc)
    return 10.0 * np.log10(sum(float(np.sum(np.abs(p.gains) ** 2))
                               for p in paths))


def test_turbulence_blurs_the_ground_comb_without_losing_energy():
    """Turbulence randomises the *relative phase* of the two rays: the pattern
    is smeared, the total energy is conserved."""
    calm = _outdoor_level_at(0.0)
    for turb in (0.1, 0.35, 0.5, 1.0):
        got = _outdoor_level_at(turb)
        assert abs(got - calm) < 0.25, (
            f"turbulence={turb} changed the received energy by {got - calm:+.2f} dB")


def test_turbulence_reduces_the_depth_of_the_ground_interference():
    """The point of the knob: the band balance gets smoother while its average
    stays where it was."""
    def balance(turb):
        sc = presets.far_field_park(ground="grass_short")
        sc.weather.turbulence = turb
        m = renderer.render(sc, 48_000).metrics
        b = np.asarray(m["band_response_db"])
        sel = (np.asarray(m["bands_hz"]) >= 250.0) & (np.asarray(m["bands_hz"]) <= 4000.0)
        return b[sel]

    calm = balance(0.0)
    rough = balance(1.0)
    assert np.ptp(rough) < np.ptp(calm) - 2.0, (
        f"comb depth {np.ptp(calm):.1f} dB calm vs {np.ptp(rough):.1f} dB turbulent")
    assert abs(np.mean(rough) - np.mean(calm)) < 1.5


# ---------------------------------------------------------------------------
# 3. The modal transfer function must be a transfer function: something the
#    synthesis (np.fft.irfft) turns into a *causal* response.
#    (AUDIT.md pass 1, finding "the modal band is inserted with a 90 deg phase
#     slip, so its impulse response comes back time-mirrored")
# ---------------------------------------------------------------------------
def _modal_probe(dims, src, dst, f_max_modes):
    """Evaluate modal_transfer on a full rfft grid; return (H, freqs, ir)."""
    fs = 48_000
    n = 1 << 15
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    alpha = Absorber("soft", (0.2, 0.2, 0.2, 0.2, 0.2, 0.2))
    h = modal.modal_transfer(dims, src, dst, freqs, [alpha, alpha, alpha],
                             f_max=f_max_modes)
    return h, freqs, np.fft.irfft(h, n=n), fs, n


def test_modal_transfer_is_the_spectrum_of_a_real_signal():
    """Whatever ``modal_transfer`` returns is written straight into the bins of
    an inverse real FFT.  A spectrum can only be synthesised that way if it is
    the positive-frequency half of a conjugate-symmetric spectrum -- in
    particular its value at DC must be real, because that is where the synthesis
    reads the mean of the response."""
    dims, src, dst = (4.0, 3.0, 2.5), (0.8, 1.4, 1.0), (2.8, 1.4, 1.2)
    h, freqs, ir, fs, n = _modal_probe(dims, src, dst, 400.0)

    h0 = complex(h[0])
    assert abs(h0.imag) <= 0.1 * abs(h0), (
        f"the DC value of the modal transfer function is {h0:.4g}: the synthesis "
        f"can only use its real part, so {abs(h0.imag) / abs(h0) * 100:.0f} % of "
        f"what the model computes at DC is silently thrown away")



# ---------------------------------------------------------------------------
# 4. Below the first eigenfrequency a closed box cannot hold a pressure
#    gradient: the bass is stiff and nearly uniform, not a 1/r free field.
#    (AUDIT.md pass 1, finding "the modal sum degenerates to free field at low f")
# ---------------------------------------------------------------------------
def test_below_the_first_mode_the_room_is_not_free_field():
    dims, src = (3.4, 2.8, 2.4), (0.8, 1.4, 1.0)
    c = speed_of_sound()
    f_first = c / (2.0 * max(dims))                # 50.5 Hz for this box
    f_probe = 0.3 * f_first                        # lambda ~ 8 x the room
    alpha = Absorber("hard", (0.05, 0.05, 0.05, 0.05, 0.05, 0.05))

    seats = [(0.9, 1.4, 1.2), (2.8, 1.4, 1.2), (1.7, 1.4, 1.2),
             (1.7, 0.4, 2.0), (1.7, 2.4, 0.4)]
    h_modal = np.array([abs(modal.modal_transfer(dims, src, seat, [f_probe],
                                                 [alpha] * 3, f_max=800.0))[0]
                        for seat in seats])
    h_free = np.array([1.0 / np.linalg.norm(np.asarray(seat) - np.asarray(src))
                       for seat in seats])

    spread_modal = 20 * np.log10(h_modal.max() / h_modal.min())
    spread_free = 20 * np.log10(h_free.max() / h_free.min())
    assert spread_modal < spread_free - 3.0, (
        f"seat-to-seat spread of the modal model at {f_probe:.1f} Hz is "
        f"{spread_modal:.1f} dB against free field's {spread_free:.1f} dB -- "
        f"the walls are doing nothing")


# ---------------------------------------------------------------------------
# 5. Time strength: "cents" are centiseconds, and TS is bounded by the decay
#    time of the room it is measured on.
#    (AUDIT.md pass 1, finding "time_strength_cents returns milliseconds")
# ---------------------------------------------------------------------------
def test_time_strength_is_reported_in_centiseconds():
    """ISO 3382 defines TS as the centroid of the -10..-25 dB range of the decay
    curve, expressed in *centiseconds* after 50 ms -- which is what the name,
    the docstring and the unit of every published TS figure all say."""
    fs = 48_000
    t60 = 0.50
    n = int(3 * fs)
    t = np.arange(n) / fs
    ir = np.sin(2 * np.pi * 1000.0 * t) * np.power(10.0, -3.0 * t / t60)

    # the same centroid, computed here in centiseconds, straight from ISO 3382
    tc, db = rooms.decay_curve_db(ir, fs)
    m = (db <= -10.0) & (db >= -25.0)
    w = ir[m] ** 2
    expected_cs = (float(np.sum(tc[m] * w) / np.sum(w)) - 0.050) * 100.0

    ts = rooms.time_strength_cents(ir, fs)
    assert abs(ts - expected_cs) < 0.5, (
        f"TS = {ts:.1f} but the centroid is {expected_cs:.1f} centiseconds -- "
        f"the function reports {ts / max(expected_cs, 1e-9):.0f}x too large a number")
