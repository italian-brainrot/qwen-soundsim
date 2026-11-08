"""Pass 2 (fixer) regression tests: the behaviour each fix was meant to produce.

Written from the physics the fix claims, not from the code that implements it:
a screen the ray misses must leave the level *exactly* as it was, a coherence
split must not create or destroy energy, a closed box below its first mode must
have the stiffness response of a volume of air and no more, and a quantity called
"dB per second" must have that dimension.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
from dataclasses import replace

from soundsim.acoustics import diffraction as diffr
from soundsim.acoustics import modal, rooms, transmission
from soundsim.acoustics.constants import speed_of_sound
from soundsim.acoustics.materials import Absorber
from soundsim.engine import presets, renderer
from soundsim.engine.paths import AIR_AREA_COEF, PathBuilder
from soundsim.engine.rir import Path, decorrelated_burst
from soundsim.engine.scene import Barrier, Outdoors, Scene, Source, Weather

C = speed_of_sound(20.0)


# ---------------------------------------------------------------------------
# 1. Screens.  A screen is only in the way if it is in the way, and what it does
#    to the level is insertion loss on the rays it crosses.  (AUDIT.md F1.)
# ---------------------------------------------------------------------------
def _park(barrier: Barrier | None, distance_m: float = 300.0) -> Scene:
    return Scene(
        name="screen test",
        source=Source(position=(0.0, 0.0, 2.0), spl_at_1m_db=110.0),
        receiver=(distance_m, 0.0, 1.6),
        outdoors=Outdoors(ground="asphalt_sealed"),
        barriers=[] if barrier is None else [barrier],
        weather=Weather(temp_c=20.0, rh_pct=50.0, turbulence=0.0),
        ism_order=0, diffuse_tail=False)


def _received_db(sc: Scene) -> float:
    from soundsim.engine.paths import build_paths
    paths, _ = build_paths(sc)
    return 10.0 * np.log10(sum(float(np.sum(np.abs(p.gains) ** 2)) for p in paths))


def test_a_screen_beside_the_ray_leaves_the_level_exactly_unchanged():
    """300 m apart, the screen 150 m out but 30 m to one side: nothing to diffract.

    Not "a small attenuation" -- *exactly* nothing.  A screen that misses the ray
    must not even change its phase or its delay.
    """
    sc_open = _park(None)
    sc_miss = _park(Barrier(center=(150.0, 30.0, 0.0), width=20.0, height=6.0,
                            along=(0, 1, 0)))
    assert _received_db(sc_miss) == pytest.approx(_received_db(sc_open), abs=1e-9)
    assert not sc_miss.barriers[0].blocks(np.array([0.0, 0.0, 2.0]),
                                         np.array([300.0, 0.0, 1.6]))


def test_a_screen_long_enough_to_cover_the_ray_shields_more_than_one_that_is_not():
    """Sound flanks past the end of a short screen -- and the geometry has to be
    one where that is even possible: with the ray crossing the middle of the
    screen, its length is irrelevant, and a test that puts it there proves nothing.

    So the screen here stands 20 m to one side of the line of sight.  12 m of it
    spans y = 14...26 and the ray, running along y = 0, misses it entirely and is
    diffracted around the near end instead; 60 m spans y = -10...50 and is squarely
    in the way.
    """
    def attn(width):
        sc = _park(Barrier(center=(150.0, 20.0, 0.0), width=width, height=6.0,
                           along=(0, 1, 0)))
        return _received_db(_park(None)) - _received_db(sc)

    flank, covered = attn(12.0), attn(60.0)
    assert covered > 5.0, f"a screen across the ray only buys {covered:.1f} dB"
    assert flank < covered - 5.0, (
        f"a screen the ray misses by 8 m buys {flank:.1f} dB against {covered:.1f} "
        f"dB for one that covers it: flanking past the end is not modelled")


def test_the_two_edges_of_a_flat_top_are_averaged_in_energy():
    """A top of finite width has a near and a far edge; both diffract."""
    ex = np.array([1.0, 6.0])                       # path excess, m
    f = 500.0
    a1, a2 = diffr.maekawa_attenuation_db(ex, f)
    group = diffr.edge_group_attenuation_db(ex, f)
    # equal-weight energy average of the two edge transfers, in dB
    want = -10.0 * np.log10(0.5 * (10 ** (-a1 / 10.0) + 10 ** (-a2 / 10.0)))
    assert float(group) == pytest.approx(float(want), rel=1e-9)
    # and one edge alone is that edge
    assert float(diffr.edge_group_attenuation_db(ex[:1], f)) == pytest.approx(
        float(a1), rel=1e-9)


def test_a_cascade_of_screens_is_the_iso_9613_2_cascade():
    """Two screens: the insertion losses add, less 10 log10 3 per extra screen,
    because each screen's loss is referenced to the field in front of it."""
    f = 1000.0
    a1 = float(diffr.maekawa_attenuation_db(1.0, f))
    a2 = float(diffr.maekawa_attenuation_db(2.0, f))
    want = a1 + a2 - 10.0 * np.log10(3.0)
    assert float(diffr.multi_screen_attenuation_db([1.0, 2.0], f)) == pytest.approx(
        want, rel=1e-9)
    assert float(diffr.series_attenuation_db([a1, a2])) == pytest.approx(want, rel=1e-9)
    # a single screen is itself: no cascade correction
    assert float(diffr.series_attenuation_db([a1])) == pytest.approx(a1, rel=1e-9)


def test_two_edges_of_one_top_do_not_double_the_benefit():
    """A flat top has two edges, and adding their *energies* would hand out 3 dB
    of extra attenuation for free -- as if the screen had two independent paths to
    the receiver.  The group is an average, so a thick top performs like the one
    edge it really presents."""
    def attn(thickness):
        sc = _park(Barrier(center=(150.0, 0.0, 0.0), width=60.0, height=6.0,
                           along=(0, 1, 0), thickness=thickness))
        return _received_db(_park(None)) - _received_db(sc)

    knife, flat = attn(0.0), attn(2.0)
    assert flat < knife + 0.6, (
        f"a 2 m flat top buys {flat - knife:+.1f} dB over a knife edge: the two "
        f"edges of one top are being added, not averaged")


# ---------------------------------------------------------------------------
# 2. Turbulence: a coherence loss between two rays, paid for out of their own
#    energy.  (AUDIT.md F2.)
# ---------------------------------------------------------------------------
def _park_turbulent(turbulence: float) -> list[Path]:
    sc = presets.far_field_park(distance_m=300.0, ground="grass_short")
    sc.weather = replace(sc.weather, turbulence=turbulence)
    from soundsim.engine.paths import build_paths
    paths, _ = build_paths(sc)
    return paths


def test_the_turbulence_split_neither_creates_nor_destroys_energy():
    """|g_coherent|^2 + |g_incoherent|^2 == |g_ground(calm)|^2, band by band.

    The waver moves the interference term; it cannot invent a quarter of the
    energy of the arrival, which is what dividing the replica fan by its width
    used to do.
    """
    calm = {(p.label, p.delay_s): p for p in _park_turbulent(0.0)}
    live = {}
    for p in _park_turbulent(1.0):
        live.setdefault(round(p.delay_s, 9), []).append(p)

    ground_calm = [p for p in calm.values() if p.label == "outdoor:ground"]
    assert ground_calm, "no ground ray to test"
    g0 = ground_calm[0].gains
    pair = None
    for group in live.values():
        labels = {p.label for p in group}
        if "outdoor:ground" in labels and any(not p.coherent for p in group):
            pair = group
    assert pair is not None, "turbulence produced no incoherent companion"
    g_coh = next(p for p in pair if p.label == "outdoor:ground").gains
    g_inc = next(p for p in pair if not p.coherent).gains
    got = np.abs(g_coh) ** 2 + np.abs(g_inc) ** 2
    assert got == pytest.approx(np.abs(g0) ** 2, rel=1e-9)


def test_turbulence_does_not_move_or_invent_an_arrival():
    """Decorrelation changes what the pair sums to, not when it arrives.  (The
    incoherent half of a split ray is a new *path* in the list, at the same delay:
    it is the same arrival carrying its remaining energy, not an extra echo.)"""
    def delays(turb):
        return sorted({round(p.delay_s, 6) for p in _park_turbulent(turb)})

    n_calm = len(_park_turbulent(0.0))
    n_live = len(_park_turbulent(1.0))
    assert n_live > n_calm, "turbulence split nothing, so there is nothing to test"
    assert delays(1.0) == pytest.approx(delays(0.0), abs=1e-6)


def test_an_incoherent_path_is_never_scattered_into_replicas():
    """A replica fan of an arrival that is itself a random phasor sum would be
    a copy of the thing it is meant to decorrelate."""
    rng = np.random.default_rng(1)
    n = 4096
    # Gains are real in this engine by convention: a reflection's phase arrives as
    # a delay (Path.delay_bands), not as a complex amplitude.
    g = np.full(24, 0.1)
    coh = Path(delay_s=0.01, gains=g, label="ray")
    inc = Path(delay_s=0.01, gains=g, label="turbulent", coherent=False)
    from soundsim.engine.rir import scatter_paths
    sl = slice(0, 1)                             # one band of the 24
    out_coh = scatter_paths([coh], 48_000, n, sl)
    out_inc = scatter_paths([inc], 48_000, n, sl)
    assert np.any(out_coh), "the coherent path produced nothing"
    assert not np.any(out_inc), "an incoherent path was scattered into replicas"
    # ...but it is synthesised, in energy, by the incoherent builder
    burst = decorrelated_burst([inc], n, 48_000, sl, rng)
    assert np.any(burst)


def test_a_complex_gain_is_refused_rather_than_quietly_dropped():
    """Every path is deposited as a zero-phase fractional delay, so a complex
    amplitude cannot be represented -- it must arrive as a delay instead, and say
    so rather than lose its phase to a numpy casting error deep in synthesis."""
    from soundsim.engine.rir import scatter_paths
    bad = Path(delay_s=0.01, gains=np.full(8, 0.1j), label="phase I forgot")
    with pytest.raises(ValueError, match="delay_bands"):
        scatter_paths([bad], 48_000, 1024, slice(0, 1))


# ---------------------------------------------------------------------------
# 3. The modal low band: a box of air, on the same scale as the direct sound,
#    with no free 90 degrees and no subsonic runaway.  (AUDIT.md F3, F4.)
# ---------------------------------------------------------------------------
def _cabin():
    sc = presets.make("inside_car")
    return sc, sc.rooms["cabin"]


def test_the_compliance_of_the_air_is_the_closed_form():
    """Well below the first mode a rigid box is a spring: 4 pi / (V k^2).

    Two independent checks of the same statement, and one of the model's honest
    limitations: what the truncated modal sum leaves out is the box's *static*
    image field (see modal_transfer), a spatially smooth term of the order of the
    free field, so the response follows the compliance to a few per cent rather
    than exactly.
    """
    sc, room = _cabin()
    dims = room.dims
    V = float(np.prod(dims))
    f = np.array([4.0, 6.0, 8.0, 12.0])          # first mode is at 39 Hz
    k = 2 * np.pi * f / speed_of_sound(sc.weather.temp_c)
    c = speed_of_sound(sc.weather.temp_c)
    axes = [renderer._BlendedAbsorber(a, b) for a, b in room.absorbers_by_axis()]
    h = modal.modal_transfer(dims, room.local(sc.source.position),
                             room.local(sc.receiver), f, axes,
                             temp_c=sc.weather.temp_c, f_max=200.0)
    # The uniform mode's own term: phi_0^2 = 1/V, k_0 = 0, damped at the Sabine
    # rate of a room whose walls are pressed equally on every side.
    areas = 2.0 * np.array([dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]])
    a_axis = []
    for west, east in room.absorbers_by_axis():
        a_axis.append(float(np.mean([np.atleast_1d(west.alpha_at(f))[0],
                                     np.atleast_1d(east.alpha_at(f))[0]])))
    a_0 = float(np.dot(a_axis, areas) / areas.sum())
    sigma_0 = c * a_0 * areas.sum() / (4.0 * V)
    want = 4.0 * np.pi / V / np.abs(-k ** 2 + 1j * k * sigma_0 / c)
    assert np.abs(h) == pytest.approx(want, rel=0.15)
    # The *shape* is the spring's, which no free field can imitate: the response
    # rises towards DC at about 12 dB per octave (a shade more, because the static
    # image field subtracts a constant), where 1/r is flat.
    rise = 20.0 * np.log10(np.abs(h[1]) / np.abs(h[3]))          # 6 Hz vs 12 Hz
    assert 5.0 < rise < 14.0, f"bass rises by only {rise:.1f} dB per octave"
    # and it is the response of a sealed box, not of a field in open air: 20 dB
    # above what 1/r would give at the same distance
    r = float(np.linalg.norm(np.array(room.local(sc.receiver))
                             - np.array(room.local(sc.source.position))))
    assert 20.0 * np.log10(np.abs(h[1]) * r) > 15.0


def test_the_modal_low_band_does_not_flood_the_render_with_subsonics():
    """A rigid box has a stiffness resonance at DC.  The render says it does not
    exist below 15 Hz, and must not write energy there whatever the model does."""
    sc, _ = _cabin()
    res = renderer.render(sc, 48_000)
    f = np.fft.rfftfreq(len(res.ir), 1.0 / res.sample_rate)
    S = np.abs(np.fft.rfft(res.ir)) ** 2
    assert S[f < renderer.SUBSONIC_CORNER_HZ].sum() < 0.01 * S.sum()


def test_the_modal_band_hands_over_without_a_step():
    """The two descriptions of the room are crossfaded, so neither may be several
    dB louder than the other where they meet."""
    sc, _ = _cabin()
    res = renderer.render(sc, 48_000)
    f_x = res.metrics["modal_cutoff_hz"]
    f = np.fft.rfftfreq(len(res.ir), 1.0 / res.sample_rate)
    db = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(res.ir)), 1e-15))
    below = db[(f > 0.70 * f_x) & (f < 0.92 * f_x)]
    above = db[(f > 1.08 * f_x) & (f < 1.35 * f_x)]
    assert abs(float(np.median(below)) - float(np.median(above))) < 5.0


def test_the_room_is_not_louder_than_the_loudspeaker_driving_it():
    """The modal band is a room *transfer function*: the source's own bottom-end
    limit belongs to it as much as it belongs to the ray paths."""
    sc, _ = _cabin()
    f3_default = sc.source.f3_db
    assert f3_default > 30.0, "the preset no longer has a limited source"

    def bass(scene):
        res = renderer.render(scene, 48_000)
        f = np.fft.rfftfreq(len(res.ir), 1.0 / res.sample_rate)
        e = np.abs(np.fft.rfft(res.ir)) ** 2
        return 10.0 * np.log10(e[(f > 16) & (f < 30)].sum())

    flat = replace(sc, source=replace(sc.source, f3_db=16.0))
    assert bass(sc) < bass(flat) - 6.0, (
        "cutting the source a whole octave lower than its modes changed the "
        "rendered bass by less than 6 dB: the modal band ignores the source")


# ---------------------------------------------------------------------------
# 4. Units and honest signatures.  (AUDIT.md F5, F7.)
# ---------------------------------------------------------------------------
def test_time_strength_is_a_centroid_of_the_decay_curve_in_centiseconds():
    """Recomputed here from the definition in the docstring: find where the
    backward-integrated decay curve sits between -10 and -25 dB, take the centroid
    of the energy there, and subtract the 50 ms the definition measures from."""
    fs = 8_000
    rng = np.random.default_rng(3)
    ir = rng.standard_normal(fs) * np.exp(-np.arange(fs) / fs * (6.908 / 0.5))
    ir[:int(0.05 * fs)] = 0.0

    e = np.cumsum(ir[::-1] ** 2)[::-1]              # energy still to come
    e_db = 10.0 * np.log10(np.maximum(e, 1e-300) / max(e[0], 1e-300))
    m = (e_db >= -25.0) & (e_db <= -10.0)
    t = np.arange(fs) / fs
    want = (float(np.sum(t[m] * ir[m] ** 2) / np.sum(ir[m] ** 2)) - 0.050) * 100.0
    assert want > 0.0, "the fixture never reaches the -10 dB window"

    got = rooms.time_strength_cents(ir, fs)
    assert got == pytest.approx(want, rel=1e-3)
    # ...and the number really is in centiseconds, not seconds or milliseconds
    assert 1.0 < got < 100.0, f"TS = {got}: wrong scale for a room"


def test_decay_per_band_and_spl_from_ir_ask_for_nothing_they_ignore():
    """A parameter that is accepted and ignored is a promise the code does not
    keep; the diffuse-field measurement used to advertise band windows."""
    assert "windows" not in inspect.signature(rooms.decay_per_band).parameters
    assert "source_power_w" not in inspect.signature(rooms.spl_from_ir).parameters
    # ...and it still measures a decay it is given
    fs = 48_000
    rng = np.random.default_rng(7)
    t = np.arange(int(3.0 * fs)) / fs
    ir = rng.standard_normal(len(t)) * np.exp(-t * (6.908 / 0.4))
    t60 = rooms.decay_per_band(ir, fs, np.array([500.0, 1000.0, 2000.0]))
    assert np.all(np.abs(t60 - 0.4) < 0.06), t60


def test_modal_decay_rate_is_db_per_second_and_slower_than_sabine_for_axials():
    """Dimensional check: a rate must go as c / (a length), not as f / (a length).

    Halving the room at the same mode number therefore halves the loss per
    second; the version that divided a frequency by a path length quartered it.
    """
    dims = (3.0, 3.0, 3.2)
    a = Absorber("flat", (0.10,) * 6)        # alpha over the six reference bands

    def rate(d):
        from soundsim.acoustics.rooms import Surface
        lx, ly, lz = d
        pairs = [[Surface(a, ly * lz, "x"), Surface(a, ly * lz, "x")],
                 [Surface(a, lx * lz, "y"), Surface(a, lx * lz, "y")],
                 [Surface(a, lx * ly, "z"), Surface(a, lx * ly, "z")]]
        return rooms.modal_decay_rate((1, 0, 0), d, pairs, 20.0)

    r = rate(dims)
    assert r == pytest.approx(-10.0 * np.log10(1.0 - 0.10) * C / dims[0], rel=1e-6)
    assert rate(tuple(2 * np.array(dims))) == pytest.approx(r / 2.0, rel=1e-6)
    # An axial mode presses two walls of a uniformly lined room, so it rings
    # S / (4 A_x) times as long as the Sabine average says: check that geometric
    # factor against Sabine directly, in dB/s.
    S = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])
    V = float(np.prod(dims))
    # Sabine in the same "dB per encounter with a wall" form -- 0.161 V/A is its
    # small-loss expansion, and the mode's loss per hit is finite, so the exact
    # logarithm is what belongs here.
    per_hit = -10.0 * np.log10(1.0 - 0.10)
    sabine_db_per_s = per_hit * C * S / (4.0 * V)
    assert r == pytest.approx(sabine_db_per_s * 4.0 * dims[1] * dims[2] / S, rel=1e-6)
    # the uniform "mode" does not oscillate and has nothing to lose
    assert rooms.modal_decay_rate((0, 0, 0), dims, [[], [], []], 20.0) == 0.0


def test_field_incidence_mass_law_is_the_hemisphere_average_it_claims_to_be():
    """R_field = 10 log10( Z^2 / ln(1 + Z^2) ), Z = w m' / 2 z0 -- checked against
    a numerical integration of the same transmission coefficient the plate model
    uses, with the Beranek weighting."""
    from soundsim.acoustics.constants import specific_impedance
    m_prime, f = 12.0, np.array([200.0, 1000.0, 4000.0])
    z0 = specific_impedance(20.0)
    z = 2 * np.pi * f * m_prime / (2 * z0)
    u = np.linspace(0.0, 1.0, 200001)
    tau = 1.0 / (1.0 + (z[:, None] * u[None, :]) ** 2)
    tau_avg = 2.0 * np.trapezoid(tau * u[None, :], u, axis=1)
    got = transmission.mass_law_db(f, m_prime, field_incidence=True, temp_c=20.0)
    assert got == pytest.approx(-10.0 * np.log10(tau_avg), abs=0.02)
    # the exact normal-incidence limit, and the (silent) zero-mass limit
    assert transmission.mass_law_db(f, m_prime, field_incidence=False) == \
        pytest.approx(10.0 * np.log10(1.0 + z ** 2), rel=1e-9)
    assert float(transmission.mass_law_db(np.array([1.0]), 1e-6)[0]) == \
        pytest.approx(0.0, abs=1e-3)


def test_the_room_tail_holds_the_energy_the_header_promises():
    """E_explicit + E_tail == E_direct (1 + 16 pi r^2 / A) -- the identity the
    module header states, and which the tail used to miss out a direct ray."""
    sc = presets.make("across_the_room")
    sc.receiver = (7.0, 3.7, 1.4)
    b = PathBuilder(sc)
    space = sc.space_at(sc.source.position)
    paths, tail = b.space_paths(space, sc.source.position, sc.receiver, sc.ism_order)
    room = sc.room(space)

    d = float(np.linalg.norm(np.array(sc.receiver) - np.array(sc.source.position)))
    areas = np.array([s.area_m2 for s in room.surface_list()])
    alpha = np.array([s.absorber.alpha_at(b.bands) for s in room.surface_list()])
    A = np.sum(alpha * areas[:, None], axis=0) + AIR_AREA_COEF * b.alpha_atm * room.volume
    target = np.abs(paths[0].gains) ** 2 * (1.0 + 16.0 * np.pi * d ** 2 / A)

    e_explicit = sum(np.abs(p.gains) ** 2 for p in paths)
    weight = rooms.mode_count_weight(b.bands, room.volume, sc.weather.temp_c)
    full = weight > 0.999                      # bands where the modal weighting is out of the way
    # The path list is pruned of inaudible rays after the tail has been sized, so
    # what the builder returns carries a hair less energy than the balance saw.
    assert e_explicit[full] + tail.energy[full] == pytest.approx(target[full], rel=2e-2)
    assert np.all(tail.energy[full] > 0.0), "the tail is empty where the room is reverberant"
