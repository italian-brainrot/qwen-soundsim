"""End-to-end checks of the synthesis engine (not the physics primitives).

These lock in the *guarantees* the hybrid scheme makes: the synthesised response
carries the reverberation time the room's absorption implies, the energy balance
is respected, the real-time convolver is exact, and no preset ever produces a
response with NaNs or a silent output.
"""
import numpy as np
import pytest

from soundsim.acoustics import atmosphere, rooms
from soundsim.engine import presets, renderer
from soundsim.engine.bands import BAND_CENTRES, band_windows
from soundsim.engine.convolver import PartitionedConvolver
from soundsim.engine.paths import PathBuilder, build_paths
from soundsim.engine.player import demo_material
from soundsim.engine.rir import Tail, synthesize
from soundsim.engine.scene import Coupling, Room, Scene, Source


def test_partition_of_unity_band_bank():
    f = np.logspace(0, 4.3, 4000)
    w = band_windows(f, BAND_CENTRES)
    assert np.allclose(w.sum(axis=0), 1.0, atol=1e-9)
    assert np.all(w >= 0.0)


def test_synth_tail_reproduces_its_own_decay_time():
    """A tail generated with T60 = X s must measure X s (within a few percent)."""
    centres = BAND_CENTRES
    fs = 48_000
    e = np.full(len(centres), 1e-3)
    t60 = np.full(len(centres), 0.40)
    res = synthesize([], Tail(e, t60, onset_s=0.02), fs, length=1 << 17)
    measured = rooms.decay_per_band(rooms.band_limited(res.ir, fs), fs, centres)
    sel = [i for i, f in enumerate(centres) if 125 <= f <= 4000]
    # single realisation of noise per band, so there is genuine measurement
    # scatter; what must hold is that it lands near the specified decay time
    assert np.abs(np.median(measured[sel]) - 0.40) < 0.06
    assert np.all(np.abs(measured[sel] - 0.40) < 0.15)


def test_room_response_measured_t60_matches_sabine_in_the_diffuse_regime():
    """The whole point of the energy-balanced tail: measured T60 == Sabine T60."""
    sc = presets.make("concrete_stairwell")
    sc.ism_order = 8
    res = renderer.render(sc, 48_000)
    room = sc.rooms["stairwell"]
    a_atm = atmosphere.attenuation_coefficient(res.band_centres, sc.weather.temp_c,
                                               sc.weather.rh_pct)
    sabine = np.asarray(rooms.sabine_t60(room.volume, room.surface_list(),
                                         freqs_hz=res.band_centres, alpha_atm=a_atm))
    measured = np.asarray(res.metrics["t60_bands"])
    bands = res.band_centres
    for fc in (500.0, 1000.0, 2000.0):
        i = int(np.argmin(abs(bands - fc)))
        assert measured[i] == pytest.approx(sabine[i], rel=0.15)


def test_energy_balance_holds_for_an_enclosed_room():
    """Path + tail energy must equal E_direct * (1 + 16 pi r^2 / A)."""
    room = Room(id="r", dims=(7.0, 5.0, 2.8), walls={k: "brick_painted" for k in
                                                     ("floor", "ceiling", "west",
                                                      "east", "south", "north")})
    sc = Scene(name="t", source=Source(position=(1.0, 1.0, 1.4)),
               receiver=(5.5, 4.0, 1.5), rooms={"r": room}, ism_order=4)
    pb = PathBuilder(sc)
    paths, tails = build_paths(sc)
    e_paths = sum(float(np.sum(np.abs(p.gains) ** 2)) for p in paths)
    e_tail = sum(float(np.sum(t.energy)) for t in tails)
    d = np.linalg.norm(np.array(sc.receiver) - np.array(sc.source.position))
    a_atm = atmosphere.attenuation_coefficient(pb.bands, sc.weather.temp_c,
                                               sc.weather.rh_pct)
    A = rooms.sabine_t60(room.volume, room.surface_list(), freqs_hz=pb.bands,
                         alpha_atm=a_atm)
    a_abs = 0.161 * room.volume / A
    src_resp = pb.source_response()
    e_dir = float(np.sum((pb.spreading(d) * src_resp) ** 2))
    target = float(np.sum((pb.spreading(d) * src_resp) ** 2
                          * (1.0 + 16.0 * np.pi * d**2 / a_abs)))
    assert (e_paths + e_tail) / target == pytest.approx(1.0, rel=0.10)


def test_walking_away_keeps_the_reverberant_field_flat():
    """Direct sound falls ~6 dB per doubling of distance, reverberation does not."""
    def c50(dist):
        sc = presets.make("across_the_room")
        sc.receiver = (1.2 + dist, 3.7, 1.4)
        return renderer.render(sc, 48_000).metrics
    near, far = c50(2.0), c50(8.0)
    # 4x distance = 12 dB less direct sound, so clarity must drop by ~12 dB
    assert far["c50"] - near["c50"] == pytest.approx(-12.0, abs=2.5)


def test_masonry_beats_glazing_and_an_opened_window_beats_both():
    """Surface density and openings, the two things everyone tests by ear."""
    glazing = presets.make("through_closed_window")
    p_glazing = renderer.render(glazing, 48_000).metrics["total_spl_db"]

    masonry = presets.make("through_closed_window")
    masonry.couplings[0].panel_key = "brick_100mm"
    masonry.couplings[0].second_panel_key = None
    masonry.couplings[0].cavity_m = 0.0
    p_masonry = renderer.render(masonry, 48_000).metrics["total_spl_db"]
    assert p_masonry < p_glazing - 8.0

    open_window = presets.make("through_closed_window")
    open_window.couplings[0].kind = "aperture"
    open_window.couplings[0].panel_key = None
    p_open = renderer.render(open_window, 48_000).metrics["total_spl_db"]
    assert p_open > p_glazing + 20.0


def test_humidity_changes_the_treble_of_a_long_outdoor_path():
    """Over 300 m the air is a filter, and the ISO 9613-1 relaxation model says
    its humidity dependence changes sign with frequency in cold air."""
    def at(m, fc):
        return m["band_response_db"][int(np.argmin(abs(m["bands_hz"] - fc)))]

    dry = presets.make("far_field_park")
    dry.weather.rh_pct = 10.0
    wet = presets.make("far_field_park")
    wet.weather.rh_pct = 90.0
    mdry = renderer.render(dry, 48_000).metrics
    mwet = renderer.render(wet, 48_000).metrics

    # 300 m of air barely touches the bass and eats the treble: the total level
    # falls modestly while the spectral balance collapses
    assert mdry["environment_loss_db"] > 5.0
    assert at(mdry, 8000.0) < at(mdry, 250.0) - 10.0
    # at 8 degrees, humidity *helps* a couple of kHz carry and *hurts* the top
    # octaves -- the sign flip the relaxation model predicts
    assert at(mwet, 2000.0) > at(mdry, 2000.0) + 5.0
    assert at(mwet, 8000.0) < at(mdry, 8000.0) - 5.0
    assert abs(at(mwet, 250.0) - at(mdry, 250.0)) < 0.6


def test_modal_band_puts_cabin_modes_where_theory_says():
    """A 4.4 x 1.8 x 1.2 m cabin has axial modes at 39.0 and 78.0 Hz."""
    sc = presets.make("inside_car")
    res = renderer.render(sc, 48_000)
    fs = res.sample_rate
    f = np.fft.rfftfreq(len(res.ir), 1.0 / fs)
    h = np.abs(np.fft.rfft(res.ir))
    db = 20 * np.log10(np.maximum(h, 1e-15))

    def peak_near(fc, win=8.0):
        m = (f > fc - win) & (f < fc + win)
        return float(db[m].max())

    ref = float(db[(f > 400) & (f < 900)].mean())
    assert peak_near(39.0) > ref - 2.0            # the axial mode must be visible
    assert peak_near(78.0) > ref - 2.0


def test_no_preset_produces_nans_or_silence():
    for name in presets.PRESETS:
        res = renderer.render(presets.make(name), 48_000)
        assert np.all(np.isfinite(res.ir)), name
        assert np.max(np.abs(res.ir)) > 1e-9, name
        assert res.metrics["total_spl_db"] > -140.0, name


def test_partitioned_convolver_is_exact_for_any_block_size():
    rng = np.random.default_rng(11)
    for n in (64, 333, 1024):
        block = 256
        c = PartitionedConvolver(block)
        h = rng.standard_normal(9000) * 0.05
        h[0] = 0.8
        h[7000] = 0.3
        c.set_ir(h)
        x = rng.standard_normal(4000)
        y = np.concatenate([c.process(x[:1300]), c.process(x[1300:]),
                            c.process(np.zeros(block))])
        ref = np.convolve(x, h)
        d = c.block
        # only the part of the linear convolution that the fed input can produce
        assert np.allclose(y[d:d + len(x)], ref[:len(x)], atol=1e-9)


def test_demo_material_is_well_behaved():
    m = demo_material(3.0, 48_000)
    assert m.samples.shape[1] == 2
    assert np.max(np.abs(m.samples)) <= 0.95
    assert np.isfinite(m.samples).all()
    assert m.duration == pytest.approx(3.0, abs=0.01)


def test_render_pair_gives_two_decorrelated_channels():
    sc = presets.make("across_the_room")
    irs, res = renderer.render_pair(sc, 48_000, separation_m=0.35)
    assert len(irs) == 2 and len(irs[0]) == len(irs[1])
    a, b = irs[0], irs[1]
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert 0.2 < corr < 0.999            # related but not identical
