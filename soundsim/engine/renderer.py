"""Scene -> impulse response, with the acoustic read-outs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .bands import BAND_CENTRES
from .paths import build_paths
from .rir import Path, Tail, RenderResult, synthesize
from .scene import Scene
from ..acoustics import atmosphere, modal, rooms
from ..acoustics.constants import speed_of_sound
from ..acoustics.rooms import early_decay_and_ratios


def render(scene: Scene, sample_rate: int = 48_000,
           band_centres=BAND_CENTRES, ir_length_s: float | None = None,
           seed: int = 12345, ir_length_cap_s: float = 6.0) -> RenderResult:
    """Build the scene's impulse response (already scaled to the source strength)."""
    paths, tails = build_paths(scene, band_centres)
    apply_source_strength(paths, tails, scene.source.spl_at_1m_db)

    length = None
    if ir_length_s:
        length = int(ir_length_s * sample_rate)
    else:
        # a very live room (concrete stairwells reach 8-13 s) would otherwise ask
        # for a multi-second-per-second buffer; cap it and say so in the metrics
        length = int(min(ir_length_cap_s, _needed_length_s(paths, tails)) * sample_rate)
    res = synthesize(paths, _merge_tails(tails), sample_rate, length=length,
                     band_centres=band_centres, seed=seed)
    res = apply_modal_lowband(res, scene)
    res.metrics.update(measure(res, scene))
    return res


def modal_cutoff_hz(scene: Scene) -> float:
    """Below this frequency an enclosed space is described by its modes.

    Uses the Schroeder frequency of the *smallest* room holding both source and
    receiver: below ~2 x Schroeder the field is a handful of modes and a
    geometrical (ray) description is the wrong model.
    """
    src_space = scene.space_at(scene.source.position)
    dst_space = scene.space_at(scene.receiver)
    room = scene.room(src_space)
    if room is None or src_space != dst_space:
        return 0.0
    t60 = float(np.mean(rooms.sabine_t60(room.volume, room.surface_list())))
    sch = rooms.Schroeder_frequency(room.volume, max(t60, 0.05))
    return float(np.clip(2.0 * sch, 60.0, 320.0))


def apply_modal_lowband(res: RenderResult, scene: Scene,
                        crossover_octaves: float = 0.6) -> RenderResult:
    """Replace the low-frequency response with the exact modal solution.

    The modal sum gives the *structure* ray methods cannot produce: where the
    bass peaks and nulls sit, and how long each mode rings.  Its level is
    anchored to the low-band energy of the path model (which is energy-balance
    validated), so the two descriptions carry the same energy and the crossfade
    is inaudible apart from the modal detail it adds.

    Only the FFT bins below the crossover are evaluated -- both for speed and to
    keep the memory footprint flat.
    """
    f_x = modal_cutoff_hz(scene)
    if f_x <= 0:
        return res
    room = scene.room(scene.space_at(scene.source.position))
    fs, ir = res.sample_rate, res.ir
    freqs = np.fft.rfftfreq(len(ir), 1.0 / fs)

    lo_lim = f_x / 2 ** crossover_octaves
    hi_lim = f_x * 2 ** crossover_octaves
    ramp = np.clip((np.log2(np.maximum(freqs, 1e-9)) - np.log2(lo_lim))
                   / (np.log2(hi_lim) - np.log2(lo_lim)), 0.0, 1.0)
    x_lo = 0.5 * (1.0 + np.cos(np.pi * ramp))          # 1 -> 0 across the crossover
    x_hi = 1.0 - x_lo

    # modal evaluation is restricted to the bins that can actually use it
    f_max = min(2.5 * hi_lim, 700.0)
    mask = freqs <= hi_lim
    if not np.any(mask):
        return res
    n_modes = len(modal.modal_frequencies(room.dims, f_max, scene.weather.temp_c)[1])
    if n_modes > 60_000:                               # keep renders responsive
        f_max = f_max * (60_000 / n_modes) ** (1 / 3)
        n_modes = len(modal.modal_frequencies(room.dims, f_max, scene.weather.temp_c)[1])

    axes = [_BlendedAbsorber(a, b) for a, b in room.absorbers_by_axis()]
    h_low = modal.modal_transfer(room.dims, room.local(scene.source.position),
                                 room.local(scene.receiver), freqs[mask], axes,
                                 temp_c=scene.weather.temp_c, f_max=f_max)
    h_low = h_low * 10.0 ** ((scene.source.spl_at_1m_db - 94.0) / 20.0)

    spec = np.fft.rfft(ir)
    e_paths = float(np.sum(np.abs(spec[mask] * x_lo[mask]) ** 2))
    e_modal = float(np.sum(np.abs(h_low * x_lo[mask]) ** 2))
    if e_paths > 0 and e_modal > 0:
        h_low = h_low * np.sqrt(e_paths / e_modal)
    spec[mask] = spec[mask] * x_hi[mask] + h_low * x_lo[mask]
    res.ir = np.fft.irfft(spec, n=len(ir))
    res.metrics["modal_cutoff_hz"] = f_x
    res.metrics["modal_modes_used"] = n_modes
    return res


class _BlendedAbsorber:
    """Area-weighted blend of the two absorbers bounding one axis."""

    def __init__(self, a, b, w=0.5):
        self.a, self.b, self.w = a, b, w

    def alpha_at(self, f):
        return self.w * self.a.alpha_at(f) + (1.0 - self.w) * self.b.alpha_at(f)


def _needed_length_s(paths, tails) -> float:
    """IR length the scene asks for: longest path + a full 60 dB decay."""
    longest = max([p.delay_s for p in paths], default=0.0)
    tail = max([float(np.max(t.t60)) for t in tails if t is not None], default=0.0)
    return longest + 1.05 * tail + 0.05


def render_pair(scene: Scene, sample_rate: int = 48_000,
                separation_m: float = 0.18, seed: int = 12345,
                ir_length_cap_s: float = 6.0) -> tuple[list[np.ndarray], RenderResult]:
    """Render a spaced-pair stereo pair, which is what the app listens with.

    Two receivers sit ``separation_m`` apart, perpendicular to the arriving wave
    (ear spacing, without a head).  Each channel is a full independent physical
    render, so inter-channel time and level differences come out of the geometry
    -- ground interference, a barrier edge, a room mode null -- rather than from
    an artificial stereo widening effect.  No HRTF: it is a spaced pair, so
    front/back confusion remains and it should be heard as "two microphones".
    """
    src = np.asarray(scene.source.position, float)
    rec = np.asarray(scene.receiver, float)
    d = rec - src
    d = d / max(np.linalg.norm(d), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    perp = np.cross(d, up)
    n = np.linalg.norm(perp)
    perp = perp / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    irs = []
    first = None
    for k, sign in enumerate((-0.5, +0.5)):
        s2 = replace(scene, receiver=tuple(rec + sign * separation_m * perp))
        r = render(s2, sample_rate, ir_length_cap_s=ir_length_cap_s,
                   seed=seed + 977 * k)
        irs.append(r.ir)
        first = first or r
    return irs, first


def apply_source_strength(paths: list[Path], tails: list[Tail], spl_at_1m_db: float):
    """Normalise everything to a source with a known free-field SPL at 1 m.

    Convention: convolving a signal whose rms is 1.0 with the returned IR gives a
    received pressure in pascals, so a source rated 94 dB SPL @ 1 m reproduces
    1 Pa rms at 1 m.
    """
    k = 10.0 ** ((spl_at_1m_db - 94.0) / 20.0)
    for p in paths:
        p.gains = p.gains * k
    for t in tails:
        t.energy = t.energy * k**2
    return paths, tails


def _merge_tails(tails: list[Tail]) -> Tail | None:
    """Combine several diffuse tails into one band description.

    Energies add.  Decay times are combined as *energy-weighted* decay rates,
    which is the correct way to merge two exponentially decaying fields: the sum
    of two exponentials decays roughly with the weighted mean rate near the
    onset, and the slower one dominates late (visible as a slight curvature of
    the decay curve, which is exactly what happens in reality when a muffled
    reverberant field leaks into a live one).
    """
    tails = [t for t in tails if t is not None and np.any(t.energy > 0)]
    if not tails:
        return None
    e = np.zeros_like(tails[0].energy)
    rate = np.zeros_like(tails[0].energy)
    onset = min(t.onset_s for t in tails)
    for t in tails:
        e = e + t.energy
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(t.t60 > 0, 1.0 / np.maximum(t.t60, 1e-6), 0.0)
        rate = rate + t.energy * r
    total = np.maximum(e, 1e-30)                      # energy-weighted mean decay
    t60 = np.where(rate > 0, total / np.maximum(rate, 1e-12), 0.0)
    return Tail(energy=e, t60=t60, onset_s=onset)


def measure(res: RenderResult, scene: Scene) -> dict:
    """Acoustic read-outs, measured on the synthesised response where possible."""
    ir, fs = res.ir, res.sample_rate
    # ISO-style parameters: measured on the 100 Hz - 4 kHz range that reverberation
    # time is actually defined over (a 20 Hz pedestal inside a short record would
    # otherwise dominate the decay fit)
    ir_iso = rooms.band_limited(ir, fs)
    out = early_decay_and_ratios(ir, fs)
    out["t60_iso"] = rooms.t60(ir_iso, fs)
    # per-band T60 measured on the response (ISO 3382 style), so it can be
    # compared with the analytic Sabine figure band by band
    from ..engine.bands import band_windows
    freqs = np.fft.rfftfreq(len(ir), 1.0 / fs)
    wins = band_windows(freqs, res.band_centres)
    t_bands = rooms.decay_per_band(rooms.band_limited(ir, fs, 60.0, 8000.0), fs,
                                   res.band_centres, wins)
    out["t60_bands"] = t_bands
    out["t60_bands_hz"] = res.band_centres
    # single-number rating: mid-band average, the way ISO 3382 describes a room.
    # A broadband decay figure is quoted alongside it, but in a small or bassy
    # space the single lowest mode dominates that fit and says nothing useful.
    mid = [t for f, t in zip(res.band_centres, t_bands) if 400.0 <= f <= 2000.0]
    mid = [t for t in mid if np.isfinite(t)]
    out["t60_mid_db"] = float(np.mean(mid)) if mid else float("nan")
    out["itdg_ms"] = _itdg(res)
    out["n_paths"] = len(res.paths)
    out["ir_length_ms"] = 1000.0 * len(ir) / fs

    # --- levels -----------------------------------------------------------
    # Free-field reference: what the receiver would get with nothing in between
    # (same distance, same air absorption).  All level figures below are relative
    # to that, which is what "how much does this environment cut the sound" means.
    from .paths import PathBuilder
    pb = PathBuilder(scene, res.band_centres)
    d = float(np.linalg.norm(np.asarray(scene.receiver, float)
                             - np.asarray(scene.source.position, float)))
    g_ff = pb.spreading_geometric(d) * pb.source_response()
    e_ff = float(np.sum(np.abs(g_ff) ** 2)) / len(res.band_centres)
    p_ff = 10.0 ** ((scene.source.spl_at_1m_db - 94.0) / 20.0) * np.sqrt(e_ff)
    out["free_field_spl_db"] = 20.0 * np.log10(max(p_ff, 1e-12) / 20e-6)

    e_total = float(np.sum(ir**2)) / len(res.band_centres)
    # the RIR is an energy response; comparing energies at equal source drive is
    # the standard direct/reverberant ratio measurement, so no decay-time
    # conversion is needed
    out["total_spl_db"] = out["free_field_spl_db"] + 10.0 * np.log10(
        max(e_total, 1e-300) / max(e_ff, 1e-300))
    # positive = the environment attenuates; negative = the rooms are focusing /
    # reverberating more energy into the receiver than free field would
    out["environment_loss_db"] = out["free_field_spl_db"] - out["total_spl_db"]

    # --- band-by-band level relative to free field ------------------------
    e_paths = np.zeros(len(res.band_centres))
    for p in res.paths:
        e_paths += np.abs(p.gains) ** 2
    for t in (res.tail and [res.tail] or []):
        e_paths += t.energy
    # the paths already carry the source's calibration (spl_at_1m_db), so the
    # reference must carry it too or the band balance would show the source
    # level rather than what the environment does to it
    k_src = 10.0 ** ((scene.source.spl_at_1m_db - 94.0) / 20.0)
    e_ref = np.abs(g_ff) ** 2 * k_src ** 2
    out["band_response_db"] = 10.0 * np.log10(
        np.maximum(e_paths, 1e-300) / np.maximum(e_ref, 1e-300))
    out["bands_hz"] = res.band_centres

    # --- analytic cross-check: Sabine T60 of every room in the scene ------
    w = scene.weather
    for room in scene.rooms.values():
        a_atm = atmosphere.attenuation_coefficient(
            res.band_centres, w.temp_c, w.rh_pct, w.pressure_pa)
        out[f"sabine_t60_{room.id}"] = rooms.sabine_t60(
            room.volume, room.surface_list(), freqs_hz=res.band_centres,
            alpha_atm=a_atm)
        out[f"schroeder_{room.id}"] = rooms.Schroeder_frequency(
            room.volume, float(np.mean(out[f"sabine_t60_{room.id}"])))
    return out


def _itdg(res: RenderResult) -> float:
    """Initial time delay gap: first arrival to the next distinct arrival.

    In a cascaded scene (through a wall, into another room) there is no single
    "reflection" class, so the gap is measured between the first and second
    *resolved* arrivals of any kind -- which is what the ear actually uses.
    """
    delays = sorted({round(p.delay_s, 5) for p in res.paths})
    if len(delays) < 2:
        return float("nan")
    return 1000.0 * (delays[1] - delays[0])
