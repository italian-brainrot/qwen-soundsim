"""Room acoustics: reverberation time, modal behaviour, ISO 3382 parameters.

Two complementary things live here:

1. **Analytic estimates** (Sabine, Eyring, modified Sabine, modal frequencies)
   used for the read-outs in the UI and to scale the diffuse tail.
2. **Measurements on the synthesised impulse response** (T20/T30/T60, EDT, C50,
   C80, D50, TS).  These are the authoritative numbers the app displays, because
   the engine's decay already contains air absorption, geometry and non-exponential
   tails that a Sabine number cannot express.

Sabine derivation used consistently throughout (so the air term is coherent):

    decay rate = (c S / 0.9214 V) * <alpha>   [dB/s]     =>  RT60 = 0.161 V / A
    decay rate_air = c * alpha_atm                          =>  A_air = 0.9214 * alpha_atm * V
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import speed_of_sound
from .materials import Absorber

AIR_ABSORPTION_AREA_COEF = 0.9214     # A_air = 0.9214 * alpha[dB/m] * V


@dataclass
class Surface:
    """An area of one material bounding a room."""
    absorber: Absorber
    area_m2: float
    name: str = ""


def average_absorption(surface_area_pairs, freqs_hz=(125, 250, 500, 1000, 2000, 4000)) -> np.ndarray:
    """Area-weighted mean absorption coefficient, band by band."""
    areas = np.array([s.area_m2 for s in surface_area_pairs], dtype=np.float64)
    areas = areas / areas.sum()
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    a = np.zeros(len(f))
    for s, w in zip(surface_area_pairs, areas):
        a += w * s.absorber.alpha_at(f)
    return a


def sabine_t60(volume_m3, surfaces, freqs_hz=(125, 250, 500, 1000, 2000, 4000),
               alpha_atm=None) -> np.ndarray:
    """Sabine reverberation time, band by band, including atmospheric absorption.

    ``alpha_atm``: atmospheric absorption coefficients (dB/m) **at the same band
    centre frequencies** -- decisive in large, dry rooms at high frequency.
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    areas = np.array([s.area_m2 for s in surfaces], dtype=np.float64)
    alpha = np.array([s.absorber.alpha_at(f) for s in surfaces])
    a = np.sum(alpha * areas[:, None], axis=0)
    if alpha_atm is not None:
        a = a + AIR_ABSORPTION_AREA_COEF * np.asarray(alpha_atm, dtype=np.float64) * volume_m3
    return 0.161 * volume_m3 / np.maximum(a, 1e-6)


def eyring_t60(volume_m3, surfaces, total_area_m2: float, alpha_atm=None,
               freqs_hz=(125, 250, 500, 1000, 2000, 4000)) -> np.ndarray:
    """Eyring reverberation time (better than Sabine in absorptive rooms).

    ``RT60 = 0.161 V / (-S ln(1 - alpha_bar))``
    """
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    areas = np.array([s.area_m2 for s in surfaces], dtype=np.float64)
    alpha = np.array([s.absorber.alpha_at(f) for s in surfaces])
    a_mean = np.sum(alpha * areas[:, None], axis=0) / areas.sum()
    a = -total_area_m2 * np.log(np.maximum(1.0 - a_mean, 1e-6))
    if alpha_atm is not None:
        a = a + AIR_ABSORPTION_AREA_COEF * np.asarray(alpha_atm, dtype=np.float64) * volume_m3
    return 0.161 * volume_m3 / np.maximum(a, 1e-6)


def modified_sabine_t60(dims, surfaces_by_axis, alpha_atm=None, temp_c: float = 20.0,
                        freqs_hz=(125, 250, 500, 1000, 2000, 4000)):
    """Fitzroy / modified Sabine for a rectangular room with different walls.

    Absorption is grouped by the axis its normal is parallel to, which accounts
    for the fact that a wave travelling along x hits the x-walls less often than
    a diffuse average implies (Fitzroy 1959):

        1/T = (c / 8V) * sum_axes Sy Sz * alpha_axis * <path factor>
    """
    lx, ly, lz = dims
    v = lx * ly * lz
    c = speed_of_sound(temp_c)
    areas = {0: ly * lz, 1: lx * lz, 2: lx * ly}
    rates = np.zeros(len(np.atleast_1d(freqs_hz)))
    for axis, pair in surfaces_by_axis.items():
        a_bar = average_absorption(pair, freqs_hz) if pair else np.zeros(len(np.atleast_1d(freqs_hz)))
        # Fitzroy weighting: -ln(1-alpha)/alpha corrects high absorption
        corr = np.where(a_bar < 1e-3, 1.0, -np.log(np.maximum(1 - a_bar, 1e-6)) / np.maximum(a_bar, 1e-6))
        rates += areas[axis] * a_bar * corr
    t = 8.0 * v / (c * np.maximum(rates, 1e-9))
    if alpha_atm is not None:
        t_air = 60.0 / (c * np.asarray(alpha_atm, dtype=np.float64))
        t = 1.0 / (1.0 / t + 1.0 / t_air)
    return t


def mean_free_path(volume_m3, total_area_m2) -> float:
    return 4.0 * volume_m3 / total_area_m2


def Schroeder_frequency(volume_m3, t60: float) -> float:
    """Modal overlap / diffuse-field onset frequency (Schroeder 1962)."""
    return 2000.0 * np.sqrt(t60 / volume_m3)


def room_modes(dims, f_max: float = 250.0, temp_c: float = 20.0):
    """Rayleigh eigenfrequencies of a rectangular room, up to ``f_max``.

    Returns a list of ``(f_hz, (n, m, l), kind)`` where kind is axial (one index
    non-zero), tangential (two) or oblique (three).  Axial modes dominate the
    low-frequency response of small rooms -- a car cabin is essentially a set of
    axial modes below ~150 Hz.
    """
    lx, ly, lz = dims
    c = speed_of_sound(temp_c)
    out = []
    n_max = int(2 * f_max * lx / c) + 1
    m_max = int(2 * f_max * ly / c) + 1
    l_max = int(2 * f_max * lz / c) + 1
    for n in range(n_max + 1):
        for m in range(m_max + 1):
            for l in range(l_max + 1):
                if n == m == l == 0:
                    continue
                f = (c / 2.0) * np.sqrt((n / lx) ** 2 + (m / ly) ** 2 + (l / lz) ** 2)
                if f > f_max:
                    continue
                nz = (n > 0) + (m > 0) + (l > 0)
                kind = {1: "axial", 2: "tangential", 3: "oblique"}[nz]
                out.append((f, (n, m, l), kind))
    out.sort()
    return out


def modal_density(f_hz: float, volume_m3, t60: float) -> float:
    """Mean modal density (modes per Hz), the high-frequency asymptote.

    ``N(f) = 24 pi V / c^3 * f^2`` in the statistical limit; the mean spacing is
    ``1/N``.  Below the Schroeder frequency the field is not diffuse and a
    continuous reverberation description stops being meaningful.
    """
    c = speed_of_sound()
    return 24.0 * np.pi * volume_m3 / c**3 * float(f_hz) ** 2


def modal_decay_rate(mode, dims, surfaces_by_axis, temp_c: float = 20.0) -> float:
    """Absorption of a single mode (Kuttruff), dB/s.

    Each mode loses energy at a rate set by the wall absorption weighted by the
    fraction of its energy near each wall: an (n,0,0) axial mode loses
    ``alpha_x`` per round trip over 2 Lx.
    """
    lx, ly, lz = dims
    c = speed_of_sound(temp_c)
    n, m, l = mode
    a_x = _axis_absorption(surfaces_by_axis[0])
    a_y = _axis_absorption(surfaces_by_axis[1])
    a_z = _axis_absorption(surfaces_by_axis[2])
    f = (c / 2.0) * np.sqrt((n / lx) ** 2 + (m / ly) ** 2 + (l / lz) ** 2)
    if f <= 0:
        return 0.0
    # energy-loss per round trip in each direction, weighted by modal participation
    fx = (n / lx) / np.sqrt((n / lx) ** 2 + (m / ly) ** 2 + (l / lz) ** 2) if n else 0.0
    fy = (m / ly) / np.sqrt((n / lx) ** 2 + (m / ly) ** 2 + (l / lz) ** 2) if m else 0.0
    fz = (l / lz) / np.sqrt((n / lx) ** 2 + (m / ly) ** 2 + (l / lz) ** 2) if l else 0.0
    # alpha per hit -> dB per hit -> hits per second
    per_hit = 0.0
    for frac, a in ((fx, a_x), (fy, a_y), (fz, a_z)):
        if a >= 1.0:
            return np.inf
        per_hit += frac * (-10.0 * np.log10(max(1.0 - a, 1e-6)))
    hits_per_s = f / max((fx * 2 * lx + fy * 2 * ly + fz * 2 * lz) / (fx + fy + fz), 1e-6)
    return per_hit * hits_per_s


def _axis_absorption(pair) -> float:
    """Mean alpha of the two walls normal to an axis (band-averaged)."""
    if not pair:
        return 0.0
    a = [s.absorber.alpha for s in pair]
    w = [s.area_m2 for s in pair]
    tot = sum(w) or 1.0
    return float(np.dot(w, np.mean(np.array(a), axis=1)) / tot)


# ---------------------------------------------------------------------------
# ISO 3382 parameters measured on an impulse response
# ---------------------------------------------------------------------------
def decay_curve_db(ir: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward-integrated decay curve, dB, versus time in seconds."""
    ir = np.asarray(ir, dtype=np.float64)
    e = np.cumsum(ir[::-1] ** 2)[::-1]
    e = np.maximum(e, 1e-300)
    db = 10.0 * np.log10(e / e[0])
    t = np.arange(len(ir)) / fs
    return t, db


def _t_from_slice(t, db, lo_db: float, hi_db: float) -> float:
    """Decay time from the slope measured over the dB range [lo, hi].

    ISO 3382's T30/T20 are just slope estimates: whatever dB range is used,
    ``T60 = 60 / |slope|`` seconds.  (The usual "multiply the measured interval
    by 2" phrasing is the same statement for a 30 dB range.)
    """
    m = (db <= lo_db) & (db >= hi_db)
    if m.sum() < 8:
        return float("nan")          # the record never got that far: not measurable
    slope = np.polyfit(t[m], db[m], 1)[0]
    if slope >= 0:
        return float("nan")
    return 60.0 / -slope


def t60(ir: np.ndarray, fs: int) -> float:
    """T60 by the T30 method (ISO 3382): slope of -5..-35 dB.

    Returns NaN when the record is shorter than the decay: a truncated response
    carries no information about T60, and fitting it produces a number that is
    not merely wrong but confidently wrong.
    """
    t, db = decay_curve_db(ir, fs)
    if np.nanmax(db) - np.nanmin(db) < 25.0:
        return float("nan")
    val = _t_from_slice(t, db, -5.0, -35.0)
    if not np.isfinite(val):
        val = _t_from_slice(t, db, -5.0, -25.0)          # T20 fallback
    if not np.isfinite(val):
        val = _t_from_slice(t, db, 0.0, -60.0)           # last resort
    return float(val)


def edt(ir: np.ndarray, fs: int) -> float:
    """Early decay time: slope of 0..-10 dB extrapolated to -60 dB."""
    t, db = decay_curve_db(ir, fs)
    return float(_t_from_slice(t, db, 0.0, -10.0))


def clarity(c: float, ir: np.ndarray, fs: int) -> float:
    """C_{c} in dB: early energy (<= c ms) / late energy."""
    ir = np.asarray(ir, float)
    n = int(round(c * fs / 1000.0))
    early = np.sum(ir[:n] ** 2)
    late = np.sum(ir[n:] ** 2)
    return 10.0 * np.log10(max(early, 1e-300) / max(late, 1e-300))


def definition(ir: np.ndarray, fs: int) -> float:
    """D50: fraction of energy in the first 50 ms, in dB."""
    ir = np.asarray(ir, float)
    n = int(round(0.05 * fs))
    tot = np.sum(ir**2)
    return 10.0 * np.log10(max(np.sum(ir[:n] ** 2), 1e-300) / max(tot, 1e-300))


def time_strength_cents(ir: np.ndarray, fs: int, db_range=(-10, -25)) -> float:
    """Time strength TS (centiseconds): centroid of the decay range."""
    ir = np.asarray(ir, float)
    t, db = decay_curve_db(ir, fs)
    m = (db >= db_range[1]) & (db <= db_range[0])
    if m.sum() < 4:
        return float("nan")
    w = ir[m] ** 2
    return float(np.sum(t[m] * w) / np.sum(w) * 1000.0 - 0.05 * 1000.0)


def early_decay_and_ratios(ir: np.ndarray, fs: int) -> dict[str, float]:
    """All of the common ISO 3382 single-channel parameters at once."""
    return {
        "t60": t60(ir, fs),
        "edt": edt(ir, fs),
        "c50": clarity(50, ir, fs),
        "c80": clarity(80, ir, fs),
        "d50": definition(ir, fs),
        "ts": time_strength_cents(ir, fs),
    }


def band_limited(ir: np.ndarray, fs: int, f_lo: float = 100.0,
                 f_hi: float = 4000.0, shoulder_octaves: float = 0.5) -> np.ndarray:
    """Band-limit a response to the range used for ISO 3382 reverberation times.

    Below ~100 Hz neither a real source nor the ear carries much energy, and a
    slowly-decaying 20 Hz pedestal inside a short record makes a *broadband* T60
    meaningless -- it measures the length of the record, not the room.
    """
    ir = np.asarray(ir, dtype=np.float64)
    freqs = np.fft.rfftfreq(len(ir), 1.0 / fs)
    lf = np.log2(np.maximum(freqs, 1e-9))

    def ramp(edge, sign):
        # raised cosine over `shoulder_octaves`, 0 -> 1 in the direction of `sign`
        d = sign * (lf - np.log2(edge)) / shoulder_octaves
        return np.where(d <= 0, 0.0, np.where(d >= 1, 1.0, 0.5 * (1 - np.cos(np.pi * d))))

    w = ramp(f_lo, +1) * ramp(f_hi, -1)
    return np.fft.irfft(np.fft.rfft(ir) * w, n=len(ir))


def decay_per_band(ir: np.ndarray, fs: int, band_centres, windows=None) -> np.ndarray:
    """T60 measured on each 1/3-octave band, with analyzer-shaped filters.

    This is what ISO 3382 prescribes (T60 at 500 Hz, 1 kHz, ...).  A single
    broadband number is meaningless when absorption varies with frequency: the
    low bands ring on while the treble is already gone.

    The filters are 2nd-order Butterworth band passes -- the shape a real
    octave/third-octave analyser uses.  Brick-wall frequency-domain masks are
    *not* used here: a narrow ideal mask rings for many cycles at low frequency,
    and that ringing is then measured as an absurdly long reverberation time.
    """
    from scipy.signal import butter, sosfiltfilt

    ir = np.asarray(ir, dtype=np.float64)
    out = np.empty(len(band_centres))
    nyq = fs / 2.0
    for b, fc in enumerate(band_centres):
        lo = fc / 2 ** (1 / 6)
        hi = min(fc * 2 ** (1 / 6), nyq * 0.99)
        if lo >= nyq:
            out[b] = float("nan")
            continue
        sos = butter(2, [max(lo / nyq, 1e-6), hi / nyq], btype="band", output="sos")
        out[b] = t60(sosfiltfilt(sos, ir, axis=0), fs)
    return out


def a_weighting(freqs_hz) -> np.ndarray:
    """IEC 61672-1 class-1 A-weighting in dB."""
    f = np.asarray(freqs_hz, dtype=np.float64)
    f1, f2, f3, f4 = 20.598997, 107.6526, 737.862, 12194.0
    num = (f4**2 * f**4)
    den = ((f**2 + f1**2) * np.sqrt((f**2 + f2**2) * (f**2 + f3**2)) * (f**2 + f4**2))
    r = num / np.maximum(den, 1e-300)
    k = (f4**2 * 1000.0**4) / ((1000.0**2 + f1**2) * np.sqrt((1000.0**2 + f2**2) * (1000.0**2 + f3**2)) * (1000.0**2 + f4**2))
    return 20.0 * np.log10(r / k) + 2.0


def spl_from_ir(ir: np.ndarray, fs: int, source_power_w: float | None = None) -> dict[str, float]:
    """Energy-based level of an impulse response, in dB (relative, and A-weighted).

    The absolute SPL of a *simulated* field requires the source strength; when
    ``source_power_w`` is given the receiver mean-square pressure is estimated
    from the total transmitted energy (see :func:`spherical_spreading`).
    """
    ir = np.asarray(ir, float)
    e = np.sum(ir**2)
    return {"lfdB": 10.0 * np.log10(max(e, 1e-300)),
            "energy_pa2_s": float(e)}
