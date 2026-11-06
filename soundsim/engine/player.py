"""Audio transport: file -> scene convolution -> output device.

Deliberately thin: a ring of decoded samples, the partitioned convolver, and a
PortAudio callback.  The scene impulse response can be replaced at any moment
(crossfaded inside the convolver), which is what lets you drag a wall's
thickness or the humidity while the music keeps playing.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

from .convolver import CrossfadeConvolver

SAMPLE_RATES = (44100, 48000, 88200, 96000)


@dataclass
class Material:
    """The thing being played: samples plus where they came from."""
    name: str
    samples: np.ndarray                 # (n, channels) float, -1..1
    sample_rate: int
    path: str | None = None

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def _libsndfile_formats() -> tuple[str, ...]:
    """Extensions libsndfile itself can decode (mp3 included since 1.1)."""
    try:
        # the dict keys are the extensions libsndfile will actually open; the
        # FormatInfo.name is the *container* it decodes into (mp3 arrives as
        # "wav"), so the keys are what matters here
        exts = {k.lower() for k in sf.available_formats()}
    except Exception:                                     # pragma: no cover
        exts = set()
    keep = {"wav", "flac", "ogg", "aiff", "aif", "au", "raw", "w64", "mp3",
            "wavpack", "caf"} & exts
    return tuple("." + k for k in sorted(keep))


SUPPORTED = _libsndfile_formats()
NEED_FFMPEG = (".m4a", ".aac", ".mp4", ".opus", ".webm", ".wma")


def pick_default_output() -> int | None:
    """Prefer the session manager's sink over a bare ALSA HDMI port.

    PortAudio sees every ALSA device, and on this kind of setup the ALSA
    "default output" is an HDMI port that goes nowhere, while the audible sink
    is the PulseAudio/PipeWire one.  Choosing it silently is the difference
    between "nothing happens" and audio.
    """
    try:
        infos = sd.query_devices()
        default_out = sd.default.device[1]
        if default_out is not None and default_out >= 0:
            d = infos[default_out]
            name = (d.get("name") or "").lower()
            looks_muted = "hdmi" in name or "digital stereo" in name
            if not looks_muted:
                return int(default_out)
        for ha in sd.query_hostapis():
            if "pulse" in ha["name"].lower() or "pipewire" in ha["name"].lower():
                dev = ha["default_output_device"]
                if dev is not None and dev >= 0:
                    return int(dev)
        for i, d in enumerate(infos):
            if d.get("max_output_channels", 0) > 0 and \
                    "pulse" in sd.query_hostapis()[d["hostapi"]]["name"].lower():
                return i
    except Exception:                                     # pragma: no cover
        pass
    return None


def load_file(path: str | Path, target_rate: int | None = None) -> Material:
    """Decode a file.  Formats libsndfile cannot read go through ffmpeg if present."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in SUPPORTED:
        data, rate = sf.read(str(path), always_2d=True, dtype="float64")
    else:
        data, rate = _decode_with_ffmpeg(path)
    if target_rate and rate != target_rate:
        g = math.gcd(int(rate), int(target_rate))
        data = resample_poly(data, target_rate // g, rate // g, axis=0)
        rate = int(target_rate)
    return Material(name=path.stem, samples=np.asarray(data, np.float64),
                    sample_rate=int(rate), path=str(path))


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Pipe a compressed file through ffmpeg as 32-bit float PCM."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{path.suffix} needs ffmpeg for decoding; install ffmpeg or use "
            f"{', '.join(SUPPORTED)}")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
           "-ac", "2", "-ar", "48000", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).astype(np.float64)
    return data, 48000


def demo_material(duration_s: float = 8.0, sample_rate: int = 48000) -> Material:
    """Built-in demonstration material.

    A plucked arpeggio (rich, decaying partials -- shows the tail), a pink-ish
    noise sweep (shows the band balance) and an impulse at the top (so the
    reverberation time is directly audible).  Makes the app useful with no audio
    files to hand.
    """
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    out = np.zeros((len(t), 2))
    # impulse
    out[0, :] = 0.7
    # plucked string arpeggio: A3 C4 E4 A4, inharmonic-free, with a soft attack
    for i, f0 in enumerate((220.0, 261.6, 329.6, 440.0)):
        t0 = 0.4 + i * 0.9
        m = t > t0
        tt = t[m] - t0
        tone = np.zeros(len(t))
        env = np.exp(-tt / 1.1) * (1.0 - np.exp(-tt / 0.006))
        s = np.zeros(len(tt))
        for k, amp in enumerate((1.0, 0.55, 0.33, 0.2, 0.12, 0.07), start=1):
            s += amp * np.sin(2 * np.pi * f0 * k * tt + 0.7 * k)
        tone[m] = env * s
        out[:, 0] += 0.16 * tone
        out[:, 1] += 0.16 * np.roll(tone, int(0.0004 * sample_rate))  # ms-scale spread
    # pink-ish noise bed (equal per octave), low level
    rng = np.random.default_rng(7)
    white = rng.standard_normal(len(t))
    pink = np.zeros(len(t))
    acc = 0.0
    for coef in (0.99, -0.9):        # crude 1/f shaping
        acc = coef * pink + white
        pink = acc
    pink /= max(np.abs(pink).max(), 1e-9)
    sweep = 0.5 * (1 + np.sin(2 * np.pi * 0.08 * t))
    out += (0.02 * pink * sweep)[:, None]
    peak = max(np.abs(out).max(), 1e-9)
    if peak > 0.9:
        out *= 0.9 / peak
    return Material(name="demo material", samples=out, sample_rate=int(sample_rate))


class Player:
    """Streams a :class:`Material` through the scene response to an output device."""

    def __init__(self, sample_rate: int = 48_000, channels: int = 2,
                 block: int = 512, device: int | None = None,
                 latency_s: float = 0.02):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.device = device
        self.latency_s = latency_s
        self.conv = CrossfadeConvolver(channels, block)
        self.material: Material | None = None
        self.index = 0                       # read position in material samples
        self.playing = False
        self.gain = 1.0
        self.gain_db = 0.0
        self.clip_flag = 0
        self.peak = 0.0
        self.error: str | None = None         # last audio-callback problem
        self.device_used: int | None = None
        self.limited_by = "scene level"
        self._lim = 1.0
        self._lim_prev = 1.0
        # amplitude wavering of a long path (see set_turbulence)
        self.turbulence_depth = 0.0
        self._turb_n = 0.0
        self._turb_rng = np.random.default_rng()
        self._turb_zi = None
        self._turb_fc = 1200.0
        self._stream: sd.OutputStream | None = None
        self._pending_ir: list[np.ndarray] | None = None
        self._pending_fade = 0.03

    # -- material -------------------------------------------------------
    def set_material(self, material: Material, keep_position: bool = False):
        """Load new material, resampling to the device rate."""
        if material.sample_rate != self.sample_rate:
            material = load_file(material.path, self.sample_rate) if material.path \
                else Material(material.name,
                              resample_poly(material.samples, self.sample_rate,
                                            material.sample_rate, axis=0),
                              self.sample_rate, material.path)
        old = self.index
        self.material = material
        self.index = min(old, len(material.samples) - 1) if keep_position else 0

    def set_position_seconds(self, seconds: float):
        if self.material is None:
            return
        self.index = int(np.clip(seconds * self.sample_rate, 0,
                                 len(self.material.samples)))

    @property
    def position_seconds(self) -> float:
        return self.index / self.sample_rate if self.material else 0.0

    @property
    def output_latency_seconds(self) -> float:
        """Fixed delay of the partitioned convolution pipeline (not a lag: it is
        a constant offset, so nothing about the sound depends on it)."""
        return self.conv.latency_samples / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        if self.material is None:
            return 0.0
        return len(self.material.samples) / self.sample_rate

    # -- output level ---------------------------------------------------
    def set_monitor_gain_for(self, received_spl_db: float,
                             target_spl_db: float = 70.0,
                             floor_db: float = -12.0, ceiling_db: float = 46.0,
                             headroom: float = 0.85):
        """Set the playback gain from the scene's predicted received level.

        The simulation is absolutely calibrated (a unit-rms signal is 1 Pa at
        1 m), which is what makes the reported dB figures meaningful -- but your
        amplifier also has gain, and a street scene at 36 dB SPL would otherwise
        be inaudible.  This maps the *predicted* level onto a comfortable one and
        is pure playback calibration: it never touches the response or the
        metrics, and the amount applied is reported so it is never mistaken for
        part of the physics.
        """
        gain_db = float(np.clip(target_spl_db - received_spl_db, floor_db, ceiling_db))
        # A live room (inside a car at 97 dB SPL) can still be far too hot for
        # the output even after the cut, so predict the real peak by convolving
        # the first seconds of the material and pull the gain down to fit.
        self.limited_by = "scene level"
        if self.material is not None and self.conv.irs:
            probe = self._probe_peak(self.material, self.conv.irs[0])
            predicted = probe * 10.0 ** (gain_db / 20.0)
            if predicted > headroom:
                gain_db -= 20.0 * math.log10(predicted / headroom)
                self.limited_by = "output headroom"
        self.gain_db = gain_db
        self.gain = 10.0 ** (gain_db / 20.0)
        return gain_db

    @staticmethod
    def _probe_peak(material: "Material", ir: np.ndarray,
                    seconds: float | None = None) -> float:
        """Worst-case sample magnitude of material * ir.

        Scans the whole material in chunks: a short probe misses a loud passage
        further in, and then the gain that was set for it clips -- which on bass
        content is heard as buzzing.
        """
        total = len(material.samples) if seconds is None else \
            min(len(material.samples), int(seconds * material.sample_rate))
        chunk = max(1, int(30.0 * material.sample_rate))
        nfft = 1 << int(np.ceil(np.log2(chunk + len(ir))))
        spec_ir = np.fft.rfft(ir, nfft)
        peak = 0.0
        for start in range(0, total, chunk):
            x = material.samples[start:min(start + chunk, total), 0]
            y = np.fft.irfft(np.fft.rfft(x, nfft) * spec_ir, nfft)
            peak = max(peak, float(np.max(np.abs(y))) if len(y) else 0.0)
            if peak <= 0.0:
                continue
        return peak

    # -- scene response -------------------------------------------------
    def set_impulse_responses(self, irs, crossfade_s: float = 0.03):
        """Install new room response(s); safe to call while playing."""
        self.conv.set_ir(irs, crossfade_s, self.sample_rate)

    # -- transport ------------------------------------------------------
    def start(self):
        """Open the output stream, trying the plausible devices and rates.

        Two things go wrong on real desktops and both look like "nothing
        happens": the system default is an HDMI port that refuses the rate, or a
        stream that died inside the callback is still holding the device.  So any
        existing stream is released first, then each candidate device is opened
        at the target rate and at 44.1 kHz.
        """
        if self._stream is not None and getattr(self._stream, "active", False):
            return
        self.stop_stream()                    # release a half-dead stream first
        devices = list(dict.fromkeys([self.device, pick_default_output(), None]))
        tried = []
        for dev in devices:
            for rate in (self.sample_rate, 44_100):
                try:
                    stream = sd.OutputStream(
                        samplerate=rate, channels=self.channels, device=dev,
                        blocksize=int(self.latency_s * rate), dtype="float32",
                        callback=self._callback,
                        finished_callback=self._stream_finished)
                    stream.start()
                except Exception as exc:
                    tried.append(f"dev={dev} {rate} Hz ({type(exc).__name__})")
                    continue
                self._stream = stream
                self.device_used = dev
                self.error = None
                if rate != self.sample_rate:
                    self._adopt_rate(rate)
                return
        raise RuntimeError("no output device accepted this stream -- tried: "
                           + "; ".join(tried))

    def _adopt_rate(self, rate: int):
        """The device wanted a different rate: re-decode the material to match."""
        self.sample_rate = int(rate)
        if self.material is None:
            return
        m = self.material
        if m.sample_rate == rate:
            return
        pos = self.position_seconds
        if m.path:
            self.material = load_file(m.path, rate)
        else:
            g = math.gcd(int(m.sample_rate), int(rate))
            self.material = Material(
                m.name, resample_poly(m.samples, rate // g,
                                      m.sample_rate // g, axis=0), int(rate), m.path)
        self.set_position_seconds(pos)

    def _stream_finished(self, *a):
        self._stream = None

    def stop_stream(self):
        """Close the stream and, crucially, hand the device back."""
        st, self._stream = self._stream, None
        if st is None:
            return
        try:
            st.stop()
        except Exception:
            pass
        try:
            st.close()
        except Exception:
            pass

    def play(self):
        self.playing = True
        self.start()

    def pause(self):
        self.playing = False

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    # -- turbulence -----------------------------------------------------
    def set_turbulence(self, depth: float, seed: int | None = None):
        """0..1 depth of the slow HF wavering a long outdoor path produces.

        Eddies refract the wave front around, so the received field is not a
        fixed filter: the high bands come and go over fractions of a second
        while the bass is essentially unaffected.  Over hundreds of metres this
        is one of the loudest cues that a sound is *far away* -- a static
        low-pass curve, however accurate, sounds like an EQ instead.

        The mechanism is physical; the depth is a knob, because it depends on
        wind, temperature gradient and site, which the scene does not know.
        """
        self.turbulence_depth = float(np.clip(depth, 0.0, 1.0))
        self._turb_zi = None
        self._turb_n = 0.0
        self._turb_rng = np.random.default_rng(seed)

    def _turbulence(self, out: np.ndarray) -> np.ndarray:
        depth = self.turbulence_depth
        if depth <= 0.0 or out.size == 0:
            return out
        from scipy.signal import butter, sosfilt

        if self._turb_zi is None:
            # 4th order so the modulated upper band barely bleeds into the
            # unmodulated lower one (with gain 1 the split is exact: low + high
            # reconstructs the input, filter phase and all)
            sos = butter(4, self._turb_fc / (self.sample_rate / 2.0), btype="low",
                         output="sos")
            self._turb_sos = sos
            self._turb_zi = np.zeros((sos.shape[0], 2, out.shape[1]))
        low = sosfilt(self._turb_sos, out, axis=0, zi=self._turb_zi)
        out_low, self._turb_zi = low[0], low[1]
        high = out - out_low

        # Ornstein-Uhlenbeck: a mean-reverting random walk, the standard model
        # for scintillation, with ~0.5 s correlation time
        tau = 0.5
        a = math.exp(-out.shape[0] / (tau * self.sample_rate))
        b = math.sqrt(max(1.0 - a * a, 1e-9))
        self._turb_n = a * self._turb_n + b * float(self._turb_rng.standard_normal())
        # up to +-9 dB of HF swing at full depth, coherent between the channels
        swing = 10.0 ** (depth * 9.0 * self._turb_n / 20.0)
        return out_low + high * swing

    # -- limiter --------------------------------------------------------
    def limit(self, out: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
        """Safety limiter: fast attack, slow release, no square-wave buzz.

        Hard clipping a bass-heavy render is exactly the "buzzing" sound people
        blame on the simulation.  The gain is normally set from a full scan of the
        material so this rarely engages; when it does it rides the level down over
        a couple of blocks instead of chopping waveform tops off.
        """
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > ceiling:
            want = ceiling / peak
            self._lim = max(want, min(self._lim, 1.0) * 0.6)   # ~1 block attack
            self.clip_flag += 1
        else:
            self._lim = min(1.0, self._lim * 1.02)             # ~1 s release
        if self._lim < 1.0:
            ramp = np.linspace(self._lim_prev, self._lim, out.shape[0])[:, None]
            out = out * ramp
        self._lim_prev = self._lim
        return out

    # -- callback -------------------------------------------------------
    @staticmethod
    def _match_channels(chunk: np.ndarray, want: int) -> np.ndarray:
        """Mono stays mono on both speakers, 5.1 keeps its front pair, etc."""
        have = chunk.shape[1]
        if have == want:
            return chunk
        if have == 1:
            return np.repeat(chunk, want, axis=1)
        if have < want:
            out = np.zeros((chunk.shape[0], want))
            out[:, :have] = chunk
            return out
        return chunk[:, :want]

    def _callback(self, outdata, frames, time_info, status):
        """Never let a glitch propagate.

        A PortAudio callback that raises either kills the stream or leaves it
        alive, holding the device and producing silence -- which from outside is
        indistinguishable from "the app does nothing".  So the callback is
        wrapped, the failure is recorded for the UI, and the device is released.
        """
        outdata[:] = 0.0
        if self.material is None:
            return
        try:
            # Feed *exactly* what the device asked for.  Over-feeding to "cover"
            # the convolver's latency looks harmless but consumes the material
            # faster than real time (2x speed, an octave of pitch shift, and a
            # queue that grows without bound); the latency is simply a fixed
            # delay, which is what output_latency_seconds reports.
            end = min(self.index + frames, len(self.material.samples))
            chunk = self.material.samples[self.index:end]
            self.index = end
            if len(chunk) < frames:                 # material ran out
                chunk = np.pad(chunk, ((0, frames - len(chunk)), (0, 0)))
                if end >= len(self.material.samples):
                    self.playing = False
            if not self.playing:
                # the material is silent but the room is not: keep feeding zeros
                # so the tail rings out instead of being cut off
                chunk = np.zeros((frames, self.channels))
            chunk = self._match_channels(chunk, self.channels)
            out = self.limit(self._turbulence(self.conv.process(chunk) * self.gain))
            peak = float(np.max(np.abs(out))) if len(out) else 0.0
            self.peak = max(0.9 * self.peak, peak)
            outdata[:] = out[:frames]
        except Exception as exc:                    # pragma: no cover
            self.error = f"audio callback: {type(exc).__name__}: {exc}"
            self.playing = False
            self.stop_stream()
