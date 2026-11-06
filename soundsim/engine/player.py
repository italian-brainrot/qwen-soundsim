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


SUPPORTED = (".wav", ".flac", ".ogg", ".aiff", ".au", ".raw", ".w64", ".aif")


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
    def duration_seconds(self) -> float:
        if self.material is None:
            return 0.0
        return len(self.material.samples) / self.sample_rate

    # -- output level ---------------------------------------------------
    def set_monitor_gain_for(self, received_spl_db: float,
                             target_spl_db: float = 70.0,
                             floor_db: float = -12.0, ceiling_db: float = 46.0):
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
        self.gain_db = gain_db
        self.gain = 10.0 ** (gain_db / 20.0)
        return gain_db

    # -- scene response -------------------------------------------------
    def set_impulse_responses(self, irs, crossfade_s: float = 0.03):
        """Install new room response(s); safe to call while playing."""
        self.conv.set_ir(irs, crossfade_s, self.sample_rate)

    # -- transport ------------------------------------------------------
    def start(self):
        """Open the output stream, falling back to the device's own rate.

        Some hosts (and some HDMI sinks) will not accept 48 kHz; rather than
        staying silent, the player re-renders nothing but resamples the material
        and uses whatever rate the device actually wants.
        """
        if self._stream is not None:
            return
        for rate in (self.sample_rate, 44_100):
            try:
                self._stream = sd.OutputStream(
                    samplerate=rate, channels=self.channels, device=self.device,
                    blocksize=int(self.latency_s * rate), dtype="float32",
                    callback=self._callback, finished_callback=self._stream_finished)
                self._stream.start()
                if rate != self.sample_rate and self.material is not None:
                    self.sample_rate = rate
                    if self.material.sample_rate != rate:
                        self.material = load_file(self.material.path, rate) \
                            if self.material.path else Material(
                                self.material.name,
                                resample_poly(self.material.samples, rate,
                                              self.material.sample_rate, axis=0),
                                rate, self.material.path)
                return
            except Exception:
                self._stream = None
        raise RuntimeError("no output device accepted this stream")

    def _stream_finished(self, *a):
        self._stream = None

    def stop_stream(self):
        if self._stream is not None:
            s, self._stream = self._stream, None
            s.stop()
            s.close()

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

    # -- callback -------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        if self.material is None:
            outdata[:] = 0.0
            return
        need = frames + self.conv.latency_samples
        end = min(self.index + need, len(self.material.samples))
        chunk = self.material.samples[self.index:end]
        self.index = end
        if len(chunk) < need:                       # ran out: pad and stop
            chunk = np.pad(chunk, ((0, need - len(chunk)), (0, 0)))
            if end >= len(self.material.samples):
                self.playing = False
        if not self.playing:
            chunk = np.zeros((need, self.channels))
        out = self.conv.process(chunk[:, :self.channels]) * self.gain
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 1.0:
            self.clip_flag += 1
            out = np.clip(out, -1.0, 1.0)
        self.peak = max(0.9 * self.peak, peak)
        outdata[:] = out[:frames]
