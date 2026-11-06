"""SoundSim desktop application.

Layout: scene presets on the left, the now-playing bar and four plots in the
middle (impulse response, decay curve, 1/3-octave balance, reverberation time
per band) with the track list under them, the physical parameters on the right,
transport along the bottom.  Changing any parameter re-renders the room impulse
response and crossfades it into the live audio, so the effect of the physics is
heard as a transition rather than as a before/after comparison.

The track list lives in a splitter, so it can be as big as you want, and audio
gets in by every route that takes text: a typed or pasted path, a chooser you
can type into, drag and drop, and Ctrl+V.
"""

from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import Qt, QSettings, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from scipy.signal import butter, sosfiltfilt

from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QPushButton, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from ..acoustics.materials import GROUNDS
from ..engine import presets, renderer
from ..engine.player import (SCAN_LIMIT, SUPPORTED, Player, demo_material,
                             expand_audio_paths, load_file, pick_default_output)
from ..engine.playlist import Playlist
from ..engine.scene import Scene
from .media import DIM, PlaylistPanel, apply_dark_theme
from .plots import Plot

PARAM_LABELS = {
    "spl": "Source level (dB SPL @ 1 m)",
    "f3": "Source −3 dB (Hz)",
    "distance": "Listener distance (m)",
    "order": "Image-source order",
    "tail": "Diffuse tail (energy balanced)",
    "modal": "Modal low band (exact box solution)",
    "temp": "Air temperature (°C)",
    "rh": "Relative humidity (%)",
    "wind": "Wind along the path (m/s, + = tailwind)",
    "ground": "Outdoor ground",
    "ir_cap": "Response length cap (s)",
    "delay_comp": "Compensate propagation delay (non-physical)",
}


# renders still running when the window closes; kept referenced until they
# finish, because destroying a live QThread aborts the process
_ORPHANS: list = []


class RenderWorker(QThread):
    """Scene -> impulse responses, off the UI thread."""
    finished_ir = Signal(object, object, float)     # irs, result, wall time
    failed = Signal(str)

    def __init__(self, scene: Scene, sample_rate: int, stereo: bool, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.sample_rate = sample_rate
        self.stereo = stereo

    def run(self):
        t0 = time.perf_counter()
        try:
            if self.stereo:
                irs, res = renderer.render_pair(self.scene, self.sample_rate)
            else:
                res = renderer.render(self.scene, self.sample_rate)
                irs = [res.ir, res.ir]
        except Exception as exc:                      # pragma: no cover
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished_ir.emit(irs, res, time.perf_counter() - t0)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        apply_dark_theme(self)
        self.setWindowTitle("SoundSim — acoustic environment simulator")
        self.resize(1360, 860)
        self.sample_rate = 48_000
        self.player = Player(sample_rate=self.sample_rate, channels=2,
                             device=pick_default_output())
        self.scene_key = "party_next_door"
        self.scene: Scene = presets.make(self.scene_key)
        self.base_distance = self._listener_distance()
        self.last_result = None
        self.worker: RenderWorker | None = None
        self.pending = []
        self.tracks = Playlist()
        self.scene_edited = False
        self._was_playing = False
        # injectable so the tests can start from an empty state file
        self.settings = settings or QSettings("SoundSim", "SoundSim")

        self._build_ui()
        self._restore_session()
        if self.player.material is None:
            self._load_material(demo_material(sample_rate=self.sample_rate))
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._start_render)
        self.request_render()
        # the position readout, the level meter and the end-of-track hand-off
        # all ride on this tick; it used to be created and never started
        self._tick = QTimer(self, interval=60)
        self._tick.timeout.connect(self._poll_player)
        self._tick.start()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- left: scenes ------------------------------------------------
        left = QVBoxLayout()
        self.scene_list = QListWidget()
        for key, (title, sub) in presets.PRESETS.items():
            self.scene_list.addItem(f"{title}")
        self.scene_list.currentRowChanged.connect(self._pick_preset)
        self.scene_list.blockSignals(True)
        self.scene_list.setCurrentRow(list(presets.PRESETS).index(self.scene_key))
        self.scene_list.blockSignals(False)
        self.scene_desc = QLabel(self.scene.description)
        self.scene_desc.setWordWrap(True)
        self.scene_desc.setStyleSheet("color:#9aa3b2; font-size:11px;")
        env_head = QHBoxLayout()
        env_head.addWidget(QLabel("<b>Environment</b>"))
        env_head.addStretch(1)
        self.btn_reset_env = QPushButton("reset")
        self.btn_reset_env.setToolTip("Put this environment's own numbers back. "
                                      "Enabled once a physics control has been "
                                      "moved; double-clicking the environment in "
                                      "the list does the same.")
        self.btn_reset_env.setEnabled(False)
        self.btn_reset_env.clicked.connect(
            lambda: self._pick_preset(self.scene_list.currentRow()))
        env_head.addWidget(self.btn_reset_env)
        self.scene_list.itemDoubleClicked.connect(
            lambda it: self._pick_preset(self.scene_list.currentRow()))
        left.addLayout(env_head)
        left.addWidget(self.scene_list, 1)
        left.addWidget(self.scene_desc)

        # ---- middle: what is playing, the plots, the playlist ------------
        now = QFrame()
        now.setObjectName("nowbar")
        now.setStyleSheet("QFrame#nowbar { background:#242832; border:1px solid "
                          "#39414d; border-radius:6px; }")
        bar = QHBoxLayout(now)
        bar.setContentsMargins(10, 6, 10, 6)
        bar.setSpacing(10)
        glyph = QLabel("\u266a")
        glyph.setStyleSheet("font-size:20px; color:#6fb3ff;")
        bar.addWidget(glyph)
        who = QVBoxLayout()
        who.setSpacing(1)
        self.track_title = QLabel("no audio loaded")
        self.track_title.setStyleSheet(
            "font-size:15px; font-weight:600; color:#e6ebf2;")
        self.track_sub = QLabel("")
        self.track_sub.setStyleSheet(f"font-size:11px; color:{DIM};")
        who.addWidget(self.track_title)
        who.addWidget(self.track_sub)
        bar.addLayout(who, 1)
        where = QVBoxLayout()
        where.setSpacing(1)
        self.env_name = QLabel("")
        self.env_name.setAlignment(Qt.AlignRight)
        self.env_name.setStyleSheet(
            "font-size:15px; font-weight:600; color:#f0c070;")
        self.env_sub = QLabel("")
        self.env_sub.setAlignment(Qt.AlignRight)
        self.env_sub.setStyleSheet(f"font-size:11px; color:{DIM};")
        where.addWidget(self.env_name)
        where.addWidget(self.env_sub)
        bar.addSpacing(12)
        bar.addLayout(where)

        self.plot_ir = Plot(title="Impulse response", xlabel="ms")
        self.plot_decay = Plot(title="Decay curve (backward integration)",
                               xlabel="ms")
        self.plot_bands = Plot(title="Band balance vs free field", xlog=True,
                               xlabel="Hz")
        self.plot_t60 = Plot(title="Reverberation time per band", xlog=True,
                             xlabel="Hz")
        grid = QWidget()
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(self.plot_ir, 0, 0)
        g.addWidget(self.plot_decay, 0, 1)
        g.addWidget(self.plot_bands, 1, 0)
        g.addWidget(self.plot_t60, 1, 1)

        self.panel = PlaylistPanel()
        self.panel.activate_row.connect(self._activate_row)
        self.panel.add_paths.connect(self._add_paths)
        self.panel.remove_rows.connect(self._remove_rows)
        self.panel.clear_list.connect(self._clear_playlist)

        # the track list is a splitter pane, not a strip: drag the handle and it
        # takes whatever room you want, plots and all
        self.stack = QSplitter(Qt.Vertical)
        self.stack.addWidget(grid)
        self.stack.addWidget(self.panel)
        self.stack.setCollapsible(0, False)
        self.stack.setCollapsible(1, False)
        self.stack.setHandleWidth(10)
        self.stack.setSizes([560, 300])

        # two rows: the seek bar gets the whole width it needs, the buttons
        # their own room, instead of each other's leftovers
        seekrow = QHBoxLayout()
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 1000)
        self.seek.sliderMoved.connect(self._seek)
        self.seek.setToolTip("Click or drag to move inside the track")
        self.time_label = QLabel("0:00 / 0:00")
        seekrow.addWidget(self.time_label)
        seekrow.addWidget(self.seek, 1)

        transport = QHBoxLayout()
        transport.setSpacing(6)
        self.btn_prev = QPushButton("\u25c0 Prev")
        self.btn_prev.setToolTip("Previous track (Alt+Left) — at less than 3 s "
                                 "into a track it goes back a track, after that "
                                 "it restarts this one, as everywhere else")
        self.btn_prev.clicked.connect(lambda: self._skip(-1))
        self.btn_play = QPushButton("\u25b6 Play")
        self.btn_play.setToolTip("Play / pause (Space)")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop = QPushButton("\u25a0 Stop")
        self.btn_stop.setToolTip("Stop and rewind")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_next = QPushButton("Next \u25b6")
        self.btn_next.setToolTip("Next track (Alt+Right)")
        self.btn_next.clicked.connect(lambda: self._skip(1))
        self.btn_shuffle = QPushButton("Shuffle")
        self.btn_shuffle.setCheckable(True)
        self.btn_shuffle.setToolTip("Play in random order (Alt+S). The current "
                                    "track stays where it is; files added while "
                                    "playing are dealt in ahead of the cursor.")
        self.btn_shuffle.toggled.connect(self._toggle_shuffle)
        self.btn_repeat = QPushButton("Repeat: off")
        self.btn_repeat.setToolTip("Cycle off / all / one (Alt+R). \u201cone\u201d "
                                    "rewinds the track when it reaches its end; "
                                    "skipping forward still moves on.")
        self.btn_repeat.clicked.connect(self._cycle_repeat)
        for b, w in ((self.btn_prev, 76), (self.btn_play, 86),
                     (self.btn_stop, 76), (self.btn_next, 76),
                     (self.btn_shuffle, 76), (self.btn_repeat, 104)):
            b.setMinimumWidth(w)
        self.counter = QLabel("0 / 0")
        self.counter.setToolTip("Position in the playlist / how many tracks")
        self.counter.setStyleSheet(f"color:{DIM};")
        self.level = QLabel("")
        self.level.setMinimumWidth(150)
        transport.addWidget(self.btn_prev)
        transport.addWidget(self.btn_play)
        transport.addWidget(self.btn_stop)
        transport.addWidget(self.btn_next)
        transport.addWidget(self.btn_shuffle)
        transport.addWidget(self.btn_repeat)
        transport.addStretch(1)
        transport.addWidget(self.counter)
        transport.addWidget(self.level)

        middle = QVBoxLayout()
        middle.addWidget(now)
        middle.addWidget(self.stack, 1)
        middle.addLayout(seekrow)
        middle.addLayout(transport)

        # ---- right: parameters + metrics ---------------------------------
        params = QGroupBox("Physics")
        form = QFormLayout(params)
        self.in_spl = QDoubleSpinBox(); self.in_spl.setRange(40, 140); self.in_spl.setValue(self.scene.source.spl_at_1m_db)
        self.in_f3 = QDoubleSpinBox(); self.in_f3.setRange(10, 500); self.in_f3.setValue(self.scene.source.f3_db); self.in_f3.setSuffix(" Hz")
        self.in_distance = QDoubleSpinBox(); self.in_distance.setRange(0.3, 2000); self.in_distance.setDecimals(2); self.in_distance.setValue(self.base_distance); self.in_distance.setSuffix(" m")
        self.in_order = QSpinBox(); self.in_order.setRange(0, 12); self.in_order.setValue(self.scene.ism_order)
        self.in_tail = QCheckBox(); self.in_tail.setChecked(self.scene.diffuse_tail)
        self.in_modal = QCheckBox(); self.in_modal.setChecked(True)
        self.in_temp = QDoubleSpinBox(); self.in_temp.setRange(-30, 50); self.in_temp.setValue(self.scene.weather.temp_c); self.in_temp.setSuffix(" °C")
        self.in_rh = QDoubleSpinBox(); self.in_rh.setRange(1, 100); self.in_rh.setValue(self.scene.weather.rh_pct); self.in_rh.setSuffix(" %")
        self.in_wind = QDoubleSpinBox(); self.in_wind.setRange(-10, 20); self.in_wind.setValue(self.scene.weather.wind_speed_ms); self.in_wind.setSuffix(" m/s")
        self.in_turb = QComboBox()
        for lab, val in (("still air", 0.0), ("slight", 0.15), ("moderate", 0.35),
                         ("unsettled", 0.8)):
            self.in_turb.addItem(lab, val)
        self.in_turb.setCurrentIndex(2)
        self.in_turb.setToolTip(
            "Eddies make the ray path length wander by centimetres. A "
            "centimetre is nothing at 100 Hz and a third of a wavelength at "
            "10 kHz, so the direct/ground interference survives in the bass and "
            "averages away in the treble. This is a static effect on the "
            "response; it does not modulate the audio.")
        self.in_waver = QComboBox()
        for lab, val in (("off", 0.0), ("slight", 0.12), ("moderate", 0.25)):
            self.in_waver.addItem(lab, val)
        self.in_waver.setCurrentIndex(0)
        self.in_waver.setToolTip(
            "Optional: the same wander, seen as it happens -- the treble of "
            "very distant sound breathes over a second or two while the bass "
            "holds. Off by default: it is real, but it is also the least "
            "certain number in the model.")
        self.in_ground = QComboBox(); self.in_ground.addItems(sorted(GROUNDS)); self.in_ground.setCurrentText(self.scene.outdoors.ground)
        self.in_cap = QDoubleSpinBox(); self.in_cap.setRange(0.5, 20); self.in_cap.setDecimals(1); self.in_cap.setValue(6.0); self.in_cap.setSuffix(" s")
        self.in_delay = QCheckBox(); self.in_delay.setChecked(False)
        self.in_lowcut = QComboBox()
        for hz, lab in ((0, "off (full physics)"), (30, "30 Hz"), (50, "50 Hz"),
                        (80, "80 Hz")):
            self.in_lowcut.addItem(lab, hz)
        self.in_lowcut.setCurrentIndex(1)      # 30 Hz: below what speakers make
        self.in_lowcut.setToolTip(
            "Playback-only bass trim for speakers that cannot reach the "
            "lowest octave. The simulation keeps everything; the response "
            "and the dB figures are unaffected.")
        self.in_auto_gain = QCheckBox(); self.in_auto_gain.setChecked(True)
        self.in_auto_gain.setToolTip("Map the predicted received level onto a "
                                     "comfortable playback level. Pure monitoring "
                                     "calibration -- the response and the dB "
                                     "figures are untouched.")
        for w in (self.in_spl, self.in_f3, self.in_distance, self.in_order, self.in_temp,
                  self.in_rh, self.in_wind, self.in_ground, self.in_turb,
                  self.in_waver, self.in_cap):
            if isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._param_changed)
            else:
                w.valueChanged.connect(self._param_changed)
        self.in_lowcut.currentIndexChanged.connect(lambda *_: self._apply_monitoring())
        for w in (self.in_tail, self.in_modal):          # physics: re-render
            w.toggled.connect(self._param_changed)
        for w in (self.in_delay, self.in_auto_gain):     # monitoring only
            w.toggled.connect(lambda *_: self._apply_monitoring())
        form.addRow(PARAM_LABELS["spl"], self.in_spl)
        form.addRow(PARAM_LABELS["f3"], self.in_f3)
        form.addRow(PARAM_LABELS["distance"], self.in_distance)
        form.addRow(PARAM_LABELS["order"], self.in_order)
        form.addRow(PARAM_LABELS["tail"], self.in_tail)
        form.addRow(PARAM_LABELS["modal"], self.in_modal)
        form.addRow(PARAM_LABELS["temp"], self.in_temp)
        form.addRow(PARAM_LABELS["rh"], self.in_rh)
        form.addRow(PARAM_LABELS["wind"], self.in_wind)
        form.addRow(PARAM_LABELS["ground"], self.in_ground)
        form.addRow("Air turbulence (path wander)", self.in_turb)
        form.addRow("Treble wavering over time", self.in_waver)
        form.addRow(PARAM_LABELS["ir_cap"], self.in_cap)
        form.addRow(PARAM_LABELS["delay_comp"], self.in_delay)
        form.addRow("Auto monitor gain (playback only)", self.in_auto_gain)
        form.addRow("Monitor low-cut (playback only)", self.in_lowcut)

        self.device_box = QComboBox()
        try:
            import sounddevice as sd
            infos = sd.query_devices()
            self.device_box.addItem("(auto: session sink)", None)
            for i, d in enumerate(infos):
                if d.get("max_output_channels", 0) > 0:
                    host = sd.query_hostapis()[d["hostapi"]]["name"]
                    self.device_box.addItem(f"[{host}] {d['name'][:40]}", i)
            want = self.player.device
            row = self.device_box.findData(want)
            if row >= 0:
                self.device_box.blockSignals(True)
                self.device_box.setCurrentIndex(row)
                self.device_box.blockSignals(False)
        except Exception:
            pass
        self.device_box.currentIndexChanged.connect(self._pick_device)
        self._good_device = self.device_box.currentIndex()

        metrics = QGroupBox("Measured on the synthesised response")
        mform = QFormLayout(metrics)
        self.metric_labels: dict[str, QLabel] = {}
        for key in ("t60", "t60_mid", "c50", "c80", "d50", "ts", "itdg",
                    "schroeder", "modal", "paths", "irlen", "spl", "loss",
                    "arrival", "render_time", "latency", "bass"):
            lab = QLabel("—")
            self.metric_labels[key] = lab
            mform.addRow(_metric_name(key), lab)

        self.status = QLabel("idle")
        self.status.setStyleSheet("color:#7fd18a;")
        right = QVBoxLayout()
        right.addWidget(params)
        right.addWidget(QLabel("Output device"))
        right.addWidget(self.device_box)
        right.addWidget(metrics)
        right.addWidget(self.status)

        splitter = QSplitter()
        w1, w2, w3 = QWidget(), QWidget(), QWidget()
        w1.setLayout(left); w2.setLayout(middle); w3.setLayout(right)
        splitter.addWidget(w1); splitter.addWidget(w2); splitter.addWidget(w3)
        splitter.setSizes([230, 760, 330])
        self.splitter = splitter
        root.addWidget(splitter)

        self.setAcceptDrops(True)
        self._install_shortcuts()
        self._update_env_label()
        self._refresh_playlist_view()
        self._refresh_plots(None)

    # ------------------------------------------------------------------
    # scene handling
    def _listener_distance(self) -> float:
        s = np.asarray(self.scene.source.position, float)
        r = np.asarray(self.scene.receiver, float)
        return float(np.linalg.norm(r - s))

    def _set_listener_distance(self, dist: float):
        s = np.asarray(self.scene.source.position, float)
        r = np.asarray(self.scene.receiver, float)
        d = r - s
        n = np.linalg.norm(d)
        if n < 1e-9:
            d = np.array([1.0, 0.0, 0.0]); n = 1.0
        self.scene.receiver = tuple(s + d / n * float(dist))

    def _pick_preset(self, row: int):
        if row < 0:
            return
        name = list(presets.PRESETS)[row]
        self.scene_key = name
        self.scene = presets.make(name)
        self.scene.diffuse_tail = self.in_tail.isChecked()
        self.base_distance = self._listener_distance()
        self.in_distance.blockSignals(True)
        self.in_distance.setValue(self.base_distance)
        self.in_distance.blockSignals(False)
        self.in_spl.blockSignals(True)
        self.in_spl.setValue(self.scene.source.spl_at_1m_db)
        self.in_spl.blockSignals(False)
        self.in_order.blockSignals(True)
        self.in_order.setValue(self.scene.ism_order)
        self.in_order.blockSignals(False)
        self.in_ground.blockSignals(True)
        self.in_ground.setCurrentText(self.scene.outdoors.ground)
        self.in_ground.blockSignals(False)
        self.in_temp.blockSignals(True)
        self.in_temp.setValue(self.scene.weather.temp_c)
        self.in_temp.blockSignals(False)
        self.in_rh.blockSignals(True)
        self.in_rh.setValue(self.scene.weather.rh_pct)
        self.in_rh.blockSignals(False)
        self.in_wind.blockSignals(True)
        self.in_wind.setValue(self.scene.weather.wind_speed_ms)
        self.in_wind.blockSignals(False)
        self.in_waver.blockSignals(True)
        self.in_waver.setCurrentIndex(0)
        self.in_waver.blockSignals(False)
        self.in_turb.blockSignals(True)
        row = self.in_turb.findData(self.scene.weather.turbulence)
        if row >= 0:
            self.in_turb.setCurrentIndex(row)
        self.in_turb.blockSignals(False)
        self.scene_desc.setText(self.scene.description)
        self.scene_edited = False
        self._update_env_label()
        self.request_render()

    def _param_changed(self, *args):
        """A physics control moved: the scene is no longer the preset as shipped."""
        self.scene_edited = True
        self._update_env_label()
        self.request_render()

    def _update_env_label(self):
        """Say which environment is applied, and whether it has been edited."""
        title, sub = presets.PRESETS.get(self.scene_key,
                                         (self.scene.name, ""))
        self.env_name.setText(title + (" \u00b7 edited" if self.scene_edited
                                       else ""))
        self.env_name.setToolTip(
            self.scene.description +
            ("\n\nSome physics control differs from this environment's "
             "defaults. Selecting it again in the list resets them."
             if self.scene_edited else ""))
        self.env_sub.setText("environment — " + sub)
        self.btn_reset_env.setEnabled(self.scene_edited)
        self._update_window_title()

    def _update_window_title(self):
        """The title bar names both, so alt-tab says what is what."""
        m = self.player.material
        track = "" if m is None else (os.path.basename(m.path) if m.path
                                      else m.name)
        title, _ = presets.PRESETS.get(self.scene_key, (self.scene.name, ""))
        self.setWindowTitle(f"{track} — {title} — SoundSim" if track
                            else "SoundSim — acoustic environment simulator")

    def _build_scene(self) -> Scene:
        sc = self.scene
        sc.source.spl_at_1m_db = self.in_spl.value()
        sc.source.f3_db = self.in_f3.value()
        sc.ism_order = self.in_order.value()
        sc.diffuse_tail = self.in_tail.isChecked()
        sc.weather.temp_c = self.in_temp.value()
        sc.weather.rh_pct = self.in_rh.value()
        sc.weather.wind_speed_ms = self.in_wind.value()
        sc.outdoors.ground = self.in_ground.currentText()
        sc.weather.turbulence = self.in_turb.currentData()
        self.player.set_turbulence(self.in_waver.currentData())
        self._set_listener_distance(self.in_distance.value())
        return sc

    def request_render(self, *args):
        self._debounce.start(220)

    def _start_render(self):
        if self.worker is not None and self.worker.isRunning():
            self.pending.append(True)
            return
        scene = self._build_scene()
        self.status.setText("rendering…")
        self.status.setStyleSheet("color:#e0c060;")
        self.worker = RenderWorker(scene, self.sample_rate, True, self)
        self.worker.finished_ir.connect(self._got_ir)
        self.worker.failed.connect(lambda m: self.status.setText(f"render failed: {m}"))
        self.worker.start()

    def _got_ir(self, irs, res, dt):
        self._raw_irs = [np.asarray(ir, float) for ir in irs]
        self.last_result = res
        self._apply_monitoring()
        self._refresh_plots(res)
        self._refresh_metrics(res, dt)
        self.status.setText(f"scene rendered in {dt:.2f} s")
        self.status.setStyleSheet("color:#7fd18a;")
        if self.pending:
            self.pending.clear()
            self.request_render()

    def _apply_monitoring(self):
        """Playback stage only: bass trim, delay trim, level.  The response the
        plots and metrics describe is the unmonitored physical one."""
        if not getattr(self, "_raw_irs", None):
            return
        irs = list(self._raw_irs)
        cut = self.in_lowcut.currentData()
        if cut:
            sos = butter(2, cut / (self.sample_rate / 2.0), btype="high",
                         output="sos")
            irs = [sosfiltfilt(sos, ir) for ir in irs]
        if self.in_delay.isChecked():
            irs = [_trim_delay(ir, self.sample_rate) for ir in irs]
        self.player.set_impulse_responses(irs, crossfade_s=0.04)
        if self.in_auto_gain.isChecked():
            self.player.set_monitor_gain_for(self.last_result.metrics["total_spl_db"])
        else:
            self.player.gain, self.player.gain_db = 1.0, 0.0

    # ------------------------------------------------------------------
    # plots / metrics
    def _refresh_plots(self, res):
        if res is None:
            return
        fs = res.sample_rate
        ir = res.ir
        env = np.abs(ir)
        t_ms = np.arange(len(ir)) / fs * 1000.0
        # decimate for drawing
        step = max(1, len(ir) // 3000)
        db = 20 * np.log10(np.maximum(env, 1e-9))
        peak = db.max()
        self.plot_ir.set_series([(t_ms[::step], np.clip(db[::step], peak - 120, peak),
                                  "IR", "#6fb3ff")],
                               hints=[f"{len(ir)/fs*1000:.0f} ms response"])
        # backward-integrated decay curve
        e = np.cumsum((ir[::-1] ** 2))[::-1]
        e = e / max(e[0], 1e-30)
        dec = 10 * np.log10(np.maximum(e, 1e-12))
        self.plot_decay.set_series([(t_ms[::step], dec[::step], "decay", "#f0a050")],
                                   hints=[f"T60 (mid bands) {_fmt(res.metrics.get('t60_mid_db'))} s"])
        bands = res.metrics.get("bands_hz", res.band_centres)
        resp = res.metrics.get("band_response_db")
        if resp is not None:
            self.plot_bands.set_series([(bands, resp, "band", "#7fd18a")],
                                       hints=["dB relative to free field at the same distance"])
        sab = None
        for k, v in res.metrics.items():
            if k.startswith("sabine_t60_"):
                sab = v
                break
        series = [(bands, res.metrics["t60_bands"], "measured", "#f07a7a")]
        if sab is not None:
            series.append((bands, np.asarray(sab), "Sabine", "#8ab8ff"))
        self.plot_t60.set_series(series,
                                 hints=[f"modal below {_fmt(res.metrics.get('modal_cutoff_hz', 0))} Hz"])

    def _refresh_metrics(self, res, dt):
        m = res.metrics
        setm = self.metric_labels
        setm["t60"].setText(_fmt(m.get("t60_iso")) + " s (broadband)")
        setm["t60_mid"].setText(_fmt(m.get("t60_mid_db")) + " s (400 Hz–2 kHz)")
        setm["c50"].setText(_fmt(m.get("c50")) + " dB")
        setm["c80"].setText(_fmt(m.get("c80")) + " dB")
        setm["d50"].setText(_fmt(m.get("d50")) + " %")
        setm["ts"].setText(_fmt(m.get("ts")) + " dB")
        setm["itdg"].setText(_fmt(m.get("itdg_ms")) + " ms")
        sch = None
        for k, v in m.items():
            if k.startswith("schroeder_"):
                sch = v
                break
        setm["schroeder"].setText(_fmt(sch) + " Hz")
        setm["modal"].setText(f"{_fmt(m.get('modal_cutoff_hz', 0))} Hz "
                              f"({m.get('modal_modes_used', 0)} modes)")
        setm["paths"].setText(str(m.get("n_paths")))
        setm["irlen"].setText(f"{len(res.ir) / res.sample_rate:.2f} s")
        setm["spl"].setText(_fmt(m.get("total_spl_db")) + " dB SPL")
        loss = m.get("environment_loss_db")
        setm["loss"].setText(
            ("attenuated" if loss > 0 else "amplified") +
            f" {abs(loss):.1f} dB vs free field")
        first = min([p.delay_s for p in res.paths], default=0.0)
        setm["arrival"].setText(f"{first * 1000:.1f} ms first arrival")
        setm["render_time"].setText(f"{dt:.2f} s")
        setm["latency"].setText(f"{self.player.output_latency_seconds * 1000:.0f} ms "
                                f"(fixed pipeline delay)")
        # how much of the played-out energy is sub-bass: the number to look at
        # when a speaker buzzes
        if getattr(self.player, "conv", None) and self.player.conv.irs:
            ir = self.player.conv.irs[0]
            fr = np.fft.rfftfreq(len(ir), 1.0 / self.sample_rate)
            S = np.abs(np.fft.rfft(ir)) ** 2
            tot = S[fr > 15].sum()
            if tot > 0:
                sub = 10.0 * np.log10(max(S[fr < 60].sum(), 1e-30) / tot)
                setm["bass"].setText(f"{sub:+.1f} dB of output energy below 60 Hz")
            else:
                setm["bass"].setText("--")
        self.level.setToolTip(
            f"monitor gain {self.player.gain_db:+.1f} dB "
            f"(playback calibration, limited by {self.player.limited_by})")

    # ------------------------------------------------------------------
    # transport
    def _say(self, text: str, bad: bool = False):
        """One status line, colour-coded: it is the only feedback channel."""
        self.status.setText(text)
        self.status.setStyleSheet("color:" + ("#e07a7a;" if bad else "#7fd18a;"))

    def _load_material(self, mat) -> bool:
        """Never let a bad decode leave the transport pointing at nothing."""
        try:
            self.player.set_material(mat)
        except Exception as exc:
            self._say(f"could not play {getattr(mat, 'name', '?')}: {exc}",
                      bad=True)
            return False
        self.seek.setValue(0)
        # the monitor gain is derived from the material's own peak, so new
        # material means the (unchanged) response has to be re-applied to it
        self._apply_monitoring()
        self._update_now_playing()
        self._poll_player()
        return True

    def _load_path(self, path: str) -> bool:
        """Decode a playlist entry and hand it to the transport."""
        try:
            mat = load_file(path, self.sample_rate)
        except Exception as exc:
            self._say(f"cannot decode {os.path.basename(path)}: {exc}", bad=True)
            return False
        return self._load_material(mat)

    def _update_play_button(self):
        self.btn_play.setText("\u2016 Pause" if self.player.playing
                              else "\u25b6 Play")

    def _toggle_play(self):
        try:
            self.player.toggle()
        except Exception as exc:
            self._say(f"cannot start audio: {exc}", bad=True)
            return
        self._update_play_button()

    def _start_playback(self):
        """Resume after loading a track while the transport was running."""
        try:
            self.player.play()
        except Exception as exc:
            self._say(f"cannot start audio: {exc}", bad=True)
        self._update_play_button()

    def _stop(self):
        self.player.pause()
        self.player.set_position_seconds(0)
        self._was_playing = False
        self._update_play_button()

    def _seek(self):
        frac = self.seek.value() / 1000.0
        self.player.set_position_seconds(frac * self.player.duration_seconds)

    def _poll_player(self):
        pos, dur = self.player.position_seconds, self.player.duration_seconds
        if dur > 0 and not self.seek.isSliderDown():
            self.seek.setValue(int(1000 * pos / dur))
        self.time_label.setText(f"{_mmss(pos)} / {_mmss(dur)}")
        pk = self.player.peak
        db = 20 * np.log10(max(pk, 1e-6))
        bars = int(np.clip((db + 60) / 60 * 20, 0, 20))
        self.level.setText("▮" * bars + f" {db:.0f} dB" +
                           ("  LIMIT" if self.player.clip_flag else ""))
        if self.player.error:
            self._say(self.player.error, bad=True)
        # the callback stops itself when the material runs out: that is the cue
        # to do what shuffle and repeat say, not a reason to simply go quiet
        mat = self.player.material
        ended = (mat is not None and self._was_playing and not self.player.playing
                 and self.player.index >= len(mat.samples))
        self._was_playing = self.player.playing
        if ended:
            self._advance()

    # ------------------------------------------------------------------
    # which audio: putting things in, picking them, stepping through them
    def _add_paths(self, specs):
        """Typed, pasted or dropped text -> tracks.

        One route for all of them, because each can carry a file, a folder, a
        glob or a whole pasted list, and which one happened to be used should
        not change what ends up in the list.
        """
        joined = "\n".join(str(s) for s in
                           ([specs] if isinstance(specs, str) else list(specs)))
        if not joined.strip():
            return
        paths = expand_audio_paths(joined)
        if not paths:
            self._say("no playable file found in: " + _clip(joined, 60), bad=True)
            return
        was_empty = not self.tracks
        added = self.tracks.add(paths)
        self.panel.remember(paths[0])
        if was_empty and added:
            self.tracks.jump_to(added[0])
            self._load_path(added[0])
        self._refresh_playlist_view()
        if not added:
            self._say(f"those {len(paths)} tracks are already in the list")
            return
        note = f"added {len(added)} track" + ("s" if len(added) != 1 else "")
        if len(paths) >= SCAN_LIMIT:
            note += f" — the folder scan is capped at {SCAN_LIMIT}"
        self._say(note)

    def _refresh_playlist_view(self):
        """Mirror the model into the list, the note and the counter."""
        self.panel.set_paths(self.tracks.items)
        self.panel.set_current(self.tracks.index)
        n = len(self.tracks)
        cur = self.tracks.index
        self.panel.set_note(
            f"{n} track" + ("s" if n != 1 else "") +
            (f" · playing {cur + 1}" if cur is not None else ""))
        self.counter.setText(f"{(cur + 1) if cur is not None else 0} / {n}")

    def _activate_row(self, index: int):
        """Double-click or Enter in the list: play that row, whatever the
        transport was doing before."""
        self._play_row(index, resume=True)

    def _play_row(self, index: int, resume: bool | None = None):
        """Load the track at a row.  *resume* True starts it, None leaves the
        transport as it was, False leaves it paused."""
        path = self.tracks.path_at(index)
        if path is None:
            return
        self.tracks.select(index)
        if not self._load_path(path):
            return
        self.panel.set_current(self.tracks.index)
        self._refresh_playlist_view()
        if resume or (resume is None and (self.player.playing or
                                          self._was_playing)):
            self._start_playback()
        else:
            self._update_play_button()

    def _skip(self, direction: int):
        """Prev / next.  Repeat-one never traps a deliberate skip."""
        if not self.tracks:
            self._say("nothing in the playlist yet — type a path above")
            return
        if direction < 0 and self.player.position_seconds > 3.0:
            self.player.set_position_seconds(0.0)     # back to the top first
            self.seek.setValue(0)
            return
        resume = self.player.playing or self._was_playing
        path = self.tracks.next() if direction > 0 else self.tracks.prev()
        if path is None:
            self._refresh_playlist_view()
            self._say("at the " + ("last" if direction > 0 else "first") +
                      " track — repeat is off")
            return
        self._play_row(self.tracks.index, resume=resume)

    def _advance(self):
        """A track ended by itself: shuffle and repeat decide what is next."""
        path = self.tracks.next(auto=True)
        if path is None:
            self._update_play_button()
            self._refresh_playlist_view()
            self.status.setText("end of the playlist")
            self.status.setStyleSheet(f"color:{DIM};")
            return
        if self._load_path(path):
            self.panel.set_current(self.tracks.index)
            self._refresh_playlist_view()
            self._start_playback()

    def _toggle_shuffle(self, on: bool):
        self.tracks.set_shuffle(on)
        self.panel.set_current(self.tracks.index)
        self._accent(self.btn_shuffle, on)
        self._say("playing in random order" if on else "playing in the order added")

    def _cycle_repeat(self):
        mode = self.tracks.cycle_repeat()
        self.btn_repeat.setText(f"Repeat: {mode}")
        self._accent(self.btn_repeat, mode != "off")
        self._say(f"repeat {mode}")

    def _accent(self, button, on: bool):
        """Fusion's 'checked' is a sunken grey; on transport buttons that reads
        as disabled, so an active mode is marked with the accent colour."""
        button.setStyleSheet("background:#3a6ea5; color:#ffffff; "
                             "border-color:#4d84c0;" if on else "")

    def _remove_rows(self, rows):
        self.tracks.remove(rows)
        self._refresh_playlist_view()
        self._say(f"removed {len(rows)} track" + ("s" if len(rows) != 1 else ""))

    def _clear_playlist(self):
        self.tracks.clear()
        self._refresh_playlist_view()
        self._say("playlist cleared — the loaded track keeps playing")

    def _update_now_playing(self):
        """What the transport is holding, where it came from and in what shape.
        Without this the four plots have no stated subject."""
        m = self.player.material
        if m is None:
            self.track_title.setText("no audio loaded")
            self.track_sub.setText("type a path below, or press Play for the "
                                   "built-in demo material")
            self.track_title.setToolTip("")
            self._update_window_title()
            return
        chans = m.samples.shape[1] if m.samples.ndim > 1 else 1
        self.track_title.setText(_clip(os.path.basename(m.path) if m.path
                                       else m.name, 70))
        self.track_title.setToolTip(m.path or "built-in demo material")
        need_ffmpeg = bool(m.path) and \
            os.path.splitext(m.path)[1].lower() not in SUPPORTED
        self.track_sub.setText(
            (_clip(os.path.dirname(m.path), 70) if m.path
             else "built into the app") +
            f"  ·  {_mmss(m.duration)}  ·  {m.sample_rate / 1000.0:g} kHz  ·  " +
            ("mono" if chans == 1 else "stereo" if chans == 2
             else f"{chans} channels") +
            ("  ·  decoded via ffmpeg" if need_ffmpeg else ""))
        self._update_window_title()

    # ------------------------------------------------------------------
    # keys
    def _install_shortcuts(self):
        """Transport keys, switched off while a text box has the focus."""
        self._shortcuts = []

        def add(seq, fn):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

        add(Qt.Key_Space, self._toggle_play)
        add(QKeySequence(Qt.Key_MediaPlay), self._toggle_play)
        add(QKeySequence(Qt.Key_MediaTogglePlayPause), self._toggle_play)
        add(QKeySequence(Qt.Key_MediaStop), self._stop)
        add(QKeySequence(Qt.Key_MediaNext), lambda: self._skip(1))
        add(QKeySequence(Qt.Key_MediaPrevious), lambda: self._skip(-1))
        add("Alt+Right", lambda: self._skip(1))
        add("Alt+Left", lambda: self._skip(-1))
        add("Alt+S", self.btn_shuffle.click)
        add("Alt+R", self._cycle_repeat)
        add("Ctrl+O", self.panel.btn_add.click)
        add("Ctrl+Shift+O", self.panel.btn_folder.click)
        add("Ctrl+F", self.panel.filter_box.setFocus)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._typing_focus)

    def _typing_focus(self, _old, now):
        """While you are typing a path, Space is a space and not Play."""
        typing = isinstance(now, (QLineEdit, QAbstractSpinBox))
        for sc in self._shortcuts:
            sc.setEnabled(not typing)

    # ------------------------------------------------------------------
    # session: the list, and where it came from, kept for the next start
    def _restore_session(self):
        s = self.settings
        items = s.value("playlist") or []
        if isinstance(items, str):
            items = [items]
        items = [str(p) for p in items if os.path.isfile(str(p))]
        self.panel.set_context(str(s.value("last_dir", os.path.expanduser("~"))),
                               [str(d) for d in (s.value("recent_dirs") or [])])
        self.tracks.add(items)
        self.tracks.set_repeat(str(s.value("repeat", "off")))
        self.btn_repeat.setText(f"Repeat: {self.tracks.repeat}")
        if self.tracks.repeat != "off":
            self._accent(self.btn_repeat, True)
        shuffle = str(s.value("shuffle", "false")).lower() in ("1", "true", "yes")
        self.btn_shuffle.setChecked(shuffle)
        self.tracks.set_shuffle(shuffle)
        try:
            cur = int(s.value("current_index", 0))
        except (TypeError, ValueError):
            cur = 0
        if self.tracks:
            self.tracks.select(cur)
            if self.tracks.current():
                self._load_path(self.tracks.current())
        self._refresh_playlist_view()
        self._restore_splitter_sizes()

    def _save_session(self):
        s = self.settings
        s.setValue("playlist", self.tracks.items)
        s.setValue("current_index", self.tracks.index or 0)
        s.setValue("shuffle", self.tracks.shuffle)
        s.setValue("repeat", self.tracks.repeat)
        s.setValue("last_dir", self.panel.last_dir)
        s.setValue("recent_dirs", self.panel.recents)
        s.setValue("split_main", self.splitter.sizes())
        s.setValue("split_audio", self.stack.sizes())
        s.sync()

    def _restore_splitter_sizes(self):
        """Whoever arranged the panes was probably trying to see something."""
        for key, pane, want in (("split_main", self.splitter, [230, 760, 330]),
                                ("split_audio", self.stack, [560, 300])):
            sizes = self.settings.value(key) or []
            try:
                sizes = [int(v) for v in sizes]
            except (TypeError, ValueError):
                sizes = []
            if len(sizes) == pane.count() and all(v > 0 for v in sizes):
                pane.setSizes(sizes)
            else:
                pane.setSizes(want)

    # ------------------------------------------------------------------
    # a file or folder dropped anywhere on the window
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        md = event.mimeData()
        text = ("\n".join(u.toLocalFile() for u in md.urls() if u.isLocalFile())
                if md.hasUrls() else md.text())
        if text.strip():
            self._add_paths(text.splitlines())
            event.acceptProposedAction()

    def _pick_device(self):
        """Switch the output device, and say so if the box refuses it."""
        dev = self.device_box.currentData()
        was_playing = self.player.playing
        self.player.stop_stream()
        self.player.device = dev
        try:
            if was_playing:
                self.player.start()
            self._good_device = self.device_box.currentIndex()
            self.status.setText("output: " + str(self.device_box.currentText()))
            self.status.setStyleSheet("color:#7fd18a;")
        except Exception as exc:
            self.status.setText(f"device refused ({type(exc).__name__}); "
                                f"staying on {self.device_box.itemText(self._good_device)}")
            self.status.setStyleSheet("color:#e07a7a;")
            self.player.device = self.device_box.itemData(self._good_device)
            self.device_box.blockSignals(True)
            self.device_box.setCurrentIndex(self._good_device)
            self.device_box.blockSignals(False)
            if was_playing:
                try:
                    self.player.start()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._save_session()
        self.player.stop_stream()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.quit()
            if not self.worker.wait(4000):
                worker = self.worker
                _ORPHANS.append(worker)
                worker.finished.connect(
                    lambda: _ORPHANS.remove(worker) if worker in _ORPHANS
                    else None)
        event.accept()


def _metric_name(key: str) -> str:
    return {
        "t60": "T60 (ISO range)",
        "t60_mid": "T60 mid-band",
        "c50": "C50 clarity",
        "c80": "C80",
        "d50": "D50 definition",
        "ts": "Ts strength",
        "itdg": "ITDG",
        "schroeder": "Schroeder frequency",
        "modal": "Modal band",
        "paths": "Paths computed",
        "irlen": "Response length",
        "spl": "Received level",
        "loss": "Environment loss",
        "arrival": "First arrival",
        "render_time": "Render time",
        "latency": "Output latency",
        "bass": "Bass in output",
    }[key]


def _trim_delay(ir: np.ndarray, fs: int, floor_db: float = -46.0) -> np.ndarray:
    """Cut the silent propagation delay off the head of a response."""
    db = 20 * np.log10(np.maximum(np.abs(ir), 1e-12))
    peak = db.max()
    idx = int(np.argmax(db > peak + floor_db))
    return ir[max(idx - 16, 0):]


def _fmt(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"


def _clip(text: str, width: int = 60) -> str:
    """Shorten in the middle-ish: the end of a path is the part that matters."""
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    head = width // 3
    return text[:head] + "…" + text[-(width - head - 1):]


def _mmss(s: float) -> str:
    s = max(0.0, float(s))
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def run() -> int:                                     # pragma: no cover
    import sys
    app = QApplication(sys.argv)
    # QSettings is where the playlist and the last folder live between runs
    app.setOrganizationName("SoundSim")
    app.setApplicationName("SoundSim")
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    code = app.exec()
    # a render that was still going when the window closed: give it a moment to
    # land before the interpreter starts pulling the rug out from under it
    for worker in list(_ORPHANS):
        worker.wait(3000)
    return code

