"""SoundSim desktop application.

Layout: scene presets on the left, four plots in the middle (impulse response,
decay curve, 1/3-octave balance, reverberation time per band), the physical
parameters on the right, transport along the bottom.  Changing any parameter
re-renders the room impulse response and crossfades it into the live audio, so
the effect of the physics is heard as a transition rather than as a before/after
comparison.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMainWindow,
    QPushButton, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from ..acoustics.materials import GROUNDS
from ..engine import presets, renderer
from ..engine.player import Player, demo_material, load_file
from ..engine.scene import Scene
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SoundSim — acoustic environment simulator")
        self.resize(1360, 860)
        self.sample_rate = 48_000
        self.player = Player(sample_rate=self.sample_rate, channels=2)
        self.scene: Scene = presets.make("party_next_door")
        self.base_distance = self._listener_distance()
        self.last_result = None
        self.worker: RenderWorker | None = None
        self.pending = []
        self.playlist: list[str] = []

        self._build_ui()
        self._load_material(demo_material(sample_rate=self.sample_rate))
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._start_render)
        self.request_render()
        self._tick = QTimer(self, interval=60)
        self._tick.timeout.connect(self._poll_player)

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
        self.scene_desc = QLabel(self.scene.description)
        self.scene_desc.setWordWrap(True)
        self.scene_desc.setStyleSheet("color:#9aa3b2; font-size:11px;")
        left.addWidget(QLabel("<b>Environment</b>"))
        left.addWidget(self.scene_list, 1)
        left.addWidget(self.scene_desc)

        # ---- middle: plots + transport ----------------------------------
        self.plot_ir = Plot(title="Impulse response", ylabel="dB")
        self.plot_decay = Plot(title="Decay curve (backward integration)", ylabel="dB")
        self.plot_bands = Plot(title="Band balance vs free field", xlog=True,
                               ylabel="dB")
        self.plot_t60 = Plot(title="Reverberation time per band", xlog=True,
                             ylabel="s")
        grid = QWidget()
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(self.plot_ir, 0, 0)
        g.addWidget(self.plot_decay, 0, 1)
        g.addWidget(self.plot_bands, 1, 0)
        g.addWidget(self.plot_t60, 1, 1)

        transport = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setShortcut(QKeySequence(Qt.Key_Space))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_files = QPushButton("Add files…")
        self.btn_files.clicked.connect(self._add_files)
        self.btn_folder = QPushButton("Add folder…")
        self.btn_folder.clicked.connect(self._add_folder)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 1000)
        self.seek.sliderMoved.connect(self._seek)
        self.time_label = QLabel("0:00 / 0:00")
        self.level = QLabel("")
        self.level.setMinimumWidth(150)
        transport.addWidget(self.btn_play)
        transport.addWidget(self.btn_stop)
        transport.addWidget(self.btn_files)
        transport.addWidget(self.btn_folder)
        transport.addWidget(self.seek, 1)
        transport.addWidget(self.time_label)
        transport.addWidget(self.level)

        self.playlist_view = QListWidget()
        self.playlist_view.setMaximumHeight(88)
        self.playlist_view.itemDoubleClicked.connect(
            lambda it: self._load_material(load_file(it.text(), self.sample_rate)))

        middle = QVBoxLayout()
        middle.addWidget(grid, 3)
        middle.addWidget(self.playlist_view, 0)
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
        self.in_ground = QComboBox(); self.in_ground.addItems(sorted(GROUNDS)); self.in_ground.setCurrentText(self.scene.outdoors.ground)
        self.in_cap = QDoubleSpinBox(); self.in_cap.setRange(0.5, 20); self.in_cap.setDecimals(1); self.in_cap.setValue(6.0); self.in_cap.setSuffix(" s")
        self.in_delay = QCheckBox(); self.in_delay.setChecked(False)
        self.in_auto_gain = QCheckBox(); self.in_auto_gain.setChecked(True)
        self.in_auto_gain.setToolTip("Map the predicted received level onto a "
                                     "comfortable playback level. Pure monitoring "
                                     "calibration -- the response and the dB "
                                     "figures are untouched.")
        for w in (self.in_spl, self.in_f3, self.in_distance, self.in_order, self.in_temp,
                  self.in_rh, self.in_wind, self.in_ground, self.in_cap):
            if isinstance(w, QComboBox):
                w.currentTextChanged.connect(self.request_render)
            else:
                w.valueChanged.connect(self.request_render)
        for w in (self.in_tail, self.in_modal, self.in_delay, self.in_auto_gain):
            w.toggled.connect(self.request_render)
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
        form.addRow(PARAM_LABELS["ir_cap"], self.in_cap)
        form.addRow(PARAM_LABELS["delay_comp"], self.in_delay)
        form.addRow("Auto monitor gain (playback only)", self.in_auto_gain)

        self.device_box = QComboBox()
        try:
            import sounddevice as sd
            infos = sd.query_devices()
            self.device_box.addItem("(system default)", None)
            for i, d in enumerate(infos):
                if d.get("max_output_channels", 0) > 0:
                    self.device_box.addItem(f"{i}: {d['name'][:44]}", i)
        except Exception:
            pass
        self.device_box.currentIndexChanged.connect(self._pick_device)

        metrics = QGroupBox("Measured on the synthesised response")
        mform = QFormLayout(metrics)
        self.metric_labels: dict[str, QLabel] = {}
        for key in ("t60", "t60_mid", "c50", "c80", "d50", "ts", "itdg",
                    "schroeder", "modal", "paths", "irlen", "spl", "loss",
                    "arrival", "render_time"):
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
        root.addWidget(splitter)

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
        self.scene_desc.setText(self.scene.description)
        self.request_render()

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
        irs = list(irs)
        if self.in_delay.isChecked():
            irs = [_trim_delay(ir, self.sample_rate) for ir in irs]
        self.player.set_impulse_responses(irs, crossfade_s=0.04)
        if self.in_auto_gain.isChecked():
            self.player.set_monitor_gain_for(res.metrics["total_spl_db"])
        self.last_result = res
        self._refresh_plots(res)
        self._refresh_metrics(res, dt)
        self.status.setText(f"scene rendered in {dt:.2f} s")
        self.status.setStyleSheet("color:#7fd18a;")
        if self.pending:
            self.pending.clear()
            self.request_render()

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
        self.level.setToolTip(f"monitor gain {self.player.gain_db:+.1f} dB "
                              "(playback calibration)")

    # ------------------------------------------------------------------
    # transport
    def _load_material(self, mat):
        self.player.set_material(mat)
        self.seek.setValue(0)
        self._poll_player()

    def _toggle_play(self):
        self.player.toggle()
        self.btn_play.setText("Pause" if self.player.playing else "Play")

    def _stop(self):
        self.player.pause()
        self.player.set_position_seconds(0)
        self.btn_play.setText("Play")

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
                           ("  CLIP" if self.player.clip_flag else ""))

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add audio", os.path.expanduser("~"),
            "Audio (*.wav *.flac *.ogg *.aiff *.au *.mp3 *.m4a *.opus)")
        self._extend_playlist(files)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder", os.path.expanduser("~"))
        if not d:
            return
        exts = (".wav", ".flac", ".ogg", ".aiff", ".au", ".mp3", ".m4a")
        files = sorted(f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                       if f.lower().endswith(exts))
        self._extend_playlist(files)

    def _extend_playlist(self, files):
        for f in files:
            self.playlist_view.addItem(f)
        if files and not self.playlist:
            self._load_material(load_file(files[0], self.sample_rate))
        self.playlist.extend(files)

    def _pick_device(self):
        dev = self.device_box.currentData()
        self.player.stop_stream()
        self.player.device = dev
        if self.player.playing:
            self.player.start()

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.player.stop_stream()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.quit()
            self.worker.wait(4000)
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


def _mmss(s: float) -> str:
    s = max(0.0, float(s))
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def run() -> int:                                     # pragma: no cover
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()

