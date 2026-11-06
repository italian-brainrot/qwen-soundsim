"""Minimal dependency-free plotting widgets (QPainter only).

Two plots are enough to see what the simulation is doing: the impulse response
with its decay curve, and the 1/3-octave band balance against the free-field
reference (with measured versus analytic reverberation times).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class Plot(QWidget):
    """A small line/step plot with optional log-x axis."""

    def __init__(self, parent=None, title: str = "", xlog: bool = False,
                 xlabel: str = "", ylabel: str = ""):
        super().__init__(parent)
        self.title = title
        self.xlog = xlog
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.series: list[tuple[np.ndarray, np.ndarray, str, str]] = []
        self.bands: list[tuple[float, float, float]] = []   # (x, y, y_alt) bars
        self.hints: list[str] = []
        self.setMinimumHeight(180)
        self.setBackgroundRole(self.backgroundRole())

    def set_series(self, series, hints=None):
        """series: list of (x, y, label, color_name)."""
        self.series = [(np.asarray(x, float), np.asarray(y, float), l, c)
                       for x, y, l, c in series]
        self.hints = list(hints or [])
        self.update()

    # -- painting -------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#16181d"))
        m = {"l": 46, "r": 10, "t": 24, "b": 26}
        pw, ph = max(w - m["l"] - m["r"], 10), max(h - m["t"] - m["b"], 10)

        xs, ys = self._range()
        if xs is None:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter, "no data")
            return
        x0, x1 = xs
        y0, y1 = ys
        if self.xlog:
            fx = lambda v: np.log2(np.clip(v, 1e-9, 1e12))
        else:
            fx = lambda v: np.asarray(v, float)
        lx0, lx1 = float(fx(x0)), float(fx(x1))
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0

        def to_px(x, y):
            tx = (float(fx(x)) - lx0) / (lx1 - lx0)
            ty = (float(y) - y0) / (y1 - y0)
            return QPointF(m["l"] + tx * pw, m["t"] + (1 - ty) * ph)

        # grid
        p.setPen(QPen(QColor("#2b2f38"), 1, Qt.DotLine))
        for k in range(5):
            yy = m["t"] + ph * k / 4
            p.drawLine(int(m["l"]), int(yy), int(m["l"] + pw), int(yy))
            p.setPen(QColor("#8a90a0"))
            p.drawText(2, int(yy) + 4, f"{y1 - (y1 - y0) * k / 4:,.0f}")
            p.setPen(QPen(QColor("#2b2f38"), 1, Qt.DotLine))
        for k in range(5):
            xx = m["l"] + pw * k / 4
            p.drawLine(int(xx), int(m["t"]), int(xx), int(m["t"] + ph))
            val = 2 ** (lx0 + (lx1 - lx0) * k / 4) if self.xlog else x0 + (x1 - x0) * k / 4
            p.setPen(QColor("#8a90a0"))
            label = self._fmt(val)
            p.drawText(int(xx) - 14, int(m["t"] + ph + 16), label)
            p.setPen(QPen(QColor("#2b2f38"), 1, Qt.DotLine))

        # zero line
        if y0 < 0 < y1:
            p.setPen(QPen(QColor("#5a6070"), 1, Qt.DashLine))
            zx = to_px(0 if not self.xlog else x0, 0.0)
            p.drawLine(int(m["l"]), int(zy := to_px(0, 0.0).y()), int(m["l"] + pw), int(zy))

        # series
        for x, y, label, color in self.series:
            if len(x) == 0:
                continue
            path = QPainterPath()
            started = False
            for xi, yi in zip(x, y):
                if not np.isfinite(yi):
                    started = False
                    continue
                pt = to_px(xi, np.clip(yi, y0, y1))
                if not started:
                    path.moveTo(pt)
                    started = True
                else:
                    path.lineTo(pt)
            p.setPen(QPen(QColor(color), 1.6))
            p.drawPath(path)

        # fonts / titles
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor("#c9cfda"))
        p.drawText(m["l"], 15, self.title)
        p.setPen(QColor("#8a90a0"))
        p.drawText(m["l"] + pw - 150, 15, "   ".join(self.hints[:2]))
        p.end()

    @staticmethod
    def _fmt(v: float) -> str:
        a = abs(v)
        if a >= 1000:
            return f"{v / 1000:g}k"
        if a >= 1:
            return f"{v:.0f}" if a >= 10 else f"{v:.1f}"
        return f"{v:.2g}"

    def _range(self):
        xs, ys = [], []
        for x, y, _, _ in self.series:
            if len(x):
                xs += [float(np.nanmin(x)), float(np.nanmax(x))]
                ys += [float(np.nanmin(y)), float(np.nanmax(y))]
        if not xs:
            return None, None
        y0, y1 = min(ys), max(ys)
        pad = 0.06 * max(y1 - y0, 1e-6)
        return (min(xs), max(xs)), (y0 - pad, y1 + pad)
