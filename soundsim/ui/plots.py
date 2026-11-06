"""Minimal dependency-free plotting widgets (QPainter only).

Two plots are enough to see what the simulation is doing: the impulse response
with its decay curve, and the 1/3-octave band balance against the free-field
reference (with measured versus analytic reverberation times).

The grid is placed on round numbers rather than on fractions of the range --
"1.84805k" is not something the eye can use as a reference, and half the point
of these plots is reading a frequency or a delay off them by eye.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QPainter, QPainterPath,
                           QPen)
from PySide6.QtWidgets import QSizePolicy, QWidget


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
        # Small enough that the track list can be dragged over most of the
        # column: a 90 px plot is still readable, and these are secondary to
        # "which of the four plots do I want to look at right now".
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
            p.end()
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

        # axis labels and grid, at decades (log) or at an even round step
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        fm = QFontMetrics(f)
        grid = QPen(QColor("#2b2f38"), 1, Qt.DotLine)
        lab = QColor("#8a90a0")
        for v in _ticks(y0, y1):
            yy = int(to_px(x0, v).y())
            if not (m["t"] - 1 <= yy <= m["t"] + ph + 1):
                continue
            p.setPen(grid)
            p.drawLine(int(m["l"]), yy, int(m["l"] + pw), yy)
            p.setPen(lab)
            p.drawText(2, yy + 4, self._fmt(v))
        for v in _ticks(x0, x1, log=self.xlog):
            xx = int(to_px(v, y0).x())
            if not (m["l"] - 1 <= xx <= m["l"] + pw + 1):
                continue
            p.setPen(grid)
            p.drawLine(xx, int(m["t"]), xx, int(m["t"] + ph))
            p.setPen(lab)
            text = self._fmt(v)
            # the last tick label would otherwise hang off the right edge
            tx = max(m["l"] - 6, min(xx - 16, m["l"] + pw - fm.horizontalAdvance(text)))
            p.drawText(tx, int(m["t"] + ph + 16), text)
        if self.xlabel:
            p.setPen(lab)
            p.drawText(QRectF(m["l"], m["t"] + ph + 3, pw, 14),
                       Qt.AlignRight | Qt.AlignVCenter, self.xlabel)

        # zero line
        if y0 < 0 < y1:
            zy = int(to_px(0 if not self.xlog else x0, 0.0).y())
            p.setPen(QPen(QColor("#5a6070"), 1, Qt.DashLine))
            p.drawLine(int(m["l"]), zy, int(m["l"] + pw), zy)

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

        # title left, the hint that fits right: they used to run into each other
        p.setPen(QColor("#c9cfda"))
        p.drawText(m["l"], 15, self.title)
        hint = "   ".join(self.hints[:2])
        if hint:
            x = m["l"] + fm.horizontalAdvance(self.title) + 14
            room = max(m["l"] + pw - x, 40)
            p.setPen(lab)
            p.drawText(QRectF(x, 3, room, 16), Qt.AlignRight | Qt.AlignVCenter,
                       fm.elidedText(hint, Qt.ElideRight, room))
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


# ---------------------------------------------------------------------------
# tick placement
# ---------------------------------------------------------------------------
def _nice_step(raw: float) -> float:
    """The smallest 1/2/2.5/5 x 10^n that is at least *raw*."""
    if raw <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(raw))
    for mul in (1, 2, 2.5, 5, 10):
        if raw <= mul * e * (1 + 1e-12):
            return mul * e
    return 10.0 * e


def _ticks(lo: float, hi: float, want: int = 5, log: bool = False) -> list[float]:
    """Round numbers to put grid lines on, inside [lo, hi].

    A log axis gets decades and their 2x and 5x subdivisions, which is how a
    frequency axis is actually read; a linear one gets an even round step.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return [float(lo)]
    if log:
        if lo <= 0:
            lo = 1e-3
        out, e = [], math.floor(math.log10(lo))
        while 10.0 ** e <= hi * 1.0001 and e < 15:
            for mul in (1.0, 2.0, 5.0):
                v = mul * 10.0 ** e
                if lo * 0.999 <= v <= hi * 1.0001:
                    out.append(float(v))
            e += 1
        while len(out) > want + 2:                 # thin out if decades crowd
            out = out[::2]
        return out
    step = _nice_step((hi - lo) / max(want - 1, 1))
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + step * 1e-6 and len(out) < 40:
        out.append(round(v, 10))
        v += step
    return out
