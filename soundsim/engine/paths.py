"""Turning a Scene into acoustic paths.

The generator walks the space graph.  For each space it emits the paths that
carry sound from one point of that space to another, and for each coupling
element it cascades them:

    p_receiver = SUM_elements [ p_source -> element ] x tau_element(f) x [ element -> p_receiver ]

Cascading matters physically: the wall/window is driven by the *whole* interior
field -- direct sound plus every interior reflection plus the reverberant tail --
so what you hear outside is the interior signal filtered by the panel's
transmission loss AND coloured by the source room's reverberation.  That is the
"music from the neighbouring flat" signature: muffled, bass-heavy, with extra
ringing whose decay time belongs to the *other* room.

Energy bookkeeping (what keeps the simulation honest):

    E_target = E_direct * (1 + 16 pi r^2 / A)        (diffuse-field energy balance)
    E_tail   = E_target - SUM(|g_early| ^ 2)

so the synthesised response carries exactly the reverberant-to-direct ratio the
room's absorption implies, and its measured T60 equals the Sabine (+ air) value
even though only a few reflection orders are computed explicitly.
"""

from __future__ import annotations

import numpy as np

from .bands import BAND_CENTRES
from .rir import Path, Tail
from .scene import Coupling, Room, Scene
from ..acoustics import atmosphere, diffraction as diffr, ground as gnd
from ..acoustics import image_source, transmission
from ..acoustics.constants import speed_of_sound
from ..acoustics.materials import absorber, panel as get_panel

AIR_AREA_COEF = 0.9214          # A_air = 0.9214 * alpha[dB/m] * V  (see rooms.py)


class PathBuilder:
    """Stateful helper holding the bands and the environmental conditions."""

    def __init__(self, scene: Scene, band_centres=BAND_CENTRES):
        self.scene = scene
        self.bands = np.asarray(band_centres, dtype=np.float64)
        self.nb = len(self.bands)
        w = scene.weather
        self.c = speed_of_sound(w.temp_c)
        # atmospheric absorption coefficient per band, dB/m
        self.alpha_atm = atmosphere.attenuation_coefficient(
            self.bands, w.temp_c, w.rh_pct, w.pressure_pa)
        # wind shortens/lengthens the effective path a little; modelled as the
        # classic effective-sound-speed change along the direct ray
        self.wind = w.effective_wind_ms

    # ------------------------------------------------------------------
    # primitive gains
    # ------------------------------------------------------------------
    def spreading_geometric(self, distance_m) -> np.ndarray:
        """Pure 1/r spreading, no atmosphere: the free-field yardstick.

        Level figures in the metrics are quoted against *this*, so that
        atmospheric absorption, ground interference, screens and partitions all
        count as part of what the environment does.
        """
        return 1.0 / max(float(distance_m), 1e-3)

    def spreading(self, distance_m) -> np.ndarray:
        """Spherical spreading 1/r plus atmospheric absorption, per band."""
        d = max(float(distance_m), 1e-3)
        g = 1.0 / d
        return g * np.power(10.0, -self.alpha_atm * d / 20.0)

    def source_response(self) -> np.ndarray:
        """The source's own low-frequency roll-off, per band.

        Every gain in the engine is multiplied by this, because the room is not
        the only filter in the chain: a loudspeaker rated 55 Hz at -3 dB simply
        does not put energy into a 16 Hz room mode, no matter how long that mode
        rings.  Modelled as two cascaded 2nd-order high-pass sections (the
        displacement response of a sealed box).
        """
        f3 = max(self.scene.source.f3_db, 1e-3)
        w = self.bands / f3
        return w ** 2 / np.sqrt(1.0 + w ** 4)     # Butterworth 2nd order, -3 dB at f3

    def delay(self, distance_m) -> float:
        """Travel time, including the first-order wind effect on the effective
        sound speed along the ray."""
        d = max(float(distance_m), 1e-3)
        return d / max(self.c + self.wind, 1.0)

    # ------------------------------------------------------------------
    # one space: source point -> receiver point
    # ------------------------------------------------------------------
    def space_paths(self, space_id: str, src, dst, order: int,
                    drive: bool = False):
        """Paths (and a tail spec) for one space.

        ``drive=True`` means ``dst`` is a wall element being *driven* by this
        space, which changes only the diffuse energy reference (the element sits
        on the boundary, where the reverberant intensity is ``pc/4``).
        """
        room = self.scene.room(space_id)
        if room is None:
            return self._outdoor_paths(src, dst), None
        return self._room_paths(room, src, dst, order, drive)

    def _room_paths(self, room: Room, src, dst, order: int, drive: bool):
        src_l = room.local(src)
        dst_l = room.local(dst)
        paths: list[Path] = []

        # ---- direct ------------------------------------------------------
        d_dir = float(np.linalg.norm(dst_l - src_l))
        paths.append(Path(self.delay(d_dir), self.spreading(d_dir),
                          kind="direct", label=f"{room.id}:direct", order=0))

        # ---- early reflections (image sources) ---------------------------
        if order > 0:
            imgs = image_source.image_source_paths(src_l, dst_l, room.dims, order)
            gains = image_source.path_gains(imgs, self.bands, room.absorbers_by_axis(),
                                           alpha_atm_db_per_band=self.alpha_atm)
            for k in range(imgs.count):
                if imgs.order[k] == 0:
                    continue
                paths.append(Path(self.delay(imgs.distance[k]), gains[k],
                                  kind="reflection",
                                  label=f"{room.id}:order{imgs.order[k]}",
                                  order=int(imgs.order[k])))

        # ---- diffuse tail from the room's own energy balance -------------
        tail = None
        if self.scene.diffuse_tail:
            tail = self._room_tail(room, paths, d_dir, drive)
        return paths, tail

    def _room_tail(self, room: Room, paths: list[Path], d_dir: float, drive: bool):
        surfaces = room.surface_list()
        areas = np.array([s.area_m2 for s in surfaces])
        alpha = np.array([s.absorber.alpha_at(self.bands) for s in surfaces])
        A = np.sum(alpha * areas[:, None], axis=0)
        A = A + AIR_AREA_COEF * self.alpha_atm * room.volume
        e_dir = np.abs(paths[0].gains) ** 2
        # reverberant energy at the receiver: 16 pi r^2 / A times the direct
        # energy (Q = 1).  For an element on the boundary the drive is the
        # incident intensity, whose energy reference is the same 16 pi r^2 / A
        # relation applied at the element's distance.
        ratio = 16.0 * np.pi * d_dir**2 / np.maximum(A, 1e-6)
        e_early = np.sum(np.abs(np.array([p.gains for p in paths])) ** 2, axis=0)
        e_tail = np.maximum(e_dir * ratio - e_early, 0.0)
        t60 = 0.161 * room.volume / np.maximum(A, 1e-6)
        onset = max(p.delay_s for p in paths) if paths else 0.0
        return Tail(energy=e_tail, t60=t60, onset_s=onset)

    def _outdoor_paths(self, src, dst):
        """Direct ray, ground image (complex reflection), screens, reflectors."""
        src = np.asarray(src, float)
        dst = np.asarray(dst, float)
        paths: list[Path] = []

        d_dir = float(np.linalg.norm(dst - src))
        paths.append(Path(self.delay(d_dir), self.spreading(d_dir),
                          kind="direct", label="outdoor:direct", order=0))

        # ---- ground image -----------------------------------------------
        g = self.scene.outdoors.ground_obj()
        if g.flow_resistivity < 1e9:
            mirror = src.copy()
            mirror[2] = -src[2]
            d_img = float(np.linalg.norm(dst - mirror))
            if d_img > 1e-3:
                graze = np.arctan2(src[2] + dst[2], max(np.hypot(dst[0] - src[0],
                                                                 dst[1] - src[1]), 1e-6))
                R = np.abs(gnd.reflection_coefficient(
                    self.bands, g, np.pi / 2.0 - graze, self.scene.weather.temp_c))
                paths.append(Path(self.delay(d_img), R * self.spreading(d_img),
                                  kind="reflection", label="outdoor:ground", order=1))

        # ---- screens (Maekawa over the top edge) -------------------------
        for bar in self.scene.barriers:
            edge = bar.edge_points(src, dst)[0]
            excess = float(np.linalg.norm(edge - src) + np.linalg.norm(dst - edge) - d_dir)
            if excess <= -1e-6:
                continue                      # screen not in the way
            g_d = diffr.maekawa_transfer(max(excess, 0.0), self.bands,
                                         self.scene.weather.temp_c)
            d_bent = float(np.linalg.norm(edge - src) + np.linalg.norm(dst - edge))
            paths.append(Path(self.delay(d_bent),
                              np.abs(g_d) * self.spreading(d_bent),
                              kind="diffraction", label=f"screen:{bar.name}", order=1))

        # ---- large exterior reflectors (facades) -------------------------
        for ref in self.scene.outdoors.reflectors:
            p0 = np.asarray(ref.point, float)
            n = np.asarray(ref.normal, float)
            n = n / np.linalg.norm(n)
            rel = src - p0
            if np.dot(rel, n) <= 0:
                continue                       # source behind the reflector
            mirror = src - 2.0 * np.dot(rel, n) * n
            d_img = float(np.linalg.norm(dst - mirror))
            paths.append(Path(self.delay(d_img), 0.85 * self.spreading(d_img),
                              kind="reflection", label=f"reflector:{ref.name}",
                              order=1))
        return paths

    # ------------------------------------------------------------------
    # coupling elements
    # ------------------------------------------------------------------
    def element_gain(self, elem: Coupling, src_in, dst_out) -> np.ndarray:
        """sqrt(S tau / 2 pi): pressure gain per metre from the driven element.

        The element is driven by the pressure that exists at its position, and
        re-radiates the transmitted power into a hemisphere, which is where the
        ``2 pi`` comes from.  Solid elements use the plate transmission loss at
        the actual incidence angle for the direct drive and the diffuse-field
        value for reverberant drive; openings use the aperture transmission of
        the air plug.
        """
        pos = np.asarray(elem.position, float)
        if elem.is_opening:
            tau = transmission.aperture_transmission(self.bands, elem.area_m2,
                                                      self.scene.weather.temp_c)
            n = np.asarray(elem.normal, float)
            n = n / max(np.linalg.norm(n), 1e-9)
            ray = np.asarray(dst_out, float) - pos
            d = float(np.linalg.norm(ray))
            # no direction given (or the receiver sits on the element): radiate
            # along the element normal, which is the peak of the baffled piston
            angle = 0.0 if d < 1e-6 else float(
                np.arccos(np.clip(abs(np.dot(ray, n)) / d, 0.0, 1.0)))
            ddir = np.abs(transmission.piston_directivity(
                self.bands, elem.area_m2, angle, self.scene.weather.temp_c))
            return np.sqrt(tau * elem.area_m2 / (2.0 * np.pi)) * ddir

        if elem.panel_key is None:
            return np.zeros(self.nb)
        p = get_panel(elem.panel_key)
        inc = self._incidence_angle(elem, src_in)
        if elem.second_panel_key and elem.cavity_m > 0:
            tl = transmission.tl_double_leaf_db(
                self.bands, p, get_panel(elem.second_panel_key), elem.cavity_m,
                field_incidence=False, temp_c=self.scene.weather.temp_c)
        else:
            tau = transmission.transmission_coefficient(
                inc, self.bands, p, temp_c=self.scene.weather.temp_c)
            tl = -10.0 * np.log10(np.clip(tau, 1e-12, 1.0))
        tau_lin = 10.0 ** (-tl / 10.0)
        return np.sqrt(tau_lin * elem.area_m2 / (2.0 * np.pi))

    def element_gain_diffuse(self, elem: Coupling) -> np.ndarray:
        """Same, for an element driven by a reverberant (diffuse) field.

        No directivity factor: a diffuse drive re-radiates over the whole
        hemisphere, which is exactly what the ``2 pi`` in the spreading term
        already accounts for.
        """
        if elem.is_opening:
            tau = transmission.aperture_transmission(self.bands, elem.area_m2,
                                                      self.scene.weather.temp_c)
            return np.sqrt(tau * elem.area_m2 / (2.0 * np.pi))
        if elem.panel_key is None:
            return np.zeros(self.nb)
        p = get_panel(elem.panel_key)
        if elem.second_panel_key and elem.cavity_m > 0:
            tl = transmission.tl_double_leaf_db(
                self.bands, p, get_panel(elem.second_panel_key), elem.cavity_m,
                field_incidence=True, temp_c=self.scene.weather.temp_c)
        else:
            tl = transmission.transmission_loss_db(self.bands, p, field_incidence=True,
                                                   temp_c=self.scene.weather.temp_c)
        tau_lin = 10.0 ** (-tl / 10.0)
        return np.sqrt(tau_lin * elem.area_m2 / (2.0 * np.pi))

    def _incidence_angle(self, elem: Coupling, src) -> float:
        """Angle from the element normal of the ray arriving from ``src``."""
        pos = np.asarray(elem.position, float)
        n = np.asarray(elem.normal, float)
        n = n / max(np.linalg.norm(n), 1e-9)
        ray = pos - np.asarray(src, float)
        d = max(np.linalg.norm(ray), 1e-6)
        return float(np.arccos(np.clip(abs(np.dot(ray, n)) / d, 0.0, 1.0)))

    def inset_to_room(self, room: Room, elem: Coupling) -> np.ndarray:
        """Pull an element point 3 cm inside a room (avoids degenerate images)."""
        pos = np.asarray(elem.position, float)
        n = np.asarray(elem.normal, float)
        n = n / max(np.linalg.norm(n), 1e-9)
        p = pos - n * 0.03
        if not room.contains(p):
            p = pos + n * 0.03
        return p


def _prune(paths: list[Path], keep: int) -> list[Path]:
    """Keep the strongest paths; return (kept, energy dropped per band).

    Multiplying two image-source sets is a genuine physical cascade, but the
    product is combinatorial: order 5 on both sides of a partition is hundreds of
    thousands of second-order pairs whose individual contributions are tiny.  The
    strongest are kept as discrete paths and the discarded energy is handed to the
    diffuse tail rather than allowed to disappear.
    """
    if len(paths) <= keep:
        return paths, np.zeros(len(paths[0].gains)) if paths else np.zeros(0)
    e = np.array([np.abs(p.gains) ** 2 for p in paths])          # (n, nbands)
    total = e.sum(axis=0)
    order = np.argsort(-e.sum(axis=1))
    kept = [paths[i] for i in order[:keep]]
    dropped = np.maximum(total - e[order[:keep]].sum(axis=0), 0.0)
    return kept, dropped


def _band_energy(paths: list[Path]) -> np.ndarray:
    if not paths:
        return np.zeros(0)
    return np.sum([np.abs(p.gains) ** 2 for p in paths], axis=0)


def build_paths(scene: Scene, band_centres=BAND_CENTRES) -> tuple[list[Path], list[Tail]]:
    """All transmission paths and diffuse tails for a scene."""
    pb = PathBuilder(scene, band_centres)
    src_space = scene.space_at(scene.source.position)
    dst_space = scene.space_at(scene.receiver)

    paths, tails = _route(pb, src_space, dst_space, scene.source.position,
                          scene.receiver, 0)
    _apply_source_response(pb, paths, tails)
    return paths, tails


MAX_CASCADE_PAIRS = 3000
MAX_SEGMENT_PATHS = 55      # per stage, before the product


def _route(pb: PathBuilder, space_a: str, space_b: str, pt_a, pt_b,
           depth: int) -> tuple[list[Path], list[Tail]]:
    """Response from a point in space_a to a point in space_b, through the graph."""
    scene = pb.scene
    order = int(scene.ism_order)
    if space_a == space_b:
        paths, tail = pb.space_paths(space_a, pt_a, pt_b, order)
        return paths, ([tail] if tail else [])

    elements = scene.couplings_between(space_a, space_b)
    if elements:
        out_p, out_t = [], []
        for elem in elements:
            p, t = _cascade(pb, elem, space_a, space_b, pt_a, pt_b)
            out_p += p
            out_t += t
        out_p, resid = _prune(out_p, MAX_CASCADE_PAIRS)
        if np.any(resid > 0):
            out_t.append(Tail(resid, np.full(len(resid), 0.2),
                              onset_s=max(p.delay_s for p in out_p)))
        return out_p, out_t

    # no shared element: try one intermediate space (room -> corridor -> room)
    if depth < 2:
        for mid in list(scene.rooms) + [scene.outdoors.id]:
            if mid in (space_a, space_b):
                continue
            e1 = scene.couplings_between(space_a, mid)
            e2 = scene.couplings_between(mid, space_b)
            if not (e1 and e2):
                continue
            out_p, out_t = [], []
            for c1 in e1:
                for c2 in e2:
                    p, t = _double_cascade(pb, c1, c2, space_a, mid, space_b,
                                           pt_a, pt_b)
                    out_p += p
                    out_t += t
            if out_p:
                out_p, resid = _prune(out_p, MAX_CASCADE_PAIRS)
                if np.any(resid > 0):
                    out_t.append(Tail(resid, np.full(len(resid), 0.2),
                                      onset_s=max(p.delay_s for p in out_p)))
                return out_p, out_t

    # nothing connects them: free field, so the app never goes silently wrong
    d = float(np.linalg.norm(np.asarray(pt_b, float) - np.asarray(pt_a, float)))
    return ([Path(pb.delay(d), pb.spreading(d), kind="direct",
                  label="unconnected:free-field")], [])


def _cascade(pb: PathBuilder, elem: Coupling, space_a: str, space_b: str,
             pt_a, pt_b) -> tuple[list[Path], list[Tail]]:
    """One element between the two spaces: the full drive x transfer x onward cascade."""
    scene = pb.scene
    order = int(scene.ism_order)
    room_a, room_b = scene.room(space_a), scene.room(space_b)
    elem_a = pb.inset_to_room(room_a, elem) if room_a else elem.position
    elem_b = pb.inset_to_room(room_b, elem) if room_b else elem.position

    drive_paths, drive_tail = pb.space_paths(space_a, pt_a, elem_a, order, drive=True)
    onward_paths, onward_tail = pb.space_paths(space_b, elem_b, pt_b, order)
    drive_paths, _ = _prune(drive_paths, MAX_SEGMENT_PATHS)
    onward_paths, _ = _prune(onward_paths, MAX_SEGMENT_PATHS)

    g_direct_elem = np.abs(drive_paths[0].gains)
    g_elem_recv = np.abs(onward_paths[0].gains)
    g_dir_elem = pb.element_gain(elem, pt_a, elem_b)
    g_diff_elem = pb.element_gain_diffuse(elem)

    paths: list[Path] = []
    tails: list[Tail] = []
    for dp in drive_paths:
        for op in onward_paths:
            g = g_dir_elem if dp.kind == "direct" else g_diff_elem
            paths.append(Path(dp.delay_s + op.delay_s, dp.gains * g * op.gains,
                              kind=_combine_kind(dp.kind, op.kind),
                              label=f"{dp.label}>{elem.label or elem.kind}>{op.label}",
                              order=dp.order + op.order))
    if drive_tail is not None:
        e = drive_tail.energy * g_diff_elem**2 * g_elem_recv**2
        if np.any(e > 0):
            tails.append(Tail(e, drive_tail.t60,
                              drive_tail.onset_s + onward_paths[0].delay_s))
    if onward_tail is not None:
        e = g_direct_elem**2 * g_dir_elem**2 * onward_tail.energy \
            / np.maximum(g_elem_recv**2, 1e-30)
        if np.any(e > 0):
            tails.append(Tail(e, onward_tail.t60, onward_tail.onset_s))
    if drive_tail is not None and onward_tail is not None:
        ratio = onward_tail.energy / np.maximum(g_elem_recv**2, 1e-30)
        e = drive_tail.energy * g_diff_elem**2 * g_elem_recv**2 * ratio
        if np.any(e > 0):
            tails.append(Tail(e, np.maximum(drive_tail.t60, onward_tail.t60),
                              drive_tail.onset_s + onward_paths[0].delay_s))
    return paths, tails


def _double_cascade(pb: PathBuilder, c1: Coupling, c2: Coupling, space_a: str,
                    space_mid: str, space_b: str, pt_a, pt_b):
    """Two elements with an intermediate space between them (A -> M -> B)."""
    scene = pb.scene
    ra, rm, rb = scene.room(space_a), scene.room(space_mid), scene.room(space_b)
    p1a = pb.inset_to_room(ra, c1) if ra else c1.position
    p1m = pb.inset_to_room(rm, c1) if rm else c1.position
    p2m = pb.inset_to_room(rm, c2) if rm else c2.position
    p2b = pb.inset_to_room(rb, c2) if rb else c2.position

    drive, drive_tail = pb.space_paths(space_a, pt_a, p1a, int(scene.ism_order), True)
    mid, mid_tail = pb.space_paths(space_mid, p1m, p2m, min(2, int(scene.ism_order)))
    onward, onward_tail = pb.space_paths(space_b, p2b, pt_b, int(scene.ism_order))
    # bound the three-way product before it is formed
    drive, _ = _prune(drive, MAX_SEGMENT_PATHS)
    mid, _ = _prune(mid, 12)
    onward, _ = _prune(onward, MAX_SEGMENT_PATHS)

    t1_dir = pb.element_gain(c1, pt_a, pt_b)
    t1_diff = pb.element_gain_diffuse(c1)
    t2_dir = pb.element_gain(c2, pt_a, pt_b)
    t2_diff = pb.element_gain_diffuse(c2)

    paths: list[Path] = []
    tails: list[Tail] = []
    for dp in drive:
        for mp in mid:
            for op in onward:
                g = (t1_dir if dp.kind == "direct" else t1_diff) * \
                    (t2_dir if (mp.kind == "direct" and op.kind == "direct") else t2_diff)
                paths.append(Path(dp.delay_s + mp.delay_s + op.delay_s,
                                  dp.gains * g * mp.gains * op.gains,
                                  kind="transmission+transmission",
                                  label=f"{dp.label}>{c1.label}>{mp.label}>{c2.label}>{op.label}",
                                  order=dp.order + mp.order + op.order))
    gmid_dir = np.abs(mid[0].gains)
    gon = np.abs(onward[0].gains)
    gd = np.abs(drive[0].gains)
    if drive_tail is not None:
        e = drive_tail.energy * t1_diff**2 * gmid_dir**2 * t2_diff**2 * gon**2
        if np.any(e > 0):
            tails.append(Tail(e, drive_tail.t60,
                              drive_tail.onset_s + mid[0].delay_s + onward[0].delay_s))
    if mid_tail is not None:
        e = gd**2 * t1_diff**2 * mid_tail.energy * t2_diff**2 * gon**2
        if np.any(e > 0):
            tails.append(Tail(e, mid_tail.t60,
                              mid_tail.onset_s + onward[0].delay_s))
    if onward_tail is not None:
        e = gd**2 * t1_dir**2 * gmid_dir**2 * t2_dir**2 * onward_tail.energy \
            / np.maximum(gon**2, 1e-30)
        if np.any(e > 0):
            tails.append(Tail(e, onward_tail.t60, onward_tail.onset_s))
    return paths, tails


def _apply_source_response(pb: PathBuilder, paths: list[Path], tails: list[Tail]):
    """Fold the source's own frequency response into every path and tail.

    Applied exactly once per scene: the transducer is part of the chain, and the
    room should not be blamed (or credited) for what the loudspeaker cannot play.
    """
    r = pb.source_response()
    for p in paths:
        p.gains = p.gains * r
    for t in tails:
        t.energy = t.energy * r ** 2
    return paths, tails


def _combine_kind(a: str, b: str) -> str:
    if a == "direct" and b == "direct":
        return "transmission"
    if "reflection" in (a, b):
        return "transmission+reflection"
    return a if a == b else f"{a}+{b}"
