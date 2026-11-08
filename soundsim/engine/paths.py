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
    E_tail   = E_target - SUM(|g_early| ^ 2)          (sum over *every* explicit path)

The direct ray is inside that sum: the explicit paths already carry it, so the
tail is what is left of the room's energy after *all* of them, not after the
reflections alone.  (An earlier revision subtracted the direct energy twice --
the target without it, the early sum with it -- which under-filled the tail by
exactly one direct-ray energy: invisible at 270 reverberant-to-direct, a real
shortfall at 1 m in a dead room.)

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
from ..acoustics import image_source, rooms as room_acoustics, transmission
from ..acoustics.constants import speed_of_sound
from ..acoustics.materials import absorber, panel as get_panel

AIR_AREA_COEF = room_acoustics.AIR_ABSORPTION_AREA_COEF   # A_air = 0.9214 a[dB/m] V


def source_roll_off(freqs_hz, f3_db: float) -> np.ndarray:
    """The loudspeaker's own low-frequency response, linear pressure gain.

    A sealed box is a 2nd-order high pass in pressure: ``w^2 / sqrt(1 + w^4)``
    with ``w = f / f3``, which is -3 dB at ``f3`` and falls at 12 dB per octave
    below it -- the velocity of the cone, not its excursion, is what radiates.
    (Earlier docstrings here claimed two cascaded sections / 4th order; the code
    has always been 2nd order, which is the physically right slope.)
    """
    w = np.asarray(freqs_hz, dtype=np.float64) / max(float(f3_db), 1e-3)
    return w ** 2 / np.sqrt(1.0 + w ** 4)


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
        rings.  See :func:`source_roll_off`.
        """
        return source_roll_off(self.bands, self.scene.source.f3_db)

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
        space.  It is reported through to the tail builder, which documents why
        the diffuse energy reference at a boundary is the same one as anywhere
        else in the room -- the distinction is real, the factor is 1.
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
        # Reverberant energy at the receiver: 16 pi r^2 / A times the direct
        # energy (Q = 1), and the total the room must hold is that plus the
        # direct sound itself -- which is why the target carries the 1 +.
        #   ``drive`` (dst is a wall element being driven by this space) does not
        # change the reference, and it would be wrong if it did: at a rigid
        # boundary the reverberant pressure doubles (+6 dB) but only half the
        # energy flux crosses a surface that lies in the plane of the wall, and
        # the classic diffuse-field intensity pc/4 is written in terms of the
        # field's own mean-square pressure.  The two cancel: the same 16 pi r^2 / A
        # relation applies at the element's distance as anywhere else in the room.
        ratio = 16.0 * np.pi * d_dir**2 / np.maximum(A, 1e-6)
        e_early = np.sum(np.abs(np.array([p.gains for p in paths])) ** 2, axis=0)
        # A diffuse field needs a dense forest of resonances to exist in.  Weyl's
        # count says a 1/3-octave band holds ~4 pi V f^2 df / c^3 modes; below
        # roughly one mode per band the *build-up* cannot reach the diffuse value,
        # and since A also shrinks with frequency an uncorrected tail dumps an
        # enormous amount of sub-60 Hz energy into the output -- inaudible, but
        # enough to pump a cone and make the limiter buzz.  The geometric early
        # reflections are left alone: they are the deterministic field.
        weight = room_acoustics.mode_count_weight(self.bands, room.volume,
                                         self.scene.weather.temp_c)
        e_tail = np.maximum((e_dir * (1.0 + ratio) - e_early) * weight, 0.0)
        t60 = 0.161 * room.volume / np.maximum(A, 1e-6)
        onset = max(p.delay_s for p in paths) if paths else 0.0
        return Tail(energy=e_tail, t60=t60, onset_s=onset)

    def _screen_insertion_loss(self, src, dst, d_dir: float):
        """Maekawa insertion loss of every screen between source and receiver.

        Returns ``(attenuation per band in dB, excess delay in s)`` -- both zero
        when nothing is in the way.  Three separate pieces of physics, in the
        order they apply:

        * a screen the ray *flies over* is not an obstacle at all.  ``Barrier``
          answers that geometrically; the Fresnel number cannot, because a path
          excess is two sides of a triangle minus the third and so is never
          negative (see the note in ``diffraction.maekawa_attenuation_db``).
        * the two edges of one flat top are two routes to the same receiver, so
          they are combined in energy.
        * several screens one behind the other are in series, which is the
          ISO 9613-2 cascade rule.
        """
        groups, bent = [], 0.0
        for bar in self.scene.barriers:
            if not bar.blocks(src, dst):
                continue
            excesses = [float(np.linalg.norm(p - src) + np.linalg.norm(dst - p) - d_dir)
                        for p in bar.edge_points(src, dst)]
            groups.append(diffr.edge_group_attenuation_db(
                excesses, self.bands, self.scene.weather.temp_c))
            bent += excesses[0]          # the near edge is the one the ray bends over
        if not groups:
            return np.zeros(self.nb), 0.0
        # the diffracted ray is longer, and that is audible as a later arrival
        return diffr.series_attenuation_db(groups), self.delay(d_dir + bent) - self.delay(d_dir)

    def _outdoor_paths(self, src, dst):
        """Direct ray, ground image (complex reflection), screens, reflectors."""
        src = np.asarray(src, float)
        dst = np.asarray(dst, float)
        paths: list[Path] = []

        d_dir = float(np.linalg.norm(dst - src))

        # ---- screens, applied as an insertion loss -----------------------
        # Maekawa / ISO 9613-2 A_bar is the loss of the *blocked ray*: a screen in
        # the shadow zone does not add a second, quieter path next to the direct
        # one, it removes the direct one and what arrives over the edge *is* the
        # direct arrival, delayed.  (Appending an attenuated copy instead makes a
        # wall add energy -- an 8 m screen across a 300 m path used to make the
        # scene louder by 0.3 dB.)  The ground ray is blocked by the same screen,
        # so it takes the same insertion loss; facade reflections are left alone
        # because they reach the receiver by a different route.
        screen_db, screen_lag = self._screen_insertion_loss(src, dst, d_dir)
        g_screen = np.power(10.0, -screen_db / 20.0)
        lag = np.full(self.nb, screen_lag)          # zeros unless something blocks

        paths.append(Path(self.delay(d_dir), self.spreading(d_dir) * g_screen,
                          kind="direct", label="outdoor:direct", order=0,
                          delay_bands=lag))

        # ---- ground image -----------------------------------------------
        g = self.scene.outdoors.ground_obj()
        # Every ground gets its image: a rigid one (asphalt, concrete, still
        # water) has R = +1, and that +6 dB over the free field is exactly why
        # sound carries over water and a car park.  Skipping the image for hard
        # grounds silently turned them into "no ground at all".
        if True:
            mirror = src.copy()
            mirror[2] = -src[2]
            d_img = float(np.linalg.norm(dst - mirror))
            if d_img > 1e-3:
                graze = np.arctan2(src[2] + dst[2], max(np.hypot(dst[0] - src[0],
                                                                 dst[1] - src[1]), 1e-6))
                Rc = gnd.reflection_coefficient(
                    self.bands, g, np.pi / 2.0 - graze, self.scene.weather.temp_c)
                # keep the reflection phase: over grass or soil the coefficient
                # rotates by more than 90 deg across the audible band, which puts
                # the ground ray partly *out of phase* with the direct one.  That
                # interference -- not the magnitude -- is what carves the midrange
                # dip that makes 300 m of air sound like 300 m of air.
                phase = np.angle(Rc)
                dphi_delay = -phase / (2.0 * np.pi * self.bands)
                # Turbulence: eddies move the wavefront, so the *path difference*
                # between the direct and the reflected ray wanders by sigma over
                # fractions of a second and their phase difference
                #     sigma_phi(f) = 2 pi f sigma / c
                # is not known.  Averaged over the wander each ray keeps its own
                # energy but the cross term is scaled by the mutual coherence
                #     gamma(f) = exp(-sigma_phi^2 / 2),
                # which is all an interference pattern physically is once the
                # medium stops holding still: the interference dies from the top
                # of the band down, exactly where the wander exceeds a fraction of
                # the wavelength, and the bass comb survives.  So the reflection is
                # split into the part still phase-locked to the direct ray
                # (amplitude gamma) and the rest, which is carried as energy; the
                # two parts hold the band energy at gamma^2 + (1 - gamma^2) = |R|^2
                # -- the knob blurs the comb, it does not swallow the reflection.
                #   A fan of five discrete delays cannot do that.  Inside a
                # 1/3-octave band it does not average the interference, it adds a
                # finer comb of its own (the pattern measured *deeper*, not
                # shallower), and dividing the amplitude by the fan size threw a
                # fixed dB away no matter how gentle the turbulence was.
                #   Calibration: sigma is the *differential* path-length wander of
                # the two rays, and it scales with the path because the eddies
                # accumulate along it.  5.5e-5 of the path -- 1.6 cm over the 300 m
                # park at turbulence 1.0, 0.6 cm at the 0.35 default.  That is a
                # couple of orders below sigma_c/c x d for a convective boundary
                # layer, because the two rays run almost side by side and only the
                # part of the wander they do *not* share moves their phase
                # difference; measurements of ground-reflected arrivals over a
                # heated surface put the residual in the 1-10 cm range over a few
                # hundred metres.  Pushing it higher does blur the comb further, but
                # it also raises the mid-band mean (filling a 10 dB null puts energy
                # back), and that is physics, not a bug -- see AUDIT.md pass 2.
                sigma_m = 5.5e-5 * d_img * max(self.scene.weather.turbulence, 0.0)
                if sigma_m > 1e-9:
                    phi = 2.0 * np.pi * self.bands * sigma_m / self.c
                    gamma = np.exp(-0.5 * phi * phi)
                else:
                    gamma = np.ones(self.nb)
                g_ground = np.abs(Rc) * self.spreading(d_img) * g_screen
                paths.append(Path(self.delay(d_img), g_ground * gamma,
                                  kind="reflection", label="outdoor:ground",
                                  order=1, delay_bands=dphi_delay + lag))
                e_lost = (1.0 - gamma * gamma) * np.abs(g_ground) ** 2
                if np.any(e_lost > 0.0):
                    paths.append(Path(self.delay(d_img), np.sqrt(e_lost),
                                      kind="reflection", label="outdoor:ground:turbulent",
                                      order=1, delay_bands=lag, coherent=False))

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
        # the plate's transmission loss is referenced to an infinite baffle; the
        # real element is finite, and below ka ~ 1 it cannot radiate what it
        # transmits, so the re-radiated power carries the piston efficiency too
        sig = transmission.radiation_efficiency(self.bands, elem.area_m2,
                                                self.scene.weather.temp_c)
        return np.sqrt(tau_lin * sig * elem.area_m2 / (2.0 * np.pi))

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
        sig = transmission.radiation_efficiency(self.bands, elem.area_m2,
                                                self.scene.weather.temp_c)
        return np.sqrt(tau_lin * sig * elem.area_m2 / (2.0 * np.pi))

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
    # ``[0]`` is the direct path only while nothing has re-sorted the list: pruning
    # returns the survivors ranked by energy, and a room whose direct sound is not
    # the strongest arrival would otherwise hand the drive reference to some
    # reflection without saying so.
    g_direct_elem = np.abs(drive_paths[0].gains)
    g_elem_recv = np.abs(onward_paths[0].gains)
    onset_onward = onward_paths[0].delay_s
    drive_paths, _ = _prune(drive_paths, MAX_SEGMENT_PATHS)
    onward_paths, _ = _prune(onward_paths, MAX_SEGMENT_PATHS)

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
                              order=dp.order + op.order,
                              coherent=dp.coherent and op.coherent))
    if drive_tail is not None:
        e = drive_tail.energy * g_diff_elem**2 * g_elem_recv**2
        if np.any(e > 0):
            tails.append(Tail(e, drive_tail.t60,
                              drive_tail.onset_s + onset_onward))
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
    # the direct arrivals and the driving levels, before pruning re-ranks the lists
    gd = np.abs(drive[0].gains)
    gmid_dir = np.abs(mid[0].gains)
    gon = np.abs(onward[0].gains)
    onset_mid, onset_on = mid[0].delay_s, onward[0].delay_s
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
                                  order=dp.order + mp.order + op.order,
                                  coherent=dp.coherent and mp.coherent and op.coherent))
    if drive_tail is not None:
        e = drive_tail.energy * t1_diff**2 * gmid_dir**2 * t2_diff**2 * gon**2
        if np.any(e > 0):
            tails.append(Tail(e, drive_tail.t60,
                              drive_tail.onset_s + onset_mid + onset_on))
    if mid_tail is not None:
        e = gd**2 * t1_diff**2 * mid_tail.energy * t2_diff**2 * gon**2
        if np.any(e > 0):
            tails.append(Tail(e, mid_tail.t60,
                              mid_tail.onset_s + onset_on))
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
