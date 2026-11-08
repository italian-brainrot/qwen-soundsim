"""The scene model: what environments are made of.

A scene is a small graph:

    spaces  --(coupling element)-->  spaces

where a *space* is either a rectangular room (with per-wall absorption) or the
outdoors (with a porous ground, weather and screens), and a *coupling element*
is a wall panel, a window, a doorway or a leak.  Sound travels through the graph
and every edge contributes its own physics:

* inside a room  -> image-source reflections + an energy-balanced diffuse tail
* through an element -> damped-plate transmission loss, or aperture radiation
* across the outdoors -> spherical spreading, atmospheric absorption, complex
  ground reflection and Maekawa screen diffraction

Positions are metres in a right-handed world frame with +z up.  Rooms occupy
``origin .. origin + dims``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..acoustics.materials import (
    GROUNDS, MATERIALS, PANELS, Absorber, Ground, Panel, absorber, panel,
)
from ..acoustics.constants import P_REF_PA


Vector = tuple[float, float, float]


@dataclass
class Weather:
    """Outdoor propagation conditions (they change the *filter*, not just the level)."""
    temp_c: float = 20.0
    rh_pct: float = 50.0
    pressure_pa: float = P_REF_PA
    wind_speed_ms: float = 0.0
    wind_towards_receiver: float = 1.0     # 1 = tail wind, -1 = head wind
    #: 0..1 strength of the phase wander eddies put on a long path.  It sets how
    #: much of the direct/ground interference survives (see _outdoor_paths).
    turbulence: float = 0.35

    @property
    def effective_wind_ms(self) -> float:
        """Component along the source->receiver line (positive = tail wind)."""
        return self.wind_speed_ms * self.wind_towards_receiver


@dataclass
class Room:
    """A rectangular enclosure with per-wall absorption."""
    id: str
    dims: Vector = (5.0, 4.0, 2.6)
    origin: Vector = (0.0, 0.0, 0.0)
    #: wall material keys, keyed by face
    walls: dict[str, str] = field(default_factory=lambda: {
        "floor": "concrete_painted", "ceiling": "plaster_smooth",
        "west": "brick_painted", "east": "brick_painted",
        "south": "brick_painted", "north": "brick_painted",
    })
    #: optional extra absorption distributed over the floor area (furniture,
    #: an audience, a car's trim): list of (material key, area in m2)
    contents: list[tuple[str, float]] = field(default_factory=list)

    # -- geometry -------------------------------------------------------
    @property
    def volume(self) -> float:
        return float(np.prod(self.dims))

    @property
    def surface_area(self) -> float:
        lx, ly, lz = self.dims
        return float(2 * (lx * ly + lx * lz + ly * lz))

    def face_area(self, wall: str) -> float:
        lx, ly, lz = self.dims
        return {"floor": lx * ly, "ceiling": lx * ly,
                "north": lx * lz, "south": lx * lz,
                "east": ly * lz, "west": ly * lz}[wall]

    def local(self, world_point) -> np.ndarray:
        """World point -> coordinates inside the room's own frame."""
        return np.asarray(world_point, float) - np.asarray(self.origin, float)

    def contains(self, world_point, margin: float = 0.02) -> bool:
        p = self.local(world_point)
        return all(-margin <= p[i] <= self.dims[i] + margin for i in range(3))

    # -- absorption -----------------------------------------------------
    def absorbers_by_axis(self) -> list[tuple[Absorber, Absorber]]:
        """``[(wall at 0, wall at L)]`` for the x, y and z axes."""
        return [
            (absorber(self.walls["west"]), absorber(self.walls["east"])),
            (absorber(self.walls["south"]), absorber(self.walls["north"])),
            (absorber(self.walls["floor"]), absorber(self.walls["ceiling"])),
        ]

    def surface_list(self) -> list:
        """(absorber, area) pairs including lumped contents."""
        from ..acoustics.rooms import Surface
        out = [Surface(absorber(m), self.face_area(w), w)
               for w, m in self.walls.items()]
        for key, area in self.contents:
            out.append(Surface(absorber(key), area, key))
        return out


@dataclass
class Outdoors:
    """The exterior space: one porous ground plane plus screens."""
    id: str = "outdoor"
    ground: str = "grass_short"
    #: facade/wall reflections are approximated by extra image paths
    reflectors: list["Reflector"] = field(default_factory=list)

    def ground_obj(self) -> Ground:
        return GROUNDS[self.ground]


@dataclass
class Reflector:
    """A large flat exterior reflector (a building facade, a cliff, a wall)."""
    point: Vector                # a point on the plane
    normal: Vector               # unit normal facing the sound field
    name: str = "facade"


@dataclass
class Barrier:
    """A finite screen: a vertical rectangle with a straight, horizontal top.

    Diffraction goes over the top edge -- the dominant path in the shadow zone.
    Two geometric details are modelled, because both are heard:

    * ``width`` is the horizontal extent along ``along``.  The diffracting point
      is the shortest source->edge->receiver route restricted to the *finite*
      top edge, so a ray that would pass beside the end of the screen is
      diffracted *around that end*, with the much larger path excess that
      implies: a 2 m screen shields a lot less than a 40 m one.
    * ``thickness`` is the width of a flat top.  A top of finite width has two
      edges, near and far, and both diffract; the engine sums the two edge
      contributions in energy (``diffraction.edge_group_attenuation_db``).
      With ``thickness = 0`` the two edges coincide and only one is returned.
    """
    center: Vector               # centre of the screen (x, y at ground level, z = 0)
    width: float                 # horizontal extent of the screen
    height: float                # top edge height, m
    along: Vector = (1.0, 0.0, 0.0)   # direction of the screen's length in plan
    name: str = "barrier"
    thickness: float = 0.0       # width of a flat top, m (0 = a zero-thickness screen)

    # -- geometry -------------------------------------------------------
    def _frame(self):
        """Unit length direction, plan normal, and the two top-edge centres."""
        c = np.asarray(self.center, float)
        d = np.asarray(self.along, float)
        d = d / max(np.linalg.norm(d), 1e-12)
        n = np.array([-d[1], d[0], 0.0])          # screen normal (in plan)
        top = c + np.array([0.0, 0.0, self.height])
        half = 0.5 * max(float(self.thickness), 0.0)
        return d, n, (top - n * half, top + n * half)

    def top_edges(self) -> tuple[tuple[Vector, Vector], ...]:
        """The top edge(s) as ``(point, direction)`` pairs, near side first.

        One entry for a thin screen, two when the top has a ``thickness``.
        """
        d, n, (near, far) = self._frame()
        if self.thickness <= 0.0:
            return ((tuple(near), tuple(d)),)
        return ((tuple(near), tuple(d)), (tuple(far), tuple(d)))

    def edge_points(self, source, receiver) -> list[np.ndarray]:
        """Point(s) on the top edge(s) that the diffracted ray passes through.

        The shortest source->edge->receiver path over each (finite) top edge,
        found by scanning the edge line between its ends (Fermat).  Clamping to
        the ends is what makes ``width`` matter: the minimiser runs off to an end
        as soon as the straight-line route would miss the screen.
        """
        s = np.asarray(source, float)
        r = np.asarray(receiver, float)
        d, n, _ = self._frame()
        half = max(0.5 * float(self.width), 0.0)
        out = []
        for top_c, _dir in self.top_edges():
            p0 = np.asarray(top_c, float)
            ts = np.linspace(-half, half, 401)
            pts = p0[None, :] + ts[:, None] * d[None, :]
            cost = (np.linalg.norm(s - pts, axis=1)
                    + np.linalg.norm(r - pts, axis=1))
            out.append(pts[int(np.argmin(cost))])
        return out

    def blocks(self, source, receiver) -> bool:
        """True when the straight source->receiver ray hits the screen's rectangle.

        The test is exact for a vertical rectangle: the plan projection of the
        ray has to cross the screen's plan trace *within its width*, and at that
        crossing the ray has to be lower than the top.  Everything else is the
        light zone, where a screen costs nothing however tall it looks in a
        drawing -- see ``diffraction.maekawa_attenuation_db``.
        """
        s = np.asarray(source, float)
        r = np.asarray(receiver, float)
        d, n, (top_c, _) = self._frame()
        half = max(0.5 * float(self.width), 0.0)
        # solve  a + t b = u d  in plan, with a = s - top_centre, b = r - s
        # (crossing the equation with d and with b respectively)
        a2 = s[:2] - top_c[:2]
        b2 = (r - s)[:2]
        cross = lambda x, y: x[0] * y[1] - x[1] * y[0]
        den = cross(b2, d)
        if abs(den) < 1e-12:
            return False                                   # ray runs along the screen
        t = -cross(a2, d) / den                            # where along the ray
        u = -cross(a2, b2) / den                           # where along the screen
        if not 0.0 <= t <= 1.0 or abs(u) > half:
            return False                                   # misses the rectangle in plan
        return s[2] + t * (r[2] - s[2]) < self.height      # and passes under the top


@dataclass
class Coupling:
    """How sound passes between two spaces.

    kind:
      ``panel``    a solid element (wall, window, car body panel) -- described by
                   a :class:`Panel`, its area and optionally a second leaf with an
                   air cavity (double glazing, stud wall);
      ``aperture`` an opening (open window, doorway, vent, gap);
      ``leak``     a small crack; treated as an aperture with its area.
    """
    kind: str
    area_m2: float
    #: world position of the element's centre; its normal points towards `to_space`
    position: Vector
    normal: Vector = (1.0, 0.0, 0.0)
    from_space: str = "room"
    to_space: str = "outdoor"
    panel_key: str | None = None
    second_panel_key: str | None = None      # double leaf
    cavity_m: float = 0.0
    label: str = ""

    @property
    def is_opening(self) -> bool:
        return self.kind in ("aperture", "leak")


@dataclass
class Source:
    """The sound source: a position plus its calibrated free-field strength."""
    position: Vector
    spl_at_1m_db: float = 94.0        # SPL a unit-rms signal produces at 1 m
    height_note: str = ""
    #: 0 = omnidirectional (physical default), 1 = strongly forward-beamed
    directivity: float = 0.0
    #: -3 dB point of the source's own low-frequency response.  A real box
    #: cannot radiate much below its box/ports resonance, and that matters here:
    #: the lowest room modes are exactly what a flat-spectrum source would
    #: over-excite.  Modelled as a 2nd-order high pass in pressure -- 12 dB per
    #: octave below f3, which is what a sealed box below its resonance does (the
    #: cone's excursion stays put, its velocity, and hence its volume velocity,
    #: falls at 6 dB per octave, and radiation costs another 6 dB).
    #: That slope is not incidental: the compliance of a closed room rises at
    #: exactly 12 dB per octave towards DC (acoustics.modal), so the two are the
    #: pair that keeps a rendered cabin from running away into inaudible bass.
    f3_db: float = 55.0


@dataclass
class Scene:
    """A complete simulation setup."""
    name: str = "scene"
    description: str = ""
    source: Source = field(default_factory=Source)
    receiver: Vector = (0.0, 0.0, 1.2)
    rooms: dict[str, Room] = field(default_factory=dict)
    outdoors: Outdoors = field(default_factory=Outdoors)
    couplings: list[Coupling] = field(default_factory=list)
    barriers: list[Barrier] = field(default_factory=list)
    weather: Weather = field(default_factory=Weather)
    #: image-source order used for early reflections (per space)
    ism_order: int = 6
    #: include the incoherent, energy-balanced diffuse tail
    diffuse_tail: bool = True

    # -- topology -------------------------------------------------------
    def space_at(self, point) -> str:
        """Which space a point lives in: the innermost room, else outdoors."""
        best = self.outdoors.id
        best_vol = np.inf
        for room in self.rooms.values():
            if room.contains(point):
                v = room.volume
                if v < best_vol:
                    best, best_vol = room.id, v
        return best

    def couplings_between(self, a: str, b: str) -> list[Coupling]:
        return [c for c in self.couplings
                if (c.from_space == a and c.to_space == b)
                or (c.from_space == b and c.to_space == a)]

    def room(self, space_id: str) -> Room | None:
        return self.rooms.get(space_id)
