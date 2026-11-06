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
    """A finite screen: a vertical quad with a straight top edge.

    Diffraction goes over the top edge (the dominant path in the shadow zone);
    the screen's thickness is modelled by two edges, as in ISO 9613-2.
    """
    center: Vector               # centre of the screen (x, y at ground level, z = 0)
    width: float                 # horizontal extent of the screen
    height: float                # top edge height, m
    along: Vector = (1.0, 0.0, 0.0)   # direction of the screen's length in plan
    name: str = "barrier"

    def top_edges(self) -> tuple[tuple[Vector, Vector], tuple[Vector, Vector]]:
        """Return the near and far top edges as (point, direction) pairs."""
        c = np.asarray(self.center, float)
        d = np.asarray(self.along, float)
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0], 0.0])          # screen normal (in plan)
        top = c + np.array([0.0, 0.0, self.height])
        return ((top, tuple(d)), (top, tuple(d)))

    def edge_points(self, source, receiver) -> list[np.ndarray]:
        """Point(s) on the top edge that the diffracted ray passes through.

        The shortest source->edge->receiver path over a straight top edge is
        found by projecting onto the edge line (Fermat).
        """
        s = np.asarray(source, float)
        r = np.asarray(receiver, float)
        top_c, top_dir = self.top_edges()[0]
        p0 = np.asarray(top_c, float)
        d = np.asarray(top_dir, float)
        d = d / np.linalg.norm(d)
        # minimise |s-(p0+t d)| + |r-(p0+t d)| over t  (golden section on a line)
        ts = np.linspace(-40.0, 40.0, 4001)
        cost = (np.linalg.norm(s - (p0[None, :] + ts[:, None] * d[None, :]), axis=1)
                + np.linalg.norm(r - (p0[None, :] + ts[:, None] * d[None, :]), axis=1))
        t = ts[int(np.argmin(cost))]
        return [p0 + t * d]


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
    #: over-excite.  4th-order (12 dB per octave each side of the corner is a
    #: 2nd-order high pass, doubled for the velocity+response behaviour of a
    #: sealed box).
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
