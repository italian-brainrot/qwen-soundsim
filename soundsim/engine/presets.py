"""Ready-made environments.

Each preset is a plain :class:`~soundsim.engine.scene.Scene` -- edit any field
afterwards in the UI or in code.  Numbers come from typical construction:
12.7 mm faced gypsum, 100 mm masonry, 4 mm window glass, 6 mm automotive glass,
mineral-wool trim in a car cabin, and so on.
"""

from __future__ import annotations

from .scene import (Barrier, Coupling, Outdoors, Reflector, Room, Scene, Source,
                    Weather)

OUTDOOR = Weather(temp_c=18.0, rh_pct=60.0)
WARM = Weather(temp_c=21.0, rh_pct=45.0)


# ---------------------------------------------------------------------------
# 1. Source inside a room, listener outside the facade
# ---------------------------------------------------------------------------
def party_next_door(distance_out: float = 8.0, window_open_fraction: float = 0.0,
                    wall_area: float = 12.0) -> Scene:
    """A party (shared) wall: source in a living room, listener in the street."""
    room = Room(id="living", dims=(6.0, 4.5, 2.7), origin=(0, 0, 0), walls={
        "floor": "wood_parquet_concrete", "ceiling": "plaster_smooth",
        "west": "brick_painted", "east": "brick_unpainted",
        "south": "brick_painted", "north": "carpet_thin",
    }, contents=[("upholstered_chair", 4.0), ("curtain_heavy", 6.0)])
    # party wall is the +x face at x = 6.0
    couplings = [Coupling("panel", wall_area, (6.0, 2.25, 1.4), (1, 0, 0),
                          "living", "outdoor", panel_key="brick_100mm",
                          label="party wall")]
    if window_open_fraction > 0:
        couplings.append(Coupling("panel", 1.6, (6.0, 3.4, 1.5), (1, 0, 0),
                                  "living", "outdoor", panel_key="glass_4mm",
                                  label="window"))
    if window_open_fraction >= 1.0:
        couplings.append(Coupling("aperture", 1.6, (6.0, 3.4, 1.5), (1, 0, 0),
                                  "living", "outdoor", label="open window"))
    elif window_open_fraction > 0:
        couplings.append(Coupling("aperture", 1.6 * window_open_fraction,
                                  (6.0, 3.4, 1.5), (1, 0, 0), "living", "outdoor",
                                  label="open part of window"))
    return Scene(
        name="Party next door",
        description=("Music playing in a brick living room, heard from the street "
                     "through the shared masonry wall."),
        source=Source(position=(1.8, 1.5, 1.3), spl_at_1m_db=98.0),
        receiver=(6.0 + distance_out, 2.25, 1.5),
        rooms={"living": room},
        outdoors=Outdoors(ground="asphalt_sealed",
                          reflectors=[Reflector((6.05, 0.0, 0.0), (-1, 0, 0), "facade")]),
        couplings=couplings, weather=OUTDOOR, ism_order=5)


def through_closed_window(distance_out: float = 6.0) -> Scene:
    """Source and receiver on opposite sides of one glazed facade."""
    room = Room(id="salon", dims=(5.0, 4.0, 2.6), walls={
        "floor": "wood_floor_on_joists", "ceiling": "plaster_smooth",
        "west": "brick_painted", "east": "glass_3mm",
        "south": "brick_painted", "north": "brick_painted",
    }, contents=[("curtain_heavy", 5.0), ("upholstered_chair", 3.0)])
    return Scene(
        name="Through the closed window",
        description=("Loudspeaker inside a carpeted salon, listener a few metres "
                     "outside the double-glazed window."),
        source=Source(position=(1.5, 2.0, 1.2), spl_at_1m_db=94.0),
        receiver=(5.0 + distance_out, 2.0, 1.5),
        rooms={"salon": room},
        outdoors=Outdoors(ground="grass_short"),
        couplings=[Coupling("panel", 2.4, (5.0, 2.0, 1.4), (1, 0, 0), "salon",
                            "outdoor", panel_key="glass_4mm",
                            second_panel_key="glass_4mm", cavity_m=0.12,
                            label="double glazing")],
        weather=OUTDOOR, ism_order=5)


def open_window_night(distance_out: float = 10.0, open_area: float = 0.6) -> Scene:
    """A partly open window at night: aperture radiation plus the wall."""
    room = Room(id="bedroom", dims=(4.0, 3.2, 2.5), walls={
        "floor": "carpet_thin", "ceiling": "plaster_smooth",
        "west": "brick_painted", "east": "glass_3mm",
        "south": "plaster_smooth", "north": "brick_painted",
    })
    return Scene(
        name="Open window at night",
        description=("Radio in a small carpeted bedroom, window ajar: the opening "
                     "behaves as a high-pass source, so outside you get the bass "
                     "and the treble detail disappears."),
        source=Source(position=(1.2, 1.6, 1.0), spl_at_1m_db=90.0),
        receiver=(4.0 + distance_out, 1.6, 1.6),
        rooms={"bedroom": room},
        outdoors=Outdoors(ground="grass_short"),
        couplings=[
            Coupling("aperture", open_area, (4.0, 2.2, 1.6), (1, 0, 0), "bedroom",
                     "outdoor", label="open window"),
            Coupling("panel", 3.0, (4.0, 1.0, 1.3), (1, 0, 0), "bedroom", "outdoor",
                     panel_key="brick_100mm", label="wall"),
        ],
        weather=Weather(temp_c=14.0, rh_pct=75.0), ism_order=5)


# ---------------------------------------------------------------------------
# 2. Room to room
# ---------------------------------------------------------------------------
def next_room(distance_in_next: float = 3.0, open_doorway: bool = False) -> Scene:
    """Two rooms coupled by a gypsum partition and an open doorway."""
    a = Room(id="source_room", dims=(5.0, 4.0, 2.6), walls={
        "floor": "wood_parquet_concrete", "ceiling": "plaster_smooth",
        "west": "brick_painted", "east": "gypsum_board_painted",
        "south": "brick_painted", "north": "brick_painted",
    }, contents=[("carpet_thick_pad", 8.0), ("upholstered_chair", 3.0)])
    b = Room(id="listen_room", dims=(4.5, 3.8, 2.6), origin=(5.0, 0, 0), walls={
        "floor": "vinyl_floor", "ceiling": "plaster_smooth",
        "west": "gypsum_board_painted", "east": "brick_painted",
        "south": "brick_painted", "north": "brick_painted",
    }, contents=[("carpet_thin", 10.0), ("upholstered_chair", 3.0),
                 ("curtain_heavy", 5.0)])
    return Scene(
        name="Next room (partition + doorway)",
        description=("Stereo behind a gypsum partition with a doorway; the "
                     "listener's own room adds its own reverberation on top."),
        source=Source(position=(1.5, 2.0, 1.1), spl_at_1m_db=94.0),
        receiver=(5.0 + distance_in_next, 1.9, 1.2),
        rooms={"source_room": a, "listen_room": b},
        couplings=[
            Coupling("panel", 8.0, (5.0, 2.0, 1.3), (1, 0, 0), "source_room",
                     "listen_room", panel_key="gypsum_board_12mm",
                     second_panel_key="gypsum_board_12mm", cavity_m=0.07,
                     label="stud wall"),
            # a *closed* door: gypsum leaf with a crack under it.  Pass
            # open_doorway=True below for the open-door variant.
            Coupling("panel", 1.9 * 0.9, (5.0, 0.95, 1.0), (1, 0, 0),
                     "source_room", "listen_room", panel_key="gypsum_board_12mm",
                     label="door leaf"),
            Coupling("leak", 0.012, (5.0, 0.95, 0.02), (1, 0, 0),
                     "source_room", "listen_room", label="under-door gap"),
        ],
        weather=WARM, ism_order=6)
    if open_doorway:
        scene.couplings.append(Coupling("aperture", 1.9 * 0.9, (5.0, 0.95, 1.0),
                                       (1, 0, 0), "source_room", "listen_room",
                                       label="open doorway"))
    return scene


def across_the_corridor() -> Scene:
    """Flat with two rooms coupled through a corridor."""
    a = Room(id="flat_a", dims=(4.6, 3.6, 2.5), walls={
        "floor": "carpet_thin", "ceiling": "plaster_smooth",
        "west": "concrete_painted", "east": "gypsum_board_painted",
        "south": "concrete_painted", "north": "concrete_painted"})
    b = Room(id="corridor", dims=(1.4, 3.6, 2.5), origin=(4.6, 0, 0), walls={
        "floor": "ceramic_tile", "ceiling": "plaster_smooth",
        "west": "gypsum_board_painted", "east": "brick_painted",
        "south": "concrete_painted", "north": "concrete_painted"})
    c = Room(id="flat_b", dims=(4.0, 3.6, 2.5), origin=(6.0, 0, 0), walls={
        "floor": "carpet_thick_pad", "ceiling": "plaster_smooth",
        "west": "brick_painted", "east": "concrete_painted",
        "south": "concrete_painted", "north": "concrete_painted"})
    return Scene(
        name="Two doors down (through a corridor)",
        description=("Sound crossing a flat, a hard tiled corridor and a second "
                     "flat -- two cascades of transmission and two reverberations."),
        source=Source(position=(1.4, 1.8, 1.1), spl_at_1m_db=92.0),
        receiver=(8.6, 1.8, 1.2),
        rooms={"flat_a": a, "corridor": b, "flat_b": c},
        couplings=[
            Coupling("panel", 6.0, (4.6, 1.8, 1.2), (1, 0, 0), "flat_a", "corridor",
                     panel_key="gypsum_board_12mm", label="partition"),
            Coupling("panel", 0.85, (4.6, 2.9, 1.05), (1, 0, 0), "flat_a",
                     "corridor", panel_key="gypsum_board_12mm", label="door leaf"),
            Coupling("leak", 0.01, (4.6, 2.9, 0.02), (1, 0, 0), "flat_a",
                     "corridor", label="door gap"),
            Coupling("panel", 6.0, (6.0, 1.8, 1.2), (1, 0, 0), "corridor", "flat_b",
                     panel_key="brick_100mm", label="masonry"),
            Coupling("panel", 0.85, (6.0, 0.7, 1.05), (1, 0, 0), "corridor",
                     "flat_b", panel_key="gypsum_board_12mm", label="door leaf"),
            Coupling("leak", 0.01, (6.0, 0.7, 0.02), (1, 0, 0), "corridor",
                     "flat_b", label="door gap"),
        ],
        weather=WARM, ism_order=5)


# ---------------------------------------------------------------------------
# 3. Vehicles
# ---------------------------------------------------------------------------
def parked_car_exterior(distance_m: float = 4.0, side: bool = True) -> Scene:
    """Car audio playing in a parked car, heard from the street."""
    cabin = Room(id="cabin", dims=(4.4, 1.8, 1.2), origin=(0, 0, 0.5), walls={
        "floor": "carpet_thin", "ceiling": "fabric_padded_wall",
        "west": "car_interior", "east": "car_interior",
        "south": "car_interior", "north": "car_interior",
    }, contents=[("upholstered_chair", 2.4)])
    y = 0.9 if side else 0.0
    z = 1.1
    return Scene(
        name="Parked car, heard outside",
        description=("Music inside a closed car: the cabin is a small modal box, "
                     "the glass is thin (coincidence near 3 kHz) and the doors are "
                     "damped panels."),
        source=Source(position=(0.9, 0.0, 1.0), spl_at_1m_db=100.0),
        receiver=(0.9, y + distance_m, z),
        rooms={"cabin": cabin},
        outdoors=Outdoors(ground="asphalt_sealed"),
        couplings=[
            Coupling("panel", 1.5, (0.9, 0.9, 1.0), (0, 1, 0), "cabin", "outdoor",
                     panel_key="car_window_glass", label="side glass"),
            Coupling("panel", 1.6, (0.9, -0.9, 1.0), (0, -1, 0), "cabin", "outdoor",
                     panel_key="car_body_panel", second_panel_key="car_body_panel",
                     cavity_m=0.06, label="far door"),
            Coupling("panel", 2.0, (2.2, 0.0, 1.1), (0, 0, 1), "cabin", "outdoor",
                     panel_key="car_body_panel", label="roof"),
            Coupling("leak", 0.01, (1.8, 0.9, 0.6), (0, 1, 0), "cabin", "outdoor",
                     label="vent gaps"),
        ],
        weather=OUTDOOR, ism_order=8)


def inside_car(distance_m: float = 1.2) -> Scene:
    """Listening inside the car: a strongly modal, damped little box."""
    cabin = Room(id="cabin", dims=(4.4, 1.8, 1.2), origin=(0, 0, 0.5), walls={
        "floor": "carpet_thin", "ceiling": "fabric_padded_wall",
        "west": "car_interior", "east": "car_interior",
        "south": "car_interior", "north": "car_interior",
    }, contents=[("upholstered_chair", 2.4)])
    return Scene(
        name="Inside the car",
        description=("The cabin below ~150 Hz is a set of axial modes (roughly "
                     "39, 78, 95, 118 Hz), which is why car audio sounds boomy or "
                     "null-y depending on where you sit."),
        source=Source(position=(0.9, 0.0, 1.0), spl_at_1m_db=100.0),
        receiver=(0.9 + distance_m, 0.0, 1.1),
        rooms={"cabin": cabin},
        weather=WARM, ism_order=10)


# ---------------------------------------------------------------------------
# 4. Outdoors / distance
# ---------------------------------------------------------------------------
def far_field_park(distance_m: float = 300.0, ground: str = "grass_short",
                   barrier_height: float = 0.0) -> Scene:
    """A distant open-air stage: air absorption, ground effect and screens."""
    barriers = []
    if barrier_height > 0:
        barriers.append(Barrier(center=(distance_m / 2.0, 0.0, 0.0), width=40.0,
                                height=barrier_height, along=(0, 1, 0),
                                name="berm"))
    return Scene(
        name="Distant stage in the park",
        description=("300 m over grass: high frequencies are eaten by the air and "
                     "by the ground, and a screen removes what little treble is "
                     "left in the shadow zone."),
        source=Source(position=(0.0, 0.0, 2.0), spl_at_1m_db=110.0),
        receiver=(distance_m, 0.0, 1.6),
        outdoors=Outdoors(ground=ground),
        barriers=barriers,
        weather=Weather(temp_c=8.0, rh_pct=80.0, wind_speed_ms=3.0,
                        wind_towards_receiver=1.0, turbulence=0.5),
        ism_order=0, diffuse_tail=False)


def across_the_room(distance_m: float = 6.0) -> Scene:
    """Same room, near vs far: the classic direct-to-reverberant walk."""
    room = Room(id="hall", dims=(11.0, 7.5, 3.6), walls={
        "floor": "marble_tile", "ceiling": "acoustic_ceiling_tile",
        "west": "brick_painted", "east": "glass_3mm",
        "south": "brick_painted", "north": "plaster_rough",
    }, contents=[("seat_auditorium_upholstered", 30.0)])
    return Scene(
        name="Across the room",
        description=("Walking away from the source in one room: the direct sound "
                     "falls 6 dB per doubling of distance while the reverberant "
                     "field stays flat."),
        source=Source(position=(1.2, 3.7, 1.4), spl_at_1m_db=94.0),
        receiver=(1.2 + distance_m, 3.7, 1.4),
        rooms={"hall": room}, weather=WARM, ism_order=6)


def concrete_stairwell() -> Scene:
    room = Room(id="stairwell", dims=(3.2, 6.5, 9.0), walls={
        "floor": "concrete_rough", "ceiling": "concrete_rough",
        "west": "concrete_painted", "east": "concrete_painted",
        "south": "concrete_painted", "north": "concrete_painted"})
    return Scene(
        name="Concrete stairwell",
        description=("Nearly dead-hard concrete: a very long reverberation time "
                     "and strong flutter between the parallel side walls."),
        source=Source(position=(1.6, 1.0, 1.4), spl_at_1m_db=90.0),
        receiver=(1.6, 5.0, 4.0),
        rooms={"stairwell": room}, weather=WARM, ism_order=8)


def small_bedroom() -> Scene:
    room = Room(id="bedroom", dims=(3.4, 2.8, 2.4), walls={
        "floor": "carpet_thick_pad", "ceiling": "plaster_smooth",
        "west": "brick_painted", "east": "brick_painted",
        "south": "brick_painted", "north": "glass_3mm"})
    return Scene(
        name="Small carpeted bedroom",
        description=("Below the Schroeder frequency the field is a handful of "
                     "axial modes: 50, 61, 71 Hz here -- that is the 'boxy' "
                     "colouration of small rooms."),
        source=Source(position=(0.8, 1.4, 1.0), spl_at_1m_db=90.0),
        receiver=(2.8, 1.4, 1.2),
        rooms={"bedroom": room}, weather=WARM, ism_order=8)


# ---------------------------------------------------------------------------
# Registry used by the UI
# ---------------------------------------------------------------------------
PRESETS: dict[str, tuple[str, str]] = {
    "party_next_door": ("Party next door", "source inside, listener in the street"),
    "through_closed_window": ("Through the closed window", "one double-glazed facade"),
    "open_window_night": ("Open window at night", "aperture radiation, bass-heavy"),
    "next_room": ("Next room", "partition + doorway, two reverberations"),
    "across_the_corridor": ("Two doors down", "room -> corridor -> room"),
    "parked_car_exterior": ("Parked car, heard outside", "thin glass, damped panels"),
    "inside_car": ("Inside the car", "modal cabin"),
    "far_field_park": ("Distant stage in the park", "air + ground + screen"),
    "across_the_room": ("Across the room", "direct vs reverberant"),
    "concrete_stairwell": ("Concrete stairwell", "very long RT"),
    "small_bedroom": ("Small carpeted bedroom", "modal, damped"),
}

_BUILDERS = {
    "party_next_door": party_next_door,
    "through_closed_window": through_closed_window,
    "open_window_night": open_window_night,
    "next_room": next_room,
    "across_the_corridor": across_the_corridor,
    "parked_car_exterior": parked_car_exterior,
    "inside_car": inside_car,
    "far_field_park": far_field_park,
    "across_the_room": across_the_room,
    "concrete_stairwell": concrete_stairwell,
    "small_bedroom": small_bedroom,
}


def make(name: str, **kw) -> Scene:
    return _BUILDERS[name](**kw)
