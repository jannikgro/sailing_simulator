"""Level definitions for the sailing school.

Each level is a frozen spec: wind character, an ordered list of objectives to
round, rocks (solid, from level 4 on) and traffic routes (from level 6 on).
``build_simulation`` turns a spec into a fresh, mutable simulation so a level
can be restarted any number of times.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .physics import (
    Rock,
    SailingSimulation,
    TrafficVessel,
    WindState,
    nav_vector,
    normalize_degrees,
)


PLAYER_START = (0.0, 0.0)


def leg(start: tuple[float, float], heading_deg: float, distance: float) -> tuple[float, float]:
    east, north = nav_vector(heading_deg)
    return start[0] + east * distance, start[1] + north * distance


def flank_rocks(
    start: tuple[float, float],
    end: tuple[float, float],
    offset: float,
    fractions: tuple[float, ...] = (0.35, 0.7),
    sides: tuple[float, ...] | None = None,
) -> tuple[Rock, ...]:
    """Rocks offset perpendicular from a leg: safe on the rhumb line,
    punishing for boats that stray. ``sides`` picks the side per rock
    (+1 starboard of the leg direction, -1 port); default alternates."""

    delta_east = end[0] - start[0]
    delta_north = end[1] - start[1]
    length = hypot(delta_east, delta_north)
    if length <= 0.0:
        return ()
    perp_east = delta_north / length
    perp_north = -delta_east / length
    rocks = []
    for index, fraction in enumerate(fractions):
        if sides is not None:
            side = sides[index]
        else:
            side = 1.0 if index % 2 == 0 else -1.0
        rocks.append(
            Rock(
                east=start[0] + delta_east * fraction + perp_east * offset * side,
                north=start[1] + delta_north * fraction + perp_north * offset * side,
            )
        )
    return tuple(rocks)


def channel_rocks(
    path: tuple[tuple[float, float], ...],
    spacing: float = 170.0,
    half_width: float = 115.0,
) -> tuple[Rock, ...]:
    """Line both sides of a path with rocks, forming a navigable channel."""

    rocks = []
    for start, end in zip(path, path[1:]):
        delta_east = end[0] - start[0]
        delta_north = end[1] - start[1]
        length = hypot(delta_east, delta_north)
        if length <= 0.0:
            continue
        dir_east = delta_east / length
        dir_north = delta_north / length
        perp_east = dir_north
        perp_north = -dir_east
        along = spacing * 0.5
        while along < length:
            center_east = start[0] + dir_east * along
            center_north = start[1] + dir_north * along
            for side in (1.0, -1.0):
                rocks.append(
                    Rock(
                        east=center_east + perp_east * half_width * side,
                        north=center_north + perp_north * half_width * side,
                    )
                )
            along += spacing
    return tuple(rocks)


@dataclass(frozen=True)
class WindSpec:
    from_deg: float
    speed: float
    minimum_speed: float
    maximum_speed: float
    gust_amplitude: float
    shift_amplitude: float

    def build(self) -> WindState:
        return WindState(
            from_deg=self.from_deg,
            speed=self.speed,
            minimum_speed=self.minimum_speed,
            maximum_speed=self.maximum_speed,
            base_speed=self.speed,
            gust_amplitude=self.gust_amplitude,
            shift_amplitude=self.shift_amplitude,
        )


@dataclass(frozen=True)
class TrafficSpec:
    id: str
    kind: str
    east: float
    north: float
    heading_deg: float
    speed: float
    waypoints: tuple[tuple[float, float], ...] = ()

    def build(self) -> TrafficVessel:
        return TrafficVessel(
            id=self.id,
            kind=self.kind,
            position_east=self.east,
            position_north=self.north,
            heading_deg=self.heading_deg,
            speed=self.speed,
            waypoints=[(east, north) for east, north in self.waypoints],
        )


@dataclass(frozen=True)
class Objective:
    east: float
    north: float
    label: str
    radius: float = 55.0


@dataclass(frozen=True)
class Level:
    number: int
    title: str
    skills: str
    briefing: tuple[str, ...]
    wind: WindSpec
    start_heading_deg: float
    objectives: tuple[Objective, ...]
    rocks: tuple[Rock, ...] = ()
    traffic: tuple[TrafficSpec, ...] = ()


def build_simulation(level: Level) -> SailingSimulation:
    simulation = SailingSimulation(
        wind=level.wind.build(),
        traffic=[spec.build() for spec in level.traffic],
        rocks=list(level.rocks),
    )
    simulation.boat.heading_deg = normalize_degrees(level.start_heading_deg)
    return simulation


def _level_1() -> Level:
    return Level(
        number=1,
        title="First Reach",
        skills="steering and sail trim",
        briefing=(
            "Steer with LEFT/RIGHT, trim the main sheet with UP/DOWN.",
            "The wind blows from the east: sailing north is a beam reach.",
            "Watch the sail efficiency panel and keep it green.",
            "Round the buoys in order to finish the level.",
        ),
        wind=WindSpec(
            from_deg=90.0,
            speed=8.0,
            minimum_speed=6.0,
            maximum_speed=10.0,
            gust_amplitude=0.5,
            shift_amplitude=0.15,
        ),
        start_heading_deg=0.0,
        objectives=(
            Objective(0.0, 320.0, "Buoy 1"),
            Objective(-80.0, 660.0, "Buoy 2"),
            Objective(80.0, 1000.0, "Buoy 3"),
            Objective(0.0, 1340.0, "Finish"),
        ),
    )


def _level_2() -> Level:
    start = PLAYER_START
    beam = leg(start, 0.0, 450.0)
    broad = leg(beam, 250.0, 430.0)
    close = leg(broad, 140.0, 430.0)
    finish = leg(close, 20.0, 420.0)
    return Level(
        number=2,
        title="Points of Sail",
        skills="trim on every course",
        briefing=(
            "Every leg needs a different trim: ease the sheet as the",
            "wind moves aft, trim in as you point closer to the wind.",
            "Legs: beam reach, broad reach, close hauled, close reach.",
            "Keep the efficiency bar green on every point of sail.",
        ),
        wind=WindSpec(
            from_deg=90.0,
            speed=9.0,
            minimum_speed=7.0,
            maximum_speed=11.0,
            gust_amplitude=1.0,
            shift_amplitude=0.3,
        ),
        start_heading_deg=0.0,
        objectives=(
            Objective(beam[0], beam[1], "Beam"),
            Objective(broad[0], broad[1], "Broad"),
            Objective(close[0], close[1], "Close"),
            Objective(finish[0], finish[1], "Finish"),
        ),
    )


def _level_3() -> Level:
    wind_from = 25.0
    start = PLAYER_START
    tack_1 = leg(start, wind_from - 42.0, 470.0)
    tack_2 = leg(tack_1, wind_from + 42.0, 470.0)
    tack_3 = leg(tack_2, wind_from - 42.0, 470.0)
    windward = leg(tack_3, wind_from + 42.0, 430.0)
    return Level(
        number=3,
        title="Tack the Ladder",
        skills="tacking and the no-go zone",
        briefing=(
            "The marks lie upwind: you cannot sail straight at them.",
            "Sail close hauled, then push the tiller to swing the bow",
            "through the wind onto the other tack. Keep speed up before",
            "each tack or you will get stuck in irons.",
        ),
        wind=WindSpec(
            from_deg=wind_from,
            speed=10.0,
            minimum_speed=7.0,
            maximum_speed=13.0,
            gust_amplitude=1.5,
            shift_amplitude=0.6,
        ),
        start_heading_deg=wind_from - 50.0,
        objectives=(
            Objective(tack_1[0], tack_1[1], "Tack 1"),
            Objective(tack_2[0], tack_2[1], "Tack 2"),
            Objective(tack_3[0], tack_3[1], "Tack 3"),
            Objective(windward[0], windward[1], "Windward"),
        ),
    )


def _level_4() -> Level:
    downwind = 20.0  # wind from 200: sailing ~20 is a run
    start = PLAYER_START
    gybe_1 = leg(start, downwind + 28.0, 420.0)
    gybe_2 = leg(gybe_1, downwind - 28.0, 420.0)
    gybe_3 = leg(gybe_2, downwind + 28.0, 420.0)
    finish = leg(gybe_3, downwind - 28.0, 420.0)
    rocks = (
        flank_rocks(start, gybe_1, 150.0)
        + flank_rocks(gybe_1, gybe_2, 150.0)
        + flank_rocks(gybe_2, gybe_3, 150.0)
        + flank_rocks(gybe_3, finish, 150.0)
    )
    return Level(
        number=4,
        title="Gybe Slalom",
        skills="controlled gybes downwind",
        briefing=(
            "A slalom dead downwind: each mark needs a gybe, swinging",
            "the stern through the wind. Trim in a little before you",
            "gybe so the boom crosses gently, then ease out again.",
            "Rocks now line the run: they are solid. Stay off them.",
        ),
        wind=WindSpec(
            from_deg=200.0,
            speed=10.0,
            minimum_speed=7.0,
            maximum_speed=13.0,
            gust_amplitude=1.5,
            shift_amplitude=0.5,
        ),
        start_heading_deg=45.0,
        objectives=(
            Objective(gybe_1[0], gybe_1[1], "Gybe 1"),
            Objective(gybe_2[0], gybe_2[1], "Gybe 2"),
            Objective(gybe_3[0], gybe_3[1], "Gybe 3"),
            Objective(finish[0], finish[1], "Finish"),
        ),
        rocks=rocks,
    )


def _level_5() -> Level:
    gate_1 = (0.0, 350.0)
    gate_2 = (-140.0, 700.0)
    gate_3 = (60.0, 1050.0)
    harbor = (0.0, 1400.0)
    rocks = channel_rocks(((0.0, -60.0), gate_1, gate_2, gate_3, harbor))
    return Level(
        number=5,
        title="Rock Channel",
        skills="precise boat handling",
        briefing=(
            "A winding channel through the rocks leads to the harbor.",
            "The walls are unforgiving: plan each curve early, carry",
            "enough speed to keep the rudder biting, and mind the",
            "wind shifting your point of sail on every bend.",
        ),
        wind=WindSpec(
            from_deg=90.0,
            speed=9.0,
            minimum_speed=7.0,
            maximum_speed=11.0,
            gust_amplitude=0.8,
            shift_amplitude=0.2,
        ),
        start_heading_deg=0.0,
        objectives=(
            Objective(gate_1[0], gate_1[1], "Gate 1"),
            Objective(gate_2[0], gate_2[1], "Gate 2"),
            Objective(gate_3[0], gate_3[1], "Gate 3"),
            Objective(harbor[0], harbor[1], "Harbor"),
        ),
        rocks=rocks,
    )


def _level_6() -> Level:
    mark_a = leg(PLAYER_START, 330.0, 480.0)
    mark_b = leg(mark_a, 105.0, 450.0)
    mark_c = leg(mark_b, 200.0, 450.0)
    finish = (-220.0, 160.0)
    return Level(
        number=6,
        title="Harbor Traffic",
        skills="right-of-way rules",
        briefing=(
            "Other vessels share the water. Overtaking boats keep clear,",
            "power gives way to sail, port tack gives way to starboard,",
            "and the windward boat keeps clear on the same tack.",
            "A red ring marks a boat you must give way to. Collisions",
            "and close passes as give-way vessel cost you a penalty.",
        ),
        wind=WindSpec(
            from_deg=60.0,
            speed=10.0,
            minimum_speed=7.0,
            maximum_speed=13.0,
            gust_amplitude=2.0,
            shift_amplitude=1.0,
        ),
        start_heading_deg=330.0,
        objectives=(
            Objective(mark_a[0], mark_a[1], "Mark A"),
            Objective(mark_b[0], mark_b[1], "Mark B"),
            Objective(mark_c[0], mark_c[1], "Mark C"),
            Objective(finish[0], finish[1], "Finish"),
        ),
        traffic=(
            TrafficSpec(
                "M1", "motorboat", -500.0, 200.0, 90.0, 5.0,
                waypoints=((500.0, 200.0), (-500.0, 200.0)),
            ),
            TrafficSpec(
                "M2", "motorboat", 400.0, 450.0, 270.0, 4.5,
                waypoints=((-400.0, 450.0), (400.0, 450.0)),
            ),
            TrafficSpec(
                "S1", "sailboat", 300.0, 100.0, 304.0, 3.0,
                waypoints=((-300.0, 500.0), (300.0, 100.0)),
            ),
        ),
    )


def _level_7() -> Level:
    wind_from = 55.0
    start = PLAYER_START
    tack_1 = leg(start, wind_from - 40.0, 500.0)
    tack_2 = leg(tack_1, wind_from + 40.0, 500.0)
    tack_3 = leg(tack_2, wind_from - 40.0, 500.0)
    windward = leg(tack_3, wind_from + 40.0, 460.0)
    run_1 = leg(windward, wind_from + 180.0, 620.0)
    finish = leg(run_1, wind_from + 180.0, 620.0)
    # The downwind return crosses back through the tacking ladder, so every
    # rock is placed on an explicitly chosen side, clear of both lanes.
    rocks = (
        flank_rocks(start, tack_1, 140.0, fractions=(0.5,), sides=(-1.0,))
        + flank_rocks(tack_2, tack_3, 140.0, fractions=(0.5,), sides=(-1.0,))
        + flank_rocks(tack_3, windward, 140.0, fractions=(0.5,), sides=(-1.0,))
        + flank_rocks(windward, run_1, 150.0, fractions=(0.5,), sides=(-1.0,))
        + flank_rocks(run_1, finish, 140.0, fractions=(0.45,), sides=(-1.0,))
        + (Rock(east=windward[0] + 180.0, north=windward[1] + 40.0),)
    )
    return Level(
        number=7,
        title="Regatta Day",
        skills="everything at once",
        briefing=(
            "The full course: beat upwind through the tacking ladder,",
            "round the windward mark and run home past the rocks.",
            "The breeze is fresh, gusty and shifty, the harbor is busy",
            "and the right-of-way rules are in force. Good luck, skipper.",
        ),
        wind=WindSpec(
            from_deg=wind_from,
            speed=12.0,
            minimum_speed=6.0,
            maximum_speed=19.0,
            gust_amplitude=4.2,
            shift_amplitude=2.4,
        ),
        start_heading_deg=wind_from - 50.0,
        objectives=(
            Objective(tack_1[0], tack_1[1], "Tack 1"),
            Objective(tack_2[0], tack_2[1], "Tack 2"),
            Objective(tack_3[0], tack_3[1], "Tack 3"),
            Objective(windward[0], windward[1], "Windward"),
            Objective(run_1[0], run_1[1], "Run"),
            Objective(finish[0], finish[1], "Finish"),
        ),
        rocks=rocks,
        traffic=(
            TrafficSpec(
                "M1", "motorboat", -300.0, 600.0, 90.0, 5.0,
                waypoints=((900.0, 600.0), (-300.0, 600.0)),
            ),
            TrafficSpec(
                "S1", "sailboat", 800.0, 200.0, 319.0, 3.2,
                waypoints=((200.0, 900.0), (800.0, 200.0)),
            ),
            TrafficSpec(
                "M2", "motorboat", 1300.0, 400.0, 305.0, 4.5,
                waypoints=((300.0, 1100.0), (1300.0, 400.0)),
            ),
            TrafficSpec(
                "S2", "sailboat", 100.0, 1200.0, 93.0, 3.0,
                waypoints=((1100.0, 1150.0), (100.0, 1200.0)),
            ),
        ),
    )


LEVELS: tuple[Level, ...] = (
    _level_1(),
    _level_2(),
    _level_3(),
    _level_4(),
    _level_5(),
    _level_6(),
    _level_7(),
)
