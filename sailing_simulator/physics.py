"""Display-free sailing simulation primitives.

The model is tuned for a readable game rather than exact hydrodynamics. It
captures the main cause-and-effect loops a player expects: sail trim and point
of sail drive speed, beam/close reaching creates heel, high heel slows the boat,
and sustained overpowering can capsize it.

Traffic vessels patrol waypoint routes and follow simplified COLREGS: the
give-way vessel alters course to starboard and slows down, the stand-on vessel
holds course and speed. Rocks are solid; hitting one (or another vessel, or
capsizing) is a failure the game layer turns into a respawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, degrees, hypot, radians, sin


NO_GO_ZONE_DEG = 35.0
MAX_SHEET_ANGLE_DEG = 80.0
MAX_TILLER = 1.0
MAX_BOAT_SPEED = 8.5
WARNING_HEEL_DEG = 36.0
DANGER_HEEL_DEG = 56.0
CAPSIZE_HEEL_DEG = 70.0
KNOCKDOWN_HEEL_DEG = 76.0
CAPSIZE_TIMER_LIMIT = 1.0
WORLD_UNITS_PER_KNOT_SECOND = 18.0
PLAYER_HULL_RADIUS = 22.0
TRAFFIC_HULL_RADIUS = 17.0
TRAFFIC_LOOKAHEAD_SECONDS = 15.0
TRAFFIC_WARNING_DISTANCE = 260.0
TRAFFIC_CPA_DISTANCE = 90.0
TRAFFIC_VIOLATION_DISTANCE = 60.0
TRAFFIC_PENALTY_SECONDS = 1.6
TRAFFIC_PENALTY_COOLDOWN = 2.4
TRAFFIC_TURN_RATE_DEG = 30.0
TRAFFIC_ACCELERATION = 1.8
TRAFFIC_AVOID_RANGE = 460.0
TRAFFIC_AVOID_CPA_DISTANCE = 130.0
TRAFFIC_AVOID_TURN_DEG = 55.0
TRAFFIC_AVOID_SPEED_FACTOR = 0.55
TRAFFIC_WAYPOINT_RADIUS = 48.0
OVERTAKING_BEARING_DEG = 112.5
OVERTAKING_COURSE_TOLERANCE_DEG = 68.0
OVERTAKING_SPEED_MARGIN = 0.3
HEAD_ON_TOLERANCE_DEG = 155.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_degrees(angle: float) -> float:
    return angle % 360.0


def signed_angle_diff(target_deg: float, origin_deg: float) -> float:
    """Return target - origin in degrees, wrapped to [-180, 180)."""

    return (target_deg - origin_deg + 180.0) % 360.0 - 180.0


def move_toward(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    if current > target:
        return max(current - max_delta, target)
    return current


def turn_toward(heading_deg: float, target_deg: float, max_delta_deg: float) -> float:
    """Rotate a heading toward a target along the shortest arc."""

    diff = signed_angle_diff(target_deg, heading_deg)
    step = clamp(diff, -max(0.0, max_delta_deg), max(0.0, max_delta_deg))
    return normalize_degrees(heading_deg + step)


def nav_vector(degrees_clockwise_from_north: float) -> tuple[float, float]:
    """Return an east/north unit vector for a navigation heading."""

    angle = radians(degrees_clockwise_from_north)
    return sin(angle), cos(angle)


def bearing_between(
    origin_east: float,
    origin_north: float,
    target_east: float,
    target_north: float,
) -> float:
    east = target_east - origin_east
    north = target_north - origin_north
    return normalize_degrees(degrees(atan2(east, north)))


@dataclass
class WindState:
    """Wind direction is where the wind comes from, in navigation degrees.

    ``base_speed``, ``gust_amplitude`` and ``shift_amplitude`` let levels dial
    the weather from a steady training breeze to gusty, shifty regatta wind.
    """

    from_deg: float = 55.0
    speed: float = 11.0
    minimum_speed: float = 4.0
    maximum_speed: float = 19.0
    base_speed: float = 11.5
    gust_amplitude: float = 4.2
    shift_amplitude: float = 2.4
    elapsed: float = 0.0

    def update(self, dt: float) -> None:
        dt = max(0.0, dt)
        self.elapsed += dt

        gust_target = (
            self.base_speed
            + self.gust_amplitude * sin(self.elapsed * 0.17)
            + self.gust_amplitude * 0.43 * sin(self.elapsed * 0.43 + 1.1)
        )
        self.speed += (gust_target - self.speed) * min(1.0, dt * 0.28)
        self.speed = clamp(self.speed, self.minimum_speed, self.maximum_speed)

        direction_rate = self.shift_amplitude * (
            sin(self.elapsed * 0.08) + 0.46 * sin(self.elapsed * 0.21 + 0.8)
        )
        self.from_deg = normalize_degrees(self.from_deg + direction_rate * dt)


@dataclass
class BoatState:
    heading_deg: float = 0.0
    speed: float = 0.0
    heel_deg: float = 0.0
    tiller: float = 0.0
    sheet: float = 0.55
    capsized: bool = False
    capsize_timer: float = 0.0
    position_east: float = 0.0
    position_north: float = 0.0

    def trim_sheet(self, delta: float) -> None:
        self.sheet = clamp(self.sheet + delta, 0.0, 1.0)

    def move_tiller(self, delta: float) -> None:
        self.tiller = clamp(self.tiller + delta, -MAX_TILLER, MAX_TILLER)

    def center_tiller(self, dt: float) -> None:
        self.tiller = move_toward(self.tiller, 0.0, max(0.0, dt) * 2.0)

    def right_after_capsize(self) -> None:
        self.capsized = False
        self.capsize_timer = 0.0
        self.speed = 0.4
        self.heel_deg = 0.0
        self.tiller = 0.0
        self.sheet = max(self.sheet, 0.75)


@dataclass(frozen=True)
class Rock:
    east: float
    north: float
    radius: float = 26.0


@dataclass
class TrafficVessel:
    id: str
    kind: str
    position_east: float
    position_north: float
    heading_deg: float
    speed: float
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    waypoint_index: int = 0
    cruise_speed: float = 0.0
    avoiding: bool = False

    def __post_init__(self) -> None:
        if self.cruise_speed <= 0.0:
            self.cruise_speed = self.speed

    def advance(self, dt: float) -> None:
        east, north = nav_vector(self.heading_deg)
        distance = self.speed * max(0.0, dt) * WORLD_UNITS_PER_KNOT_SECOND
        self.position_east += east * distance
        self.position_north += north * distance

    def current_waypoint(self) -> tuple[float, float] | None:
        if not self.waypoints:
            return None
        return self.waypoints[self.waypoint_index % len(self.waypoints)]

    def desired_heading(self) -> float:
        waypoint = self.current_waypoint()
        if waypoint is None:
            return self.heading_deg
        if (
            hypot(waypoint[0] - self.position_east, waypoint[1] - self.position_north)
            < TRAFFIC_WAYPOINT_RADIUS
        ):
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            waypoint = self.waypoints[self.waypoint_index]
        return bearing_between(
            self.position_east, self.position_north, waypoint[0], waypoint[1]
        )


@dataclass(frozen=True)
class VesselDescriptor:
    """Everything the COLREGS helpers need to know about one vessel."""

    east: float
    north: float
    heading_deg: float
    speed: float
    kind: str  # "sailboat" | "motorboat"


def descriptor_for_boat(boat: BoatState) -> VesselDescriptor:
    return VesselDescriptor(
        east=boat.position_east,
        north=boat.position_north,
        heading_deg=boat.heading_deg,
        speed=boat.speed,
        kind="sailboat",
    )


def descriptor_for_traffic(vessel: TrafficVessel) -> VesselDescriptor:
    return VesselDescriptor(
        east=vessel.position_east,
        north=vessel.position_north,
        heading_deg=vessel.heading_deg,
        speed=vessel.speed,
        kind=vessel.kind,
    )


@dataclass(frozen=True)
class EncounterGeometry:
    distance: float
    cpa_distance: float
    cpa_time: float
    closing: bool


@dataclass(frozen=True)
class RightOfWayAssessment:
    vessel_id: str
    vessel_kind: str
    distance: float
    cpa_distance: float
    cpa_time: float
    player_must_give_way: bool
    reason: str
    warning: bool
    violation: bool


@dataclass(frozen=True)
class SailForces:
    target_speed: float
    target_heel_deg: float
    trim_efficiency: float
    heel_load: float
    angle_efficiency: float
    relative_wind_deg: float
    ideal_sheet: float
    overpower: float


@dataclass(frozen=True)
class StabilityState:
    heel_abs_deg: float
    heel_limit_deg: float
    heel_fraction: float
    capsize_progress: float
    risk: float
    status: str


def ideal_sheet_for_angle(abs_relative_wind_deg: float) -> float:
    if abs_relative_wind_deg < NO_GO_ZONE_DEG:
        return 0.0
    reach_progress = (abs_relative_wind_deg - NO_GO_ZONE_DEG) / (
        180.0 - NO_GO_ZONE_DEG
    )
    return clamp(reach_progress, 0.0, 1.0)


def calculate_sail_forces(boat: BoatState, wind: WindState) -> SailForces:
    relative = signed_angle_diff(wind.from_deg, boat.heading_deg)
    abs_relative = abs(relative)

    ideal_sheet = ideal_sheet_for_angle(abs_relative)
    trim_error = abs(boat.sheet - ideal_sheet)
    trim_efficiency = clamp(1.0 - trim_error * 1.75, 0.0, 1.0)
    overtrim = max(0.0, ideal_sheet - boat.sheet)
    undertrim = max(0.0, boat.sheet - ideal_sheet)
    heel_load = clamp(
        1.0 - undertrim * 2.25 - overtrim * 0.42 + overtrim * 0.35,
        0.0,
        1.12,
    )

    if abs_relative < NO_GO_ZONE_DEG:
        angle_efficiency = 0.0
    else:
        lift_drive = max(0.0, sin(radians(abs_relative))) ** 0.72
        downwind_drive = 0.68 * (
            (abs_relative - NO_GO_ZONE_DEG) / (180.0 - NO_GO_ZONE_DEG)
        ) ** 1.18
        angle_efficiency = clamp(max(lift_drive, downwind_drive), 0.0, 1.0)

    heel_side_force = max(0.0, sin(radians(min(abs_relative, 165.0))))
    if abs_relative > 145.0:
        heel_side_force *= 0.45
    sheet_power = 1.18 - boat.sheet * 0.38
    target_heel = (
        heel_side_force
        * wind.speed
        * wind.speed
        / 3.15
        * heel_load
        * sheet_power
    )
    target_heel = clamp(target_heel, 0.0, 86.0)
    if relative < 0.0:
        target_heel *= -1.0

    heel_penalty = clamp((abs(boat.heel_deg) - 38.0) / 42.0, 0.0, 0.55)
    target_speed = (
        wind.speed * 0.58 * angle_efficiency * trim_efficiency * (1.0 - heel_penalty)
    )
    target_speed = clamp(target_speed, 0.0, MAX_BOAT_SPEED)

    overpower = clamp((abs(target_heel) - DANGER_HEEL_DEG) / 28.0, 0.0, 1.0)
    return SailForces(
        target_speed=target_speed,
        target_heel_deg=target_heel,
        trim_efficiency=trim_efficiency,
        heel_load=heel_load,
        angle_efficiency=angle_efficiency,
        relative_wind_deg=relative,
        ideal_sheet=ideal_sheet,
        overpower=overpower,
    )


def vessel_tack(heading_deg: float, wind: WindState) -> str:
    relative = signed_angle_diff(wind.from_deg, heading_deg)
    if abs(relative) < 4.0 or abs(relative) > 176.0:
        return "neutral"
    return "starboard" if relative > 0.0 else "port"


def vessel_velocity_world(heading_deg: float, speed: float) -> tuple[float, float]:
    east, north = nav_vector(heading_deg)
    return (
        east * speed * WORLD_UNITS_PER_KNOT_SECOND,
        north * speed * WORLD_UNITS_PER_KNOT_SECOND,
    )


def is_overtaking(overtaker: VesselDescriptor, ahead: VesselDescriptor) -> bool:
    """COLREGS rule 13: coming up from more than 22.5 deg abaft the beam.

    Geometry alone is not enough: the overtaker must also be on a broadly
    similar course and actually faster, otherwise a boat merely sitting astern
    would count as "overtaking" forever.
    """

    bearing_from_ahead = bearing_between(
        ahead.east, ahead.north, overtaker.east, overtaker.north
    )
    relative_bearing = abs(signed_angle_diff(bearing_from_ahead, ahead.heading_deg))
    coming_from_astern = relative_bearing > OVERTAKING_BEARING_DEG
    similar_course = (
        abs(signed_angle_diff(overtaker.heading_deg, ahead.heading_deg))
        < OVERTAKING_COURSE_TOLERANCE_DEG
    )
    faster = overtaker.speed > ahead.speed + OVERTAKING_SPEED_MARGIN
    return coming_from_astern and similar_course and faster


def is_windward(
    own: VesselDescriptor, other: VesselDescriptor, wind: WindState
) -> bool:
    """True when ``own`` sits further upwind than ``other``."""

    up_east, up_north = nav_vector(wind.from_deg)
    return (own.east - other.east) * up_east + (own.north - other.north) * up_north > 0.0


def give_way_reason(
    own: VesselDescriptor, other: VesselDescriptor, wind: WindState
) -> tuple[bool, str]:
    """Decide whether ``own`` must keep clear of ``other`` and explain why.

    Rule order follows COLREGS: overtaking trumps everything, then the sailing
    rules (12), then power gives way to sail (18), then the power-power
    head-on/crossing rules (14/15).
    """

    if is_overtaking(own, other):
        return True, "overtaking vessel keeps clear"
    if is_overtaking(other, own):
        return False, "stand on: you are being overtaken"

    own_sail = own.kind == "sailboat"
    other_sail = other.kind == "sailboat"

    if own_sail and other_sail:
        own_tack = vessel_tack(own.heading_deg, wind)
        other_tack = vessel_tack(other.heading_deg, wind)
        if own_tack == "port" and other_tack == "starboard":
            return True, "port tack gives way to starboard tack"
        if own_tack == "starboard" and other_tack == "port":
            return False, "stand on: starboard tack over port"
        if own_tack == other_tack and own_tack != "neutral":
            if is_windward(own, other, wind):
                return True, "windward boat keeps clear"
            return False, "stand on: leeward boat"
        return False, "no clear give-way rule"

    if own_sail and not other_sail:
        return False, "stand on: power gives way to sail"
    if other_sail and not own_sail:
        return True, "power gives way to sail"

    heading_diff = abs(signed_angle_diff(own.heading_deg, other.heading_deg))
    if heading_diff > HEAD_ON_TOLERANCE_DEG:
        return True, "head-on: alter course to starboard"
    bearing_to_other = bearing_between(own.east, own.north, other.east, other.north)
    relative_bearing = signed_angle_diff(bearing_to_other, own.heading_deg)
    if 0.0 < relative_bearing < OVERTAKING_BEARING_DEG:
        return True, "crossing: give way to vessel to starboard"
    return False, "stand on: crossing vessel to port"


def encounter_geometry(
    own: VesselDescriptor, other: VesselDescriptor
) -> EncounterGeometry:
    rel_east = other.east - own.east
    rel_north = other.north - own.north
    distance = hypot(rel_east, rel_north)

    own_v_east, own_v_north = vessel_velocity_world(own.heading_deg, own.speed)
    other_v_east, other_v_north = vessel_velocity_world(other.heading_deg, other.speed)
    rel_v_east = other_v_east - own_v_east
    rel_v_north = other_v_north - own_v_north
    rel_speed_sq = rel_v_east * rel_v_east + rel_v_north * rel_v_north

    approach = rel_east * rel_v_east + rel_north * rel_v_north
    closing = approach < 0.0
    if rel_speed_sq <= 0.0001:
        cpa_time = 0.0
    else:
        cpa_time = clamp(-approach / rel_speed_sq, 0.0, TRAFFIC_LOOKAHEAD_SECONDS)

    cpa_east = rel_east + rel_v_east * cpa_time
    cpa_north = rel_north + rel_v_north * cpa_time
    return EncounterGeometry(
        distance=distance,
        cpa_distance=hypot(cpa_east, cpa_north),
        cpa_time=cpa_time,
        closing=closing,
    )


def assess_right_of_way(
    player: BoatState, traffic: TrafficVessel, wind: WindState
) -> RightOfWayAssessment:
    own = descriptor_for_boat(player)
    other = descriptor_for_traffic(traffic)
    geometry = encounter_geometry(own, other)
    player_must_give_way, reason = give_way_reason(own, other, wind)

    warning = (
        geometry.distance < TRAFFIC_WARNING_DISTANCE
        and geometry.closing
        and geometry.cpa_time < TRAFFIC_LOOKAHEAD_SECONDS
        and geometry.cpa_distance < TRAFFIC_CPA_DISTANCE
    )
    violation = player_must_give_way and (
        geometry.distance < TRAFFIC_VIOLATION_DISTANCE
        or (
            geometry.cpa_time < 4.0
            and geometry.cpa_distance < TRAFFIC_VIOLATION_DISTANCE * 0.65
        )
    )
    return RightOfWayAssessment(
        vessel_id=traffic.id,
        vessel_kind=traffic.kind,
        distance=geometry.distance,
        cpa_distance=geometry.cpa_distance,
        cpa_time=geometry.cpa_time,
        player_must_give_way=player_must_give_way,
        reason=reason,
        warning=warning,
        violation=violation,
    )


def assess_traffic_right_of_way(
    player: BoatState, traffic: list[TrafficVessel], wind: WindState
) -> list[RightOfWayAssessment]:
    return [assess_right_of_way(player, vessel, wind) for vessel in traffic]


def update_traffic_vessel(
    vessel: TrafficVessel, player: BoatState, wind: WindState, dt: float
) -> None:
    """Steer one traffic vessel: follow its route, keep clear when required."""

    desired = vessel.desired_heading()
    target_speed = vessel.cruise_speed

    own = descriptor_for_traffic(vessel)
    other = descriptor_for_boat(player)
    geometry = encounter_geometry(own, other)
    gives_way, _reason = give_way_reason(own, other, wind)

    vessel.avoiding = (
        gives_way
        and geometry.distance < TRAFFIC_AVOID_RANGE
        and geometry.cpa_distance < TRAFFIC_AVOID_CPA_DISTANCE
        and geometry.cpa_time < TRAFFIC_LOOKAHEAD_SECONDS
    )
    if vessel.avoiding:
        desired = normalize_degrees(desired + TRAFFIC_AVOID_TURN_DEG)
        target_speed = vessel.cruise_speed * TRAFFIC_AVOID_SPEED_FACTOR

    vessel.heading_deg = turn_toward(
        vessel.heading_deg, desired, TRAFFIC_TURN_RATE_DEG * max(0.0, dt)
    )
    vessel.speed = move_toward(vessel.speed, target_speed, TRAFFIC_ACCELERATION * dt)
    vessel.advance(dt)


def find_rock_collision(boat: BoatState, rocks: list[Rock]) -> Rock | None:
    for rock in rocks:
        distance = hypot(
            rock.east - boat.position_east, rock.north - boat.position_north
        )
        if distance < rock.radius + PLAYER_HULL_RADIUS * 0.7:
            return rock
    return None


def find_vessel_collision(
    boat: BoatState, traffic: list[TrafficVessel]
) -> TrafficVessel | None:
    for vessel in traffic:
        distance = hypot(
            vessel.position_east - boat.position_east,
            vessel.position_north - boat.position_north,
        )
        if distance < PLAYER_HULL_RADIUS + TRAFFIC_HULL_RADIUS:
            return vessel
    return None


def stability_state(boat: BoatState) -> StabilityState:
    heel_abs = abs(boat.heel_deg)
    heel_fraction = clamp(heel_abs / CAPSIZE_HEEL_DEG, 0.0, 1.25)
    capsize_progress = clamp(boat.capsize_timer / CAPSIZE_TIMER_LIMIT, 0.0, 1.0)
    risk = clamp(max(heel_fraction, capsize_progress), 0.0, 1.0)

    if boat.capsized:
        status = "capsized"
    elif heel_abs >= DANGER_HEEL_DEG or capsize_progress >= 0.72:
        status = "danger"
    elif heel_abs >= WARNING_HEEL_DEG or capsize_progress >= 0.35:
        status = "warning"
    else:
        status = "stable"

    return StabilityState(
        heel_abs_deg=heel_abs,
        heel_limit_deg=CAPSIZE_HEEL_DEG,
        heel_fraction=heel_fraction,
        capsize_progress=capsize_progress,
        risk=risk,
        status=status,
    )


def point_of_sail(abs_relative_wind_deg: float) -> str:
    if abs_relative_wind_deg < NO_GO_ZONE_DEG:
        return "in irons"
    if abs_relative_wind_deg < 60.0:
        return "close hauled"
    if abs_relative_wind_deg < 112.0:
        return "beam reach"
    if abs_relative_wind_deg < 155.0:
        return "broad reach"
    return "running"


def sailing_terms(boat: BoatState, wind: WindState) -> dict[str, str]:
    relative = signed_angle_diff(wind.from_deg, boat.heading_deg)
    abs_relative = abs(relative)

    if boat.capsized:
        tack = "capsized"
        beam = "capsized"
        point = "capsized"
    elif abs_relative < 4.0 or abs_relative > 176.0:
        tack = "neutral"
        beam = "centerline"
        point = point_of_sail(abs_relative)
    else:
        tack_side = "starboard" if relative > 0.0 else "port"
        tack = f"{tack_side} tack"
        if abs_relative < NO_GO_ZONE_DEG:
            beam = "on the bow"
        elif abs_relative > 155.0:
            beam = "over the stern"
        else:
            beam = f"{tack_side} beam"
        point = point_of_sail(abs_relative)

    return {
        "tack": tack,
        "beam": beam,
        "point_of_sail": point,
    }


def classify_crossing(old_relative: float, new_relative: float) -> str | None:
    old_sign = 1 if old_relative > 0.0 else -1 if old_relative < 0.0 else 0
    new_sign = 1 if new_relative > 0.0 else -1 if new_relative < 0.0 else 0
    if old_sign == 0 or new_sign == 0 or old_sign == new_sign:
        return None

    if min(abs(old_relative), abs(new_relative)) < 95.0:
        return "you have tacked"
    return "you have gybed"


@dataclass
class ActionLog:
    capacity: int = 6
    messages: list[str] = field(default_factory=list)

    def push(self, message: str) -> None:
        if self.messages and self.messages[-1] == message:
            return
        self.messages.append(message)
        if len(self.messages) > self.capacity:
            self.messages = self.messages[-self.capacity :]

    @property
    def latest(self) -> str:
        if not self.messages:
            return "ready to sail"
        return self.messages[-1]


@dataclass
class SailingSimulation:
    boat: BoatState = field(default_factory=BoatState)
    wind: WindState = field(default_factory=WindState)
    traffic: list[TrafficVessel] = field(default_factory=list)
    rocks: list[Rock] = field(default_factory=list)
    actions: ActionLog = field(default_factory=ActionLog)
    action_cooldown: float = 0.0
    right_of_way: list[RightOfWayAssessment] = field(default_factory=list)
    penalty_timer: float = 0.0
    penalty_cooldown: float = 0.0
    penalty_count: int = 0
    grounded: bool = False
    collided_vessel_id: str | None = None

    @property
    def wrecked(self) -> bool:
        """True when the boat cannot sail on without a respawn or recovery."""

        return self.boat.capsized or self.grounded or self.collided_vessel_id is not None

    def adjust_tiller(self, direction: float, dt: float) -> None:
        self.boat.move_tiller(direction * dt * 1.85)

    def adjust_sheet(self, direction: float, dt: float) -> None:
        self.set_sheet(self.boat.sheet + direction * dt * 0.72)

    def set_sheet(self, sheet: float) -> None:
        old_sheet = self.boat.sheet
        self.boat.sheet = clamp(sheet, 0.0, 1.0)
        if abs(self.boat.sheet - old_sheet) > 0.002:
            if self.boat.sheet < old_sheet:
                self.actions.push("you trimmed in the main sheet")
            else:
                self.actions.push("you eased the main sheet")

    def center_tiller(self, dt: float) -> None:
        self.boat.center_tiller(dt)

    def right_boat(self) -> None:
        if self.boat.capsized:
            self.boat.right_after_capsize()
            self.actions.push("you righted the boat")

    def respawn_at(self, east: float, north: float, heading_deg: float) -> None:
        boat = self.boat
        boat.position_east = east
        boat.position_north = north
        boat.heading_deg = normalize_degrees(heading_deg)
        boat.speed = 0.5
        boat.heel_deg = 0.0
        boat.tiller = 0.0
        boat.sheet = 0.6
        boat.capsized = False
        boat.capsize_timer = 0.0
        self.grounded = False
        self.collided_vessel_id = None
        self.penalty_timer = 0.0
        self.actions.push("back on the water")

    def step(self, dt: float) -> SailForces:
        dt = clamp(dt, 0.0, 0.08)
        old_relative = signed_angle_diff(self.wind.from_deg, self.boat.heading_deg)

        self.wind.update(dt)
        forces = calculate_sail_forces(self.boat, self.wind)
        self._integrate_boat(dt, forces)
        self._integrate_traffic(dt)
        self._check_collisions()

        new_relative = signed_angle_diff(self.wind.from_deg, self.boat.heading_deg)
        self._record_sailing_events(dt, old_relative, new_relative)
        self._record_traffic_events(dt)
        return calculate_sail_forces(self.boat, self.wind)

    def _integrate_traffic(self, dt: float) -> None:
        for vessel in self.traffic:
            update_traffic_vessel(vessel, self.boat, self.wind, dt)

    def _integrate_boat(self, dt: float, forces: SailForces) -> None:
        boat = self.boat
        if boat.capsized:
            boat.speed = move_toward(boat.speed, 0.0, dt * 1.6)
            boat.heel_deg = move_toward(boat.heel_deg, 90.0, dt * 18.0)
            return
        if self.grounded or self.collided_vessel_id is not None:
            boat.speed = move_toward(boat.speed, 0.0, dt * 4.0)
            return

        rudder_effect = clamp(0.22 + boat.speed / 4.0, 0.22, 1.0)
        turn_rate = boat.tiller * 58.0 * rudder_effect

        if forces.overpower > 0.0 and abs(forces.relative_wind_deg) > NO_GO_ZONE_DEG:
            weather_helm = (1.0 if forces.relative_wind_deg > 0.0 else -1.0) * (
                13.0 * forces.overpower
            )
            turn_rate += weather_helm

        boat.heading_deg = normalize_degrees(boat.heading_deg + turn_rate * dt)
        boat.speed += (forces.target_speed - boat.speed) * min(1.0, dt * 0.78)
        boat.heel_deg += (forces.target_heel_deg - boat.heel_deg) * min(1.0, dt * 1.65)

        capsize_pressure = max(abs(forces.target_heel_deg), abs(boat.heel_deg))
        if capsize_pressure > DANGER_HEEL_DEG:
            load_excess = clamp(
                (capsize_pressure - DANGER_HEEL_DEG)
                / (CAPSIZE_HEEL_DEG - DANGER_HEEL_DEG),
                0.0,
                1.6,
            )
            righting_loss = clamp(
                (abs(boat.heel_deg) - WARNING_HEEL_DEG)
                / (KNOCKDOWN_HEEL_DEG - WARNING_HEEL_DEG),
                0.0,
                1.0,
            )
            boat.capsize_timer += dt * (
                0.35 + load_excess * 1.15 + righting_loss * 1.05 + forces.overpower * 0.65
            )
        else:
            recovery_rate = 1.35 if capsize_pressure < WARNING_HEEL_DEG else 0.45
            boat.capsize_timer = max(0.0, boat.capsize_timer - dt * recovery_rate)

        if abs(boat.heel_deg) > KNOCKDOWN_HEEL_DEG or boat.capsize_timer > CAPSIZE_TIMER_LIMIT:
            boat.capsized = True
            boat.speed *= 0.25
            boat.heel_deg = 88.0 if boat.heel_deg >= 0.0 else -88.0
            self.actions.push("you capsized")
            return

        east, north = nav_vector(boat.heading_deg)
        distance = boat.speed * dt * WORLD_UNITS_PER_KNOT_SECOND
        boat.position_east += east * distance
        boat.position_north += north * distance

    def _check_collisions(self) -> None:
        if self.boat.capsized:
            return
        if not self.grounded and find_rock_collision(self.boat, self.rocks):
            self.grounded = True
            self.boat.speed = 0.0
            self.actions.push("you ran aground on a rock")
            return
        if self.collided_vessel_id is None:
            vessel = find_vessel_collision(self.boat, self.traffic)
            if vessel is not None:
                self.collided_vessel_id = vessel.id
                self.boat.speed = 0.0
                self.actions.push(f"you collided with {vessel.id}")

    def _record_sailing_events(
        self, dt: float, old_relative: float, new_relative: float
    ) -> None:
        crossing = classify_crossing(old_relative, new_relative)
        if crossing:
            self.actions.push(crossing)
            self.action_cooldown = 0.85
            return

        self.action_cooldown = max(0.0, self.action_cooldown - dt)
        if self.action_cooldown > 0.0:
            return

        old_abs = abs(old_relative)
        new_abs = abs(new_relative)
        if old_abs < NO_GO_ZONE_DEG or old_abs > 170.0:
            return

        delta = old_abs - new_abs
        if delta > 2.4:
            self.actions.push("you are heading up")
            self.action_cooldown = 0.75
        elif delta < -2.4 and new_abs < 170.0:
            self.actions.push("you are bearing away")
            self.action_cooldown = 0.75

    def _record_traffic_events(self, dt: float) -> None:
        self.penalty_timer = max(0.0, self.penalty_timer - dt)
        self.penalty_cooldown = max(0.0, self.penalty_cooldown - dt)
        self.right_of_way = assess_traffic_right_of_way(
            self.boat, self.traffic, self.wind
        )
        if self.penalty_cooldown > 0.0:
            return

        for assessment in self.right_of_way:
            if assessment.violation:
                self.penalty_timer = TRAFFIC_PENALTY_SECONDS
                self.penalty_cooldown = TRAFFIC_PENALTY_COOLDOWN
                self.penalty_count += 1
                self.actions.push(
                    f"rule violation: {assessment.reason} near {assessment.vessel_id}"
                )
                return
