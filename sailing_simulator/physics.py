"""Display-free sailing simulation primitives.

The model is tuned for a readable game rather than exact hydrodynamics. It
captures the main cause-and-effect loops a player expects: sail trim and point
of sail drive speed, beam/close reaching creates heel, high heel slows the boat,
and sustained overpowering can capsize it.
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
TRAFFIC_LOOKAHEAD_SECONDS = 12.0
TRAFFIC_WARNING_DISTANCE = 170.0
TRAFFIC_CPA_DISTANCE = 70.0
TRAFFIC_VIOLATION_DISTANCE = 58.0
TRAFFIC_PENALTY_SECONDS = 1.6
TRAFFIC_PENALTY_COOLDOWN = 2.4


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
    """Wind direction is where the wind comes from, in navigation degrees."""

    from_deg: float = 55.0
    speed: float = 11.0
    minimum_speed: float = 4.0
    maximum_speed: float = 19.0
    elapsed: float = 0.0

    def update(self, dt: float) -> None:
        dt = max(0.0, dt)
        self.elapsed += dt

        gust_target = 11.5 + 4.2 * sin(self.elapsed * 0.17) + 1.8 * sin(
            self.elapsed * 0.43 + 1.1
        )
        self.speed += (gust_target - self.speed) * min(1.0, dt * 0.28)
        self.speed = clamp(self.speed, self.minimum_speed, self.maximum_speed)

        direction_rate = 2.4 * sin(self.elapsed * 0.08) + 1.1 * sin(
            self.elapsed * 0.21 + 0.8
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


@dataclass
class TrafficVessel:
    id: str
    kind: str
    position_east: float
    position_north: float
    heading_deg: float
    speed: float

    def advance(self, dt: float) -> None:
        east, north = nav_vector(self.heading_deg)
        distance = self.speed * max(0.0, dt) * WORLD_UNITS_PER_KNOT_SECOND
        self.position_east += east * distance
        self.position_north += north * distance


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


def default_traffic() -> list[TrafficVessel]:
    return [
        TrafficVessel("M1", "motorboat", -430.0, 220.0, 82.0, 5.1),
        TrafficVessel("S1", "sailboat", 320.0, 520.0, 220.0, 3.1),
        TrafficVessel("M2", "motorboat", 420.0, 980.0, 282.0, 4.8),
        TrafficVessel("S2", "sailboat", -360.0, 1320.0, 24.0, 2.8),
    ]


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


def is_overtaking(
    overtaker_east: float,
    overtaker_north: float,
    target_east: float,
    target_north: float,
    target_heading_deg: float,
) -> bool:
    bearing_from_target = bearing_between(
        target_east, target_north, overtaker_east, overtaker_north
    )
    relative_bearing = abs(signed_angle_diff(bearing_from_target, target_heading_deg))
    return relative_bearing > 112.5


def player_is_windward(
    player: BoatState, traffic: TrafficVessel, wind: WindState
) -> bool:
    windward_east, windward_north = nav_vector(wind.from_deg)
    traffic_from_player_east = traffic.position_east - player.position_east
    traffic_from_player_north = traffic.position_north - player.position_north
    return (
        traffic_from_player_east * windward_east
        + traffic_from_player_north * windward_north
        < 0.0
    )


def right_of_way_reason(
    player: BoatState, traffic: TrafficVessel, wind: WindState
) -> tuple[bool, str]:
    if is_overtaking(
        player.position_east,
        player.position_north,
        traffic.position_east,
        traffic.position_north,
        traffic.heading_deg,
    ):
        return True, "overtaking vessel must keep clear"

    if is_overtaking(
        traffic.position_east,
        traffic.position_north,
        player.position_east,
        player.position_north,
        player.heading_deg,
    ):
        return False, "traffic is overtaking you"

    if traffic.kind == "motorboat":
        return False, "sail has right of way over power"

    player_tack = vessel_tack(player.heading_deg, wind)
    traffic_tack = vessel_tack(traffic.heading_deg, wind)
    if player_tack == "port" and traffic_tack == "starboard":
        return True, "port tack gives way to starboard tack"
    if player_tack == "starboard" and traffic_tack == "port":
        return False, "starboard tack stands on"

    if player_tack == traffic_tack and player_tack != "neutral":
        if player_is_windward(player, traffic, wind):
            return True, "windward boat keeps clear"
        return False, "leeward boat stands on"

    return False, "no clear give-way rule"


def assess_right_of_way(
    player: BoatState, traffic: TrafficVessel, wind: WindState
) -> RightOfWayAssessment:
    rel_east = traffic.position_east - player.position_east
    rel_north = traffic.position_north - player.position_north
    distance = hypot(rel_east, rel_north)

    player_v_east, player_v_north = vessel_velocity_world(player.heading_deg, player.speed)
    traffic_v_east, traffic_v_north = vessel_velocity_world(
        traffic.heading_deg, traffic.speed
    )
    rel_v_east = traffic_v_east - player_v_east
    rel_v_north = traffic_v_north - player_v_north
    rel_speed_sq = rel_v_east * rel_v_east + rel_v_north * rel_v_north
    if rel_speed_sq <= 0.0001:
        cpa_time = 0.0
    else:
        cpa_time = -(
            rel_east * rel_v_east + rel_north * rel_v_north
        ) / rel_speed_sq
        cpa_time = clamp(cpa_time, 0.0, TRAFFIC_LOOKAHEAD_SECONDS)

    cpa_east = rel_east + rel_v_east * cpa_time
    cpa_north = rel_north + rel_v_north * cpa_time
    cpa_distance = hypot(cpa_east, cpa_north)
    player_must_give_way, reason = right_of_way_reason(player, traffic, wind)

    warning = (
        distance < TRAFFIC_WARNING_DISTANCE
        and cpa_time < TRAFFIC_LOOKAHEAD_SECONDS
        and cpa_distance < TRAFFIC_CPA_DISTANCE
    )
    violation = player_must_give_way and (
        distance < TRAFFIC_VIOLATION_DISTANCE
        or (cpa_time < 4.0 and cpa_distance < TRAFFIC_VIOLATION_DISTANCE * 0.65)
    )
    return RightOfWayAssessment(
        vessel_id=traffic.id,
        vessel_kind=traffic.kind,
        distance=distance,
        cpa_distance=cpa_distance,
        cpa_time=cpa_time,
        player_must_give_way=player_must_give_way,
        reason=reason,
        warning=warning,
        violation=violation,
    )


def assess_traffic_right_of_way(
    player: BoatState, traffic: list[TrafficVessel], wind: WindState
) -> list[RightOfWayAssessment]:
    return [assess_right_of_way(player, vessel, wind) for vessel in traffic]


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
    actions: ActionLog = field(default_factory=ActionLog)
    action_cooldown: float = 0.0
    right_of_way: list[RightOfWayAssessment] = field(default_factory=list)
    penalty_timer: float = 0.0
    penalty_cooldown: float = 0.0

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

    def step(self, dt: float) -> SailForces:
        dt = clamp(dt, 0.0, 0.08)
        old_relative = signed_angle_diff(self.wind.from_deg, self.boat.heading_deg)

        self.wind.update(dt)
        forces = calculate_sail_forces(self.boat, self.wind)
        self._integrate_boat(dt, forces)
        self._integrate_traffic(dt)

        new_relative = signed_angle_diff(self.wind.from_deg, self.boat.heading_deg)
        self._record_sailing_events(dt, old_relative, new_relative)
        self._record_traffic_events(dt)
        return calculate_sail_forces(self.boat, self.wind)

    def _integrate_traffic(self, dt: float) -> None:
        for vessel in self.traffic:
            vessel.advance(dt)

    def _integrate_boat(self, dt: float, forces: SailForces) -> None:
        boat = self.boat
        if boat.capsized:
            boat.speed = move_toward(boat.speed, 0.0, dt * 1.6)
            boat.heel_deg = move_toward(boat.heel_deg, 90.0, dt * 18.0)
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
                self.actions.push(
                    f"rule violation: {assessment.reason} near {assessment.vessel_id}"
                )
                return
