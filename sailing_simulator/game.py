"""PyGame interface for the sailing simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin
from typing import Iterable

import pygame

from .physics import (
    MAX_SHEET_ANGLE_DEG,
    CAPSIZE_HEEL_DEG,
    DANGER_HEEL_DEG,
    NO_GO_ZONE_DEG,
    SailingSimulation,
    TrafficVessel,
    WARNING_HEEL_DEG,
    calculate_sail_forces,
    clamp,
    default_traffic,
    nav_vector,
    sailing_terms,
    signed_angle_diff,
    stability_state,
)


WIDTH = 1000
HEIGHT = 700
BOAT_CENTER = (WIDTH // 2, HEIGHT // 2)
WATER = (26, 94, 126)
WATER_DARK = (17, 67, 92)
FOAM = (204, 232, 229)
HULL = (238, 244, 239)
HULL_SHADOW = (54, 71, 78)
SAIL = (245, 248, 234)
SAIL_SHADE = (205, 218, 205)
WOOD = (140, 84, 42)
PANEL = (12, 30, 40)
PANEL_BORDER = (115, 159, 166)
TEXT = (236, 246, 244)
MUTED = (172, 205, 204)
WARNING = (255, 196, 87)
DANGER = (245, 98, 76)
BLACK = (5, 10, 12)
ROCK = (92, 83, 74)
ROCK_LIGHT = (136, 127, 112)
BUOY_RED = (220, 68, 58)
BUOY_GREEN = (54, 172, 107)
BUOY_YELLOW = (242, 205, 83)
MOTORBOAT = (232, 128, 64)
TRAFFIC_SAIL = (226, 232, 218)
BOOM_LENGTH = 68.0
SAIL_HEAD_FORWARD = 58.0
MIN_BOOM_ANGLE_DEG = 8.0
GAME_SPEED = 0.72
DEFAULT_COURSE_WIND_FROM_DEG = 55.0
POINT_OF_SAIL_BANDS = [
    ("irons", 0.0, NO_GO_ZONE_DEG, DANGER),
    ("close", NO_GO_ZONE_DEG, 60.0, WARNING),
    ("beam", 60.0, 112.0, (78, 198, 133)),
    ("broad", 112.0, 155.0, BUOY_YELLOW),
    ("run", 155.0, 180.0, MUTED),
]


@dataclass
class CourseMarker:
    kind: str
    east: float
    north: float
    label: str = ""
    color: tuple[int, int, int] = BUOY_YELLOW


@dataclass
class NoSailZoneToggle:
    rect: pygame.Rect
    enabled: bool = True

    @classmethod
    def default(cls) -> "NoSailZoneToggle":
        return cls(pygame.Rect(WIDTH - 220, 150, 190, 52))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.enabled = not self.enabled
                return True
        return False


@dataclass
class MainSheetLever:
    rect: pygame.Rect
    dragging: bool = False

    @classmethod
    def default(cls) -> "MainSheetLever":
        return cls(pygame.Rect(WIDTH - 230, HEIGHT - 252, 204, 222))

    @property
    def track_rect(self) -> pygame.Rect:
        return pygame.Rect(self.rect.right - 66, self.rect.y + 58, 18, self.rect.height - 98)

    def knob_rect(self, sim: SailingSimulation) -> pygame.Rect:
        track = self.track_rect
        knob_y = sheet_to_lever_y(sim.boat.sheet, track)
        return pygame.Rect(0, 0, 50, 24).move(
            track.centerx - 25,
            int(round(knob_y)) - 12,
        )

    def hit_rect(self, sim: SailingSimulation) -> pygame.Rect:
        return self.knob_rect(sim).union(self.track_rect.inflate(30, 20))

    def set_sheet_from_pos(self, sim: SailingSimulation, pos: tuple[int, int]) -> None:
        sim.set_sheet(sheet_from_lever_y(pos[1], self.track_rect))

    def handle_event(self, sim: SailingSimulation, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hit_rect(sim).collidepoint(event.pos):
                self.dragging = True
                self.set_sheet_from_pos(sim, event.pos)
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_sheet_from_pos(sim, event.pos)
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_dragging = self.dragging
            self.dragging = False
            return was_dragging
        return False


@dataclass
class SailVisualState:
    """Smoothed top-down mainsail state for rendering.

    Positive boom angles are to starboard, negative angles are to port, measured
    from the boat's aft centerline. Keeping this separate from the physics state
    gives the sail a little inertia instead of snapping on every tack or gybe.
    """

    boom_angle_deg: float = -52.0
    flutter_phase: float = 0.0

    @classmethod
    def from_sim(cls, sim: SailingSimulation) -> "SailVisualState":
        return cls(boom_angle_deg=desired_boom_angle(sim))

    def update(self, sim: SailingSimulation, dt: float) -> None:
        dt = clamp(dt, 0.0, 0.08)
        target = desired_boom_angle(sim)
        response = 9.0
        if self.boom_angle_deg * target < 0.0:
            response = 4.2
        if sim.boat.capsized:
            response = 2.0

        self.boom_angle_deg += (target - self.boom_angle_deg) * min(1.0, dt * response)

        forces = calculate_sail_forces(sim.boat, sim.wind)
        luffing = max(0.0, 1.0 - forces.trim_efficiency)
        irons = 1.0 if abs(forces.relative_wind_deg) < 38.0 else 0.0
        self.flutter_phase += dt * (2.3 + sim.wind.speed * 0.18 + luffing * 6.0 + irons * 5.0)


@dataclass(frozen=True)
class SailGeometry:
    mast: tuple[float, float]
    head: tuple[float, float]
    belly: tuple[float, float]
    clew: tuple[float, float]
    sheet_anchor: tuple[float, float]


@dataclass(frozen=True)
class TillerRudderGeometry:
    rudder_pivot: tuple[float, float]
    rudder_tip: tuple[float, float]
    tiller_pivot: tuple[float, float]
    tiller_handle: tuple[float, float]


def sheet_percent_in(sheet: float) -> int:
    return int(round((1.0 - clamp(sheet, 0.0, 1.0)) * 100.0))


def sheet_to_lever_y(sheet: float, track: pygame.Rect) -> float:
    return track.top + clamp(sheet, 0.0, 1.0) * track.height


def sheet_from_lever_y(y: float, track: pygame.Rect) -> float:
    return clamp((y - track.top) / track.height, 0.0, 1.0)


def scaled_game_dt(dt: float) -> float:
    return max(0.0, dt) * GAME_SPEED


def sail_efficiency(sim: SailingSimulation) -> float:
    forces = calculate_sail_forces(sim.boat, sim.wind)
    return clamp(forces.trim_efficiency * forces.angle_efficiency, 0.0, 1.0)


def efficiency_color(efficiency: float) -> tuple[int, int, int]:
    if efficiency >= 0.72:
        return (78, 198, 133)
    if efficiency >= 0.38:
        return WARNING
    return DANGER


def point_of_sail_marker_x(relative_wind_deg: float, track: pygame.Rect) -> int:
    fraction = clamp(abs(relative_wind_deg) / 180.0, 0.0, 1.0)
    return track.x + int(round(track.width * fraction))


def course_point(
    start: tuple[float, float], heading_deg: float, distance: float
) -> tuple[float, float]:
    east, north = nav_vector(heading_deg)
    return start[0] + east * distance, start[1] + north * distance


def default_course_markers() -> list[CourseMarker]:
    start = (0.0, 260.0)
    windward_starboard = DEFAULT_COURSE_WIND_FROM_DEG - 40.0
    windward_port = DEFAULT_COURSE_WIND_FROM_DEG + 40.0
    downwind = (DEFAULT_COURSE_WIND_FROM_DEG + 180.0) % 360.0

    tack_1 = course_point(start, windward_starboard, 520.0)
    tack_2 = course_point(tack_1, windward_port, 520.0)
    tack_3 = course_point(tack_2, windward_starboard, 520.0)
    windward = course_point(tack_3, windward_port, 470.0)
    run_1 = course_point(windward, downwind, 660.0)
    run_2 = course_point(run_1, downwind, 660.0)
    finish = course_point(run_2, downwind, 520.0)

    return [
        CourseMarker("buoy", start[0], start[1], "Start", BUOY_YELLOW),
        CourseMarker("buoy", tack_1[0], tack_1[1], "Tack 1", BUOY_RED),
        CourseMarker("buoy", tack_2[0], tack_2[1], "Tack 2", BUOY_GREEN),
        CourseMarker("buoy", tack_3[0], tack_3[1], "Tack 3", BUOY_RED),
        CourseMarker("buoy", windward[0], windward[1], "Windward", BUOY_YELLOW),
        CourseMarker("buoy", run_1[0], run_1[1], "Run 1", BUOY_GREEN),
        CourseMarker("buoy", run_2[0], run_2[1], "Run 2", BUOY_RED),
        CourseMarker("buoy", finish[0], finish[1], "Finish", BUOY_YELLOW),
        CourseMarker("rock", -260.0, 455.0),
        CourseMarker("rock", 470.0, 610.0),
        CourseMarker("rock", 700.0, 1035.0),
        CourseMarker("rock", 1120.0, 930.0),
        CourseMarker("rock", 365.0, 625.0),
    ]


def world_to_screen(
    sim: SailingSimulation,
    east: float,
    north: float,
    center: tuple[int, int] = BOAT_CENTER,
) -> tuple[float, float]:
    return (
        center[0] + east - sim.boat.position_east,
        center[1] - (north - sim.boat.position_north),
    )


def visible_course_markers(
    sim: SailingSimulation,
    markers: Iterable[CourseMarker],
    margin: float = 80.0,
) -> list[tuple[CourseMarker, tuple[float, float]]]:
    visible = []
    for marker in markers:
        pos = world_to_screen(sim, marker.east, marker.north)
        if -margin <= pos[0] <= WIDTH + margin and -margin <= pos[1] <= HEIGHT + margin:
            visible.append((marker, pos))
    return visible


def visible_traffic(
    sim: SailingSimulation,
    margin: float = 80.0,
) -> list[tuple[TrafficVessel, tuple[float, float]]]:
    visible = []
    for vessel in sim.traffic:
        pos = world_to_screen(sim, vessel.position_east, vessel.position_north)
        if -margin <= pos[0] <= WIDTH + margin and -margin <= pos[1] <= HEIGHT + margin:
            visible.append((vessel, pos))
    return visible


def no_sail_zone_points(
    wind_from_deg: float,
    center: tuple[int, int] = BOAT_CENTER,
    radius: float = 135.0,
    steps: int = 14,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = [center]
    start = wind_from_deg - NO_GO_ZONE_DEG
    span = NO_GO_ZONE_DEG * 2.0
    for index in range(steps + 1):
        heading = start + span * index / steps
        dx, dy = screen_vector(heading, radius)
        points.append((center[0] + dx, center[1] + dy))
    return points


def screen_vector(nav_degrees: float, length: float) -> tuple[float, float]:
    east, north = nav_vector(nav_degrees)
    return east * length, -north * length


def rotate_point(
    local_x: float, local_y: float, heading_deg: float, origin: tuple[float, float]
) -> tuple[float, float]:
    angle = radians(heading_deg)
    sx = local_x * cos(angle) - local_y * sin(angle)
    sy = local_x * sin(angle) + local_y * cos(angle)
    return origin[0] + sx, origin[1] + sy


def boat_polygon(
    center: tuple[int, int], heading_deg: float, length: float = 96.0, width: float = 36.0
) -> list[tuple[float, float]]:
    points = [
        (0.0, -length * 0.55),
        (width * 0.50, -length * 0.12),
        (width * 0.42, length * 0.36),
        (0.0, length * 0.52),
        (-width * 0.42, length * 0.36),
        (-width * 0.50, -length * 0.12),
    ]
    return [rotate_point(x, y, heading_deg, center) for x, y in points]


def traffic_hull_polygon(
    center: tuple[float, float], heading_deg: float, length: float = 52.0, width: float = 20.0
) -> list[tuple[float, float]]:
    points = [
        (0.0, -length * 0.52),
        (width * 0.48, -length * 0.08),
        (width * 0.34, length * 0.38),
        (0.0, length * 0.52),
        (-width * 0.34, length * 0.38),
        (-width * 0.48, -length * 0.08),
    ]
    return [rotate_point(x, y, heading_deg, center) for x, y in points]


def desired_boom_angle(sim: SailingSimulation) -> float:
    relative = signed_angle_diff(sim.wind.from_deg, sim.boat.heading_deg)
    if abs(relative) < 2.0:
        side = -1.0 if sim.boat.heel_deg >= 0.0 else 1.0
    else:
        side = -1.0 if relative > 0.0 else 1.0
    return side * (MIN_BOOM_ANGLE_DEG + sim.boat.sheet * MAX_SHEET_ANGLE_DEG)


def aft_local_vector(angle_deg: float, length: float) -> tuple[float, float]:
    angle = radians(angle_deg)
    return sin(angle) * length, cos(angle) * length


def sail_geometry(
    sim: SailingSimulation,
    sail_visual: SailVisualState,
    center: tuple[int, int] = BOAT_CENTER,
) -> SailGeometry:
    boat = sim.boat
    forces = calculate_sail_forces(boat, sim.wind)
    luffing = max(0.0, 1.0 - forces.trim_efficiency)
    in_irons = 1.0 if abs(forces.relative_wind_deg) < 38.0 else 0.0
    flutter = sin(sail_visual.flutter_phase) * (2.0 + luffing * 7.0 + in_irons * 5.0)
    boom_angle = sail_visual.boom_angle_deg + flutter
    side = 1.0 if boom_angle >= 0.0 else -1.0

    mast_local = (0.0, -8.0)
    boom_x, boom_y = aft_local_vector(boom_angle, BOOM_LENGTH)
    clew_local = (mast_local[0] + boom_x, mast_local[1] + boom_y)
    head_local = (side * (2.0 + luffing * 2.0), mast_local[1] - SAIL_HEAD_FORWARD)

    belly_x = mast_local[0] + boom_x * 0.52 + side * (10.0 + luffing * 5.0)
    belly_y = mast_local[1] + boom_y * 0.50 - 8.0 + sin(sail_visual.flutter_phase * 1.7) * luffing * 4.0
    belly_local = (belly_x, belly_y)

    sheet_anchor_local = (0.0, 36.0)

    return SailGeometry(
        mast=rotate_point(*mast_local, boat.heading_deg, center),
        head=rotate_point(*head_local, boat.heading_deg, center),
        belly=rotate_point(*belly_local, boat.heading_deg, center),
        clew=rotate_point(*clew_local, boat.heading_deg, center),
        sheet_anchor=rotate_point(*sheet_anchor_local, boat.heading_deg, center),
    )


def tiller_rudder_geometry(
    sim: SailingSimulation,
    center: tuple[int, int] = BOAT_CENTER,
) -> TillerRudderGeometry:
    boat = sim.boat
    tiller = clamp(boat.tiller, -1.0, 1.0)

    rudder_pivot_local = (0.0, 44.0)
    rudder_tip_local = (tiller * 22.0, 78.0)
    tiller_pivot_local = (0.0, 34.0)
    tiller_handle_local = (-tiller * 30.0, 0.0)

    return TillerRudderGeometry(
        rudder_pivot=rotate_point(*rudder_pivot_local, boat.heading_deg, center),
        rudder_tip=rotate_point(*rudder_tip_local, boat.heading_deg, center),
        tiller_pivot=rotate_point(*tiller_pivot_local, boat.heading_deg, center),
        tiller_handle=rotate_point(*tiller_handle_local, boat.heading_deg, center),
    )


def draw_arrow(
    surface: pygame.Surface,
    start: tuple[float, float],
    nav_degrees: float,
    length: float,
    color: tuple[int, int, int],
    width: int = 4,
) -> None:
    dx, dy = screen_vector(nav_degrees, length)
    end = (start[0] + dx, start[1] + dy)
    pygame.draw.line(surface, color, start, end, width)

    angle = atan2(dy, dx)
    head_len = 15
    left = (
        end[0] - head_len * cos(angle - 0.55),
        end[1] - head_len * sin(angle - 0.55),
    )
    right = (
        end[0] - head_len * cos(angle + 0.55),
        end[1] - head_len * sin(angle + 0.55),
    )
    pygame.draw.polygon(surface, color, [end, left, right])


def draw_water(surface: pygame.Surface, wind_from_deg: float, time_s: float) -> None:
    surface.fill(WATER)
    wind_to_deg = (wind_from_deg + 180.0) % 360.0
    dx, dy = screen_vector(wind_to_deg + 90.0, 1.0)
    for idx in range(18):
        y = 28 + idx * 40 + sin(time_s * 0.7 + idx) * 5
        offset = sin(time_s * 0.35 + idx * 0.8) * 40
        for x in range(-120, WIDTH + 120, 170):
            start = (x + offset, y)
            end = (start[0] + dx * 58, start[1] + dy * 13)
            pygame.draw.line(surface, WATER_DARK, start, end, 2)


def draw_course_markers(
    surface: pygame.Surface,
    font: pygame.font.Font,
    sim: SailingSimulation,
    markers: Iterable[CourseMarker],
    time_s: float,
) -> None:
    marker_list = list(markers)
    buoy_points = [
        world_to_screen(sim, marker.east, marker.north)
        for marker in marker_list
        if marker.kind == "buoy"
    ]
    if len(buoy_points) > 1:
        pygame.draw.lines(surface, (117, 180, 175), False, buoy_points, 2)

    for marker, pos in visible_course_markers(sim, marker_list):
        bob = sin(time_s * 2.0 + marker.east * 0.017 + marker.north * 0.006) * 3.0
        x, y = pos[0], pos[1] + bob
        if marker.kind == "rock":
            shadow = [(x - 22, y + 14), (x + 24, y + 15), (x + 12, y + 23), (x - 18, y + 21)]
            rock = [(x - 24, y + 10), (x - 12, y - 12), (x + 9, y - 18), (x + 25, y + 5), (x + 12, y + 18), (x - 17, y + 17)]
            pygame.draw.polygon(surface, (13, 48, 63), shadow)
            pygame.draw.polygon(surface, ROCK, rock)
            pygame.draw.line(surface, ROCK_LIGHT, (x - 11, y - 8), (x + 8, y - 13), 2)
            pygame.draw.line(surface, (65, 59, 54), (x - 15, y + 12), (x + 15, y + 10), 2)
        else:
            pygame.draw.line(surface, BLACK, (x, y + 22), (x, y - 18), 3)
            pygame.draw.circle(surface, marker.color, (round(x), round(y)), 13)
            pygame.draw.circle(surface, (246, 246, 230), (round(x), round(y - 3)), 5)
            pygame.draw.polygon(
                surface,
                marker.color,
                [(x, y - 24), (x + 9, y - 10), (x - 9, y - 10)],
            )
            if marker.label:
                label = font.render(marker.label, True, TEXT)
                surface.blit(label, (x - label.get_width() / 2, y + 18))


def draw_traffic_vessels(
    surface: pygame.Surface,
    font: pygame.font.Font,
    sim: SailingSimulation,
    time_s: float,
) -> None:
    assessments = {assessment.vessel_id: assessment for assessment in sim.right_of_way}
    for vessel, pos in visible_traffic(sim):
        x, y = pos
        wake_dx, wake_dy = screen_vector((vessel.heading_deg + 180.0) % 360.0, 34)
        pygame.draw.line(
            surface,
            FOAM,
            (x + wake_dx * 0.2, y + wake_dy * 0.2),
            (x + wake_dx, y + wake_dy),
            2,
        )

        hull_color = MOTORBOAT if vessel.kind == "motorboat" else HULL
        hull = traffic_hull_polygon((x, y), vessel.heading_deg)
        pygame.draw.polygon(surface, HULL_SHADOW, hull)
        inner = traffic_hull_polygon((x, y), vessel.heading_deg, length=42.0, width=13.0)
        pygame.draw.polygon(surface, hull_color, inner)

        if vessel.kind == "motorboat":
            cabin = traffic_hull_polygon((x, y), vessel.heading_deg, length=18.0, width=10.0)
            pygame.draw.polygon(surface, (250, 224, 184), cabin)
        else:
            mast = rotate_point(0.0, -2.0, vessel.heading_deg, (x, y))
            head = rotate_point(0.0, -28.0, vessel.heading_deg, (x, y))
            sail_side = -1.0 if signed_angle_diff(sim.wind.from_deg, vessel.heading_deg) > 0.0 else 1.0
            clew = rotate_point(sail_side * 18.0, 13.0, vessel.heading_deg, (x, y))
            pygame.draw.polygon(surface, TRAFFIC_SAIL, [mast, head, clew])
            pygame.draw.line(surface, BLACK, mast, head, 2)

        assessment = assessments.get(vessel.id)
        if assessment and assessment.warning:
            ring_color = DANGER if assessment.player_must_give_way else WARNING
            pygame.draw.circle(surface, ring_color, (round(x), round(y)), 34, 2)

        bob = sin(time_s * 2.0 + vessel.position_east * 0.01) * 2.0
        label = font.render(vessel.id, True, TEXT)
        surface.blit(label, (x - label.get_width() / 2, y + 24 + bob))


def draw_no_sail_zone(surface: pygame.Surface, sim: SailingSimulation, enabled: bool) -> None:
    if not enabled:
        return

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    points = no_sail_zone_points(sim.wind.from_deg)
    pygame.draw.polygon(overlay, (222, 50, 50, 60), points)
    pygame.draw.lines(overlay, (255, 116, 100, 140), False, points[1:], 2)
    surface.blit(overlay, (0, 0))


def draw_rule_penalty(surface: pygame.Surface, font: pygame.font.Font, sim: SailingSimulation) -> None:
    if sim.penalty_timer <= 0.0:
        return

    intensity = clamp(sim.penalty_timer / 1.6, 0.0, 1.0)
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((220, 40, 32, int(60 * intensity)))
    surface.blit(overlay, (0, 0))
    border = pygame.Rect(8, 8, WIDTH - 16, HEIGHT - 16)
    pygame.draw.rect(surface, DANGER, border, max(3, int(9 * intensity)), border_radius=8)
    text = font.render("RIGHT-OF-WAY VIOLATION", True, (255, 238, 230))
    surface.blit(text, (WIDTH / 2 - text.get_width() / 2, 82))


def stability_color(status: str) -> tuple[int, int, int]:
    if status in {"capsized", "danger"}:
        return DANGER
    if status == "warning":
        return WARNING
    return TEXT


def draw_boat(
    surface: pygame.Surface,
    sim: SailingSimulation,
    sail_visual: SailVisualState | None = None,
) -> None:
    boat = sim.boat
    center = BOAT_CENTER
    if sail_visual is None:
        sail_visual = SailVisualState.from_sim(sim)

    heel = clamp(abs(boat.heel_deg) / 85.0, 0.0, 1.0)
    hull_width = 36.0 * (1.0 - heel * 0.42)
    hull = boat_polygon(center, boat.heading_deg, width=hull_width)

    wake_length = 48 + boat.speed * 18
    wake_dx, wake_dy = screen_vector((boat.heading_deg + 180.0) % 360.0, wake_length)
    pygame.draw.line(
        surface,
        FOAM,
        (center[0] + wake_dx * 0.35, center[1] + wake_dy * 0.35),
        (center[0] + wake_dx, center[1] + wake_dy),
        max(1, int(boat.speed * 0.9)),
    )

    pygame.draw.polygon(surface, HULL_SHADOW, hull)
    inner = boat_polygon(center, boat.heading_deg, length=84.0, width=hull_width * 0.62)
    pygame.draw.polygon(surface, HULL, inner)

    sail = sail_geometry(sim, sail_visual)
    pygame.draw.polygon(surface, SAIL_SHADE, [sail.mast, sail.head, sail.belly, sail.clew])
    pygame.draw.aaline(surface, SAIL, sail.head, sail.belly)
    pygame.draw.aaline(surface, SAIL, sail.belly, sail.clew)
    pygame.draw.line(surface, BLACK, sail.mast, sail.head, 3)
    pygame.draw.line(surface, WOOD, sail.mast, sail.clew, 5)
    pygame.draw.aaline(surface, MUTED, sail.clew, sail.sheet_anchor)

    batten_1 = (
        sail.mast[0] * 0.68 + sail.belly[0] * 0.32,
        sail.mast[1] * 0.68 + sail.belly[1] * 0.32,
    )
    batten_2 = (
        sail.head[0] * 0.55 + sail.belly[0] * 0.45,
        sail.head[1] * 0.55 + sail.belly[1] * 0.45,
    )
    pygame.draw.aaline(surface, (224, 232, 218), batten_1, sail.belly)
    pygame.draw.aaline(surface, (224, 232, 218), batten_2, sail.belly)

    steering = tiller_rudder_geometry(sim)
    pygame.draw.line(
        surface,
        HULL_SHADOW,
        steering.rudder_pivot,
        steering.rudder_tip,
        6,
    )
    pygame.draw.line(
        surface,
        WOOD,
        steering.tiller_pivot,
        steering.tiller_handle,
        5,
    )
    pygame.draw.circle(surface, WOOD, (round(steering.tiller_pivot[0]), round(steering.tiller_pivot[1])), 4)
    pygame.draw.circle(surface, (188, 119, 62), (round(steering.tiller_handle[0]), round(steering.tiller_handle[1])), 4)

    if boat.capsized:
        pygame.draw.circle(surface, DANGER, center, 64, 5)


def info_lines(sim: SailingSimulation) -> list[tuple[str, tuple[int, int, int]]]:
    boat = sim.boat
    wind = sim.wind
    terms = sailing_terms(boat, wind)
    forces = calculate_sail_forces(boat, wind)
    stability = stability_state(boat)
    wind_to = (wind.from_deg + 180.0) % 360.0

    heel_color = stability_color(stability.status)

    traffic_line = "Traffic: clear"
    traffic_color = MUTED
    active_assessments = [
        assessment
        for assessment in sim.right_of_way
        if assessment.warning or assessment.distance < 120.0
    ]
    if active_assessments:
        nearest = min(active_assessments, key=lambda assessment: assessment.distance)
        if nearest.player_must_give_way:
            traffic_line = f"Traffic: give way to {nearest.vessel_id}"
            traffic_color = DANGER if nearest.violation else WARNING
        else:
            traffic_line = f"Traffic: stand on vs {nearest.vessel_id}"
            traffic_color = TEXT

    return [
        (f"Tack: {terms['tack']}", TEXT),
        (f"Beam: {terms['beam']}", TEXT),
        (f"Point: {terms['point_of_sail']}", TEXT),
        (f"Speed: {boat.speed:0.1f} kt", TEXT),
        (traffic_line, traffic_color),
        (f"Wind: {wind.speed:0.1f} kt from {wind.from_deg:03.0f}", TEXT),
        (f"Wind arrow points to {wind_to:03.0f}", MUTED),
        (f"Main sheet: {sheet_percent_in(boat.sheet)}% in", TEXT),
        (f"Tiller: {boat.tiller:+0.2f}", TEXT),
        (f"Tilt: {stability.heel_abs_deg:0.0f}/{stability.heel_limit_deg:0.0f} deg", heel_color),
        (f"Capsize risk: {stability.risk * 100:0.0f}%", heel_color),
        (f"Sail efficiency: {sail_efficiency(sim) * 100:0.0f}%", MUTED),
        (f"Recent: {sim.actions.latest}", WARNING if boat.capsized else TEXT),
    ]


def draw_tilt_meter(surface: pygame.Surface, rect: pygame.Rect, sim: SailingSimulation) -> None:
    stability = stability_state(sim.boat)
    meter = pygame.Rect(rect.x + 16, rect.bottom - 38, rect.width - 32, 14)
    pygame.draw.rect(surface, (8, 25, 34), meter, border_radius=7)

    fill_width = int(meter.width * clamp(stability.risk, 0.0, 1.0))
    if fill_width > 0:
        fill_rect = pygame.Rect(meter.x, meter.y, fill_width, meter.height)
        pygame.draw.rect(surface, stability_color(stability.status), fill_rect, border_radius=7)

    warning_x = meter.x + int(meter.width * WARNING_HEEL_DEG / CAPSIZE_HEEL_DEG)
    danger_x = meter.x + int(meter.width * DANGER_HEEL_DEG / CAPSIZE_HEEL_DEG)
    pygame.draw.line(surface, WARNING, (warning_x, meter.y - 3), (warning_x, meter.bottom + 3), 2)
    pygame.draw.line(surface, DANGER, (danger_x, meter.y - 3), (danger_x, meter.bottom + 3), 2)
    pygame.draw.rect(surface, PANEL_BORDER, meter, 1, border_radius=7)


def draw_info_box(
    surface: pygame.Surface, font: pygame.font.Font, small: pygame.font.Font, sim: SailingSimulation
) -> None:
    rect = pygame.Rect(24, 24, 305, 430)
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((*PANEL, 224))
    surface.blit(panel, rect)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)

    y = rect.y + 16
    for text, color in info_lines(sim):
        rendered = font.render(text, True, color)
        surface.blit(rendered, (rect.x + 16, y))
        y += 25

    if len(sim.actions.messages) > 1:
        y += 4
        for message in sim.actions.messages[-3:-1]:
            rendered = small.render(message, True, MUTED)
            surface.blit(rendered, (rect.x + 16, y))
            y += 19

    draw_tilt_meter(surface, rect, sim)


def draw_wind_indicator(
    surface: pygame.Surface, font: pygame.font.Font, sim: SailingSimulation
) -> None:
    rect = pygame.Rect(WIDTH - 220, 26, 190, 110)
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((6, 19, 27, 190))
    surface.blit(panel, rect)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)
    label = font.render("Wind", True, TEXT)
    surface.blit(label, (rect.x + 16, rect.y + 12))

    wind_to_deg = (sim.wind.from_deg + 180.0) % 360.0
    start = (rect.centerx - 34, rect.centery + 18)
    draw_arrow(surface, start, wind_to_deg, 72, WARNING, 5)
    speed = font.render(f"{sim.wind.speed:0.1f} kt", True, TEXT)
    surface.blit(speed, (rect.x + 16, rect.y + 72))


def draw_no_sail_toggle(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small: pygame.font.Font,
    toggle: NoSailZoneToggle,
) -> None:
    rect = toggle.rect
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((6, 19, 27, 190))
    surface.blit(panel, rect)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)

    label = font.render("No-sail zone", True, TEXT)
    hint = small.render("N", True, MUTED)
    surface.blit(label, (rect.x + 14, rect.y + 8))
    surface.blit(hint, (rect.x + 14, rect.y + 30))

    switch = pygame.Rect(rect.right - 70, rect.y + 16, 48, 22)
    pygame.draw.rect(surface, (12, 36, 46), switch, border_radius=11)
    knob_x = switch.right - 18 if toggle.enabled else switch.left + 4
    fill = DANGER if toggle.enabled else MUTED
    pygame.draw.rect(surface, fill, switch, 2, border_radius=11)
    pygame.draw.circle(surface, fill, (knob_x + 7, switch.centery), 7)


def draw_sail_efficiency_panel(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small: pygame.font.Font,
    sim: SailingSimulation,
) -> None:
    rect = pygame.Rect(WIDTH - 230, 218, 204, 176)
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((6, 19, 27, 205))
    surface.blit(panel, rect)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)

    forces = calculate_sail_forces(sim.boat, sim.wind)
    efficiency = sail_efficiency(sim)
    color = efficiency_color(efficiency)
    title = font.render("Sail efficiency", True, TEXT)
    value = font.render(f"{efficiency * 100:0.0f}%", True, color)
    surface.blit(title, (rect.x + 14, rect.y + 12))
    surface.blit(value, (rect.right - value.get_width() - 16, rect.y + 12))

    efficiency_track = pygame.Rect(rect.x + 16, rect.y + 48, rect.width - 32, 14)
    pygame.draw.rect(surface, (8, 25, 34), efficiency_track, border_radius=7)
    fill_width = int(round(efficiency_track.width * efficiency))
    if fill_width > 0:
        fill = pygame.Rect(efficiency_track.x, efficiency_track.y, fill_width, efficiency_track.height)
        pygame.draw.rect(surface, color, fill, border_radius=7)
    pygame.draw.rect(surface, PANEL_BORDER, efficiency_track, 1, border_radius=7)

    point_label = small.render("iron  close  beam  broad  run", True, MUTED)
    surface.blit(point_label, (rect.x + 14, rect.y + 76))

    point_track = pygame.Rect(rect.x + 16, rect.y + 100, rect.width - 32, 16)
    for _label, start_deg, end_deg, band_color in POINT_OF_SAIL_BANDS:
        start_x = point_of_sail_marker_x(start_deg, point_track)
        end_x = point_of_sail_marker_x(end_deg, point_track)
        band = pygame.Rect(start_x, point_track.y, max(1, end_x - start_x), point_track.height)
        pygame.draw.rect(surface, band_color, band, border_radius=4)
    pygame.draw.rect(surface, PANEL_BORDER, point_track, 1, border_radius=4)

    marker_x = point_of_sail_marker_x(forces.relative_wind_deg, point_track)
    pygame.draw.polygon(
        surface,
        TEXT,
        [
            (marker_x, point_track.y - 7),
            (marker_x - 6, point_track.y - 1),
            (marker_x + 6, point_track.y - 1),
        ],
    )
    pygame.draw.line(surface, BLACK, (marker_x, point_track.y - 1), (marker_x, point_track.bottom + 5), 2)

    trim = small.render(f"trim {forces.trim_efficiency * 100:0.0f}%", True, MUTED)
    point = small.render(f"angle {forces.angle_efficiency * 100:0.0f}%", True, MUTED)
    surface.blit(trim, (rect.x + 16, rect.y + 136))
    surface.blit(point, (rect.right - point.get_width() - 16, rect.y + 136))


def draw_main_sheet_lever(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small: pygame.font.Font,
    sim: SailingSimulation,
    lever: MainSheetLever,
) -> None:
    rect = lever.rect
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((6, 19, 27, 205))
    surface.blit(panel, rect)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)

    percent = sheet_percent_in(sim.boat.sheet)
    title = font.render("Main sheet", True, TEXT)
    value = font.render(f"{percent}%", True, WARNING)
    surface.blit(title, (rect.x + 16, rect.y + 14))
    surface.blit(value, (rect.x + 16, rect.y + 42))

    status = "trimmed in" if percent >= 52 else "eased"
    status_text = small.render(status, True, MUTED)
    surface.blit(status_text, (rect.x + 16, rect.y + 70))

    track = lever.track_rect
    pygame.draw.rect(surface, (8, 25, 34), track.inflate(8, 8), border_radius=10)
    pygame.draw.rect(surface, PANEL_BORDER, track, border_radius=8)

    knob = lever.knob_rect(sim)
    fill_top = sheet_to_lever_y(sim.boat.sheet, track)
    active_fill = pygame.Rect(track.x, int(round(fill_top)), track.width, track.bottom - int(round(fill_top)))
    if active_fill.height > 0:
        pygame.draw.rect(surface, WARNING, active_fill, border_radius=8)

    pygame.draw.line(surface, MUTED, (track.left - 20, track.top), (track.left - 4, track.top), 2)
    pygame.draw.line(
        surface,
        MUTED,
        (track.left - 20, track.bottom),
        (track.left - 4, track.bottom),
        2,
    )
    top_label = small.render("100", True, MUTED)
    bottom_label = small.render("0", True, MUTED)
    surface.blit(top_label, (track.left - 48, track.top - 8))
    surface.blit(bottom_label, (track.left - 34, track.bottom - 9))

    knob_color = (255, 210, 104) if lever.dragging else (244, 188, 75)
    pygame.draw.rect(surface, BLACK, knob.move(2, 3), border_radius=6)
    pygame.draw.rect(surface, knob_color, knob, border_radius=6)
    pygame.draw.rect(surface, (255, 238, 170), knob.inflate(-10, -12), border_radius=3)


def key_is_down(keys: Iterable[bool], key: int) -> bool:
    try:
        return bool(keys[key])  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return False


def handle_keys(sim: SailingSimulation, keys: Iterable[bool], dt: float) -> bool:
    quit_requested = False
    if key_is_down(keys, pygame.K_ESCAPE):
        quit_requested = True

    left = key_is_down(keys, pygame.K_LEFT) or key_is_down(keys, pygame.K_a)
    right = key_is_down(keys, pygame.K_RIGHT) or key_is_down(keys, pygame.K_d)
    if left and not right:
        sim.adjust_tiller(-1.0, dt)
    elif right and not left:
        sim.adjust_tiller(1.0, dt)
    else:
        sim.center_tiller(dt)

    if key_is_down(keys, pygame.K_SPACE):
        sim.boat.tiller = 0.0

    trim_in = key_is_down(keys, pygame.K_UP) or key_is_down(keys, pygame.K_w)
    ease = key_is_down(keys, pygame.K_DOWN) or key_is_down(keys, pygame.K_s)
    if trim_in and not ease:
        sim.adjust_sheet(-1.0, dt)
    elif ease and not trim_in:
        sim.adjust_sheet(1.0, dt)

    if key_is_down(keys, pygame.K_r):
        sim.right_boat()

    return quit_requested


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Sailing Simulator")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18)
    small = pygame.font.SysFont("arial", 15)
    sim = SailingSimulation(traffic=default_traffic())
    sail_visual = SailVisualState.from_sim(sim)
    sheet_lever = MainSheetLever.default()
    no_sail_toggle = NoSailZoneToggle.default()
    course_markers = default_course_markers()
    animation_time_s = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                no_sail_toggle.enabled = not no_sail_toggle.enabled
            else:
                if not no_sail_toggle.handle_event(event):
                    sheet_lever.handle_event(sim, event)

        keys = pygame.key.get_pressed()
        if handle_keys(sim, keys, dt):
            running = False

        sim_dt = scaled_game_dt(dt)
        animation_time_s += sim_dt
        sim.step(sim_dt)
        sail_visual.update(sim, sim_dt)
        draw_water(screen, sim.wind.from_deg, animation_time_s)
        draw_course_markers(screen, small, sim, course_markers, animation_time_s)
        draw_traffic_vessels(screen, small, sim, animation_time_s)
        draw_no_sail_zone(screen, sim, no_sail_toggle.enabled)
        draw_boat(screen, sim, sail_visual)
        draw_info_box(screen, font, small, sim)
        draw_wind_indicator(screen, font, sim)
        draw_no_sail_toggle(screen, font, small, no_sail_toggle)
        draw_sail_efficiency_panel(screen, font, small, sim)
        draw_main_sheet_lever(screen, font, small, sim, sheet_lever)
        draw_rule_penalty(screen, font, sim)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
