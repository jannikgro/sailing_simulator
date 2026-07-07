"""PyGame interface for the sailing simulator.

The app is a small state machine: a level-select menu, the level itself, and a
level-complete screen. Failing a level (capsize, rock, collision) shows a
three-second countdown and respawns the boat at the last rounded buoy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, ceil, cos, hypot, radians, sin
from typing import Iterable

import pygame

from .levels import LEVELS, Level, Objective, build_simulation
from .physics import (
    MAX_SHEET_ANGLE_DEG,
    CAPSIZE_HEEL_DEG,
    DANGER_HEEL_DEG,
    NO_GO_ZONE_DEG,
    SailingSimulation,
    TrafficVessel,
    WARNING_HEEL_DEG,
    bearing_between,
    calculate_sail_forces,
    clamp,
    nav_vector,
    normalize_degrees,
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
BUOY_DONE = (116, 132, 130)
MOTORBOAT = (232, 128, 64)
TRAFFIC_SAIL = (226, 232, 218)
ACCENT = (120, 214, 196)
BOOM_LENGTH = 68.0
SAIL_HEAD_FORWARD = 58.0
MIN_BOOM_ANGLE_DEG = 8.0
GAME_SPEED = 0.72
RESPAWN_SECONDS = 3.0
BRIEFING_SECONDS = 10.0
POINT_OF_SAIL_BANDS = [
    ("irons", 0.0, NO_GO_ZONE_DEG, DANGER),
    ("close", NO_GO_ZONE_DEG, 60.0, WARNING),
    ("beam", 60.0, 112.0, (78, 198, 133)),
    ("broad", 112.0, 155.0, BUOY_YELLOW),
    ("run", 155.0, 180.0, MUTED),
]


# --- level session -----------------------------------------------------------


def sailable_heading(wind_from_deg: float, desired_heading_deg: float) -> float:
    """Nudge a heading out of the no-go zone so a respawned boat can sail."""

    relative = signed_angle_diff(wind_from_deg, desired_heading_deg)
    if abs(relative) >= NO_GO_ZONE_DEG + 10.0:
        return normalize_degrees(desired_heading_deg)
    side = 1.0 if relative >= 0.0 else -1.0
    return normalize_degrees(wind_from_deg - side * (NO_GO_ZONE_DEG + 15.0))


@dataclass
class LevelSession:
    """Progress through one level: objectives, checkpoints, fail-and-respawn."""

    level: Level
    sim: SailingSimulation
    target_index: int = 0
    elapsed: float = 0.0
    respawn_timer: float = 0.0
    failure_reason: str = ""
    completed: bool = False
    checkpoint: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def begin(cls, level: Level) -> "LevelSession":
        return cls(level=level, sim=build_simulation(level))

    @property
    def failed(self) -> bool:
        return self.respawn_timer > 0.0

    def current_objective(self) -> Objective | None:
        if self.target_index < len(self.level.objectives):
            return self.level.objectives[self.target_index]
        return None

    def update(self, dt: float) -> None:
        self.sim.step(dt)
        if self.completed:
            return
        if self.failed:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0.0:
                self.respawn_timer = 0.0
                self._respawn()
            return

        self.elapsed += dt
        if self.sim.wrecked:
            self.failure_reason = self._describe_failure()
            self.respawn_timer = RESPAWN_SECONDS
            return

        objective = self.current_objective()
        if objective is None:
            return
        boat = self.sim.boat
        distance = hypot(
            objective.east - boat.position_east, objective.north - boat.position_north
        )
        if distance < objective.radius:
            self.checkpoint = (objective.east, objective.north)
            self.sim.actions.push(f"rounded {objective.label}")
            self.target_index += 1
            if self.target_index >= len(self.level.objectives):
                self.completed = True

    def _describe_failure(self) -> str:
        if self.sim.boat.capsized:
            return "Capsized!"
        if self.sim.grounded:
            return "You hit a rock!"
        if self.sim.collided_vessel_id is not None:
            return f"Collision with {self.sim.collided_vessel_id}!"
        return "Wrecked!"

    def _respawn(self) -> None:
        objective = self.current_objective()
        if objective is not None:
            desired = bearing_between(
                self.checkpoint[0], self.checkpoint[1], objective.east, objective.north
            )
        else:
            desired = self.level.start_heading_deg
        heading = sailable_heading(self.sim.wind.from_deg, desired)
        self.failure_reason = ""
        self.sim.respawn_at(self.checkpoint[0], self.checkpoint[1], heading)


# --- menu --------------------------------------------------------------------


def menu_row_rect(index: int) -> pygame.Rect:
    return pygame.Rect(64, 148 + index * 64, 440, 56)


@dataclass
class MenuState:
    selected: int = 0
    completed: set[int] = field(default_factory=set)

    def move(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(LEVELS)

    def handle_event(self, event: pygame.event.Event) -> str | None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.move(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.move(1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return "start"
            elif event.key == pygame.K_ESCAPE:
                return "quit"
            elif pygame.K_1 <= event.key <= pygame.K_9:
                index = event.key - pygame.K_1
                if index < len(LEVELS):
                    self.selected = index
                    return "start"
        elif event.type == pygame.MOUSEMOTION:
            for index in range(len(LEVELS)):
                if menu_row_rect(index).collidepoint(event.pos):
                    self.selected = index
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index in range(len(LEVELS)):
                if menu_row_rect(index).collidepoint(event.pos):
                    self.selected = index
                    return "start"
        return None


# --- widgets -----------------------------------------------------------------


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


def on_screen(pos: tuple[float, float], margin: float = 80.0) -> bool:
    return -margin <= pos[0] <= WIDTH + margin and -margin <= pos[1] <= HEIGHT + margin


def visible_traffic(
    sim: SailingSimulation,
    margin: float = 80.0,
) -> list[tuple[TrafficVessel, tuple[float, float]]]:
    visible = []
    for vessel in sim.traffic:
        pos = world_to_screen(sim, vessel.position_east, vessel.position_north)
        if on_screen(pos, margin):
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


def rock_polygon(x: float, y: float, radius: float) -> list[tuple[float, float]]:
    scale = radius / 24.0
    points = [(-24, 10), (-12, -12), (9, -18), (25, 5), (12, 18), (-17, 17)]
    return [(x + px * scale, y + py * scale) for px, py in points]


def draw_rocks(surface: pygame.Surface, sim: SailingSimulation) -> None:
    for rock in sim.rocks:
        pos = world_to_screen(sim, rock.east, rock.north)
        if not on_screen(pos):
            continue
        x, y = pos
        scale = rock.radius / 24.0
        shadow = [
            (x - 22 * scale, y + 14 * scale),
            (x + 24 * scale, y + 15 * scale),
            (x + 12 * scale, y + 23 * scale),
            (x - 18 * scale, y + 21 * scale),
        ]
        pygame.draw.polygon(surface, (13, 48, 63), shadow)
        pygame.draw.polygon(surface, ROCK, rock_polygon(x, y, rock.radius))
        pygame.draw.line(
            surface,
            ROCK_LIGHT,
            (x - 11 * scale, y - 8 * scale),
            (x + 8 * scale, y - 13 * scale),
            2,
        )
        pygame.draw.line(
            surface,
            (65, 59, 54),
            (x - 15 * scale, y + 12 * scale),
            (x + 15 * scale, y + 10 * scale),
            2,
        )


def objective_color(index: int, total: int) -> tuple[int, int, int]:
    if index == total - 1:
        return BUOY_YELLOW
    return BUOY_RED if index % 2 == 0 else BUOY_GREEN


def draw_objectives(
    surface: pygame.Surface,
    font: pygame.font.Font,
    session: LevelSession,
    time_s: float,
) -> None:
    sim = session.sim
    objectives = session.level.objectives
    remaining = [
        world_to_screen(sim, objective.east, objective.north)
        for objective in objectives[session.target_index :]
    ]
    if remaining and not session.completed:
        pygame.draw.lines(
            surface, (117, 180, 175), False, [BOAT_CENTER] + remaining, 2
        )

    for index, objective in enumerate(objectives):
        pos = world_to_screen(sim, objective.east, objective.north)
        if not on_screen(pos):
            continue
        bob = sin(time_s * 2.0 + objective.east * 0.017 + objective.north * 0.006) * 3.0
        x, y = pos[0], pos[1] + bob
        rounded = index < session.target_index
        color = BUOY_DONE if rounded else objective_color(index, len(objectives))

        if index == session.target_index and not session.completed:
            pulse = 26 + sin(time_s * 4.0) * 4.0
            pygame.draw.circle(surface, WARNING, (round(x), round(y)), int(pulse), 2)
            pygame.draw.circle(
                surface,
                (255, 255, 255),
                (round(x), round(y)),
                int(objective.radius),
                1,
            )

        pygame.draw.line(surface, BLACK, (x, y + 22), (x, y - 18), 3)
        pygame.draw.circle(surface, color, (round(x), round(y)), 13)
        pygame.draw.circle(surface, (246, 246, 230), (round(x), round(y - 3)), 5)
        pygame.draw.polygon(
            surface,
            color,
            [(x, y - 24), (x + 9, y - 10), (x - 9, y - 10)],
        )
        label = font.render(objective.label, True, TEXT if not rounded else MUTED)
        surface.blit(label, (x - label.get_width() / 2, y + 18))


def draw_target_arrow(
    surface: pygame.Surface, font: pygame.font.Font, session: LevelSession
) -> None:
    objective = session.current_objective()
    if objective is None:
        return
    boat = session.sim.boat
    distance = hypot(
        objective.east - boat.position_east, objective.north - boat.position_north
    )
    if distance < 150.0:
        return
    bearing = bearing_between(
        boat.position_east, boat.position_north, objective.east, objective.north
    )
    dx, dy = screen_vector(bearing, 96.0)
    start = (BOAT_CENTER[0] + dx, BOAT_CENTER[1] + dy)
    draw_arrow(surface, start, bearing, 30.0, ACCENT, 3)
    label = font.render(f"{objective.label}  {distance:0.0f}", True, ACCENT)
    surface.blit(label, (start[0] - label.get_width() / 2, start[1] + 14))


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


# --- level overlays ----------------------------------------------------------


def format_time(seconds: float) -> str:
    minutes = int(seconds) // 60
    return f"{minutes:02d}:{int(seconds) % 60:02d}"


def draw_hud_banner(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small: pygame.font.Font,
    session: LevelSession,
) -> None:
    rect = pygame.Rect(WIDTH // 2 - 230, 14, 460, 58)
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((6, 19, 27, 205))
    surface.blit(panel, rect)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)

    level = session.level
    title = font.render(f"Level {level.number} - {level.title}", True, TEXT)
    surface.blit(title, (rect.centerx - title.get_width() / 2, rect.y + 8))

    objective = session.current_objective()
    if session.completed:
        progress = "course complete"
    elif objective is not None:
        progress = (
            f"next: {objective.label}"
            f"  ({session.target_index + 1}/{len(level.objectives)})"
        )
    else:
        progress = ""
    detail = small.render(
        f"{progress}   time {format_time(session.elapsed)}"
        f"   penalties {session.sim.penalty_count}",
        True,
        MUTED,
    )
    surface.blit(detail, (rect.centerx - detail.get_width() / 2, rect.y + 34))


def draw_briefing(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small: pygame.font.Font,
    session: LevelSession,
) -> None:
    if session.elapsed > BRIEFING_SECONDS or session.failed or session.completed:
        return
    lines = session.level.briefing
    rect = pygame.Rect(24, HEIGHT - 60 - len(lines) * 20 - 44, 560, len(lines) * 20 + 44)
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((6, 19, 27, 215))
    surface.blit(panel, rect)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=6)

    header = font.render(session.level.skills.capitalize(), True, ACCENT)
    surface.blit(header, (rect.x + 16, rect.y + 10))
    y = rect.y + 38
    for line in lines:
        rendered = small.render(line, True, TEXT)
        surface.blit(rendered, (rect.x + 16, y))
        y += 20


def draw_respawn_countdown(
    surface: pygame.Surface,
    big: pygame.font.Font,
    font: pygame.font.Font,
    session: LevelSession,
) -> None:
    if not session.failed:
        return
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((8, 14, 20, 130))
    surface.blit(overlay, (0, 0))

    reason = font.render(session.failure_reason, True, DANGER)
    surface.blit(reason, (WIDTH / 2 - reason.get_width() / 2, HEIGHT / 2 - 130))

    count = big.render(str(max(1, ceil(session.respawn_timer))), True, TEXT)
    surface.blit(count, (WIDTH / 2 - count.get_width() / 2, HEIGHT / 2 - 70))

    hint = font.render("respawning at the last buoy...", True, MUTED)
    surface.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 40))


def draw_level_complete(
    surface: pygame.Surface,
    big: pygame.font.Font,
    font: pygame.font.Font,
    session: LevelSession,
) -> None:
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((8, 14, 20, 130))
    surface.blit(overlay, (0, 0))

    rect = pygame.Rect(WIDTH // 2 - 260, HEIGHT // 2 - 130, 520, 260)
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((6, 19, 27, 235))
    surface.blit(panel, rect)
    pygame.draw.rect(surface, ACCENT, rect, 2, border_radius=8)

    title = big.render("Level Complete!", True, ACCENT)
    surface.blit(title, (rect.centerx - title.get_width() / 2, rect.y + 26))

    stats = font.render(
        f"time {format_time(session.elapsed)}   penalties {session.sim.penalty_count}",
        True,
        TEXT,
    )
    surface.blit(stats, (rect.centerx - stats.get_width() / 2, rect.y + 110))

    last_level = session.level.number >= len(LEVELS)
    next_hint = (
        "You finished the sailing school!"
        if last_level
        else "ENTER: next level"
    )
    hint_1 = font.render(next_hint, True, TEXT if last_level else MUTED)
    hint_2 = font.render("ESC: back to menu", True, MUTED)
    surface.blit(hint_1, (rect.centerx - hint_1.get_width() / 2, rect.y + 160))
    surface.blit(hint_2, (rect.centerx - hint_2.get_width() / 2, rect.y + 192))


# --- menu drawing -------------------------------------------------------------


def draw_menu(
    surface: pygame.Surface,
    big: pygame.font.Font,
    font: pygame.font.Font,
    small: pygame.font.Font,
    menu: MenuState,
    time_s: float,
) -> None:
    draw_water(surface, 90.0, time_s)

    title = big.render("SAILING SCHOOL", True, TEXT)
    surface.blit(title, (64, 48))
    subtitle = font.render(
        "learn to sail, from the first reach to regatta day", True, MUTED
    )
    surface.blit(subtitle, (66, 100))

    for index, level in enumerate(LEVELS):
        rect = menu_row_rect(index)
        selected = index == menu.selected
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((10, 32, 44, 235) if selected else (6, 19, 27, 195))
        surface.blit(panel, rect)
        pygame.draw.rect(
            surface, ACCENT if selected else PANEL_BORDER, rect, 2, border_radius=6
        )

        name = font.render(f"{level.number}. {level.title}", True, TEXT)
        skills = small.render(level.skills, True, ACCENT if selected else MUTED)
        surface.blit(name, (rect.x + 16, rect.y + 8))
        surface.blit(skills, (rect.x + 16, rect.y + 32))

        if level.number in menu.completed:
            check = font.render("done", True, BUOY_GREEN)
            surface.blit(check, (rect.right - check.get_width() - 16, rect.y + 16))

    level = LEVELS[menu.selected]
    info = pygame.Rect(540, 148, 420, 64 * len(LEVELS) - 8)
    panel = pygame.Surface(info.size, pygame.SRCALPHA)
    panel.fill((6, 19, 27, 215))
    surface.blit(panel, info)
    pygame.draw.rect(surface, PANEL_BORDER, info, 2, border_radius=6)

    header = font.render(level.title, True, ACCENT)
    surface.blit(header, (info.x + 18, info.y + 14))
    y = info.y + 48
    for line in level.briefing:
        rendered = small.render(line, True, TEXT)
        surface.blit(rendered, (info.x + 18, y))
        y += 21

    y += 14
    wind = level.wind
    facts = [
        f"wind: {wind.speed:0.0f} kt from {wind.from_deg:03.0f}",
        f"buoys: {len(level.objectives)}",
        f"rocks: {'yes' if level.rocks else 'no'}",
        f"traffic: {len(level.traffic) or 'no'}",
    ]
    for fact in facts:
        rendered = small.render(fact, True, MUTED)
        surface.blit(rendered, (info.x + 18, y))
        y += 21

    footer = small.render(
        "UP/DOWN select   ENTER sail   ESC quit", True, MUTED
    )
    surface.blit(footer, (64, HEIGHT - 40))


# --- input and main loop ------------------------------------------------------


def key_is_down(keys: Iterable[bool], key: int) -> bool:
    try:
        return bool(keys[key])  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return False


def handle_keys(sim: SailingSimulation, keys: Iterable[bool], dt: float) -> None:
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


def draw_scene(
    screen: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    session: LevelSession,
    sail_visual: SailVisualState,
    sheet_lever: MainSheetLever,
    no_sail_toggle: NoSailZoneToggle,
    time_s: float,
) -> None:
    sim = session.sim
    draw_water(screen, sim.wind.from_deg, time_s)
    draw_rocks(screen, sim)
    draw_objectives(screen, fonts["small"], session, time_s)
    draw_traffic_vessels(screen, fonts["small"], sim, time_s)
    draw_no_sail_zone(screen, sim, no_sail_toggle.enabled)
    draw_boat(screen, sim, sail_visual)
    draw_target_arrow(screen, fonts["small"], session)
    draw_info_box(screen, fonts["font"], fonts["small"], sim)
    draw_wind_indicator(screen, fonts["font"], sim)
    draw_no_sail_toggle(screen, fonts["font"], fonts["small"], no_sail_toggle)
    draw_sail_efficiency_panel(screen, fonts["font"], fonts["small"], sim)
    draw_main_sheet_lever(screen, fonts["font"], fonts["small"], sim, sheet_lever)
    draw_hud_banner(screen, fonts["font"], fonts["small"], session)
    draw_briefing(screen, fonts["font"], fonts["small"], session)
    draw_rule_penalty(screen, fonts["font"], sim)
    draw_respawn_countdown(screen, fonts["big"], fonts["font"], session)


def main(max_frames: int | None = None) -> None:
    pygame.init()
    pygame.display.set_caption("Sailing Simulator")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    fonts = {
        "font": pygame.font.SysFont("arial", 18),
        "small": pygame.font.SysFont("arial", 15),
        "big": pygame.font.SysFont("arial", 56, bold=True),
    }
    menu = MenuState()
    session: LevelSession | None = None
    sail_visual = SailVisualState()
    sheet_lever = MainSheetLever.default()
    no_sail_toggle = NoSailZoneToggle.default()
    state = "menu"
    animation_time_s = 0.0
    frames = 0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        if state == "menu":
            for event in events:
                action = menu.handle_event(event)
                if action == "quit":
                    running = False
                elif action == "start":
                    session = LevelSession.begin(LEVELS[menu.selected])
                    sail_visual = SailVisualState.from_sim(session.sim)
                    sheet_lever = MainSheetLever.default()
                    state = "playing"
            animation_time_s += dt

        elif state == "playing" and session is not None:
            for event in events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    state = "menu"
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    no_sail_toggle.enabled = not no_sail_toggle.enabled
                elif not no_sail_toggle.handle_event(event):
                    if not session.failed and not session.completed:
                        sheet_lever.handle_event(session.sim, event)

            if state == "playing":
                keys = pygame.key.get_pressed()
                if not session.failed and not session.completed:
                    handle_keys(session.sim, keys, dt)

                sim_dt = scaled_game_dt(dt)
                animation_time_s += sim_dt
                session.update(sim_dt)
                sail_visual.update(session.sim, sim_dt)
                if session.completed:
                    menu.completed.add(session.level.number)
                    state = "complete"

        elif state == "complete" and session is not None:
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        state = "menu"
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        next_index = session.level.number  # numbers are 1-based
                        if next_index < len(LEVELS):
                            menu.selected = next_index
                            session = LevelSession.begin(LEVELS[next_index])
                            sail_visual = SailVisualState.from_sim(session.sim)
                            sheet_lever = MainSheetLever.default()
                            state = "playing"
                        else:
                            state = "menu"
            animation_time_s += dt

        if state == "menu" or session is None:
            draw_menu(
                screen, fonts["big"], fonts["font"], fonts["small"], menu, animation_time_s
            )
        else:
            draw_scene(
                screen,
                fonts,
                session,
                sail_visual,
                sheet_lever,
                no_sail_toggle,
                animation_time_s,
            )
            if state == "complete":
                draw_level_complete(screen, fonts["big"], fonts["font"], session)

        pygame.display.flip()
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
