import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from sailing_simulator.game import (
    BOAT_CENTER,
    HEIGHT,
    WIDTH,
    DEFAULT_COURSE_WIND_FROM_DEG,
    GAME_SPEED,
    MainSheetLever,
    NoSailZoneToggle,
    POINT_OF_SAIL_BANDS,
    SailVisualState,
    boat_polygon,
    course_point,
    default_course_markers,
    desired_boom_angle,
    draw_boat,
    draw_course_markers,
    draw_info_box,
    draw_main_sheet_lever,
    draw_no_sail_toggle,
    draw_no_sail_zone,
    draw_rule_penalty,
    draw_sail_efficiency_panel,
    draw_tilt_meter,
    draw_traffic_vessels,
    draw_water,
    draw_wind_indicator,
    efficiency_color,
    handle_keys,
    info_lines,
    no_sail_zone_points,
    point_of_sail_marker_x,
    sail_geometry,
    sail_efficiency,
    scaled_game_dt,
    sheet_from_lever_y,
    sheet_percent_in,
    sheet_to_lever_y,
    stability_color,
    traffic_hull_polygon,
    tiller_rudder_geometry,
    visible_traffic,
    visible_course_markers,
    world_to_screen,
)
from sailing_simulator.physics import (
    BoatState,
    SailingSimulation,
    TrafficVessel,
    WindState,
    bearing_between,
    signed_angle_diff,
)


def test_boat_polygon_keeps_boat_centered() -> None:
    points = boat_polygon(BOAT_CENTER, heading_deg=30)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    assert min(xs) < BOAT_CENTER[0] < max(xs)
    assert min(ys) < BOAT_CENTER[1] < max(ys)


def test_info_lines_include_required_fields() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0),
        wind=WindState(from_deg=90, speed=12),
    )

    lines = [text for text, _color in info_lines(sim)]

    assert any(text.startswith("Tack:") for text in lines)
    assert any(text.startswith("Beam:") for text in lines)
    assert any(text.startswith("Recent:") for text in lines)
    assert any(text.startswith("Tilt:") for text in lines)
    assert any(text.startswith("Capsize risk:") for text in lines)
    assert any(text.startswith("Sail efficiency:") for text in lines)
    assert any(text.startswith("Traffic:") for text in lines)


def test_stability_color_escalates_with_status() -> None:
    assert stability_color("stable") != stability_color("warning")
    assert stability_color("danger") == stability_color("capsized")


def test_scaled_game_dt_slows_simulation_without_reversing_time() -> None:
    assert 0.0 < GAME_SPEED < 1.0
    assert scaled_game_dt(1.0) == GAME_SPEED
    assert scaled_game_dt(-1.0) == 0.0


def test_sail_efficiency_combines_trim_and_angle_efficiency() -> None:
    good = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.38),
        wind=WindState(from_deg=90, speed=12),
    )
    irons = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.0),
        wind=WindState(from_deg=10, speed=12),
    )

    assert sail_efficiency(good) > 0.9
    assert sail_efficiency(irons) == 0.0


def test_efficiency_color_escalates_with_efficiency() -> None:
    assert efficiency_color(0.1) != efficiency_color(0.5)
    assert efficiency_color(0.8) != efficiency_color(0.5)


def test_point_of_sail_marker_and_bands_include_close_and_beam() -> None:
    track = pygame.Rect(10, 10, 180, 16)
    labels = [band[0] for band in POINT_OF_SAIL_BANDS]

    assert "close" in labels
    assert "beam" in labels
    assert point_of_sail_marker_x(0, track) == track.x
    assert point_of_sail_marker_x(180, track) == track.right
    assert point_of_sail_marker_x(-90, track) == track.x + 90


def test_sheet_lever_maps_top_to_trimmed_in_and_bottom_to_eased() -> None:
    lever = MainSheetLever.default()
    track = lever.track_rect

    assert sheet_from_lever_y(track.top, track) == 0.0
    assert sheet_from_lever_y(track.bottom, track) == 1.0
    assert sheet_to_lever_y(0.0, track) == track.top
    assert sheet_to_lever_y(1.0, track) == track.bottom
    assert sheet_percent_in(0.0) == 100
    assert sheet_percent_in(1.0) == 0


def test_sheet_lever_mouse_drag_updates_main_sheet() -> None:
    pygame.init()
    sim = SailingSimulation(boat=BoatState(sheet=0.55))
    lever = MainSheetLever.default()
    track = lever.track_rect

    down = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN,
        {"button": 1, "pos": (track.centerx, track.top)},
    )
    motion = pygame.event.Event(
        pygame.MOUSEMOTION,
        {"pos": (track.centerx, track.bottom)},
    )
    up = pygame.event.Event(
        pygame.MOUSEBUTTONUP,
        {"button": 1, "pos": (track.centerx, track.bottom)},
    )

    assert lever.handle_event(sim, down)
    assert lever.dragging
    assert sim.boat.sheet == 0.0
    assert sim.actions.latest == "you trimmed in the main sheet"

    assert lever.handle_event(sim, motion)
    assert sim.boat.sheet == 1.0
    assert sim.actions.latest == "you eased the main sheet"

    assert lever.handle_event(sim, up)
    assert not lever.dragging
    pygame.quit()


def test_default_course_contains_buoys_and_rocks() -> None:
    markers = default_course_markers()
    kinds = {marker.kind for marker in markers}

    assert "buoy" in kinds
    assert "rock" in kinds
    assert len([marker for marker in markers if marker.kind == "buoy"]) >= 4


def test_course_point_moves_along_navigation_heading() -> None:
    north = course_point((0.0, 0.0), 0.0, 100.0)
    east = course_point((0.0, 0.0), 90.0, 100.0)

    assert round(north[0], 6) == 0.0
    assert round(north[1], 6) == 100.0
    assert round(east[0], 6) == 100.0
    assert round(east[1], 6) == 0.0


def test_default_course_requires_upwind_zig_zags_then_downwind_run() -> None:
    buoys = [marker for marker in default_course_markers() if marker.kind == "buoy"]

    assert [buoy.label for buoy in buoys] == [
        "Start",
        "Tack 1",
        "Tack 2",
        "Tack 3",
        "Windward",
        "Run 1",
        "Run 2",
        "Finish",
    ]

    leg_angles = [
        bearing_between(start.east, start.north, end.east, end.north)
        for start, end in zip(buoys, buoys[1:])
    ]
    upwind_offsets = [
        signed_angle_diff(angle, DEFAULT_COURSE_WIND_FROM_DEG)
        for angle in leg_angles[:4]
    ]

    assert [offset < 0 for offset in upwind_offsets] == [True, False, True, False]
    assert all(36.0 <= abs(offset) <= 44.0 for offset in upwind_offsets)

    downwind_offsets = [
        signed_angle_diff(angle, DEFAULT_COURSE_WIND_FROM_DEG)
        for angle in leg_angles[4:]
    ]
    assert all(abs(abs(offset) - 180.0) < 0.1 for offset in downwind_offsets)


def test_world_to_screen_scrolls_markers_as_boat_moves() -> None:
    sim = SailingSimulation()
    before = world_to_screen(sim, 0.0, 260.0)

    sim.boat.position_north = 100.0
    after = world_to_screen(sim, 0.0, 260.0)

    assert after[1] > before[1]


def test_visible_course_markers_filters_distant_objects() -> None:
    sim = SailingSimulation()
    markers = default_course_markers()

    visible = visible_course_markers(sim, markers)

    assert any(marker.label == "Start" for marker, _pos in visible)
    assert all(-80 <= pos[0] <= WIDTH + 80 for _marker, pos in visible)


def test_visible_traffic_filters_distant_vessels() -> None:
    sim = SailingSimulation(
        traffic=[
            TrafficVessel("near", "motorboat", 20.0, 20.0, 90.0, 3.0),
            TrafficVessel("far", "sailboat", 4000.0, 4000.0, 0.0, 3.0),
        ]
    )

    visible = visible_traffic(sim)

    assert [vessel.id for vessel, _pos in visible] == ["near"]


def test_traffic_hull_polygon_rotates_around_center() -> None:
    points = traffic_hull_polygon((100.0, 100.0), 45.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    assert min(xs) < 100.0 < max(xs)
    assert min(ys) < 100.0 < max(ys)


def test_no_sail_zone_points_form_small_wind_centered_sector() -> None:
    points = no_sail_zone_points(0.0, radius=100.0, steps=2)

    assert points[0] == BOAT_CENTER
    assert len(points) == 4
    assert all(point[1] < BOAT_CENTER[1] for point in points[1:])


def test_no_sail_toggle_mouse_click_flips_state() -> None:
    toggle = NoSailZoneToggle.default()
    event = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN,
        {"button": 1, "pos": toggle.rect.center},
    )

    assert toggle.enabled
    assert toggle.handle_event(event)
    assert not toggle.enabled


def test_sail_geometry_keeps_boom_aft_and_to_leeward() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.55),
        wind=WindState(from_deg=90, speed=12),
    )
    visual = SailVisualState.from_sim(sim)
    sail = sail_geometry(sim, visual)

    assert sail.clew[1] > sail.mast[1]
    assert sail.clew[0] < sail.mast[0]


def test_sail_geometry_rotates_with_the_boat() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=90, sheet=0.35),
        wind=WindState(from_deg=180, speed=12),
    )
    visual = SailVisualState.from_sim(sim)
    sail = sail_geometry(sim, visual)

    assert sail.head[0] > sail.mast[0]


def test_sail_visual_state_eases_across_tack_changes() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.55),
        wind=WindState(from_deg=90, speed=12),
    )
    visual = SailVisualState.from_sim(sim)
    original = visual.boom_angle_deg

    sim.wind.from_deg = 270
    target = desired_boom_angle(sim)
    visual.update(sim, 0.05)

    assert original < 0
    assert target > 0
    assert original < visual.boom_angle_deg < 0


def test_centered_tiller_and_rudder_stay_attached_to_stern() -> None:
    sim = SailingSimulation(boat=BoatState(heading_deg=0, tiller=0.0))
    steering = tiller_rudder_geometry(sim)

    assert steering.rudder_tip[1] > steering.rudder_pivot[1]
    assert steering.tiller_handle[1] < steering.tiller_pivot[1]
    assert steering.rudder_tip[0] == steering.rudder_pivot[0]
    assert steering.tiller_handle[0] == steering.tiller_pivot[0]


def test_deflected_tiller_and_rudder_move_opposite_each_other() -> None:
    sim = SailingSimulation(boat=BoatState(heading_deg=0, tiller=0.8))
    steering = tiller_rudder_geometry(sim)

    tiller_side = steering.tiller_handle[0] - steering.tiller_pivot[0]
    rudder_side = steering.rudder_tip[0] - steering.rudder_pivot[0]
    assert tiller_side < 0
    assert rudder_side > 0


def test_tiller_and_rudder_rotate_with_boat_heading() -> None:
    sim = SailingSimulation(boat=BoatState(heading_deg=90, tiller=0.0))
    steering = tiller_rudder_geometry(sim)

    assert steering.tiller_handle[0] > steering.tiller_pivot[0]
    assert steering.rudder_tip[0] < steering.rudder_pivot[0]


def test_handle_keys_can_be_imported_without_opening_a_window() -> None:
    pygame.init()
    key_count = 512
    keys = [False] * key_count
    sim = SailingSimulation()

    assert handle_keys(sim, keys, 0.016) is False
    pygame.quit()


def test_can_render_one_frame_to_a_surface() -> None:
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))
    font = pygame.font.SysFont("arial", 18)
    small = pygame.font.SysFont("arial", 15)
    sim = SailingSimulation(
        traffic=[
            TrafficVessel("M", "motorboat", 70.0, 40.0, 250.0, 4.0),
            TrafficVessel("S", "sailboat", -80.0, 60.0, 20.0, 3.0),
        ]
    )
    sim.penalty_timer = 1.0

    draw_water(surface, sim.wind.from_deg, 0.0)
    draw_course_markers(surface, small, sim, default_course_markers(), 0.0)
    draw_traffic_vessels(surface, small, sim, 0.0)
    draw_no_sail_zone(surface, sim, True)
    draw_boat(surface, sim)
    draw_info_box(surface, font, small, sim)
    draw_tilt_meter(surface, pygame.Rect(24, 24, 305, 404), sim)
    draw_wind_indicator(surface, font, sim)
    draw_no_sail_toggle(surface, font, small, NoSailZoneToggle.default())
    draw_sail_efficiency_panel(surface, font, small, sim)
    draw_main_sheet_lever(surface, font, small, sim, MainSheetLever.default())
    draw_rule_penalty(surface, font, sim)

    center_color = surface.get_at(BOAT_CENTER)
    assert center_color != surface.get_at((0, 0))
    pygame.quit()
