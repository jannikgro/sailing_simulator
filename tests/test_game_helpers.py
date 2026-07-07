import os
from collections import defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from sailing_simulator.game import (
    BOAT_CENTER,
    BRIEFING_SECONDS,
    HEIGHT,
    WIDTH,
    GAME_SPEED,
    RESPAWN_SECONDS,
    LevelSession,
    MainSheetLever,
    MenuState,
    NoSailZoneToggle,
    POINT_OF_SAIL_BANDS,
    SailVisualState,
    boat_polygon,
    desired_boom_angle,
    draw_boat,
    draw_briefing,
    draw_hud_banner,
    draw_info_box,
    draw_level_complete,
    draw_main_sheet_lever,
    draw_menu,
    draw_no_sail_toggle,
    draw_no_sail_zone,
    draw_objectives,
    draw_respawn_countdown,
    draw_rocks,
    draw_rule_penalty,
    draw_sail_efficiency_panel,
    draw_scene,
    draw_target_arrow,
    draw_tilt_meter,
    draw_traffic_vessels,
    draw_water,
    draw_wind_indicator,
    efficiency_color,
    format_time,
    handle_keys,
    info_lines,
    main,
    menu_row_rect,
    no_sail_zone_points,
    on_screen,
    point_of_sail_marker_x,
    sail_geometry,
    sail_efficiency,
    sailable_heading,
    scaled_game_dt,
    sheet_from_lever_y,
    sheet_percent_in,
    sheet_to_lever_y,
    stability_color,
    traffic_hull_polygon,
    tiller_rudder_geometry,
    visible_traffic,
    world_to_screen,
)
from sailing_simulator.levels import LEVELS
from sailing_simulator.physics import (
    NO_GO_ZONE_DEG,
    BoatState,
    SailingSimulation,
    TrafficVessel,
    WindState,
    signed_angle_diff,
)


def make_fonts() -> dict[str, pygame.font.Font]:
    return {
        "font": pygame.font.SysFont("arial", 18),
        "small": pygame.font.SysFont("arial", 15),
        "big": pygame.font.SysFont("arial", 56, bold=True),
    }


# --- level session -----------------------------------------------------------


def test_session_begins_at_level_start_with_level_wind() -> None:
    level = LEVELS[2]
    session = LevelSession.begin(level)

    assert session.sim.boat.position_east == 0.0
    assert session.sim.boat.position_north == 0.0
    assert session.sim.boat.heading_deg == level.start_heading_deg % 360.0
    assert session.sim.wind.from_deg == level.wind.from_deg
    assert session.target_index == 0
    assert not session.completed
    assert not session.failed


def test_session_rounds_objectives_and_moves_checkpoint() -> None:
    session = LevelSession.begin(LEVELS[0])
    first = session.level.objectives[0]
    session.sim.boat.position_east = first.east
    session.sim.boat.position_north = first.north

    session.update(0.02)

    assert session.target_index == 1
    assert session.checkpoint == (first.east, first.north)
    assert f"rounded {first.label}" in session.sim.actions.messages


def test_session_completes_after_last_objective() -> None:
    session = LevelSession.begin(LEVELS[0])
    for objective in session.level.objectives:
        session.sim.boat.position_east = objective.east
        session.sim.boat.position_north = objective.north
        session.update(0.02)

    assert session.completed
    assert session.current_objective() is None


def test_capsize_fails_the_session_and_starts_the_countdown() -> None:
    session = LevelSession.begin(LEVELS[0])
    session.sim.boat.capsized = True

    session.update(0.02)

    assert session.failed
    assert session.respawn_timer > RESPAWN_SECONDS - 0.1
    assert session.failure_reason == "Capsized!"


def test_hitting_a_rock_reports_the_rock_failure() -> None:
    session = LevelSession.begin(LEVELS[3])
    session.sim.grounded = True

    session.update(0.02)

    assert session.failed
    assert session.failure_reason == "You hit a rock!"


def test_respawn_happens_after_the_three_second_countdown() -> None:
    session = LevelSession.begin(LEVELS[0])
    first = session.level.objectives[0]
    session.sim.boat.position_east = first.east
    session.sim.boat.position_north = first.north
    session.update(0.02)
    assert session.target_index == 1

    session.sim.boat.position_east += 200.0
    session.sim.boat.capsized = True
    session.update(0.02)
    assert session.failed

    total = 0.0
    while session.failed and total < RESPAWN_SECONDS + 0.5:
        session.update(0.1)
        total += 0.1

    assert total >= RESPAWN_SECONDS - 0.2
    assert not session.failed
    assert not session.sim.boat.capsized
    # The boat respawns exactly at the checkpoint buoy and sails on from there.
    assert abs(session.sim.boat.position_east - first.east) < 20.0
    assert abs(session.sim.boat.position_north - first.north) < 20.0
    assert session.target_index == 1


def test_elapsed_time_freezes_while_failed() -> None:
    session = LevelSession.begin(LEVELS[0])
    session.update(0.5)
    assert session.elapsed > 0.0

    session.sim.boat.capsized = True
    session.update(0.02)
    frozen = session.elapsed
    session.update(0.5)

    assert session.elapsed == frozen


def test_sailable_heading_stays_outside_the_no_go_zone() -> None:
    wind_from = 90.0
    into_wind = sailable_heading(wind_from, 92.0)
    fine_course = sailable_heading(wind_from, 0.0)

    assert abs(signed_angle_diff(wind_from, into_wind)) >= NO_GO_ZONE_DEG
    assert fine_course == 0.0


# --- menu --------------------------------------------------------------------


def test_menu_selection_wraps_in_both_directions() -> None:
    menu = MenuState()
    menu.move(-1)
    assert menu.selected == len(LEVELS) - 1
    menu.move(1)
    assert menu.selected == 0


def test_menu_keyboard_events_select_and_start() -> None:
    pygame.init()
    menu = MenuState()

    down = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN})
    enter = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RETURN})
    escape = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})

    assert menu.handle_event(down) is None
    assert menu.selected == 1
    assert menu.handle_event(enter) == "start"
    assert menu.handle_event(escape) == "quit"
    pygame.quit()


def test_menu_number_key_jumps_to_level_and_starts() -> None:
    pygame.init()
    menu = MenuState()
    three = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_3})

    assert menu.handle_event(three) == "start"
    assert menu.selected == 2
    pygame.quit()


def test_menu_mouse_click_on_row_starts_that_level() -> None:
    pygame.init()
    menu = MenuState()
    row = menu_row_rect(4)
    click = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": row.center}
    )

    assert menu.handle_event(click) == "start"
    assert menu.selected == 4
    pygame.quit()


def test_menu_rows_do_not_overlap() -> None:
    for first, second in zip(range(len(LEVELS)), range(1, len(LEVELS))):
        assert menu_row_rect(first).bottom <= menu_row_rect(second).top + 8


# --- drawing helpers ----------------------------------------------------------


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


def test_format_time_renders_minutes_and_seconds() -> None:
    assert format_time(0.0) == "00:00"
    assert format_time(83.4) == "01:23"


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


def test_world_to_screen_scrolls_markers_as_boat_moves() -> None:
    sim = SailingSimulation()
    before = world_to_screen(sim, 0.0, 260.0)

    sim.boat.position_north = 100.0
    after = world_to_screen(sim, 0.0, 260.0)

    assert after[1] > before[1]
    assert on_screen(before)
    assert not on_screen((-500.0, -500.0))


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


def test_handle_keys_moves_tiller_without_a_window() -> None:
    pygame.init()
    keys = defaultdict(bool)
    keys[pygame.K_RIGHT] = True
    sim = SailingSimulation()

    handle_keys(sim, keys, 0.1)

    assert sim.boat.tiller > 0.0
    pygame.quit()


# --- rendering smoke tests -----------------------------------------------------


def test_can_render_a_full_level_scene() -> None:
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))
    fonts = make_fonts()
    session = LevelSession.begin(LEVELS[-1])
    session.sim.penalty_timer = 1.0
    visual = SailVisualState.from_sim(session.sim)

    draw_scene(
        surface,
        fonts,
        session,
        visual,
        MainSheetLever.default(),
        NoSailZoneToggle.default(),
        time_s=0.5,
    )

    center_color = surface.get_at(BOAT_CENTER)
    assert center_color != surface.get_at((0, 0))
    pygame.quit()


def test_can_render_rocks_objectives_and_overlays() -> None:
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))
    fonts = make_fonts()
    session = LevelSession.begin(LEVELS[4])

    draw_water(surface, session.sim.wind.from_deg, 0.0)
    draw_rocks(surface, session.sim)
    draw_objectives(surface, fonts["small"], session, 0.0)
    draw_target_arrow(surface, fonts["small"], session)
    draw_hud_banner(surface, fonts["font"], fonts["small"], session)
    draw_briefing(surface, fonts["font"], fonts["small"], session)
    assert session.elapsed <= BRIEFING_SECONDS

    session.sim.boat.capsized = True
    session.update(0.02)
    draw_respawn_countdown(surface, fonts["big"], fonts["font"], session)

    session.respawn_timer = 0.0
    session.completed = True
    draw_level_complete(surface, fonts["big"], fonts["font"], session)
    pygame.quit()


def test_can_render_the_menu() -> None:
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))
    fonts = make_fonts()
    menu = MenuState(selected=2, completed={1})

    draw_menu(surface, fonts["big"], fonts["font"], fonts["small"], menu, 0.0)

    row = menu_row_rect(2)
    assert surface.get_at(row.center) != surface.get_at((WIDTH - 4, HEIGHT - 4))
    pygame.quit()


def test_can_render_panels_on_a_plain_simulation() -> None:
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))
    fonts = make_fonts()
    sim = SailingSimulation(
        traffic=[
            TrafficVessel("M", "motorboat", 70.0, 40.0, 250.0, 4.0),
            TrafficVessel("S", "sailboat", -80.0, 60.0, 20.0, 3.0),
        ]
    )
    sim.penalty_timer = 1.0

    draw_water(surface, sim.wind.from_deg, 0.0)
    draw_traffic_vessels(surface, fonts["small"], sim, 0.0)
    draw_no_sail_zone(surface, sim, True)
    draw_boat(surface, sim)
    draw_info_box(surface, fonts["font"], fonts["small"], sim)
    draw_tilt_meter(surface, pygame.Rect(24, 24, 305, 404), sim)
    draw_wind_indicator(surface, fonts["font"], sim)
    draw_no_sail_toggle(surface, fonts["font"], fonts["small"], NoSailZoneToggle.default())
    draw_sail_efficiency_panel(surface, fonts["font"], fonts["small"], sim)
    draw_main_sheet_lever(surface, fonts["font"], fonts["small"], sim, MainSheetLever.default())
    draw_rule_penalty(surface, fonts["font"], sim)
    pygame.quit()


def test_main_loop_runs_a_few_frames_headless() -> None:
    main(max_frames=3)
