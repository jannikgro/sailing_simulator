from math import hypot

from sailing_simulator.levels import (
    LEVELS,
    PLAYER_START,
    build_simulation,
    channel_rocks,
    flank_rocks,
    leg,
)
from sailing_simulator.physics import (
    NO_GO_ZONE_DEG,
    bearing_between,
    signed_angle_diff,
)


MIN_ROCK_LEG_CLEARANCE = 80.0


def level_waypoints(level) -> list[tuple[float, float]]:
    return [PLAYER_START] + [(o.east, o.north) for o in level.objectives]


def point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    seg_east = end[0] - start[0]
    seg_north = end[1] - start[1]
    length_sq = seg_east * seg_east + seg_north * seg_north
    if length_sq == 0.0:
        return hypot(point[0] - start[0], point[1] - start[1])
    t = ((point[0] - start[0]) * seg_east + (point[1] - start[1]) * seg_north) / length_sq
    t = max(0.0, min(1.0, t))
    closest = (start[0] + seg_east * t, start[1] + seg_north * t)
    return hypot(point[0] - closest[0], point[1] - closest[1])


def test_leg_moves_along_navigation_heading() -> None:
    north = leg((0.0, 0.0), 0.0, 100.0)
    east = leg((0.0, 0.0), 90.0, 100.0)

    assert round(north[0], 6) == 0.0
    assert round(north[1], 6) == 100.0
    assert round(east[0], 6) == 100.0
    assert round(east[1], 6) == 0.0


def test_flank_rocks_sit_offset_from_the_leg() -> None:
    rocks = flank_rocks((0.0, 0.0), (0.0, 1000.0), 150.0)

    assert len(rocks) == 2
    for rock in rocks:
        assert abs(abs(rock.east) - 150.0) < 1e-6
        assert 0.0 < rock.north < 1000.0


def test_channel_rocks_line_both_sides_of_the_path() -> None:
    rocks = channel_rocks(((0.0, 0.0), (0.0, 800.0)), spacing=200.0, half_width=100.0)

    east_side = [rock for rock in rocks if rock.east > 0]
    west_side = [rock for rock in rocks if rock.east < 0]
    assert len(east_side) == len(west_side) == 4
    assert all(abs(abs(rock.east) - 100.0) < 1e-6 for rock in rocks)


def test_levels_are_numbered_in_order_with_unique_titles() -> None:
    assert [level.number for level in LEVELS] == list(range(1, len(LEVELS) + 1))
    assert len(LEVELS) >= 7
    assert len({level.title for level in LEVELS}) == len(LEVELS)
    assert all(level.briefing for level in LEVELS)


def test_early_levels_have_no_traffic_and_later_levels_do() -> None:
    for level in LEVELS[:5]:
        assert not level.traffic, f"level {level.number} should have no traffic"
    for level in LEVELS[5:]:
        assert level.traffic, f"level {level.number} should have traffic"


def test_rocks_are_absent_early_and_used_in_later_levels() -> None:
    for level in LEVELS[:3]:
        assert not level.rocks, f"level {level.number} should have no rocks"
    rocky_levels = [level for level in LEVELS if level.rocks]
    assert len(rocky_levels) >= 3
    assert LEVELS[-1].rocks, "the final level should use rocks"


def test_difficulty_ramps_wind_variability() -> None:
    first = LEVELS[0].wind
    last = LEVELS[-1].wind

    assert first.gust_amplitude < last.gust_amplitude
    assert first.shift_amplitude < last.shift_amplitude
    assert first.maximum_speed < last.maximum_speed


def test_every_leg_is_sailable_outside_the_no_go_zone() -> None:
    for level in LEVELS:
        points = level_waypoints(level)
        for start, end in zip(points, points[1:]):
            bearing = bearing_between(start[0], start[1], end[0], end[1])
            relative = abs(signed_angle_diff(level.wind.from_deg, bearing))
            assert relative >= NO_GO_ZONE_DEG + 3.0, (
                f"level {level.number}: leg {start}->{end} points into the wind"
            )


def test_start_heading_is_sailable() -> None:
    for level in LEVELS:
        relative = abs(
            signed_angle_diff(level.wind.from_deg, level.start_heading_deg)
        )
        assert relative >= NO_GO_ZONE_DEG + 5.0


def test_consecutive_objectives_are_spread_out() -> None:
    for level in LEVELS:
        points = level_waypoints(level)
        for start, end in zip(points, points[1:]):
            assert hypot(end[0] - start[0], end[1] - start[1]) >= 250.0


def test_rocks_keep_clear_of_the_course_line_and_buoys() -> None:
    for level in LEVELS:
        points = level_waypoints(level)
        for rock in level.rocks:
            for start, end in zip(points, points[1:]):
                distance = point_segment_distance((rock.east, rock.north), start, end)
                assert distance >= MIN_ROCK_LEG_CLEARANCE, (
                    f"level {level.number}: rock at ({rock.east:.0f}, {rock.north:.0f}) "
                    f"is {distance:.0f} from leg {start}->{end}"
                )
            for objective in level.objectives:
                distance = hypot(rock.east - objective.east, rock.north - objective.north)
                assert distance >= objective.radius + rock.radius + 10.0
            start_distance = hypot(rock.east, rock.north)
            assert start_distance >= 100.0


def test_traffic_starts_away_from_the_player() -> None:
    for level in LEVELS:
        for spec in level.traffic:
            assert hypot(spec.east, spec.north) >= 300.0
            assert spec.kind in {"motorboat", "sailboat"}
            assert len(spec.waypoints) >= 2


def test_build_simulation_creates_fresh_state_each_time() -> None:
    level = LEVELS[-1]
    first = build_simulation(level)
    second = build_simulation(level)

    first.traffic[0].position_east += 999.0
    first.wind.from_deg += 90.0

    assert second.traffic[0].position_east != first.traffic[0].position_east
    assert second.wind.from_deg == level.wind.from_deg
    assert second.boat.position_east == 0.0
    assert second.boat.position_north == 0.0
    assert len(second.rocks) == len(level.rocks)
    assert all(vessel.waypoints for vessel in second.traffic)


def test_built_wind_matches_the_spec() -> None:
    for level in LEVELS:
        wind = level.wind.build()
        assert wind.from_deg == level.wind.from_deg
        assert wind.speed == level.wind.speed
        assert wind.base_speed == level.wind.speed
        assert wind.minimum_speed == level.wind.minimum_speed
        assert wind.maximum_speed == level.wind.maximum_speed
