from sailing_simulator.physics import (
    BoatState,
    SailingSimulation,
    TrafficVessel,
    WindState,
    assess_right_of_way,
    calculate_sail_forces,
    classify_crossing,
    default_traffic,
    ideal_sheet_for_angle,
    sailing_terms,
    stability_state,
    signed_angle_diff,
)


def run_steps(sim: SailingSimulation, count: int, dt: float = 0.05) -> None:
    for _ in range(count):
        sim.step(dt)


def test_signed_angle_diff_wraps_to_shortest_direction() -> None:
    assert signed_angle_diff(10, 350) == 20
    assert signed_angle_diff(350, 10) == -20
    assert signed_angle_diff(180, 0) == -180


def test_sailing_terms_identify_starboard_and_port_tacks() -> None:
    starboard = sailing_terms(BoatState(heading_deg=0), WindState(from_deg=90))
    port = sailing_terms(BoatState(heading_deg=0), WindState(from_deg=270))

    assert starboard["tack"] == "starboard tack"
    assert starboard["beam"] == "starboard beam"
    assert starboard["point_of_sail"] == "beam reach"
    assert port["tack"] == "port tack"
    assert port["beam"] == "port beam"


def test_sailing_terms_identify_irons_and_running() -> None:
    irons = sailing_terms(BoatState(heading_deg=0), WindState(from_deg=15))
    running = sailing_terms(BoatState(heading_deg=0), WindState(from_deg=180))

    assert irons["point_of_sail"] == "in irons"
    assert irons["beam"] == "on the bow"
    assert running["point_of_sail"] == "running"
    assert running["beam"] == "centerline"


def test_sheet_target_eases_as_relative_wind_moves_aft() -> None:
    assert ideal_sheet_for_angle(40) < ideal_sheet_for_angle(90)
    assert ideal_sheet_for_angle(90) < ideal_sheet_for_angle(170)


def test_good_trim_produces_more_speed_than_bad_trim() -> None:
    wind = WindState(from_deg=90, speed=12)
    good = BoatState(heading_deg=0, sheet=ideal_sheet_for_angle(90))
    bad = BoatState(heading_deg=0, sheet=0.0)

    good_forces = calculate_sail_forces(good, wind)
    bad_forces = calculate_sail_forces(bad, wind)

    assert good_forces.target_speed > bad_forces.target_speed * 2.5
    assert good_forces.trim_efficiency > bad_forces.trim_efficiency


def test_boat_accelerates_on_a_trimmed_beam_reach() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=ideal_sheet_for_angle(90)),
        wind=WindState(from_deg=90, speed=10),
    )

    run_steps(sim, 80)

    assert sim.boat.speed > 4.0
    assert not sim.boat.capsized


def test_boat_world_position_advances_with_speed_and_heading() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=ideal_sheet_for_angle(90)),
        wind=WindState(from_deg=90, speed=10),
    )

    run_steps(sim, 80)

    assert sim.boat.position_north > 60.0
    assert abs(sim.boat.position_east) < 25.0


def test_overtrimmed_reach_has_low_drive_but_high_heeling_load() -> None:
    wind = WindState(from_deg=90, speed=13)
    good = BoatState(heading_deg=0, sheet=ideal_sheet_for_angle(90))
    overtrimmed = BoatState(heading_deg=0, sheet=0.0)

    good_forces = calculate_sail_forces(good, wind)
    overtrimmed_forces = calculate_sail_forces(overtrimmed, wind)

    assert overtrimmed_forces.target_speed < good_forces.target_speed * 0.5
    assert overtrimmed_forces.target_heel_deg > good_forces.target_heel_deg
    assert overtrimmed_forces.heel_load > 0.9


def test_boat_stays_slow_in_irons() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.0),
        wind=WindState(from_deg=5, speed=14),
    )

    run_steps(sim, 80)

    assert sim.boat.speed < 1.0
    assert sailing_terms(sim.boat, sim.wind)["point_of_sail"] == "in irons"


def test_wind_changes_gently_and_stays_bounded() -> None:
    wind = WindState(from_deg=70, speed=11)
    speeds = []
    directions = []
    for _ in range(180):
        old_direction = wind.from_deg
        wind.update(0.1)
        speeds.append(wind.speed)
        directions.append(abs(signed_angle_diff(wind.from_deg, old_direction)))

    assert min(speeds) >= wind.minimum_speed
    assert max(speeds) <= wind.maximum_speed
    assert max(directions) < 0.6
    assert abs(speeds[-1] - speeds[0]) > 0.5


def test_default_traffic_includes_motorboats_and_sailboats() -> None:
    traffic = default_traffic()
    kinds = {vessel.kind for vessel in traffic}

    assert "motorboat" in kinds
    assert "sailboat" in kinds


def test_traffic_vessel_advances_along_heading() -> None:
    vessel = TrafficVessel("T", "motorboat", 0.0, 0.0, 0.0, 2.0)

    vessel.advance(1.0)

    assert vessel.position_east == 0.0
    assert vessel.position_north == 36.0


def test_sail_stands_on_against_powerboat_unless_overtaking() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=0, speed=3)
    motorboat = TrafficVessel("M", "motorboat", 50.0, 0.0, 270.0, 4.0)

    assessment = assess_right_of_way(player, motorboat, wind)

    assert not assessment.player_must_give_way
    assert "sail" in assessment.reason


def test_overtaking_player_must_keep_clear_even_of_motorboat() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=0, speed=4, position_north=0)
    motorboat = TrafficVessel("M", "motorboat", 0.0, 60.0, 0.0, 2.0)

    assessment = assess_right_of_way(player, motorboat, wind)

    assert assessment.player_must_give_way
    assert "overtaking" in assessment.reason


def test_port_tack_player_gives_way_to_starboard_tack_sailboat() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=180, speed=3)
    starboard = TrafficVessel("S", "sailboat", 40.0, 0.0, 0.0, 3.0)

    assessment = assess_right_of_way(player, starboard, wind)

    assert assessment.player_must_give_way
    assert "port tack" in assessment.reason


def test_windward_player_gives_way_on_same_tack() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=0, speed=3)
    leeward_boat = TrafficVessel("S", "sailboat", -45.0, 0.0, 0.0, 3.0)

    assessment = assess_right_of_way(player, leeward_boat, wind)

    assert assessment.player_must_give_way
    assert "windward" in assessment.reason


def test_rule_violation_triggers_red_penalty_when_player_must_give_way() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=180, speed=2),
        wind=WindState(from_deg=90, speed=10),
        traffic=[TrafficVessel("S", "sailboat", 35.0, 0.0, 0.0, 2.0)],
    )

    sim.step(0.01)

    assert sim.penalty_timer > 0.0
    assert sim.right_of_way[0].violation
    assert sim.actions.latest.startswith("rule violation:")


def test_stand_on_player_does_not_get_penalized_for_close_traffic() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, speed=2),
        wind=WindState(from_deg=90, speed=10),
        traffic=[TrafficVessel("S", "sailboat", 35.0, 0.0, 180.0, 2.0)],
    )

    sim.step(0.01)

    assert sim.penalty_timer == 0.0
    assert not sim.right_of_way[0].player_must_give_way


def test_sustained_overpowering_capsizes_the_boat() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.15),
        wind=WindState(from_deg=82, speed=18),
    )

    run_steps(sim, 90)

    assert sim.boat.capsized
    assert sim.actions.latest == "you capsized"


def test_overtrimmed_beam_reach_capsizes_in_a_fresh_breeze() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.0),
        wind=WindState(from_deg=90, speed=13),
    )

    run_steps(sim, 80)

    assert sim.boat.capsized
    assert stability_state(sim.boat).status == "capsized"


def test_eased_sheet_survives_strong_wind_longer() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.95),
        wind=WindState(from_deg=125, speed=16),
    )

    run_steps(sim, 60)

    assert not sim.boat.capsized
    assert abs(sim.boat.heel_deg) < 55


def test_stability_state_reports_tilt_and_capsize_progress() -> None:
    stable = stability_state(BoatState(heel_deg=12, capsize_timer=0.0))
    warning = stability_state(BoatState(heel_deg=-40, capsize_timer=0.0))
    danger = stability_state(BoatState(heel_deg=45, capsize_timer=0.8))

    assert stable.status == "stable"
    assert stable.heel_abs_deg == 12
    assert warning.status == "warning"
    assert danger.status == "danger"
    assert danger.risk > warning.risk


def test_righting_boat_resets_capsize_state_and_eases_sheet() -> None:
    sim = SailingSimulation(boat=BoatState(capsized=True, sheet=0.2, speed=4, heel_deg=88))

    sim.right_boat()

    assert not sim.boat.capsized
    assert sim.boat.speed == 0.4
    assert sim.boat.sheet >= 0.75
    assert sim.actions.latest == "you righted the boat"


def test_crossing_classification_distinguishes_tack_and_gybe() -> None:
    assert classify_crossing(42, -39) == "you have tacked"
    assert classify_crossing(-170, 168) == "you have gybed"
    assert classify_crossing(80, 85) is None


def test_positive_tiller_command_turns_boat_starboard() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, speed=4, tiller=1.0),
        wind=WindState(from_deg=140, speed=8),
    )

    sim.step(0.05)

    assert 0 < sim.boat.heading_deg < 10


def test_action_log_records_heading_up_and_bearing_away() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, speed=4, tiller=1.0),
        wind=WindState(from_deg=85, speed=10),
    )

    run_steps(sim, 14)
    assert "you are heading up" in sim.actions.messages

    sim.boat.tiller = -1.0
    sim.action_cooldown = 0.0
    run_steps(sim, 14)
    assert "you are bearing away" in sim.actions.messages


def test_adjustments_are_clamped_and_logged() -> None:
    sim = SailingSimulation()

    sim.adjust_tiller(1.0, 10.0)
    sim.adjust_sheet(-1.0, 10.0)
    assert sim.boat.tiller == 1.0
    assert sim.boat.sheet == 0.0
    assert sim.actions.latest == "you trimmed in the main sheet"

    sim.adjust_tiller(-1.0, 10.0)
    sim.adjust_sheet(1.0, 10.0)
    assert sim.boat.tiller == -1.0
    assert sim.boat.sheet == 1.0
    assert sim.actions.latest == "you eased the main sheet"
