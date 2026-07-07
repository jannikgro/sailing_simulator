from math import hypot

from sailing_simulator.physics import (
    BoatState,
    Rock,
    SailingSimulation,
    TrafficVessel,
    VesselDescriptor,
    WindState,
    assess_right_of_way,
    calculate_sail_forces,
    classify_crossing,
    descriptor_for_boat,
    descriptor_for_traffic,
    encounter_geometry,
    find_rock_collision,
    find_vessel_collision,
    give_way_reason,
    ideal_sheet_for_angle,
    is_overtaking,
    sailing_terms,
    stability_state,
    signed_angle_diff,
    turn_toward,
    update_traffic_vessel,
)


def run_steps(sim: SailingSimulation, count: int, dt: float = 0.05) -> None:
    for _ in range(count):
        sim.step(dt)


def make_descriptor(
    east: float = 0.0,
    north: float = 0.0,
    heading_deg: float = 0.0,
    speed: float = 3.0,
    kind: str = "sailboat",
) -> VesselDescriptor:
    return VesselDescriptor(
        east=east, north=north, heading_deg=heading_deg, speed=speed, kind=kind
    )


def test_signed_angle_diff_wraps_to_shortest_direction() -> None:
    assert signed_angle_diff(10, 350) == 20
    assert signed_angle_diff(350, 10) == -20
    assert signed_angle_diff(180, 0) == -180


def test_turn_toward_moves_heading_along_shortest_arc() -> None:
    assert turn_toward(350.0, 10.0, 30.0) == 10.0
    assert turn_toward(350.0, 10.0, 5.0) == 355.0
    assert turn_toward(10.0, 350.0, 5.0) == 5.0
    assert turn_toward(90.0, 90.0, 5.0) == 90.0


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


def test_wind_spec_parameters_allow_a_steady_training_breeze() -> None:
    wind = WindState(
        from_deg=90,
        speed=8,
        base_speed=8.0,
        gust_amplitude=0.4,
        shift_amplitude=0.1,
        minimum_speed=6.0,
        maximum_speed=10.0,
    )
    for _ in range(300):
        wind.update(0.1)

    assert 7.0 < wind.speed < 9.0
    assert abs(signed_angle_diff(wind.from_deg, 90.0)) < 6.0


# --- right of way -----------------------------------------------------------


def test_overtaking_requires_astern_similar_course_and_more_speed() -> None:
    ahead = make_descriptor(north=60.0, heading_deg=0.0, speed=2.0)

    behind_faster = make_descriptor(north=0.0, heading_deg=0.0, speed=4.0)
    behind_slower = make_descriptor(north=0.0, heading_deg=0.0, speed=1.0)
    behind_heading_away = make_descriptor(north=0.0, heading_deg=180.0, speed=4.0)
    abeam = make_descriptor(east=60.0, north=60.0, heading_deg=0.0, speed=4.0)

    assert is_overtaking(behind_faster, ahead)
    assert not is_overtaking(behind_slower, ahead)
    assert not is_overtaking(behind_heading_away, ahead)
    assert not is_overtaking(abeam, ahead)


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


def test_player_being_overtaken_stands_on() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=0, speed=2, position_north=60.0)
    overtaker = TrafficVessel("M", "motorboat", 0.0, 0.0, 0.0, 5.0)

    assessment = assess_right_of_way(player, overtaker, wind)

    assert not assessment.player_must_give_way
    assert "overtaken" in assessment.reason


def test_port_tack_player_gives_way_to_starboard_tack_sailboat() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=180, speed=3)
    starboard = TrafficVessel("S", "sailboat", 40.0, 0.0, 0.0, 3.0)

    assessment = assess_right_of_way(player, starboard, wind)

    assert assessment.player_must_give_way
    assert "port tack" in assessment.reason


def test_starboard_tack_player_stands_on_against_port_tack() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=0, speed=3)
    port_boat = TrafficVessel("S", "sailboat", 40.0, 0.0, 180.0, 3.0)

    assessment = assess_right_of_way(player, port_boat, wind)

    assert not assessment.player_must_give_way
    assert "starboard" in assessment.reason


def test_windward_player_gives_way_on_same_tack() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=0, speed=3)
    leeward_boat = TrafficVessel("S", "sailboat", -45.0, 0.0, 0.0, 3.0)

    assessment = assess_right_of_way(player, leeward_boat, wind)

    assert assessment.player_must_give_way
    assert "windward" in assessment.reason


def test_leeward_player_stands_on_on_same_tack() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=0, speed=3)
    windward_boat = TrafficVessel("S", "sailboat", 45.0, 0.0, 0.0, 3.0)

    assessment = assess_right_of_way(player, windward_boat, wind)

    assert not assessment.player_must_give_way
    assert "leeward" in assessment.reason


def test_give_way_rules_are_symmetric_between_perspectives() -> None:
    wind = WindState(from_deg=90, speed=10)
    player = BoatState(heading_deg=180, speed=3)
    vessel = TrafficVessel("S", "sailboat", 40.0, 0.0, 0.0, 3.0)

    player_gives, _ = give_way_reason(
        descriptor_for_boat(player), descriptor_for_traffic(vessel), wind
    )
    vessel_gives, _ = give_way_reason(
        descriptor_for_traffic(vessel), descriptor_for_boat(player), wind
    )

    assert player_gives
    assert not vessel_gives


def test_power_power_crossing_gives_way_to_starboard_side() -> None:
    wind = WindState(from_deg=0, speed=10)
    own = make_descriptor(heading_deg=0.0, speed=4.0, kind="motorboat")
    to_starboard = make_descriptor(
        east=80.0, north=80.0, heading_deg=270.0, speed=4.0, kind="motorboat"
    )

    gives_way, reason = give_way_reason(own, to_starboard, wind)
    other_gives, _ = give_way_reason(to_starboard, own, wind)

    assert gives_way
    assert "starboard" in reason
    assert not other_gives


def test_power_power_head_on_both_alter_to_starboard() -> None:
    wind = WindState(from_deg=0, speed=10)
    own = make_descriptor(heading_deg=0.0, speed=4.0, kind="motorboat")
    oncoming = make_descriptor(
        north=200.0, heading_deg=180.0, speed=4.0, kind="motorboat"
    )

    own_gives, reason = give_way_reason(own, oncoming, wind)
    other_gives, _ = give_way_reason(oncoming, own, wind)

    assert own_gives
    assert other_gives
    assert "head-on" in reason


def test_rule_violation_triggers_red_penalty_and_counts() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=180, speed=2),
        wind=WindState(from_deg=90, speed=10),
        traffic=[TrafficVessel("S", "sailboat", 45.0, 0.0, 0.0, 2.0)],
    )

    sim.step(0.01)

    assert sim.penalty_timer > 0.0
    assert sim.penalty_count == 1
    assert sim.right_of_way[0].violation
    assert sim.actions.latest.startswith("rule violation:")


def test_stand_on_player_does_not_get_penalized_for_close_traffic() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, speed=2),
        wind=WindState(from_deg=90, speed=10),
        traffic=[TrafficVessel("S", "sailboat", 45.0, 0.0, 180.0, 2.0)],
    )

    sim.step(0.01)

    assert sim.penalty_timer == 0.0
    assert not sim.right_of_way[0].player_must_give_way


def test_encounter_geometry_reports_closing_and_cpa() -> None:
    own = make_descriptor(heading_deg=0.0, speed=4.0)
    oncoming = make_descriptor(north=400.0, heading_deg=180.0, speed=4.0)
    diverging = make_descriptor(north=400.0, heading_deg=0.0, speed=4.0)

    closing = encounter_geometry(own, oncoming)
    opening = encounter_geometry(own, diverging)

    assert closing.closing
    assert closing.cpa_distance < 10.0
    assert not opening.closing


# --- traffic behaviour ------------------------------------------------------


def test_traffic_vessel_advances_along_heading() -> None:
    vessel = TrafficVessel("T", "motorboat", 0.0, 0.0, 0.0, 2.0)

    vessel.advance(1.0)

    assert vessel.position_east == 0.0
    assert vessel.position_north == 36.0


def test_traffic_vessel_steers_toward_waypoints_and_loops() -> None:
    vessel = TrafficVessel(
        "T",
        "motorboat",
        0.0,
        0.0,
        0.0,
        4.0,
        waypoints=[(0.0, 300.0), (300.0, 300.0)],
    )
    player = BoatState(position_east=5000.0, position_north=5000.0)
    wind = WindState(from_deg=90, speed=10)

    for _ in range(400):
        update_traffic_vessel(vessel, player, wind, 0.05)

    assert vessel.waypoint_index > 0
    assert not vessel.avoiding


def test_give_way_traffic_alters_course_and_slows_down() -> None:
    wind = WindState(from_deg=90, speed=10)
    # Motorboat crossing ahead of the sailing player: power gives way to sail.
    vessel = TrafficVessel("M", "motorboat", -250.0, 250.0, 90.0, 5.0)
    player = BoatState(heading_deg=0, speed=4.0)

    original_heading = vessel.heading_deg
    for _ in range(30):
        update_traffic_vessel(vessel, player, wind, 0.05)

    assert vessel.avoiding
    assert signed_angle_diff(vessel.heading_deg, original_heading) > 5.0
    assert vessel.speed < vessel.cruise_speed


def test_stand_on_traffic_holds_course_and_speed() -> None:
    wind = WindState(from_deg=90, speed=10)
    # Starboard-tack sailboat crossing a port-tack player: it stands on.
    vessel = TrafficVessel("S", "sailboat", -250.0, 250.0, 90.0, 3.0)
    player = BoatState(heading_deg=180, speed=3.0)

    for _ in range(30):
        update_traffic_vessel(vessel, player, wind, 0.05)

    assert not vessel.avoiding
    assert vessel.heading_deg == 90.0
    assert vessel.speed == 3.0


# --- rocks and collisions ---------------------------------------------------


def test_rock_collision_detected_within_hull_distance() -> None:
    boat = BoatState(position_east=0.0, position_north=0.0)
    near = Rock(east=30.0, north=0.0, radius=26.0)
    far = Rock(east=300.0, north=0.0, radius=26.0)

    assert find_rock_collision(boat, [far]) is None
    assert find_rock_collision(boat, [far, near]) is near


def test_hitting_a_rock_grounds_and_stops_the_boat() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=ideal_sheet_for_angle(90)),
        wind=WindState(from_deg=90, speed=10),
        rocks=[Rock(east=0.0, north=120.0)],
    )

    run_steps(sim, 120)

    assert sim.grounded
    assert sim.wrecked
    assert sim.boat.speed < 0.5
    assert "you ran aground on a rock" in sim.actions.messages
    assert hypot(sim.boat.position_east, sim.boat.position_north - 120.0) < 80.0


def test_colliding_with_traffic_vessel_is_detected() -> None:
    boat = BoatState()
    near = TrafficVessel("M", "motorboat", 20.0, 10.0, 0.0, 0.0)

    assert find_vessel_collision(boat, [near]) is near

    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, speed=3),
        wind=WindState(from_deg=90, speed=10),
        traffic=[TrafficVessel("M", "motorboat", 0.0, 30.0, 0.0, 0.0)],
    )
    sim.step(0.05)

    assert sim.collided_vessel_id == "M"
    assert sim.wrecked


def test_respawn_resets_boat_and_failure_flags() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=175, speed=5, heel_deg=70, capsized=True),
        wind=WindState(from_deg=90, speed=10),
    )
    sim.grounded = True
    sim.collided_vessel_id = "M"
    sim.penalty_timer = 1.0

    sim.respawn_at(100.0, 200.0, 45.0)

    assert not sim.wrecked
    assert sim.boat.position_east == 100.0
    assert sim.boat.position_north == 200.0
    assert sim.boat.heading_deg == 45.0
    assert sim.boat.heel_deg == 0.0
    assert sim.penalty_timer == 0.0
    assert sim.actions.latest == "back on the water"


# --- stability and events ---------------------------------------------------


def test_sustained_overpowering_capsizes_the_boat() -> None:
    sim = SailingSimulation(
        boat=BoatState(heading_deg=0, sheet=0.15),
        wind=WindState(from_deg=82, speed=18),
    )

    run_steps(sim, 90)

    assert sim.boat.capsized
    assert sim.wrecked
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
