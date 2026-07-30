"""
Unit tests for obstacle_planner.py's state machine. Synthetic
Detection/LaneGeometry fixtures only -- this validates the FSM's
transition LOGIC (hysteresis, perception-gating, timeouts, emergency
override), not real-world detection/depth accuracy, which is validated
separately against real tubs via scripts/replay_cone_planner.py.
"""
from types import SimpleNamespace

import pytest

from donkeycar.parts.obstacle_planner import ObstaclePlanner
from donkeycar.parts.obstacle_types import BBox, Detection, LaneGeometry, PlannerState


def make_cfg(**overrides):
    cfg = SimpleNamespace(
        CONE_DETECT_CONFIRM_FRAMES=3,
        CONE_CLEAR_CONFIRM_FRAMES=3,
        CONE_LANE_ACQUIRE_CONFIRM_FRAMES=3,
        CONE_WATCH_DISTANCE_MM=3000,
        CONE_COMMIT_DISTANCE_MM=1200,
        CONE_EMERGENCY_DISTANCE_MM=400,
        PLANNER_PREPARE_MIN_FRAMES=2,
        PLANNER_HOLD_MIN_FRAMES=2,
        PLANNER_MANEUVER_TIMEOUT_FRAMES=50,
        PLANNER_SAFE_STOP_RECOVER_FRAMES=3,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def lane(yellow_x=200.0, white_x=260.0, width_px=60.0, white_right_of_yellow=True, primary_row_y=200.0):
    return LaneGeometry(yellow_x=yellow_x, white_x=white_x, width_px=width_px,
                         white_right_of_yellow=white_right_of_yellow, primary_row_y=primary_row_y)


def det_in_lane(distance_mm=2000.0, cx=230, w=20, h=40):
    # our lane spans [200,260] by default -- center it inside that
    return Detection(bbox=BBox(int(cx - w / 2), 150, w, h), area_px=w * h,
                      distance_mm=distance_mm, distance_valid=True, distance_source='depth')


def det_other_lane(distance_mm=2000.0, cx=100, w=20, h=40):
    return Detection(bbox=BBox(int(cx - w / 2), 150, w, h), area_px=w * h,
                      distance_mm=distance_mm, distance_valid=True, distance_source='depth')


def run_n(planner, n, detection, geometry):
    decision = None
    for _ in range(n):
        decision = planner.step(detection, geometry)
    return decision


def test_stays_in_follow_lane_when_no_cone():
    p = ObstaclePlanner(make_cfg())
    d = run_n(p, 10, None, lane())
    assert p.state == PlannerState.FOLLOW_LANE
    assert d.requested_lane is None
    assert d.throttle_scale == 1.0


def test_relevant_cone_far_away_only_reaches_object_watch():
    p = ObstaclePlanner(make_cfg())
    d = run_n(p, 5, det_in_lane(distance_mm=2800), lane())
    assert p.state == PlannerState.OBJECT_WATCH
    assert d.requested_lane is None  # no lane request yet -- just watching


def test_cone_outside_path_never_triggers_watch():
    p = ObstaclePlanner(make_cfg())
    run_n(p, 20, det_other_lane(distance_mm=1000), lane())
    assert p.state == PlannerState.FOLLOW_LANE, \
        "a cone outside the current lane's corridor must never be treated as blocking"


def test_single_noisy_frame_does_not_flip_state():
    cfg = make_cfg(CONE_DETECT_CONFIRM_FRAMES=3)
    p = ObstaclePlanner(cfg)
    p.step(det_in_lane(distance_mm=2800), lane())  # 1 relevant frame
    p.step(None, lane())                            # noise: miss
    assert p.state == PlannerState.FOLLOW_LANE
    # watch_count should have reset on the miss, not accumulated across it
    for _ in range(2):
        p.step(det_in_lane(distance_mm=2800), lane())
    assert p.state == PlannerState.FOLLOW_LANE, "2 consecutive relevant frames < confirm threshold of 3"


def test_full_happy_path_switch_hold_return():
    cfg = make_cfg()
    p = ObstaclePlanner(cfg)
    geom = lane(white_right_of_yellow=True)  # currently in 'right' lane
    neighbor_geom = lane(yellow_x=50.0, white_x=-10.0, white_right_of_yellow=False)

    # 1) approach: confirm in-path (3 ticks to cross detect_confirm_frames),
    # then a 4th tick evaluates OBJECT_WATCH's own commit-distance check.
    run_n(p, 3, det_in_lane(distance_mm=2500), geom)
    assert p.state == PlannerState.OBJECT_WATCH
    p.step(det_in_lane(distance_mm=1000), geom)   # within commit distance
    assert p.state == PlannerState.CONFIRMED_IN_PATH

    d = p.step(det_in_lane(distance_mm=900), geom)  # CONFIRMED_IN_PATH -> PREPARE_SLOW (immediate)
    assert p.state == PlannerState.PREPARE_SLOW
    assert d.throttle_scale < 1.0

    # 2) prepare_min_frames elapse (ticks spent IN PREPARE_SLOW, not counting
    # the transition tick itself) with lane geometry available -> switch
    d = run_n(p, cfg.PLANNER_PREPARE_MIN_FRAMES, det_in_lane(distance_mm=850), geom)
    assert p.state == PlannerState.SWITCH_TO_NEIGHBOR_LANE
    assert d.requested_lane == 'left'

    # 3) neighbor lane geometry acquired for N frames -> HOLD_UNTIL_CLEAR
    d = run_n(p, cfg.CONE_LANE_ACQUIRE_CONFIRM_FRAMES, det_in_lane(distance_mm=800), neighbor_geom)
    assert p.state == PlannerState.HOLD_UNTIL_CLEAR
    assert d.requested_lane == 'left'

    # 4) cone recedes past its closest point by a real margin, sustained -> RETURN.
    # Still fed neighbor_geom here: LaneFollower hasn't been switched back yet
    # in this simulation (set_lane(original) only happens once RETURN begins).
    # Step one tick at a time (rather than pre-computing an exact tick count)
    # until the transition fires, so this isn't brittle to the exact
    # hold_min_frames/clear_confirm_frames arithmetic -- in real integration,
    # set_lane(original) only takes effect starting the tick AFTER the
    # planner requests it, so once RETURN fires here, subsequent ticks must
    # simulate the geometry actually having switched (step 5, below).
    d = None
    for _ in range(50):
        d = p.step(None, neighbor_geom)
        if p.state != PlannerState.HOLD_UNTIL_CLEAR:
            break
    assert p.state == PlannerState.RETURN_TO_ORIGINAL_LANE
    assert d.requested_lane == 'right'

    # 5) original lane geometry re-acquired (now fed geom, representing
    # LaneFollower having actually switched back per requested_lane above)
    # -> FOLLOW_LANE
    d = run_n(p, cfg.CONE_LANE_ACQUIRE_CONFIRM_FRAMES, None, geom)
    assert p.state == PlannerState.FOLLOW_LANE
    assert d.requested_lane is None
    assert d.throttle_scale == 1.0


def test_cone_clears_during_prepare_aborts_to_follow_lane():
    cfg = make_cfg()
    p = ObstaclePlanner(cfg)
    geom = lane()
    run_n(p, cfg.CONE_DETECT_CONFIRM_FRAMES, det_in_lane(distance_mm=1000), geom)
    assert p.state == PlannerState.OBJECT_WATCH
    p.step(det_in_lane(distance_mm=1000), geom)  # OBJECT_WATCH -> CONFIRMED_IN_PATH
    assert p.state == PlannerState.CONFIRMED_IN_PATH
    p.step(det_in_lane(distance_mm=1000), geom)  # CONFIRMED_IN_PATH -> PREPARE_SLOW
    assert p.state == PlannerState.PREPARE_SLOW
    # cone clears (goes out of path) before commit -- must abort, not switch lanes
    d = run_n(p, cfg.CONE_CLEAR_CONFIRM_FRAMES, None, geom)
    assert p.state == PlannerState.FOLLOW_LANE
    assert d.requested_lane is None


def test_emergency_close_cone_forces_safe_stop():
    p = ObstaclePlanner(make_cfg())
    d = p.step(det_in_lane(distance_mm=300), lane())  # instantly emergency-close
    assert p.state == PlannerState.SAFE_STOP
    assert d.throttle_scale == 0.0
    assert d.requested_lane is None


def test_maneuver_timeout_forces_safe_stop():
    cfg = make_cfg(PLANNER_MANEUVER_TIMEOUT_FRAMES=5, CONE_LANE_ACQUIRE_CONFIRM_FRAMES=100)
    p = ObstaclePlanner(cfg)
    geom = lane()
    run_n(p, cfg.CONE_DETECT_CONFIRM_FRAMES, det_in_lane(distance_mm=1000), geom)
    p.step(det_in_lane(distance_mm=1000), geom)  # -> CONFIRMED_IN_PATH
    p.step(det_in_lane(distance_mm=1000), geom)  # -> PREPARE_SLOW
    run_n(p, cfg.PLANNER_PREPARE_MIN_FRAMES, det_in_lane(distance_mm=900), geom)
    assert p.state == PlannerState.SWITCH_TO_NEIGHBOR_LANE
    # neighbor lane never acquires (acquire threshold impossibly high) --
    # must eventually safe-stop rather than hang in the maneuver forever
    d = run_n(p, 10, det_in_lane(distance_mm=900), geom)
    assert p.state == PlannerState.SAFE_STOP


def test_safe_stop_auto_recovers_when_clean():
    cfg = make_cfg(PLANNER_SAFE_STOP_RECOVER_FRAMES=3)
    p = ObstaclePlanner(cfg)
    p.step(det_in_lane(distance_mm=300), lane())
    assert p.state == PlannerState.SAFE_STOP
    d = run_n(p, 3, None, lane())
    assert p.state == PlannerState.FOLLOW_LANE
    assert d.throttle_scale == 1.0


def test_emergency_cone_sustained_in_safe_stop_is_not_a_repeated_transition():
    """A cone that stays emergency-close for many consecutive frames (e.g.
    a stationary/parked cone right in front of the car) must not cause
    SAFE_STOP -> SAFE_STOP self-transitions every tick -- found via real
    tub replay (cone_static_right), where this previously reset
    _frames_in_state to 0 every frame and produced hundreds of identical
    log lines."""
    p = ObstaclePlanner(make_cfg())
    p.step(det_in_lane(distance_mm=300), lane())
    assert p.state == PlannerState.SAFE_STOP
    for _ in range(20):
        p.step(det_in_lane(distance_mm=300), lane())
    # frames_in_state should have accumulated normally, not been reset
    # every tick by a spurious self-transition
    assert p._frames_in_state == 20


def test_safe_stop_does_not_recover_when_disabled():
    cfg = make_cfg(PLANNER_SAFE_STOP_AUTO_RECOVER=False)
    p = ObstaclePlanner(cfg)
    p.step(det_in_lane(distance_mm=300), lane())
    assert p.state == PlannerState.SAFE_STOP
    run_n(p, 100, None, lane())
    assert p.state == PlannerState.SAFE_STOP, "auto-recover disabled -- must require explicit reset"


def test_decision_always_populated_even_in_disabled_mode():
    cfg = make_cfg()
    cfg.OBSTACLE_AVOIDANCE_MODE = 'disabled'
    p = ObstaclePlanner(cfg)
    d = p.step(det_in_lane(distance_mm=1000), lane())
    assert d is not None
    assert d.state is not None
