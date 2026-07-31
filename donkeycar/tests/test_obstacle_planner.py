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


def lane(yellow_x=200.0, white_x=350.0, width_px=150.0, white_right_of_yellow=True, primary_row_y=200.0,
         yellow_lost_frames=0):
    # gap (here 150px) must clear ObstaclePlanner's CORRIDOR_WIDTH_MIN_PX
    # default (90) or every detection is rejected by the corridor-
    # plausibility gate regardless of position -- real lane widths on the
    # actual car run ~150-330px (see obstacle_planner.py), so 150 is a
    # realistic default, not just "big enough to pass the gate".
    # yellow_lost_frames=0 default means "genuinely detected this frame"
    # -- most tests want a normally-tracking lane; staleness itself is
    # tested explicitly (see test_stale_yellow_estimate_is_not_acquired).
    return LaneGeometry(yellow_x=yellow_x, white_x=white_x, width_px=width_px,
                         white_right_of_yellow=white_right_of_yellow, primary_row_y=primary_row_y,
                         yellow_lost_frames=yellow_lost_frames)


def det_in_lane(distance_mm=2000.0, cx=275, w=20, h=40):
    # our lane spans [200,350] by default (narrowed to [220,330] at the
    # default -20 corridor margin) -- center it inside that
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
    neighbor_geom = lane(yellow_x=200.0, white_x=50.0, white_right_of_yellow=False)

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
    d = p.step(det_in_lane(distance_mm=150), lane())  # below CONE_CRITICAL_DISTANCE_MM -- instant, no confirm needed  # instantly emergency-close
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
    p.step(det_in_lane(distance_mm=150), lane())
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
    p.step(det_in_lane(distance_mm=150), lane())
    assert p.state == PlannerState.SAFE_STOP
    for _ in range(20):
        p.step(det_in_lane(distance_mm=150), lane())
    # frames_in_state should have accumulated normally, not been reset
    # every tick by a spurious self-transition
    assert p._frames_in_state == 20


def test_safe_stop_does_not_recover_when_disabled():
    cfg = make_cfg(PLANNER_SAFE_STOP_AUTO_RECOVER=False)
    p = ObstaclePlanner(cfg)
    p.step(det_in_lane(distance_mm=150), lane())
    assert p.state == PlannerState.SAFE_STOP
    run_n(p, 100, None, lane())
    assert p.state == PlannerState.SAFE_STOP, "auto-recover disabled -- must require explicit reset"


def test_emergency_requires_confirmation_unless_critical():
    """A non-critical emergency-distance reading (below CONE_EMERGENCY_
    DISTANCE_MM but at/above CONE_CRITICAL_DISTANCE_MM) must be confirmed
    over CONE_EMERGENCY_CONFIRM_FRAMES consecutive frames before
    triggering SAFE_STOP - added after avoid_cone's frame-162 anomaly (an
    unexplained single-frame emergency reading right after completing a
    maneuver)."""
    cfg = make_cfg(CONE_EMERGENCY_CONFIRM_FRAMES=2, CONE_CRITICAL_DISTANCE_MM=200)
    p = ObstaclePlanner(cfg)
    p.step(det_in_lane(distance_mm=300), lane())  # emergency but not critical -- 1st frame
    assert p.state != PlannerState.SAFE_STOP, "a single non-critical emergency frame must not trigger SAFE_STOP"
    p.step(det_in_lane(distance_mm=300), lane())  # 2nd consecutive frame -- now confirmed
    assert p.state == PlannerState.SAFE_STOP


def test_critical_distance_triggers_emergency_instantly():
    cfg = make_cfg(CONE_CRITICAL_DISTANCE_MM=200)
    p = ObstaclePlanner(cfg)
    d = p.step(det_in_lane(distance_mm=150), lane())  # below critical -- must not wait for confirmation
    assert p.state == PlannerState.SAFE_STOP
    assert d.throttle_scale == 0.0


def test_lateral_sweep_past_is_not_treated_as_in_path():
    """A cone that's laterally sweeping toward one edge of the frame as it
    closes (the real signature of an object in the neighboring lane being
    driven past - see cone_negative2 in this project's replay findings)
    must not be classified as in-path, even though its raw screen
    position momentarily overlaps the corridor."""
    cfg = make_cfg(CONE_LATERAL_HISTORY_FRAMES=10, CONE_LATERAL_SWEEP_REJECT_PX=25,
                   CONE_LATERAL_MIN_HEIGHT_PX=45)
    p = ObstaclePlanner(cfg)
    geom = lane()
    # bbox center sweeps steadily from inside the corridor toward the edge
    # as bbox height grows (closing distance) -- mirrors the real
    # cone_negative2 numbers (cx 143->31 over ~45 frames, bbox_h 42->200+)
    cx_start, h_start = 275, 50
    for i in range(15):
        cx = cx_start - i * 6
        h = h_start + i * 4
        det = Detection(bbox=BBox(int(cx - 20 / 2), 150, 20, h), area_px=20 * h,
                         distance_mm=2000 - i * 80, distance_valid=True, distance_source='depth')
        d = p.step(det, geom)
    assert p.state in (PlannerState.FOLLOW_LANE, PlannerState.OBJECT_WATCH), \
        f"sweeping cone should never reach CONFIRMED_IN_PATH, got {p.state}"
    assert d.cone_in_path is False


def test_stable_centered_approach_is_not_rejected_by_lateral_check():
    """The complementary case: a cone that stays laterally stable as it
    closes (real cone_approach_right2 signature: bbox center held ~190-
    231px, max 10-frame drift 18px once bbox_h>=45) must NOT be vetoed by
    the lateral-sweep check - it should still be able to reach
    CONFIRMED_IN_PATH."""
    cfg = make_cfg(CONE_LATERAL_HISTORY_FRAMES=10, CONE_LATERAL_SWEEP_REJECT_PX=25,
                   CONE_LATERAL_MIN_HEIGHT_PX=45, CONE_COMMIT_DISTANCE_MM=1200)
    p = ObstaclePlanner(cfg)
    geom = lane()
    cx, h = 275, 50
    d = None
    for i in range(15):
        h += 10  # closing distance, growing blob, near-stable center
        det = Detection(bbox=BBox(int(cx - 20 / 2), 150, 20, h), area_px=20 * h,
                         distance_mm=max(2000 - i * 150, 900), distance_valid=True, distance_source='depth')
        d = p.step(det, geom)
    # must have progressed well past OBJECT_WATCH -- proves the lateral
    # check doesn't block a genuine stable/centered approach
    assert p.state not in (PlannerState.FOLLOW_LANE, PlannerState.OBJECT_WATCH), f"got {p.state}"


def test_stale_yellow_estimate_is_not_acquired():
    """The exact real bug: yellow_x present and plausible, but
    yellow_lost_frames says it's stale (carried-forward telemetry, not a
    genuine detection) - must not count as acquired. Mirrors
    tub_122_26-07-30's real failure (yellow frozen at one value for 45+
    frames after set_lane('left'))."""
    cfg = make_cfg(YELLOW_FRESHNESS_MAX_FRAMES=4)
    p = ObstaclePlanner(cfg)
    stale_geom = lane(yellow_x=200.0, white_x=None, white_right_of_yellow=False, yellow_lost_frames=45)
    assert p._lane_acquired(stale_geom) is False


def test_fresh_yellow_only_is_acquired_even_without_white():
    """The actual fix: yellow alone, fresh and plausibly-placed, is
    sufficient - white's absence must not block acquisition (real
    on-car frames show the far white edge is never in view from the
    right lane at all for a right-to-left switch)."""
    cfg = make_cfg(YELLOW_FRESHNESS_MAX_FRAMES=4)
    p = ObstaclePlanner(cfg)
    fresh_geom = lane(yellow_x=200.0, white_x=None, width_px=150.0,
                       white_right_of_yellow=False, yellow_lost_frames=0)
    assert p._lane_acquired(fresh_geom) is True


def test_yellow_within_coast_frames_still_counts_as_fresh():
    cfg = make_cfg(YELLOW_FRESHNESS_MAX_FRAMES=4)
    p = ObstaclePlanner(cfg)
    coasting_geom = lane(yellow_x=200.0, white_x=None, white_right_of_yellow=False, yellow_lost_frames=3)
    assert p._lane_acquired(coasting_geom) is True


def test_implausible_yellow_offset_is_not_acquired():
    """Fresh yellow, but the implied lane-center estimate is nowhere
    near image center (a sanity bound, not a "must already be
    centered" requirement)."""
    cfg = make_cfg(YELLOW_FRESHNESS_MAX_FRAMES=4, YELLOW_OFFSET_PLAUSIBLE_PX=200)
    p = ObstaclePlanner(cfg)
    # yellow_x=10, width=150, white_right_of_yellow=False -> sign=-1,
    # estimated_center = 10 - 75 = -65; image_center default ~213 ->
    # offset ~278, past the 200px plausibility bound
    implausible_geom = lane(yellow_x=10.0, white_x=None, width_px=150.0,
                             white_right_of_yellow=False, yellow_lost_frames=0)
    assert p._lane_acquired(implausible_geom) is False


def test_full_switch_cycle_using_yellow_only_after_switch():
    """End-to-end: white genuinely unavailable for the ENTIRE neighbor-
    lane portion (not just briefly) - the fixed planner must still
    complete switch -> hold -> return using yellow alone, unlike the
    real tub_122 failure this fix targets."""
    cfg = make_cfg()
    p = ObstaclePlanner(cfg)
    geom = lane(white_right_of_yellow=True)
    run_n(p, cfg.CONE_DETECT_CONFIRM_FRAMES, det_in_lane(distance_mm=1000), geom)
    p.step(det_in_lane(distance_mm=1000), geom)
    p.step(det_in_lane(distance_mm=1000), geom)
    d = run_n(p, cfg.PLANNER_PREPARE_MIN_FRAMES, det_in_lane(distance_mm=900), geom)
    assert p.state == PlannerState.SWITCH_TO_NEIGHBOR_LANE

    # neighbor lane: yellow only, fresh every frame, white never found
    neighbor_geom = lane(yellow_x=200.0, white_x=None, width_px=150.0,
                         white_right_of_yellow=False, yellow_lost_frames=0)
    d = run_n(p, cfg.CONE_LANE_ACQUIRE_CONFIRM_FRAMES, det_in_lane(distance_mm=800), neighbor_geom)
    assert p.state == PlannerState.HOLD_UNTIL_CLEAR
    assert d.requested_lane == 'left'


def test_implausible_corridor_width_is_not_trusted():
    """A corridor whose width falls outside CORRIDOR_WIDTH_MIN/MAX_PX is
    not trusted for path classification, even if the cone's raw position
    would otherwise overlap it."""
    cfg = make_cfg(CORRIDOR_WIDTH_MIN_PX=90, CORRIDOR_WIDTH_MAX_PX=340)
    p = ObstaclePlanner(cfg)
    implausibly_narrow = lane(yellow_x=200.0, white_x=230.0, width_px=30.0)  # 30px gap, below the 90px floor
    run_n(p, 20, det_in_lane(distance_mm=1000, cx=215), implausibly_narrow)
    assert p.state == PlannerState.FOLLOW_LANE


def test_require_valid_depth_to_switch_blocks_bbox_only_commit():
    """CONE_REQUIRE_VALID_DEPTH_TO_SWITCH=True (the default) must not let
    a bbox-height-only (no real depth) detection commit to
    CONFIRMED_IN_PATH, even if it's past the bbox-height commit
    threshold."""
    cfg = make_cfg(CONE_REQUIRE_VALID_DEPTH_TO_SWITCH=True, CONE_COMMIT_BBOX_HEIGHT_PX=90)
    p = ObstaclePlanner(cfg)
    geom = lane()
    det = Detection(bbox=BBox(265, 150, 20, 120), area_px=20 * 120,
                     distance_mm=None, distance_valid=False, distance_source='bbox_height')
    run_n(p, 20, det, geom)
    assert p.state == PlannerState.OBJECT_WATCH, \
        "must not commit past OBJECT_WATCH without valid depth when the gate is enabled"


def test_decision_always_populated_even_in_disabled_mode():
    cfg = make_cfg()
    cfg.OBSTACLE_AVOIDANCE_MODE = 'disabled'
    p = ObstaclePlanner(cfg)
    d = p.step(det_in_lane(distance_mm=1000), lane())
    assert d is not None
    assert d.state is not None
