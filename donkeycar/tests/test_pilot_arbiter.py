"""
Tests for pilot_arbiter.py -- the Vehicle-part glue wiring LaneFollower,
ConeDetector, and ObstaclePlanner together. Focused on the mode dispatch
contract: observe/shadow must NEVER touch driving (no set_lane(), no
throttle change), active must apply both once the planner actually
requests them. Detection/planning correctness is covered by
test_cone_detector.py/test_obstacle_planner.py; this file assumes those
work and only checks the adapter logic.
"""
from types import SimpleNamespace

import cv2
import numpy as np
from simple_pid import PID

from donkeycar.parts.cone_detector import ConeDetector
from donkeycar.parts.lane_follower3 import LaneFollower
from donkeycar.parts.obstacle_planner import ObstaclePlanner
from donkeycar.parts.pilot_arbiter import PilotArbiter

W, H = 426, 240
ORANGE_HSV = (8, 180, 200)


def make_cfg(mode):
    cfg = SimpleNamespace(
        OVERLAY_IMAGE=False, THROTTLE_INITIAL=0.15, THROTTLE_STEP=0.05,
        THROTTLE_MAX=0.35, THROTTLE_MIN=0.15, SCAN_Y=168, SCAN_HEIGHT=20,
        WHITE_RIGHT_OF_YELLOW=True,
        OBSTACLE_AVOIDANCE_MODE=mode,
        # confirm/commit fast so a short test can reach SWITCH quickly
        CONE_DETECT_CONFIRM_FRAMES=2, CONE_CLEAR_CONFIRM_FRAMES=2,
        CONE_LANE_ACQUIRE_CONFIRM_FRAMES=2,
        CONE_COMMIT_DISTANCE_MM=1200, CONE_COMMIT_BBOX_HEIGHT_PX=50,
        CONE_EMERGENCY_BBOX_HEIGHT_PX=1000,  # unreachable -- keep this test out of SAFE_STOP
        PLANNER_PREPARE_MIN_FRAMES=1, PLANNER_HOLD_MIN_FRAMES=1,
        PLANNER_MANEUVER_TIMEOUT_FRAMES=200,
        CORRIDOR_SAFETY_MARGIN_PX=0,
        CORRIDOR_WIDTH_MIN_PX=50,   # this file's synthetic painted lane gap is narrower than the real-world default
        # this file has no synthetic depth array (arbiter.run(..., None, ...)) --
        # the depth-required gate is about detection accuracy, not mode
        # dispatch, which is what this file actually tests
        CONE_REQUIRE_VALID_DEPTH_TO_SWITCH=False,
    )
    return cfg


def cone_frame():
    """A frame with a plausible yellow dash + white edge at the LaneFollower
    scan row (y=168, SCAN_HEIGHT=20 -> band ~158-178), plus a cone painted
    well above that band so its color can't bleed into lane detection.
    Without real lane geometry, ObstaclePlanner's SWITCH_TO_NEIGHBOR_LANE
    gate (_lane_acquired) never passes -- this is required for the
    active-mode test to actually reach a lane switch, not just an
    artifact of this test file."""
    frame = np.full((H, W, 3), 60, dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[80:160, 200:230] = ORANGE_HSV       # cone -- centered, tall, close
    hsv[150:190, 180:205] = (25, 150, 180)  # yellow dash, within YELLOW_HSV default range
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    frame[150:190, 260:285] = (240, 240, 240)  # white edge, right of yellow
    return frame


def make_stack(mode):
    cfg = make_cfg(mode)
    pid = PID(-0.01, 0.0, -0.00035)
    lf = LaneFollower(pid, cfg)
    detector = ConeDetector(cfg)
    planner = ObstaclePlanner(cfg)
    arbiter = PilotArbiter(cfg, lf, detector, planner)
    return lf, arbiter


def run_ticks(lf, arbiter, n):
    frame = cone_frame()
    result = None
    for _ in range(n):
        lf.run(frame)
        result = arbiter.run(frame, None, lf.steering, lf.throttle,
                              lf.last_yellow_x, lf.last_white_x, lf.lane_width_px)
    return result


def test_observe_mode_never_calls_set_lane_or_changes_throttle():
    lf, arbiter = make_stack('observe')
    starting_lane = lf.current_lane
    result = run_ticks(lf, arbiter, 15)
    out_steering, out_throttle = result[0], result[1]
    assert lf.current_lane == starting_lane, "observe mode must never call set_lane()"
    assert out_throttle == lf.throttle, "observe mode must never scale throttle"
    assert out_steering == lf.steering


def test_shadow_mode_never_calls_set_lane_or_changes_throttle():
    lf, arbiter = make_stack('shadow')
    starting_lane = lf.current_lane
    result = run_ticks(lf, arbiter, 15)
    out_throttle = result[1]
    assert lf.current_lane == starting_lane, "shadow mode must never call set_lane()"
    assert out_throttle == lf.throttle, "shadow mode must never scale throttle"


def test_active_mode_eventually_calls_set_lane_and_scales_throttle():
    """This synthetic frame only paints ONE white-colored patch (on the
    right, for the right lane); with the yellow-anchor acquisition fix
    the planner now completes the full cycle quickly and repeats it
    indefinitely against this same static frame - so this test checks
    'did active mode dispatch a real set_lane() call and scale
    throttle at some point', not the state at one arbitrarily-chosen
    tick count (which depends on cycle phase). Full-maneuver validation
    against real recorded lane geometry is scripts/replay_cone_planner.py."""
    lf, arbiter = make_stack('active')
    frame = cone_frame()
    starting_lane = lf.current_lane
    lanes_seen = set()
    scaled_down = False
    for _ in range(40):
        lf.run(frame)
        result = arbiter.run(frame, None, lf.steering, lf.throttle,
                              lf.last_yellow_x, lf.last_white_x, lf.lane_width_px)
        lanes_seen.add(lf.current_lane)
        if result[1] < lf.throttle:
            scaled_down = True
    assert lanes_seen != {starting_lane}, "active mode should have switched lanes at some point"
    assert scaled_down, "throttle should be scaled down at some point during the maneuver"


def test_active_mode_floors_throttle_instead_of_compounding_scales():
    """The exact real bug (cone_test4, 2026-07-30): LaneFollower's own
    confidence-based speed policy already drops pilot_throttle to
    THROTTLE_MIN as soon as detection confidence falls (which happens
    right after a lane switch) - PLANNER_SWITCH_THROTTLE_SCALE=0.5 then
    multiplied THAT already-reduced value, not nominal cruising
    throttle, landing on throttle=0.075 (0.15*0.5) for the entire
    maneuver - too low to actually move the car (recorded cone distance
    never changed for the full 300-frame/15s timeout). The floor must
    win over the compounded (lower) product."""
    from donkeycar.parts.obstacle_types import PlannerDecision, PlannerState

    class FakeDetector:
        def detect(self, cam_img, depth_img):
            return None, {}

    class FakePlanner:
        mode = None
        def step(self, detection, geometry, raw_steering=None):
            return PlannerDecision(state=PlannerState.SWITCH_TO_NEIGHBOR_LANE, reason='test',
                                    requested_lane=None, throttle_scale=0.5, cone_in_path=False)

    from donkeycar.parts.obstacle_types import RolloutMode
    cfg = make_cfg('active')
    cfg.PLANNER_MIN_MANEUVER_THROTTLE = 0.15
    pid = PID(-0.01, 0.0, -0.00035)
    lf = LaneFollower(pid, cfg)
    fake_planner = FakePlanner()
    fake_planner.mode = RolloutMode.ACTIVE
    arbiter = PilotArbiter(cfg, lf, FakeDetector(), fake_planner)

    frame = cone_frame()
    # LaneFollower's own low-confidence crawl -- already-reduced input
    low_confidence_pilot_throttle = 0.15
    result = arbiter.run(frame, None, 0.0, low_confidence_pilot_throttle, None, None, 150.0)
    out_throttle = result[1]

    compounded = low_confidence_pilot_throttle * 0.5  # = 0.075, the actual observed bug value
    assert out_throttle > compounded, "throttle must not be left at the stalling compounded value"
    assert out_throttle == 0.15, "floored at PLANNER_MIN_MANEUVER_THROTTLE"


def test_safe_stop_throttle_scale_is_not_floored():
    """throttle_scale=0.0 (SAFE_STOP) must still mean a real stop -- the
    floor only applies to genuine in-between maneuver scaling."""
    from donkeycar.parts.obstacle_types import PlannerDecision, PlannerState, RolloutMode

    class FakeDetector:
        def detect(self, cam_img, depth_img):
            return None, {}

    class FakePlanner:
        def step(self, detection, geometry, raw_steering=None):
            return PlannerDecision(state=PlannerState.SAFE_STOP, reason='test',
                                    requested_lane=None, throttle_scale=0.0, cone_in_path=False)

    cfg = make_cfg('active')
    cfg.PLANNER_MIN_MANEUVER_THROTTLE = 0.15
    pid = PID(-0.01, 0.0, -0.00035)
    lf = LaneFollower(pid, cfg)
    fake_planner = FakePlanner()
    fake_planner.mode = RolloutMode.ACTIVE
    arbiter = PilotArbiter(cfg, lf, FakeDetector(), fake_planner)

    frame = cone_frame()
    result = arbiter.run(frame, None, 0.0, 0.30, None, None, 150.0)
    assert result[1] == 0.0, "SAFE_STOP must still fully stop the car, not get floored"


def test_returned_debug_fields_are_populated_when_cone_present():
    lf, arbiter = make_stack('observe')
    result = run_ticks(lf, arbiter, 3)
    (out_steering, out_throttle, state, reason, requested_lane, cone_in_path,
     bbox_x, bbox_y, bbox_w, bbox_h, distance_mm, distance_valid, distance_source) = result
    assert isinstance(state, str) and state
    assert isinstance(reason, str) and reason
    assert bbox_w is not None and bbox_h is not None
    assert distance_source in ('depth', 'bbox_height')
