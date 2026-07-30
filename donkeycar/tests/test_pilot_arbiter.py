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
    hsv[160:176, 185:200] = (25, 150, 180)  # yellow dash, within YELLOW_HSV default range
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    frame[160:176, 260:280] = (240, 240, 240)  # white edge, right of yellow
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
    lf, arbiter = make_stack('active')
    starting_lane = lf.current_lane
    result = run_ticks(lf, arbiter, 15)
    state = result[2]
    assert lf.current_lane != starting_lane, \
        f"active mode should have switched lanes by now (planner state={state})"
    out_throttle = result[1]
    assert out_throttle <= lf.throttle, "throttle should be scaled down during the maneuver"


def test_returned_debug_fields_are_populated_when_cone_present():
    lf, arbiter = make_stack('observe')
    result = run_ticks(lf, arbiter, 3)
    (out_steering, out_throttle, state, reason, requested_lane, cone_in_path,
     bbox_x, bbox_y, bbox_w, bbox_h, distance_mm, distance_valid, distance_source) = result
    assert isinstance(state, str) and state
    assert isinstance(reason, str) and reason
    assert bbox_w is not None and bbox_h is not None
    assert distance_source in ('depth', 'bbox_height')
