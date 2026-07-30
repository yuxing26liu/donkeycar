"""
Tests for LaneFollower.set_lane() (donkeycar/parts/lane_follower3.py) -
the runtime lane-switch interface added for cone avoidance on the
cone-dodger3000 branch.

These tests only exercise the switch/reset contract itself (no cone
detector or planner involved yet): idempotence, which state resets vs.
which state is preserved, and that normal run() behavior survives a
lane switch without raising. They do not assert anything about
detection accuracy - that's covered separately for the new cone
detector.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from simple_pid import PID

from donkeycar.parts.lane_follower3 import LaneFollower


def make_cfg(**overrides):
    cfg = SimpleNamespace(
        OVERLAY_IMAGE=False,
        THROTTLE_INITIAL=0.15,
        THROTTLE_STEP=0.05,
        THROTTLE_MAX=0.35,
        THROTTLE_MIN=0.15,
        SCAN_Y=168,
        SCAN_HEIGHT=20,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_follower(**cfg_overrides):
    cfg = make_cfg(**cfg_overrides)
    pid = PID(-0.01, 0.0, -0.00035)
    return LaneFollower(pid, cfg)


def test_default_lane_matches_config():
    lf = make_follower(WHITE_RIGHT_OF_YELLOW=True)
    assert lf.current_lane == 'right'
    lf2 = make_follower(WHITE_RIGHT_OF_YELLOW=False)
    assert lf2.current_lane == 'left'


def test_invalid_lane_raises():
    lf = make_follower()
    with pytest.raises(ValueError):
        lf.set_lane('sideways')


def test_set_lane_switches_and_resets_invalidated_state():
    lf = make_follower(WHITE_RIGHT_OF_YELLOW=True)
    # simulate an actively-tracking follower before the switch
    for tracker in lf.yellow_trackers + lf.white_trackers:
        tracker.tracked_position = 123.0
        tracker.smoothed_position = 123.0
    # in real operation, run() keeps self.lane_width_px in sync with
    # row_lane_width_px[0] every frame (see run()'s LANE_WIDTH_PX comment) -
    # set both here to match that invariant.
    lf.lane_width_px = 222.0
    lf.row_lane_width_px = [222.0 for _ in lf.scan_rows]
    lf.row_center_offset = [17.0 for _ in lf.scan_rows]
    lf._last_position = 150.0
    lf.last_yellow_x = 140.0
    lf.last_white_x = 200.0
    lf.lost_frames = 5

    lf.set_lane('left')

    assert lf.current_lane == 'left'
    assert lf.white_right_of_yellow is False
    for tracker in lf.yellow_trackers + lf.white_trackers:
        assert tracker.tracked_position is None
        assert tracker.smoothed_position is None
    # seeded from the last known width, not corrupted/misshapen
    assert lf.row_lane_width_px == [222.0 for _ in lf.scan_rows]
    assert lf.row_center_offset == [0.0 for _ in lf.scan_rows]
    assert lf._last_position is None
    assert lf.last_yellow_x is None
    assert lf.last_white_x is None
    assert lf.lost_frames == 0


def test_set_lane_same_lane_is_a_true_noop():
    lf = make_follower(WHITE_RIGHT_OF_YELLOW=True)
    for tracker in lf.yellow_trackers + lf.white_trackers:
        tracker.tracked_position = 77.0
        tracker.smoothed_position = 77.0
    lf.row_center_offset = [9.0 for _ in lf.scan_rows]
    lf._last_position = 88.0

    lf.set_lane('right')  # already 'right' -> must not touch anything

    for tracker in lf.yellow_trackers + lf.white_trackers:
        assert tracker.tracked_position == 77.0
        assert tracker.smoothed_position == 77.0
    assert lf.row_center_offset == [9.0 for _ in lf.scan_rows]
    assert lf._last_position == 88.0


def test_repeated_alternating_calls_do_not_corrupt_state():
    lf = make_follower(WHITE_RIGHT_OF_YELLOW=True)
    n_rows = len(lf.scan_rows)
    for i in range(10):
        lf.set_lane('left' if i % 2 == 0 else 'right')
        assert len(lf.row_lane_width_px) == n_rows
        assert len(lf.row_center_offset) == n_rows
        assert all(np.isfinite(w) for w in lf.row_lane_width_px)
        assert all(np.isfinite(o) for o in lf.row_center_offset)
    # ended on an odd number of alternations from 'right' -> back to 'right'
    assert lf.current_lane == 'right'


def test_run_survives_a_lane_switch_mid_drive():
    lf = make_follower(WHITE_RIGHT_OF_YELLOW=True)
    img = (np.random.rand(240, 426, 3) * 255).astype(np.uint8)

    result = lf.run(img)
    assert len(result) == 6

    lf.set_lane('left')
    result = lf.run(img)
    assert len(result) == 6
    steering, throttle, out_img, yellow_x, white_x, width_px = result
    assert -1.0 <= steering <= 1.0
    assert out_img.shape == img.shape
