#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from types import SimpleNamespace

import numpy as np
from simple_pid import PID

from donkeycar.parts.lane_follower import LaneFollower, _LineTracker, _select_line_blob


# chosen well inside the default YELLOW/WHITE RGB threshold bands in cfg_cv_control.py
YELLOW_RGB = (220, 180, 50)   # within (190,120,0)-(255,220,100)
WHITE_RGB = (230, 230, 230)   # within (190,190,190)-(255,255,255)
BACKGROUND_RGB = (100, 100, 100)  # mid-gray: outside both threshold bands


def _make_cfg(**overrides):
    cfg = SimpleNamespace(
        OVERLAY_IMAGE=False,
        SCAN_Y=100,
        LANE_SCAN_ROWS=[{'scan_y': 100, 'weight': 1.0}],
        LANE_SCAN_HEIGHT=20,
        YELLOW_COLOR_THRESHOLD_LOW=(190, 120, 0),
        YELLOW_COLOR_THRESHOLD_HIGH=(255, 220, 100),
        WHITE_COLOR_THRESHOLD_LOW=(190, 190, 190),
        WHITE_COLOR_THRESHOLD_HIGH=(255, 255, 255),
        MIN_LINE_AREA_PX=150,
        MAX_LINE_WIDTH_PX=250,
        MIN_LINE_ASPECT_RATIO=0.15,
        MORPH_KERNEL_SIZE=0,  # disabled: keeps synthetic test blobs exact
        MAX_JUMP_PIXELS=40,
        REACQUIRE_AFTER_FRAMES=15,
        POSITION_SMOOTHING_ALPHA=0.4,
        WHITE_RIGHT_OF_YELLOW=True,
        LANE_WIDTH_PX=150,
        LANE_WIDTH_SMOOTHING_ALPHA=0.1,
        LANE_TARGET_PIXEL=None,
        LANE_TARGET_THRESHOLD=10,
        THROTTLE_INITIAL=0.15,
        THROTTLE_STEP=0.05,
        THROTTLE_MAX=0.3,
        THROTTLE_MIN=0.15,
        MAX_LOST_FRAMES=5,
        LOST_STEERING_DECAY=0.85,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_frame(width=320, height=240, background=BACKGROUND_RGB, lines=()):
    '''
    lines: iterable of (scan_y, scan_height, center_x, half_width, rgb) -
    paints a filled rectangle for each entry, simulating a line's cross-section.
    '''
    img = np.full((height, width, 3), background, dtype=np.uint8)
    for scan_y, scan_height, center_x, half_width, rgb in lines:
        x0 = max(0, center_x - half_width)
        x1 = min(width, center_x + half_width)
        img[scan_y: scan_y + scan_height, x0:x1, :] = rgb
    return img


def _scan_line(center_x, width=320, height=20, half_width=8, rgb=YELLOW_RGB, background=BACKGROUND_RGB):
    img = np.full((height, width, 3), background, dtype=np.uint8)
    x0 = max(0, center_x - half_width)
    x1 = min(width, center_x + half_width)
    img[:, x0:x1, :] = rgb
    return img


def _blank_scan_line(width=320, height=20, background=BACKGROUND_RGB):
    return np.full((height, width, 3), background, dtype=np.uint8)


class TestSelectLineBlob(unittest.TestCase):

    def test_picks_the_only_valid_blob(self):
        mask = np.zeros((20, 320), dtype=np.uint8)
        mask[:, 100:115] = 255  # width 15, area 300
        x, area = _select_line_blob(mask, min_area_px=150, max_width_px=250, min_aspect_ratio=0.15)
        self.assertIsNotNone(x)
        self.assertAlmostEqual(x, 107.0, delta=1.0)
        self.assertEqual(area, 300)

    def test_rejects_too_small_blob(self):
        mask = np.zeros((20, 320), dtype=np.uint8)
        mask[:2, 100:103] = 255  # tiny speckle: 2x3 = 6px area
        x, area = _select_line_blob(mask, min_area_px=150, max_width_px=250, min_aspect_ratio=0.15)
        self.assertIsNone(x)

    def test_rejects_too_wide_blob(self):
        mask = np.zeros((20, 320), dtype=np.uint8)
        mask[:, 0:300] = 255  # a wall/sunlit patch spanning almost the whole row
        x, area = _select_line_blob(mask, min_area_px=150, max_width_px=250, min_aspect_ratio=0.15)
        self.assertIsNone(x)

    def test_picks_largest_of_multiple_valid_blobs(self):
        mask = np.zeros((20, 320), dtype=np.uint8)
        mask[:, 50:65] = 255    # width 15, area 300
        mask[:, 200:230] = 255  # width 30, area 600 (aspect 20/30=0.67, still valid)
        x, area = _select_line_blob(mask, min_area_px=150, max_width_px=250, min_aspect_ratio=0.15)
        self.assertAlmostEqual(x, 215.0, delta=1.0)
        self.assertEqual(area, 600)


class TestLineTracker(unittest.TestCase):

    def _tracker(self, **overrides):
        cfg = _make_cfg(**overrides)
        return _LineTracker(cfg.YELLOW_COLOR_THRESHOLD_LOW, cfg.YELLOW_COLOR_THRESHOLD_HIGH, cfg)

    def test_locks_on_first_detection(self):
        tracker = self._tracker()
        x, _mask = tracker.update(_scan_line(150))
        self.assertAlmostEqual(x, 150, delta=1.0)
        self.assertEqual(tracker.tracked_position, x)

    def test_rejects_implausible_jump(self):
        tracker = self._tracker(MAX_JUMP_PIXELS=40)
        tracker.update(_scan_line(150))
        x, _mask = tracker.update(_scan_line(250))  # 100px jump, over the 40px gate
        self.assertIsNone(x)
        self.assertEqual(tracker.lost_frames, 1)

    def test_accepts_small_jump(self):
        tracker = self._tracker(MAX_JUMP_PIXELS=40)
        tracker.update(_scan_line(150))
        x, _mask = tracker.update(_scan_line(170))  # 20px jump, within the gate
        self.assertIsNotNone(x)

    def test_jump_gate_widens_after_consecutive_misses(self):
        # regression test for the outer-turn stall: a dashed line's gap plus
        # in-turn drift can put the next real detection farther than
        # MAX_JUMP_PIXELS from the last *accepted* position, even though it's
        # a legitimate detection - the gate should widen with lost_frames
        # instead of freezing at a stale anchor and rejecting it outright.
        tracker = self._tracker(MAX_JUMP_PIXELS=40, REACQUIRE_AFTER_FRAMES=15)
        tracker.update(_scan_line(150))
        tracker.update(_blank_scan_line())  # miss 1: lost_frames=1, gate=80
        tracker.update(_blank_scan_line())  # miss 2: lost_frames=2, gate=120
        # 100px jump from the stale anchor (150) - rejected by a fixed 40px
        # gate (see test_rejects_implausible_jump), accepted once widened to
        # 120px by two consecutive misses, and well under the 15-frame
        # REACQUIRE_AFTER_FRAMES threshold so continuity hasn't been dropped
        x, _mask = tracker.update(_scan_line(250))
        self.assertIsNotNone(x)  # accepted, not rejected as an implausible jump
        self.assertAlmostEqual(tracker.tracked_position, 250, delta=1.0)
        self.assertEqual(tracker.lost_frames, 0)

    def test_reacquires_after_sustained_loss(self):
        tracker = self._tracker(MAX_JUMP_PIXELS=40, REACQUIRE_AFTER_FRAMES=3)
        tracker.update(_scan_line(150))
        for _ in range(4):  # exceed REACQUIRE_AFTER_FRAMES with no detection
            tracker.update(_blank_scan_line())
        # a far-away line is now accepted despite the large jump, because
        # continuity was dropped after the sustained loss
        x, _mask = tracker.update(_scan_line(250))
        self.assertIsNotNone(x)
        self.assertAlmostEqual(x, 250, delta=1.0)

    def test_smoothing_blends_toward_new_position(self):
        tracker = self._tracker(MAX_JUMP_PIXELS=40, POSITION_SMOOTHING_ALPHA=0.5)
        tracker.update(_scan_line(150))
        x2, _mask = tracker.update(_scan_line(170))
        # smoothed position moves part-way from 150 toward 170, not straight to it
        self.assertGreater(x2, 150)
        self.assertLess(x2, 170)
        self.assertAlmostEqual(x2, 160, delta=1.0)  # alpha=0.5 -> ~midpoint


class TestLaneFollower(unittest.TestCase):

    def _pid(self, kp=0.01):
        return PID(Kp=kp, Ki=0.0, Kd=0.0)

    def test_centers_between_both_lines(self):
        cfg = _make_cfg(LANE_TARGET_PIXEL=160.0)
        follower = LaneFollower(self._pid(), cfg)
        img = _make_frame(lines=[
            (100, 20, 130, 8, YELLOW_RGB),
            (100, 20, 190, 8, WHITE_RGB),
        ])
        steering, throttle, _img, yellow_x, white_x, _lane_width = follower.run(img)
        self.assertAlmostEqual(yellow_x, 130, delta=2.0)
        self.assertAlmostEqual(white_x, 190, delta=2.0)
        self.assertAlmostEqual(steering, 0.0, delta=0.05)  # measured lane center ~= target

    def test_steers_toward_lane_center_when_off_target(self):
        cfg = _make_cfg(LANE_TARGET_PIXEL=160.0)
        follower = LaneFollower(self._pid(), cfg)
        img = _make_frame(lines=[
            (100, 20, 180, 8, YELLOW_RGB),
            (100, 20, 240, 8, WHITE_RGB),
        ])  # lane center ~210, well right of target 160
        steering, _throttle, _img, _yx, _wx, _lw = follower.run(img)
        self.assertLess(steering, 0.0)

    def test_estimates_lane_center_from_yellow_only(self):
        cfg = _make_cfg(LANE_TARGET_PIXEL=160.0, LANE_WIDTH_PX=60, WHITE_RIGHT_OF_YELLOW=True)
        follower = LaneFollower(self._pid(), cfg)
        img = _make_frame(lines=[(100, 20, 160, 8, YELLOW_RGB)])
        steering, _throttle, _img, yellow_x, white_x, _lw = follower.run(img)
        self.assertIsNotNone(yellow_x)
        self.assertIsNone(white_x)
        # lane center estimated as yellow_x + lane_width/2 = 160 + 30 = 190,
        # right of target 160 -> negative steering
        self.assertLess(steering, 0.0)

    def test_lane_width_estimate_updates_when_both_lines_visible(self):
        cfg = _make_cfg(LANE_WIDTH_PX=100, LANE_WIDTH_SMOOTHING_ALPHA=1.0)  # alpha=1 -> snap to measured
        follower = LaneFollower(self._pid(), cfg)
        img = _make_frame(lines=[
            (100, 20, 130, 8, YELLOW_RGB),
            (100, 20, 190, 8, WHITE_RGB),
        ])
        follower.run(img)
        self.assertAlmostEqual(follower.lane_width_px, 60, delta=2.0)

    def test_throttle_ramps_down_to_stop_after_sustained_loss(self):
        cfg = _make_cfg(MAX_LOST_FRAMES=3, THROTTLE_INITIAL=0.15, THROTTLE_MIN=0.15, THROTTLE_STEP=0.05)
        follower = LaneFollower(self._pid(), cfg)
        blank = _make_frame()
        throttle = None
        for _ in range(10):
            _steering, throttle, *_rest = follower.run(blank)
        self.assertEqual(throttle, 0.0)

    def test_multi_row_weighting_shifts_position_toward_far_row_on_curve(self):
        cfg = _make_cfg(
            LANE_TARGET_PIXEL=160.0,
            LANE_SCAN_ROWS=[{'scan_y': 100, 'weight': 1.0}, {'scan_y': 70, 'weight': 1.0}],
        )
        follower = LaneFollower(self._pid(), cfg)
        # near row (scan_y=100) is centered on target; far row (scan_y=70) is
        # shifted right, simulating an upcoming curve the far row sees first
        img = _make_frame(lines=[
            (100, 20, 130, 8, YELLOW_RGB), (100, 20, 190, 8, WHITE_RGB),  # near: center ~160
            (70, 20, 160, 8, YELLOW_RGB), (70, 20, 220, 8, WHITE_RGB),    # far: center ~190
        ])
        steering, _throttle, _img, _yx, _wx, _lw = follower.run(img)
        # equal-weighted average (160, 190) is right of target -> negative steering,
        # even though the near row alone would be exactly on target
        self.assertLess(steering, 0.0)

    def test_none_image_returns_safe_defaults(self):
        cfg = _make_cfg()
        follower = LaneFollower(self._pid(), cfg)
        steering, throttle, img, yellow_x, white_x, _lw = follower.run(None)
        self.assertEqual(steering, 0)
        self.assertEqual(throttle, 0)
        self.assertIsNone(img)
        self.assertIsNone(yellow_x)
        self.assertIsNone(white_x)


if __name__ == '__main__':
    unittest.main()
