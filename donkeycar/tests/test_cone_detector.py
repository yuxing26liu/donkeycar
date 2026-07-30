"""
Unit tests for cone_detector.py.

These are synthetic-geometry tests of the detector's own logic (shape
filters, continuity, mask-fraction guard, depth percentile/fallback) --
NOT a substitute for validation against real camera footage. That
validation is done separately via scripts/replay_cone_planner.py against
the real recorded tubs (cone_static_right, cone_approach_right,
cone_negative, avoid_cone, tub_30_26-07-28) -- see this project's
architecture-planning notes on why origin/blue-tape-detect's and
origin/marcus-object-detection's synthetic-only test suites couldn't have
caught their real perspective/geometry bugs, and why this project treats
"passes its unit tests" and "works on real footage" as separate claims.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from donkeycar.parts.cone_detector import ConeDetector

W, H = 426, 240
ORANGE_HSV = (8, 180, 200)  # mid-range of the measured real cone signature


def hsv_to_rgb_frame(base_gray=60):
    frame = np.full((H, W, 3), base_gray, dtype=np.uint8)
    return frame


def paint_hsv_rect(rgb, x, y, w, h, hsv_color):
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[y:y + h, x:x + w] = hsv_color
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def make_cfg(**overrides):
    cfg = SimpleNamespace()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_detects_tall_cone_shaped_blob():
    det = ConeDetector(make_cfg())
    frame = hsv_to_rgb_frame()
    frame = paint_hsv_rect(frame, 200, 100, 20, 60, ORANGE_HSV)  # h/w = 3.0
    result, debug = det.detect(frame)
    assert result is not None, debug
    assert result.bbox.w == 20 and result.bbox.h == 60


def test_rejects_wide_flat_blob_by_aspect_ratio():
    det = ConeDetector(make_cfg())
    frame = hsv_to_rgb_frame()
    frame = paint_hsv_rect(frame, 100, 100, 100, 20, ORANGE_HSV)  # h/w = 0.2
    result, debug = det.detect(frame)
    assert result is None
    assert debug['reason'] == 'no_candidates'


def test_ignores_non_matching_color():
    det = ConeDetector(make_cfg())
    frame = hsv_to_rgb_frame()
    # paint a tall blue-ish rectangle -- should not match the orange window
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[100:160, 200:220] = (110, 180, 200)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    result, debug = det.detect(frame)
    assert result is None


def test_rejects_full_frame_overexposure_via_mask_fraction():
    det = ConeDetector(make_cfg(CONE_MAX_MASK_FRACTION=0.6))
    frame = hsv_to_rgb_frame()
    frame = paint_hsv_rect(frame, 0, 0, W, H, ORANGE_HSV)  # 100% of frame
    result, debug = det.detect(frame)
    assert result is None
    assert debug['reason'] == 'mask_fraction_too_large'


def test_continuity_prefers_blob_near_last_track_over_larger_distractor():
    det = ConeDetector(make_cfg(CONE_MAX_JUMP_PX=60))
    frame1 = hsv_to_rgb_frame()
    frame1 = paint_hsv_rect(frame1, 200, 100, 16, 50, ORANGE_HSV)
    r1, _ = det.detect(frame1)
    assert r1 is not None
    last_cx = r1.bbox.cx

    # frame 2: a small blob near the old position (continuity target) AND
    # a much larger distractor far away (would win on pure area alone)
    frame2 = hsv_to_rgb_frame()
    frame2 = paint_hsv_rect(frame2, 205, 102, 16, 52, ORANGE_HSV)     # near last track
    frame2 = paint_hsv_rect(frame2, 350, 40, 40, 120, ORANGE_HSV)     # far, much bigger
    r2, debug = det.detect(frame2)
    assert r2 is not None, debug
    assert abs(r2.bbox.cx - last_cx) < 20, "continuity should have preferred the nearby blob"


def test_cold_start_with_no_prior_track_picks_largest():
    det = ConeDetector(make_cfg())
    frame = hsv_to_rgb_frame()
    frame = paint_hsv_rect(frame, 50, 100, 14, 40, ORANGE_HSV)   # smaller
    frame = paint_hsv_rect(frame, 300, 60, 30, 100, ORANGE_HSV)  # larger
    result, debug = det.detect(frame)
    assert result is not None
    assert result.bbox.w == 30 and result.bbox.h == 100


def test_depth_percentile_estimate_ignores_noise_floor():
    det = ConeDetector(make_cfg(CONE_DEPTH_MIN_VALID_MM=150, CONE_DEPTH_MIN_VALID_PX=10))
    frame = hsv_to_rgb_frame()
    frame = paint_hsv_rect(frame, 100, 50, 40, 100, ORANGE_HSV)

    depth = np.full((H, W), 6000, dtype=np.uint16)
    # real cone surface inside the bbox ROI
    depth[70:140, 110:130] = 900
    # a ring of near-lens noise (below min-valid) around the whole frame,
    # simulating stereo-noise floor -- must NOT drag the estimate down
    depth[0:5, :] = 50

    result, debug = det.detect(frame, depth=depth)
    assert result is not None
    assert result.distance_source == 'depth'
    assert result.distance_valid is True
    assert 850 <= result.distance_mm <= 950


def test_depth_falls_back_to_bbox_height_when_insufficient_valid_pixels():
    det = ConeDetector(make_cfg(CONE_DEPTH_MIN_VALID_PX=1000))  # impossible to satisfy
    frame = hsv_to_rgb_frame()
    frame = paint_hsv_rect(frame, 100, 50, 40, 100, ORANGE_HSV)
    depth = np.full((H, W), 900, dtype=np.uint16)

    result, debug = det.detect(frame, depth=depth)
    assert result is not None
    assert result.distance_source == 'bbox_height'
    assert result.distance_valid is False
    assert result.distance_mm is None
    assert result.bbox.h == 100  # caller uses this as the ordinal proxy


def test_no_depth_array_uses_bbox_height_fallback():
    det = ConeDetector(make_cfg())
    frame = hsv_to_rgb_frame()
    frame = paint_hsv_rect(frame, 100, 50, 40, 100, ORANGE_HSV)
    result, debug = det.detect(frame, depth=None)
    assert result is not None
    assert result.distance_source == 'bbox_height'
    assert result.distance_valid is False
