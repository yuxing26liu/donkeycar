import logging

import numpy as np
import pytest

from donkeycar.parts.obstacle_avoider import ObstacleAvoider, _lane_bounds


IMAGE_W = 426
IMAGE_H = 240
GRAY = (120, 120, 120)   # stand-in for plain concrete
BLUE = (0, 0, 255)       # RGB pure blue -> HSV hue ~120 on OpenCV's 0..179 scale
ORANGE = (255, 90, 0)    # RGB traffic-cone orange -> HSV hue ~10 on OpenCV's 0..179 scale
                          # (matches real cone pixels sampled from tub_33_26-07-24)
GREEN = (34, 139, 34)    # stand-in for a leaf / other debris on the track
WHITE = (230, 230, 230)
YELLOW = (230, 200, 40)


class _Cfg:
    CONE_SCAN_Y = 60
    CONE_SCAN_HEIGHT = 30
    BLUE_HSV_THRESHOLD_LOW = (95, 100, 60)
    BLUE_HSV_THRESHOLD_HIGH = (130, 255, 255)
    ORANGE_HSV_THRESHOLD_LOW = (0, 90, 60)
    ORANGE_HSV_THRESHOLD_HIGH = (18, 255, 255)
    CONE_MIN_AREA_PX = 80
    CONE_MAX_WIDTH_PX = 250
    WHITE_RIGHT_OF_YELLOW = True
    LANE_SHIFT_MARGIN_PX = 10
    CONE_TRIGGER_FRAMES = 2
    MORPH_KERNEL_SIZE = 3
    OVERLAY_IMAGE = False
    CONE_LOG_INTERVAL_FRAMES = 10

    # avoidance maneuver (Phase 2) - same values as cfg_cv_control.py's
    # defaults for the constants it reuses (PID_P/I/D, THROTTLE_*,
    # MAX_LOST_FRAMES, LOST_STEERING_DECAY), spelled out explicitly here so
    # test behavior doesn't depend on ObstacleAvoider's internal fallbacks
    PID_P = -0.01
    PID_I = 0.0
    PID_D = -0.0001
    LANE_TARGET_THRESHOLD = 10
    THROTTLE_STEP = 0.05
    THROTTLE_MIN = 0.1
    THROTTLE_MAX = 0.3
    MAX_LOST_FRAMES = 40
    LOST_STEERING_DECAY = 0.85


def _make_frame(patches=()):
    '''patches: iterable of (x0, x1, y0, y1, rgb_color) rectangles to paint
    onto an otherwise plain gray frame.'''
    img = np.full((IMAGE_H, IMAGE_W, 3), GRAY, dtype=np.uint8)
    for x0, x1, y0, y1, color in patches:
        img[y0:y1, x0:x1] = color
    return img


# our lane: [200, 260] (yellow_x=200, white_x=260)
# other lane: [140, 200] (mirrored across yellow_x by lane_width_px=60)
YELLOW_X, WHITE_X, LANE_WIDTH_PX = 200.0, 260.0, 60.0


class TestLaneBounds:
    def test_both_lines_visible(self):
        assert _lane_bounds(YELLOW_X, WHITE_X, LANE_WIDTH_PX, True) == (200.0, 260.0)

    def test_other_lane_mirrors_across_yellow(self):
        assert _lane_bounds(YELLOW_X, WHITE_X, LANE_WIDTH_PX, True, other_lane=True) == (140.0, 200.0)

    def test_yellow_only_extrapolates_our_lane(self):
        assert _lane_bounds(YELLOW_X, None, LANE_WIDTH_PX, True) == (200.0, 260.0)

    def test_white_only_extrapolates_our_lane(self):
        assert _lane_bounds(None, WHITE_X, LANE_WIDTH_PX, True) == (200.0, 260.0)

    def test_other_lane_needs_yellow(self):
        assert _lane_bounds(None, WHITE_X, LANE_WIDTH_PX, True, other_lane=True) == (None, None)

    def test_neither_line_visible(self):
        assert _lane_bounds(None, None, LANE_WIDTH_PX, True) == (None, None)

    def test_white_left_of_yellow_flips_sign(self):
        # left-lane car: white boundary is on the left of the yellow centerline
        assert _lane_bounds(200.0, 140.0, LANE_WIDTH_PX, False) == (140.0, 200.0)


class TestObstacleAvoiderConeDetection:
    def _run_n(self, avoider, cam_img, n):
        result = None
        for _ in range(n):
            result = avoider.run(cam_img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        return result

    def test_no_blue_no_detection(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame()
        steering, throttle, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert steering == 0.0 and throttle == 0.2  # pure passthrough (Phase 1)

    def test_cone_in_our_lane_latches_after_trigger_frames(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(210, 230, 65, 85, BLUE)])  # centered ~220, inside [200,260]

        # single frame: not yet latched (CONE_TRIGGER_FRAMES=2)
        _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is False
        assert avoider.cone_in_our_lane is True

        # second consecutive frame: latches
        _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is True
        assert avoider.cone_x == pytest.approx(220, abs=3)

    def test_cone_in_other_lane_is_ignored(self):
        avoider = ObstacleAvoider(_Cfg())
        # blue square at x=[150,170], inside the OTHER lane [140,200], well
        # outside our lane's [190,270] margin-extended bounds
        img = _make_frame([(150, 170, 65, 85, BLUE)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert avoider.cone_x is not None       # detected...
        assert avoider.cone_in_our_lane is False  # ...but correctly not "in our lane"

    def test_blue_outside_scan_band_is_ignored(self):
        avoider = ObstacleAvoider(_Cfg())
        # same x-position as the positive case, but well below the scan band
        # (CONE_SCAN_Y=60, CONE_SCAN_HEIGHT=30 -> band is rows [60,90))
        img = _make_frame([(210, 230, 150, 170, BLUE)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert avoider.cone_x is None

    def test_tiny_blue_speck_below_min_area_ignored(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(220, 223, 65, 68, BLUE)])  # 3x3 px, well under CONE_MIN_AREA_PX=80
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False

    def test_leaf_like_debris_does_not_trigger(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(210, 230, 65, 85, GREEN)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert avoider.cone_x is None

    def test_track_lines_do_not_trigger(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([
            (0, IMAGE_W, 65, 68, WHITE),
            (195, 205, 65, 68, YELLOW),
        ])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False

    def test_losing_the_cone_resets_the_debounce(self):
        avoider = ObstacleAvoider(_Cfg())
        with_cone = _make_frame([(210, 230, 65, 85, BLUE)])
        without_cone = _make_frame()

        avoider.run(with_cone, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        avoider.run(without_cone, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        # the one in-lane frame shouldn't carry over after a miss
        _s, _t, _cv, detected = avoider.run(with_cone, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is False

    def test_cam_img_none_passes_through(self):
        avoider = ObstacleAvoider(_Cfg())
        steering, throttle, cv_img, detected = avoider.run(None, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.3, 0.25)
        assert (steering, throttle, cv_img, detected) == (0.3, 0.25, None, False)

    def test_overlay_draws_without_raising_and_leaves_control_output_unchanged(self):
        cfg = _Cfg()
        cfg.OVERLAY_IMAGE = True
        avoider = ObstacleAvoider(cfg)
        img = _make_frame([(210, 230, 65, 85, BLUE)])
        cv_img = np.copy(img)
        steering, throttle, out_cv_img, _detected = avoider.run(
            img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2, cv_img)
        assert steering == 0.0 and throttle == 0.2
        assert out_cv_img is not None
        assert out_cv_img.shape == img.shape


class TestObstacleAvoiderOrangeConeDetection:
    '''
    Mirrors TestObstacleAvoiderConeDetection's key cases for the orange-cone
    detector, added after tub_33_26-07-24 (real on-car footage) showed the
    cones on this track have no blue tape marker at all - see class
    docstring in obstacle_avoider.py.
    '''

    def _run_n(self, avoider, cam_img, n):
        result = None
        for _ in range(n):
            result = avoider.run(cam_img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        return result

    def test_orange_cone_in_our_lane_latches_after_trigger_frames(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(210, 230, 65, 85, ORANGE)])  # centered ~220, inside [200,260]

        _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is False
        assert avoider.cone_in_our_lane is True
        assert avoider.cone_color == 'orange cone'

        _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is True
        assert avoider.cone_x == pytest.approx(220, abs=3)

    def test_orange_cone_in_other_lane_is_ignored(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(150, 170, 65, 85, ORANGE)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert avoider.cone_x is not None
        assert avoider.cone_in_our_lane is False

    def test_orange_outside_scan_band_is_ignored(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(210, 230, 150, 170, ORANGE)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert avoider.cone_x is None

    def test_tiny_orange_speck_below_min_area_ignored(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(220, 223, 65, 68, ORANGE)])  # 3x3 px, well under CONE_MIN_AREA_PX=80
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False

    def test_larger_blob_wins_when_both_colors_present(self):
        # a big orange cone in our lane and a small blue speck (below
        # CONE_MIN_AREA_PX, e.g. background glare) elsewhere in the band:
        # only the orange blob should pass the shape filter at all, so it
        # wins regardless of the area-comparison tiebreak.
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([
            (210, 230, 65, 85, ORANGE),   # 20x20 = 400px, well over CONE_MIN_AREA_PX
            (10, 13, 65, 68, BLUE),       # 3x3 = 9px, rejected by shape filter
        ])
        _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert avoider.cone_color == 'orange cone'
        assert avoider.cone_x == pytest.approx(220, abs=3)


class TestObstacleAvoiderAvoidanceManeuver:
    '''
    Phase 2: once a cone latches as detected, ObstacleAvoider must actually
    override steering/throttle to swerve toward the other lane, and must
    NOT swerve back to the original lane afterward - the maneuver latches
    permanently (see class docstring in obstacle_avoider.py and the
    "don't return to the original lane" decision behind it).
    '''

    def _trigger(self, avoider, cone_patch=(210, 230, 65, 85, ORANGE)):
        img = _make_frame([cone_patch])
        for _ in range(2):  # CONE_TRIGGER_FRAMES=2
            result = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        return result

    def test_avoidance_triggers_and_overrides_steering(self):
        avoider = ObstacleAvoider(_Cfg())
        steering, throttle, _cv, detected = self._trigger(avoider)

        assert detected is True
        assert avoider.avoiding is True
        # no longer plain passthrough (0.0) now that the maneuver is active
        assert steering != 0.0

    def test_avoidance_steers_toward_other_lane_center(self):
        # other lane center = yellow_x - lane_width_px/2 = 200 - 30 = 170,
        # which is left of image center (426/2 = 213) - the PID should
        # therefore command a nonzero correction, not hold at 0
        avoider = ObstacleAvoider(_Cfg())
        self._trigger(avoider)
        assert avoider.avoid_target_pixel == pytest.approx(IMAGE_W / 2.0)
        assert avoider.avoid_pid.setpoint == pytest.approx(IMAGE_W / 2.0)

    def test_avoidance_does_not_return_to_original_lane_once_cone_clears(self):
        avoider = ObstacleAvoider(_Cfg())
        self._trigger(avoider)
        assert avoider.avoiding is True

        without_cone = _make_frame()
        for _ in range(10):
            steering, throttle, _cv, detected = avoider.run(
                without_cone, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)

        # cone_detected clears (it's no longer visible/in-lane), but the
        # maneuver itself must stay latched and keep steering toward the
        # other lane's center, not revert to passthrough/original-lane
        assert detected is False
        assert avoider.avoiding is True
        assert steering != 0.0

    def test_avoidance_holds_last_steering_when_lane_geometry_lost(self):
        avoider = ObstacleAvoider(_Cfg())
        self._trigger(avoider)
        steering_before = avoider.avoid_steering

        img = _make_frame()
        steering, throttle, _cv, _detected = avoider.run(img, None, None, LANE_WIDTH_PX, 0.0, 0.2)

        # passive fallback (Decision 3): decay the last steering rather
        # than guess off missing geometry, and don't raise
        assert steering == pytest.approx(steering_before * avoider.lost_steering_decay)

    def test_no_trigger_leaves_passthrough_unaffected(self):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame()
        steering, throttle, _cv, detected = self._run_plain(avoider, img)
        assert detected is False
        assert avoider.avoiding is False
        assert steering == 0.0 and throttle == 0.2

    def _run_plain(self, avoider, img, n=5):
        result = None
        for _ in range(n):
            result = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        return result


class TestObstacleAvoiderDiagnostics:
    '''
    Terminal-output checks for on-car verification: a raw blue-blob
    detection must print its sampled color value regardless of whether lane
    geometry is available, and a missing lane/yellow_x + lane/white_x must
    be surfaced as a one-time warning rather than silently never triggering
    (see project_doc/obstacle_avoidance.md, "would this work on the pi").
    '''

    def test_raw_detection_logs_sampled_color_even_without_lane_geometry(self, caplog):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(210, 230, 65, 85, BLUE)])
        with caplog.at_level(logging.INFO, logger='donkeycar.parts.obstacle_avoider'):
            avoider.run(img, None, None, LANE_WIDTH_PX, 0.0, 0.2)
        messages = [r.message for r in caplog.records]
        color_logs = [m for m in messages if 'blue tape candidate at' in m]
        assert len(color_logs) == 1
        assert 'HSV=' in color_logs[0] and 'RGB=' in color_logs[0]
        assert 'NOT in our lane' in color_logs[0]

        # the sampled color should reflect the pure-blue patch itself, not a
        # blend with the surrounding gray track (regression check: the patch
        # must be centered on the marker's vertical extent within the scan
        # band, not averaged over the whole scan band height)
        hue = int(color_logs[0].split('HSV=(')[1].split(',')[0])
        assert hue == pytest.approx(120, abs=5)

    def test_orange_raw_detection_logs_sampled_color(self, caplog):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame([(210, 230, 65, 85, ORANGE)])
        with caplog.at_level(logging.INFO, logger='donkeycar.parts.obstacle_avoider'):
            avoider.run(img, None, None, LANE_WIDTH_PX, 0.0, 0.2)
        messages = [r.message for r in caplog.records]
        color_logs = [m for m in messages if 'orange cone candidate at' in m]
        assert len(color_logs) == 1
        hue = int(color_logs[0].split('HSV=(')[1].split(',')[0])
        assert hue == pytest.approx(10, abs=5)

    def test_startup_line_logs_both_color_thresholds(self, caplog):
        with caplog.at_level(logging.INFO, logger='donkeycar.parts.obstacle_avoider'):
            ObstacleAvoider(_Cfg())
        messages = [r.message for r in caplog.records]
        assert any('ObstacleAvoider active' in m and 'blue tape' in m and 'orange cone' in m
                    for m in messages)

    def test_raw_detection_log_clears_on_loss(self, caplog):
        avoider = ObstacleAvoider(_Cfg())
        with_cone = _make_frame([(210, 230, 65, 85, BLUE)])
        without_cone = _make_frame()
        with caplog.at_level(logging.INFO, logger='donkeycar.parts.obstacle_avoider'):
            avoider.run(with_cone, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
            avoider.run(without_cone, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        messages = [r.message for r in caplog.records]
        assert any('no longer visible' in m for m in messages)

    def test_missing_lane_geometry_warns_once(self, caplog):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame()
        with caplog.at_level(logging.INFO, logger='donkeycar.parts.obstacle_avoider'):
            for _ in range(5):
                avoider.run(img, None, None, LANE_WIDTH_PX, 0.0, 0.2)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                    and 'CV_CONTROLLER_OUTPUTS' in r.message]
        assert len(warnings) == 1  # logged once, not every frame

    def test_present_lane_geometry_never_warns(self, caplog):
        avoider = ObstacleAvoider(_Cfg())
        img = _make_frame()
        with caplog.at_level(logging.INFO, logger='donkeycar.parts.obstacle_avoider'):
            for _ in range(5):
                avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == []
