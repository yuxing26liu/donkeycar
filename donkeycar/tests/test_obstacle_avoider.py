import numpy as np
import pytest

from donkeycar.parts.obstacle_avoider import ObstacleAvoider, _lane_bounds


IMAGE_W = 426
IMAGE_H = 240
GRAY = (120, 120, 120)   # stand-in for plain concrete
BLUE = (0, 0, 255)       # RGB pure blue -> HSV hue ~120 on OpenCV's 0..179 scale
GREEN = (34, 139, 34)    # stand-in for a leaf / other debris on the track
WHITE = (230, 230, 230)
YELLOW = (230, 200, 40)


class _Cfg:
    CONE_SCAN_Y = 60
    CONE_SCAN_HEIGHT = 30
    BLUE_HSV_THRESHOLD_LOW = (95, 100, 60)
    BLUE_HSV_THRESHOLD_HIGH = (130, 255, 255)
    CONE_MIN_AREA_PX = 80
    CONE_MAX_WIDTH_PX = 250
    WHITE_RIGHT_OF_YELLOW = True
    LANE_SHIFT_MARGIN_PX = 10
    CONE_TRIGGER_FRAMES = 2
    MORPH_KERNEL_SIZE = 3
    OVERLAY_IMAGE = False


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
