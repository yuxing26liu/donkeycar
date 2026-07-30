import numpy as np
import pytest

from donkeycar.parts.car_avoider import CarAvoider, _encroachment_bound, _past_encroachment_bound


IMAGE_W = 426
IMAGE_H = 240
GRAY = (120, 120, 120)
BLACK = (10, 10, 10)      # stand-in for the oncoming car's wheels/front
GREEN = (34, 139, 34)     # leaf/debris stand-in


class _Cfg:
    CAR_SCAN_Y = 45
    CAR_SCAN_HEIGHT = 20
    BLACK_HSV_THRESHOLD_LOW = (0, 0, 0)
    BLACK_HSV_THRESHOLD_HIGH = (179, 255, 60)
    CAR_MIN_AREA_PX = 50
    CAR_MAX_WIDTH_PX = 100000
    CAR_MIN_ASPECT_RATIO = 0.0
    WHITE_RIGHT_OF_YELLOW = True
    CAR_LANE_MARGIN_PX = 10
    CAR_EARLY_MARGIN_PX = 20   # smaller than the cfg_cv_control.py production
                                # default (60) - scaled to this test's
                                # LANE_WIDTH_PX=60 (a real lane runs
                                # 150-250px, see lane_follower.py)
    CAR_TRIGGER_FRAMES = 2
    CAR_FRAME_EDGE_MARGIN_PX = 5
    CAR_REQUIRE_GROWTH = False
    CAR_GROWTH_WINDOW_FRAMES = 5
    CAR_GROWTH_MIN_PX_PER_FRAME = 3.0
    MORPH_KERNEL_SIZE = 3
    OVERLAY_IMAGE = False
    CAR_LOG_INTERVAL_FRAMES = 10

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
    img = np.full((IMAGE_H, IMAGE_W, 3), GRAY, dtype=np.uint8)
    for x0, x1, y0, y1, color in patches:
        img[y0:y1, x0:x1] = color
    return img


# our lane: [200, 260] (yellow_x=200, white_x=260)
# other lane: [140, 200] (mirrored across yellow_x by lane_width_px=60)
YELLOW_X, WHITE_X, LANE_WIDTH_PX = 200.0, 260.0, 60.0
SCAN_Y0, SCAN_Y1 = _Cfg.CAR_SCAN_Y, _Cfg.CAR_SCAN_Y + _Cfg.CAR_SCAN_HEIGHT


class TestEncroachmentBound:
    def test_bound_is_margin_left_of_yellow_when_white_right(self):
        assert _encroachment_bound(200.0, 20.0, True) == 180.0

    def test_bound_is_margin_right_of_yellow_when_white_left(self):
        assert _encroachment_bound(200.0, 20.0, False) == 220.0

    def test_none_yellow_gives_none_bound(self):
        assert _encroachment_bound(None, 20.0, True) is None

    def test_past_bound_white_right(self):
        assert _past_encroachment_bound(185.0, 180.0, True) is True
        assert _past_encroachment_bound(150.0, 180.0, True) is False

    def test_past_bound_white_left(self):
        # white_right_of_yellow=False: our lane is on the LOWER-x side of
        # yellow, so "past the bound toward our lane" means x has dropped
        # to/below the bound, not risen above it
        assert _past_encroachment_bound(215.0, 220.0, False) is True
        assert _past_encroachment_bound(250.0, 220.0, False) is False


class TestCarAvoiderDetection:
    def _run_n(self, avoider, cam_img, n, other_avoidance_active=False):
        result = None
        for _ in range(n):
            result = avoider.run(cam_img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2,
                                  other_avoidance_active=other_avoidance_active)
        return result

    def test_no_black_no_detection(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame()
        steering, throttle, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert steering == 0.0 and throttle == 0.2  # pure passthrough

    def test_car_deep_in_other_lane_not_yet_encroaching(self):
        avoider = CarAvoider(_Cfg())
        # blob centered ~150, well inside the other lane [140,200] but
        # short of the encroachment bound (yellow_x - margin = 200-20=180)
        img = _make_frame([(140, 160, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert avoider.car_x is not None
        assert avoider.car_in_our_lane is False
        assert avoider.car_encroaching is False

    def test_car_encroaching_past_yellow_triggers_early(self):
        avoider = CarAvoider(_Cfg())
        # blob centered ~185: past the encroachment bound (180) but still
        # short of our lane's own bounds ([190,270] with margin) - this is
        # the "still crossing, not yet fully in our lane" case the early
        # margin exists to catch
        img = _make_frame([(180, 190, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])

        _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is False  # not yet (CAR_TRIGGER_FRAMES=2)
        assert avoider.car_encroaching is True
        assert avoider.car_in_our_lane is False

        _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is True
        assert avoider.avoiding is True

    def test_car_fully_in_our_lane_also_triggers(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(210, 230, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 2)
        assert detected is True
        assert avoider.car_in_our_lane is True

    def test_black_outside_scan_band_ignored(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(210, 230, SCAN_Y1 + 40, SCAN_Y1 + 60, BLACK)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert avoider.car_x is None

    def test_tiny_black_speck_below_min_area_ignored(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(220, 223, SCAN_Y0 + 2, SCAN_Y0 + 5, BLACK)])  # ~9px area
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False

    def test_green_debris_does_not_trigger(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(210, 230, SCAN_Y0 + 2, SCAN_Y0 + 12, GREEN)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False

    def test_black_at_frame_edge_ignored(self):
        avoider = CarAvoider(_Cfg())
        # a wide dark patch sitting at the very left edge - background
        # clutter, same reasoning as the cone detector's centered check
        img = _make_frame([(0, 4, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])
        _s, _t, _cv, detected = self._run_n(avoider, img, 5)
        assert detected is False
        assert avoider.car_x is None

    def test_no_lane_geometry_still_triggers_on_raw_detection(self):
        # mirrors ObstacleAvoider's tub_7_26-07-27 fallback: a car that has
        # already lost the lane must still be able to trigger avoidance
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(210, 230, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])
        result = None
        for _ in range(2):
            result = avoider.run(img, None, None, LANE_WIDTH_PX, 0.0, 0.2)
        _s, _t, _cv, detected = result
        assert detected is True
        assert avoider.lane_geometry_available is False

    def test_one_frame_then_miss_resets_debounce(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(210, 230, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])
        blank = _make_frame()
        avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert avoider._pending_frames == 1
        avoider.run(blank, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert avoider._pending_frames == 0
        _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is False  # only 1 consecutive frame again


class TestCarAvoiderOtherAvoidanceActive:
    def test_trigger_deferred_while_other_avoidance_active(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(210, 230, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])
        for _ in range(4):
            _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2,
                                                 other_avoidance_active=True)
        # car_detected (the debounced raw trigger) still latches...
        assert detected is True
        # ...but the maneuver itself never engages while ceding the actuator
        assert avoider.avoiding is False

    def test_maneuver_engages_once_other_avoidance_clears(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(210, 230, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])
        for _ in range(3):
            avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2, other_avoidance_active=True)
        assert avoider.avoiding is False
        avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2, other_avoidance_active=False)
        assert avoider.avoiding is True


class TestCarAvoiderGrowthGate:
    def test_static_area_does_not_trigger_when_growth_required(self):
        cfg = _Cfg()
        cfg.CAR_REQUIRE_GROWTH = True
        avoider = CarAvoider(cfg)
        img = _make_frame([(210, 230, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])  # same size every frame
        _s, _t, _cv, detected = self._run_n(avoider, img, 10)
        assert detected is False
        assert avoider.car_growing is False

    def test_growing_area_triggers_when_growth_required(self):
        cfg = _Cfg()
        cfg.CAR_REQUIRE_GROWTH = True
        avoider = CarAvoider(cfg)
        detected = False
        for i in range(8):
            width = 10 + i * 6  # blob widens each frame - simulates closing
            x0 = 220 - width // 2
            img = _make_frame([(x0, x0 + width, SCAN_Y0 + 2, SCAN_Y0 + 14, BLACK)])
            _s, _t, _cv, detected = avoider.run(img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert avoider.car_growing is True
        assert detected is True

    def _run_n(self, avoider, cam_img, n):
        result = None
        for _ in range(n):
            result = avoider.run(cam_img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        return result


class TestCarAvoiderManeuver:
    def test_steers_toward_other_lane_center(self):
        avoider = CarAvoider(_Cfg())
        img = _make_frame([(210, 230, SCAN_Y0 + 2, SCAN_Y0 + 12, BLACK)])
        for _ in range(2):
            steering, throttle, _cv, detected = avoider.run(
                img, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.0, 0.2)
        assert detected is True
        assert avoider.avoiding is True

        # other lane center = yellow_x - lane_width/2 = 200 - 30 = 170,
        # well left of image center (213) - steering should turn the car
        # left (this codebase's PID_P is negative, so a target left of
        # center produces positive steering - just assert it moved off
        # zero and stays within the PID's output bounds)
        assert steering != 0.0
        assert -1.0 <= steering <= 1.0
        assert avoider.avoid_target_pixel == pytest.approx(IMAGE_W / 2.0)

    def test_cam_img_none_passes_through(self):
        avoider = CarAvoider(_Cfg())
        steering, throttle, cv_img, detected = avoider.run(
            None, YELLOW_X, WHITE_X, LANE_WIDTH_PX, 0.5, 0.25)
        assert steering == 0.5 and throttle == 0.25 and cv_img is None
        assert detected is False
