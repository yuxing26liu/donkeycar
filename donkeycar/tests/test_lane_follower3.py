import numpy as np
import pytest
from simple_pid import PID

from donkeycar.parts.lane_follower3 import LaneFollower


IMAGE_W = 426
IMAGE_H = 240
GRAY = (100, 100, 100)   # stand-in for plain concrete
WHITE = (220, 220, 220)  # bright, low-saturation - passes the LAB-adaptive test
# moderate lightness / high saturation, like real paint - not pure RGB
# yellow (255,255,0), whose LAB lightness rivals WHITE's and skews a
# combined row's stddev enough to trip the adaptive mask's own
# two-lighting-conditions guard (ADAPTIVE_MAX_STD) as an unrelated
# side effect
YELLOW = (190, 160, 30)  # -> HSV hue ~24 on OpenCV's 0..179 scale


class _Cfg:
    OVERLAY_IMAGE = False
    SCAN_Y = 100  # accessed eagerly as a getattr() default even when LANE_SCAN_ROWS is set
    LANE_SCAN_ROWS = [{'scan_y': 100, 'weight': 1.0}]
    LANE_SCAN_HEIGHT = 20
    YELLOW_HSV_THRESHOLD_LOW = (15, 60, 60)
    YELLOW_HSV_THRESHOLD_HIGH = (35, 255, 255)
    WHITE_RIGHT_OF_YELLOW = True
    LANE_WIDTH_PX = 150
    THROTTLE_INITIAL = 0.2
    THROTTLE_STEP = 0.05
    THROTTLE_MAX = 0.3
    THROTTLE_MIN = 0.1


def _make_frame(patches=()):
    img = np.full((IMAGE_H, IMAGE_W, 3), GRAY, dtype=np.uint8)
    for x0, x1, y0, y1, color in patches:
        img[y0:y1, x0:x1] = color
    return img


def _make_follower():
    pid = PID(Kp=-0.01, Ki=0.0, Kd=-0.0001)
    return LaneFollower(pid, _Cfg())


class TestLaneFollowerColdStartTwoLineGuard:
    '''
    Added after tub_7_26-07-27: the car swerved to a near-max steering lock
    1.5s into a drive and never recovered. Replaying the recorded frames
    through the real white-tracker code showed a false single-line "white"
    lock (a bright, low-saturation blob - in that footage, a sunlit
    building wall - that passed the same shape/color tests a real line
    would) immediately produced a confidently-acted-on lane-center estimate
    with no established anchor to check it against. See the module
    docstring's changelog entry 9 and the guard in LaneFollower.run().
    '''

    def test_single_line_only_on_cold_start_does_not_set_steering_anchor(self):
        lf = _make_follower()
        # a single bright "white" blob, no yellow anywhere - exactly the
        # false-lock shape found in the real footage (min_area_px=150
        # default: 80px wide x 20px tall = 1600px, well clear of it)
        img = _make_frame([(150, 230, 100, 120, WHITE)])

        steering, throttle, _img, yellow_x, white_x, _lane_w = lf.run(img)

        assert yellow_x is None
        # the tracker itself still detects and publishes the raw blob -
        # only whether LaneFollower ACTS on it for steering is guarded
        assert white_x is not None
        # never established an anchor from the single-line fallback -
        # this is the actual behavioral assertion: without it, self.steering
        # would already reflect a confidently-acted-on (and wrong) command
        assert lf._last_position is None
        assert lf.lost_frames == 1
        # full-loss path, not a real lock: throttle ramps toward the floor
        assert throttle == pytest.approx(_Cfg.THROTTLE_INITIAL - _Cfg.THROTTLE_STEP)

    def test_repeated_single_line_only_frames_never_lock_in_without_yellow(self):
        lf = _make_follower()
        img = _make_frame([(150, 230, 100, 120, WHITE)])
        for _ in range(10):
            steering, throttle, _img, _y, _w, _lw = lf.run(img)
        # ten frames of the same unconfirmed single-line blob: still no
        # anchor, still decaying toward a stop, never a confident lock
        assert lf._last_position is None
        assert steering == pytest.approx(0.0)
        assert throttle == pytest.approx(_Cfg.THROTTLE_MIN)

    def test_two_line_detection_establishes_anchor_immediately(self):
        lf = _make_follower()
        # yellow at x~110, white at x~270 (both inside the row, clearly
        # separated) - a genuine two-line sighting
        img = _make_frame([
            (90, 130, 100, 120, YELLOW),
            (250, 330, 100, 120, WHITE),
        ])
        _steering, _throttle, _img, yellow_x, white_x, _lane_w = lf.run(img)

        assert yellow_x is not None
        assert white_x is not None
        assert lf._last_position is not None

    def test_single_line_frames_are_trusted_once_an_anchor_exists(self):
        # regression check: the guard must only gate the FIRST-EVER lock,
        # not single-line tracking in general once driving is underway -
        # otherwise every dashed-yellow-line gap would wrongly freeze
        # steering mid-drive.
        lf = _make_follower()
        two_line_img = _make_frame([
            (90, 130, 100, 120, YELLOW),
            (250, 330, 100, 120, WHITE),
        ])
        lf.run(two_line_img)
        assert lf._last_position is not None
        anchor_after_two_line = lf._last_position

        # now yellow disappears (e.g. a dash gap) - only white remains.
        # run() returns self.last_yellow_x ("last known", not "this
        # frame's fresh hit" - see the module docstring's dash-gap
        # coasting rationale), so it stays at its previous value here;
        # the tracker's OWN fresh-miss state is what actually matters for
        # this regression check.
        white_only_img = _make_frame([(250, 330, 100, 120, WHITE)])
        _steering, _throttle, _img, _yellow_x, white_x, _lane_w = lf.run(white_only_img)

        assert lf.yellow_trackers[0].lost_frames > 0  # genuinely missed this frame
        assert white_x is not None
        # position updated from the single-line fallback, exactly as
        # before this guard existed - the anchor already established from
        # the earlier two-line frame is what makes this trustworthy
        assert lf._last_position is not None
        assert lf.lost_frames == 0
