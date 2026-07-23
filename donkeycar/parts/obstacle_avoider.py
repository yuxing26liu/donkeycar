import logging

import cv2
import numpy as np

from donkeycar.parts.lane_follower import _select_line_blob

logger = logging.getLogger(__name__)


def _lane_bounds(yellow_x, white_x, lane_width_px, white_right_of_yellow, other_lane=False):
    '''
    The pixel [low, high] extent of a lane, derived the same way
    LaneFollower._lane_center derives a lane's *center* (see lane_follower.py)
    - deliberately duplicated here rather than importing a shared helper, to
    keep this file's footprint in lane_follower.py at zero lines; see
    project_doc/obstacle_avoidance.md for that tradeoff.

    other_lane=False: our own lane, [white_x, yellow_x] (or one of those
    extrapolated from the other via lane_width_px, same as _lane_center).
    other_lane=True: the lane on the far side of the yellow centerline, which
    LaneFollower never tracks a boundary line for (our lane doesn't need
    one - see its class docstring). Only derivable from yellow_x, mirrored
    across it by lane_width_px - this assumes both lanes are about the same
    width, which is the only assumption possible without a tracked far edge.
    '''
    sign = 1.0 if white_right_of_yellow else -1.0

    if other_lane:
        if yellow_x is None:
            return None, None
        far_edge = yellow_x - sign * lane_width_px
        return tuple(sorted((yellow_x, far_edge)))

    if yellow_x is not None and white_x is not None:
        return tuple(sorted((yellow_x, white_x)))
    if yellow_x is not None:
        edge = yellow_x + sign * lane_width_px
        return tuple(sorted((yellow_x, edge)))
    if white_x is not None:
        edge = white_x - sign * lane_width_px
        return tuple(sorted((white_x, edge)))
    return None, None


class ObstacleAvoider:
    '''
    Obstacle avoidance layered on top of LaneFollower (donkeycar/parts/lane_follower.py)
    for the "two-way road navigation" mission (see CLAUDE.md). See
    project_doc/obstacle_avoidance.md for the full decision-by-decision design
    (detection strategy for each obstacle type, and why) and current build
    status - this class is being built incrementally, one detector at a time,
    before any steering override is wired up.

    Phase 1 (this increment): detect the traffic cone's position by its
    ground marker - a blue tape square laid down at the cone's spot on the
    track (see the reference track photos) - rather than the cone itself.
    A large flat tape square is a far more reliable classical-CV target at
    camera resolution/distance than a small cone silhouette would be, and
    its color doesn't collide with anything else in play here (gray
    concrete, white line, yellow dashed line). This phase is detection-only:
    run() always passes pilot/steering and pilot/throttle through unchanged.
    The car's driving behavior is unaffected even when this part is enabled
    - it only reports what it sees (self.cone_detected / self.cone_x, plus
    the 'obstacle/cone_detected' output and, if OVERLAY_IMAGE is set, a box
    drawn on cv/image_array) so the detector can be tuned and verified
    against real camera footage on the car before any avoidance maneuver is
    built on top of it.

    Not yet implemented (see project_doc/obstacle_avoidance.md "Next steps"):
    detecting the oncoming car (planned: color-key its black wheels/front,
    decision 2 in the design doc) and the actual avoidance maneuver (swerve
    to the other lane's center, then return - decision the design doc
    already worked out: a dedicated PID retargeting to
    _lane_bounds(..., other_lane=True)'s center).

    Zero changes to lane_follower.py: this part is purely downstream of it,
    reusing its already-published per-frame outputs (lane/yellow_x,
    lane/white_x, lane/width_px - see LaneFollower's own class docstring,
    which anticipated exactly this) instead of re-deriving lane geometry,
    and reusing its `_select_line_blob` connected-component shape filter
    for its own color-blob detection instead of a second implementation.

    Detection method: color-keyed in HSV (BLUE_HSV_THRESHOLD_LOW/HIGH),
    the same technique lane_follower.py's yellow _LineTracker uses and for
    the same reason - a solid, saturated color is a far more reliable
    signal than shape alone, and a positive color match (vs. e.g. "anything
    not already known") is naturally robust to background clutter like
    leaves/debris on the track, which won't be blue. Restricted to a single
    forward scan band (CONE_SCAN_Y/CONE_SCAN_HEIGHT) - the same slice-based
    approach every CV part in this codebase uses - both so a blob is only
    ever evaluated against the track's actual pixel-x extent at that row
    (background outside the track, e.g. the building/planter/recycling bin
    visible in the reference photos, never enters the candidate pool) and
    so there's lead distance to react before the marker reaches
    LaneFollower's own nearer scan rows.

    A detection only "counts" (self.cone_detected) if it falls within OUR
    lane's pixel bounds (_lane_bounds, with a small LANE_SHIFT_MARGIN_PX
    margin) AND holds for CONE_TRIGGER_FRAMES consecutive frames - a cone
    marked in the other lane is detected but doesn't count, and a one-frame
    misclassification (e.g. a sliver of glare) can't flip the flag by
    itself. This mirrors the "don't react to a single frame" caution
    lane_follower.py's continuity gating uses for the dashed yellow line,
    applied here to noise rejection instead of dash-gap tolerance.
    '''

    def __init__(self, cfg):
        self.overlay_image = getattr(cfg, 'OVERLAY_IMAGE', False)

        self.scan_y = getattr(cfg, 'CONE_SCAN_Y', 60)
        self.scan_height = getattr(cfg, 'CONE_SCAN_HEIGHT', 30)
        self.morph_kernel_size = getattr(cfg, 'MORPH_KERNEL_SIZE', 3)

        self.blue_low = np.asarray(getattr(cfg, 'BLUE_HSV_THRESHOLD_LOW', (95, 100, 60)))
        self.blue_high = np.asarray(getattr(cfg, 'BLUE_HSV_THRESHOLD_HIGH', (130, 255, 255)))
        self.cone_min_area_px = getattr(cfg, 'CONE_MIN_AREA_PX', 80)
        self.cone_max_width_px = getattr(cfg, 'CONE_MAX_WIDTH_PX', 250)

        self.white_right_of_yellow = getattr(cfg, 'WHITE_RIGHT_OF_YELLOW', True)
        self.lane_margin_px = getattr(cfg, 'LANE_SHIFT_MARGIN_PX', 10)
        self.cone_trigger_frames = getattr(cfg, 'CONE_TRIGGER_FRAMES', 2)

        # public detection state - what a future avoidance maneuver (or a
        # test) reads; updated every run() call
        self.cone_x = None              # raw detected x this frame (any lane), or None
        self.cone_in_our_lane = False   # raw in-our-lane test this frame, pre-debounce
        self.cone_detected = False      # debounced: True once cone_in_our_lane has held
                                         # for cone_trigger_frames consecutive frames
        self._pending_frames = 0

    def _open(self, mask):
        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def detect_cone(self, band_hsv):
        '''
        input: band_hsv, HSV numpy array of the forward scan band
        output: x position in pixels of the blue tape marker's blob centroid, or None
        '''
        mask = self._open(cv2.inRange(band_hsv, self.blue_low, self.blue_high))
        x, _area = _select_line_blob(mask, self.cone_min_area_px, self.cone_max_width_px,
                                      min_aspect_ratio=0.0, log_tag='cone_tape')
        return x

    def _x_in_bounds(self, x, bounds):
        lo, hi = bounds
        if x is None or lo is None:
            return False
        return lo - self.lane_margin_px <= x <= hi + self.lane_margin_px

    def run(self, cam_img, yellow_x, white_x, lane_width_px, steering, throttle, cv_img=None):
        '''
        main runloop
        input: cam_img, raw RGB camera frame (cam/image_array - independent of
               whatever LaneFollower drew on cv/image_array); yellow_x,
               white_x, lane_width_px - LaneFollower's own published lane
               geometry (lane/yellow_x, lane/white_x, lane/width_px);
               steering, throttle - LaneFollower's pilot output, passed
               through unchanged (see class docstring: Phase 1 is
               detection-only); cv_img, the already-annotated display image
               to optionally draw the detection on top of.
        output: steering, throttle (unchanged), cv_img, cone_detected
        '''
        if cam_img is None:
            return steering, throttle, cv_img, self.cone_detected

        band_rgb = cam_img[self.scan_y: self.scan_y + self.scan_height, :, :]
        if band_rgb.size == 0:
            logger.warning(f"Empty cone scan slice at scan_y={self.scan_y}: "
                            f"cam_img shape={cam_img.shape}; check CONE_SCAN_Y/HEIGHT")
            return steering, throttle, cv_img, self.cone_detected
        band_hsv = cv2.cvtColor(band_rgb, cv2.COLOR_RGB2HSV)

        our_lane = _lane_bounds(yellow_x, white_x, lane_width_px, self.white_right_of_yellow, other_lane=False)

        self.cone_x = self.detect_cone(band_hsv)
        self.cone_in_our_lane = self._x_in_bounds(self.cone_x, our_lane)

        if self.cone_in_our_lane:
            self._pending_frames += 1
        else:
            self._pending_frames = 0

        was_detected = self.cone_detected
        self.cone_detected = self._pending_frames >= self.cone_trigger_frames
        if self.cone_detected and not was_detected:
            logger.info(f"cone marker detected in our lane at x={self.cone_x:.1f} "
                         f"(held {self._pending_frames} frames)")
        elif was_detected and not self.cone_detected:
            logger.info("cone marker no longer in our lane")

        if self.overlay_image and cv_img is not None:
            cv_img = self.overlay_display(cv_img)

        return steering, throttle, cv_img, self.cone_detected

    def overlay_display(self, cv_img):
        y0, y1 = self.scan_y, self.scan_y + self.scan_height
        if self.cone_x is not None:
            color = (255, 140, 0) if self.cone_in_our_lane else (150, 150, 150)
            cv2.rectangle(cv_img, (int(self.cone_x) - 8, y0), (int(self.cone_x) + 8, y1),
                          color=color, thickness=2)
        cv2.putText(cv_img, f"CONE:{self.cone_detected}", org=(10, cv_img.shape[0] - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))
        return cv_img
