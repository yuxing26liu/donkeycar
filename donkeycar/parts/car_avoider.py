import logging
import math
from collections import deque

import cv2
import numpy as np
from simple_pid import PID

from donkeycar.parts.lane_follower import _select_line_blob
from donkeycar.parts.obstacle_avoider import _lane_bounds, _other_lane_center

logger = logging.getLogger(__name__)


def _x_in_bounds(x, bounds, margin_px):
    lo, hi = bounds
    if x is None or lo is None:
        return False
    return lo - margin_px <= x <= hi + margin_px


def _encroachment_bound(yellow_x, margin_px, white_right_of_yellow):
    '''
    The x position, margin_px on the OPPOSITE-lane side of the yellow
    centerline, past which an oncoming car counts as "heading into our
    lane" for CarAvoider's purposes - deliberately short of _lane_bounds'
    actual our-lane edge (yellow_x itself), so the trigger fires while the
    car is still crossing rather than only once it has fully arrived. See
    CarAvoider's class docstring for why this exists: unlike the stationary
    cone in obstacle_avoider.py, the oncoming car is closing at combined
    speed, so waiting for it to be strictly "in our lane" before reacting
    leaves far less time to complete the swerve.
    '''
    if yellow_x is None:
        return None
    sign = 1.0 if white_right_of_yellow else -1.0
    return yellow_x - sign * margin_px


def _past_encroachment_bound(x, bound, white_right_of_yellow):
    if x is None or bound is None:
        return False
    return x >= bound if white_right_of_yellow else x <= bound


class CarAvoider:
    '''
    Oncoming-car detection + avoidance, layered on top of LaneFollower
    (donkeycar/parts/lane_follower.py) the same way
    donkeycar/parts/obstacle_avoider.py's ObstacleAvoider is - a separate
    downstream part, zero changes to lane_follower.py, reusing its
    published lane/yellow_x, lane/white_x, lane/width_px outputs instead of
    re-deriving lane geometry. See project_doc/obstacle_avoidance.md,
    Decision 2 ("How to detect the oncoming car") and the "Next steps"
    section this file implements the first item of.

    This intentionally imports _lane_bounds/_other_lane_center directly
    from obstacle_avoider.py rather than re-deriving or re-duplicating that
    lane-geometry math a second time (obstacle_avoider.py already made the
    "duplicate a small helper rather than touch lane_follower.py" tradeoff
    once; a third copy of the same ~10-line formula would be pure
    liability). The avoidance maneuver itself (_avoid_step below) IS a
    near-duplicate of ObstacleAvoider._avoid_step, for the same reason
    obstacle_avoider.py's own docstring gives for duplicating
    LaneFollower's lane-center math: this maneuver's PID/rate-limit state
    (self.avoid_pid, self._avoid_last_position, ...) belongs to this part
    instance, not ObstacleAvoider's, so the two avoidance maneuvers'
    integral/anchor state can never cross-contaminate even when both parts
    are wired into the same car.

    Detection (Decision 2, option A): color-key the car's black
    wheels/front in HSV (BLACK_HSV_THRESHOLD_LOW/HIGH - a low Value
    threshold, independent of hue/saturation), reusing
    lane_follower.py's `_select_line_blob` for the same connected-
    component shape/size filter every other color-blob detector in this
    codebase uses, restricted to its own forward scan band
    (CAR_SCAN_Y/CAR_SCAN_HEIGHT - see Decision 4: a farther/earlier
    lookahead than the cone's CONE_SCAN_Y, since the car is closing at
    combined speed rather than sitting still). Two known false-positive
    risks (Decision 2's table): cast shadows and the track's own dark
    expansion-joint seams. Mitigated here by:
      - the same shape/size filter every line/blob detector uses
        (CAR_MIN_AREA_PX/CAR_MAX_WIDTH_PX),
      - a frame-edge sanity gate (CAR_FRAME_EDGE_MARGIN_PX) rejecting
        blobs at the extreme image edges, the same place off-track
        clutter (recycling bin, kiosk sign) sat in the cone detector's
        real footage (see obstacle_avoider.py's class docstring),
      - an optional frame-over-frame blob-growth requirement
        (CAR_REQUIRE_GROWTH, off by default - Decision 2's option C, "held
        in reserve" per the design doc until real footage shows shadows/
        seams are a recurring false trigger): a real approaching car's
        apparent size should grow frame to frame; a static shadow or seam
        shouldn't. self.car_growing is computed and published every frame
        regardless of CAR_REQUIRE_GROWTH, so it's visible for tuning
        before deciding whether to gate on it.
    None of these thresholds have been checked against real on-car
    footage of the actual oncoming car yet (no such footage existed at the
    time this file was written - unlike CONE_HSV_THRESHOLD, which was
    calibrated against real sampled cone pixels) - BLACK_HSV_THRESHOLD is a
    starting guess in the same spirit as BLUE_HSV_THRESHOLD in
    obstacle_avoider.py, not a measurement.

    Triggering "early" (this file's actual novel piece over just porting
    the cone's approach): the cone detector waits for a detection to be
    centered AND close - reasonable for a stationary object, since "close"
    is the only signal the cone is worth reacting to at all. An oncoming
    car is a moving hazard closing at (its speed + our speed), so waiting
    for it to be fully inside our lane's pixel bounds before reacting
    leaves much less time to complete a swerve than the cone maneuver
    gets. self.car_encroaching (_encroachment_bound/_past_encroachment_bound)
    extends the trigger zone CAR_EARLY_MARGIN_PX past the yellow
    centerline into the OTHER lane, so the maneuver can begin while the
    car is still crossing rather than only once it has arrived -
    car_ready fires on car_in_our_lane OR car_encroaching, not only the
    former.

    Same lane-geometry-lost exception as ObstacleAvoider (see its class
    docstring's "Exception" paragraph, added after tub_7_26-07-27): when
    LaneFollower publishes no lane geometry at all this frame,
    car_in_our_lane/car_encroaching are structurally False (both depend on
    yellow_x), so a car that has already lost the lane could otherwise
    never trigger avoidance no matter how obviously an oncoming car sat
    dead ahead. Falls back to "a plausible black blob was detected in the
    scan band at all" in that case - see run().

    The avoidance maneuver itself (_avoid_step) steers toward
    _other_lane_center exactly like ObstacleAvoider._avoid_step does,
    reusing the same hard-won defenses (soft-saturated error, position and
    steering rate limits, a plausibility gate on yellow_x jumps, passive
    hold/decay when lane geometry is lost mid-maneuver) - see
    obstacle_avoider.py's class/`_avoid_step` docstrings for the on-car
    incidents each of those defenses was added after. Skipping any of them
    here would just reproduce the same bugs a second time.

    Like ObstacleAvoider's cone maneuver (Decision 1.5), this maneuver
    latches permanently once triggered (self.avoiding never un-latches) -
    the same "one maneuver, nothing to get wrong about when to swerve
    back" simplicity tradeoff, not yet revisited for the oncoming-car case
    specifically (see project_doc/obstacle_avoidance.md's "Next steps" #2
    for the known gap: how this interacts with a car that swerved for a
    cone is still unsolved).

    Decision 5 (project_doc/obstacle_avoidance.md): "once a maneuver is
    active for one obstacle type, ignore triggers of the other type until
    back to cruising - first-detected wins." Since this is a separate Part
    instance from ObstacleAvoider (no shared instance state), run() accepts
    an optional `other_avoidance_active` flag - cv_control.py/manage2.py
    wire `obstacle/avoiding` into it (a Lambda reading
    ObstacleAvoider.avoiding directly off the live instance - see the
    comment there for why `obstacle/cone_detected` is NOT the right signal:
    it un-latches once the cone leaves the scan band even though the
    permanent swerve is still underway) so a cone maneuver already active
    isn't immediately overridden the same frame the car crosses in; this
    part's own trigger simply waits (debounce keeps counting) until that
    flag clears.
    '''

    def __init__(self, cfg):
        self.overlay_image = getattr(cfg, 'OVERLAY_IMAGE', False)

        self.scan_y = getattr(cfg, 'CAR_SCAN_Y', 45)
        self.scan_height = getattr(cfg, 'CAR_SCAN_HEIGHT', 20)
        self.morph_kernel_size = getattr(cfg, 'MORPH_KERNEL_SIZE', 3)

        # Low Value (brightness), any hue/saturation - "black" regardless of
        # what's tinting it. Untuned guess (see class docstring) - no real
        # footage of the oncoming car existed to sample from when this was
        # written, unlike ORANGE_HSV_THRESHOLD in obstacle_avoider.py.
        self.black_low = np.asarray(getattr(cfg, 'BLACK_HSV_THRESHOLD_LOW', (0, 0, 0)))
        self.black_high = np.asarray(getattr(cfg, 'BLACK_HSV_THRESHOLD_HIGH', (179, 255, 60)))
        self.car_min_area_px = getattr(cfg, 'CAR_MIN_AREA_PX', 50)
        # Unbounded by default, same reasoning as CONE_MAX_WIDTH_PX in
        # obstacle_avoider.py: a close real car legitimately fills much of
        # the frame width, so a width cap borrowed from a line tracker's
        # shape filter would eventually reject exactly the closest,
        # most-urgent detections - see that constant's comment there for
        # the two on-car incidents this already caused for the cone.
        self.car_max_width_px = getattr(cfg, 'CAR_MAX_WIDTH_PX', 100000)
        self.car_min_aspect_ratio = getattr(cfg, 'CAR_MIN_ASPECT_RATIO', 0.0)

        self.white_right_of_yellow = getattr(cfg, 'WHITE_RIGHT_OF_YELLOW', True)
        self.lane_margin_px = getattr(cfg, 'CAR_LANE_MARGIN_PX', 10)
        # How far PAST the yellow centerline, into the opposite lane, the
        # trigger zone extends - see _encroachment_bound and the class
        # docstring's "Triggering early" section. Untuned starting guess,
        # same caveat as everything else here - needs a swept range of
        # real closing-speed footage to tune properly, not just synthetic
        # test images.
        self.early_margin_px = getattr(cfg, 'CAR_EARLY_MARGIN_PX', 60)
        self.trigger_frames = getattr(cfg, 'CAR_TRIGGER_FRAMES', 2)
        # Rejects a detection sitting at the extreme image edges - the same
        # place off-track background clutter (recycling bin, kiosk sign)
        # sat in the cone detector's real footage (see
        # obstacle_avoider.py's class docstring) - cheap enough to keep on
        # by default even though it's unlikely to matter much given the
        # encroachment-zone gating already does most of the real work here.
        self.frame_edge_margin_px = getattr(cfg, 'CAR_FRAME_EDGE_MARGIN_PX', 5)

        # Frame-over-frame blob-growth requirement (Decision 2 option C) -
        # off by default per the design doc ("held in reserve... only
        # worth building if [color-key + shape/size filter] isn't enough in
        # on-car testing"). self.car_growing is still computed and
        # published every frame regardless, so it's visible for tuning
        # before flipping this on.
        self.require_growth = getattr(cfg, 'CAR_REQUIRE_GROWTH', False)
        self.growth_window_frames = getattr(cfg, 'CAR_GROWTH_WINDOW_FRAMES', 5)
        self.growth_min_px_per_frame = getattr(cfg, 'CAR_GROWTH_MIN_PX_PER_FRAME', 3.0)
        self._area_history = deque(maxlen=self.growth_window_frames)

        # Re-smoothed/clamped copy of lane/width_px, independent of
        # ObstacleAvoider's own copy of the same idea (separate Part
        # instance, separate state) - see CONE_LANE_WIDTH_SMOOTHING_ALPHA's
        # comment in obstacle_avoider.py for why raw lane/width_px is too
        # noisy to trust directly for either the in-lane test or the
        # maneuver's steering target.
        self.lane_width_smoothing_alpha = getattr(cfg, 'CAR_LANE_WIDTH_SMOOTHING_ALPHA', 0.1)
        self.lane_width_min_px = getattr(cfg, 'CAR_LANE_WIDTH_MIN_PX', 80)
        self.lane_width_max_px = getattr(cfg, 'CAR_LANE_WIDTH_MAX_PX', 300)
        self.smoothed_lane_width_px = None

        self.log_interval_frames = getattr(cfg, 'CAR_LOG_INTERVAL_FRAMES', 10)

        # Avoidance maneuver - own dedicated PID (never LaneFollower's
        # pid_st, never ObstacleAvoider's avoid_pid), same default gains as
        # the main steering PID, same reasoning as
        # obstacle_avoider.py's AVOID_PID_*.
        self.avoid_pid = PID(
            Kp=getattr(cfg, 'AVOID_CAR_PID_P', getattr(cfg, 'PID_P', -0.01)),
            Ki=getattr(cfg, 'AVOID_CAR_PID_I', getattr(cfg, 'PID_I', 0.0)),
            Kd=getattr(cfg, 'AVOID_CAR_PID_D', getattr(cfg, 'PID_D', -0.0001)),
        )
        self.avoid_pid.output_limits = (-1.0, 1.0)
        self.avoid_error_saturation_px = getattr(cfg, 'AVOID_CAR_ERROR_SATURATION_PX',
                                                  getattr(cfg, 'LANE_ERROR_SATURATION_PX', 80))
        self.avoid_position_rate_limit_px = getattr(cfg, 'AVOID_CAR_POSITION_RATE_LIMIT_PX',
                                                     getattr(cfg, 'LANE_POSITION_RATE_LIMIT_PX', 25))
        self._avoid_last_position = None
        self.avoid_steering_rate_limit = getattr(cfg, 'AVOID_CAR_STEERING_RATE_LIMIT', 0.04)
        self.avoid_yellow_max_jump_px = getattr(cfg, 'AVOID_CAR_YELLOW_MAX_JUMP_PX',
                                                 getattr(cfg, 'MAX_JUMP_PIXELS', 40))
        self.avoid_yellow_reacquire_frames = getattr(cfg, 'AVOID_CAR_YELLOW_REACQUIRE_FRAMES',
                                                       getattr(cfg, 'REACQUIRE_AFTER_FRAMES', 15))
        self._avoid_last_yellow_x = None
        self._avoid_yellow_reject_frames = 0
        self.avoid_target_threshold = getattr(cfg, 'LANE_TARGET_THRESHOLD', 10)
        self.throttle_step = getattr(cfg, 'THROTTLE_STEP', 0.05)
        self.throttle_min = getattr(cfg, 'THROTTLE_MIN', 0.1)
        self.throttle_max = getattr(cfg, 'THROTTLE_MAX', 0.3)
        self.max_lost_frames = getattr(cfg, 'MAX_LOST_FRAMES', 40)
        self.lost_steering_decay = getattr(cfg, 'LOST_STEERING_DECAY', 0.85)

        self.avoiding = False           # latches True permanently once triggered
        self.avoid_target_pixel = None
        self.avoid_throttle = None
        self.avoid_steering = 0.0
        self.avoid_lost_frames = 0

        # public detection state, updated every run() call
        self.car_x = None
        self.car_area = 0
        self.car_growing = False
        self.car_in_our_lane = False
        self.car_encroaching = False
        self.lane_geometry_available = False
        self.car_ready = False
        self.car_detected = False
        self._pending_frames = 0

        self._was_raw_detected = False
        self._frame_count = 0
        self._warned_no_cam_img = False

        logger.info(
            f"[car_avoider] CarAvoider active - scanning rows "
            f"[{self.scan_y},{self.scan_y + self.scan_height}) for black "
            f"HSV={tuple(self.black_low.tolist())}-{tuple(self.black_high.tolist())}, "
            f"early trigger margin={self.early_margin_px}px past the yellow centerline"
        )

    def _open(self, mask):
        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _smoothed_lane_width(self, lane_width_px):
        if lane_width_px is None:
            return self.smoothed_lane_width_px

        clamped = min(max(lane_width_px, self.lane_width_min_px), self.lane_width_max_px)
        if self.smoothed_lane_width_px is None:
            self.smoothed_lane_width_px = clamped
        else:
            self.smoothed_lane_width_px = (self.lane_width_smoothing_alpha * clamped
                                            + (1 - self.lane_width_smoothing_alpha) * self.smoothed_lane_width_px)
        return self.smoothed_lane_width_px

    def detect_car(self, band_hsv):
        '''
        input: band_hsv, HSV numpy array of the forward scan band
        output: (x, area, black_mask) - centroid x of the winning black blob
                (or None), its pixel area (0 if x is None), and the binary
                mask (returned for diagnostics)
        '''
        black_mask = self._open(cv2.inRange(band_hsv, self.black_low, self.black_high))
        x, area = _select_line_blob(black_mask, self.car_min_area_px, self.car_max_width_px,
                                     min_aspect_ratio=self.car_min_aspect_ratio, log_tag='car_black')
        return x, area, black_mask

    def _update_growth_history(self, x, area):
        '''
        Frame-over-frame blob-growth tracking (Decision 2 option C) - see
        the class docstring's "Detection" section. A gap in detection
        (x is None) clears the history rather than letting it span the
        gap: growth is only meaningful measured across CONSECUTIVE
        detections of what's presumably the same real object, not across
        an interruption that might be a different blob entirely.
        '''
        if x is None:
            self._area_history.clear()
            self.car_growing = False
            return

        self._area_history.append(area)
        if len(self._area_history) == self._area_history.maxlen:
            growth = self._area_history[-1] - self._area_history[0]
            frames = self._area_history.maxlen - 1
            self.car_growing = growth >= self.growth_min_px_per_frame * frames
        else:
            self.car_growing = False

    def _log_raw_detection(self):
        raw_detected = self.car_x is not None
        self._frame_count += 1
        heartbeat_due = self._frame_count % self.log_interval_frames == 0

        if raw_detected and not self._was_raw_detected:
            lane_note = "IN our lane" if self.car_in_our_lane else \
                        ("encroaching toward our lane" if self.car_encroaching else "not yet a threat")
            logger.info(
                f"[car_avoider] black blob at x={self.car_x:.1f}, area={self.car_area}px "
                f"- {lane_note}, growing={self.car_growing} - "
                f"{'AVOIDING' if self.avoiding else ('SWERVE pending' if self.car_ready else 'hold lane')}"
            )
        elif not raw_detected and self._was_raw_detected:
            logger.info("[car_avoider] black blob no longer visible in scan band")
        elif heartbeat_due and raw_detected:
            logger.info(
                f"[car_avoider] black blob still at x={self.car_x:.1f} area={self.car_area}px "
                f"growing={self.car_growing} - {'AVOIDING' if self.avoiding else 'hold lane'}"
            )

        self._was_raw_detected = raw_detected

    def run(self, cam_img, yellow_x, white_x, lane_width_px, steering, throttle, cv_img=None,
            other_avoidance_active=False):
        '''
        main runloop
        input: cam_img, raw RGB camera frame; yellow_x, white_x, lane_width_px
               - LaneFollower's published lane geometry; steering, throttle -
               upstream pilot output, passed through unchanged until this
               maneuver triggers; cv_img, the display image to optionally
               draw on; other_avoidance_active - True while another
               avoidance maneuver (e.g. ObstacleAvoider's cone swerve) is
               already underway, so this part's own trigger waits rather
               than fighting it for the steering actuator (Decision 5).
        output: steering, throttle (overridden while self.avoiding), cv_img,
                car_detected
        '''
        if cam_img is None:
            if not self._warned_no_cam_img:
                logger.warning("[car_avoider] cam_img is None - cam/image_array not populated yet")
                self._warned_no_cam_img = True
            return steering, throttle, cv_img, self.car_detected

        band_rgb = cam_img[self.scan_y: self.scan_y + self.scan_height, :, :]
        if band_rgb.size == 0:
            logger.warning(f"Empty car scan slice at scan_y={self.scan_y}: "
                            f"cam_img shape={cam_img.shape}; check CAR_SCAN_Y/HEIGHT")
            return steering, throttle, cv_img, self.car_detected
        band_hsv = cv2.cvtColor(band_rgb, cv2.COLOR_RGB2HSV)

        lane_width_px = self._smoothed_lane_width(lane_width_px)
        our_lane = _lane_bounds(yellow_x, white_x, lane_width_px, self.white_right_of_yellow, other_lane=False)
        self.lane_geometry_available = our_lane[0] is not None

        self.car_x, self.car_area, black_mask = self.detect_car(band_hsv)

        frame_width = cam_img.shape[1]
        if self.car_x is not None and (self.car_x < self.frame_edge_margin_px
                                        or self.car_x > frame_width - self.frame_edge_margin_px):
            # background clutter at the extreme image edges - see class
            # docstring; treat exactly like no detection this frame
            self.car_x, self.car_area = None, 0

        self._update_growth_history(self.car_x, self.car_area)

        self.car_in_our_lane = _x_in_bounds(self.car_x, our_lane, self.lane_margin_px)
        bound = _encroachment_bound(yellow_x, self.early_margin_px, self.white_right_of_yellow)
        self.car_encroaching = _past_encroachment_bound(self.car_x, bound, self.white_right_of_yellow)

        if self.lane_geometry_available:
            car_positioned = self.car_in_our_lane or self.car_encroaching
        else:
            # lane fully lost - same fallback ObstacleAvoider uses (see its
            # class docstring's "Exception" paragraph, added after
            # tub_7_26-07-27): car_in_our_lane/car_encroaching are
            # structurally False whenever yellow_x is None, so a car that
            # already lost the lane could otherwise never trigger no matter
            # how obviously an oncoming car sat dead ahead.
            car_positioned = self.car_x is not None

        self.car_ready = (self.car_x is not None and car_positioned
                           and (not self.require_growth or self.car_growing))

        if self.car_ready:
            self._pending_frames += 1
        else:
            self._pending_frames = 0

        was_detected = self.car_detected
        self.car_detected = self._pending_frames >= self.trigger_frames
        if self.car_detected and not self.avoiding and not other_avoidance_active:
            self.avoiding = True
            logger.info(
                f"[car_avoider] oncoming car confirmed at x={self.car_x:.1f} "
                f"(held {self._pending_frames} frames) - swerving to the other lane; "
                f"will NOT return to the original lane afterward"
            )
        elif was_detected and not self.car_detected:
            logger.info("[car_avoider] oncoming car no longer positioned to trigger avoidance")

        if self.avoiding:
            steering, throttle = self._avoid_step(cam_img, yellow_x, lane_width_px, throttle)

        self._log_raw_detection()

        if self.overlay_image and cv_img is not None:
            cv_img = self.overlay_display(cv_img)

        return steering, throttle, cv_img, self.car_detected

    def _avoid_step(self, cam_img, yellow_x, lane_width_px, throttle):
        '''
        One frame of the avoidance maneuver - see obstacle_avoider.py's
        _avoid_step for the detailed rationale behind each defense here;
        this is deliberately the same mechanism (soft-saturated PID error,
        target position/steering rate limits, a plausibility gate on
        yellow_x jumps, passive hold/decay on lane loss) applied with this
        part's own state, since those defenses were each added after a real
        on-car failure of exactly this maneuver shape and skipping any one
        of them here would just reproduce that failure a second time.
        '''
        if self.avoid_target_pixel is None:
            self.avoid_target_pixel = cam_img.shape[1] / 2.0
            self.avoid_pid.setpoint = self.avoid_target_pixel
        if self.avoid_throttle is None:
            self.avoid_throttle = throttle

        if yellow_x is not None and self._avoid_last_yellow_x is not None:
            jump = abs(yellow_x - self._avoid_last_yellow_x)
            if (jump > self.avoid_yellow_max_jump_px
                    and self._avoid_yellow_reject_frames < self.avoid_yellow_reacquire_frames):
                self._avoid_yellow_reject_frames += 1
                yellow_x = None
            else:
                self._avoid_yellow_reject_frames = 0
                self._avoid_last_yellow_x = yellow_x
        elif yellow_x is not None:
            self._avoid_last_yellow_x = yellow_x
            self._avoid_yellow_reject_frames = 0

        other_center = _other_lane_center(yellow_x, lane_width_px, self.white_right_of_yellow)

        if other_center is None:
            self.avoid_lost_frames += 1
            self.avoid_steering *= self.lost_steering_decay
            if self.avoid_lost_frames > 15:
                self._avoid_last_position = None
                self._avoid_last_yellow_x = None
                self._avoid_yellow_reject_frames = 0
            if self.avoid_lost_frames > self.max_lost_frames:
                self.avoid_throttle = max(self.avoid_throttle - self.throttle_step, 0.0)
            return self.avoid_steering, self.avoid_throttle

        self.avoid_lost_frames = 0

        if self.avoid_position_rate_limit_px > 0 and self._avoid_last_position is not None:
            step = other_center - self._avoid_last_position
            if abs(step) > self.avoid_position_rate_limit_px:
                other_center = self._avoid_last_position + math.copysign(
                    self.avoid_position_rate_limit_px, step)
        self._avoid_last_position = other_center

        error_px = other_center - self.avoid_target_pixel
        k = self.avoid_error_saturation_px
        soft_error_px = k * math.tanh(error_px / k)
        pid_steering = self.avoid_pid(self.avoid_target_pixel + soft_error_px)

        if self.avoid_steering_rate_limit > 0:
            delta = pid_steering - self.avoid_steering
            if abs(delta) > self.avoid_steering_rate_limit:
                pid_steering = self.avoid_steering + math.copysign(
                    self.avoid_steering_rate_limit, delta)
        self.avoid_steering = pid_steering

        if abs(other_center - self.avoid_target_pixel) > self.avoid_target_threshold:
            self.avoid_throttle = max(self.avoid_throttle - self.throttle_step, self.throttle_min)
        else:
            self.avoid_throttle = min(self.avoid_throttle + self.throttle_step, self.throttle_max)

        return self.avoid_steering, self.avoid_throttle

    def overlay_display(self, cv_img):
        y0, y1 = self.scan_y, self.scan_y + self.scan_height
        if self.car_x is not None:
            if self.car_ready:
                color = (255, 0, 0)
            elif self.car_in_our_lane or self.car_encroaching:
                color = (150, 60, 60)
            else:
                color = (150, 150, 150)
            cv2.rectangle(cv_img, (int(self.car_x) - 8, y0), (int(self.car_x) + 8, y1),
                          color=color, thickness=2)
        label = f"CAR:{self.car_detected}"
        if self.avoiding:
            label += " AVOIDING"
        cv2.putText(cv_img, label, org=(10, cv_img.shape[0] - 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))
        return cv_img
