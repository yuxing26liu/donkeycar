import logging
import math

import cv2
import numpy as np
from simple_pid import PID

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
        if yellow_x is None or lane_width_px is None:
            return None, None
        far_edge = yellow_x - sign * lane_width_px
        return tuple(sorted((yellow_x, far_edge)))

    if yellow_x is not None and white_x is not None:
        return tuple(sorted((yellow_x, white_x)))
    if yellow_x is not None:
        if lane_width_px is None:
            return None, None
        edge = yellow_x + sign * lane_width_px
        return tuple(sorted((yellow_x, edge)))
    if white_x is not None:
        if lane_width_px is None:
            return None, None
        edge = white_x - sign * lane_width_px
        return tuple(sorted((white_x, edge)))
    return None, None


def _other_lane_center(yellow_x, lane_width_px, white_right_of_yellow):
    '''
    The other lane's center pixel - the steering target once a cone
    avoidance maneuver is underway (see ObstacleAvoider._avoid_step).

    Mirrors LaneFollower._lane_center's single-anchor extrapolation, and
    only ever derives from yellow_x, same as _lane_bounds(..., other_lane=True)
    above: yellow is the one line shared by both lanes, so it's the only
    reliable anchor for the far lane's position (white_x is our own lane's
    *outer* edge - two extrapolation hops removed from the other lane's
    center, so it's not used here even when visible). Returns None if
    yellow_x isn't visible - see Decision 3 in project_doc/obstacle_avoidance.md:
    no reliable anchor means no maneuver steering this frame, not a guess.
    '''
    if yellow_x is None or lane_width_px is None:
        return None
    sign = 1.0 if white_right_of_yellow else -1.0
    return yellow_x - sign * lane_width_px / 2.0


class ObstacleAvoider:
    '''
    Obstacle avoidance layered on top of LaneFollower (donkeycar/parts/lane_follower.py)
    for the "two-way road navigation" mission (see CLAUDE.md). See
    project_doc/obstacle_avoidance.md for the full decision-by-decision design
    (detection strategy for each obstacle type, and why) and current build
    status - this class is being built incrementally, one detector at a time.

    Phase 1: detect the traffic cone's position by color - either its
    ground marker (a blue tape square laid down at the cone's spot, per
    the original design in project_doc/obstacle_avoidance.md Decision 1)
    or the cone's own orange body. Both color-keyed detectors run every
    frame; whichever produces the larger valid blob wins (see
    detect_cone). Orange detection was added after tub_33_26-07-24 (real
    on-car footage) showed cones placed on the track with **no** blue tape
    marker under them at all - the recorded frames only ever matched blue
    on background clutter (a recycling bin and a kiosk sign at the image
    edges), never on the track surface - while the cones themselves are
    clearly, consistently orange. BLUE_HSV_THRESHOLD_LOW/HIGH stays wired
    up (project_doc's Decision 1 option C, "union of masks") in case a
    tape marker is used on some other track/lap.

    Phase 2: once a cone is confirmed in our lane (self.cone_detected
    holding for CONE_TRIGGER_FRAMES consecutive frames), steer toward the
    other lane's center (_other_lane_center) using a dedicated PID
    (self.avoid_pid - never LaneFollower's own pid_st, so the two loops'
    integral state can't corrupt each other) instead of passing
    pilot/steering/pilot/throttle through unchanged - see _avoid_step.
    The maneuver latches permanently once triggered (self.avoiding): by
    this project's current design the car does **not** swerve back to its
    original lane once the cone is cleared, it just keeps driving centered
    in whichever lane it ends up in for the rest of the drive - simpler
    and lower-risk than a return maneuver, and this track only has the one
    cone to clear. See _avoid_step for the passive fallback if lane
    geometry is lost mid-maneuver.

    Not yet implemented (see project_doc/obstacle_avoidance.md "Next steps"):
    detecting the oncoming car (planned: color-key its black wheels/front,
    decision 2 in the design doc).

    Zero changes to lane_follower.py: this part is purely downstream of it,
    reusing its already-published per-frame outputs (lane/yellow_x,
    lane/white_x, lane/width_px - see LaneFollower's own class docstring,
    which anticipated exactly this) instead of re-deriving lane geometry,
    and reusing its `_select_line_blob` connected-component shape filter
    for its own color-blob detection instead of a second implementation.

    Detection method: color-keyed in HSV (BLUE_HSV_THRESHOLD_LOW/HIGH and
    ORANGE_HSV_THRESHOLD_LOW/HIGH), the same technique lane_follower.py's
    yellow _LineTracker uses and for the same reason - a solid, saturated
    color is a far more reliable signal than shape alone, and a positive
    color match (vs. e.g. "anything not already known") is naturally robust
    to background clutter like leaves/debris on the track, which won't be
    blue or orange. Restricted to a single
    forward scan band (CONE_SCAN_Y/CONE_SCAN_HEIGHT) - the same slice-based
    approach every CV part in this codebase uses - both so a blob is only
    ever evaluated against the track's actual pixel-x extent at that row
    (background outside the track, e.g. the building/planter/recycling bin
    visible in the reference photos, never enters the candidate pool) and
    so there's lead distance to react before the marker reaches
    LaneFollower's own nearer scan rows.

    A detection only "counts" (self.cone_ready) if it falls within OUR
    lane's pixel bounds (_lane_bounds, with a small LANE_SHIFT_MARGIN_PX
    margin) AND is roughly centered in the raw image (_is_centered,
    CONE_CENTER_MARGIN_PX) AND is close enough to matter (_is_close,
    CONE_CLOSE_MIN_AREA_PX) - and self.cone_detected only latches once
    cone_ready holds for CONE_TRIGGER_FRAMES consecutive frames. The
    centered/close gates were added after on-car testing showed a cone in
    the OTHER lane still latching cone_detected: lane/width_px can be noisy
    enough (see project_doc/obstacle_avoidance.md's "noisy lane width"
    entries) to fool the lane-bounds test on its own, but a far-lane cone
    still reads as off-center and/or small in the raw frame regardless of
    what the lane-bounds test thinks, so these two checks hold even when
    lane geometry is wrong. A cone marked in the other lane is detected but
    doesn't count, and a one-frame misclassification (e.g. a sliver of
    glare) can't flip the flag by itself. This mirrors the "don't react to
    a single frame" caution lane_follower.py's continuity gating uses for
    the dashed yellow line, applied here to noise rejection instead of
    dash-gap tolerance.

    Exception: when LaneFollower publishes NO lane geometry at all this
    frame (yellow_x and white_x both None - self.lane_geometry_available
    False), cone_in_our_lane is structurally False (see _x_in_bounds) and
    can never be satisfied, so requiring it would mean a car that has
    already lost the lane can never trigger avoidance no matter how
    obviously a cone sits dead ahead. Added after tub_7_26-07-27: the car
    drove straight into a cone with no swerve while already off the marked
    lane from an earlier incident. In that specific case cone_ready falls
    back to centered AND close alone - see run().
    '''

    def __init__(self, cfg):
        self.overlay_image = getattr(cfg, 'OVERLAY_IMAGE', False)

        self.scan_y = getattr(cfg, 'CONE_SCAN_Y', 60)
        self.scan_height = getattr(cfg, 'CONE_SCAN_HEIGHT', 30)
        self.morph_kernel_size = getattr(cfg, 'MORPH_KERNEL_SIZE', 3)

        self.blue_low = np.asarray(getattr(cfg, 'BLUE_HSV_THRESHOLD_LOW', (95, 100, 60)))
        self.blue_high = np.asarray(getattr(cfg, 'BLUE_HSV_THRESHOLD_HIGH', (130, 255, 255)))
        # calibrated against real cone pixels sampled from tub_33_26-07-24
        # (two cones, two frames, saturation-filtered) - not a blind guess
        # like BLUE_HSV_THRESHOLD above; see class docstring
        self.orange_low = np.asarray(getattr(cfg, 'ORANGE_HSV_THRESHOLD_LOW', (0, 90, 60)))
        self.orange_high = np.asarray(getattr(cfg, 'ORANGE_HSV_THRESHOLD_HIGH', (18, 255, 255)))
        self.cone_min_area_px = getattr(cfg, 'CONE_MIN_AREA_PX', 80)
        # Effectively unbounded by default - see the CONE_MAX_WIDTH_PX
        # comment in cfg_cv_control.py for the two on-car incidents that
        # got this raised twice (250 -> 400 after tub_41_26-07-24, then
        # this default after tub_7_26-07-27 showed even 400 still rejects
        # a real cone at the exact moment avoidance matters most: closer
        # than ~(400/IMAGE_W) of the frame width away). A width cap makes
        # sense for a LINE tracker (rejecting a wide sunlit patch of
        # pavement pretending to be a paint stripe) but not for a real 3D
        # cone, which legitimately fills the entire frame width at close
        # range - unlike a painted line, "wider" is stronger evidence of a
        # real cone, not weaker, so this only exists as a knob for a track
        # that specifically needs one, not as a default safety net.
        self.cone_max_width_px = getattr(cfg, 'CONE_MAX_WIDTH_PX', 100000)

        self.white_right_of_yellow = getattr(cfg, 'WHITE_RIGHT_OF_YELLOW', True)
        self.lane_margin_px = getattr(cfg, 'LANE_SHIFT_MARGIN_PX', 10)
        self.cone_trigger_frames = getattr(cfg, 'CONE_TRIGGER_FRAMES', 2)

        # Added after on-car testing showed a cone sitting in the OTHER lane
        # still latching cone_detected: the lane-bounds test above
        # (_lane_bounds/_x_in_bounds) depends entirely on LaneFollower's
        # published yellow_x/white_x/lane_width_px, which project_doc's
        # "noisy lane width" incident already showed can swing far enough in
        # a single frame to misjudge which lane a detection is actually in -
        # smoothing (_smoothed_lane_width) narrowed that window but doesn't
        # close it. These two gates are independent of lane geometry
        # entirely, so they hold even when yellow_x/white_x/lane_width_px are
        # themselves wrong: a cone worth swerving for is one we're about to
        # hit, which from a forward-facing camera means it reads as roughly
        # centered in the raw image (our lane, being the lane the camera
        # looks down, sits close to image-center; the other lane sits well
        # off to the side - see _is_centered) AND its blob has grown large
        # enough to mean "close" (a real 3D object's apparent size grows as
        # it nears the camera, same reasoning CONE_MAX_WIDTH_PX's docstring
        # already uses - see _is_close). Both are required in addition to,
        # not instead of, the lane-bounds test - a cheap extra filter, not a
        # replacement for a working one.
        self.cone_center_margin_px = getattr(cfg, 'CONE_CENTER_MARGIN_PX', 60)
        # CAUTION: 200 is only just above CONE_MIN_AREA_PX's 80px noise
        # floor, not calibrated against how large a real close cone gets -
        # tub_41_26-07-24's on-car log recorded a genuinely close, centered
        # cone's blob at ~8654px (see project_doc/obstacle_avoidance.md,
        # the CONE_MAX_WIDTH_PX incident), two orders of magnitude bigger
        # than this default. 200 is a deliberately low starting floor (so
        # this gate doesn't accidentally reject real detections before
        # there's real near/far footage to calibrate against), not a
        # "close" threshold in the sense a human would mean it - watch the
        # `area=` value now printed in _log_raw_detection against real
        # distances on the car and raise this once there's data.
        self.cone_close_min_area_px = getattr(cfg, 'CONE_CLOSE_MIN_AREA_PX', 200)

        # lane/width_px (published by LaneFollower) swung 150->392->133px
        # across a handful of frames during on-car avoidance testing - real
        # lane width can't change that fast, so both the "is this cone in
        # our lane" test and the avoid-maneuver's steering target (both
        # derived from this value, see _lane_bounds/_other_lane_center)
        # were reacting to noise, not real geometry. Re-smoothed and
        # clamped here, independently of whatever smoothing LaneFollower
        # already does internally (zero changes to lane_follower.py) -
        # see _smoothed_lane_width.
        self.lane_width_smoothing_alpha = getattr(cfg, 'CONE_LANE_WIDTH_SMOOTHING_ALPHA', 0.1)
        self.lane_width_min_px = getattr(cfg, 'CONE_LANE_WIDTH_MIN_PX', 80)
        self.lane_width_max_px = getattr(cfg, 'CONE_LANE_WIDTH_MAX_PX', 300)
        self.smoothed_lane_width_px = None

        self.log_interval_frames = getattr(cfg, 'CONE_LOG_INTERVAL_FRAMES', 10)

        # Avoidance maneuver (Phase 2, see class docstring): a dedicated PID,
        # never LaneFollower's own pid_st, so the two loops' integral state
        # can't corrupt each other. Defaults to the same gains as the main
        # steering PID (cfg.PID_P/I/D) since it's steering the same physical
        # car/camera toward a target pixel - only override AVOID_PID_* if
        # on-car testing shows the maneuver needs different gains.
        self.avoid_pid = PID(
            Kp=getattr(cfg, 'AVOID_PID_P', getattr(cfg, 'PID_P', -0.01)),
            Ki=getattr(cfg, 'AVOID_PID_I', getattr(cfg, 'PID_I', 0.0)),
            Kd=getattr(cfg, 'AVOID_PID_D', getattr(cfg, 'PID_D', -0.0001)),
        )
        self.avoid_pid.output_limits = (-1.0, 1.0)
        # Mirrors LaneFollower's own two defenses against a single noisy
        # frame driving the PID straight to full steering lock (see
        # lane_follower.py's module docstring, points 1 and the
        # position_rate_limit_px comment in its __init__) - added here after
        # on-car testing showed the avoid maneuver swerving hard enough to
        # leave the track entirely. other_center (the avoid target, derived
        # from yellow_x/lane_width_px - see _other_lane_center) can jump
        # between frames the same way LaneFollower's own lane-center
        # estimate can; without either defense the raw jump goes straight
        # into the P term.
        self.avoid_error_saturation_px = getattr(cfg, 'AVOID_ERROR_SATURATION_PX',
                                                  getattr(cfg, 'LANE_ERROR_SATURATION_PX', 80))
        self.avoid_position_rate_limit_px = getattr(cfg, 'AVOID_POSITION_RATE_LIMIT_PX',
                                                     getattr(cfg, 'LANE_POSITION_RATE_LIMIT_PX', 25))
        self._avoid_last_position = None
        # Steering rate limit - added after user feedback that the maneuver
        # turned too sharply into the other lane, leaving LaneFollower(3)
        # too little time to pick up the new lane's own white boundary
        # before the turn was already mostly complete. avoid_position_rate_limit_px
        # (above) only slews the PID's *target*; the PID itself can still
        # jump straight to a large output the instant the maneuver begins
        # (e.g. the first frame's error is already sizable). This caps how
        # much self.avoid_steering itself may change per frame, on top of
        # that - a deliberately slow turn-in so the car is still looking
        # roughly down the lane (rather than already committed to a hard
        # turn) while the new white line comes into the scan band. 0 disables.
        self.avoid_steering_rate_limit = getattr(cfg, 'AVOID_STEERING_RATE_LIMIT', 0.04)
        # reuse LaneFollower's own turning/straight throttle scheme
        # (cfg.LANE_TARGET_THRESHOLD/THROTTLE_STEP/THROTTLE_MIN/THROTTLE_MAX)
        # rather than inventing new tuning knobs - see _avoid_step
        self.avoid_target_threshold = getattr(cfg, 'LANE_TARGET_THRESHOLD', 10)
        self.throttle_step = getattr(cfg, 'THROTTLE_STEP', 0.05)
        self.throttle_min = getattr(cfg, 'THROTTLE_MIN', 0.1)
        self.throttle_max = getattr(cfg, 'THROTTLE_MAX', 0.3)
        # reuse LaneFollower's own sustained-loss handling constants for the
        # same passive fallback (Decision 3) applied to the avoid maneuver
        self.max_lost_frames = getattr(cfg, 'MAX_LOST_FRAMES', 40)
        self.lost_steering_decay = getattr(cfg, 'LOST_STEERING_DECAY', 0.85)

        self.avoiding = False           # latches True permanently once a cone
                                         # avoidance triggers - see class docstring
        self.avoid_target_pixel = None  # resolved to image center on first use
        self.avoid_throttle = None      # seeded from the pass-through throttle
                                         # the frame avoidance begins
        self.avoid_steering = 0.0
        self.avoid_lost_frames = 0

        # public detection state - what a future avoidance maneuver (or a
        # test) reads; updated every run() call
        self.cone_x = None              # raw detected x this frame (any lane), or None
        self.cone_area = 0              # winning blob's pixel area this frame, or 0
        self.cone_color = None          # 'blue tape' or 'orange cone' - which detector
                                         # won this frame, or None if cone_x is None
        self.cone_in_our_lane = False   # raw in-our-lane test this frame (lane geometry), pre-debounce
        self.cone_centered = False      # raw centered-in-frame test this frame (see __init__)
        self.cone_close = False         # raw close-enough test this frame (see __init__)
        self.lane_geometry_available = False  # yellow_x or white_x published this frame - see run()
        self.cone_ready = False         # normally in_our_lane AND centered AND close; centered AND
                                         # close alone when lane_geometry_available is False - see run()
        self.cone_detected = False      # debounced: True once cone_ready has held
                                         # for cone_trigger_frames consecutive frames
        self._pending_frames = 0

        # diagnostic-logging state only (see _log_raw_detection /
        # _warn_if_lane_geometry_missing) - not used for detection itself
        self._was_raw_detected = False
        self._was_raw_color = None
        self._frame_count = 0
        self._warned_no_lane_geometry = False
        self._warned_no_cam_img = False

        # one-time startup line - confirms this part is actually alive and
        # scanning as soon as `python manage.py drive` constructs it,
        # independent of whether anything is detected yet. Added because the
        # prior silent failure mode (HAVE_OBSTACLE_AVOIDANCE=False, or an
        # out-of-date manage.py that never wires this part in at all) is
        # otherwise indistinguishable from "wired in but nothing detected
        # yet" - see project_doc/obstacle_avoidance.md.
        logger.info(
            f"[cone_tape] ObstacleAvoider active - scanning rows "
            f"[{self.scan_y},{self.scan_y + self.scan_height}) for blue tape "
            f"HSV={tuple(self.blue_low.tolist())}-{tuple(self.blue_high.tolist())} or "
            f"orange cone HSV={tuple(self.orange_low.tolist())}-{tuple(self.orange_high.tolist())}"
        )

    def _open(self, mask):
        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _smoothed_lane_width(self, lane_width_px):
        '''
        EMA-smoothed, clamped version of LaneFollower's published
        lane/width_px - see the CONE_LANE_WIDTH_SMOOTHING_ALPHA comment in
        __init__ for why: the raw value can swing far faster than a real
        lane's width does, and that noise otherwise feeds directly into
        both the in-our-lane test (_lane_bounds) and the avoid maneuver's
        steering target (_other_lane_center). Called every run(), not just
        while avoiding, so the estimate is already settled by the time a
        maneuver starts instead of starting cold.

        input: lane_width_px, this frame's raw lane/width_px (or None)
        output: the smoothed width in pixels, or None if nothing has ever
                been observed yet
        '''
        if lane_width_px is None:
            return self.smoothed_lane_width_px

        clamped = min(max(lane_width_px, self.lane_width_min_px), self.lane_width_max_px)
        if self.smoothed_lane_width_px is None:
            self.smoothed_lane_width_px = clamped
        else:
            self.smoothed_lane_width_px = (self.lane_width_smoothing_alpha * clamped
                                            + (1 - self.lane_width_smoothing_alpha) * self.smoothed_lane_width_px)
        return self.smoothed_lane_width_px

    def detect_cone(self, band_hsv):
        '''
        input: band_hsv, HSV numpy array of the forward scan band
        output: (x, area, color_label, blue_mask, orange_mask) - x position
                in pixels of the winning blob's centroid (or None if neither
                color produced one), its pixel area (0 if x is None; see
                _is_close's docstring for why area is what run() uses as a
                "how close is it" proxy), color_label is 'blue tape' or
                'orange cone' (whichever won) or None, and both binary color
                masks (returned for _describe_mask's diagnostics below, so
                they aren't recomputed twice per frame)

        Runs both color detectors every frame - a real cone should only ever
        match one of them, so this isn't "wasted" work, it's just not
        assuming in advance which marking this particular track/lap uses.
        If both somehow produce a valid blob in the same frame (e.g. a
        tape-marked cone), the larger one by area wins - same "biggest blob
        wins" contract _select_line_blob already uses within a single mask.
        '''
        blue_mask = self._open(cv2.inRange(band_hsv, self.blue_low, self.blue_high))
        orange_mask = self._open(cv2.inRange(band_hsv, self.orange_low, self.orange_high))

        blue_x, blue_area = _select_line_blob(blue_mask, self.cone_min_area_px, self.cone_max_width_px,
                                               min_aspect_ratio=0.0, log_tag='cone_tape_blue')
        orange_x, orange_area = _select_line_blob(orange_mask, self.cone_min_area_px, self.cone_max_width_px,
                                                    min_aspect_ratio=0.0, log_tag='cone_tape_orange')

        candidates = [(area, x, label) for area, x, label in
                      ((blue_area, blue_x, 'blue tape'), (orange_area, orange_x, 'orange cone'))
                      if x is not None]
        if not candidates:
            return None, 0, None, blue_mask, orange_mask
        area, x, color_label = max(candidates)
        return x, area, color_label, blue_mask, orange_mask

    def _describe_mask(self, blue_mask, orange_mask):
        '''
        Diagnostics-only, independent of the shape filter in _select_line_blob:
        that function only logs *why* a blob was rejected (too small/too wide)
        when the root logger is at DEBUG - which on this car would also spam
        LaneFollower's own per-frame yellow/white rejections. This reports the
        same thing for both cone-color masks, at the default INFO level, so
        "why isn't it detecting the cone" is answerable from a normal `python
        manage.py drive` run: was either color threshold ever matched at all
        (raw_pixel_count), and if so, did the largest blob fail the shape
        filter and why.

        output: {'blue tape': (raw_pixel_count, reasons), 'orange cone': (...)}
                reasons is a list of strings describing why that color's
                largest raw blob (if any) was rejected by the shape filter,
                or [] if either no blob exists or one passed
        '''
        def describe(mask):
            raw_pixel_count = int(np.count_nonzero(mask))
            if raw_pixel_count == 0:
                return raw_pixel_count, []

            num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            best_label, best_area = None, 0
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > best_area:
                    best_area, best_label = area, label
            if best_label is None:
                return raw_pixel_count, []

            width = stats[best_label, cv2.CC_STAT_WIDTH]
            reasons = []
            if best_area < self.cone_min_area_px:
                reasons.append(f"largest blob area {best_area}px < CONE_MIN_AREA_PX {self.cone_min_area_px}")
            if width > self.cone_max_width_px:
                reasons.append(f"largest blob width {width}px > CONE_MAX_WIDTH_PX {self.cone_max_width_px}")
            return raw_pixel_count, reasons

        return {'blue tape': describe(blue_mask), 'orange cone': describe(orange_mask)}

    def _decide_action(self):
        '''
        Human-readable recommended action for the terminal diagnostics below -
        describes what run() actually does this frame (see _avoid_step),
        not a hypothetical.
        '''
        if self.avoiding:
            return f"AVOIDING - steering toward other lane (triggered by {self.cone_color or 'cone'}, stays this way for the rest of the drive)"
        if self.cone_detected:
            basis = "centered and close (no lane geometry to check)" if not self.lane_geometry_available \
                else "in our lane, centered, and close"
            return f"SWERVE - {self.cone_color} confirmed {basis} - steer toward other lane"
        if self.cone_ready:
            basis = "centered and close (no lane geometry to check)" if not self.lane_geometry_available \
                else "in our lane, centered, and close"
            return (f"{self.cone_color} candidate {basis} - confirming "
                     f"({self._pending_frames}/{self.cone_trigger_frames} frames) - hold lane for now")
        if not self.lane_geometry_available and self.cone_x is not None:
            reasons = []
            if not self.cone_centered:
                reasons.append("not centered in frame")
            if not self.cone_close:
                reasons.append("not close enough")
            return (f"{self.cone_color} candidate but {' / '.join(reasons)} "
                     f"(area={self.cone_area}px, no lane geometry to check) - hold lane for now")
        if self.cone_in_our_lane:
            reasons = []
            if not self.cone_centered:
                reasons.append("not centered in frame")
            if not self.cone_close:
                reasons.append("not close enough")
            return (f"{self.cone_color} candidate in our lane but {' / '.join(reasons)} "
                     f"(area={self.cone_area}px) - hold lane for now")
        if self.cone_x is not None:
            return f"{self.cone_color} blob seen but not in our lane - hold lane"
        return "no cone (blue tape or orange) visible - hold lane"

    def _x_in_bounds(self, x, bounds):
        lo, hi = bounds
        if x is None or lo is None:
            return False
        return lo - self.lane_margin_px <= x <= hi + self.lane_margin_px

    def _is_centered(self, x, frame_width):
        '''
        True if x sits within CONE_CENTER_MARGIN_PX of the raw image's
        horizontal center - see the CONE_CENTER_MARGIN_PX comment in
        __init__ for why this is checked independently of the lane-bounds
        test. Deliberately the image's center, not our lane's center: a
        forward-facing camera looking down our own lane already reads our
        lane as roughly centered in frame, while the other lane - the case
        this is meant to reject - sits well off to one side.
        '''
        if x is None:
            return False
        return abs(x - frame_width / 2.0) <= self.cone_center_margin_px

    def _is_close(self, area):
        '''
        True if the winning blob's pixel area meets CONE_CLOSE_MIN_AREA_PX -
        see the CONE_CLOSE_MIN_AREA_PX comment in __init__. A real 3D cone's
        apparent size grows as it nears the camera (the same reasoning
        CONE_MAX_WIDTH_PX's docstring already relies on), so area is a cheap
        proxy for distance without needing a second, nearer scan row.
        '''
        return area >= self.cone_close_min_area_px

    def _sample_color(self, band_rgb, band_hsv, x, radius=4):
        '''
        Mean HSV/RGB over a small patch centered on a detected x AND on the
        scan band's vertical midline, for the terminal diagnostics below -
        lets BLUE_HSV_THRESHOLD_LOW/HIGH be checked against what the camera
        is actually seeing on the car, without needing to pull frames off
        the Pi first. Deliberately a small patch, not the full band height:
        the tape marker may not fill CONE_SCAN_HEIGHT, and averaging over
        rows outside it would blend in the gray track surface and wash out
        the reported color - this stays inside the marker as long as it
        crosses the band's vertical midline, which is the same assumption
        CONE_SCAN_Y/HEIGHT being sized to the marker already makes.
        '''
        h, w = band_hsv.shape[:2]
        x0, x1 = max(0, int(round(x)) - radius), min(w, int(round(x)) + radius + 1)
        y_mid = h // 2
        y0, y1 = max(0, y_mid - radius), min(h, y_mid + radius + 1)
        mean_hsv = band_hsv[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        mean_rgb = band_rgb[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        return mean_hsv, mean_rgb

    def _log_raw_detection(self, band_rgb, band_hsv, blue_mask, orange_mask):
        '''
        Terminal diagnostics, separate from run()'s in-lane/debounce gating
        (already settled by the time this runs, see run()): prints whenever
        a blue-tape or orange-cone blob enters/leaves the scan band, with
        the actually-sampled HSV/RGB color value - what to watch when tuning
        BLUE_HSV_THRESHOLD_LOW/HIGH, ORANGE_HSV_THRESHOLD_LOW/HIGH, or
        CONE_SCAN_Y/HEIGHT against the real cone/tape on the car. It fires
        regardless of whether lane geometry is available, so it still
        confirms the color detectors themselves are working even if
        lane/yellow_x etc. turn out not to be wired (see
        _warn_if_lane_geometry_missing).

        Also prints a heartbeat every CONE_LOG_INTERVAL_FRAMES frames
        *regardless* of whether anything is detected, so `python manage.py
        drive`'s terminal always shows current status + recommended action
        (this is the actual per-run() answer to "is it seeing a cone and
        what would it do about it") - and, when nothing passes the shape
        filter, *why*, for both colors independently (see _describe_mask):
        raw_pixel_count==0 for a color means its threshold never matched
        anything this frame (tune that color's HSV bounds), while a nonzero
        count with rejection reasons means a blob of that color exists but
        is the wrong size/shape (tune CONE_MIN_AREA_PX/CONE_MAX_WIDTH_PX or
        check CONE_SCAN_Y/HEIGHT).
        '''
        raw_detected = self.cone_x is not None
        action = self._decide_action()
        self._frame_count += 1
        heartbeat_due = self._frame_count % self.log_interval_frames == 0

        if raw_detected and not self._was_raw_detected:
            mean_hsv, mean_rgb = self._sample_color(band_rgb, band_hsv, self.cone_x)
            lane_note = "IN our lane" if self.cone_in_our_lane else "NOT in our lane (or lane unknown)"
            centered_note = "centered" if self.cone_centered else "off-center"
            close_note = "close" if self.cone_close else "far"
            logger.info(
                f"[cone_tape] {self.cone_color} candidate at x={self.cone_x:.1f}, area={self.cone_area}px, "
                f"scan_y={self.scan_y} - sampled color HSV=({mean_hsv[0]:.0f},{mean_hsv[1]:.0f},{mean_hsv[2]:.0f}) "
                f"RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f}) - {lane_note}, {centered_note}, {close_note} - "
                f"ACTION: {action}"
            )
        elif not raw_detected and self._was_raw_detected:
            logger.info(f"[cone_tape] {self._was_raw_color} no longer visible in scan band - ACTION: {action}")
        elif heartbeat_due:
            if raw_detected:
                mean_hsv, mean_rgb = self._sample_color(band_rgb, band_hsv, self.cone_x)
                logger.info(
                    f"[cone_tape] {self.cone_color} still at x={self.cone_x:.1f} "
                    f"HSV=({mean_hsv[0]:.0f},{mean_hsv[1]:.0f},{mean_hsv[2]:.0f}) - ACTION: {action}"
                )
            else:
                masks_desc = self._describe_mask(blue_mask, orange_mask)
                details = []
                for label, (raw_pixel_count, reasons) in masks_desc.items():
                    if raw_pixel_count == 0:
                        details.append(f"{label}: no pixels matched")
                    elif reasons:
                        details.append(f"{label}: {raw_pixel_count}px matched but rejected ({'; '.join(reasons)})")
                    else:
                        details.append(f"{label}: {raw_pixel_count}px matched, no blob")
                logger.info(f"[cone_tape] no cone detected ({'; '.join(details)}) - ACTION: {action}")

        self._was_raw_detected = raw_detected
        if raw_detected:
            self._was_raw_color = self.cone_color

    def _warn_if_lane_geometry_missing(self, yellow_x, white_x):
        if yellow_x is not None or white_x is not None or self._warned_no_lane_geometry:
            return
        logger.warning(
            "[cone_tape] lane/yellow_x and lane/white_x are both None - cone-in-our-lane "
            "detection can never trigger this way. Check myconfig.py: CV_CONTROLLER_CLASS "
            "must be 'LaneFollower', and CV_CONTROLLER_OUTPUTS must be the full "
            "['pilot/steering','pilot/throttle','cv/image_array','lane/yellow_x','lane/white_x',"
            "'lane/width_px'] (LaneFollower.run() returns all 6; Memory.put() silently drops "
            "anything past the end of CV_CONTROLLER_OUTPUTS, so a shorter list here still lets "
            "the car steer normally while leaving lane/yellow_x etc. permanently unset)."
        )
        self._warned_no_lane_geometry = True

    def run(self, cam_img, yellow_x, white_x, lane_width_px, steering, throttle, cv_img=None):
        '''
        main runloop
        input: cam_img, raw RGB camera frame (cam/image_array - independent of
               whatever LaneFollower drew on cv/image_array); yellow_x,
               white_x, lane_width_px - LaneFollower's own published lane
               geometry (lane/yellow_x, lane/white_x, lane/width_px);
               steering, throttle - LaneFollower's pilot output, passed
               through unchanged until a cone avoidance triggers (see class
               docstring: Phase 2), after which _avoid_step's output
               overrides them, permanently, for the rest of the drive;
               cv_img, the already-annotated display image to optionally
               draw the detection on top of.
        output: steering, throttle (overridden while self.avoiding), cv_img,
                cone_detected
        '''
        if cam_img is None:
            # cam/image_array not populated yet (e.g. camera part hasn't
            # produced a frame this loop) - silent otherwise, which is
            # indistinguishable from run_condition='run_pilot' skipping this
            # part's run() entirely (see Vehicle.update_parts). Logged once,
            # not every frame, since this is expected transiently at startup.
            if not self._warned_no_cam_img:
                logger.warning("[cone_tape] cam_img is None - cam/image_array not populated yet")
                self._warned_no_cam_img = True
            return steering, throttle, cv_img, self.cone_detected

        band_rgb = cam_img[self.scan_y: self.scan_y + self.scan_height, :, :]
        if band_rgb.size == 0:
            logger.warning(f"Empty cone scan slice at scan_y={self.scan_y}: "
                            f"cam_img shape={cam_img.shape}; check CONE_SCAN_Y/HEIGHT")
            return steering, throttle, cv_img, self.cone_detected
        band_hsv = cv2.cvtColor(band_rgb, cv2.COLOR_RGB2HSV)

        # smoothed/clamped in place of the raw value for both the in-our-lane
        # test below and (once avoiding) the maneuver's steering target -
        # see _smoothed_lane_width
        lane_width_px = self._smoothed_lane_width(lane_width_px)

        our_lane = _lane_bounds(yellow_x, white_x, lane_width_px, self.white_right_of_yellow, other_lane=False)
        self.lane_geometry_available = our_lane[0] is not None

        self.cone_x, self.cone_area, self.cone_color, blue_mask, orange_mask = self.detect_cone(band_hsv)
        self.cone_in_our_lane = self._x_in_bounds(self.cone_x, our_lane)
        self.cone_centered = self._is_centered(self.cone_x, cam_img.shape[1])
        self.cone_close = self._is_close(self.cone_area)

        if self.lane_geometry_available:
            # Normal case: all three gates required (see the
            # CONE_CENTER_MARGIN_PX/CONE_CLOSE_MIN_AREA_PX comments in
            # __init__) - in our lane per the (noisy) lane geometry, roughly
            # centered in the raw frame, and close enough to actually matter.
            self.cone_ready = self.cone_in_our_lane and self.cone_centered and self.cone_close
        else:
            # No lane geometry AT ALL this frame (yellow_x and white_x both
            # None - e.g. LaneFollower has fully lost the track, which is
            # exactly the state a car that's already run off-course tends to
            # be in). Added after tub_7_26-07-27: the car went straight into
            # a second cone with no swerve at all, in a stretch where it had
            # already drifted off the marked lane from an earlier incident -
            # cone_in_our_lane is structurally False whenever our_lane is
            # (None, None) (see _x_in_bounds), so requiring it AND the other
            # two gates meant a lost car could never trigger avoidance no
            # matter how obviously a cone sat dead ahead. cone_in_our_lane
            # was the ONE gate that depends on lane geometry in the first
            # place (see class docstring); centered/close were added
            # specifically to be reliable even when that geometry is wrong
            # (Decision from the earlier "cone in the other lane" fix) - so
            # when there's no geometry to test against at all, falling back
            # to those two alone is strictly better than never triggering.
            self.cone_ready = self.cone_centered and self.cone_close

        if self.cone_ready:
            self._pending_frames += 1
        else:
            self._pending_frames = 0

        # cone_detected (and therefore _decide_action's "SWERVE" verdict)
        # must be settled *before* _log_raw_detection runs below, so the
        # printed ACTION reflects this frame's decision instead of lagging
        # one frame behind it.
        was_detected = self.cone_detected
        self.cone_detected = self._pending_frames >= self.cone_trigger_frames
        if self.cone_detected and not self.avoiding:
            self.avoiding = True
            basis = "centered and close (no lane geometry to check)" if not self.lane_geometry_available \
                else "in our lane, centered, and close"
            logger.info(
                f"[cone_tape] cone confirmed {basis} at x={self.cone_x:.1f} "
                f"(held {self._pending_frames} frames) - beginning avoidance maneuver "
                f"toward the other lane; will NOT return to the original lane "
                f"afterward (see class docstring)"
            )
        elif was_detected and not self.cone_detected:
            logger.info(f"cone marker no longer in our lane / centered / close - ACTION: {self._decide_action()}")

        if self.avoiding:
            steering, throttle = self._avoid_step(cam_img, yellow_x, lane_width_px, throttle)

        self._log_raw_detection(band_rgb, band_hsv, blue_mask, orange_mask)
        self._warn_if_lane_geometry_missing(yellow_x, white_x)

        if self.overlay_image and cv_img is not None:
            cv_img = self.overlay_display(cv_img)

        return steering, throttle, cv_img, self.cone_detected

    def _avoid_step(self, cam_img, yellow_x, lane_width_px, throttle):
        '''
        One frame of the avoidance maneuver, called every run() once
        self.avoiding has latched True (see class docstring - it never
        un-latches). Mirrors LaneFollower.run()'s own steering scheme
        exactly: the PID's setpoint is the fixed image-center pixel, and
        the *detected* position - here, the other lane's estimated center
        (_other_lane_center) instead of our own lane's - is fed in as the
        process variable, so the PID steers to bring that position under
        the image's centerline, i.e. centers the car in the other lane the
        same way LaneFollower centers it in our own.

        input: cam_img, this frame's raw RGB camera frame (used only to
               resolve the image-center setpoint on first use); yellow_x,
               lane_width_px - LaneFollower's published lane geometry (the
               same near-field values LaneFollower itself steers from);
               throttle - LaneFollower's pass-through throttle this frame,
               used only to seed self.avoid_throttle the first frame the
               maneuver is active
        output: (steering, throttle) to actually drive with this frame
        '''
        if self.avoid_target_pixel is None:
            self.avoid_target_pixel = cam_img.shape[1] / 2.0
            self.avoid_pid.setpoint = self.avoid_target_pixel
        if self.avoid_throttle is None:
            self.avoid_throttle = throttle

        other_center = _other_lane_center(yellow_x, lane_width_px, self.white_right_of_yellow)

        if other_center is None:
            # no yellow line to steer against right now - passive fallback
            # (Decision 3, project_doc/obstacle_avoidance.md): hold/decay
            # the last known steering rather than guess off stale/missing
            # geometry, same caution LaneFollower's own sustained-loss
            # handling uses for its own lane.
            self.avoid_lost_frames += 1
            self.avoid_steering *= self.lost_steering_decay
            if self.avoid_lost_frames > 15:
                # sustained loss, not a brief flicker - drop the rate-limiter
                # anchor so reacquiring snaps fresh instead of slewing from a
                # stale position (mirrors LaneFollower's own reset at the
                # same threshold, see its run())
                self._avoid_last_position = None
            if self.avoid_lost_frames > self.max_lost_frames:
                self.avoid_throttle = max(self.avoid_throttle - self.throttle_step, 0.0)
            return self.avoid_steering, self.avoid_throttle

        self.avoid_lost_frames = 0

        # Rate-limit the target before it reaches the PID - mirrors
        # LaneFollower's own position_rate_limit_px (see its __init__): a
        # reacquire/noise-driven jump in other_center otherwise steps the
        # PID's process variable most of the way across the image in one
        # frame, which is exactly what turned into a full-lock swerve off
        # the track during on-car testing.
        if self.avoid_position_rate_limit_px > 0 and self._avoid_last_position is not None:
            step = other_center - self._avoid_last_position
            if abs(step) > self.avoid_position_rate_limit_px:
                other_center = self._avoid_last_position + math.copysign(
                    self.avoid_position_rate_limit_px, step)
        self._avoid_last_position = other_center

        # Soft-saturate the remaining error (k*tanh(error/k)) before handing
        # it to the PID - same technique as LaneFollower's
        # error_saturation_px, for the same reason: a large error still
        # shouldn't drive the proportional term straight to full steering
        # lock.
        error_px = other_center - self.avoid_target_pixel
        k = self.avoid_error_saturation_px
        soft_error_px = k * math.tanh(error_px / k)
        pid_steering = self.avoid_pid(self.avoid_target_pixel + soft_error_px)

        # Steering rate limit - see AVOID_STEERING_RATE_LIMIT in __init__:
        # slows how fast the turn itself is allowed to develop (distinct
        # from avoid_position_rate_limit_px above, which only slews the
        # PID's *target*), so LaneFollower(3) has more frames with the
        # camera still looking roughly down the lane to pick up the new
        # lane's white boundary before the car has already committed to a
        # hard turn.
        rate_limited = False
        if self.avoid_steering_rate_limit > 0:
            delta = pid_steering - self.avoid_steering
            if abs(delta) > self.avoid_steering_rate_limit:
                pid_steering = self.avoid_steering + math.copysign(
                    self.avoid_steering_rate_limit, delta)
                rate_limited = True
        self.avoid_steering = pid_steering

        if rate_limited or abs(other_center - self.avoid_target_pixel) > self.avoid_target_threshold:
            # turning hard toward the other lane - slow down, same rule
            # LaneFollower uses for its own lane-keeping. Also slows down
            # while the steering rate limiter is actively capping the turn
            # (rate_limited True), even if the remaining pixel error is
            # already under avoid_target_threshold - a big correction is
            # still underway, just spread over more frames, and the car
            # should stay slow for the whole stretch it's turning in.
            self.avoid_throttle = max(self.avoid_throttle - self.throttle_step, self.throttle_min)
        else:
            self.avoid_throttle = min(self.avoid_throttle + self.throttle_step, self.throttle_max)

        return self.avoid_steering, self.avoid_throttle

    def overlay_display(self, cv_img):
        y0, y1 = self.scan_y, self.scan_y + self.scan_height
        if self.cone_x is not None:
            # bright orange only once the cone is actually "ready" (in
            # lane, centered, close - see cone_ready in run()); a dimmer
            # orange for in-lane-but-not-ready-yet, gray for anything else
            if self.cone_ready:
                color = (255, 140, 0)
            elif self.cone_in_our_lane:
                color = (180, 120, 60)
            else:
                color = (150, 150, 150)
            cv2.rectangle(cv_img, (int(self.cone_x) - 8, y0), (int(self.cone_x) + 8, y1),
                          color=color, thickness=2)
        label = f"CONE:{self.cone_detected}" + (f" ({self.cone_color})" if self.cone_color else "")
        if self.avoiding:
            label += " AVOIDING"
        cv2.putText(cv_img, label, org=(10, cv_img.shape[0] - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))
        return cv_img
