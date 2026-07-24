"""
lf3_kalman_position_tracker.py

Single-variable candidate forked from lane_follower2.py. The color masking
(_adaptive_lab_mask), blob selection/shape-filter (_select_line_blob), PID
loop, lane-center geometry, and overlay are byte-for-byte unchanged - the
only thing that changes is how _LineTracker turns this frame's raw blob
pick into the position handed to the PID: the old ad hoc trio (a fixed
max-pixel-jump gate, a hard reacquire-after-N-frames cutover, and a fixed
exponential-smoothing blend) is replaced with a constant-velocity 1D Kalman
filter (state = [position, velocity]).

Why: the old scheme freezes at the last smoothed position during a gap
(dashed-line miss, momentary occlusion, a rejected jump) and then either
waits for the next raw pick to be close enough or, after
REACQUIRE_AFTER_FRAMES, snaps hard onto whatever the strongest candidate
is - both are blind to the fact that the line was very likely still moving
during the gap (e.g. mid-turn). A constant-velocity Kalman filter carries
the last known velocity forward through a gap (dead reckoning) instead of
freezing, which should track a moving line through a dashed-line gap more
smoothly, and uses the same predicted position (not a stale fixed value)
as the proximity anchor for _select_line_blob's tiebreak, so continuity
gating and reacquisition emerge from the filter's own uncertainty growth
rather than a separate fixed-pixel gate. This is expected to help
lighting-robustness indirectly too: a real detection dropout (e.g. white
collapsing at dusk, see lane_follower2.py's docstring) now degrades into
smooth extrapolation rather than a frozen value or a wrong hard snap.

REACQUIRE_AFTER_FRAMES is kept with its original meaning (a sustained-loss
cutover), but the mechanism changes: instead of "next raw pick wins no
matter how far", a sustained loss now resets the filter entirely (state
back to None/cold-start) so an extrapolating filter doesn't keep
confidently projecting a stale velocity further and further from reality -
the same "don't trust a value blindly forever" intent as before, just
implemented as a hard reset instead of a gate relaxation.

measurement_variance and process_variance (CFG_OVERRIDES below) are
per-color configurable via the same _shape_param convention as every other
per-color override in this file (e.g. YELLOW_KALMAN_MEASUREMENT_VAR
overrides KALMAN_MEASUREMENT_VAR for the yellow tracker only). The
defaults (process variance 4.0, well below measurement variance 25.0) bias
the filter toward trusting the constant-velocity motion model between
frames over any single raw measurement - appropriate for a position that
is expected to move smoothly frame-to-frame rather than jump. These are
unvalidated starting guesses carried over verbatim from the design spec,
not a calibration - re-tune against real tub footage before trusting them.

FIX (2026-07-24, after initial offline scoring): the first version had no
decay or bound on the velocity state - it accumulated from every
detection's innovation indefinitely. Measured effect: full-lock-steering
fraction roughly doubled on every one of the 10 test tubs uniformly
(independent of lighting/curve content), the signature of a runaway state
variable rather than a strategy that simply underperformed. Added
KALMAN_VELOCITY_DECAY (velocity shrinks each predict step absent a
corroborating measurement, instead of persisting through a gap
indefinitely) and KALMAN_MAX_VELOCITY_PX (a hard clamp against one large
innovation setting an implausible velocity outright) - see _LineTracker's
__init__/update for the exact mechanism.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


CFG_OVERRIDES = {
    "KALMAN_PROCESS_VAR": 4.0,
    "KALMAN_MEASUREMENT_VAR": 25.0,
}


def _select_line_blob(mask, min_area_px, max_width_px, min_aspect_ratio, log_tag=None, preferred_x=None):
    '''
    Pick the best line-shaped connected component in a binary mask.

    Ported from BetterLineFollower.select_line_component (origin/better-line-follower):
    runs connected-components on the mask and rejects blobs that don't look like a
    line cross-section - too small (noise/gravel/glare), too wide (a sunlit patch of
    pavement or a wall), or too flat relative to their width (same wide-patch case,
    but resolution-independent).

    Of what's left: if preferred_x is given, the candidate *closest* to it wins;
    otherwise (cold start, no prior position) the largest by area wins. Added
    after tub_31_26-07-24 showed the old "largest always wins" rule flip-flopping
    between two simultaneously-visible real line-shaped blobs (this track's two
    solid edges, both in frame on a wide section) whenever their areas happened
    to be close - confirmed directly: frame 3448 had a real, correctly-tracked
    line at x=331.8 (area 364, matching the last several frames' smooth position)
    and an unrelated line-shaped blob at x=127.8 (area 382) that won by 18px of
    area, for 16 consecutive frames, before the tracker gave up waiting for a
    close-enough candidate and snapped hard to whatever was biggest by then -
    a full steering-lock swerve. Preferring proximity to the last tracked
    position (passed in by the caller, typically the Kalman filter's predicted
    position - see _LineTracker.update below) picks the correct blob directly
    instead of relying on the continuity gate to reject the wrong one after
    the fact - it doesn't reduce false positives within a single blob, it
    fixes *choosing between two true positives*, which the area rule was
    never meant to arbitrate.

    input: mask, binary (0/255) uint8 image; log_tag, optional label (e.g. color
           name) used to identify which tracker a rejection log line came from;
           preferred_x, optional last-known x position - when given, breaks ties
           by proximity instead of area
    output: (x, area) of the winning blob's centroid x and pixel area, or (None, 0)
    '''
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates = []  # (x, area) for every blob that passes the shape filter
    rejected = []  # only populated when candidates stays empty and DEBUG is on
    log_rejections = log_tag is not None and logger.isEnabledFor(logging.DEBUG)

    if log_rejections and num_labels <= 1:
        # label 0 (background) is the only label - the color mask had zero
        # matching pixels this frame, full stop. Distinct from the "no blob
        # passed shape filter" case below: that means candidates existed and
        # were rejected for looking wrong-shaped; this means the color
        # threshold itself never matched anything, so MIN_LINE_AREA_PX/
        # MAX_LINE_WIDTH_PX/MIN_LINE_ASPECT_RATIO are irrelevant here - check
        # the *_COLOR_THRESHOLD_LOW/HIGH bounds against real sampled pixels.
        logger.debug(f"[{log_tag}] color mask had zero matching pixels this frame - "
                      f"check the color threshold bounds, not the shape filter")

    for label in range(1, num_labels):  # label 0 is background, always skip it
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        aspect = (height / width) if width > 0 else 0

        if area < min_area_px:
            if log_rejections:
                rejected.append(f"area={area}<{min_area_px}")
            continue
        if width > max_width_px:
            if log_rejections:
                rejected.append(f"width={width}>{max_width_px}")
            continue
        if aspect < min_aspect_ratio:
            if log_rejections:
                rejected.append(f"aspect={aspect:.2f}<{min_aspect_ratio} (w={width},h={height})")
            continue

        candidates.append((float(centroids[label][0]), int(area)))

    if not candidates:
        if rejected:
            logger.debug(f"[{log_tag}] no blob passed shape filter; candidates rejected by "
                          f"{', '.join(rejected)}")
        return None, 0

    if preferred_x is not None:
        best_x, best_area = min(candidates, key=lambda c: abs(c[0] - preferred_x))
    else:
        best_x, best_area = max(candidates, key=lambda c: c[1])

    return best_x, best_area


def _shape_param(cfg, color_name, key, default):
    '''
    Per-color override with fallback to the shared key, then a hardcoded
    default: f'{COLOR}_{key}' (e.g. YELLOW_MIN_LINE_AREA_PX) takes
    precedence over the shared f'{key}' (MIN_LINE_AREA_PX) if it's set in
    cfg, so each color's filter can be tuned independently. Backward
    compatible: if no per-color keys are set, behavior is unchanged.
    '''
    if color_name:
        per_color_key = f'{color_name.upper()}_{key}'
        if hasattr(cfg, per_color_key):
            return getattr(cfg, per_color_key)
    return getattr(cfg, key, default)


def _adaptive_lab_mask(scan_line_rgb, k_std, min_std, max_std, max_saturation, clahe=None):
    '''
    Lighting-robust "brighter than the surrounding pavement" mask, used
    for white instead of a fixed absolute RGB threshold (see
    lane_follower2.py's module docstring for why: tub_9_26-07-22 showed the
    fixed threshold collapse to 0% hit-rate for multi-second stretches as
    shadows fell). Unchanged from lane_follower2.py - this candidate only
    touches _LineTracker's continuity/smoothing logic, not color masking.

    Works in LAB's L channel (perceptual lightness - more shadow/
    highlight-invariant than raw RGB per-channel values) and thresholds
    relative to *this frame's own* scan-band brightness rather than a
    constant: mean + k_std * stddev.

    A near-uniform band (stddev below min_std) has no reliable local
    contrast to threshold against, so this returns an all-zero mask.
    Symmetrically, a band with stddev above max_std is untrustworthy in
    the other direction - not one lighting condition with some contrast,
    but two different lighting conditions (e.g. a hard shadow/sun boundary
    crossing the band).

    If clahe is given (an OpenCV CLAHE object), it's applied to the L
    channel before any of the above.

    L-channel brightness alone can't tell a genuine white line from a
    sunlit yellow dash - both are "brighter than this frame's pavement."
    Real white paint and bare pavement are both low-saturation, so any
    pixel that cleared the brightness bar but is more saturated than
    max_saturation is dropped from the mask.

    input: scan_line_rgb, an RGB numpy array (one scan row's cropped band);
           max_saturation, HSV saturation (0-255) above which a pixel is
           excluded even if it passed the brightness threshold;
           clahe, optional cv2.CLAHE instance (None = disabled)
    output: mask, binary (0/255) uint8, same height/width as scan_line_rgb
    '''
    lab = cv2.cvtColor(scan_line_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    if clahe is not None:
        l_channel = clahe.apply(l_channel)

    mean = float(np.mean(l_channel))
    std = float(np.std(l_channel))
    if std < min_std or std > max_std:
        return np.zeros(l_channel.shape, dtype=np.uint8)

    threshold = mean + k_std * std
    mask = np.where(l_channel >= threshold, 255, 0).astype(np.uint8)

    saturation = cv2.cvtColor(scan_line_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    mask[saturation > max_saturation] = 0

    return mask


class _LineTracker:
    '''
    Per-color, per-scan-row line detector with continuity tracking, now via
    a constant-velocity Kalman filter (state = [position, velocity]) instead
    of lane_follower2.py's fixed-jump-gate + reacquire-frames +
    exponential-smoothing trio. Color masking (_adaptive_lab_mask) and blob
    selection (_select_line_blob) are unchanged - ported straight from
    lane_follower2.py.

    Each frame: predict (position += velocity, uncertainty grows by
    process_variance), then either correct against this frame's raw blob
    pick (if one passed the shape filter) or, on a miss, just keep the
    predicted state - dead reckoning through a gap using the last known
    velocity, rather than freezing at the last smoothed value the way fixed
    exponential smoothing does. The predicted position (not a fixed last
    value) is what _select_line_blob uses as its proximity anchor, so the
    same continuity-tiebreak behavior lane_follower2.py relied on
    (tub_31_26-07-24) still works, just anchored to a filter's best current
    guess of where the line is *now* instead of where it last was.

    REACQUIRE_AFTER_FRAMES keeps its original meaning (give up on a
    sustained loss) but the mechanism is a full filter reset - state back
    to cold-start (position=None, velocity=0) - rather than relaxing the
    jump gate, since letting a constant-velocity filter keep extrapolating
    indefinitely through a real, prolonged loss would just confidently
    project the position further and further from reality with no
    correcting signal.
    '''

    def __init__(self, color_low, color_high, cfg, color_name=None, color_space='RGB'):
        self.color_thr_low = np.asarray(color_low)
        self.color_thr_hi = np.asarray(color_high)
        self.color_name = color_name  # only used to tag debug log lines
        # 'RGB' (default), 'HSV', or 'LAB_ADAPTIVE' (see lane_follower2.py's
        # module docstring). color_thr_low/high are unused in LAB_ADAPTIVE
        # mode - the threshold is computed fresh per frame instead.
        self.color_space = color_space

        self.min_area_px = _shape_param(cfg, color_name, 'MIN_LINE_AREA_PX', 150)
        self.max_width_px = _shape_param(cfg, color_name, 'MAX_LINE_WIDTH_PX', 250)
        # 0.10 (not the old 0.15) so a foreshortened yellow dash seen from a low
        # camera angle still clears the bar - see MIN_LINE_ASPECT_RATIO in
        # cfg_cv_control.py for the reasoning
        self.min_aspect_ratio = _shape_param(cfg, color_name, 'MIN_LINE_ASPECT_RATIO', 0.10)
        self.morph_kernel_size = getattr(cfg, 'MORPH_KERNEL_SIZE', 3)

        # Glare/overexposure guard (unchanged from lane_follower2.py): a mask
        # matching more than this fraction of the whole scan band is
        # rejected outright as unreliable rather than handed to the shape
        # filter. 0.25 is a starting guess, not a calibration.
        self.max_mask_fraction = _shape_param(cfg, color_name, 'MAX_MASK_FRACTION', 0.25)

        # Only used when color_space == 'LAB_ADAPTIVE'; see _adaptive_lab_mask.
        self.adaptive_k_std = _shape_param(cfg, color_name, 'ADAPTIVE_K_STD', 1.5)
        self.adaptive_min_std = _shape_param(cfg, color_name, 'ADAPTIVE_MIN_STD', 5.0)
        self.adaptive_max_std = _shape_param(cfg, color_name, 'ADAPTIVE_MAX_STD', 50.0)
        self.adaptive_max_saturation = _shape_param(cfg, color_name, 'ADAPTIVE_MAX_SATURATION', 60)

        self.use_clahe = _shape_param(cfg, color_name, 'ADAPTIVE_USE_CLAHE', False)
        self.clahe = None
        if self.use_clahe and color_space == 'LAB_ADAPTIVE':
            clip_limit = _shape_param(cfg, color_name, 'ADAPTIVE_CLAHE_CLIP_LIMIT', 2.0)
            tile_grid = _shape_param(cfg, color_name, 'ADAPTIVE_CLAHE_TILE_GRID', (8, 1))
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))

        # Kept with its original meaning - see class docstring for how the
        # mechanism (full filter reset, not jump-gate relaxation) differs.
        self.reacquire_after_frames = getattr(cfg, 'REACQUIRE_AFTER_FRAMES', 15)

        # Kalman filter tuning, per-color overridable via the same
        # _shape_param convention as everything else in this file (e.g.
        # YELLOW_KALMAN_MEASUREMENT_VAR). See module docstring: process
        # variance well below measurement variance biases the filter
        # toward trusting the constant-velocity motion model between
        # frames over any single noisy measurement - unvalidated starting
        # guesses, not a calibration.
        self.process_variance = _shape_param(cfg, color_name, 'KALMAN_PROCESS_VAR', 4.0)
        self.measurement_variance = _shape_param(cfg, color_name, 'KALMAN_MEASUREMENT_VAR', 25.0)

        # Simplified joint position/velocity correction: rather than
        # deriving the full 2x2 covariance cross-terms, the velocity
        # correction uses the same innovation and position Kalman gain but
        # damped by this fixed fraction, so a single noisy measurement
        # nudges the velocity estimate without letting it overshoot on
        # that measurement alone. Not per-color configurable - this is a
        # simplification-of-mechanism constant, not a tuning knob callers
        # are expected to need to touch per color.
        self.velocity_gain_factor = 0.5

        # FIX (2026-07-24, after initial offline scoring): the first version
        # had no decay or bound on velocity at all, so one ambiguous
        # detection's innovation could leave the filter dead-reckoning
        # further off-track on every subsequent miss instead of settling
        # back toward the raw detections - measured directly: full-lock
        # steering frequency roughly doubled on every one of 10 tubs
        # uniformly (not just curve-heavy or low-light ones), which is the
        # signature of a runaway state variable, not a strategy that simply
        # didn't help. Two independent safety nets, both conservative
        # starting guesses:
        #   - velocity_decay: each predict step multiplies the carried-
        #     forward velocity by this factor (<1), so an unconfirmed
        #     velocity estimate bleeds back toward zero over a gap instead
        #     of being trusted indefinitely - only a real, repeated
        #     measurement (the correction step below) can sustain a
        #     nonzero velocity.
        #   - max_velocity_px: a hard clamp, independent of decay, so a
        #     single large innovation (e.g. a brief misidentified blob)
        #     can't set a velocity so large that even one predict step
        #     shoots the position implausibly far.
        self.velocity_decay = _shape_param(cfg, color_name, 'KALMAN_VELOCITY_DECAY', 0.8)
        self.max_velocity_px = _shape_param(cfg, color_name, 'KALMAN_MAX_VELOCITY_PX', 15.0)

        # Kalman filter state: None means cold-start (no track yet, or just
        # reset after a sustained loss) - mirrors the old
        # tracked_position/smoothed_position both starting at None.
        self.position = None
        self.velocity = 0.0
        self.variance_position = None
        self.lost_frames = 0

    def update(self, scan_line_rgb):
        '''
        input: scan_line_rgb, an RGB numpy array (one scan row's cropped band)
        output: (position, mask) - position is the Kalman filter's current
                position estimate (predicted-through-gap if this frame had
                no detection, corrected-by-measurement if it did), or None
                if the filter is cold (no detection yet, or since the last
                sustained-loss reset); mask is always returned for the
                overlay regardless of detection status
        '''
        if self.color_space == 'HSV':
            scan_line = cv2.cvtColor(scan_line_rgb, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(scan_line, self.color_thr_low, self.color_thr_hi)
        elif self.color_space == 'LAB_ADAPTIVE':
            mask = _adaptive_lab_mask(scan_line_rgb, self.adaptive_k_std, self.adaptive_min_std,
                                       self.adaptive_max_std, self.adaptive_max_saturation,
                                       clahe=self.clahe)
        else:
            mask = cv2.inRange(scan_line_rgb, self.color_thr_low, self.color_thr_hi)

        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Kalman predict step, every frame. A cold-started tracker
        # (self.position is None) has no state to predict from yet - it
        # waits for a first detection to initialize on, same as the old
        # tracked_position/smoothed_position pair starting at None.
        if self.position is not None:
            predicted_position = self.position + self.velocity
            # Decay, not carry-forward - see __init__ comment on
            # velocity_decay/max_velocity_px: an unconfirmed velocity
            # estimate should bleed toward zero absent a corroborating
            # measurement, not persist indefinitely.
            predicted_velocity = self.velocity * self.velocity_decay
            predicted_variance = self.variance_position + self.process_variance
        else:
            predicted_position = None
            predicted_velocity = 0.0
            predicted_variance = None

        # Glare/overexposure guard - see __init__ comment. Checked here,
        # before blob selection, since a diffuse frame-filling false
        # positive isn't guaranteed to get broken into shape-filter-
        # rejectable pieces by morphology alone. Treated identically to
        # "no blob passed the shape filter" below - both are just "no
        # usable measurement this frame."
        mask_fraction = np.count_nonzero(mask) / mask.size
        if mask_fraction > self.max_mask_fraction:
            if logger.isEnabledFor(logging.DEBUG):
                tag = f"[{self.color_name}] " if self.color_name else ""
                logger.debug(f"{tag}rejecting mask: {mask_fraction * 100:.1f}% of scan band matched "
                              f"(> {self.max_mask_fraction * 100:.0f}%) - likely glare/overexposure")
            raw_x = None
        else:
            # predicted_position (not a fixed last value) anchors the
            # proximity tiebreak - see class docstring and
            # _select_line_blob's docstring.
            raw_x, _area = _select_line_blob(mask, self.min_area_px, self.max_width_px, self.min_aspect_ratio,
                                              log_tag=self.color_name, preferred_x=predicted_position)

        if raw_x is None:
            self.lost_frames += 1
            if predicted_position is not None:
                # Dead reckoning: carry the motion model through the gap
                # instead of freezing at the last value - this is expected
                # to handle a normal dashed-line gap at least as gracefully
                # as the old fixed exponential smoothing did, since
                # velocity carries the trend forward rather than stalling.
                self.position = predicted_position
                self.velocity = predicted_velocity
                self.variance_position = predicted_variance
            # else: still cold, nothing to carry forward - self.position
            # stays None.

            if self.lost_frames > self.reacquire_after_frames:
                # Sustained loss - reset the filter entirely rather than
                # let it keep confidently extrapolating position further
                # and further from reality with no correcting measurement.
                # Mirrors the old reacquire behavior's intent (give up
                # waiting and start fresh) even though the mechanism here
                # is a hard reset rather than a gate relaxation.
                self.position = None
                self.velocity = 0.0
                self.variance_position = None

            return self.position, mask

        # Detection this frame.
        if predicted_position is None:
            # Cold start (first-ever detection, or the first detection
            # after a sustained-loss reset): initialize the filter
            # directly on this measurement - no prior motion to blend
            # against, so there's nothing a Kalman correction step would
            # add over simply snapping, same as the old
            # tracked_position/smoothed_position both snapping to the
            # first raw_x.
            self.position = raw_x
            self.velocity = 0.0
            self.variance_position = self.measurement_variance
        else:
            innovation = raw_x - predicted_position
            kalman_gain = predicted_variance / (predicted_variance + self.measurement_variance)
            self.position = predicted_position + kalman_gain * innovation
            # Simplified velocity correction - see __init__ comment on
            # velocity_gain_factor. Clamped (see velocity_decay/
            # max_velocity_px comment) so one large innovation can't set an
            # implausibly large velocity outright.
            self.velocity = predicted_velocity + kalman_gain * self.velocity_gain_factor * innovation
            self.velocity = max(-self.max_velocity_px, min(self.max_velocity_px, self.velocity))
            self.variance_position = (1 - kalman_gain) * predicted_variance

        self.lost_frames = 0
        return self.position, mask


class LaneFollower:
    '''
    OpenCV based lane-keeping controller - single-variable candidate forked
    from lane_follower2.py. See this module's docstring for the one thing
    that's different (a Kalman filter replaces the old jump-gate/reacquire/
    smoothing trio inside _LineTracker); everything else, including this
    docstring's description of the base behavior, is unchanged from that
    file.

    LineFollower tracks one line with one horizontal scan row and one PID loop.
    This class instead tracks the pair of lines that bound a lane - a solid
    outer boundary and a dashed centerline - and steers to keep their midpoint
    centered, so the car follows the lane rather than a single stripe. Curves
    are anticipated by scanning multiple rows at different lookahead distances
    (LANE_SCAN_ROWS) and combining each row's estimate of lane center into a
    weighted average, rather than reacting only to the single nearest row.

    Yellow uses HSV (on-car testing showed the dashed line and this track's
    plain concrete surface are nearly the same RGB triplet under overcast
    light and only really separate on saturation). White uses the
    LAB_ADAPTIVE mode (see _adaptive_lab_mask) instead of a fixed RGB
    threshold.

    When only one line is visible (e.g. the dashed line is mid-gap, or the
    solid line briefly leaves the scan band on a curve), the lane center is
    estimated as an offset from whichever line *is* visible, using a running
    estimate of the lane's pixel width (self.lane_width_px, exponentially
    smoothed off the primary scan row whenever both lines are visible
    together).

    Drop-in replacement for LineFollower (or lane_follower.py's/
    lane_follower2.py's LaneFollower): same constructor signature (pid,
    cfg). run(cam_img) returns a 6-tuple - (steering, throttle, image,
    yellow_x, white_x, lane_width_px) - matching lane_follower.py's
    CV_CONTROLLER_OUTPUTS.
    '''

    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE

        self.scan_height = getattr(cfg, 'LANE_SCAN_HEIGHT', getattr(cfg, 'SCAN_HEIGHT', 20))
        self.scan_rows = getattr(cfg, 'LANE_SCAN_ROWS', [{'scan_y': cfg.SCAN_Y, 'weight': 1.0}])

        yellow_low = getattr(cfg, 'YELLOW_HSV_THRESHOLD_LOW', (15, 60, 60))
        yellow_high = getattr(cfg, 'YELLOW_HSV_THRESHOLD_HIGH', (35, 255, 255))
        # white_low/high are read for backward compatibility but unused by
        # LAB_ADAPTIVE mode - the threshold is computed per frame instead.
        white_low = getattr(cfg, 'WHITE_COLOR_THRESHOLD_LOW', (190, 190, 190))
        white_high = getattr(cfg, 'WHITE_COLOR_THRESHOLD_HIGH', (255, 255, 255))

        self.yellow_trackers = [_LineTracker(yellow_low, yellow_high, cfg, color_name='yellow', color_space='HSV')
                                 for _ in self.scan_rows]
        self.white_trackers = [_LineTracker(white_low, white_high, cfg, color_name='white', color_space='LAB_ADAPTIVE')
                                for _ in self.scan_rows]

        # geometry: which side of the dashed centerline our lane's solid
        # boundary is on. True = white is to the right of yellow (our lane is
        # the right lane), used to derive a lane-center estimate when only
        # one of the two lines is visible.
        self.white_right_of_yellow = getattr(cfg, 'WHITE_RIGHT_OF_YELLOW', True)
        self.lane_width_px = getattr(cfg, 'LANE_WIDTH_PX', 150)
        self.lane_width_smoothing_alpha = getattr(cfg, 'LANE_WIDTH_SMOOTHING_ALPHA', 0.1)

        self.target_pixel = getattr(cfg, 'LANE_TARGET_PIXEL', None)
        self.target_threshold = getattr(cfg, 'LANE_TARGET_THRESHOLD', 10)

        self.steering = 0.0  # from -1 to 1
        self.throttle = cfg.THROTTLE_INITIAL  # from -1 to 1
        self.delta_th = cfg.THROTTLE_STEP
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        self.max_lost_frames = getattr(cfg, 'MAX_LOST_FRAMES', 40)
        self.lost_steering_decay = getattr(cfg, 'LOST_STEERING_DECAY', 0.85)
        self.lost_frames = 0

        self.last_yellow_x = None
        self.last_white_x = None

        self.pid_st = pid
        # bounds the output *and* caps the internal integral accumulator so
        # it can't wind up past what the actuator can use - see
        # BetterLineFollower's docstring (origin/better-line-follower) for
        # why this must be set on the pid object, not clipped post-hoc.
        self.pid_st.output_limits = (-1.0, 1.0)

    def _lane_center(self, yellow_x, white_x):
        '''
        Combine whichever of the two lines is visible into a single "center of
        our lane" pixel position.

        When only one line is visible, the other's position is estimated as an
        offset of self.lane_width_px using WHITE_RIGHT_OF_YELLOW - the only
        signal available, since (per this track's 2-line design) there is no
        second boundary line on the far side of the road to measure against
        directly. This assumes the lane is roughly the same pixel width at
        this scan row every frame, which self.lane_width_px's continuous
        re-estimation (see run()) keeps reasonably current.
        '''
        if yellow_x is not None and white_x is not None:
            return (yellow_x + white_x) / 2.0

        sign = 1.0 if self.white_right_of_yellow else -1.0
        if yellow_x is not None:
            return yellow_x + sign * self.lane_width_px / 2.0
        if white_x is not None:
            return white_x - sign * self.lane_width_px / 2.0
        return None

    def run(self, cam_img):
        '''
        main runloop of the CV controller
        input: cam_image, an RGB numpy array
        output: steering, throttle, image, yellow_x, white_x, lane_width_px
        '''
        if cam_img is None:
            return 0, 0, None, None, None, self.lane_width_px

        if self.target_pixel is None:
            # center of the actual incoming frame, resolved on first use -
            # see class docstring for why this isn't latched onto frame 1's
            # detection (LineFollower's original behavior).
            self.target_pixel = cam_img.shape[1] / 2.0
            logger.info(f"Defaulting lane target pixel to image center = {self.target_pixel}")

        if self.pid_st.setpoint != self.target_pixel:
            self.pid_st.setpoint = self.target_pixel

        row_centers = []
        row_weights = []
        near_yellow_x = None
        near_white_x = None
        overlay_rows = []

        for i, row in enumerate(self.scan_rows):
            scan_y = row['scan_y']
            weight = row.get('weight', 1.0)
            scan_line = cam_img[scan_y: scan_y + self.scan_height, :, :]

            if scan_line.size == 0:
                logger.warning(
                    f"Empty lane scan slice at scan_y={scan_y}: cam_img shape={cam_img.shape}; "
                    f"check LANE_SCAN_ROWS against the actual camera resolution")
                continue

            yellow_x, yellow_mask = self.yellow_trackers[i].update(scan_line)
            white_x, white_mask = self.white_trackers[i].update(scan_line)
            overlay_rows.append((scan_y, yellow_mask, white_mask, yellow_x, white_x))

            if i == 0:
                # the primary (nearest) row is authoritative for the
                # published lane geometry and the lane-width estimate below
                near_yellow_x, near_white_x = yellow_x, white_x

            center = self._lane_center(yellow_x, white_x)
            if center is not None:
                row_centers.append(center)
                row_weights.append(weight)

        # keep the lane-width estimate current off the primary row only -
        # perspective makes a farther row's apparent width less reliable
        if near_yellow_x is not None and near_white_x is not None:
            measured_width = abs(near_white_x - near_yellow_x)
            self.lane_width_px = (self.lane_width_smoothing_alpha * measured_width
                                   + (1 - self.lane_width_smoothing_alpha) * self.lane_width_px)

        if near_yellow_x is not None:
            self.last_yellow_x = near_yellow_x
        if near_white_x is not None:
            self.last_white_x = near_white_x

        if row_centers:
            self.lost_frames = 0

            # weighted average across scan rows: the near row keeps the car
            # centered right now, farther rows anticipate an upcoming curve
            # before the near row's detection would otherwise catch it
            position = sum(c * w for c, w in zip(row_centers, row_weights)) / sum(row_weights)

            self.steering = self.pid_st(position)

            if abs(position - self.target_pixel) > self.target_threshold:
                # turning - slow down
                self.throttle = max(self.throttle - self.delta_th, self.throttle_min)
            else:
                # straight - speed up
                self.throttle = min(self.throttle + self.delta_th, self.throttle_max)
        else:
            # neither line visible in any scan row this frame - a genuine
            # loss, not just the dashed line's expected per-frame gap (that's
            # already tolerated inside _LineTracker). Ease toward stopped
            # instead of continuing to act on a stale command; see
            # BetterLineFollower/RobustLineFollower docstrings for the
            # real-tub-replay motivation for this behavior.
            self.lost_frames += 1
            self.steering *= self.lost_steering_decay
            if self.lost_frames > self.max_lost_frames:
                if self.lost_frames == self.max_lost_frames + 1:
                    logger.warning(
                        f"Lane lost for more than MAX_LOST_FRAMES={self.max_lost_frames} "
                        f"consecutive frames; stopping instead of holding stale output.")
                self.throttle = max(self.throttle - self.delta_th, 0.0)
            else:
                logger.info(
                    f"No lane line detected in any scan row "
                    f"({self.lost_frames}/{self.max_lost_frames} consecutive)")
                self.throttle = max(self.throttle - self.delta_th, self.throttle_min)

        if self.overlay_image and overlay_rows:
            cam_img = self.overlay_display(cam_img, overlay_rows)

        return self.steering, self.throttle, cam_img, self.last_yellow_x, self.last_white_x, self.lane_width_px

    def overlay_display(self, cam_img, overlay_rows):
        '''
        composite each scan row's color masks on top of the original image,
        mark the detected yellow/white positions and the target, and show
        the current control values - so the multi-row detection and the
        lane-width estimate are visible while tuning, not just their end
        effect on steering.
        '''
        img = np.copy(cam_img)
        target_x = int(self.target_pixel) if self.target_pixel is not None else None

        for scan_y, yellow_mask, white_mask, yellow_x, white_x in overlay_rows:
            combined = cv2.bitwise_or(yellow_mask, white_mask)
            mask_exp = np.stack((combined,) * 3, axis=-1)
            band = img[scan_y: scan_y + self.scan_height, :, :]
            img[scan_y: scan_y + self.scan_height, :, :] = np.where(mask_exp > 0, mask_exp, band)

            if target_x is not None:
                cv2.line(img, (target_x, scan_y), (target_x, scan_y + self.scan_height),
                         color=(255, 255, 255), thickness=1)
            if yellow_x is not None:
                # img is RGB (see module docstring), so yellow is (255,255,0) -
                # not the (0,255,255) that would be yellow if this were BGR
                cv2.line(img, (int(yellow_x), scan_y), (int(yellow_x), scan_y + self.scan_height),
                         color=(255, 255, 0), thickness=2)
            if white_x is not None:
                cv2.line(img, (int(white_x), scan_y), (int(white_x), scan_y + self.scan_height),
                         color=(0, 0, 255), thickness=2)

        display_str = [
            "STEERING:{:.2f}".format(self.steering),
            "THROTTLE:{:.2f}".format(self.throttle),
            "YELLOW_X:{}".format(int(self.last_yellow_x) if self.last_yellow_x is not None else "None"),
            "WHITE_X:{}".format(int(self.last_white_x) if self.last_white_x is not None else "None"),
            "LANE_WIDTH:{:.0f}".format(self.lane_width_px),
        ]
        y = 10
        x = 10
        for s in display_str:
            cv2.putText(img, s, color=(0, 0, 0), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
            y += 10

        return img
