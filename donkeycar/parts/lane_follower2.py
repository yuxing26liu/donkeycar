"""
lane_follower2.py

Experimental fork of lane_follower.py - NOT wired up anywhere, NOT a
replacement, sibling file only. Explores two ideas raised after analyzing
tub_9_26-07-22 (recorded near sunset): white detection collapsed to a
sustained 0% hit-rate for multi-second stretches as ambient light dimmed
and shadows fell across the track, while a handful of other frames swung
the opposite way - white hit-rate spiking to 30-55% (vs. a normal ~3-5%)
during what looked like glare/overexposure, almost certainly a false
positive rather than an actual line.

Changes from lane_follower.py, all isolated to white detection:

  1. White now uses an adaptive, per-frame-relative threshold in LAB's L
     channel instead of a fixed absolute RGB threshold. LAB's L channel
     is closer to human perceptual lightness than raw RGB and holds up
     better under shadow than RGB's per-channel triplet (this is the
     same reasoning that already moved yellow from RGB to HSV after
     tub_4_26-07-22 - see _LineTracker.color_space below - applied to
     white with a channel built for exactly this). "Adaptive" means the
     cutoff is computed fresh each frame from that frame's own scan-band
     mean/stddev ("significantly brighter than this frame's own
     pavement"), not a constant tuned for one lighting condition - which
     matters because tub_9's scan-band brightness didn't fade smoothly
     with the sunset, it swung between ~80 and ~170 multiple times in a
     single session (shaded vs. sunlit sections of track), so a single
     manually-recalibrated constant would still fail somewhere in the lap.

  2. A mask-level glare guard: if a color mask matches more than
     MAX_MASK_FRACTION of the whole scan band, it's rejected outright as
     unreliable (glare/overexposure/blown-out sky) rather than handed to
     the shape filter, which operates per-blob and isn't guaranteed to
     reject a diffuse, frame-filling false positive the way it reliably
     rejects a compact noise blob.

  3. A saturation ceiling on the LAB_ADAPTIVE mask (ADAPTIVE_MAX_SATURATION):
     the L channel alone is purely a brightness signal, so a sunlit yellow
     dash - which is *brighter than the surrounding pavement* just like a
     real white line is - passed the white check as readily as genuine
     white paint. Confirmed on tub_13_26-07-23: on frames where the yellow
     tracker had a real detection, the same pixels passed the LAB_ADAPTIVE
     white mask 90-98% of the time on a meaningful fraction of those
     frames. Since real white paint and bare pavement are both
     low-saturation (that's the whole reason RGB/LAB brightness works for
     white to begin with) while yellow paint is not, rejecting any would-be
     "white" pixel above a saturation ceiling filters out the yellow-dash
     false positives without touching genuine white detections - the same
     HSV saturation signal that already separates yellow from concrete for
     the yellow tracker, applied here as an exclusion instead of an
     inclusion band.

  4. A ceiling on the L channel's own stddev (ADAPTIVE_MAX_STD), and
     optional CLAHE contrast normalization before the adaptive threshold
     is computed (ADAPTIVE_USE_CLAHE). Both address the same failure,
     found on tub_25_26-07-23: "brighter than this frame's own mean/std"
     assumes the scan band represents one lighting condition with paint
     as the brightness anomaly within it. When the band straddles a hard
     shadow/sun boundary, that's false - the band contains two lighting
     conditions, and sunlit *bare pavement* is brighter than the real,
     shaded white paint elsewhere in the same band. Measured directly:
     normal bands have L-channel stddev ~16-20; bands straddling a
     shadow/sun boundary hit 55-65+. Worse, when this pushes the
     threshold high enough that nothing passes for several consecutive
     frames, the continuity tracker's reacquire-after-loss logic (see
     _LineTracker.update below) kicks in and confidently locks onto
     whatever the next frame's strongest candidate is - which is often
     exactly the sunlit patch, since it really is the brightest thing in
     the band. ADAPTIVE_MAX_STD rejects the frame outright when this
     happens (mirrors the existing ADAPTIVE_MIN_STD guard for the
     opposite case - a band too uniform to threshold against) so the
     tracker fails safe (loses lock, car slows) instead of failing
     confidently wrong (locks onto pavement). ADAPTIVE_USE_CLAHE was a
     complementary idea tried alongside this - locally re-normalize
     contrast in tiles across the band before the mean/std math runs, so
     a shadow/sun split doesn't dominate the per-band statistics the way
     a single global mean/std does - but it tested worse in practice:
     replayed against tub_25's confirmed false-lock frames, CLAHE alone
     still picked nearly the same wrong (bare-pavement) position on most
     of them, and combined with ADAPTIVE_MAX_STD it made tub_16
     dramatically worse (white hit-rate 11.8%->4.1%) - CLAHE changes the
     L channel's own statistics enough that a max-std threshold
     calibrated on raw images isn't calibrated for CLAHE's output
     anymore. Left in as an option (off by default) for future
     experimentation, not because it's known to help - ADAPTIVE_MAX_STD
     alone is the actual, verified fix.

Yellow's detection, the continuity/shape/PID/sustained-loss machinery,
and the overlay are all unchanged from lane_follower.py - only the things
above differ. Same constructor signature (pid, cfg) and same 6-tuple
return as lane_follower.py, so it's a drop-in alternative if
CV_CONTROLLER_MODULE is ever pointed at this file instead - not done yet.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
    position (passed in by the caller, typically _LineTracker.tracked_position)
    picks the correct blob directly instead of relying on the continuity gate
    to reject the wrong one after the fact - it doesn't reduce false positives
    within a single blob, it fixes *choosing between two true positives*, which
    the area rule was never meant to arbitrate.

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
    for white instead of a fixed absolute RGB threshold (see module
    docstring for why: tub_9_26-07-22 showed the fixed threshold
    collapse to 0% hit-rate for multi-second stretches as shadows fell).

    Works in LAB's L channel (perceptual lightness - more shadow/
    highlight-invariant than raw RGB per-channel values) and thresholds
    relative to *this frame's own* scan-band brightness rather than a
    constant: mean + k_std * stddev. That self-adjusts across both a
    slow sunset fade and the faster shaded/sunlit swings a single lap
    can have (tub_9's scan-band brightness bounced between ~80 and ~170
    multiple times in one session, not a single monotonic fade - a
    fixed threshold re-tuned for "evening" would still fail part of
    that same lap).

    A near-uniform band (stddev below min_std - e.g. deep uniform
    shadow, or a blown-out overexposed patch) has no reliable local
    contrast to threshold against; below that floor there's nothing
    trustworthy to call "brighter than," so this returns an all-zero
    mask rather than manufacturing a threshold from noise. Symmetrically,
    a band with stddev *above* max_std (see module docstring point 4) is
    just as untrustworthy in the other direction - not "one lighting
    condition with some contrast," but two different lighting conditions
    (e.g. a hard shadow/sun boundary crossing the band), where "brighter
    than this band's own mean" picks out sunlit bare pavement rather than
    the real, shaded paint. Confirmed on tub_25_26-07-23: normal bands
    run ~16-20 stddev; shadow/sun-split bands hit 55-65+.

    If clahe is given (an OpenCV CLAHE object), it's applied to the L
    channel before any of the above - locally renormalizing contrast in
    tiles across the band so a large-scale brightness gradient (like a
    shadow/sun split) doesn't dominate the whole-band mean/std the way it
    otherwise would, in principle letting real paint-vs-pavement contrast
    stand out relative to its own local neighborhood on both sides of the
    split rather than only the globally brightest region winning.

    L-channel brightness alone can't tell a genuine white line from a
    sunlit yellow dash (see module docstring point 3) - both are
    "brighter than this frame's pavement." Real white paint and bare
    pavement are both low-saturation, so any pixel that cleared the
    brightness bar but is more saturated than max_saturation is dropped
    from the mask - this is what actually excludes the yellow-dash false
    positives without touching genuine white detections.

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
    Per-color, per-scan-row line detector with continuity tracking and smoothing.

    Combines two techniques ported from unmerged teammate branches (see
    LaneFollower's docstring below for why): BetterLineFollower's connected-component
    blob-shape filter (_select_line_blob) picks a single best line-shaped blob each
    frame; RobustLineFollower's continuity gating - prefer the candidate nearest the
    last tracked position, drop the gate and re-acquire on the strongest candidate
    after a sustained loss - plus exponential smoothing make that per-frame pick
    stable enough for a *dashed* line, whose blob disappears every other frame by
    design and would otherwise look identical to a genuine loss.

    color_space adds a third mode over lane_follower.py's 'RGB'/'HSV':
    'LAB_ADAPTIVE' (see _adaptive_lab_mask) - used for white in this file.
    '''

    def __init__(self, color_low, color_high, cfg, color_name=None, color_space='RGB'):
        self.color_thr_low = np.asarray(color_low)
        self.color_thr_hi = np.asarray(color_high)
        self.color_name = color_name  # only used to tag debug log lines
        # 'RGB' (default), 'HSV', or 'LAB_ADAPTIVE' (see module docstring).
        # color_thr_low/high are unused in LAB_ADAPTIVE mode - the threshold
        # is computed fresh per frame instead.
        self.color_space = color_space

        self.min_area_px = _shape_param(cfg, color_name, 'MIN_LINE_AREA_PX', 150)
        self.max_width_px = _shape_param(cfg, color_name, 'MAX_LINE_WIDTH_PX', 250)
        # 0.10 (not the old 0.15) so a foreshortened yellow dash seen from a low
        # camera angle still clears the bar - see MIN_LINE_ASPECT_RATIO in
        # cfg_cv_control.py for the reasoning
        self.min_aspect_ratio = _shape_param(cfg, color_name, 'MIN_LINE_ASPECT_RATIO', 0.10)
        self.morph_kernel_size = getattr(cfg, 'MORPH_KERNEL_SIZE', 3)

        # Glare/overexposure guard (new in this file): a mask matching more
        # than this fraction of the whole scan band is rejected outright as
        # unreliable rather than handed to the shape filter - see module
        # docstring's point 2. 0.25 is a starting guess (a real thin line
        # should never come close to a quarter of the band), not a
        # calibration - watch how often this actually fires on real footage.
        self.max_mask_fraction = _shape_param(cfg, color_name, 'MAX_MASK_FRACTION', 0.25)

        # Only used when color_space == 'LAB_ADAPTIVE'; see _adaptive_lab_mask.
        self.adaptive_k_std = _shape_param(cfg, color_name, 'ADAPTIVE_K_STD', 1.5)
        self.adaptive_min_std = _shape_param(cfg, color_name, 'ADAPTIVE_MIN_STD', 5.0)
        # New: rejects a band whose L-channel stddev is implausibly high -
        # not "some contrast to threshold against" but "this band spans
        # two different lighting conditions" (see module docstring point
        # 4 and _adaptive_lab_mask). Calibrated across tub_4/8/9/11/13/16/25:
        # normal bands run ~16-20 stddev, with occasional legitimate spikes
        # into the 30s-40s; tub_16 and tub_25 (the two tubs with a real
        # shadow/sun split within the band) average 30-50 with a long tail
        # to 70+. 50 sits above the normal tubs' occasional legitimate
        # spikes (p99 up to ~58 on tub_9) but well below where a genuine
        # split-lighting band sits - re-check against new tubs before
        # assuming this is final, it's from one round of calibration.
        self.adaptive_max_std = _shape_param(cfg, color_name, 'ADAPTIVE_MAX_STD', 50.0)
        # Conceptually this should track whatever YELLOW_HSV_THRESHOLD_LOW's
        # saturation floor is calibrated to for the current lighting -
        # anything less saturated than "counts as yellow paint" is
        # presumed genuinely low-chroma (white paint or bare pavement).
        # 60 here is just a generic fallback if myconfig doesn't set one;
        # set ADAPTIVE_MAX_SATURATION explicitly to keep it in sync.
        self.adaptive_max_saturation = _shape_param(cfg, color_name, 'ADAPTIVE_MAX_SATURATION', 60)

        # New: optional CLAHE contrast normalization before the adaptive
        # threshold math (see module docstring point 4). Off by default -
        # the min/max std guards above are the load-bearing fix; this is
        # a complementary, more ambitious attempt to reduce how often a
        # band's stats are split-lighting-contaminated in the first place.
        # Only relevant in LAB_ADAPTIVE mode.
        self.use_clahe = _shape_param(cfg, color_name, 'ADAPTIVE_USE_CLAHE', False)
        self.clahe = None
        if self.use_clahe and color_space == 'LAB_ADAPTIVE':
            clip_limit = _shape_param(cfg, color_name, 'ADAPTIVE_CLAHE_CLIP_LIMIT', 2.0)
            tile_grid = _shape_param(cfg, color_name, 'ADAPTIVE_CLAHE_TILE_GRID', (8, 1))
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))

        self.max_jump_pixels = getattr(cfg, 'MAX_JUMP_PIXELS', 40)
        self.reacquire_after_frames = getattr(cfg, 'REACQUIRE_AFTER_FRAMES', 15)
        self.smoothing_alpha = getattr(cfg, 'POSITION_SMOOTHING_ALPHA', 0.4)

        self.tracked_position = None
        self.smoothed_position = None
        self.lost_frames = 0
        self.just_reacquired = False

    def update(self, scan_line_rgb):
        '''
        input: scan_line_rgb, an RGB numpy array (one scan row's cropped band)
        output: (smoothed_x, mask) if a plausible line was found this frame,
                 else (None, mask)
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

        # Glare/overexposure guard - see __init__ comment and module
        # docstring point 2. Checked here, before blob selection, since a
        # diffuse frame-filling false positive isn't guaranteed to get
        # broken into shape-filter-rejectable pieces by morphology alone.
        mask_fraction = np.count_nonzero(mask) / mask.size
        if mask_fraction > self.max_mask_fraction:
            if logger.isEnabledFor(logging.DEBUG):
                tag = f"[{self.color_name}] " if self.color_name else ""
                logger.debug(f"{tag}rejecting mask: {mask_fraction * 100:.1f}% of scan band matched "
                              f"(> {self.max_mask_fraction * 100:.0f}%) - likely glare/overexposure")
            self.lost_frames += 1
            self.just_reacquired = False
            return None, mask

        raw_x, _area = _select_line_blob(mask, self.min_area_px, self.max_width_px, self.min_aspect_ratio,
                                          log_tag=self.color_name, preferred_x=self.tracked_position)

        if raw_x is None:
            self.lost_frames += 1
            self.just_reacquired = False
            return None, mask

        if self.tracked_position is None or self.lost_frames > self.reacquire_after_frames:
            # no track yet, or lost long enough that we stop waiting and
            # re-acquire on whatever the strongest candidate is
            accepted_x = raw_x
            self.just_reacquired = True
        elif abs(raw_x - self.tracked_position) <= self.max_jump_pixels:
            accepted_x = raw_x
            self.just_reacquired = False
        else:
            # implausible jump (e.g. this row's blob filter picked up the
            # *other* line, or track clutter) - treat this frame as a miss
            if logger.isEnabledFor(logging.DEBUG):
                tag = f"[{self.color_name}] " if self.color_name else ""
                logger.debug(f"{tag}rejecting jump: raw_x={raw_x:.1f} vs tracked={self.tracked_position:.1f} "
                              f"(delta={abs(raw_x - self.tracked_position):.1f} > max_jump={self.max_jump_pixels})")
            self.lost_frames += 1
            self.just_reacquired = False
            return None, mask

        self.tracked_position = accepted_x
        self.lost_frames = 0

        if self.smoothed_position is None or self.just_reacquired:
            # fresh lock: snap instead of blending in slowly from a stale value
            self.smoothed_position = accepted_x
        else:
            self.smoothed_position = (self.smoothing_alpha * accepted_x
                                       + (1 - self.smoothing_alpha) * self.smoothed_position)

        return self.smoothed_position, mask


class LaneFollower:
    '''
    OpenCV based lane-keeping controller - experimental fork of the
    LaneFollower in lane_follower.py. See this module's docstring for
    what's different (white detection + a glare guard); everything else,
    including this docstring's description of the base behavior, is
    unchanged from that file.

    LineFollower tracks one line with one horizontal scan row and one PID loop.
    This class instead tracks the pair of lines that bound a lane - a solid
    outer boundary and a dashed centerline - and steers to keep their midpoint
    centered, so the car follows the lane rather than a single stripe. Curves
    are anticipated by scanning multiple rows at different lookahead distances
    (LANE_SCAN_ROWS) and combining each row's estimate of lane center into a
    weighted average, rather than reacting only to the single nearest row.

    Yellow uses HSV (on-car testing showed the dashed line and this track's
    plain concrete surface are nearly the same RGB triplet under overcast
    light and only really separate on saturation). White uses the new
    LAB_ADAPTIVE mode (see _adaptive_lab_mask) instead of lane_follower.py's
    fixed RGB threshold.

    When only one line is visible (e.g. the dashed line is mid-gap, or the
    solid line briefly leaves the scan band on a curve), the lane center is
    estimated as an offset from whichever line *is* visible, using a running
    estimate of the lane's pixel width (self.lane_width_px, exponentially
    smoothed off the primary scan row whenever both lines are visible
    together).

    Drop-in replacement for LineFollower (or lane_follower.py's
    LaneFollower): same constructor signature (pid, cfg). run(cam_img)
    returns a 6-tuple - (steering, throttle, image, yellow_x, white_x,
    lane_width_px) - matching lane_follower.py's CV_CONTROLLER_OUTPUTS.
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
