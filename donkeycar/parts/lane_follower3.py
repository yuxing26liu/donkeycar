"""
lane_follower3.py

Next-generation lane follower - built step by step on top of the lessons
from lane_follower.py and lane_follower2.py (see CLAUDE.md's "Current
focus" section and this project's memory notes for what those two
established and where they hit a ceiling).

Built via an offline candidate-testing pass (2026-07-24): five single-variable
strategy candidates were each implemented as an isolated diff from
lane_follower2.py and scored against donkeycar/tools/tub_eval.py across all
10 recorded tubs (autonomous runs at varied lighting + the two manually-driven
"stay centered" ground-truth tubs), compared against a lane_follower2.py
baseline. See donkeycar/parts/_candidates/ for the full set and their
individual results.

Merged into this file so far:

  1. Soft-saturating lateral error (from lf3_soft_saturating_error.py) -
     targets "turns too tight / crosses lane boundary". The pixel error
     handed to the steering PID is passed through k*tanh(error/k) instead of
     fed raw. Small errors are numerically ~unchanged (tanh(x/k) ~= x/k near
     zero); large errors (tight turns, post-loss reacquire snaps) compress
     instead of growing linearly, so a big momentary error is less likely to
     drive the PID's proportional term straight to full steering lock and
     overshoot a boundary. Verified: full-lock-steering fraction dropped
     roughly 10x worst-case (0.309 -> 0.026) across every one of the 10
     tubs with zero regressions on any other metric (mean_abs_steering,
     steering-vs-human-driving error on the two ground-truth tubs, sign-flip
     rate, stopped fraction all flat or improved). k = LANE_ERROR_SATURATION_PX
     (see myconfig.py), a starting estimate (~image_width/5), not yet
     calibrated against real on-car turning behavior. CONFIRMED ON CAR
     (tub_51_26-07-24, first real drive): full-lock steering 0.1% of frames,
     zero sustained full-lock runs - the mechanism works as designed.

  2. Geometric plausibility constraint (added after tub_51_26-07-24's
     on-car feedback). The two most common failures on that drive: the
     white tracker picking the SAME physical line as the yellow tracker,
     and picking a white line on the WRONG side of yellow (the opposite
     lane's outer edge, chosen when this lane's white line left the frame
     during a sharp turn). Both are geometrically impossible viewed from
     inside the right lane - white is always to the right of the yellow
     centerline (WHITE_RIGHT_OF_YELLOW), and the two lines never share an
     x position. Fix: white's blob candidates are hard-bounded to
     x >= yellow_ref + LANE_MIN_SEPARATION_PX (mirrored when
     WHITE_RIGHT_OF_YELLOW is False), where yellow_ref is this frame's
     yellow detection or its recently-tracked position. Enforced inside
     blob selection itself, so it holds even when the impossible candidate
     is the only candidate - the case the proximity/area rules can't help
     with, and the reason the earlier joint_edge_selection candidate
     (which needed candidates from both colors to compare) never fired on
     exactly these failures. Deliberately one-directional (yellow bounds
     white, never the reverse) - see the __init__ comment.

  3. Outer-line turn bias, config-gated (added after the same drive; user
     observation: even with detection working, a lane-midpoint target
     leaves no margin toward the inside of a turn, so the car clips the
     inside line - the target should sit closer to whichever line is on
     the OUTSIDE of the turn). The effective steering target is shifted by
     LANE_OUTER_BIAS_GAIN_PX * (previous frame's steering output); the
     gain's sign maps "outside of the turn" onto an image direction and
     was selected by offline sweep against the human-driven ground-truth
     laps (steering error vs. human), not derived from first principles.
     0.0 disables the bias entirely.

Not yet merged - bugs found and fixed, but not yet good enough to adopt
(see donkeycar/parts/_candidates/):

  - boundary_margin_bias: an asymmetric steering bias meant to push away
    from whichever lane edge is closer than a safety margin. First version
    was a complete no-op - the two edge-margin correction terms cancel
    exactly whenever LANE_SCAN_ROWS has only one row (today's config),
    since position is then always exactly the primary row's midpoint.
    Fixed (2026-07-24) to react to only the single nearer edge, verified
    directly to now produce a real nonzero bias on real footage. Re-scored:
    no regressions, but the effect is a small mixed wash (some tubs
    marginally better, others marginally worse, all within tolerance) -
    LANE_SAFETY_MARGIN_PX/LANE_MARGIN_BIAS_GAIN are still uncalibrated
    starting guesses now that the mechanism actually works. Not adopted
    yet - needs real tuning, not just a working mechanism, before it earns
    a place in this file.
  - kalman_position_tracker: replaces each line tracker's ad hoc
    jump-gate/reacquire-frames/smoothing-alpha trio with a small
    constant-velocity Kalman filter. First version regressed full-lock-
    steering fraction on every tub uniformly - traced to the velocity term
    accumulating from every detection's innovation with no decay or clamp.
    Fixed (2026-07-24) with a velocity decay term and a hard magnitude
    clamp, verified velocity no longer grows unbounded. Re-scored: the
    worst-hit tub (two_outer_laps) improved dramatically (full-lock
    30.9% -> 16.4%), but full-lock fraction still regressed on 5 of the
    other 9 tubs - a more fundamental tuning issue (the constant-velocity
    model likely overshooting through real curves) rather than the single
    bug that's now fixed. Not adopted - needs real tuning work on the
    process/measurement variance and velocity-gain-factor constants, not
    further blind iteration on this offline dataset alone.

Also tested, not yet merged (works, no regressions, but held back so this
file's first on-car test isolates one change at a time rather than combining
two behavioral changes before either has been driven):

  - feedforward_speed_profile: continuous (not binary) throttle-vs-error law
    plus a small steering feedforward term proportional to the lane-center's
    frame-to-frame rate of change. Reduced full-lock fraction similarly to
    the soft-saturating-error change, but raised sign-flip rate (steering
    direction changes) on 7 of 9 non-ground-truth tubs - a plausible side
    effect of the feedforward term amplifying detection jitter. Stayed
    within the harness's regression tolerance everywhere, but worth its own
    isolated on-car check given this project's known PID-wobble history,
    rather than folding it in alongside the soft-saturating-error change.
  - joint_edge_selection: jointly selects the (yellow, white) blob pair by
    lane-width plausibility instead of each color picking independently.
    Not shown to help OR hurt - the joint-selection condition (both colors
    having a shape-passing candidate simultaneously on the primary row)
    rarely triggers on this tub set, so it's essentially untested rather
    than validated.
"""

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _select_line_blob(mask, min_area_px, max_width_px, min_aspect_ratio, log_tag=None, preferred_x=None,
                       x_min=None, x_max=None):
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

    x_min/x_max (optional) are HARD plausibility bounds on the winning blob's
    centroid x, applied before the preferred_x/area pick: a candidate outside
    them is discarded outright, even if it's the only candidate. Added after
    tub_51_26-07-24 (the first real lane_follower3 drive): the two most
    common on-car failures were the white tracker picking the *same* physical
    line as the yellow tracker, and picking a white line to the LEFT of the
    yellow centerline (the opposite lane's outer edge) when this lane's white
    line left the frame during a sharp turn - both geometrically impossible
    for the real lane being driven (white is always to the right of yellow
    from the right lane, and the two lines are never at the same x). The
    proximity/area rules can't reject these - when the true line is out of
    frame, the impossible candidate is often the only one - so the caller
    passes bounds derived from the *other* color's position, and this filter
    makes the impossible pick structurally unavailable rather than merely
    unlikely.

    input: mask, binary (0/255) uint8 image; log_tag, optional label (e.g. color
           name) used to identify which tracker a rejection log line came from;
           preferred_x, optional last-known x position - when given, breaks ties
           by proximity instead of area;
           x_min/x_max, optional hard bounds on plausible centroid x
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

        cx = float(centroids[label][0])
        if x_min is not None and cx < x_min:
            if log_rejections:
                rejected.append(f"x={cx:.1f}<x_min={x_min:.1f} (geometric plausibility bound)")
            continue
        if x_max is not None and cx > x_max:
            if log_rejections:
                rejected.append(f"x={cx:.1f}>x_max={x_max:.1f} (geometric plausibility bound)")
            continue

        candidates.append((cx, int(area)))

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

        # Glare/overexposure guard: a mask matching more than this fraction
        # of the whole scan band is rejected outright as unreliable rather
        # than handed to the shape filter - see module docstring's point 2.
        # 0.25 is a starting guess (a real thin line should never come close
        # to a quarter of the band), not a calibration - watch how often
        # this actually fires on real footage.
        self.max_mask_fraction = _shape_param(cfg, color_name, 'MAX_MASK_FRACTION', 0.25)

        # Only used when color_space == 'LAB_ADAPTIVE'; see _adaptive_lab_mask.
        self.adaptive_k_std = _shape_param(cfg, color_name, 'ADAPTIVE_K_STD', 1.5)
        self.adaptive_min_std = _shape_param(cfg, color_name, 'ADAPTIVE_MIN_STD', 5.0)
        # Rejects a band whose L-channel stddev is implausibly high - not
        # "some contrast to threshold against" but "this band spans two
        # different lighting conditions" (see module docstring point 4 and
        # _adaptive_lab_mask). Calibrated across tub_4/8/9/11/13/16/25:
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

        # Optional CLAHE contrast normalization before the adaptive
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

    def update(self, scan_line_rgb, x_min=None, x_max=None):
        '''
        input: scan_line_rgb, an RGB numpy array (one scan row's cropped band);
               x_min/x_max, optional hard plausibility bounds on the detected
               x, forwarded to _select_line_blob (see its docstring) - used by
               LaneFollower.run to keep the two colors' picks geometrically
               consistent with each other (white right of yellow, minimum
               separation). A frame whose only candidates violate the bounds
               counts as a miss, same as any other no-detection frame - and if
               the tracker had previously latched onto a now-out-of-bounds
               line, the normal sustained-loss reacquire path is what recovers
               it onto a legal candidate.
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
                                          log_tag=self.color_name, preferred_x=self.tracked_position,
                                          x_min=x_min, x_max=x_max)

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
    OpenCV based lane-keeping controller - see this module's docstring for
    the changelog of what's been merged into this file vs. lane_follower2.py.

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

    New in this file (see module docstring): the lateral pixel error handed
    to the steering PID is soft-saturated (k*tanh(error/k)) instead of fed
    raw, to reduce full-steering-lock overshoot on large momentary errors.

    Drop-in replacement for LineFollower (or lane_follower.py's/
    lane_follower2.py's LaneFollower): same constructor signature (pid, cfg).
    run(cam_img) returns a 6-tuple - (steering, throttle, image, yellow_x,
    white_x, lane_width_px) - matching lane_follower.py's
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

        # The lateral pixel error (position - target_pixel) is soft-clipped
        # through k*tanh(error/k) before being handed to the PID, where k is
        # this value (see module docstring). 80px is a starting estimate
        # (~image_width/5 for a 426px-wide frame), not a calibrated constant -
        # watch real turning behavior on the car before trusting it further.
        self.error_saturation_px = getattr(cfg, 'LANE_ERROR_SATURATION_PX', 80)

        self.steering = 0.0  # from -1 to 1
        self.throttle = cfg.THROTTLE_INITIAL  # from -1 to 1
        self.delta_th = cfg.THROTTLE_STEP
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        self.max_lost_frames = getattr(cfg, 'MAX_LOST_FRAMES', 40)
        self.lost_steering_decay = getattr(cfg, 'LOST_STEERING_DECAY', 0.85)
        self.lost_frames = 0

        # Geometric plausibility constraint (new after tub_51_26-07-24, the
        # first on-car drive of this file - see module docstring): the white
        # tracker's candidates are hard-bounded to sit at least this many
        # pixels on WHITE_RIGHT_OF_YELLOW's side of the yellow line, so
        # "white and yellow picked the same physical line" and "white picked
        # on the wrong side of yellow" (both observed repeatedly on that
        # drive, both geometrically impossible from inside the lane) are
        # structurally unavailable rather than merely unlikely. One-
        # directional by design: yellow (HSV saturation, historically the
        # reliable detector here) constrains white (LAB-adaptive, the known
        # weak point, and the side both observed failures were on) - a
        # white-constrains-yellow bound is deliberately NOT applied, so a
        # white tracker that has latched onto something wrong can't veto
        # correct yellow detections while it recovers.
        self.min_separation_px = getattr(cfg, 'LANE_MIN_SEPARATION_PX', 50)

        # Outer-line turn bias (new after tub_51_26-07-24): shift the
        # steering target toward the OUTSIDE line of the current turn -
        # user-observed failure mode is crossing the inside line mid-turn
        # even with detection working, because a midpoint target leaves no
        # margin for the unseen blind zone between the camera's scan row
        # and the car. Bias is proportional to the previous frame's
        # steering output (already smoothed, cheap turn-direction signal):
        # bias_px = LANE_OUTER_BIAS_GAIN_PX * self.steering. The gain's
        # SIGN encodes which image direction "outside of the turn" is for
        # this camera/steering polarity - it was chosen by offline sweep
        # against the human-driven ground-truth laps, not derived from
        # first principles; if the drivetrain's steering polarity ever
        # changes, this sign must be re-checked. 0.0 disables the bias
        # entirely (behavior identical to before this feature existed).
        self.outer_bias_gain_px = getattr(cfg, 'LANE_OUTER_BIAS_GAIN_PX', 0.0)

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

            # Geometric plausibility bound for white, derived from yellow -
            # see the LANE_MIN_SEPARATION_PX comment in __init__. Prefer this
            # frame's fresh yellow detection; fall back to yellow's tracked
            # position while it's only briefly lost (a normal dashed-line
            # gap), since the line hasn't plausibly moved far in that time.
            # No yellow reference at all (cold start, or yellow lost past its
            # reacquire window) -> no bound; the constraint can only be as
            # available as the line it's derived from.
            yellow_tracker = self.yellow_trackers[i]
            if yellow_x is not None:
                yellow_ref = yellow_x
            elif (yellow_tracker.tracked_position is not None
                    and yellow_tracker.lost_frames <= yellow_tracker.reacquire_after_frames):
                yellow_ref = yellow_tracker.tracked_position
            else:
                yellow_ref = None

            white_x_min = None
            white_x_max = None
            if yellow_ref is not None:
                if self.white_right_of_yellow:
                    white_x_min = yellow_ref + self.min_separation_px
                else:
                    white_x_max = yellow_ref - self.min_separation_px

            white_x, white_mask = self.white_trackers[i].update(scan_line,
                                                                 x_min=white_x_min, x_max=white_x_max)
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

            # Outer-line turn bias (see LANE_OUTER_BIAS_GAIN_PX in __init__):
            # shift the effective target toward the outside of the current
            # turn, using the previous frame's steering output as the turn
            # signal. Applied to the target rather than the measurement so
            # the soft-saturation below still operates on the true remaining
            # error. self.steering here is last frame's value - by the time
            # it's overwritten below, the bias for THIS frame is already
            # baked into pid_input.
            effective_target = self.target_pixel + self.outer_bias_gain_px * self.steering

            # Soft-saturate the lateral error before handing it to the PID
            # (see module docstring): for small errors this is numerically
            # ~identical to using position directly (tanh(x/k) ~= x/k near
            # 0), but large errors are compressed instead of growing
            # linearly, so a big momentary error is less likely to drive
            # the proportional term straight to full steering lock.
            error_px = position - effective_target
            k = self.error_saturation_px
            soft_error_px = k * math.tanh(error_px / k)
            pid_input = self.target_pixel + soft_error_px
            self.steering = self.pid_st(pid_input)

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
