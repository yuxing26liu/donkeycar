"""
lf3_joint_edge_selection.py

Single-variable candidate fork of lane_follower2.py. Copies it exactly
(including the LAB_ADAPTIVE white detection and glare guard) and changes
only how the primary scan row's yellow and white blobs are chosen when
more than one shape-passing candidate exists for either color.

Today (lane_follower2.py / lane_follower.py), _select_line_blob picks each
color's winning blob independently - nearest to that color's own last
tracked position, or largest on cold start. That already fixed "two
real, similarly-sized blobs of the SAME color both look plausible"
(tub_31_26-07-24, see _select_line_blob's docstring), but it still has no
way to notice when this frame's independently-plausible yellow pick and
independently-plausible white pick don't make sense *together* - e.g.
yellow's shape filter accepting a saturated shadow edge that happens to
sit close to yellow's last tracked position, while white's independent
pick is a real line, producing a yellow/white pair whose implied lane
width is nothing like the lane's actual, continuously-tracked width.
Each color's pick was locally defensible; only reasoning about the pair
exposes the problem.

The one change here: on the primary scan row only (i == 0, matching how
lane_width_px is already only ever updated from that row), when BOTH
colors have at least one shape-passing candidate this frame, every
(yellow_candidate, white_candidate) pair is scored by how well it matches
the running lane-width estimate (self.lane_width_px) plus how close each
candidate is to that color's own last tracked position, and the
minimum-scoring pair is used - instead of each tracker choosing its
member of the pair in isolation. When only one color has a candidate (or
this isn't the primary row), there is nothing to jointly reason about, so
each tracker still picks independently exactly as before. The winning
pair is still run through each tracker's existing jump-gate + smoothing
logic unchanged, so a jointly-selected candidate that's still an
implausible jump from that color's own track is rejected exactly as it
always would be - joint selection only changes which candidate is
*offered* to that machinery, not the machinery itself.

No new tuning constant was needed for this change (the score reuses
self.lane_width_px and each tracker's existing tracked_position, both of
which already exist), so CFG_OVERRIDES is empty.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CFG_OVERRIDES = {}


def _select_line_blob_candidates(mask, min_area_px, max_width_px, min_aspect_ratio, log_tag=None):
    '''
    Run connected-components on a binary mask and return every blob that
    passes the line-shape filter, as a list of (x, area) pairs - the same
    filtering logic _select_line_blob uses before it picks a single winner,
    factored out so a caller can reason about the full set of plausible
    candidates instead of only the winner (see module docstring: joint
    yellow/white selection needs both colors' full candidate lists, not
    just each color's independently-chosen best blob).

    input: mask, binary (0/255) uint8 image; log_tag, optional label (e.g.
           color name) used to identify which tracker a rejection log line
           came from
    output: list of (x, area) - centroid x and pixel area - for every blob
            that passes the shape filter, in connected-component label
            order (not sorted); empty list if none passed
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

    if not candidates and rejected:
        logger.debug(f"[{log_tag}] no blob passed shape filter; candidates rejected by "
                      f"{', '.join(rejected)}")

    return candidates


def _pick_best_candidate(candidates, preferred_x=None):
    '''
    Given a list of (x, area) candidates that already passed the shape
    filter, pick one: nearest to preferred_x if given, else largest by
    area. Shared by _select_line_blob (independent per-color choice) and
    _LineTracker.select_and_track (which may be handed a full candidate
    list for independent choice, or a single already-jointly-chosen
    candidate - see module docstring) so both apply the identical rule.

    input: candidates, list of (x, area); preferred_x, optional last-known
           x position to break ties by proximity instead of area
    output: (x, area) of the winning blob, or (None, 0) if candidates is empty
    '''
    if not candidates:
        return None, 0
    if preferred_x is not None:
        return min(candidates, key=lambda c: abs(c[0] - preferred_x))
    return max(candidates, key=lambda c: c[1])


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

    Implemented as the shape filter (_select_line_blob_candidates) plus the
    winner rule (_pick_best_candidate) so both are independently reusable -
    see module docstring for why the winner rule alone needs to be reusable
    (joint yellow/white candidate-pair selection on the primary scan row).

    input: mask, binary (0/255) uint8 image; log_tag, optional label (e.g. color
           name) used to identify which tracker a rejection log line came from;
           preferred_x, optional last-known x position - when given, breaks ties
           by proximity instead of area
    output: (x, area) of the winning blob's centroid x and pixel area, or (None, 0)
    '''
    candidates = _select_line_blob_candidates(mask, min_area_px, max_width_px, min_aspect_ratio, log_tag=log_tag)
    return _pick_best_candidate(candidates, preferred_x=preferred_x)


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
    lane_follower2.py's module docstring for the full rationale - this
    function is unchanged from that file).

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
    blob-shape filter picks line-shaped candidate blobs each frame; RobustLineFollower's
    continuity gating - prefer the candidate nearest the last tracked position, drop the
    gate and re-acquire on the strongest candidate after a sustained loss - plus
    exponential smoothing make that per-frame pick stable enough for a *dashed* line,
    whose blob disappears every other frame by design and would otherwise look
    identical to a genuine loss.

    Split into two phases (new in this file, see module docstring) instead of the
    single update() method lane_follower2.py has:

      - compute_candidates(scan_line_rgb): builds the color mask (including the
        glare guard) and runs the shape filter, returning every candidate blob
        that passed - no tracker state is touched yet.
      - select_and_track(candidates): given a candidate list - the full list for
        independent per-color choice, or a single pre-chosen candidate when the
        caller (LaneFollower.run(), on the primary row) has already jointly
        decided the winner with the other color - picks a winner if more than
        one is given (nearest tracked_position, or largest by area on cold
        start/reacquire), then runs it through the existing jump-gate +
        smoothing/continuity logic exactly as lane_follower2.py always has.

    update(scan_line_rgb) is kept as a thin wrapper calling both phases in
    sequence, so this class's public behavior is identical to
    lane_follower2.py's for any caller that doesn't need the split.

    color_space adds a third mode over lane_follower.py's 'RGB'/'HSV':
    'LAB_ADAPTIVE' (see _adaptive_lab_mask) - used for white in this file,
    unchanged from lane_follower2.py.
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
        # than handed to the shape filter. 0.25 is a starting guess (a real
        # thin line should never come close to a quarter of the band), not
        # a calibration - watch how often this actually fires on real footage.
        self.max_mask_fraction = _shape_param(cfg, color_name, 'MAX_MASK_FRACTION', 0.25)

        # Only used when color_space == 'LAB_ADAPTIVE'; see _adaptive_lab_mask.
        self.adaptive_k_std = _shape_param(cfg, color_name, 'ADAPTIVE_K_STD', 1.5)
        self.adaptive_min_std = _shape_param(cfg, color_name, 'ADAPTIVE_MIN_STD', 5.0)
        self.adaptive_max_std = _shape_param(cfg, color_name, 'ADAPTIVE_MAX_STD', 50.0)
        # Conceptually this should track whatever YELLOW_HSV_THRESHOLD_LOW's
        # saturation floor is calibrated to for the current lighting -
        # anything less saturated than "counts as yellow paint" is
        # presumed genuinely low-chroma (white paint or bare pavement).
        # 60 here is just a generic fallback if myconfig doesn't set one;
        # set ADAPTIVE_MAX_SATURATION explicitly to keep it in sync.
        self.adaptive_max_saturation = _shape_param(cfg, color_name, 'ADAPTIVE_MAX_SATURATION', 60)

        # Optional CLAHE contrast normalization before the adaptive threshold
        # math. Off by default - the min/max std guards above are the
        # load-bearing fix; this is a complementary, more ambitious attempt to
        # reduce how often a band's stats are split-lighting-contaminated in
        # the first place. Only relevant in LAB_ADAPTIVE mode.
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

    def compute_candidates(self, scan_line_rgb):
        '''
        Phase 1: build this color's mask (including the glare guard) and run
        the shape filter, without picking a winner or touching any tracker
        state (tracked_position/lost_frames/etc are read here, never
        written) - split out so LaneFollower.run() can look at both colors'
        full candidate lists together before either commits to a choice
        (see module docstring and _LineTracker's own docstring).

        input: scan_line_rgb, an RGB numpy array (one scan row's cropped band)
        output: (candidates, mask) - candidates is a list of (x, area) pairs
                that passed the shape filter (empty list if the color mask
                matched nothing, nothing passed the shape filter, or the
                glare guard rejected the frame); mask is the binary uint8
                mask (all-zero if glare-rejected)
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

        # Glare/overexposure guard - see __init__ comment. Checked here,
        # before blob selection, since a diffuse frame-filling false
        # positive isn't guaranteed to get broken into shape-filter-
        # rejectable pieces by morphology alone. No tracker state is
        # touched here (unlike lane_follower2.py, which incremented
        # lost_frames at this point) - an empty candidate list on its own
        # already signals "no detection this frame" to select_and_track,
        # which is now the single place lost_frames/just_reacquired change,
        # so this can't double-count a rejection.
        mask_fraction = np.count_nonzero(mask) / mask.size
        if mask_fraction > self.max_mask_fraction:
            if logger.isEnabledFor(logging.DEBUG):
                tag = f"[{self.color_name}] " if self.color_name else ""
                logger.debug(f"{tag}rejecting mask: {mask_fraction * 100:.1f}% of scan band matched "
                              f"(> {self.max_mask_fraction * 100:.0f}%) - likely glare/overexposure")
            return [], mask

        candidates = _select_line_blob_candidates(mask, self.min_area_px, self.max_width_px,
                                                    self.min_aspect_ratio, log_tag=self.color_name)
        return candidates, mask

    def select_and_track(self, candidates):
        '''
        Phase 2: given a candidate list for this frame, pick a winner (if
        more than one candidate is given - nearest tracked_position, or
        largest by area on cold start/after a sustained loss) and run it
        through the jump-gate + smoothing/continuity logic. This is the
        only place tracker state (tracked_position, lost_frames,
        just_reacquired, smoothed_position) changes.

        A single-element candidates list (as LaneFollower.run() passes on
        the primary row once joint yellow/white selection has already
        picked this tracker's member of the winning pair - see module
        docstring) skips the winner choice trivially and goes straight
        through the same jump-gate/smoothing path as any other frame, so
        continuity tracking and smoothing behave identically whether this
        tracker chose independently or was handed a pre-chosen candidate.

        input: candidates, list of (x, area) pairs that already passed the
               shape filter (as returned by compute_candidates, or a
               caller-narrowed subset of it)
        output: smoothed_x if a plausible line was found this frame, else None
        '''
        raw_x, _area = _pick_best_candidate(candidates, preferred_x=self.tracked_position)

        if raw_x is None:
            self.lost_frames += 1
            self.just_reacquired = False
            return None

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
            return None

        self.tracked_position = accepted_x
        self.lost_frames = 0

        if self.smoothed_position is None or self.just_reacquired:
            # fresh lock: snap instead of blending in slowly from a stale value
            self.smoothed_position = accepted_x
        else:
            self.smoothed_position = (self.smoothing_alpha * accepted_x
                                       + (1 - self.smoothing_alpha) * self.smoothed_position)

        return self.smoothed_position

    def update(self, scan_line_rgb):
        '''
        Thin wrapper: compute_candidates() then select_and_track() in
        sequence, matching lane_follower2.py's single-method update() for
        any caller that doesn't need the two phases split apart.

        input: scan_line_rgb, an RGB numpy array (one scan row's cropped band)
        output: (smoothed_x, mask) if a plausible line was found this frame,
                 else (None, mask)
        '''
        candidates, mask = self.compute_candidates(scan_line_rgb)
        smoothed_x = self.select_and_track(candidates)
        return smoothed_x, mask


class LaneFollower:
    '''
    OpenCV based lane-keeping controller - candidate fork of lane_follower2.py
    (see this module's docstring for the one change: joint yellow/white
    candidate-pair selection on the primary scan row). Everything else,
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
    light and only really separate on saturation). White uses the
    LAB_ADAPTIVE mode (see _adaptive_lab_mask), unchanged from lane_follower2.py.

    When only one line is visible (e.g. the dashed line is mid-gap, or the
    solid line briefly leaves the scan band on a curve), the lane center is
    estimated as an offset from whichever line *is* visible, using a running
    estimate of the lane's pixel width (self.lane_width_px, exponentially
    smoothed off the primary scan row whenever both lines are visible
    together). That same self.lane_width_px is also the reference the
    primary row's joint yellow/white selection scores candidate pairs
    against (see module docstring).

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

    def _select_joint_pair(self, i, yellow_candidates, white_candidates):
        '''
        Score every (yellow_candidate, white_candidate) pair for scan row i
        and return the minimum-score pair - see module docstring for why
        this only runs on the primary row (i == 0) and only when both
        colors have at least one shape-passing candidate.

        score = width_error + continuity_error, where:
          - width_error is how far the pair's implied lane width
            (abs(white_x - yellow_x)) is from the running lane-width
            estimate (self.lane_width_px) - a pair that doesn't look like
            this lane's actual width is probably not two edges of the same
            lane.
          - continuity_error is each candidate's distance from that color's
            own last tracked position (0 if that tracker has no track yet)
            - a pair that also matches where each line was last seen is
            preferred over one that merely has a plausible width.

        input: i, scan row index; yellow_candidates/white_candidates, lists
               of (x, area) pairs, both guaranteed non-empty by the caller
        output: (yellow_candidate, white_candidate), the minimum-score pair
        '''
        yellow_tracked = self.yellow_trackers[i].tracked_position
        white_tracked = self.white_trackers[i].tracked_position

        best_pair = None
        best_score = None
        for y_cand in yellow_candidates:
            for w_cand in white_candidates:
                width = abs(w_cand[0] - y_cand[0])
                width_error = abs(width - self.lane_width_px)

                continuity_error = 0.0
                if yellow_tracked is not None:
                    continuity_error += abs(y_cand[0] - yellow_tracked)
                if white_tracked is not None:
                    continuity_error += abs(w_cand[0] - white_tracked)

                score = width_error + continuity_error
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (y_cand, w_cand)

        return best_pair

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

            yellow_candidates, yellow_mask = self.yellow_trackers[i].compute_candidates(scan_line)
            white_candidates, white_mask = self.white_trackers[i].compute_candidates(scan_line)

            if i == 0 and yellow_candidates and white_candidates:
                # Joint selection (see module docstring): reason about every
                # (yellow, white) candidate pair together instead of each
                # color picking its winner in isolation, since a locally
                # plausible pick for one color can still be the wrong blob
                # once you also know what the other color found this frame.
                best_yellow, best_white = self._select_joint_pair(i, yellow_candidates, white_candidates)
                yellow_x = self.yellow_trackers[i].select_and_track([best_yellow])
                white_x = self.white_trackers[i].select_and_track([best_white])
            else:
                # Only one color has a candidate this frame (or this isn't
                # the primary row) - nothing to jointly reason about, so
                # each tracker chooses independently exactly as
                # lane_follower2.py does.
                yellow_x = self.yellow_trackers[i].select_and_track(yellow_candidates)
                white_x = self.white_trackers[i].select_and_track(white_candidates)

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
