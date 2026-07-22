import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _select_line_blob(mask, min_area_px, max_width_px, min_aspect_ratio, log_tag=None):
    '''
    Pick the best line-shaped connected component in a binary mask.

    Ported from BetterLineFollower.select_line_component (origin/better-line-follower):
    runs connected-components on the mask and rejects blobs that don't look like a
    line cross-section - too small (noise/gravel/glare), too wide (a sunlit patch of
    pavement or a wall), or too flat relative to their width (same wide-patch case,
    but resolution-independent). Of what's left, the largest by area wins.

    input: mask, binary (0/255) uint8 image; log_tag, optional label (e.g. color
           name) used to identify which tracker a rejection log line came from
    output: (x, area) of the winning blob's centroid x and pixel area, or (None, 0)
    '''
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_label = None
    best_area = 0
    rejected = []  # only populated when best_label stays None and DEBUG is on
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

        if area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        if rejected:
            logger.debug(f"[{log_tag}] no blob passed shape filter; candidates rejected by "
                          f"{', '.join(rejected)}")
        return None, 0
    return float(centroids[best_label][0]), int(best_area)


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
    '''

    def __init__(self, color_low, color_high, cfg, color_name=None):
        self.color_thr_low = np.asarray(color_low)
        self.color_thr_hi = np.asarray(color_high)
        self.color_name = color_name  # only used to tag debug log lines

        self.min_area_px = getattr(cfg, 'MIN_LINE_AREA_PX', 150)
        self.max_width_px = getattr(cfg, 'MAX_LINE_WIDTH_PX', 250)
        # 0.10 (not the old 0.15) so a foreshortened yellow dash seen from a low
        # camera angle still clears the bar - see MIN_LINE_ASPECT_RATIO in
        # cfg_cv_control.py for the reasoning
        self.min_aspect_ratio = getattr(cfg, 'MIN_LINE_ASPECT_RATIO', 0.10)
        self.morph_kernel_size = getattr(cfg, 'MORPH_KERNEL_SIZE', 3)

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
        # thresholded directly in RGB, not HSV - simpler to reason about and to
        # measure sample values for (just read R/G/B off a few pixels), and this
        # project doesn't ever need to isolate hue on its own
        mask = cv2.inRange(scan_line_rgb, self.color_thr_low, self.color_thr_hi)

        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        raw_x, _area = _select_line_blob(mask, self.min_area_px, self.max_width_px, self.min_aspect_ratio,
                                          log_tag=self.color_name)

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
    OpenCV based lane-keeping controller - dual-line upgrade of LineFollower
    (see line_follower.py, the project's single-line baseline autopilot).

    LineFollower tracks one line with one horizontal scan row and one PID loop.
    This class instead tracks the pair of lines that bound a lane - a solid
    outer boundary and a dashed centerline - and steers to keep their midpoint
    centered, so the car follows the lane rather than a single stripe. Curves
    are anticipated by scanning multiple rows at different lookahead distances
    (LANE_SCAN_ROWS) and combining each row's estimate of lane center into a
    weighted average, rather than reacting only to the single nearest row.

    Techniques ported from unmerged teammate branches:
      - origin/better-line-follower's BetterLineFollower: connected-component
        blob-shape filtering (_select_line_blob) instead of a flattened
        column-histogram argmax, PID output_limits clamp set on the pid object
        itself (bounds the integral term too, not just the output), and
        sustained-loss handling that stops the car instead of holding stale
        output forever.
      - origin/robustRyan's RobustLineFollower: continuity-tracking (prefer the
        candidate nearest the last tracked position) and exponential smoothing
        of the tracked position, both essential here because the yellow
        centerline is *dashed* and disappears from the scan row every other
        segment by design - without continuity tracking that reads as "line
        lost" on a normal, expected gap.
      - origin/estella's ObjectAvoider/line_follower.py hardening: defaulting
        the target pixel to image-center instead of latching onto whatever
        matched on the first frame (which could be background clutter).

    Colors are thresholded directly in RGB (cv2.inRange on the raw camera
    frame), not HSV - LineFollower and the ported branches above all convert
    to HSV first, but plain RGB bounds are simpler to read pixel samples for
    and measure/tune against, and this project has no need to isolate hue
    from brightness on its own.

    New in this class, not present in any of the above (none of them do
    dual-line or curve-aware detection): when only one line is visible (e.g.
    the dashed line is mid-gap, or the solid line briefly leaves the scan
    band on a curve), the lane center is estimated as an offset from
    whichever line *is* visible, using a running estimate of the lane's
    pixel width (self.lane_width_px, exponentially smoothed off the primary
    scan row whenever both lines are visible together).

    Drop-in replacement for LineFollower: same constructor signature (pid,
    cfg). run(cam_img) returns a 6-tuple instead of LineFollower's 3-tuple -
    (steering, throttle, image, yellow_x, white_x, lane_width_px) - so that
    obstacle-avoidance parts added later (Phase 2/3 of the lane-following
    project) can read the raw lane geometry without re-running any CV; see
    CV_CONTROLLER_OUTPUTS in cfg_cv_control.py, which must list all six names.
    '''

    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE

        self.scan_height = getattr(cfg, 'LANE_SCAN_HEIGHT', getattr(cfg, 'SCAN_HEIGHT', 20))
        self.scan_rows = getattr(cfg, 'LANE_SCAN_ROWS', [{'scan_y': cfg.SCAN_Y, 'weight': 1.0}])

        # RGB, not HSV - see class docstring
        yellow_low = getattr(cfg, 'YELLOW_COLOR_THRESHOLD_LOW', (190, 120, 0))
        yellow_high = getattr(cfg, 'YELLOW_COLOR_THRESHOLD_HIGH', (255, 220, 100))
        white_low = getattr(cfg, 'WHITE_COLOR_THRESHOLD_LOW', (190, 190, 190))
        white_high = getattr(cfg, 'WHITE_COLOR_THRESHOLD_HIGH', (255, 255, 255))

        self.yellow_trackers = [_LineTracker(yellow_low, yellow_high, cfg, color_name='yellow')
                                 for _ in self.scan_rows]
        self.white_trackers = [_LineTracker(white_low, white_high, cfg, color_name='white')
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
