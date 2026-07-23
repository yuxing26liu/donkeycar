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

        self.log_interval_frames = getattr(cfg, 'CONE_LOG_INTERVAL_FRAMES', 10)

        # public detection state - what a future avoidance maneuver (or a
        # test) reads; updated every run() call
        self.cone_x = None              # raw detected x this frame (any lane), or None
        self.cone_in_our_lane = False   # raw in-our-lane test this frame, pre-debounce
        self.cone_detected = False      # debounced: True once cone_in_our_lane has held
                                         # for cone_trigger_frames consecutive frames
        self._pending_frames = 0

        # diagnostic-logging state only (see _log_raw_detection /
        # _warn_if_lane_geometry_missing) - not used for detection itself
        self._was_raw_detected = False
        self._frame_count = 0
        self._warned_no_lane_geometry = False

    def _open(self, mask):
        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def detect_cone(self, band_hsv):
        '''
        input: band_hsv, HSV numpy array of the forward scan band
        output: (x, mask) - x position in pixels of the blue tape marker's blob
                centroid (or None), and the binary color mask used to find it
                (returned for _describe_mask's diagnostics below, so the mask
                isn't recomputed twice per frame)
        '''
        mask = self._open(cv2.inRange(band_hsv, self.blue_low, self.blue_high))
        x, _area = _select_line_blob(mask, self.cone_min_area_px, self.cone_max_width_px,
                                      min_aspect_ratio=0.0, log_tag='cone_tape')
        return x, mask

    def _describe_mask(self, mask):
        '''
        Diagnostics-only, independent of the shape filter in _select_line_blob:
        that function only logs *why* a blob was rejected (too small/too wide)
        when the root logger is at DEBUG - which on this car would also spam
        LaneFollower's own per-frame yellow/white rejections. This reports the
        same thing for the cone-tape mask alone, at the default INFO level, so
        "why isn't it detecting the tape" is answerable from a normal `python
        manage.py drive` run: was the color threshold ever matched at all
        (raw_pixel_count), and if so, did the largest blob fail the shape
        filter and why.

        output: (raw_pixel_count, reasons) - reasons is a list of strings
                describing why the largest raw blob (if any) was rejected by
                the shape filter, or [] if either no blob exists or one passed
        '''
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

    def _decide_action(self):
        '''
        Human-readable recommended action for the terminal diagnostics below.
        Phase 1 is detection-only (see class docstring) - this never actually
        changes steering/throttle, it just states what a future avoidance
        maneuver *would* do, so the detector can be verified end-to-end before
        that maneuver is built.
        '''
        if self.cone_detected:
            return "SWERVE - cone confirmed in our lane, steer toward other lane"
        if self.cone_in_our_lane:
            return (f"cone candidate in our lane, confirming "
                     f"({self._pending_frames}/{self.cone_trigger_frames} frames) - hold lane for now")
        if self.cone_x is not None:
            return "blue blob seen but not in our lane - hold lane"
        return "no blue tape visible - hold lane"

    def _x_in_bounds(self, x, bounds):
        lo, hi = bounds
        if x is None or lo is None:
            return False
        return lo - self.lane_margin_px <= x <= hi + self.lane_margin_px

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

    def _log_raw_detection(self, band_rgb, band_hsv, mask):
        '''
        Terminal diagnostics, separate from run()'s in-lane/debounce gating
        (already settled by the time this runs, see run()): prints whenever
        a blue blob enters/leaves the scan band, with
        the actually-sampled HSV/RGB color value - what to watch when tuning
        BLUE_HSV_THRESHOLD_LOW/HIGH or CONE_SCAN_Y/HEIGHT against the real
        tape on the car. It fires regardless of whether lane geometry is
        available, so it still confirms the color detector itself is working
        even if lane/yellow_x etc. turn out not to be wired (see
        _warn_if_lane_geometry_missing).

        Also prints a heartbeat every CONE_LOG_INTERVAL_FRAMES frames
        *regardless* of whether anything is detected, so `python manage.py
        drive`'s terminal always shows current status + recommended action
        (this is the actual per-run() answer to "is it seeing the tape and
        what would it do about it") - and, when nothing passes the shape
        filter, *why* (see _describe_mask): raw_pixel_count==0 means the
        color threshold itself never matched anything (tune
        BLUE_HSV_THRESHOLD_LOW/HIGH), while a nonzero count with rejection
        reasons means a blue blob exists but is the wrong size/shape (tune
        CONE_MIN_AREA_PX/CONE_MAX_WIDTH_PX or check CONE_SCAN_Y/HEIGHT).
        '''
        raw_detected = self.cone_x is not None
        action = self._decide_action()
        self._frame_count += 1
        heartbeat_due = self._frame_count % self.log_interval_frames == 0

        if raw_detected and not self._was_raw_detected:
            mean_hsv, mean_rgb = self._sample_color(band_rgb, band_hsv, self.cone_x)
            lane_note = "IN our lane" if self.cone_in_our_lane else "NOT in our lane (or lane unknown)"
            logger.info(
                f"[cone_tape] blue tape candidate at x={self.cone_x:.1f}, scan_y={self.scan_y} - "
                f"sampled color HSV=({mean_hsv[0]:.0f},{mean_hsv[1]:.0f},{mean_hsv[2]:.0f}) "
                f"RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f}) - {lane_note} - "
                f"ACTION: {action}"
            )
        elif not raw_detected and self._was_raw_detected:
            logger.info(f"[cone_tape] blue tape no longer visible in scan band - ACTION: {action}")
        elif heartbeat_due:
            if raw_detected:
                mean_hsv, mean_rgb = self._sample_color(band_rgb, band_hsv, self.cone_x)
                logger.info(
                    f"[cone_tape] blue tape still at x={self.cone_x:.1f} "
                    f"HSV=({mean_hsv[0]:.0f},{mean_hsv[1]:.0f},{mean_hsv[2]:.0f}) - ACTION: {action}"
                )
            else:
                raw_pixel_count, reasons = self._describe_mask(mask)
                if raw_pixel_count == 0:
                    detail = "no pixels matched BLUE_HSV_THRESHOLD_LOW/HIGH in scan band"
                elif reasons:
                    detail = f"{raw_pixel_count}px matched color but rejected: " + "; ".join(reasons)
                else:
                    detail = f"{raw_pixel_count}px matched color, no blob"
                logger.info(f"[cone_tape] no blue tape detected ({detail}) - ACTION: {action}")

        self._was_raw_detected = raw_detected

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

        self.cone_x, mask = self.detect_cone(band_hsv)
        self.cone_in_our_lane = self._x_in_bounds(self.cone_x, our_lane)

        if self.cone_in_our_lane:
            self._pending_frames += 1
        else:
            self._pending_frames = 0

        # cone_detected (and therefore _decide_action's "SWERVE" verdict)
        # must be settled *before* _log_raw_detection runs below, so the
        # printed ACTION reflects this frame's decision instead of lagging
        # one frame behind it.
        was_detected = self.cone_detected
        self.cone_detected = self._pending_frames >= self.cone_trigger_frames
        if self.cone_detected and not was_detected:
            logger.info(f"cone marker detected in our lane at x={self.cone_x:.1f} "
                         f"(held {self._pending_frames} frames) - ACTION: {self._decide_action()}")
        elif was_detected and not self.cone_detected:
            logger.info(f"cone marker no longer in our lane - ACTION: {self._decide_action()}")

        self._log_raw_detection(band_rgb, band_hsv, mask)
        self._warn_if_lane_geometry_missing(yellow_x, white_x)

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
