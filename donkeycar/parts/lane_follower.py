"""
lane_follower.py

Successor to LineFollower (line_follower.py) and RobustLineFollower
(robust_line_follower.py). Those track a single line; this reasons about
the full lane so the car can drive in the right lane of a two-lane track
(bounded by a solid edge line on the outside and a dashed yellow line on
the inside).

Pipeline ("advanced lane finding" style - color/gradient threshold +
perspective warp + sliding-window polynomial fit, chosen over a plain
Hough-transform approach specifically because the sliding window carries
the last known position through empty windows, which is what lets it
track a *dashed* line through its gaps):

  1. threshold the incoming RGB frame for yellow (center line) and white
     (edge lines) pixels, optionally OR'd with a Sobel gradient mask.
  2. warp that binary mask to a bird's-eye view so lines that are
     physically parallel stay parallel in image space regardless of
     camera tilt.
  3. take a histogram across the bottom band of the warped mask and
     cluster it into candidate line base positions (however many lines
     are actually visible, not a fixed left/right pair).
  4. for each base position, slide a search window up the frame,
     re-centering on the mean x of whatever pixels fall in each window
     and collecting them all for the fit.
  5. fit a 2nd-degree polynomial (x as a function of y) through each
     line's collected pixels, and classify each fitted line as
     yellow/white by majority vote over the pixels that fed it.
  6. the yellow line is the lane's center boundary; the rightmost white
     line to its right is the outer edge; the midpoint between them
     (extrapolated from a fixed lane width if only one is visible) is
     the target the steering PID chases.

Obstacle avoidance (swerving the target lane-center when something is in
the way, using the OAK-D's depth stream) is the next phase and is not
implemented here yet - this file only drives the right lane.
"""

import cv2
import numpy as np
from simple_pid import PID
import logging

logger = logging.getLogger(__name__)


class LaneFollower:
    '''
    OpenCV based lane-following controller.

    Tracks the dashed yellow center line and the solid right edge line of
    a two-lane track (see module docstring for the pipeline), and steers
    to keep the car centered in the right lane rather than following a
    single line.
    '''

    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE
        self.image_w = cfg.IMAGE_W
        self.image_h = cfg.IMAGE_H

        # --- perspective warp (bird's-eye view) ---
        # Source trapezoid is in the original camera frame; narrower at
        # the top (far from the car) than at the bottom (near the car).
        # This needs the same real-footage verification SCAN_Y got - it's
        # a starting point, not a calibration.
        src = np.float32(cfg.LANE_WARP_SRC)  # [top_left, top_right, bottom_right, bottom_left]
        dst = np.float32([
            [0, 0],
            [self.image_w - 1, 0],
            [self.image_w - 1, self.image_h - 1],
            [0, self.image_h - 1],
        ])
        self.warp_matrix = cv2.getPerspectiveTransform(src, dst)
        self.unwarp_matrix = cv2.getPerspectiveTransform(dst, src)

        # --- color thresholds (HSV) ---
        self.yellow_low = np.asarray(cfg.LANE_YELLOW_THRESHOLD_LOW)
        self.yellow_high = np.asarray(cfg.LANE_YELLOW_THRESHOLD_HIGH)
        self.white_low = np.asarray(cfg.LANE_WHITE_THRESHOLD_LOW)
        self.white_high = np.asarray(cfg.LANE_WHITE_THRESHOLD_HIGH)

        # --- gradient threshold (optional, OR'd into the combined mask) ---
        self.use_gradient = getattr(cfg, 'LANE_USE_GRADIENT', True)
        self.sobel_kernel = getattr(cfg, 'LANE_SOBEL_KERNEL', 3)
        self.sobel_thresh_low = getattr(cfg, 'LANE_SOBEL_THRESH_LOW', 40)
        self.sobel_thresh_high = getattr(cfg, 'LANE_SOBEL_THRESH_HIGH', 255)

        # --- line-base detection (histogram over the bottom band) ---
        self.histogram_band_frac = getattr(cfg, 'LANE_HISTOGRAM_BAND_FRAC', 0.35)
        self.min_line_pixels = getattr(cfg, 'LANE_MIN_LINE_PIXELS', 30)
        self.min_line_separation = getattr(cfg, 'LANE_MIN_LINE_SEPARATION', 20)

        # --- sliding window search ---
        self.n_windows = getattr(cfg, 'LANE_N_WINDOWS', 8)
        self.window_margin = getattr(cfg, 'LANE_WINDOW_MARGIN', 40)
        self.min_pixels_recenter = getattr(cfg, 'LANE_MIN_PIXELS_RECENTER', 15)
        self.min_fit_points = getattr(cfg, 'LANE_MIN_FIT_POINTS', 40)

        # --- lane geometry / steering target ---
        # Bird's-eye lane width in warped pixels, used to extrapolate the
        # lane center when only one boundary line is visible (e.g. the
        # dashed line is mid-gap and only the solid edge was found).
        self.lane_width_px = getattr(cfg, 'LANE_WIDTH_PX', 220)
        # Where in the warped frame the car's own forward centerline
        # projects to - None auto-defaults to the middle of the warped
        # canvas on first use (assumes the warp trapezoid above is
        # centered on the car; override in myconfig.py if the camera
        # mount is off-center).
        self.target_pixel = getattr(cfg, 'LANE_TARGET_PIXEL', None)
        self.target_threshold = getattr(cfg, 'LANE_TARGET_THRESHOLD', 15)

        # --- throttle (shared with the other CV controllers - physical
        # car property, not specific to this detection method) ---
        self.throttle = cfg.THROTTLE_INITIAL
        self.delta_th = cfg.THROTTLE_STEP
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        # --- lost-lane handling ---
        self.lost_steering_decay = getattr(cfg, 'LANE_LOST_STEERING_DECAY', 0.85)
        # After this many *consecutive* lost frames, stop instead of just
        # decaying steering while continuing to creep at throttle_min - a
        # sustained loss (drove off the track, missed a turn entirely) means
        # "stop and wait to be picked up," not "keep easing toward straight
        # forever." Ported from a teammate's lane_follower.py (object-
        # detection branch) after it was exercised by a real sharp-turn
        # failure on the car.
        self.max_lost_frames = getattr(cfg, 'LANE_MAX_LOST_FRAMES', 40)
        self.overlay_alpha = getattr(cfg, 'LANE_OVERLAY_ALPHA', 0.3)

        self.steering = 0.0
        self.lost_frames = 0
        self.pid_st = pid
        # Clamp on the pid object itself, not just the returned value - this
        # also bounds simple_pid's internal integral accumulator so it can't
        # wind up past what the actuator can ever use. Same reasoning as
        # BetterLineFollower (origin/better-line-follower).
        self.pid_st.output_limits = (-1.0, 1.0)

    # ------------------------------------------------------------------
    # thresholding
    # ------------------------------------------------------------------
    def _threshold(self, cam_img):
        '''
        input: cam_image, an RGB numpy array
        output: (combined_mask, yellow_mask, white_mask), each a uint8
                0/255 binary mask the same size as cam_img.
        '''
        img_hsv = cv2.cvtColor(cam_img, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(img_hsv, self.yellow_low, self.yellow_high)
        white_mask = cv2.inRange(img_hsv, self.white_low, self.white_high)
        combined = cv2.bitwise_or(yellow_mask, white_mask)

        if self.use_gradient:
            gray = cv2.cvtColor(cam_img, cv2.COLOR_RGB2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
            abs_sobel = np.absolute(sobel_x)
            max_val = np.max(abs_sobel)
            scaled_sobel = (
                np.uint8(255 * abs_sobel / max_val) if max_val > 0
                else np.zeros_like(gray, dtype=np.uint8)
            )
            gradient_mask = cv2.inRange(
                scaled_sobel, self.sobel_thresh_low, self.sobel_thresh_high
            )
            combined = cv2.bitwise_or(combined, gradient_mask)

        return combined, yellow_mask, white_mask

    # ------------------------------------------------------------------
    # line-base detection
    # ------------------------------------------------------------------
    def _find_line_bases(self, warped_mask):
        '''
        Cluster the bottom band of a warped binary mask into candidate
        line base positions - however many distinct lines are visible,
        not a fixed left/right pair.
        output: list of (center_x, weight) tuples, sorted left to right.
        '''
        h, w = warped_mask.shape
        band_start = int(h * (1.0 - self.histogram_band_frac))
        band = warped_mask[band_start:, :]
        col_counts = np.count_nonzero(band, axis=0)

        nonzero_cols = np.where(col_counts > 0)[0]
        if nonzero_cols.size == 0:
            return []

        groups = [[nonzero_cols[0]]]
        for c in nonzero_cols[1:]:
            if c - groups[-1][-1] <= self.min_line_separation:
                groups[-1].append(c)
            else:
                groups.append([c])

        bases = []
        for g in groups:
            g = np.array(g)
            weight = int(col_counts[g].sum())
            if weight < self.min_line_pixels:
                continue
            center = float(np.average(g, weights=col_counts[g]))
            bases.append((center, weight))

        return bases

    # ------------------------------------------------------------------
    # sliding window search
    # ------------------------------------------------------------------
    def _sliding_window_search(self, warped_mask, base_x):
        '''
        Search up from a base x position, re-centering each window on
        the pixels found in it. A window with no pixels (e.g. a gap in
        a dashed line) simply leaves the search center unchanged instead
        of losing the line, so the search keeps following the same
        column band until the line reappears.
        output: (points_x, points_y) - all collected pixel coordinates,
                as parallel python lists.
        '''
        h, w = warped_mask.shape
        window_height = h // self.n_windows
        current_x = base_x

        points_x = []
        points_y = []

        for window in range(self.n_windows):
            y_high = h - window * window_height
            y_low = 0 if window == self.n_windows - 1 else h - (window + 1) * window_height

            x_low = int(max(0, current_x - self.window_margin))
            x_high = int(min(w, current_x + self.window_margin))
            if x_high <= x_low:
                continue

            window_slice = warped_mask[y_low:y_high, x_low:x_high]
            ys, xs = np.nonzero(window_slice)

            if xs.size > 0:
                points_x.extend((xs + x_low).tolist())
                points_y.extend((ys + y_low).tolist())

                if xs.size >= self.min_pixels_recenter:
                    current_x = x_low + float(np.mean(xs))

        return points_x, points_y

    def _fit_line(self, points_x, points_y):
        '''Fit x = f(y) (2nd degree). Returns None if too few points.'''
        if len(points_x) < self.min_fit_points:
            return None
        return np.polyfit(points_y, points_x, 2)

    def _eval_fit(self, fit, y):
        return fit[0] * y ** 2 + fit[1] * y + fit[2]

    def _classify_color(self, points_x, points_y, yellow_mask, white_mask):
        '''Majority vote of which color mask the line's pixels came from.'''
        ys = np.asarray(points_y, dtype=np.int32)
        xs = np.asarray(points_x, dtype=np.int32)
        yellow_hits = int(np.count_nonzero(yellow_mask[ys, xs]))
        white_hits = int(np.count_nonzero(white_mask[ys, xs]))
        return 'yellow' if yellow_hits >= white_hits else 'white'

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, cam_img):
        '''
        main runloop of the CV controller
        input: cam_image, an RGB numpy array
        output: steering, throttle, and the (optionally annotated) image.
        '''
        if cam_img is None:
            return 0.0, 0.0, None

        combined_mask, yellow_mask, white_mask = self._threshold(cam_img)

        warp_size = (self.image_w, self.image_h)
        warped_mask = cv2.warpPerspective(
            combined_mask, self.warp_matrix, warp_size, flags=cv2.INTER_NEAREST
        )
        warped_yellow = cv2.warpPerspective(
            yellow_mask, self.warp_matrix, warp_size, flags=cv2.INTER_NEAREST
        )
        warped_white = cv2.warpPerspective(
            white_mask, self.warp_matrix, warp_size, flags=cv2.INTER_NEAREST
        )

        bases = self._find_line_bases(warped_mask)

        lines = []
        for base_x, _weight in bases:
            points_x, points_y = self._sliding_window_search(warped_mask, base_x)
            fit = self._fit_line(points_x, points_y)
            if fit is None:
                continue
            color = self._classify_color(points_x, points_y, warped_yellow, warped_white)
            lines.append({
                'x_base': base_x,
                'fit': fit,
                'points_x': points_x,
                'points_y': points_y,
                'color': color,
            })

        yellow_candidates = [l for l in lines if l['color'] == 'yellow']
        white_candidates = [l for l in lines if l['color'] == 'white']

        center_line = (
            max(yellow_candidates, key=lambda l: len(l['points_x']))
            if yellow_candidates else None
        )

        if center_line is not None:
            right_candidates = [l for l in white_candidates if l['x_base'] > center_line['x_base']]
        else:
            right_candidates = white_candidates

        # Always the rightmost candidate, not "most pixels" - with no
        # center line to bound the search, picking by pixel count risks
        # locking onto the LEFT solid edge instead (e.g. if it's more
        # visible than the right one that frame), which would steer the
        # car the wrong way rather than just losing the lane.
        right_edge_line = (
            max(right_candidates, key=lambda l: l['x_base'])
            if right_candidates else None
        )

        eval_y = self.image_h - 1
        yellow_x = self._eval_fit(center_line['fit'], eval_y) if center_line else None
        right_x = self._eval_fit(right_edge_line['fit'], eval_y) if right_edge_line else None

        if yellow_x is not None and right_x is not None:
            lane_center = (yellow_x + right_x) / 2.0
        elif yellow_x is not None:
            lane_center = yellow_x + self.lane_width_px / 2.0
        elif right_x is not None:
            lane_center = right_x - self.lane_width_px / 2.0
        else:
            lane_center = None

        if self.target_pixel is None:
            self.target_pixel = self.image_w / 2.0
            logger.info(f"Defaulting lane target to warped image center = {self.target_pixel}")

        if self.pid_st.setpoint != self.target_pixel:
            self.pid_st.setpoint = self.target_pixel

        if lane_center is not None:
            self.lost_frames = 0
            self.steering = self.pid_st(lane_center)

            if abs(lane_center - self.target_pixel) > self.target_threshold:
                self.throttle = max(self.throttle - self.delta_th, self.throttle_min)
            else:
                self.throttle = min(self.throttle + self.delta_th, self.throttle_max)
        else:
            self.lost_frames += 1
            self.steering *= self.lost_steering_decay
            if self.lost_frames > self.max_lost_frames:
                if self.lost_frames == self.max_lost_frames + 1:
                    logger.warning(
                        f"Lane lost for more than LANE_MAX_LOST_FRAMES={self.max_lost_frames} "
                        f"consecutive frames; stopping instead of holding stale output.")
                self.throttle = max(self.throttle - self.delta_th, 0.0)
            else:
                logger.info(f"No lane boundaries detected ({self.lost_frames}/{self.max_lost_frames})")
                self.throttle = max(self.throttle - self.delta_th, self.throttle_min)

        if self.overlay_image:
            cam_img = self._overlay_display(cam_img, center_line, right_edge_line, lane_center)

        return self.steering, self.throttle, cam_img

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------
    def _unwarp_points(self, xs, ys):
        pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
        unwarped = cv2.perspectiveTransform(pts, self.unwarp_matrix)
        return unwarped.reshape(-1, 2).astype(np.int32)

    def _overlay_display(self, cam_img, center_line, right_edge_line, lane_center):
        img = np.copy(cam_img)
        ys = np.arange(0, self.image_h)

        if center_line is not None or right_edge_line is not None:
            if center_line is not None:
                left_xs = self._eval_fit(center_line['fit'], ys)
            else:
                left_xs = self._eval_fit(right_edge_line['fit'], ys) - self.lane_width_px

            if right_edge_line is not None:
                right_xs = self._eval_fit(right_edge_line['fit'], ys)
            else:
                right_xs = self._eval_fit(center_line['fit'], ys) + self.lane_width_px

            left_pts = np.stack([left_xs, ys], axis=1)
            right_pts = np.flipud(np.stack([right_xs, ys], axis=1))
            polygon_warped = np.concatenate([left_pts, right_pts], axis=0).astype(np.int32)

            warped_overlay = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
            cv2.fillPoly(warped_overlay, [polygon_warped], (0, 200, 0))
            unwarped_overlay = cv2.warpPerspective(
                warped_overlay, self.unwarp_matrix, (self.image_w, self.image_h)
            )
            img = cv2.addWeighted(img, 1.0, unwarped_overlay, self.overlay_alpha, 0)

        for line, color in ((center_line, (255, 255, 0)), (right_edge_line, (255, 0, 255))):
            if line is None:
                continue
            xs = self._eval_fit(line['fit'], ys)
            in_bounds = (xs >= 0) & (xs < self.image_w)
            if not np.any(in_bounds):
                continue
            pts = self._unwarp_points(xs[in_bounds], ys[in_bounds])
            cv2.polylines(img, [pts], False, color, 2)

        display_str = []
        display_str.append("STEERING:{:.2f}".format(self.steering))
        display_str.append("THROTTLE:{:.2f}".format(self.throttle))
        display_str.append(
            "LANE_CENTER:{}".format(f"{lane_center:.0f}" if lane_center is not None else "LOST")
        )
        display_str.append("TARGET:{:.0f}".format(self.target_pixel))

        y = 10
        x = 10
        for s in display_str:
            cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
            y += 10

        return img
