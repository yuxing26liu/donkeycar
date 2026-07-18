import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RobustLineFollower:
    '''
    OpenCV based controller. Based on LineFollower (see line_follower.py),
    with the same horizontal-slice + HSV-threshold + PID approach, but
    hardened against two specific failure modes seen on a real track:

    1. Small false-positive patches (background clutter, glare) getting
       treated as the line, because the original picks whichever single
       column has the most matching pixels with no notion of how wide or
       solid that match is.
    2. The tracked line jumping to a different, unrelated line elsewhere
       in the scan row (e.g. another lane's edge), because the original
       has no memory of where the line was last seen - it just re-picks
       the global strongest column every frame.

    This still tracks exactly one line (whichever color you configure via
    COLOR_THRESHOLD_LOW/HIGH) - it does not try to fuse multiple lines.
    '''
    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE
        self.scan_y = cfg.SCAN_Y   # num pixels from the top to start horiz scan
        self.scan_height = cfg.SCAN_HEIGHT  # num pixels high to grab from horiz scan
        self.color_thr_low = np.asarray(cfg.COLOR_THRESHOLD_LOW)
        self.color_thr_hi = np.asarray(cfg.COLOR_THRESHOLD_HIGH)
        self.target_pixel = cfg.TARGET_PIXEL
        self.target_threshold = cfg.TARGET_THRESHOLD
        self.confidence_threshold = cfg.CONFIDENCE_THRESHOLD
        self.steering = 0.0
        self.throttle = cfg.THROTTLE_INITIAL
        self.delta_th = cfg.THROTTLE_STEP
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        self.pid_st = pid

        # --- new config, all optional (getattr with defaults) so this part
        # --- works with an existing cv_control myconfig.py unmodified ---

        # CHANGE 1: minimum width (in pixels) a contiguous run of matching
        # columns must have to be treated as a real line rather than noise.
        self.min_line_width = getattr(cfg, 'MIN_LINE_WIDTH', 8)

        # CHANGE 2: tracking/continuity state and its two tuning knobs.
        # tracked_position is the last position we locked onto; a new
        # candidate is only accepted if it's within max_jump_pixels of it.
        # If we go reacquire_after_frames frames without a plausible
        # candidate, we drop the gate and lock onto whatever is strongest,
        # so the algorithm can still recover after a real, sustained loss
        # (e.g. driving past the end of a dashed segment).
        self.tracked_position = None
        self.lost_frames = 0
        self.max_jump_pixels = getattr(cfg, 'MAX_JUMP_PIXELS', 40)
        self.reacquire_after_frames = getattr(cfg, 'REACQUIRE_AFTER_FRAMES', 15)

        # CHANGE 1 (cont'd): morphological opening kernel used to erase
        # speckled false-positive pixels before we look for line-shaped
        # blobs. Set to 0 or 1 to disable.
        self.morph_kernel_size = getattr(cfg, 'MORPH_KERNEL_SIZE', 3)

        # CHANGE 4: how much to ease steering back toward straight on each
        # consecutive frame where no line is found/tracked, instead of
        # freezing at the last commanded steering value.
        self.lost_steering_decay = getattr(cfg, 'LOST_STEERING_DECAY', 0.85)

        # CHANGE 5: exponential smoothing applied to the measured position
        # before it reaches the PID, to remove frame-to-frame jitter (e.g.
        # from a dashed line's edge shifting slightly dash-to-dash) that
        # would otherwise get amplified by the PID's derivative term into
        # oscillating ("wobbly") steering. Lower alpha = smoother/slower
        # to react; higher alpha = snappier/noisier.
        self.smoothing_alpha = getattr(cfg, 'POSITION_SMOOTHING_ALPHA', 0.4)
        self.smoothed_position = None
        # set inside get_i_color(): True when we just (re)locked onto a
        # line with no prior continuity (startup, or recovered from a long
        # loss) - in that case change 5's smoothing should snap to the new
        # position immediately instead of slowly blending in from a stale
        # smoothed value that may be far away.
        self.just_reacquired = False

    def get_i_color(self, cam_img):
        '''
        input: cam_image, an RGB numpy array
        output: (position, confidence, mask)
          position: x index of the tracked line, or None if nothing
                     plausible was found this frame.
          confidence: fraction of the scanned area's pixels that belong to
                     the chosen line (0..1).
          mask: the thresholded (and cleaned-up) binary mask, for overlay.
        '''
        # take a horizontal slice of the image
        iSlice = self.scan_y
        scan_line = cam_img[iSlice: iSlice + self.scan_height, :, :]

        # convert to HSV color space
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_RGB2HSV)

        # make a mask of the colors in our range we are looking for
        mask = cv2.inRange(img_hsv, self.color_thr_low, self.color_thr_hi)

        # CHANGE 1: erase small speckled false-positive pixels (glare,
        # background clutter) before we try to find the line in the mask.
        # An "opening" (erode then dilate) removes anything narrower than
        # the kernel while leaving a genuinely solid line intact.
        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # per-column count of matching pixels (0..scan_height)
        col_counts = np.count_nonzero(mask, axis=0)

        # CHANGE 1 (cont'd): instead of a single argmax column, find every
        # contiguous run of matching columns ("blobs") and discard any run
        # narrower than min_line_width. A stray patch of a few noisy
        # columns can no longer out-compete or masquerade as the line.
        is_match = col_counts > 0
        padded = np.concatenate(([0], is_match.astype(int), [0]))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        candidates = []
        for start, end in zip(starts, ends):
            width = end - start
            if width >= self.min_line_width:
                weight = int(np.sum(col_counts[start:end]))
                center = start + width / 2.0
                candidates.append((center, weight))

        if not candidates:
            self.lost_frames += 1
            return None, 0.0, mask

        # CHANGE 2: pick the candidate nearest to where we last tracked the
        # line, not the single strongest one - unless we don't have a
        # track yet, or have been lost long enough that we should stop
        # waiting and re-acquire on the strongest candidate available.
        if self.tracked_position is None or self.lost_frames > self.reacquire_after_frames:
            chosen = max(candidates, key=lambda c: c[1])
            # CHANGE 5 (cont'd): flag this as a fresh lock so the smoothing
            # filter in run() snaps to it instead of blending in slowly.
            self.just_reacquired = True
        else:
            nearest = min(candidates, key=lambda c: abs(c[0] - self.tracked_position))
            if abs(nearest[0] - self.tracked_position) <= self.max_jump_pixels:
                chosen = nearest
                self.just_reacquired = False
            else:
                # the closest candidate is still an implausible jump away
                # (e.g. a different line elsewhere in the row) - treat this
                # frame as a miss rather than snapping to it.
                self.lost_frames += 1
                return None, 0.0, mask

        self.tracked_position = chosen[0]
        self.lost_frames = 0

        total_pixels = mask.shape[0] * mask.shape[1]
        confidence = chosen[1] / float(total_pixels)

        return chosen[0], confidence, mask

    def run(self, cam_img):
        '''
        main runloop of the CV controller
        input: cam_image, an RGB numpy array
        output: steering, throttle, and the image.
        '''
        if cam_img is None:
            return 0, 0, None

        # CHANGE 6: default the target to the horizontal center of the
        # image, as soon as we know the image width - not "wherever the
        # line happened to be on the first frame we detected it"
        # (LineFollower's behavior, inherited unchanged in earlier
        # versions of this file). That auto-capture-from-first-frame
        # behavior is meant for a two-lane "stay in whichever lane you
        # started in" setup; for a single isolated line you want to
        # actually drive centered on it, so if TARGET_PIXEL isn't
        # explicitly set in config, this now means "the middle of the
        # image" rather than "whatever offset we started at."
        if self.target_pixel is None:
            self.target_pixel = cam_img.shape[1] / 2.0
            logger.info(f"Defaulting target line position to image center = {self.target_pixel}")

        if self.pid_st.setpoint != self.target_pixel:
            self.pid_st.setpoint = self.target_pixel

        position, confidence, mask = self.get_i_color(cam_img)

        # CHANGE 5 (cont'd): smooth the raw per-frame measurement before it
        # reaches the PID. On a fresh (re)lock, snap directly to the new
        # position instead of blending from a possibly-stale old value.
        if position is not None:
            if self.smoothed_position is None or self.just_reacquired:
                self.smoothed_position = position
            else:
                self.smoothed_position = (self.smoothing_alpha * position
                                           + (1 - self.smoothing_alpha) * self.smoothed_position)

        if position is not None and confidence >= self.confidence_threshold:
            self.steering = self.pid_st(self.smoothed_position)

            # slow down linearly when away from ideal, and speed up when close
            if abs(self.smoothed_position - self.target_pixel) > self.target_threshold:
                self.throttle = max(self.throttle - self.delta_th, self.throttle_min)
            else:
                self.throttle = min(self.throttle + self.delta_th, self.throttle_max)
        else:
            # CHANGE 4: the original holds steering/throttle unchanged here,
            # which means a lost line drives "blind" at whatever command it
            # last had. Instead, ease steering back toward straight and
            # slow down each consecutive frame the line isn't found, so a
            # brief dropout degrades gracefully instead of committing to a
            # stale command.
            logger.info(f"No line detected: confidence {confidence:.4f} < {self.confidence_threshold}")
            self.steering *= self.lost_steering_decay
            self.throttle = max(self.throttle - self.delta_th, self.throttle_min)

        if self.overlay_image:
            cam_img = self.overlay_display(cam_img, mask, position, confidence)

        return self.steering, self.throttle, cam_img

    def overlay_display(self, cam_img, mask, position, confidence):
        '''
        composite mask on top the original image.
        show some values we are using for control
        '''
        mask_exp = np.stack((mask,) * 3, axis=-1)
        iSlice = self.scan_y
        img = np.copy(cam_img)
        img[iSlice: iSlice + self.scan_height, :, :] = mask_exp

        # CHANGE 2 (cont'd) / CHANGE 5/6 (cont'd): mark the raw detection,
        # the smoothed value actually driving the PID, and the target, so
        # it's visually obvious in the web UI when the raw track jumps,
        # how much the smoothing is correcting for it, and whether the
        # car is actually centering on the target.
        if position is not None:
            xi = int(np.clip(position, 0, img.shape[1] - 1))
            cv2.line(img, (xi, iSlice), (xi, iSlice + self.scan_height), (0, 255, 255), 1)  # raw = yellow
        if self.smoothed_position is not None:
            xs = int(np.clip(self.smoothed_position, 0, img.shape[1] - 1))
            cv2.line(img, (xs, iSlice), (xs, iSlice + self.scan_height), (255, 0, 0), 2)  # smoothed = blue
        if self.target_pixel is not None:
            xt = int(np.clip(self.target_pixel, 0, img.shape[1] - 1))
            cv2.line(img, (xt, iSlice), (xt, iSlice + self.scan_height), (0, 255, 0), 1)  # target = green

        display_str = []
        display_str.append("STEERING:{:.1f}".format(self.steering))
        display_str.append("THROTTLE:{:.2f}".format(self.throttle))
        display_str.append("I LINE:{}".format(position if position is not None else "LOST"))
        display_str.append("CONF:{:.4f}".format(confidence))

        y = 10
        x = 10
        for s in display_str:
            cv2.putText(img, s, color=(0, 0, 0), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
            y += 10

        return img
