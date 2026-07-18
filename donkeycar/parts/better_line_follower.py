import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BetterLineFollower:
    '''
    OpenCV based controller - first iteration on top of LineFollower.

    Same overall shape as LineFollower: take a horizontal slice of the
    image, threshold it in HSV to find "yellow", and steer a PID
    controller to keep that position centered. Three changes on top of
    that, aimed specifically at the reported wobble/veering caused by
    sunlit rocks and gravel in the background matching the yellow
    threshold:

      1. Connected-component blob filtering (replaces the raw
         "argmax over a flattened column histogram"). The color mask
         is broken into separate connected blobs with
         cv2.connectedComponentsWithStats, and each blob is scored
         against its physical shape - pixel area, bounding-box width,
         and height/width aspect ratio - before it is allowed to be a
         candidate at all. A handful of small rock/gravel speckles can
         still win a flattened per-column sum even though no single
         blob is line-shaped; per-blob filtering rejects them
         individually instead. Wide, flat blobs (a sunlit patch of
         pavement or a wall) are rejected the same way. Whichever
         surviving blob has the largest area is chosen as the line.
         This is a stateless, single-frame filter - it does not track
         the line across frames (see "Not included" below).

      2. RGB channel-dominance test, applied in addition to the
         existing HSV threshold. A pixel must have R and G both
         clearly brighter than B, and R and G close to each other, to
         count as "yellow". This is a different signal than HSV: a
         shadow falling across the tape lowers its saturation and
         value, which can push it outside a tight HSV band, but it
         does not change the fact that the tape's red and green
         channels stay well above blue. Conversely gray pavement and
         green foliage fail the dominance test regardless of how HSV
         is tuned. The two tests catch different failure modes, so
         they're combined (AND), not one replacing the other.

      3. Target pixel defaults to the image center, not the first
         detection. LineFollower's TARGET_PIXEL=None means "whatever
         color-matches on the very first frame becomes the steering
         target forever after" - if that first frame happens to catch
         a rock instead of the line, the controller tracks the rock
         from then on. This computes the center from the actual
         incoming frame's width on the first call to run(), rather
         than from a configured resolution, so it can't be wrong even
         if IMAGE_W/IMAGE_H in config doesn't match what the camera is
         actually producing (see camera-resolution note below).
         cfg.TARGET_PIXEL still wins if explicitly set to a number.

    Also fixed: LineFollower's run() returns a 4-tuple
    (0, 0, False, None) on its "no frame yet" path but a 3-tuple
    everywhere else. Vehicle.py's Memory.put() zips positionally
    against the 3 configured output keys, so this doesn't crash, but
    it silently stores cv/image_array as the boolean False instead of
    None on that path. This version always returns a 3-tuple.

    Deliberately NOT included in this pass (each is a bigger change
    and easier to validate in isolation once these three are proven
    out on the car):
      - persistent frame-to-frame tracking, a max-jump clamp, or a
        "reacquire after N lost frames" state machine
      - reference-resolution scaling of SCAN_Y / SCAN_HEIGHT / etc.
      - stopping the car (vs. just holding last output) after
        sustained loss-of-line

    Drop-in replacement for LineFollower: same constructor signature
    (pid, cfg) and same run(cam_img) -> (steering, throttle, image)
    contract, so it can be swapped in via CV_CONTROLLER_MODULE /
    CV_CONTROLLER_CLASS in config without touching manage.py.
    '''

    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE
        self.scan_y = cfg.SCAN_Y                # num pixels from the top to start horiz scan
        self.scan_height = cfg.SCAN_HEIGHT      # num pixels high to grab from horiz scan
        self.color_thr_low = np.asarray(cfg.COLOR_THRESHOLD_LOW)   # hsv dark yellow
        self.color_thr_hi = np.asarray(cfg.COLOR_THRESHOLD_HIGH)   # hsv light yellow

        # change 3: if None, resolved from the first real frame's width
        # in run() below, instead of from the first frame's detection.
        self.target_pixel = cfg.TARGET_PIXEL
        self.target_threshold = cfg.TARGET_THRESHOLD  # min distance from target_pixel before a steering change is made

        self.steering = 0.0             # from -1 to 1
        self.throttle = cfg.THROTTLE_INITIAL  # from -1 to 1
        self.delta_th = cfg.THROTTLE_STEP     # how much to change throttle when off
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        # change 2: RGB dominance test, layered on top of the HSV mask.
        # New knobs (all optional; sane defaults if myconfig doesn't set them).
        self.enable_color_dominance = getattr(cfg, "ENABLE_COLOR_DOMINANCE", True)
        self.color_min_dominance = getattr(cfg, "COLOR_MIN_DOMINANCE", 8)
        self.color_max_channel_diff = getattr(cfg, "COLOR_MAX_CHANNEL_DIFF", 30)

        # change 1: connected-component shape filtering knobs.
        self.morph_kernel_px = getattr(cfg, "MASK_MORPH_KERNEL_PX", 3)
        self.min_line_area_px = getattr(cfg, "MIN_LINE_AREA_PX", 150)
        self.max_line_width_px = getattr(cfg, "MAX_LINE_WIDTH_PX", 250)
        self.min_line_aspect_ratio = getattr(cfg, "MIN_LINE_ASPECT_RATIO", 0.15)

        self.pid_st = pid

    def get_color_mask(self, scan_line):
        '''
        input: scan_line, an RGB numpy array (the cropped scan band)
        output: binary (0/255) uint8 mask of pixels that pass the HSV
        threshold and, if enabled, the RGB dominance test
        '''
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, self.color_thr_low, self.color_thr_hi)

        if self.enable_color_dominance:
            # int16 so R-B / G-B / R-G can't wrap around like uint8 would
            r = scan_line[:, :, 0].astype(np.int16)
            g = scan_line[:, :, 1].astype(np.int16)
            b = scan_line[:, :, 2].astype(np.int16)
            dominance = (
                (r - b >= self.color_min_dominance)
                & (g - b >= self.color_min_dominance)
                & (np.abs(r - g) <= self.color_max_channel_diff)
            )
            mask = cv2.bitwise_and(mask, (dominance.astype(np.uint8) * 255))

        if self.morph_kernel_px > 1:
            # clears isolated 1-2px noise before component analysis;
            # keep this small so it doesn't erode thin/distant line segments
            kernel = np.ones((self.morph_kernel_px, self.morph_kernel_px), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def select_line_component(self, mask):
        '''
        input: mask, binary (0/255) uint8 image (the scan band's color mask)
        output: (x, area) of the selected component's centroid x and its
        pixel area, or (None, 0) if nothing in the mask looks like a line

        Runs connected-components on the mask and rejects blobs that
        don't look like a line cross-section: too small (noise/gravel),
        too wide (a wall or a sunlit patch of pavement), or too flat
        relative to their width (same wide-patch case, but as a
        resolution-independent ratio rather than a fixed pixel count).
        Of what's left, the largest by area wins - this is a direct,
        shape-aware replacement for the old "biggest column sum" pick,
        not a temporal tracker (nothing here depends on previous frames).
        '''
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        best_label = None
        best_area = 0
        for label in range(1, num_labels):  # label 0 is background, always skip it
            area = stats[label, cv2.CC_STAT_AREA]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]

            if area < self.min_line_area_px:
                continue
            if width > self.max_line_width_px:
                continue
            aspect = (height / width) if width > 0 else 0
            if aspect < self.min_line_aspect_ratio:
                continue

            if area > best_area:
                best_area = area
                best_label = label

        if best_label is None:
            return None, 0

        x = centroids[best_label][0]
        return x, int(best_area)

    def run(self, cam_img):
        '''
        main runloop of the CV controller
        input: cam_image, an RGB numpy array
        output: steering, throttle, and the image.
        If overlay_image is True, then the output image includes an
        overlay showing how the algorithm is working; otherwise the
        image is just passed through untouched.
        '''
        if cam_img is None:
            return 0, 0, None

        if self.target_pixel is None:
            # change 3: center of the actual incoming frame - see class
            # docstring for why this isn't taken from cfg.IMAGE_W instead.
            self.target_pixel = cam_img.shape[1] // 2
            logger.info(f"Defaulting target pixel to image center = {self.target_pixel}")

        iSlice = self.scan_y
        scan_line = cam_img[iSlice: iSlice + self.scan_height, :, :]

        if scan_line.size == 0:
            # SCAN_Y/SCAN_HEIGHT fall outside the actual frame - most likely
            # cause is cfg geometry tuned for a different resolution than
            # the camera is currently producing. Hold last output and warn
            # instead of failing deeper inside OpenCV.
            logger.warning(
                f"Empty scan slice: cam_img shape={cam_img.shape}, "
                f"SCAN_Y={self.scan_y}, SCAN_HEIGHT={self.scan_height}")
            return self.steering, self.throttle, cam_img

        mask = self.get_color_mask(scan_line)
        line_x, area = self.select_line_component(mask)

        if self.pid_st.setpoint != self.target_pixel:
            # this is the target of our steering PID controller
            self.pid_st.setpoint = self.target_pixel

        if line_x is not None:
            # invoke the controller with the current line position
            # get the new steering value as it chases the ideal
            self.steering = self.pid_st(line_x)

            # slow down linearly when away from ideal, and speed up when close
            if abs(line_x - self.target_pixel) > self.target_threshold:
                # we will be turning, so slow down
                if self.throttle > self.throttle_min:
                    self.throttle -= self.delta_th
                if self.throttle < self.throttle_min:
                    self.throttle = self.throttle_min
            else:
                # we are going straight, so speed up
                if self.throttle < self.throttle_max:
                    self.throttle += self.delta_th
                if self.throttle > self.throttle_max:
                    self.throttle = self.throttle_max
        else:
            logger.info("No line detected: no component passed shape filtering")

        # show some diagnostics
        if self.overlay_image:
            cam_img = self.overlay_display(cam_img, mask, line_x, area)

        return self.steering, self.throttle, cam_img

    def overlay_display(self, cam_img, mask, line_x, area):
        '''
        composite mask on top of the original image, plus the selected
        component's centroid and the target pixel, so the two new
        filtering stages (shape filter, dominance test) are visible
        while testing, not just their end effect on steering.
        '''
        mask_exp = np.stack((mask,) * 3, axis=-1)
        iSlice = self.scan_y
        img = np.copy(cam_img)
        img[iSlice: iSlice + self.scan_height, :, :] = mask_exp

        target_x = int(self.target_pixel)
        cv2.line(img, (target_x, iSlice), (target_x, iSlice + self.scan_height),
                 color=(255, 255, 255), thickness=1)

        if line_x is not None:
            mid_y = iSlice + self.scan_height // 2
            cv2.circle(img, (int(line_x), mid_y), radius=5, color=(255, 0, 0), thickness=-1)

        display_str = []
        display_str.append("STEERING:{:.2f}".format(self.steering))
        display_str.append("THROTTLE:{:.2f}".format(self.throttle))
        display_str.append("LINE_X:{}".format(int(line_x) if line_x is not None else "None"))
        display_str.append("AREA:{:d}".format(area))

        y = 10
        x = 10

        for s in display_str:
            cv2.putText(img, s, color=(0, 0, 0), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
            y += 10

        return img
