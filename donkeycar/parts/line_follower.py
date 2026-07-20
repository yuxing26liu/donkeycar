import cv2
import numpy as np
from simple_pid import PID
import logging

logger = logging.getLogger(__name__)


class LineFollower:
    '''
    OpenCV based controller
    This controller takes a horizontal slice of the image at a set Y coordinate.
    Then it converts to HSV and does a color thresh hold to find the yellow pixels.
    It does a histogram to find the pixel of maximum yellow. Then is uses that iPxel
    to guid a PID controller which seeks to maintain the max yellow at the same point
    in the image.
    '''
    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE
        self.scan_y = cfg.SCAN_Y   # num pixels from the top to start horiz scan
        self.scan_height = cfg.SCAN_HEIGHT  # num pixels high to grab from horiz scan
        self.color_thr_low = np.asarray(cfg.COLOR_THRESHOLD_LOW)  # hsv dark yellow
        self.color_thr_hi = np.asarray(cfg.COLOR_THRESHOLD_HIGH)  # hsv light yellow
        self.line_width_min = cfg.LINE_WIDTH_MIN  # narrowest pixel width of a valid tape dash
        self.line_width_max = cfg.LINE_WIDTH_MAX  # widest pixel width of a valid tape dash
        self.target_pixel = cfg.TARGET_PIXEL  # of the N slots above, which is the ideal relationship target
        self.target_threshold = cfg.TARGET_THRESHOLD # minimum distance from target_pixel before a steering change is made.
        self.confidence_threshold = cfg.CONFIDENCE_THRESHOLD  # percentage of yellow pixels that must be in target_pixel slice
        self.steering = 0.0 # from -1 to 1
        self.throttle = cfg.THROTTLE_INITIAL # from -1 to 1
        self.delta_th = cfg.THROTTLE_STEP  # how much to change throttle when off
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        self.pid_st = pid


    def get_i_color(self, cam_img):
        '''
        get the horizontal index of the color at the given slice of the image
        input: cam_image, an RGB numpy array
        output: index of max color, value of cumulative color at that index, mask of
                pixels in range, and the pixel width of the blob that was picked (for
                tuning LINE_WIDTH_MIN/MAX -- 0 if nothing matched)
        '''
        # take a horizontal slice of the image
        iSlice = self.scan_y
        scan_line = cam_img[iSlice : iSlice + self.scan_height, :, :]

        # convert to HSV color space
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_RGB2HSV)

        # make a mask of the colors in our range we are looking for
        mask = cv2.inRange(img_hsv, self.color_thr_low, self.color_thr_hi)

        # Group matched pixels into connected blobs and keep only the ones whose
        # width matches the known width of a tape dash. A same-hue object off the
        # track (e.g. an orange/yellow rock) forms a wider or narrower blob than
        # the tape, so this rejects it even though the color match alone can't
        # tell them apart.
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        best_label = None
        best_area = 0
        for label in range(1, num_labels):  # label 0 is the background
            width = stats[label, cv2.CC_STAT_WIDTH]
            if self.line_width_min <= width <= self.line_width_max:
                area = stats[label, cv2.CC_STAT_AREA]
                if area > best_area:
                    best_label = label
                    best_area = area

        if best_label is None:
            return 0, 0, mask, 0

        max_yellow = int(round(centroids[best_label][0]))
        width = stats[best_label, cv2.CC_STAT_WIDTH]

        return max_yellow, best_area, mask, width


    def run(self, cam_img):
        '''
        main runloop of the CV controller
        input: cam_image, an RGB numpy array
        output: steering, throttle, and the image.
        If overlay_image is True, then the output image
        includes and overlay that shows how the 
        algorithm is working; otherwise the image
        is just passed-through untouched. 
        '''
        if cam_img is None:
            return 0, 0, False, None

        max_yellow, confidence, mask, width = self.get_i_color(cam_img)
        conf_thresh = 0.001

        # width of the matched blob -- point the camera at a tape dash vs. the
        # offending object and record this to pick LINE_WIDTH_MIN/MAX.
        # enable with LOGLEVEL=DEBUG (default level is INFO, so this is silent
        # during normal driving).
        logger.debug(f"line width: {width} px (I_YELLOW={max_yellow}, CONF={confidence})")

        if self.target_pixel is None:
            # Default to the image center rather than latching onto whatever
            # the first frame's strongest color match happens to be -- that
            # match can be background clutter (a wall, plant, reflection)
            # rather than the actual line, causing the car to track the
            # wrong object for the rest of the run.
            self.target_pixel = cam_img.shape[1] / 2
            logger.info(f"No TARGET_PIXEL configured; defaulting to image center = {self.target_pixel}")

        if self.pid_st.setpoint != self.target_pixel:
            # this is the target of our steering PID controller
            self.pid_st.setpoint = self.target_pixel

        if confidence >= self.confidence_threshold:
            # invoke the controller with the current yellow line position
            # get the new steering value as it chases the ideal
            self.steering = self.pid_st(max_yellow)

            # slow down linearly when away from ideal, and speed up when close
            if abs(max_yellow - self.target_pixel) > self.target_threshold:
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
            logger.info(f"No line detected: confidence {confidence} < {self.confidence_threshold}")

        # show some diagnostics
        if self.overlay_image:
            cam_img = self.overlay_display(cam_img, mask, max_yellow, confidence, width)

        return self.steering, self.throttle, cam_img

    def overlay_display(self, cam_img, mask, max_yellow, confidense, width):
        '''
        composite mask on top the original image.
        show some values we are using for control
        '''

        mask_exp = np.stack((mask, ) * 3, axis=-1)
        iSlice = self.scan_y
        img = np.copy(cam_img)
        img[iSlice : iSlice + self.scan_height, :, :] = mask_exp
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        display_str = []
        display_str.append("STEERING:{:.1f}".format(self.steering))
        display_str.append("THROTTLE:{:.2f}".format(self.throttle))
        display_str.append("I YELLOW:{:d}".format(max_yellow))
        display_str.append("CONF:{:.2f}".format(confidense))
        # width of the matched blob -- use this to tune LINE_WIDTH_MIN/MAX:
        # watch this value over the tape dashes vs. over a rock/off-track object.
        display_str.append("WIDTH:{:d}".format(width))

        y = 10
        x = 10

        for s in display_str:
            cv2.putText(img, s, color=(0, 0, 0), org=(x ,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
            y += 10

        return img

